#include "loina_terminal.h"
#include "linenoise.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

int linenoiseEditInsert(struct linenoiseState *, const char *, size_t);

static struct {
    int fd, input_fd, rows, cols, bottom, flags, column, escape;
    char utf8[4];
    size_t utf8_len;
    bool active, editing, eof, busy;
    bool *thinking;
    volatile sig_atomic_t *interrupt;
    struct linenoiseState edit;
    char input[16384], prompt[32], history[4096];
    char **queue;
    char *draft;
    size_t queued;
    pthread_mutex_t mutex;
    const char *state;
    loina_job *job;
    unsigned context, used, generated;
    int progress;
    double rate;
} ui;

void loina_terminal_job(loina_job *job) {
    if (!ui.active) return;
    pthread_mutex_lock(&ui.mutex);
    ui.job=job;
    if (job && *ui.interrupt) loina_job_cancel(job);
    pthread_mutex_unlock(&ui.mutex);
}

static void cancel(void) {
    *ui.interrupt=1;
    pthread_mutex_lock(&ui.mutex);
    if (ui.job) loina_job_cancel(ui.job);
    pthread_mutex_unlock(&ui.mutex);
}

static void cursor(int row, int col) {
    dprintf(ui.fd, "\x1b[%d;%dH", row, col);
}

static int layout(struct linenoiseState *edit, size_t prompt, size_t status, void *unused) {
    (void)edit; (void)unused;
    int reserved = (int)(prompt + status);
    if (reserved < 2) reserved = 2;
    if (reserved > ui.rows - 2) reserved = ui.rows - 2;
    int bottom = ui.rows - reserved;
    if (bottom != ui.bottom) {
        /* Make room for a growing draft by scrolling output, as ds4 does. */
        if (ui.bottom > bottom) {
            cursor(ui.bottom, 1);
            for (int i = bottom; i < ui.bottom; i++) dprintf(ui.fd, "\r\n");
        }
        ui.bottom = bottom;
        dprintf(ui.fd, "\x1b[1;%dr", bottom);
        cursor(bottom, ui.column < ui.cols ? ui.column + 1 : ui.cols);
        dprintf(ui.fd, "\0337");
        for (int row = bottom + 1; row <= ui.rows; row++) {
            cursor(row, 1);
            dprintf(ui.fd, "\x1b[2K");
        }
    }
    return bottom + 1;
}

static void status(void) {
    char activity[256], text[512], bar[97], used[32];
    pthread_mutex_lock(&ui.mutex);
    if (!strcmp(ui.state, "reading")) {
        bar[0] = 0;
        for (int i = 0; i < 32; i++) strcat(bar, i < ui.progress * 32 / 100 ? "▶" : "·");
        snprintf(activity, sizeof(activity), "reading [%s] %d%%", bar, ui.progress);
    } else if (!strcmp(ui.state, "generation")) {
        snprintf(activity, sizeof(activity), "generation %u tokens %.1f t/s", ui.generated, ui.rate);
    } else if (!strcmp(ui.state, "preparing tool call")) {
        snprintf(activity, sizeof(activity), "preparing tool call %u tokens %.1f t/s", ui.generated, ui.rate);
    } else snprintf(activity, sizeof(activity), "%s", ui.state);
    if (ui.used < 1000) snprintf(used, sizeof(used), "%u", ui.used);
    else snprintf(used, sizeof(used), "%.1fk", ui.used / 1000.0);
    snprintf(text, sizeof(text), "ctx %s/%.1fk | %s | think %s (Tab)",
             used, ui.context / 1000.0, activity, *ui.thinking ? "on" : "off");
    pthread_mutex_unlock(&ui.mutex);
    char footer[1024];
    if (ui.queued) {
        int preview = ui.cols * 2 - 48;
        if (preview < 8) preview = 8;
        snprintf(footer, sizeof(footer), "\x1b[38;5;87;1mqueued: %.*s (ctrl+x to edit, ESC to send ASAP)\x1b[0m\n\x1b[48;5;238;38;5;252m%s",
                 preview, ui.queue[0], text);
    } else snprintf(footer, sizeof(footer), "%s", text);
    if (ui.edit.status && !strcmp(ui.edit.status, footer)) return;
    linenoiseEditSetStatus(&ui.edit, footer, ui.queued ? "" : "\x1b[48;5;238;38;5;252m", "\x1b[0m");
}

static void show(void) {
    status();
    dprintf(ui.fd, "\x1b[0m");
    cursor(ui.bottom + 1, 1);
    linenoiseShow(&ui.edit);
}

static void output_column(unsigned char c) {
    if (ui.escape == 1) { ui.escape = c == '[' ? 2 : c == ']' ? 3 : 0; return; }
    if (ui.escape == 2) { if (c >= 0x40 && c <= 0x7e) ui.escape = 0; return; }
    if (ui.escape == 3) { if (c == 7) ui.escape = 0; else if (c == 27) ui.escape = 4; return; }
    if (ui.escape == 4) { ui.escape = c == '\\' ? 0 : 3; return; }
    if (c == 27) { ui.escape = 1; return; }
    if (c == '\n' || c == '\r') { ui.column = 0; return; }
    if (c < 32) return;
    ui.utf8[ui.utf8_len++] = (char)c;
    uint32_t cp;
    size_t n = linenoiseUtf8Decode(ui.utf8, ui.utf8_len, &cp);
    if (!n) return;
    ui.utf8_len = 0;
    int width = linenoiseCharacterWidth(cp);
    if (width < 0) width = 1;
    if (ui.column + width > ui.cols) ui.column = 0;
    ui.column += width;
}

static void output(const char *text, size_t size) {
    if (ui.editing) linenoiseHide(&ui.edit);
    dprintf(ui.fd, "\0338");
    for (size_t i = 0; i < size; i++) {
        if (text[i] == '\n' && (!i || text[i-1] != '\r')) write(ui.fd, "\r", 1);
        write(ui.fd, text + i, 1);
        output_column((unsigned char)text[i]);
    }
    dprintf(ui.fd, "\0337");
    if (ui.editing) show();
}

static void stop(void) {
    if (!ui.editing) return;
    linenoiseHide(&ui.edit);
    linenoiseEditStop(&ui.edit);
    ui.editing = false;
    fcntl(ui.input_fd, F_SETFL, ui.flags);
    dprintf(ui.fd, "\0338");
}

static bool start(void) {
    if (ui.editing) return true;
    dprintf(ui.fd, "\0337");
    cursor(ui.bottom + 1, 1);
    if (linenoiseEditStart(&ui.edit, ui.input_fd, ui.fd, ui.input, sizeof(ui.input), ui.prompt))
        return false;
    ui.editing = true;
    fcntl(ui.input_fd, F_SETFL, ui.flags | O_NONBLOCK);
    linenoiseEditSetLayoutCallback(&ui.edit, layout, NULL);
    show();
    if (ui.draft) {
        linenoiseEditInsert(&ui.edit, ui.draft, strlen(ui.draft));
        free(ui.draft); ui.draft = NULL;
    }
    return true;
}

static void resize(void) {
    struct winsize ws;
    if (ioctl(ui.fd, TIOCGWINSZ, &ws) || ws.ws_row < 8 || ws.ws_col < 20) return;
    if (ws.ws_row == ui.rows && ws.ws_col == ui.cols) return;
    linenoiseHide(&ui.edit);
    ui.rows = ws.ws_row; ui.cols = ws.ws_col;
    ui.edit.cols = ws.ws_col;
    ui.bottom = 0;
    layout(&ui.edit, 1, 1, NULL);
    show();
}

bool loina_terminal_active(void) { return ui.active; }

bool loina_terminal_command(loina_session *session, const char *command) {
    if (!strcmp(command, "/help")) {
        puts("Commands:\n"
             "  /help        Show this help.\n"
             "  /save        Save the current session.\n"
             "  /list        List saved sessions.\n"
             "  /switch ID   Load a saved session.\n"
             "  /del ID      Delete a saved session.\n"
             "  /history [N] Show N recent user turns from the current session.\n"
             "  /new         Start a fresh session.\n"
             "  /read FILE   Attach an image or text document.\n"
             "  /think       Enable thinking.\n"
             "  /nothink     Disable thinking.\n"
             "  /quit, /exit Exit.\n"
             "  Ctrl+C       Interrupt generation; clear edited text.\n"
             "  Tab          Toggle thinking at an empty idle prompt.\n"
             "  Enter        Queue text while the model is busy.\n"
             "  Ctrl+X       Edit the first queued prompt.\n"
             "  ESC          Interrupt and send queued prompt immediately.\n"
             "  Ctrl+D       Exit from an empty prompt.");
        return true;
    }
    if (strncmp(command, "/history", 8) || (command[8] && command[8] != ' ')) return false;
    char *end;
    unsigned long turns = command[8] ? strtoul(command + 9, &end, 10) : 10;
    if (command[8] && (!turns || *end || turns > 1000)) {
        puts("usage: /history [1..1000]");
        return true;
    }
    size_t count = loina_session_count(session), first = count;
    while (first) {
        loina_message m = loina_session_get(session, --first);
        if (m.role == LOINA_USER && strncmp(m.text, "<tool_response>", 15) && !--turns) break;
    }
    for (size_t i = first; i < count; i++) {
        loina_message m = loina_session_get(session, i);
        if (m.role == LOINA_USER && strncmp(m.text, "<tool_response>", 15))
            printf("* %s\n\n", m.text);
    }
    return true;
}

bool loina_terminal_open(const char *mode, uint32_t context, bool *thinking,
                         volatile sig_atomic_t *interrupt) {
    struct winsize ws;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO) ||
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) || ws.ws_row < 8 || ws.ws_col < 20) return false;
    ui.fd = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    if (ui.fd < 0) return false;
    fcntl(ui.fd, F_SETFD, FD_CLOEXEC);
    /* stdin and stdout may share one open file description. Reopen input so
     * nonblocking reads cannot make terminal writes drop redraw frames. */
    const char *terminal = ttyname(STDIN_FILENO);
    ui.input_fd = terminal ? open(terminal, O_RDONLY | O_NOCTTY | O_CLOEXEC) : -1;
    if (ui.input_fd < 0) { close(ui.fd); return false; }
    ui.flags = fcntl(ui.input_fd, F_GETFL);
    ui.rows = ws.ws_row; ui.cols = ws.ws_col;
    ui.context = context; ui.thinking = thinking; ui.interrupt = interrupt;
    ui.state = "idle";
    pthread_mutex_init(&ui.mutex, NULL);
    ui.active = true;
    snprintf(ui.prompt, sizeof(ui.prompt), "loina%s> ", !strcmp(mode, "Agent") ? "-agent" : "");
    const char *home = getenv("HOME");
    snprintf(ui.history, sizeof(ui.history), "%s/.loina%s_history", home ? home : ".",
             !strcmp(mode, "Agent") ? "_agent" : "");
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(ui.history);
    dprintf(ui.fd, "\x1b[0m\x1b[2J\x1b[H");
    layout(NULL, 1, 1, NULL);
    char banner[256];
    snprintf(banner, sizeof(banner), "\x1b[1m" LOINA_ACCENT "Loina\x1b[0m 💬 %s, context %.1fk tokens\n\n", mode, context / 1000.0);
    output(banner, strlen(banner));
    if (!strcmp(mode, "Agent")) {
        char directory[4096], info[4160];
        if (getcwd(directory, sizeof(directory))) {
            snprintf(info, sizeof(info), "Working directory: %s\n\n", directory);
            output(info, strlen(info));
        }
    }
    return true;
}

void loina_terminal_close(void) {
    if (!ui.active) return;
    stop();
    dprintf(ui.fd, "\x1b[0m\x1b[r");
    for (int row = ui.bottom + 1; row <= ui.rows; row++) {
        cursor(row, 1); dprintf(ui.fd, "\x1b[2K");
    }
    cursor(ui.rows, 1);
    dprintf(ui.fd, "\r\n");
    close(ui.fd);
    close(ui.input_fd);
    for (size_t i = 0; i < ui.queued; i++) free(ui.queue[i]);
    free(ui.queue);
    free(ui.draft);
    ui.active = false;
    pthread_mutex_destroy(&ui.mutex);
    memset(&ui,0,sizeof(ui));
}

static char *take(void) {
    if (!ui.queued) return NULL;
    char *line = ui.queue[0];
    memmove(ui.queue, ui.queue + 1, --ui.queued * sizeof(*ui.queue));
    return line;
}

static char *feed(void) {
    unsigned char buf[4096];
    ssize_t n = read(ui.input_fd, buf, sizeof(buf));
    if (!n) { ui.eof = true; if (ui.busy) cancel(); return NULL; }
    if (n < 0) return NULL;
    for (ssize_t i = 0; i < n; i++) {
        unsigned char c = buf[i];
        if (c == 3 && n == 1 && !ui.edit.paste_active) {
            if (ui.busy) cancel();
            else { linenoiseEditClear(&ui.edit); output("\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n", strlen("\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n")); }
        } else if (c == '\t' && n == 1 && !ui.edit.paste_active && !ui.edit.len && !ui.busy) {
            *ui.thinking = !*ui.thinking;
        } else if (c == 24 && n == 1 && !ui.edit.paste_active && ui.queued) {
            char *line = take();
            linenoiseEditInsert(&ui.edit, line, strlen(line)); free(line);
        } else if (c == 27 && n == 1 && ui.queued && ui.busy) {
            struct pollfd key = {.fd=ui.input_fd, .events=POLLIN};
            if (poll(&key, 1, 20) == 0) cancel();
            else linenoiseEditQueueInput(&ui.edit, (const char *)&c, 1);
        } else linenoiseEditQueueInput(&ui.edit, (const char *)&c, 1);
    }
    while (linenoiseEditQueuedInput(&ui.edit)) {
        errno = 0;
        char *line = linenoiseEditFeed(&ui.edit);
        if (line == linenoiseEditMore) continue;
        if (!line) {
            if (errno == EAGAIN) { linenoiseEditClear(&ui.edit); if (ui.busy) cancel(); }
            else { ui.eof = true; if (ui.busy) cancel(); }
            return NULL;
        }
        stop();
        return line;
    }
    return NULL;
}

char *loina_terminal_read(char *line, size_t size) {
    if (!ui.active) return fgets(line, (int)size, stdin);
    if (ui.eof) { stop(); return NULL; }
    char *value = take();
    if (value) stop();
    while (!value && !ui.eof) {
        if (!start()) return NULL;
        struct pollfd fd = {.fd=ui.input_fd, .events=POLLIN};
        int rc = poll(&fd, 1, 100);
        if (*ui.interrupt) { *ui.interrupt = 0; linenoiseEditClear(&ui.edit); }
        resize();
        if (rc > 0) value = feed();
        if (rc < 0 && errno != EINTR) break;
        if (ui.editing) { status(); linenoiseRefreshStatus(&ui.edit); }
    }
    stop();
    if (!value) return NULL;
    snprintf(line, size, "%s", value);
    if (*value) {
        linenoiseHistoryAdd(value);
        linenoiseHistorySave(ui.history);
        dprintf(ui.fd, LOINA_ACCENT "*\x1b[1;97m %s\x1b[0m\r\n\r\n", value);
        ui.column = 0;
        dprintf(ui.fd, "\0337");
    }
    linenoiseFree(value);
    return line;
}

void loina_terminal_event(loina_event event, const char *text, int progress) {
    if (!ui.active) return;
    pthread_mutex_lock(&ui.mutex);
    if (event == LOINA_TOKEN) ui.state = "generation";
    else if (event == LOINA_LOADING) ui.state = "loading";
    else if (event == LOINA_PROGRESS) { ui.state = "reading"; ui.progress = progress; }
    else if (event == LOINA_TOOL) {
        ui.state = !strcmp(text, "preparing tool call") ? "preparing tool call" :
                   !strcmp(text, "running tool") ? "running tool" : "reading the web";
    }
    else if (event == LOINA_METRICS) {
        const char *p = strstr(text, "\"context\":");
        if (p) ui.used = (unsigned)strtoul(p + 10, NULL, 10);
        p = strstr(text, "\"generated\":");
        if (p) ui.generated = (unsigned)strtoul(p + 12, NULL, 10);
        p = strstr(text, "\"decode\":");
        if (p) ui.rate = strtod(p + 9, NULL);
    }
    pthread_mutex_unlock(&ui.mutex);
}

typedef struct {
    int (*run)(void *);
    void *user;
    int result;
    atomic_bool done;
} task;

static void *worker(void *pointer) {
    task *t = pointer;
    t->result = t->run(t->user);
    fflush(stdout); fflush(stderr);
    atomic_store(&t->done, true);
    return NULL;
}

int loina_terminal_run(int (*run)(void *), void *user) {
    if (!ui.active) return run(user);
    int channel[2];
    if (pipe(channel)) return run(user);
    fflush(stdout); fflush(stderr);
    int out = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3), err = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
    if (out < 0 || err < 0) {
        if (out >= 0) close(out);
        if (err >= 0) close(err);
        close(channel[0]); close(channel[1]);
        return run(user);
    }
    fcntl(channel[0], F_SETFL, O_NONBLOCK);
    start();
    ui.busy = true;
    pthread_mutex_lock(&ui.mutex);
    ui.state = "reading"; ui.progress = 0; ui.generated = 0; ui.rate = 0;
    pthread_mutex_unlock(&ui.mutex);
    dup2(channel[1], STDOUT_FILENO); dup2(channel[1], STDERR_FILENO);
    close(channel[1]);
    task t = {.run=run, .user=user};
    atomic_init(&t.done, false);
    pthread_t thread;
    int created = pthread_create(&thread, NULL, worker, &t);
    if (!created) {
        while (!atomic_load(&t.done)) {
            struct pollfd fd[2] = {{.fd=channel[0], .events=POLLIN}, {.fd=ui.input_fd, .events=POLLIN}};
            poll(fd, 2, 100);
            char buf[4096]; ssize_t n;
            while ((n = read(channel[0], buf, sizeof(buf))) > 0) output(buf, (size_t)n);
            if (fd[1].revents) {
                char *line = feed();
                if (line && *line) {
                    char **queue = realloc(ui.queue, (ui.queued + 1) * sizeof(*queue));
                    if (queue) { ui.queue = queue; ui.queue[ui.queued++] = line; }
                    else free(line);
                    start();
                } else if (line) { free(line); start(); }
            }
            linenoiseRestoreRawMode();
            resize();
            if (ui.editing) { status(); linenoiseRefreshStatus(&ui.edit); }
        }
        pthread_join(thread, NULL);
    }
    dup2(out, STDOUT_FILENO); dup2(err, STDERR_FILENO);
    close(out); close(err);
    char buf[4096]; ssize_t n;
    while ((n = read(channel[0], buf, sizeof(buf))) > 0) output(buf, (size_t)n);
    close(channel[0]);
    ui.busy = false;
    pthread_mutex_lock(&ui.mutex);
    ui.state = "idle";
    pthread_mutex_unlock(&ui.mutex);
    if (created) { stop(); return run(user); }
    if (ui.edit.len) ui.draft = strdup(ui.edit.buf);
    stop();
    return t.result;
}

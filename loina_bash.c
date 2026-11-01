#include "loina_bash.h"
#include "loina_renderer.h"
#include "loina_terminal.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct bash_job {
    struct bash_job *next;
    pthread_t thread;
    pthread_mutex_t mutex;
    pid_t pid;
    int id, input, log, status, error;
    bool done, timed_out, observed;
    double start, end, timeout, stop_deadline;
    size_t bytes, lines;
    char last, path[64];
} bash_job;

static bash_job *jobs;
static int sequence;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void *monitor(void *pointer) {
    bash_job *j = pointer;
    bool reaped = false, eof = false;
    int status = 0;
    while (!reaped || !eof) {
        char data[4096];
        for (int chunk=0; !eof && chunk<16; chunk++) {
            ssize_t n = read(j->input, data, sizeof(data));
            if (!n) { eof = true; break; }
            if (n < 0) {
                if (errno != EINTR && errno != EAGAIN) eof = true;
                break;
            }
            size_t written = 0;
            while (written < (size_t)n) {
                ssize_t w = write(j->log, data+written, (size_t)n-written);
                if (w < 0 && errno == EINTR) continue;
                if (w <= 0) {
                    pthread_mutex_lock(&j->mutex); j->error = errno; pthread_mutex_unlock(&j->mutex);
                    kill(-j->pid, SIGKILL); eof = true; break;
                }
                written += (size_t)w;
            }
            pthread_mutex_lock(&j->mutex);
            j->bytes += written;
            for (size_t i=0; i<written; i++) if (data[i]=='\n') j->lines++;
            if (written) j->last = data[written-1];
            pthread_mutex_unlock(&j->mutex);
        }
        if (!reaped) {
            pid_t pid=waitpid(j->pid, &status, WNOHANG);
            if (pid==j->pid) reaped=true;
            else if (pid<0 && errno!=EINTR) { reaped=true; status=-1; }
        }
        if (now()-j->start >= j->timeout && (!reaped || !eof)) {
            kill(-j->pid, SIGKILL);
            pthread_mutex_lock(&j->mutex); j->timed_out = true; pthread_mutex_unlock(&j->mutex);
        }
        pthread_mutex_lock(&j->mutex);
        bool expired=(j->timed_out && now()-j->start>=j->timeout+1) ||
                     (j->stop_deadline && now()>=j->stop_deadline);
        if (expired && reaped && !eof) { eof=true; j->error=EIO; }
        pthread_mutex_unlock(&j->mutex);
        if (!reaped || !eof) {
            struct pollfd fd = {.fd=eof ? -1 : j->input, .events=POLLIN};
            poll(&fd, 1, 20);
        }
    }
    close(j->input);
    pthread_mutex_lock(&j->mutex);
    j->status = WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? 128+WTERMSIG(status) : -1;
    j->end = now(); j->done = true;
    pthread_mutex_unlock(&j->mutex);
    return NULL;
}

static bool done(bash_job *j) {
    pthread_mutex_lock(&j->mutex);
    bool value = j->done;
    pthread_mutex_unlock(&j->mutex);
    return value;
}

static void stop_job(bash_job *j) {
    pthread_mutex_lock(&j->mutex);
    if (!j->stop_deadline) j->stop_deadline=now()+1;
    pthread_mutex_unlock(&j->mutex);
    kill(-j->pid,SIGKILL);
    kill(j->pid,SIGKILL);
}

static void destroy(bash_job *j) {
    if (!done(j)) stop_job(j);
    pthread_join(j->thread, NULL);
    close(j->log);
    pthread_mutex_destroy(&j->mutex);
    free(j);
}

void loina_bash_cleanup(void) {
    while (jobs) { bash_job *j=jobs; jobs=j->next; destroy(j); }
}

static char *observe(bash_job *j, long refresh, bool stop, volatile sig_atomic_t *interrupt) {
    if (stop && !done(j)) kill(-j->pid, SIGTERM);
    double until = now() + (stop && refresh<1 ? 1 : refresh);
    while (!done(j) && now()<until) {
        if (interrupt && *interrupt) { stop_job(j); break; }
        struct timespec pause = {0,20000000}; nanosleep(&pause,NULL);
    }
    if ((stop || (interrupt && *interrupt)) && !done(j)) {
        stop_job(j);
        double deadline=now()+1.2;
        while (!done(j) && now()<deadline) { struct timespec pause={0,20000000}; nanosleep(&pause,NULL); }
    }
    pthread_mutex_lock(&j->mutex);
    bool finished=j->done, first=!j->observed;
    size_t bytes=j->bytes, lines=j->lines + (j->bytes && j->last!='\n');
    int status=j->status, timed_out=j->timed_out, error=j->error;
    double elapsed=(finished ? j->end : now())-j->start;
    j->observed=true;
    pthread_mutex_unlock(&j->mutex);
    size_t limit=first ? 8192 : 32768;
    size_t offset=first || bytes<limit ? 0 : bytes-limit;
    size_t size=bytes-offset;
    if (size>limit) size=limit;
    char *body=malloc(size+1), *out=malloc(size+2048);
    if (!body || !out) { free(body); free(out); return strdup("Tool error: cannot allocate bash observation"); }
    size_t got=0;
    while (got<size) {
        ssize_t n=pread(j->log,body+got,size-got,(off_t)(offset+got));
        if (n<0 && errno==EINTR) continue;
        if (n<=0) break;
        got+=(size_t)n;
    }
    body[got]=0;
    char *text=body;
    int display=first ? 100 : finished ? 20 : 4;
    if (first) {
        int count=0;
        for (size_t i=0; i<got; i++) if (body[i]=='\n' && ++count==display) {
            got=i+1; body[got]=0; break;
        }
    } else {
        int count=got && body[got-1]!='\n';
        for (size_t i=got; i>0; i--) if (body[i-1]=='\n' && ++count>display) { text=body+i; break; }
        while (((unsigned char)*text & 0xc0)==0x80) text++;
    }
    bool truncated=offset || got<bytes;
    int used;
    if (finished) used=snprintf(out,size+2048,
        "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\nexit_status=%d\n",
        j->id,(long)j->pid,elapsed,timed_out,status);
    else used=snprintf(out,size+2048,
        "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
        j->id,(long)j->pid,elapsed,j->timeout);
    if (error) used+=snprintf(out+used,size+2048-(size_t)used,
        "Tool error: command output could not be captured completely: %s\n",strerror(error));
    if (bytes && (!first || truncated || !finished)) {
        used+=snprintf(out+used,size+2048-(size_t)used,
            "output_path=%s (%zu bytes, %zu lines)\n",j->path,bytes,lines);
        used+=snprintf(out+used,size+2048-(size_t)used,"<%s -%d %s>\n",first ? "head" : "tail",display,j->path);
    } else used+=snprintf(out+used,size+2048-(size_t)used,"<output>\n");
    const char *label=bytes && (!first || truncated || !finished) ? first ? "head" : "tail" : "output";
    used+=snprintf(out+used,size+2048-(size_t)used,"%s%s</%s>\n",text,
                   *text && text[strlen(text)-1]!='\n' ? "\n" : "",label);
    if (!finished) snprintf(out+used,size+2048-(size_t)used,
        "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",j->id,j->id);
    if (*text) {
        char *safe=loina_renderer_safe_text(text,strlen(text));
        bool color=loina_terminal_active() || isatty(STDERR_FILENO);
        if (color && finished && status) fputs("\x1b[38;5;208m",stderr);
        fputs(safe,stderr); free(safe);
        if (text[strlen(text)-1]!='\n') fputc('\n',stderr);
        if (color) fputs("\x1b[0m",stderr);
        fflush(stderr);
    }
    free(body);
    if (finished) {
        bash_job **p=&jobs;
        while (*p && *p!=j) p=&(*p)->next;
        if (*p) *p=j->next;
        destroy(j);
    }
    return out;
}

extern char **environ;
static int spawn_shell(pid_t *pid, int tmpfd, const int pipefd[2],
                            const char *cmd) {
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    int rc = posix_spawnattr_init(&attr);
    if (rc) return rc;
    rc = posix_spawn_file_actions_init(&actions);
    if (rc) {
        posix_spawnattr_destroy(&attr);
        return rc;
    }
    rc = posix_spawnattr_setpgroup(&attr, 0);
    if (!rc) rc = posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    if (!rc) rc = posix_spawn_file_actions_addclose(&actions, tmpfd);
    if (!rc) rc = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    if (!rc) rc = posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    if (!rc) rc = posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO);
    if (!rc) rc = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    /* A tool must not inherit or change the live terminal's input mode. */
    if (!rc) rc = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO,
                                                 "/dev/null", O_RDONLY, 0);
    char *argv[] = {"sh", "-c", (char *)(cmd ? cmd : ""), NULL};
    if (!rc) rc = posix_spawn(pid, "/bin/sh", &actions, &attr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    return rc;
}

char *loina_bash_run(const char *command, long timeout, long refresh, volatile sig_atomic_t *interrupt) {
    if (!command || !*command) return strdup("Tool error: bash requires command\n");
    if (timeout<=0) timeout=3600;
    if (refresh<0) refresh=60;
    bash_job *j=calloc(1,sizeof(*j));
    int channel[2];
    if (!j) return strdup("Tool error: cannot allocate bash job\n");
    strcpy(j->path,"/tmp/loina-bash-XXXXXX");
    j->log=mkstemp(j->path);
    if (j->log<0) { free(j); return strdup("Tool error: cannot capture bash output\n"); }
    if (pipe(channel)) { close(j->log); unlink(j->path); free(j); return strdup("Tool error: cannot capture bash output\n"); }
    fcntl(j->log,F_SETFD,FD_CLOEXEC);
    fcntl(channel[0],F_SETFD,FD_CLOEXEC);
    fcntl(channel[1],F_SETFD,FD_CLOEXEC);
    int rc=spawn_shell(&j->pid,j->log,channel,command);
    close(channel[1]);
    if (rc) {
        close(channel[0]); close(j->log); unlink(j->path); free(j);
        return strdup("Tool error: cannot start bash\n");
    }
    j->input=channel[0]; fcntl(j->input,F_SETFL,O_NONBLOCK);
    j->id=++sequence; j->start=now(); j->timeout=(double)timeout;
    pthread_mutex_init(&j->mutex,NULL);
    if (pthread_create(&j->thread,NULL,monitor,j)) {
        kill(-j->pid,SIGKILL); waitpid(j->pid,NULL,0);
        close(j->input); close(j->log); pthread_mutex_destroy(&j->mutex); free(j);
        return strdup("Tool error: cannot monitor bash\n");
    }
    j->next=jobs; jobs=j;
    return observe(j,refresh,false,interrupt);
}

char *loina_bash_status(int id, int pid, long refresh, bool stop, volatile sig_atomic_t *interrupt) {
    for (bash_job *j=jobs; j; j=j->next)
        if ((id && j->id==id) || (!id && pid && j->pid==pid)) return observe(j,refresh,stop,interrupt);
    char message[128];
    snprintf(message,sizeof(message),"Tool error: bash job not found: job=%d pid=%d\n",id,pid);
    return strdup(message);
}

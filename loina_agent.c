#include "loina.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fnmatch.h>
#include <limits.h>
#include <regex.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* loina-agent, ds4 style.
 *
 * A coding agent on top of loina-core (llama.cpp runtime) using loina
 * commands: loina_open / loina_job_create / loina_job_run with the same
 * immutable catalog packages as the app. Interactive REPL mirrors ./loina
 * (sessions, /read vision, thinking) plus agent tools.
 *
 * Tool protocol (ASCII, intentionally distinct from the core web-tool markers
 * so loina_chat streams them as plain text for the agent to dispatch):
 *
 *   <loina_tool_calls>
 *   <loina_invoke name="read">
 *   <loina_param name="path">src/main.c</loina_param>
 *   </loina_invoke>
 *   </loina_tool_calls>
 *
 * Tool calls are never inside <think>. The agent executes them locally
 * (read/more/write/edit/search/list/bash/view_image/web_search/web_visit),
 * feeds observations back as <tool_response> user turns, and loops until the
 * model answers without tools. Inspired by ds4_agent.c and q36_agent.c. */

static volatile sig_atomic_t g_interrupted;
static loina_job *volatile g_job;

static void on_sigint(int s) {
    (void)s;
    g_interrupted = 1;
    if (g_job)
        loina_job_cancel(g_job);
}

/* ---------- small utils ---------- */

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("loina-agent: malloc");
        exit(1);
    }
    return p;
}

static char *xstrdup(const char *s) {
    if (!s)
        s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        perror("loina-agent: realloc");
        exit(1);
    }
    return q;
}

static bool parse_u32(const char *s, uint32_t *n) {
    if (!s || !*s || *s == '-')
        return false;
    char *e;
    errno = 0;
    unsigned long v = strtoul(s, &e, 10);
    if (errno || *e || v > UINT32_MAX)
        return false;
    *n = (uint32_t)v;
    return true;
}

static bool parse_real(const char *s, float *n) {
    char *e;
    errno = 0;
    *n = strtof(s, &e);
    return !errno && e != s && !*e;
}

static int parse_int_exit(const char *s, const char *opt) {
    uint32_t n;
    if (!parse_u32(s, &n) || !n || n > INT32_MAX) {
        fprintf(stderr, "loina-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)n;
}

static float parse_float_exit(const char *s, const char *opt, float lo, float hi) {
    float v;
    if (!parse_real(s, &v) || v < lo || v > hi) {
        fprintf(stderr, "loina-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "loina-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

/* ---------- streaming renderer (thinking grey + tool hiding) ---------- */

typedef struct {
    loina_job *job;
    char *buf;
    size_t len, cap;
    bool use_color;
    bool in_think;
    bool color_open;
    bool last_nl;
    bool in_tool;
    char pending[32];
    size_t pending_len;
    double prefill, decode;
    bool have_metrics;
} stream_t;

static void s_grey(stream_t *s) {
    if (s->use_color && !s->color_open) {
        fputs("\x1b[90m", stdout);
        s->color_open = true;
    }
}

static void s_reset(stream_t *s) {
    if (s->use_color && s->color_open) {
        fputs("\x1b[0m", stdout);
        s->color_open = false;
    }
}

static void s_emit(stream_t *s, const char *p, size_t n) {
    if (n > 16 * 1024 * 1024)
        return;
    if (s->len + n + 1 > s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 8192;
        while (nc < s->len + n + 1)
            nc *= 2;
        s->buf = xrealloc(s->buf, nc);
        s->cap = nc;
    }
    memcpy(s->buf + s->len, p, n);
    s->len += n;
    s->buf[s->len] = 0;
}

static bool starts(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n >= m && !memcmp(p, s, m);
}

static bool partial(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n < m && !memcmp(s, p, n);
}

static void s_write(stream_t *s, const char *text, size_t n, bool finish) {
    size_t total = s->pending_len + n;
    char *b = xmalloc(total ? total : 1);
    if (s->pending_len)
        memcpy(b, s->pending, s->pending_len);
    if (n)
        memcpy(b + s->pending_len, text, n);
    s->pending_len = 0;
    size_t i = 0;
    const char *tool_open = "<loina_tool_calls>";
    const char *tool_close = "</loina_tool_calls>";
    while (i < total) {
        const char *cur = b + i;
        size_t rem = total - i;
        if (!s->in_tool && starts(cur, rem, tool_open)) {
            s->in_tool = true;
            i += strlen(tool_open);
            continue;
        }
        if (s->in_tool && starts(cur, rem, tool_close)) {
            s->in_tool = false;
            i += strlen(tool_close);
            continue;
        }
        if (s->in_tool) {
            i++;
            continue; /* hide tool XML; announced after the turn */
        }
        if (starts(cur, rem, "<think>")) {
            s->in_think = true;
            i += 7;
            continue;
        }
        if (starts(cur, rem, "</think>")) {
            s->in_think = false;
            s_reset(s);
            if (!s->last_nl) {
                fputc('\n', stdout);
                s->last_nl = true;
            }
            i += 8;
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (partial(cur, rem, "<think>") || partial(cur, rem, "</think>") ||
                partial(cur, rem, tool_open) || partial(cur, rem, tool_close))) {
            if (rem < sizeof(s->pending)) {
                memcpy(s->pending, cur, rem);
                s->pending_len = rem;
            }
            break;
        }
        s_emit(s, cur, 1);
        if (s->in_think)
            s_grey(s);
        fputc((unsigned char)cur[0], stdout);
        s->last_nl = (cur[0] == '\n');
        i++;
    }
    free(b);
}

static void stream_event(void *user, loina_event ev, const char *text, size_t n, int progress) {
    stream_t *s = user;
    (void)progress;
    if (g_interrupted && s->job)
        loina_job_cancel(s->job);
    if (ev == LOINA_TOKEN) {
        s_write(s, text, n, false);
        fflush(stdout);
    } else if (ev == LOINA_LOADING) {
        fprintf(stderr, "Loading %.*s\n", (int)n, text);
    } else if (ev == LOINA_TOOL) {
        fprintf(stderr, "Reading the web...\n");
    } else if (ev == LOINA_COMPACTED) {
        fprintf(stderr, "Earlier turns remain in the saved transcript.\n");
    } else if (ev == LOINA_METRICS) {
        const char *p = strstr(text, "\"prefill\""), *d = strstr(text, "\"decode\"");
        if (p)
            s->prefill = strtod(strchr(p, ':') ? strchr(p, ':') + 1 : p, NULL);
        if (d)
            s->decode = strtod(strchr(d, ':') ? strchr(d, ':') + 1 : d, NULL);
        s->have_metrics = true;
    }
}

/* ---------- Linux web bridge (same shape as ./loina CLI) ---------- */

static void web_fail(loina_error *e, loina_status c, const char *m) {
    if (e) {
        e->code = c;
        snprintf(e->message, sizeof(e->message), "%s", m);
    }
}

static void sh_quote(const char *s, char *o, size_t n) {
    size_t u = 0;
    if (u < n)
        o[u++] = '\'';
    for (; *s && u + 4 < n; s++) {
        if (*s == '\'') {
            memcpy(o + u, "'\\''", 4);
            u += 4;
        } else
            o[u++] = *s;
    }
    if (u < n)
        o[u++] = '\'';
    o[u < n ? u : n - 1] = 0;
}

static char *fetch_url(const char *url) {
    char q[8192 + 16], cmd[8192 + 128];
    sh_quote(url, q, sizeof(q));
    snprintf(cmd, sizeof(cmd),
        "curl -sL --max-time 25 --proto '=https' --user-agent 'Loina/1.0' %s 2>/dev/null | head -c 1048576", q);
    FILE *f = popen(cmd, "r");
    if (!f)
        return NULL;
    size_t cap = 1 << 20, len = 0;
    char *b = xmalloc(cap + 1);
    size_t n;
    while ((n = fread(b + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap)
            break;
    }
    pclose(f);
    b[len] = 0;
    if (!len) {
        free(b);
        return NULL;
    }
    return b;
}

static void buf_add(char **b, size_t *l, size_t *c, const char *s, size_t n) {
    if (!n)
        return;
    if (*l + n + 1 > *c) {
        size_t nc = *c ? *c * 2 : 8192;
        while (nc < *l + n + 1)
            nc *= 2;
        *b = xrealloc(*b, nc);
        *c = nc;
    }
    memcpy(*b + *l, s, n);
    *l += n;
    (*b)[*l] = 0;
}

static char *strip_tags(const char *h, size_t a, size_t b) {
    char *o = NULL;
    size_t l = 0, c = 0;
    bool sp = true;
    for (size_t i = a; i < b;) {
        if (h[i] == '<') {
            size_t j = i + 1;
            while (j < b && h[j] != '>')
                j++;
            if (j >= b)
                break;
            i = j + 1;
            continue;
        }
        if (isspace((unsigned char)h[i])) {
            if (!sp) {
                buf_add(&o, &l, &c, " ", 1);
                sp = true;
            }
        } else {
            buf_add(&o, &l, &c, h + i, 1);
            sp = false;
        }
        i++;
    }
    if (!o)
        o = xstrdup("");
    char *s = o;
    while (*s && isspace((unsigned char)*s))
        s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1]))
        s[--n] = 0;
    if (s != o)
        memmove(o, s, n + 1);
    return o;
}

static const char *ci_find(const char *h, const char *n) {
    size_t nl = strlen(n);
    if (!nl)
        return h;
    for (; *h; h++) {
        size_t i = 0;
        while (i < nl && h[i] && tolower((unsigned char)h[i]) == tolower((unsigned char)n[i]))
            i++;
        if (i == nl)
            return h;
    }
    return NULL;
}

static char *page_text(const char *html) {
    size_t n = strlen(html);
    /* drop script/style blocks naively */
    char *clean = NULL;
    size_t l = 0, c = 0;
    for (size_t i = 0; i < n;) {
        if (!strncasecmp(html + i, "<script", 7) || !strncasecmp(html + i, "<style", 6)) {
            const char *e = strchr(html + i, '>');
            if (!e)
                break;
            i = (size_t)(e - html) + 1;
            const char *c1 = ci_find(html + i, "</script");
            const char *c2 = ci_find(html + i, "</style");
            const char *cl = c1 && c2 ? (c1 < c2 ? c1 : c2) : (c1 ? c1 : c2);
            if (cl) {
                const char *gt = strchr(cl, '>');
                i = gt ? (size_t)(gt - html) + 1 : n;
            } else
                i = n;
            continue;
        }
        buf_add(&clean, &l, &c, html + i, 1);
        i++;
    }
    if (!clean)
        clean = xstrdup(html);
    char *t = strip_tags(clean, 0, strlen(clean));
    free(clean);
    const char *ti = ci_find(html, "<title>");
    char title[256] = {0};
    if (ti) {
        ti += 7;
        const char *te = ci_find(ti, "</title>");
        if (te && (size_t)(te - ti) < 500) {
            char *tt = strip_tags(html, (size_t)(ti - html), (size_t)(te - html));
            strncpy(title, tt, sizeof(title) - 1);
            free(tt);
        }
    }
    char *out = NULL;
    size_t ol = 0, oc = 0;
    buf_add(&out, &ol, &oc, "# ", 2);
    buf_add(&out, &ol, &oc, title[0] ? title : "page", strlen(title[0] ? title : "page"));
    buf_add(&out, &ol, &oc, "\n\n## Page text\n", 14);
    if (strlen(t) > 6000)
        t[6000] = 0;
    buf_add(&out, &ol, &oc, t, strlen(t));
    free(t);
    return out;
}

static loina_status agent_web(void *user, const char *url, char **text, loina_job *job, loina_error *error) {
    (void)user;
    if (text)
        *text = NULL;
    if (!url || strncmp(url, "https://", 8)) {
        web_fail(error, LOINA_INVALID, "Web tools require an HTTPS URL");
        return LOINA_INVALID;
    }
    if (loina_job_cancelled(job)) {
        web_fail(error, LOINA_CANCELLED, "Web request cancelled");
        return LOINA_CANCELLED;
    }
    char *html = fetch_url(url);
    if (!html) {
        if (strstr(url, "html.duckduckgo.com")) {
            const char *q = strstr(url, "?q=");
            if (q) {
                char bing[8448];
                snprintf(bing, sizeof(bing), "https://www.bing.com/search?q=%s", q + 3);
                html = fetch_url(bing);
            }
        }
        if (!html) {
            web_fail(error, LOINA_IO, "Tool error: Web page could not be loaded");
            return LOINA_IO;
        }
    }
    bool search = strstr(url, "duckduckgo.com") || strstr(url, "bing.com/search");
    char *out = NULL;
    if (search) {
        size_t ol = 0, oc = 0;
        buf_add(&out, &ol, &oc, "# Search results\n\nURL: ", 22);
        buf_add(&out, &ol, &oc, url, strlen(url));
        buf_add(&out, &ol, &oc, "\n", 1);
        int found = 0;
        size_t pos = 0, n = strlen(html);
        while (found < 3 && pos < n) {
            const char *a = strstr(html + pos, "<a");
            if (!a)
                break;
            pos = (size_t)(a - html) + 2;
            const char *te = strchr(a, '>');
            if (!te)
                break;
            size_t tl = (size_t)(te - a);
            char *tag = xstrndup(a, tl);
            for (char *cp = tag; *cp; cp++)
                *cp = (char)tolower((unsigned char)*cp);
            bool hit = strstr(tag, "result__a") || strstr(tag, "b_algo") || strstr(tag, "b_title");
            free(tag);
            if (!hit)
                continue;
            const char *href = strstr(a, "href=");
            char link[4096] = {0};
            if (href) {
                href += 5;
                while (*href == ' ' || *href == '"' || *href == '\'')
                    href++;
                size_t li = 0;
                while (*href && *href != '"' && *href != '\'' && *href != ' ' && *href != '>' && li + 1 < sizeof(link))
                    link[li++] = *href++;
                char *ud = strstr(link, "uddg=");
                if (ud) {
                    memmove(link, ud + 5, strlen(ud + 5) + 1);
                    char *amp = strchr(link, '&');
                    if (amp)
                        *amp = 0;
                    /* percent-decode minimal */
                    char *w = link;
                    for (char *r = link; *r; r++) {
                        if (*r == '%' && isxdigit((unsigned char)r[1]) && isxdigit((unsigned char)r[2])) {
                            char hex[3] = {r[1], r[2], 0};
                            *w++ = (char)strtol(hex, NULL, 16);
                            r += 2;
                        } else if (*r == '+')
                            *w++ = ' ';
                        else
                            *w++ = *r;
                    }
                    *w = 0;
                } else if (!strncmp(link, "//", 2)) {
                    memmove(link + 6, link + 2, strlen(link + 2) + 1);
                    memcpy(link, "https:", 6);
                }
            }
            const char *ls = te + 1, *le = strstr(ls, "</a>");
            if (!le) {
                pos = (size_t)(te - html) + 1;
                continue;
            }
            if (strncmp(link, "https://", 8) || strstr(link, "duckduckgo.com") || strstr(link, "bing.com")) {
                pos = (size_t)(le - html) + 4;
                continue;
            }
            char *label = strip_tags(html, (size_t)(ls - html), (size_t)(le - html));
            if (!label || strlen(label) < 3) {
                free(label);
                pos = (size_t)(le - html) + 4;
                continue;
            }
            if (strlen(label) > 160)
                label[160] = 0;
            buf_add(&out, &ol, &oc, "\n- ", 3);
            buf_add(&out, &ol, &oc, label, strlen(label));
            buf_add(&out, &ol, &oc, ": ", 2);
            buf_add(&out, &ol, &oc, link, strlen(link));
            buf_add(&out, &ol, &oc, "\n", 1);
            free(label);
            found++;
            pos = (size_t)(le - html) + 4;
        }
        if (!found) {
            free(html);
            free(out);
            web_fail(error, LOINA_IO, "Tool error: Search returned no readable results. Try a more specific query.");
            return LOINA_IO;
        }
    } else {
        out = page_text(html);
    }
    free(html);
    if (loina_job_cancelled(job)) {
        free(out);
        web_fail(error, LOINA_CANCELLED, "Web request cancelled");
        return LOINA_CANCELLED;
    }
    *text = out;
    return LOINA_OK;
}

/* ---------- local tools (ds4 contracts, loina-sized) ---------- */

typedef struct {
    char more_path[PATH_MAX];
    int more_next;
    bool more_valid;
} more_state_t;

static char *tool_err(const char *fmt, const char *a) {
    char b[1024];
    snprintf(b, sizeof(b), "Tool error: %s%s", fmt, a ? a : "");
    return xstrdup(b);
}

static bool under_cwd(const char *path, char out[PATH_MAX]) {
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd)))
        return false;
    char tmp[PATH_MAX];
    if (path[0] == '/') {
        if (strlen(path) >= sizeof(tmp)) return false;
        snprintf(tmp, sizeof(tmp), "%s", path);
    } else {
        if (strlen(cwd) + 1 + strlen(path) >= sizeof(tmp)) return false;
        snprintf(tmp, sizeof(tmp), "%s/%s", cwd, path);
    }
    /* Collapse ./ and trailing /. No symlink resolution (tools follow the
     * target like ds4; hard-link rejection is best-effort below). */
    char *w = out;
    const char *r = tmp;
    if (*r == '/')
        *w++ = *r++;
    while (*r) {
        while (*r == '/')
            r++;
        if (!*r)
            break;
        const char *seg = r;
        while (*r && *r != '/')
            r++;
        size_t n = (size_t)(r - seg);
        if (n == 1 && seg[0] == '.')
            continue;
        if (n == 2 && seg[0] == '.' && seg[1] == '.') {
            /* Refuse to escape the workspace for writes; reads may use
             * absolute paths like the app document picker. */
            if (w > out + 1) {
                w--;
                while (w > out && *(w - 1) != '/')
                    w--;
            }
            continue;
        }
        if ((size_t)(w - out) + n + 2 >= PATH_MAX)
            return false;
        if (w != out && *(w - 1) != '/')
            *w++ = '/';
        memcpy(w, seg, n);
        w += n;
    }
    *w = 0;
    if (!out[0])
        strcpy(out, "/");
    return true;
}

static char *tool_read(const char *path, long start, long maxl, bool whole, bool raw, more_state_t *more) {
    char ap[PATH_MAX];
    if (!path || !*path)
        return tool_err("read needs path", NULL);
    if (!under_cwd(path, ap))
        return tool_err("invalid path: ", path);
    FILE *f = fopen(ap, "rb");
    if (!f)
        return tool_err("cannot open file: ", path);
    struct stat st;
    if (fstat(fileno(f), &st) || !S_ISREG(st.st_mode)) {
        fclose(f);
        return tool_err("not a regular file: ", path);
    }
    if (st.st_size > 8 * 1024 * 1024) {
        fclose(f);
        return tool_err("file too large (8 MB limit): ", path);
    }
    size_t cap = (size_t)st.st_size + 1;
    char *data = xmalloc(cap);
    size_t n = fread(data, 1, (size_t)st.st_size, f);
    fclose(f);
    data[n] = 0;
    if (memchr(data, 0, n)) {
        free(data);
        return tool_err("binary file, use view_image for images: ", path);
    }
    /* Split lines. */
    size_t lines = 1;
    for (size_t i = 0; i < n; i++)
        if (data[i] == '\n')
            lines++;
    long s = start > 0 ? start : 1;
    long m = maxl > 0 ? maxl : (whole ? (long)lines : 160);
    if (!whole && maxl <= 0)
        m = 80; /* first look default 80-160 lines, like the agent prompt */
    if (whole && (n > 128 * 1024)) {
        free(data);
        return tool_err("whole=true fails on files over 128 KiB; use chunks with more", NULL);
    }
    if (s < 1)
        s = 1;
    /* find byte offsets */
    size_t off = 0;
    long cur = 1;
    while (off < n && cur < s) {
        if (data[off++] == '\n')
            cur++;
    }
    size_t end = off;
    long shown = 0;
    while (end < n && shown < m) {
        if (data[end++] == '\n')
            shown++;
    }
    bool truncated = end < n;
    /* If byte-limited mid-line, 'more' continues exactly there. */
    size_t chunk = end - off;
    if (!whole && chunk > 128 * 1024) {
        chunk = 128 * 1024;
        end = off + chunk;
        truncated = true;
    }
    char *out = NULL;
    size_t ol = 0, oc = 0;
    if (!raw) {
        char head[512];
        snprintf(head, sizeof(head), "lines %ld-%ld of %zu%s\n", s, s + shown - (shown ? 1 : 0), lines,
            truncated ? " (more available)" : "");
        buf_add(&out, &ol, &oc, head, strlen(head));
        long ln = s;
        size_t p = off;
        char lb[32];
        snprintf(lb, sizeof(lb), "%ld: ", ln);
        buf_add(&out, &ol, &oc, lb, strlen(lb));
        while (p < end) {
            if (data[p] == '\n') {
                buf_add(&out, &ol, &oc, "\n", 1);
                ln++;
                if (p + 1 < end) {
                    snprintf(lb, sizeof(lb), "%ld: ", ln);
                    buf_add(&out, &ol, &oc, lb, strlen(lb));
                }
            } else {
                buf_add(&out, &ol, &oc, data + p, 1);
            }
            p++;
        }
        if (ol && out[ol - 1] != '\n')
            buf_add(&out, &ol, &oc, "\n", 1);
    } else {
        buf_add(&out, &ol, &oc, data + off, chunk);
    }
    /* Save more cursor (byte-exact continuation). */
    snprintf(more->more_path, sizeof(more->more_path), "%s", ap);
    more->more_next = (int)(s + shown);
    more->more_valid = truncated;
    free(data);
    return out;
}

static char *tool_more(long count, more_state_t *more) {
    if (!more->more_valid)
        return tool_err("nothing more to read; call read first", NULL);
    long m = count > 0 ? count : 160;
    /* Re-read from saved byte line: read starting at more_next. */
    char rel[PATH_MAX];
    snprintf(rel, sizeof(rel), "%s", more->more_path);
    return tool_read(rel, more->more_next, m, false, false, more);
}

static bool is_hardlink(const char *ap) {
    struct stat st;
    return !stat(ap, &st) && st.st_nlink > 1;
}

static char *tool_write(const char *path, const char *content) {
    char ap[PATH_MAX];
    if (!path || !*path)
        return tool_err("write needs path", NULL);
    if (!under_cwd(path, ap))
        return tool_err("invalid path: ", path);
    if (is_hardlink(ap))
        return tool_err("refusing hard-linked file: ", path);
    /* Follow existing symlink target like ds4 (write through the link). */
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", ap);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = 0;
        if (dir[0]) {
            /* create parents */
            char tmp[PATH_MAX];
            snprintf(tmp, sizeof(tmp), "%s", dir);
            for (char *p = tmp + 1; *p; p++) {
                if (*p == '/') {
                    *p = 0;
                    mkdir(tmp, 0755);
                    *p = '/';
                }
            }
            mkdir(tmp, 0755);
        }
    }
    char tmp[PATH_MAX + 8];
    snprintf(tmp, sizeof(tmp), "%s.tmp-%d", ap, (int)getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return tool_err("cannot write file: ", path);
    size_t n = content ? strlen(content) : 0;
    if (n && fwrite(content, 1, n, f) != n) {
        fclose(f);
        remove(tmp);
        return tool_err("write failed: ", path);
    }
    if (fclose(f)) {
        remove(tmp);
        return tool_err("write failed: ", path);
    }
    if (rename(tmp, ap)) {
        remove(tmp);
        return tool_err("cannot replace file: ", path);
    }
    char ok[PATH_MAX + 32];
    snprintf(ok, sizeof(ok), "wrote %s (%zu bytes)", path, n);
    return xstrdup(ok);
}

static char *tool_edit(const char *path, const char *old_t, const char *new_t, bool allow_upto) {
    char ap[PATH_MAX];
    if (!path || !*path || !old_t || !new_t)
        return tool_err("edit needs path, old and new", NULL);
    if (!under_cwd(path, ap))
        return tool_err("invalid path: ", path);
    if (is_hardlink(ap))
        return tool_err("refusing hard-linked file: ", path);
    FILE *f = fopen(ap, "rb");
    if (!f)
        return tool_err("cannot open file: ", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n < 0 || n > 8 * 1024 * 1024) {
        fclose(f);
        return tool_err("file too large: ", path);
    }
    char *data = xmalloc((size_t)n + 1);
    if (n && fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data);
        fclose(f);
        return tool_err("read failed: ", path);
    }
    data[n] = 0;
    fclose(f);
    char *result = NULL;
    if (allow_upto) {
        const char *up = strstr(old_t, "[upto]");
        if (up) {
            size_t hl = (size_t)(up - old_t);
            const char *tail = up + 7;
            /* head must match once, tail once after head. */
            char *head = xstrndup(old_t, hl);
            const char *h1 = strstr(data, head);
            const char *h2 = h1 ? strstr(h1 + 1, head) : NULL;
            const char *t1 = h1 ? strstr(h1 + hl, tail) : NULL;
            const char *t2 = t1 ? strstr(t1 + 1, tail) : NULL;
            if (!h1 || h2 || !t1 || t2 || !*head || !*tail) {
                free(head);
                free(data);
                return tool_err("edit fails: head/tail must each match exactly once", NULL);
            }
            size_t hn = strlen(head), tn = strlen(tail);
            size_t nn = (size_t)n - ((size_t)(t1 + tn - h1)) + strlen(new_t);
            result = xmalloc(nn + 1);
            size_t pre = (size_t)(h1 - data);
            memcpy(result, data, pre);
            memcpy(result + pre, new_t, strlen(new_t));
            strcpy(result + pre + strlen(new_t), t1 + tn);
            (void)hn;
            free(head);
            free(data);
            /* atomic replace */
            char tmp[PATH_MAX + 8];
            snprintf(tmp, sizeof(tmp), "%s.tmp-%d", ap, (int)getpid());
            FILE *o = fopen(tmp, "wb");
            if (!o) {
                free(result);
                return tool_err("cannot write file: ", path);
            }
            fwrite(result, 1, strlen(result), o);
            fclose(o);
            if (rename(tmp, ap)) {
                remove(tmp);
                free(result);
                return tool_err("cannot replace file: ", path);
            }
            char ok[128];
            snprintf(ok, sizeof(ok), "edited %s (anchored [upto])", path);
            free(result);
            return xstrdup(ok);
        }
    }
    /* exact-once */
    const char *a = strstr(data, old_t);
    if (!a || !*old_t) {
        free(data);
        return tool_err("edit fails: old text not found exactly once", NULL);
    }
    if (strstr(a + 1, old_t)) {
        free(data);
        return tool_err("edit fails: old text matches multiple times", NULL);
    }
    size_t nn = (size_t)n - strlen(old_t) + strlen(new_t);
    result = xmalloc(nn + 1);
    size_t pre = (size_t)(a - data);
    memcpy(result, data, pre);
    memcpy(result + pre, new_t, strlen(new_t));
    strcpy(result + pre + strlen(new_t), a + strlen(old_t));
    free(data);
    char tmp[PATH_MAX + 8];
    snprintf(tmp, sizeof(tmp), "%s.tmp-%d", ap, (int)getpid());
    FILE *o = fopen(tmp, "wb");
    if (!o) {
        free(result);
        return tool_err("cannot write file: ", path);
    }
    fwrite(result, 1, strlen(result), o);
    fclose(o);
    if (rename(tmp, ap)) {
        remove(tmp);
        free(result);
        return tool_err("cannot replace file: ", path);
    }
    char ok[128];
    snprintf(ok, sizeof(ok), "edited %s", path);
    free(result);
    return xstrdup(ok);
}

typedef struct {
    char *query;
    char *path;
    char *mode;
    char *glob;
    long context;
    long max_results;
    bool case_sens;
} search_args_t;

static char *tool_search(search_args_t *a) {
    if (!a->query || !*a->query)
        return tool_err("search needs query", NULL);
    const char *root = (a->path && *a->path) ? a->path : ".";
    bool regex_mode = a->mode && !strcmp(a->mode, "regex");
    long ctx = a->context;
    if (ctx < 0)
        ctx = 0;
    if (ctx > 5)
        ctx = 5;
    long maxr = a->max_results > 0 ? a->max_results : 50;
    if (maxr < 1)
        maxr = 1;
    if (maxr > 500)
        maxr = 500;
    regex_t re;
    bool have_re = false;
    if (regex_mode) {
        int fl = REG_EXTENDED | (a->case_sens ? 0 : REG_ICASE);
        if (regcomp(&re, a->query, fl)) {
            return tool_err("invalid regex: ", a->query);
        }
        have_re = true;
    }
    char *out = NULL;
    size_t ol = 0, oc = 0;
    long shown = 0;
    bool incomplete = false;
    /* Walk with opendir recursion (skip .git, nested symlinks). */
    char stack[64][PATH_MAX];
    int depth = 0;
    char ap[PATH_MAX];
    if (!under_cwd(root, ap)) {
        if (have_re)
            regfree(&re);
        return tool_err("invalid path: ", root);
    }
    snprintf(stack[0], sizeof(stack[0]), "%s", ap);
    depth = 1;
    char qlow[2048];
    snprintf(qlow, sizeof(qlow), "%s", a->query);
    for (char *c = qlow; *c; c++)
        *c = (char)tolower((unsigned char)*c);
    while (depth > 0 && shown < maxr) {
        char dir[PATH_MAX];
        snprintf(dir, sizeof(dir), "%s", stack[--depth]);
        DIR *d = opendir(dir);
        if (!d) {
            incomplete = true;
            continue;
        }
        struct dirent *e;
        while ((e = readdir(d)) && shown < maxr) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..") || !strcmp(e->d_name, ".git"))
                continue;
            char full[PATH_MAX];
            if (strlen(dir) + 1 + strlen(e->d_name) >= sizeof(full)) { incomplete = true; continue; }
            snprintf(full, sizeof(full), "%s/%s", dir, e->d_name);
            struct stat st;
            if (lstat(full, &st))
                continue;
            if (S_ISLNK(st.st_mode))
                continue; /* skip nested symlinks */
            if (S_ISDIR(st.st_mode)) {
                if (depth < 64) {
                    snprintf(stack[depth++], PATH_MAX, "%s", full);
                } else
                    incomplete = true;
                continue;
            }
            if (!S_ISREG(st.st_mode) || st.st_size > 2 * 1024 * 1024)
                continue;
            if (a->glob && *a->glob && fnmatch(a->glob, e->d_name, 0))
                continue;
            FILE *f = fopen(full, "rb");
            if (!f) {
                incomplete = true;
                continue;
            }
            char *line = NULL;
            size_t ll = 0;
            ssize_t nl;
            long lineno = 0;
            /* keep context ring */
            char *ring[5] = {0};
            int ri = 0;
            while ((nl = getline(&line, &ll, f)) >= 0) {
                lineno++;
                if (memchr(line, 0, (size_t)nl))
                    break; /* binary */
                bool hit = false;
                if (have_re) {
                    hit = !regexec(&re, line, 0, NULL, 0);
                } else if (a->case_sens) {
                    hit = strstr(line, a->query) != NULL;
                } else {
                    char *low = xstrndup(line, (size_t)nl);
                    for (char *c = low; *c; c++)
                        *c = (char)tolower((unsigned char)*c);
                    hit = strstr(low, qlow) != NULL;
                    free(low);
                }
                if (hit) {
                    /* emit with context */
                    char head[PATH_MAX + 64];
                    const char *rel = full;
                    char cwd[PATH_MAX];
                    if (getcwd(cwd, sizeof(cwd)) && !strncmp(full, cwd, strlen(cwd)))
                        rel = full + strlen(cwd) + (full[strlen(cwd)] == '/' ? 1 : 0);
                    /* prior context */
                    for (long k = ctx; k > 0; k--) {
                        int idx = (ri - (int)k + 20) % 5;
                        if (ctx > 0 && ring[idx]) {
                            snprintf(head, sizeof(head), "%s:%ld-%s", rel, lineno - k, "");
                            buf_add(&out, &ol, &oc, head, strlen(head));
                            buf_add(&out, &ol, &oc, ring[idx], strlen(ring[idx]));
                        }
                    }
                    snprintf(head, sizeof(head), "%s:%ld: ", rel, lineno);
                    buf_add(&out, &ol, &oc, head, strlen(head));
                    size_t ln2 = strlen(line);
                    if (ln2 > 500) {
                        line[500] = 0;
                        ln2 = 500;
                    }
                    buf_add(&out, &ol, &oc, line, ln2);
                    if (ol && out[ol - 1] != '\n')
                        buf_add(&out, &ol, &oc, "\n", 1);
                    shown++;
                    if (shown >= maxr)
                        break;
                    /* following context handled by ring naturally on next hits;
                     * for simplicity emit next ctx lines eagerly */
                    for (long k = 0; k < ctx && shown < maxr; k++) {
                        ssize_t n2 = getline(&line, &ll, f);
                        if (n2 < 0)
                            break;
                        lineno++;
                        snprintf(head, sizeof(head), "%s:%ld-%s", rel, lineno, "");
                        buf_add(&out, &ol, &oc, head, strlen(head));
                        buf_add(&out, &ol, &oc, line, strlen(line));
                    }
                }
                if (ctx > 0) {
                    free(ring[ri]);
                    ring[ri] = xstrndup(line, strlen(line));
                    ri = (ri + 1) % 5;
                }
                if (ol > 120 * 1024)
                    break;
            }
            for (int k = 0; k < 5; k++)
                free(ring[k]);
            free(line);
            fclose(f);
            if (ol > 120 * 1024) {
                incomplete = true;
                break;
            }
        }
        closedir(d);
    }
    if (have_re)
        regfree(&re);
    if (!shown) {
        free(out);
        return xstrdup("no matches");
    }
    if (incomplete) {
        buf_add(&out, &ol, &oc, "(incomplete coverage: skipped unreadable paths)\n", 47);
    }
    return out;
}

static char *tool_list(const char *path) {
    char ap[PATH_MAX];
    const char *p = (path && *path) ? path : ".";
    if (!under_cwd(p, ap))
        return tool_err("invalid path: ", p);
    DIR *d = opendir(ap);
    if (!d)
        return tool_err("cannot list directory: ", p);
    char *out = NULL;
    size_t ol = 0, oc = 0;
    struct dirent *e;
    int n = 0;
    while ((e = readdir(d)) && n < 200) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char full[PATH_MAX];
        if (strlen(ap) + 1 + strlen(e->d_name) >= sizeof(full)) continue;
        snprintf(full, sizeof(full), "%s/%s", ap, e->d_name);
        struct stat st;
        if (stat(full, &st))
            continue;
        char line[PATH_MAX + 64];
        if (S_ISDIR(st.st_mode))
            snprintf(line, sizeof(line), "%s/\n", e->d_name);
        else
            snprintf(line, sizeof(line), "%s (%lld bytes)\n", e->d_name, (long long)st.st_size);
        buf_add(&out, &ol, &oc, line, strlen(line));
        n++;
    }
    closedir(d);
    if (!out)
        return xstrdup("(empty directory)");
    return out;
}

static char *tool_bash(const char *command, long timeout_sec) {
    if (!command || !*command)
        return tool_err("bash needs command", NULL);
    if (strlen(command) > 8000)
        return tool_err("command too long", NULL);
    long to = timeout_sec > 0 ? timeout_sec : 30;
    if (to < 1)
        to = 1;
    if (to > 600)
        to = 600;
    int pipefd[2];
    if (pipe(pipefd))
        return tool_err("cannot run command", NULL);
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return tool_err("cannot fork", NULL);
    }
    if (pid == 0) {
        /* child: run through sh, merge stderr, enforce timeout via alarm */
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        alarm((unsigned)to);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    char *out = NULL;
    size_t ol = 0, oc = 0;
    char tmp[4096];
    ssize_t n;
    time_t deadline = time(NULL) + to + 2;
    FILE *f = fdopen(pipefd[0], "r");
    if (f) {
        /* blocking read; child alarm bounds execution */
        while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
            if (ol + (size_t)n > 256 * 1024) {
                size_t keep = 256 * 1024 - ol;
                if (keep)
                    buf_add(&out, &ol, &oc, tmp, keep);
                break;
            }
            buf_add(&out, &ol, &oc, tmp, (size_t)n);
            if (time(NULL) > deadline)
                break;
        }
        fclose(f);
    } else {
        close(pipefd[0]);
    }
    int status = 0;
    /* wait with timeout */
    time_t wend = time(NULL) + to + 5;
    pid_t w;
    while ((w = waitpid(pid, &status, WNOHANG)) == 0 && time(NULL) < wend)
        usleep(50000);
    if (w == 0) {
        kill(pid, 9);
        waitpid(pid, &status, 0);
        buf_add(&out, &ol, &oc, "\n[timeout]\n", 11);
    }
    char tail[128];
    if (WIFEXITED(status))
        snprintf(tail, sizeof(tail), "\n[exit code: %d]\n", WEXITSTATUS(status));
    else if (WIFSIGNALED(status))
        snprintf(tail, sizeof(tail), "\n[signal: %d]\n", WTERMSIG(status));
    else
        snprintf(tail, sizeof(tail), "\n[done]\n");
    buf_add(&out, &ol, &oc, tail, strlen(tail));
    if (!out)
        return xstrdup("[exit code: 0]\n");
    return out;
}

/* ---------- DSML parsing (agent tool protocol) ---------- */

typedef struct {
    char *name;
    char *value;
} targ_t;

typedef struct {
    char *name;
    targ_t *args;
    int argc, cap;
} tcall_t;

typedef struct {
    tcall_t *v;
    int len, cap;
    char error[256];
} tcalls_t;

static void tcalls_free(tcalls_t *t) {
    for (int i = 0; i < t->len; i++) {
        free(t->v[i].name);
        for (int j = 0; j < t->v[i].argc; j++) {
            free(t->v[i].args[j].name);
            free(t->v[i].args[j].value);
        }
        free(t->v[i].args);
    }
    free(t->v);
    t->v = NULL;
    t->len = t->cap = 0;
}

static void unescape_param(char *s) {
    /* &lt;/loina_param> -> </loina_param>, &amp;lt; -> &lt; */
    char *w = s;
    for (char *r = s; *r;) {
        if (!strncmp(r, "&lt;/loina_param>", 18)) {
            memcpy(w, "</loina_param>", 15);
            w += 15;
            r += 18;
        } else if (!strncmp(r, "&amp;lt;", 8)) {
            memcpy(w, "&lt;", 4);
            w += 4;
            r += 8;
        } else {
            *w++ = *r++;
        }
    }
    *w = 0;
}

static int parse_calls(const char *reply, tcalls_t *out) {
    memset(out, 0, sizeof(*out));
    const char *block = strstr(reply, "<loina_tool_calls>");
    if (!block)
        return 0;
    const char *bend = strstr(block, "</loina_tool_calls>");
    if (!bend) {
        snprintf(out->error, sizeof(out->error), "unclosed <loina_tool_calls>");
        return -1;
    }
    const char *p = block + strlen("<loina_tool_calls>");
    int count = 0;
    while (count < 4) {
        const char *inv = strstr(p, "<loina_invoke");
        if (!inv || inv >= bend)
            break;
        const char *nq = strstr(inv, "name=\"");
        if (!nq || nq >= bend) {
            snprintf(out->error, sizeof(out->error), "invoke without name");
            goto fail;
        }
        nq += 6;
        const char *nq2 = strchr(nq, '"');
        if (!nq2 || nq2 >= bend) {
            snprintf(out->error, sizeof(out->error), "unterminated tool name");
            goto fail;
        }
        char *name = xstrndup(nq, (size_t)(nq2 - nq));
        const char *gt = strchr(nq2, '>');
        const char *iend = strstr(gt ? gt : nq2, "</loina_invoke>");
        if (!gt || !iend || iend > bend) {
            snprintf(out->error, sizeof(out->error), "unclosed invoke %.64s", name ? name : "?");
            free(name);
            goto fail;
        }
        if (out->len == out->cap) {
            out->cap = out->cap ? out->cap * 2 : 4;
            out->v = xrealloc(out->v, (size_t)out->cap * sizeof(*out->v));
        }
        tcall_t *c = &out->v[out->len++];
        memset(c, 0, sizeof(*c));
        c->name = name;
        const char *q = gt + 1;
        while (q < iend) {
            const char *pa = strstr(q, "<loina_param");
            if (!pa || pa >= iend)
                break;
            const char *an = strstr(pa, "name=\"");
            if (!an || an >= iend) {
                snprintf(out->error, sizeof(out->error), "param without name in %s", name);
                goto fail;
            }
            an += 6;
            const char *an2 = strchr(an, '"');
            if (!an2 || an2 >= iend) {
                snprintf(out->error, sizeof(out->error), "unterminated param name in %s", name);
                goto fail;
            }
            char *aname = xstrndup(an, (size_t)(an2 - an));
            const char *pgt = strchr(an2, '>');
            if (!pgt || pgt >= iend) {
                snprintf(out->error, sizeof(out->error), "unclosed param %.64s", aname ? aname : "?");
                free(aname);
                goto fail;
            }
            const char *pend = strstr(pgt, "</loina_param>");
            if (!pend || pend > iend) {
                free(aname);
                snprintf(out->error, sizeof(out->error), "unclosed param value");
                goto fail;
            }
            char *val = xstrndup(pgt + 1, (size_t)(pend - pgt - 1));
            unescape_param(val);
            /* trim one surrounding newline (ds4 convention) */
            while (*val == '\n')
                memmove(val, val + 1, strlen(val));
            size_t vl = strlen(val);
            while (vl && val[vl - 1] == '\n')
                val[--vl] = 0;
            if (c->argc == c->cap) {
                c->cap = c->cap ? c->cap * 2 : 4;
                c->args = xrealloc(c->args, (size_t)c->cap * sizeof(*c->args));
            }
            c->args[c->argc].name = aname;
            c->args[c->argc].value = val;
            c->argc++;
            q = pend + strlen("</loina_param>");
        }
        p = iend + strlen("</loina_invoke>");
        count++;
    }
    /* Trailing garbage after the block is the model's final text; tool-only
     * turns have none. The caller decides. */
    return out->len;
fail:
    tcalls_free(out);
    return -1;
}

static const char *arg_val(tcall_t *c, const char *name) {
    for (int i = 0; i < c->argc; i++)
        if (!strcmp(c->args[i].name, name))
            return c->args[i].value;
    return NULL;
}

static bool arg_bool(const char *v, bool def) {
    if (!v)
        return def;
    return !strcmp(v, "true") || !strcmp(v, "1") || !strcmp(v, "yes");
}

static long arg_long(const char *v, long def) {
    if (!v || !*v)
        return def;
    char *e;
    long n = strtol(v, &e, 10);
    return *e ? def : n;
}

/* ---------- agent configuration ---------- */

typedef struct {
    loina_config engine_cfg;
    char engine_dir[PATH_MAX];
    char tmpl_dir[PATH_MAX];
    const char *package_id;
    const char *model_file;
    const char *import;
    const char *session_id;
    const char *prompt;
    const char *prompt_file;
    const char *chdir_path;
    const char *vision_path;
    bool non_interactive;
    bool edit_upto;
    bool n_set;
    uint32_t max_tokens;
    bool thinking;
    unsigned backend;
    float temp, top_p, min_p;
    uint32_t seed;
    bool seed_set;
    int threads;
} agent_cfg_t;

static void agent_help(void) {
    puts("loina-agent - ds4-style coding agent on loina-core + llama.cpp\n"
         "Usage: loina-agent [options]\n"
         "  -p TEXT               Run one task non-interactively; otherwise REPL\n"
         "  --prompt-file FILE    Read the initial task from a file\n"
         "  --non-interactive     Never prompt; exit after tool loop\n"
         "  --edit-upto           Allow [upto] anchored edits\n"
         "  --chdir PATH          Run local tools in PATH\n"
         "  -m FILE               Import catalog model file (siblings if needed)\n"
         "  --package ID          Use an immutable package\n"
         "  --import DIRECTORY    Install selected package from local artifacts\n"
         "  --session ID          Resume a saved conversation\n"
         "  --cache-dir PATH      Packages, transcripts, KV (default .loina)\n"
         "  --ctx N               Context size, max 32768 (default 32768)\n"
         "  -n N                  Max tokens per turn (default 2048 thinking, 1024 plain)\n"
         "  --threads N           CPU threads\n"
         "  --backend NAME        CPU, Vulkan, OpenCL, Metal or CUDA\n"
         "  --temp F --top-p F --min-p F  Sampling\n"
         "  --seed N              Sampling seed\n"
         "  --thinking on|off     Reasoning when the package allows it (default on)\n"
         "  --vision FILE         Attach an image to the first task\n"
         "REPL: /help /think /nothink /save /list /switch ID /del ID /new /read FILE /quit\n"
         "Tools: read more write edit search list bash view_image web_search web_visit.");
}

/* Agent system template: overrides bundled system.jinja so the model learns
 * the local DSML tool protocol instead of the app web-tool text. */
static const char agent_system_jinja[] =
    "Today's date: {{ date }}.\n"
    "You are Loina, a coding agent running locally in the user's workspace. Use tools for local work. "
    "Avoid printing large files; create or edit files with tools, then summarize briefly.\n\n"
    "## Tools\n\n"
    "Invoke tools with exactly this shape (never inside <think>):\n\n"
    "<loina_tool_calls>\n"
    "<loina_invoke name=\"read\">\n"
    "<loina_param name=\"path\">src/main.c</loina_param>\n"
    "<loina_param name=\"start_line\">1</loina_param>\n"
    "<loina_param name=\"max_lines\">120</loina_param>\n"
    "</loina_invoke>\n"
    "</loina_tool_calls>\n\n"
    "Read returns a bounded chunk, not the whole file. For first looks prefer max_lines 80-160; "
    "if more lines are available, call more with count. Use whole=true only when asked for the complete file; "
    "it fails over 128 KiB. raw=true omits line numbers.\n"
    "Search is literal (default) or regex (POSIX extended). Defaults: path=., case_sensitive=true, context=0, max_results=50. "
    "context is 0-5, max_results 1-500. Skips .git and symlinks.\n"
    "bash runs a shell command (timeout_sec default 30, max 600). Long commands should set timeout_sec.\n"
    "Write and edit replace files atomically and refuse hard-linked files.\n"
    "Use edit with path, old, new. old must match exactly once. For inserts, anchor on unique text.\n"
    "Use view_image with path to see a local PNG/JPEG.\n"
    "Use web_search(query) for current facts and web_visit(url) for a known HTTPS page.\n\n"
    "### Schemas\n\n"
    "{\"name\":\"read\",\"params\":[\"path\",\"start_line\",\"max_lines\",\"whole\",\"raw\"]}\n"
    "{\"name\":\"more\",\"params\":[\"count\"]}\n"
    "{\"name\":\"write\",\"params\":[\"path\",\"content\"]}\n"
    "{\"name\":\"edit\",\"params\":[\"path\",\"old\",\"new\"]}\n"
    "{\"name\":\"search\",\"params\":[\"query\",\"path\",\"mode\",\"glob\",\"context\",\"max_results\",\"case_sensitive\"]}\n"
    "{\"name\":\"list\",\"params\":[\"path\"]}\n"
    "{\"name\":\"bash\",\"params\":[\"command\",\"timeout_sec\"]}\n"
    "{\"name\":\"view_image\",\"params\":[\"path\"]}\n"
    "{\"name\":\"web_search\",\"params\":[\"query\"]}\n"
    "{\"name\":\"web_visit\",\"params\":[\"url\"]}\n\n"
    "# Rules\n"
    "- State the target filename before editing; put path first.\n"
    "- Prefer read/search + edit over rewriting whole files.\n"
    "- Preserve system configuration unless asked otherwise.\n";

static bool write_agent_template(const char *dir) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/system.jinja", dir);
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    size_t n = strlen(agent_system_jinja);
    bool ok = fwrite(agent_system_jinja, 1, n, f) == n && !fclose(f);
    if (!ok)
        remove(path);
    return ok;
}

/* ---------- one inference turn ---------- */

typedef struct {
    loina_engine *engine;
    loina_session *session;
    agent_cfg_t *cfg;
    more_state_t *more;
} agent_t;

static char *run_turn(agent_t *a, const char *prompt, const char *vision_path, bool *cancelled) {
    loina_request r = loina_request_default(vision_path ? LOINA_VISION : LOINA_TEXT);
    r.session = a->session;
    r.prompt = prompt;
    r.input_path = vision_path;
    r.tools = false; /* agent DSML handled outside the core web loop */
    r.thinking = a->cfg->thinking;
    r.max_tokens = a->cfg->n_set ? a->cfg->max_tokens : (a->cfg->thinking ? 2048 : 1024);
    r.temperature = a->cfg->temp;
    r.top_p = a->cfg->top_p;
    r.min_p = a->cfg->min_p;
    if (a->cfg->seed_set)
        r.seed = a->cfg->seed;
    stream_t s = {.use_color = isatty(fileno(stdout)) != 0, .last_nl = true};
    loina_error err = {0};
    loina_job *job = loina_job_create(a->engine, &r, stream_event, &s, &err);
    if (!job) {
        fprintf(stderr, "%s\n", err.message);
        return NULL;
    }
    s.job = job;
    g_job = job;
    g_interrupted = 0;
    loina_status st = loina_job_run(job, &err);
    g_job = NULL;
    s_write(&s, NULL, 0, true);
    if (s.color_open) {
        fputs("\x1b[0m", stdout);
        s.color_open = false;
    }
    if (!s.last_nl) {
        putchar('\n');
        fflush(stdout);
    }
    if (s.have_metrics && !st) {
        bool ttyerr = isatty(fileno(stderr)) != 0;
        fprintf(stderr, "%sloina-agent: prefill: %.2f t/s, generation: %.2f t/s%s\n",
            ttyerr ? "\x1b[32m" : "", s.prefill, s.decode, ttyerr ? "\x1b[0m" : "");
    }
    char *result = NULL;
    if (!st) {
        result = xstrdup(loina_job_result(job));
    } else {
        if (st == LOINA_CANCELLED) {
            fprintf(stderr, "Interrupted.\n");
            if (cancelled)
                *cancelled = true;
        } else {
            fprintf(stderr, "%s\n", err.message);
        }
    }
    free(s.buf);
    loina_job_free(job);
    return result;
}

static char *run_vision_desc(agent_t *a, const char *image_path) {
    loina_request r = loina_request_default(LOINA_VISION);
    r.session = NULL; /* transient extraction, like loina_chat vision */
    r.prompt = "Describe this image and extract its visible text faithfully. Include details needed to answer questions about it.";
    r.input_path = image_path;
    r.tools = false;
    r.thinking = false;
    r.max_tokens = 1024;
    r.temperature = a->cfg->temp;
    stream_t s = {.use_color = false, .last_nl = true};
    loina_error err = {0};
    loina_job *job = loina_job_create(a->engine, &r, stream_event, &s, &err);
    if (!job) {
        fprintf(stderr, "%s\n", err.message);
        return NULL;
    }
    s.job = job;
    g_job = job;
    loina_status st = loina_job_run(job, &err);
    g_job = NULL;
    char *res = NULL;
    if (!st)
        res = xstrdup(loina_job_result(job));
    else
        fprintf(stderr, "%s\n", err.message);
    free(s.buf);
    loina_job_free(job);
    return res;
}

static char *dispatch_tool(agent_t *a, tcall_t *c) {
    const char *n = c->name;
    if (!strcmp(n, "read")) {
        return tool_read(arg_val(c, "path"), arg_long(arg_val(c, "start_line"), 1),
            arg_long(arg_val(c, "max_lines"), 0), arg_bool(arg_val(c, "whole"), false),
            arg_bool(arg_val(c, "raw"), false), a->more);
    }
    if (!strcmp(n, "more")) {
        return tool_more(arg_long(arg_val(c, "count"), 0), a->more);
    }
    if (!strcmp(n, "write")) {
        const char *p = arg_val(c, "path"), *co = arg_val(c, "content");
        if (!p || !co)
            return tool_err("write needs path and content", NULL);
        return tool_write(p, co);
    }
    if (!strcmp(n, "edit")) {
        const char *p = arg_val(c, "path"), *o = arg_val(c, "old"), *nw = arg_val(c, "new");
        if (!p || !o || !nw)
            return tool_err("edit needs path, old and new", NULL);
        return tool_edit(p, o, nw, a->cfg->edit_upto);
    }
    if (!strcmp(n, "search")) {
        search_args_t sa = {.query = (char *)arg_val(c, "query"), .path = (char *)arg_val(c, "path"),
            .mode = (char *)arg_val(c, "mode"), .glob = (char *)arg_val(c, "glob"),
            .context = arg_long(arg_val(c, "context"), 0),
            .max_results = arg_long(arg_val(c, "max_results"), 50),
            .case_sens = arg_bool(arg_val(c, "case_sensitive"), true)};
        return tool_search(&sa);
    }
    if (!strcmp(n, "list")) {
        return tool_list(arg_val(c, "path"));
    }
    if (!strcmp(n, "bash")) {
        return tool_bash(arg_val(c, "command"), arg_long(arg_val(c, "timeout_sec"), 30));
    }
    if (!strcmp(n, "view_image")) {
        const char *p = arg_val(c, "path");
        if (!p || !*p)
            return tool_err("view_image needs path", NULL);
        char *desc = run_vision_desc(a, p);
        if (!desc)
            return tool_err("cannot view image: ", p);
        size_t need = strlen(desc) + strlen(p) + 64;
        char *o = xmalloc(need);
        snprintf(o, need, "[Image %s]\n%s", p, desc);
        free(desc);
        return o;
    }
    if (!strcmp(n, "web_search") || !strcmp(n, "google_search")) {
        const char *q = arg_val(c, "query");
        if (!q || !*q)
            return tool_err("web_search needs query", NULL);
        /* Reuse the core DuckDuckGo URL shape via the agent bridge. */
        char url[8192];
        size_t u = 0;
        const char *pre = "https://html.duckduckgo.com/html/?q=";
        strcpy(url, pre);
        u = strlen(pre);
        for (const unsigned char *p = (const unsigned char *)q; *p && u + 4 < sizeof(url); p++) {
            if (isalnum(*p)) {
                url[u++] = (char)*p;
            } else {
                snprintf(url + u, 4, "%%%02X", *p);
                u += 3;
            }
        }
        url[u] = 0;
        char *text = NULL;
        loina_error err = {0};
        if (agent_web(NULL, url, &text, NULL, &err)) {
            char *e = tool_err(err.message[0] ? err.message : "web search failed", NULL);
            free(text);
            return e;
        }
        return text;
    }
    if (!strcmp(n, "web_visit") || !strcmp(n, "visit_page")) {
        const char *u = arg_val(c, "url");
        if (!u || strncmp(u, "https://", 8))
            return tool_err("web_visit needs an HTTPS url", NULL);
        char *text = NULL;
        loina_error err = {0};
        if (agent_web(NULL, u, &text, NULL, &err)) {
            char *e = tool_err(err.message[0] ? err.message : "web visit failed", NULL);
            free(text);
            return e;
        }
        return text;
    }
    /* bash_status/stop are ds4 background controls; this agent runs bash
     * synchronously, so report the contract instead of hanging. */
    if (!strcmp(n, "bash_status") || !strcmp(n, "bash_stop")) {
        return tool_err("background bash jobs are not used; call bash with timeout_sec", NULL);
    }
    return tool_err("unknown tool: ", n);
}

/* Run a full user task: up to 10 tool rounds, then the final answer. */
static int run_task(agent_t *a, const char *user_text, const char *vision_path) {
    char *prompt = xstrdup(user_text);
    const char *vision = vision_path;
    for (int round = 0; round < 11; round++) {
        if (g_interrupted) {
            free(prompt);
            return 130;
        }
        bool cancelled = false;
        char *reply = run_turn(a, prompt, vision, &cancelled);
        free(prompt);
        prompt = NULL;
        vision = NULL;
        if (!reply)
            return cancelled ? 130 : 1;
        tcalls_t calls;
        int nc = parse_calls(reply, &calls);
        /* Visible answer = reply minus tool blocks + think handling already
         * streamed. Decide by stripping tool blocks here. */
        char *visible = NULL;
        {
            size_t ol = 0, oc = 0;
            const char *p = reply;
            while (*p) {
                const char *b = strstr(p, "<loina_tool_calls>");
                if (!b) {
                    buf_add(&visible, &ol, &oc, p, strlen(p));
                    break;
                }
                buf_add(&visible, &ol, &oc, p, (size_t)(b - p));
                const char *e = strstr(b, "</loina_tool_calls>");
                p = e ? e + strlen("</loina_tool_calls>") : b + strlen("<loina_tool_calls>");
            }
            if (!visible)
                visible = xstrdup("");
            /* strip think for the empty check */
            char *chk = xstrdup(visible);
            /* crude: remove <think>..</think> for emptiness test */
            char *t1 = strstr(chk, "<think>");
            char *t2 = t1 ? strstr(t1, "</think>") : NULL;
            if (t1 && t2)
                memmove(t1, t2 + 8, strlen(t2 + 8) + 1);
            bool empty = true;
            for (char *c = chk; *c; c++)
                if (!isspace((unsigned char)*c)) {
                    empty = false;
                    break;
                }
            free(chk);
            if (nc < 0) {
                fprintf(stderr, "%s\n", calls.error);
                fprintf(stderr, "Retrying with the tool syntax reminder.\n");
                tcalls_free(&calls);
                free(visible);
                free(reply);
                prompt = xstrdup("Tool error: Invalid tool format. Use <loina_tool_calls><loina_invoke name=\"...\">"
                                 "<loina_param name=\"...\">...</loina_param></loina_invoke></loina_tool_calls> "
                                 "with no extra parameters. Only the documented tools exist.");
                /* feed as tool_response so the transcript stays a tool turn */
                char *wrapped = xmalloc(strlen(prompt) + 64);
                sprintf(wrapped, "<tool_response>\n%s\n</tool_response>", prompt);
                free(prompt);
                prompt = wrapped;
                continue;
            }
            if (nc == 0) {
                if (empty) {
                    fprintf(stderr, "The model did not finish an answer; try again.\n");
                    free(visible);
                    free(reply);
                    return 1;
                }
                free(visible);
                free(reply);
                return 0; /* final answer already streamed */
            }
            /* Announce tools ds4-style, then execute. */
            for (int i = 0; i < calls.len; i++) {
                if (isatty(fileno(stderr)))
                    fprintf(stderr, "\x1b[36m» %s\x1b[0m", calls.v[i].name);
                else
                    fprintf(stderr, "» %s", calls.v[i].name);
                for (int j = 0; j < calls.v[i].argc; j++) {
                    const char *vn = calls.v[i].args[j].name;
                    if (!strcmp(vn, "path") || !strcmp(vn, "command") || !strcmp(vn, "query") || !strcmp(vn, "url"))
                        fprintf(stderr, " %s=%.60s", vn, calls.v[i].args[j].value);
                }
                fputc('\n', stderr);
            }
            free(visible);
        }
        if (round == 10) {
            fprintf(stderr, "Tool budget reached; answering from evidence so far.\n");
            tcalls_free(&calls);
            free(reply);
            /* One last plain turn without tools. */
            prompt = xstrdup("Answer the user's question directly using the available evidence. Do not call tools.");
            char *w2 = xmalloc(strlen(prompt) + 64);
            sprintf(w2, "<tool_response>\n%s\n</tool_response>", prompt);
            free(prompt);
            prompt = w2;
            bool c2 = false;
            char *fin = run_turn(a, prompt, NULL, &c2);
            free(prompt);
            free(fin);
            tcalls_free(&calls);
            return c2 ? 130 : 0;
        }
        /* Execute and wrap observations for the MiniCPM <tool_response> path. */
        char *obs = NULL;
        size_t ol = 0, oc = 0;
        for (int i = 0; i < calls.len; i++) {
            char *r = dispatch_tool(a, &calls.v[i]);
            if (i)
                buf_add(&obs, &ol, &oc, "\n\n", 2);
            buf_add(&obs, &ol, &oc, r, strlen(r));
            free(r);
            if (ol > 96 * 1024)
                break;
        }
        tcalls_free(&calls);
        free(reply);
        if (!obs)
            obs = xstrdup("Tool error: no output");
        if (strlen(obs) > 96 * 1024) {
            obs[96 * 1024] = 0;
            char *t = xmalloc(strlen(obs) + 64);
            sprintf(t, "%s\n[truncated to 96 KiB]", obs);
            free(obs);
            obs = t;
        }
        prompt = xmalloc(strlen(obs) + 64);
        sprintf(prompt, "<tool_response>\n%s\n</tool_response>", obs);
        free(obs);
    }
    free(prompt);
    return 0;
}

static void list_cb(void *u, const char *id) {
    (void)u;
    puts(id);
}

static loina_session *agent_new_session(loina_engine *e, loina_error *err) {
    static unsigned seq;
    char id[64];
    snprintf(id, sizeof(id), "agent-%llu-%u", (unsigned long long)time(NULL), ++seq);
    return loina_session_create(e, id, err);
}

static void hist_append(const char *line) {
    const char *h = getenv("HOME");
    if (!h || !*h || !*line)
        return;
    char p[PATH_MAX];
    snprintf(p, sizeof(p), "%s/.loina_agent_history", h);
    FILE *f = fopen(p, "a");
    if (f) {
        fprintf(f, "%s\n", line);
        fclose(f);
    }
}

int main(int argc, char **argv) {
    agent_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.thinking = true;
    cfg.temp = 0.1f;
    cfg.top_p = 1.0f;
    const char *cache_dir = ".loina";
    const char *templates_override = NULL;
    uint32_t ctx = 0;
    bool show_models = false, show_device = false;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            const char *t = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[i + 1] : NULL;
            (void)t;
            agent_help();
            return 0;
        }
        if (!strcmp(a, "--non-interactive")) {
            cfg.non_interactive = true;
            continue;
        }
        if (!strcmp(a, "--edit-upto")) {
            cfg.edit_upto = true;
            continue;
        }
        if (!strcmp(a, "--models")) {
            show_models = true;
            continue;
        }
        if (!strcmp(a, "--device")) {
            show_device = true;
            continue;
        }
        const char *v = need_arg(&i, argc, argv, a);
        bool ok = true;
        uint32_t n;
        if (!strcmp(a, "-p"))
            cfg.prompt = v;
        else if (!strcmp(a, "--prompt-file"))
            cfg.prompt_file = v;
        else if (!strcmp(a, "--chdir"))
            cfg.chdir_path = v;
        else if (!strcmp(a, "-m"))
            cfg.model_file = v;
        else if (!strcmp(a, "--package"))
            cfg.package_id = v;
        else if (!strcmp(a, "--import"))
            cfg.import = v;
        else if (!strcmp(a, "--session"))
            cfg.session_id = v;
        else if (!strcmp(a, "--cache-dir"))
            cache_dir = v;
        else if (!strcmp(a, "--templates"))
            templates_override = v;
        else if (!strcmp(a, "--vision"))
            cfg.vision_path = v;
        else if (!strcmp(a, "--ctx")) {
            ok = parse_u32(v, &n) && n >= 512 && n <= 32768;
            if (ok)
                ctx = n;
        } else if (!strcmp(a, "-n")) {
            ok = parse_u32(v, &cfg.max_tokens) && cfg.max_tokens > 0;
            cfg.n_set = ok;
        } else if (!strcmp(a, "--threads"))
            cfg.threads = parse_int_exit(v, a);
        else if (!strcmp(a, "--backend")) {
            if (!strcmp(v, "CPU"))
                cfg.backend = LOINA_CPU;
            else if (!strcmp(v, "Vulkan"))
                cfg.backend = LOINA_VULKAN;
            else if (!strcmp(v, "OpenCL"))
                cfg.backend = LOINA_OPENCL;
            else if (!strcmp(v, "Metal"))
                cfg.backend = LOINA_METAL;
            else if (!strcmp(v, "CUDA"))
                cfg.backend = LOINA_CUDA;
            else
                ok = false;
        } else if (!strcmp(a, "--temp"))
            cfg.temp = parse_float_exit(v, a, 0.0f, 10.0f);
        else if (!strcmp(a, "--top-p"))
            cfg.top_p = parse_float_exit(v, a, 0.0f, 1.0f);
        else if (!strcmp(a, "--min-p"))
            cfg.min_p = parse_float_exit(v, a, 0.0f, 1.0f);
        else if (!strcmp(a, "--seed")) {
            ok = parse_u32(v, &cfg.seed);
            cfg.seed_set = ok;
        } else if (!strcmp(a, "--thinking")) {
            ok = !strcmp(v, "on") || !strcmp(v, "off");
            cfg.thinking = !strcmp(v, "on");
        } else {
            fprintf(stderr, "loina-agent: unknown option: %s\n", a);
            return 2;
        }
        if (!ok) {
            fprintf(stderr, "loina-agent: invalid value for %s: %s\n", a, v);
            return 2;
        }
    }
    if (cfg.chdir_path && chdir(cfg.chdir_path)) {
        fprintf(stderr, "loina-agent: cannot chdir to %s\n", cfg.chdir_path);
        return 1;
    }
    char *owned_prompt = NULL;
    if (cfg.prompt_file) {
        FILE *f = fopen(cfg.prompt_file, "rb");
        if (!f) {
            fprintf(stderr, "loina-agent: cannot open %s\n", cfg.prompt_file);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        if (sz < 0 || sz > 16 * 1024 * 1024) {
            fclose(f);
            fprintf(stderr, "loina-agent: bad prompt file\n");
            return 1;
        }
        owned_prompt = xmalloc((size_t)sz + 1);
        if (sz && fread(owned_prompt, 1, (size_t)sz, f) != (size_t)sz) {
            fclose(f);
            fprintf(stderr, "loina-agent: cannot read prompt file\n");
            return 1;
        }
        owned_prompt[sz] = 0;
        fclose(f);
        cfg.prompt = owned_prompt;
    }
    if (cfg.prompt)
        cfg.non_interactive = true;

    loina_config ecfg = {.directory = cache_dir, .web = agent_web, .context = ctx};
    if (cfg.threads)
        ecfg.device.threads = cfg.threads;
    /* Agent template override: local tool protocol instead of app web text. */
    char auto_tmpl[PATH_MAX] = {0};
    if (!templates_override) {
        snprintf(auto_tmpl, sizeof(auto_tmpl), "%s/agent-templates", cache_dir);
        /* mkdir -p */
        char tmp[PATH_MAX];
        snprintf(tmp, sizeof(tmp), "%s", auto_tmpl);
        for (char *p = tmp + 1; *p; p++) {
            if (*p == '/') {
                *p = 0;
                mkdir(tmp, 0755);
                *p = '/';
            }
        }
        mkdir(tmp, 0755);
        if (!write_agent_template(auto_tmpl)) {
            fprintf(stderr, "loina-agent: cannot write agent templates\n");
            return 1;
        }
        ecfg.template_directory = auto_tmpl;
    } else {
        ecfg.template_directory = templates_override;
    }
    signal(SIGINT, on_sigint);
    loina_error err = {0};
    loina_engine *engine = loina_open(&ecfg, &err);
    if (!engine) {
        fprintf(stderr, "%s\n", err.message);
        free(owned_prompt);
        return 1;
    }
    if (cfg.backend && loina_set_backend(engine, cfg.backend, &err)) {
        fprintf(stderr, "%s\n", err.message);
        loina_close(engine);
        free(owned_prompt);
        return 1;
    }
    if (cfg.package_id) {
        const loina_package *p = loina_catalog_find(cfg.package_id);
        if (!p) {
            fprintf(stderr, "Unknown package; see --models\n");
            loina_close(engine);
            free(owned_prompt);
            return 1;
        }
    }
    if (cfg.model_file) {
        const char *base = strrchr(cfg.model_file, '/');
        base = base ? base + 1 : cfg.model_file;
        const loina_package *found = NULL;
        for (size_t i = 0; i < loina_catalog_count(); i++) {
            const loina_package *p = loina_catalog_get(i);
            if ((p->feature == LOINA_TEXT || p->feature == LOINA_VISION) && !strcmp(p->files[0].name, base)) {
                found = p;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Unknown model file; see --models\n");
            loina_close(engine);
            free(owned_prompt);
            return 1;
        }
        if (!loina_package_installed(engine, found)) {
            char dir[PATH_MAX];
            snprintf(dir, sizeof(dir), "%s", cfg.model_file);
            char *sl = strrchr(dir, '/');
            if (sl)
                *sl = 0;
            else
                strcpy(dir, ".");
            if (loina_install(engine, found, dir, NULL, &err)) {
                fprintf(stderr, "%s\n", err.message);
                loina_close(engine);
                free(owned_prompt);
                return 1;
            }
        }
        if (cfg.package_id && strcmp(cfg.package_id, found->id)) {
            fprintf(stderr, "Conflicting --package and -m\n");
            loina_close(engine);
            free(owned_prompt);
            return 1;
        }
        cfg.package_id = found->id;
    }
    if (cfg.import) {
        const loina_package *p = cfg.package_id ? loina_catalog_find(cfg.package_id)
                                                : loina_resolve(engine, LOINA_TEXT, &err);
        if (!p || loina_install(engine, p, cfg.import, NULL, &err)) {
            fprintf(stderr, "%s\n", err.message);
            loina_close(engine);
            free(owned_prompt);
            return 1;
        }
    }
    if (show_device) {
        const loina_device *d = loina_get_device(engine);
        printf("RAM: %llu; available: %llu; free storage: %llu; threads: %d; backends: 0x%x\n",
            (unsigned long long)d->ram_bytes, (unsigned long long)d->available_ram_bytes,
            (unsigned long long)d->storage_bytes, d->threads, d->backends);
    }
    if (show_models) {
        for (size_t i = 0; i < loina_catalog_count(); i++) {
            const loina_package *p = loina_catalog_get(i);
            printf("%s\n  %s | %s | %llu bytes | %s\n", p->id, p->name, p->license,
                (unsigned long long)loina_package_bytes(p),
                loina_package_installed(engine, p) ? "installed" : "not installed");
        }
    }
    if (show_models || show_device) {
        loina_close(engine);
        free(owned_prompt);
        return 0;
    }

    loina_session *session = cfg.session_id ? loina_session_load(engine, cfg.session_id, &err)
                                            : agent_new_session(engine, &err);
    if (!session) {
        fprintf(stderr, "%s\n", err.message);
        loina_close(engine);
        free(owned_prompt);
        return 1;
    }
    more_state_t more = {0};
    agent_t agent = {.engine = engine, .session = session, .cfg = &cfg, .more = &more};
    int rc = 0;
    if (cfg.non_interactive) {
        if (!cfg.prompt) {
            fprintf(stderr, "loina-agent: --non-interactive needs -p\n");
            rc = 2;
        } else {
            rc = run_task(&agent, cfg.prompt, cfg.vision_path);
        }
    } else {
        bool tty = isatty(fileno(stdin)) != 0;
        /* ds4-style welcome: colored name + thought-balloon instead of the
         * whale (ds4) or crystal ball (quarkstar). */
        char ctxs[32];
        uint32_t eff = ctx ? ctx : 32768;
        if (eff % 1000 == 0)
            snprintf(ctxs, sizeof(ctxs), "%uk", eff / 1000);
        else
            snprintf(ctxs, sizeof(ctxs), "%.1fk", (double)eff / 1000.0);
        if (isatty(fileno(stdout)))
            printf("\x1b[1;97mLoina\x1b[0m \xf0\x9f\x92\xad Agent, context %s tokens\n", ctxs);
        else
            printf("Loina Agent, context %s tokens\n", ctxs);
        puts("Tools run locally; web via curl bridge. /help for commands.");
        char line[16384];
        while (fputs(tty ? "loina> " : "", stdout), fflush(stdout), fgets(line, sizeof(line), stdin)) {
            line[strcspn(line, "\r\n")] = 0;
            char *cmd = line;
            while (*cmd && isspace((unsigned char)*cmd))
                cmd++;
            char *e = cmd + strlen(cmd);
            while (e > cmd && isspace((unsigned char)e[-1]))
                *--e = 0;
            if (!*cmd)
                continue;
            hist_append(cmd);
            if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit"))
                break;
            if (!strcmp(cmd, "/help")) {
                agent_help();
                puts("Agent tools run locally in the current directory (see --chdir).");
                continue;
            }
            if (!strcmp(cmd, "/think")) {
                cfg.thinking = true;
                puts("Thinking on (when the package allows it).");
                continue;
            }
            if (!strcmp(cmd, "/nothink")) {
                cfg.thinking = false;
                puts("Thinking off.");
                continue;
            }
            if (!strcmp(cmd, "/save")) {
                if (loina_session_save(session, &err))
                    fprintf(stderr, "%s\n", err.message);
                else
                    puts(loina_session_id(session));
                continue;
            }
            if (!strcmp(cmd, "/list")) {
                loina_session_list(engine, list_cb, NULL, &err);
                continue;
            }
            if (!strncmp(cmd, "/del ", 5)) {
                if (loina_session_delete(engine, cmd + 5, &err))
                    fprintf(stderr, "%s\n", err.message);
                continue;
            }
            if (!strcmp(cmd, "/new") || !strncmp(cmd, "/switch ", 8)) {
                loina_session *nx = !strcmp(cmd, "/new") ? agent_new_session(engine, &err)
                                                         : loina_session_load(engine, cmd + 8, &err);
                if (nx) {
                    loina_session_free(session);
                    session = nx;
                    agent.session = session;
                    memset(&more, 0, sizeof(more));
                } else
                    fprintf(stderr, "%s\n", err.message);
                continue;
            }
            if (!strncmp(cmd, "/read ", 6)) {
                char *rp = cmd + 6;
                while (*rp && isspace((unsigned char)*rp))
                    rp++;
                if (!*rp) {
                    fprintf(stderr, "/read needs a file path\n");
                    continue;
                }
                /* Image -> vision observation; text -> document turn. */
                FILE *tf = fopen(rp, "rb");
                bool img = false;
                if (tf) {
                    unsigned char mg[12] = {0};
                    size_t nn = fread(mg, 1, 12, tf);
                    fclose(tf);
                    if ((nn >= 8 && !memcmp(mg, "\x89PNG\r\n\x1a\n", 8)) ||
                        (nn >= 2 && mg[0] == 0xFF && mg[1] == 0xD8))
                        img = true;
                    else {
                        const char *dot = strrchr(rp, '.');
                        if (dot && (!strcasecmp(dot, ".png") || !strcasecmp(dot, ".jpg") || !strcasecmp(dot, ".jpeg")))
                            img = true;
                    }
                }
                if (img) {
                    rc = run_task(&agent, "Describe this attached image and extract its text.", rp);
                } else {
                    FILE *f = fopen(rp, "rb");
                    if (!f) {
                        fprintf(stderr, "Cannot open %s\n", rp);
                        continue;
                    }
                    fseek(f, 0, SEEK_END);
                    long sz = ftell(f);
                    rewind(f);
                    if (sz < 0 || sz > 1024 * 1024) {
                        fclose(f);
                        fprintf(stderr, "Text limit is 1 MB: %s\n", rp);
                        continue;
                    }
                    char *d = xmalloc((size_t)sz + 1);
                    if (sz && fread(d, 1, (size_t)sz, f) != (size_t)sz) {
                        free(d);
                        fclose(f);
                        continue;
                    }
                    d[sz] = 0;
                    fclose(f);
                    const char *b = strrchr(rp, '/');
                    b = b ? b + 1 : rp;
                    char *task = xmalloc((size_t)sz + strlen(b) + 64);
                    sprintf(task, "[Document: %s]\n%s", b, d);
                    free(d);
                    rc = run_task(&agent, task, NULL);
                    free(task);
                }
                if (rc == 130) {
                    g_interrupted = 0;
                    rc = 0;
                } else if (rc) {
                    fprintf(stderr, "task failed\n");
                    rc = 0;
                }
                continue;
            }
            if (cmd[0] == '/') {
                fprintf(stderr, "unknown command: %s\n", cmd);
                continue;
            }
            int tr = run_task(&agent, cmd, NULL);
            if (tr == 130) {
                g_interrupted = 0;
            } else if (tr) {
                fprintf(stderr, "task failed\n");
            }
        }
    }
    loina_session_free(session);
    loina_close(engine);
    free(owned_prompt);
    return rc;
}

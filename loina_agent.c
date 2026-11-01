#include "loina.h"
#include "loina_terminal.h"
#include "loina_renderer.h"
#include "loina_bash.h"
#include "loina_storage.h"
#include "loina_agent_tools.h"
#include "ds4_tool_text.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fnmatch.h>
#include <fcntl.h>
#ifdef __linux__
#include <sys/xattr.h>
#elif defined(__APPLE__)
#include <copyfile.h>
#endif
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

/* Host tools use the model's native protocol; the core keeps app tools separate. */

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

/* ---------- streaming reasoning and native tool visualization ---------- */

typedef struct {
    loina_job *job;
    bool use_color;
    unsigned context;
    bool in_think;
    bool last_nl;
    bool in_tool;
    bool native_tool;
    char tool_tag[256], tool_name[64], tool_param[64];
    loina_renderer *renderer;
    size_t tool_tag_len;
    bool tool_value, tool_first, tool_newline;
    int streamed_tools;
    char pending[32];
    size_t pending_len;
    double prefill, decode;
    bool have_metrics;
} stream_t;



static loina_renderer *s_renderer(stream_t *s) {
    if (!s->renderer) {
        s->renderer = loina_renderer_new(s->use_color);
        loina_renderer_context(s->renderer, s->context);
    }
    return s->renderer;
}

static bool starts(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n >= m && !memcmp(p, s, m);
}

static bool partial(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n < m && !memcmp(s, p, n);
}

static void tool_stream_char(stream_t *s, char c) {
    loina_renderer_param_byte(s_renderer(s), c);
    s->last_nl = loina_renderer_last_nl(s->renderer);
}

static void tool_stream_value(stream_t *s, char c) {
    if (!s->tool_value) return;
    if (s->tool_first) {
        s->tool_first = false;
        if (c == '\n') return;
    }
    /* Hold just the delimiter newline; embedded blank lines still stream. */
    if (s->tool_newline) tool_stream_char(s, '\n');
    s->tool_newline = c == '\n';
    if (c != '\n') tool_stream_char(s, c);
}

static void tool_stream_end(stream_t *s) {
    s->tool_value = s->tool_newline = false;
    loina_renderer_tool_end(s_renderer(s));
    s->last_nl = loina_renderer_last_nl(s->renderer);
    s->tool_name[0] = s->tool_param[0] = 0;
}

static void tool_stream_tag(stream_t *s) {
    char *tag = s->tool_tag;
    if (!strncmp(tag, "<function=", 10)) {
        size_t n = strlen(tag) - 11;
        snprintf(s->tool_name, sizeof(s->tool_name), "%.*s", (int)n, tag + 10);
        loina_renderer_tool_start(s_renderer(s));
        loina_renderer_tool_name(s->renderer, s->tool_name);
        s->streamed_tools++;
    } else if (!strncmp(tag, "<parameter=", 11)) {
        size_t n = strlen(tag) - 12;
        snprintf(s->tool_param, sizeof(s->tool_param), "%.*s", (int)n, tag + 11);
        s->tool_value = s->tool_first = true;
        s->tool_newline = false;
        loina_renderer_param_start(s_renderer(s), s->tool_param);
    } else if (!strcmp(tag, "</parameter>")) {
        s->tool_value = s->tool_newline = false;
        loina_renderer_param_end(s_renderer(s));
    } else if (!strcmp(tag, "</function>")) tool_stream_end(s);
    s->last_nl = loina_renderer_last_nl(s_renderer(s));
}

static void tool_stream_byte(stream_t *s, char c) {
    if (s->in_think) return;
    if (!s->tool_tag_len && c != '<') { tool_stream_value(s, c); return; }
    if (s->tool_tag_len + 1 >= sizeof(s->tool_tag)) {
        for (size_t i = 0; i < s->tool_tag_len; i++) tool_stream_value(s, s->tool_tag[i]);
        s->tool_tag_len = 0;
    }
    s->tool_tag[s->tool_tag_len++] = c;
    s->tool_tag[s->tool_tag_len] = 0;
    const char *tags[] = {"<function=", "</function>", "<parameter=", "</parameter>"};
    bool possible = false;
    for (size_t i = 0; i < sizeof(tags)/sizeof(*tags); i++) {
        if (s->tool_value && i != 3) continue;
        size_t len = strlen(tags[i]);
        if (!strncmp(s->tool_tag, tags[i], s->tool_tag_len < len ? s->tool_tag_len : len))
            possible = true;
    }
    if (!possible) {
        for (size_t i = 0; i < s->tool_tag_len; i++) tool_stream_value(s, s->tool_tag[i]);
        s->tool_tag_len = 0;
    } else if (c == '>') {
        tool_stream_tag(s);
        s->tool_tag_len = 0;
    }
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
            loina_terminal_event(LOINA_TOOL, "preparing tool call", 0);
            i += strlen(tool_open);
            continue;
        }
        if (!s->in_tool && starts(cur, rem, "<tool_call>")) {
            s->in_tool = s->native_tool = true;
            s->tool_tag_len = 0;
            loina_terminal_event(LOINA_TOOL, "preparing tool call", 0);
            i += strlen("<tool_call>");
            continue;
        }
        if (s->in_tool && (!s->native_tool || !s->tool_value) && starts(cur, rem, s->native_tool ? "</tool_call>" : tool_close)) {
            if (s->native_tool && *s->tool_name) tool_stream_end(s);
            s->in_tool = false;
            i += strlen(s->native_tool ? "</tool_call>" : tool_close);
            s->native_tool = false;
            continue;
        }
        if (s->in_tool && !finish && cur[0] == '<' &&
            partial(cur, rem, s->native_tool ? "</tool_call>" : tool_close)) {
            memcpy(s->pending, cur, rem); s->pending_len = rem;
            break;
        }
        if (s->in_tool) {
            if (s->native_tool) tool_stream_byte(s, cur[0]);
            i++;
            continue;
        }
        if (starts(cur, rem, "<think>")) {
            s->in_think = true;
            loina_renderer_feed(s_renderer(s), "<think>", 7);
            i += 7;
            continue;
        }
        if (starts(cur, rem, "</think>")) {
            s->in_think = false;
            loina_renderer_feed(s_renderer(s), "</think>", 8);
            s->last_nl = loina_renderer_last_nl(s->renderer);
            i += 8;
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (partial(cur, rem, "<think>") || partial(cur, rem, "</think>") ||
                partial(cur, rem, tool_open) || partial(cur, rem, tool_close) ||
                partial(cur, rem, "<tool_call>") || partial(cur, rem, "</tool_call>"))) {
            if (rem < sizeof(s->pending)) {
                memcpy(s->pending, cur, rem);
                s->pending_len = rem;
            }
            break;
        }
        loina_renderer_feed(s_renderer(s), cur, 1);
        s->last_nl = loina_renderer_last_nl(s->renderer);
        i++;
    }
    free(b);
    if (finish && s->renderer) {
        loina_renderer_finish(s->renderer);
        s->last_nl = loina_renderer_last_nl(s->renderer);
    }
}

static void stream_event(void *user, loina_event ev, const char *text, size_t n, int progress) {
    stream_t *s = user;
    if (ev == LOINA_TOKEN && s->in_tool)
        loina_terminal_event(LOINA_TOOL, "preparing tool call", 0);
    else loina_terminal_event(ev, text, progress);
    if (g_interrupted && s->job)
        loina_job_cancel(s->job);
    if (ev == LOINA_TOKEN) {
        s_write(s, text, n, false);
        fflush(stdout);
    } else if (ev == LOINA_LOADING && !loina_terminal_active()) {
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

/* Local file contracts adapted from ds4_agent.c (MIT; licenses/ds4.txt). */

typedef struct {
    char more_path[PATH_MAX];
    long more_next;
    off_t more_offset;
    bool more_valid, more_raw, more_mid_line;
    unsigned context;
} more_state_t;

static char *tool_err(const char *fmt, const char *a) {
    char b[1024];
    snprintf(b, sizeof(b), "Tool error: %s%s", fmt, a ? a : "");
    return xstrdup(b);
}

static bool absolute_path(const char *path, char out[PATH_MAX]) {
    if (!path || !*path) return false;
    if (*path == '/') return snprintf(out, PATH_MAX, "%s", path) < PATH_MAX;
    char cwd[PATH_MAX];
    return getcwd(cwd, sizeof(cwd)) && snprintf(out, PATH_MAX, "%s/%s", cwd, path) < PATH_MAX;
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    size_t limit;
    bool truncated;
} agent_buf;

#define AGENT_TOOL_MAX_BYTES (128 * 1024)

static void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = b->limit ? b->limit : SIZE_MAX - 1;
    if (n > max - b->len) {
        n = max > b->len ? max - b->len : 0;
        while (n && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

static char *agent_buf_take(agent_buf *b) {
    if (b->truncated) {
        b->truncated = false;
        b->limit = 0;
        agent_buf_puts(b, "\n[Output truncated at the tool byte limit. Narrow the request.]\n");
    }
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static int read_default_lines(const more_state_t *more) {
    return more->context && more->context <= 8192 ? 120 : more->context && more->context <= 16384 ? 240 : 500;
}

static FILE *open_regular_file(const char *path) {
    int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    struct stat st;
    if (fd < 0) return NULL;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode)) { close(fd); errno=EINVAL; return NULL; }
    FILE *f = fdopen(fd, "rb");
    if (!f) close(fd);
    return f;
}

static char *tool_read_from(more_state_t *w, const char *path, long start_line,
                              long max_lines, bool whole_file, bool bare,
                              bool set_more, off_t offset, bool mid_line) {
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    agent_buf out = {0}, body = {0};
    FILE *fp = open_regular_file(path);
    if (!fp) goto failed;
    struct stat st;
    if (fstat(fileno(fp), &st) != 0) goto failed;
    if (!S_ISREG(st.st_mode)) { errno = EINVAL; goto failed; }
    if (start_line < 1) start_line = 1;
    if (max_lines <= 0) max_lines = read_default_lines(w);
    int c; long line = 1, lines = 0;
    if (offset > 0) {
        if (fseeko(fp, offset, SEEK_SET) != 0) goto failed;
        line = start_line;
    }
    while (line < start_line && (c = fgetc(fp)) != EOF) {
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
        }
        if (c == '\n' || c == '\r') line++;
    }
    bool beginning = !mid_line, prefix = true;
    long last_line = 0;
    while ((whole_file || lines < max_lines) && (c = fgetc(fp)) != EOF) {
        if (body.len >= (128*1024) - 4096 &&
            ((c & 0xc0) != 0x80 || body.len >= (128*1024) - 4096 + 3)) {
            ungetc(c, fp);
            break;
        }
        if (!c) {
            agent_buf_puts(&out, "Tool error: read encountered binary data\n");
            goto done;
        }
        last_line = line;
        if (prefix && !bare) {
            char label[80];
            snprintf(label, sizeof(label), "%ld%s ", line, beginning ? "" : " (continued)");
            agent_buf_puts(&body, label);
        }
        prefix = false;
        beginning = false;
        char ch = (char)c;
        if (c == '\r') {
            int next = fgetc(fp);
            if (bare) agent_buf_append(&body, &ch, 1);
            if (next == '\n') {
                ch = '\n';
            } else {
                if (next != EOF) ungetc(next, fp);
                if (bare) ch = 0;
                else ch = '\n';
            }
        }
        if (ch) agent_buf_append(&body, &ch, 1);
        if (c == '\n' || c == '\r') {
            if (line == LONG_MAX) { errno = EOVERFLOW; goto failed; }
            line++;
            lines++;
            beginning = prefix = true;
        }
    }
    if (ferror(fp)) goto failed;
    off_t next_offset = ftello(fp);
    if (next_offset < 0) goto failed;
    c = fgetc(fp);
    bool more = c != EOF;
    if (ferror(fp)) goto failed;
    if (whole_file && more) {
        agent_buf_puts(&out, "Tool error: whole read exceeds the 128 KiB output limit; use read and more for chunks\n");
        goto done;
    }
    if (!bare) {
        char header[PATH_MAX + 128];
        snprintf(header, sizeof(header), "%s: lines %ld-%ld%s\n", path,
                 last_line ? start_line : 0, last_line, more ? " (partial read)" : " (end of file)");
        agent_buf_puts(&out, header);
    }
    agent_buf_append(&out, body.ptr, body.len);
    if (more) {
        char note[256];
        snprintf(note, sizeof(note), "\n[Read truncated. continue_offset=%ld; continue_byte_offset=%lld%s. Call more to continue.]\n",
                 line, (long long)next_offset, beginning ? "" : " (within line)");
        agent_buf_puts(&out, note);
    }
    if (set_more) {
        if (more && path != w->more_path) snprintf(w->more_path, sizeof(w->more_path), "%s", path);
        w->more_next = line; w->more_valid = more; w->more_raw = bare;
        w->more_offset = next_offset;
        w->more_mid_line = !beginning;
    }
    free(body.ptr);
    fclose(fp);
    return agent_buf_take(&out);
failed:
    agent_buf_puts(&out, "Tool error: read failed: ");
    agent_buf_puts(&out, strerror(errno));
    agent_buf_puts(&out, "\n");
done:
    if (fp) fclose(fp);
    free(body.ptr);
    if (set_more) w->more_valid = false;
    return agent_buf_take(&out);
}

static char *tool_read(const char *path, long start, long maxl, bool whole, bool raw, more_state_t *more) {
    char ap[PATH_MAX];
    if (!absolute_path(path, ap)) return tool_err("read requires path", NULL);
    return tool_read_from(more, ap, start, maxl, whole, raw, true, 0, false);
}

static char *tool_more(long count, more_state_t *more) {
    if (!more->more_valid) return tool_err("no previous output to continue", NULL);
    return tool_read_from(more, more->more_path, more->more_next, count, false,
                          more->more_raw, true, more->more_offset, more->more_mid_line);
}

static bool agent_same_file_version(const struct stat *a, const struct stat *b) {
#ifdef __APPLE__
    struct timespec am = a->st_mtimespec, bm = b->st_mtimespec;
    struct timespec ac = a->st_ctimespec, bc = b->st_ctimespec;
#else
    struct timespec am = a->st_mtim, bm = b->st_mtim;
    struct timespec ac = a->st_ctim, bc = b->st_ctim;
#endif
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && a->st_nlink == b->st_nlink &&
           am.tv_sec == bm.tv_sec && am.tv_nsec == bm.tv_nsec &&
           ac.tv_sec == bc.tv_sec && ac.tv_nsec == bc.tv_nsec;
}

#ifdef __linux__
/* Linux stores ACLs as xattrs too. A metadata copy failure must abort the
 * replacement, not silently remove access rules from the edited file. */
static int agent_copy_file_xattrs(int src, int dst) {
    if (fremovexattr(dst, "system.posix_acl_access") != 0 &&
        errno != ENODATA && errno != ENOTSUP) return -1;
    ssize_t count = flistxattr(src, NULL, 0);
    if (count < 0) return errno == ENOTSUP ? 0 : -1;
    if (!count) return 0;
    char *names = xmalloc((size_t)count);
    count = flistxattr(src, names, (size_t)count);
    int rc = -1, saved_errno;
    if (count < 0) goto done;
    for (ssize_t pos = 0; pos < count;) {
        const char *name = names + pos;
        ssize_t len = fgetxattr(src, name, NULL, 0);
        if (len < 0) goto done;
        void *value = xmalloc(len ? (size_t)len : 1);
        ssize_t readlen = fgetxattr(src, name, value, (size_t)len);
        int copied = readlen < 0 ? -1 : fsetxattr(dst, name, value, (size_t)readlen, 0);
        int saved_errno = errno;
        free(value);
        errno = saved_errno;
        if (copied != 0) goto done;
        pos += (ssize_t)strlen(name) + 1;
    }
    rc = 0;
done:
    saved_errno = errno;
    free(names);
    errno = saved_errno;
    return rc;
}
#endif

/* Commit only complete files. Follow existing symlinks, but refuse hard links:
 * an atomic rename cannot preserve their shared-inode semantics. The version
 * checks catch concurrent edits, though unrelated writers do not take a lock. */
static int agent_replace_file(const char *path, const char *data, size_t len,
                              const char *expected, size_t expected_len,
                              char *err, size_t errlen) {
    char target[PATH_MAX], temp[PATH_MAX] = "";
    struct stat before, after;
    int src = -1, fd = -1, rc = -1;
    bool exists = lstat(path, &before) == 0;
    if (!exists && errno != ENOENT) goto failed;
    if (exists) {
        if (!realpath(path, target)) goto failed;
        src = open(target, O_RDWR | O_NONBLOCK | O_NOFOLLOW);
        if (src < 0 || fstat(src, &before) != 0) goto failed;
        if (!S_ISREG(before.st_mode) || before.st_nlink != 1) {
            snprintf(err, errlen, "refusing to replace non-regular or hard-linked file: %s", path);
            goto done;
        }
        if (expected) {
            if (before.st_size < 0 || (uintmax_t)before.st_size != expected_len)
                goto changed;
            char buf[8192];
            size_t pos = 0;
            while (pos < expected_len) {
                size_t count = expected_len - pos;
                if (count > sizeof(buf)) count = sizeof(buf);
                ssize_t n = read(src, buf, count);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) goto failed;
                if (!n || memcmp(buf, expected + pos, (size_t)n)) goto changed;
                pos += (size_t)n;
            }
        }
    } else {
        if (expected) goto changed;
        if (snprintf(target, sizeof(target), "%s", path) >= (int)sizeof(target)) {
            errno = ENAMETOOLONG;
            goto failed;
        }
    }
    if (snprintf(temp, sizeof(temp), "%s.loina-XXXXXX", target) >= (int)sizeof(temp)) {
        temp[0] = 0;
        errno = ENAMETOOLONG;
        goto failed;
    }
    fd = mkstemp(temp);
    if (fd < 0) { temp[0] = 0; goto failed; }
    if (!exists) {
        /* Let open apply the process umask without changing it in this thread. */
        if (unlink(temp) != 0) goto failed;
        close(fd);
        fd = open(temp, O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd < 0) { temp[0] = 0; goto failed; }
    }
    for (size_t pos = 0; pos < len;) {
        ssize_t n = write(fd, data + pos, len - pos);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; goto failed; }
        pos += (size_t)n;
    }
    if (exists) {
        if (fchown(fd, before.st_uid, before.st_gid) != 0) goto failed;
        if (fchmod(fd, before.st_mode & 07777) != 0) goto failed;
#ifdef __APPLE__
        if (fcopyfile(src, fd, NULL, COPYFILE_ACL | COPYFILE_XATTR) != 0) goto failed;
#elif defined(__linux__)
        if (agent_copy_file_xattrs(src, fd) != 0) goto failed;
#endif
    }
    if (fsync(fd) != 0) goto failed;
    if (close(fd) != 0) { fd = -1; goto failed; }
    fd = -1;
    if (exists) {
        char resolved[PATH_MAX];
        if (!realpath(path, resolved) || strcmp(resolved, target) ||
            stat(target, &after) != 0 || !agent_same_file_version(&before, &after))
            goto changed;
        if (rename(temp, target) != 0) goto failed;
    } else {
        /* Unlike rename, link will not overwrite a concurrently created file. */
        if (link(temp, target) != 0) goto failed;
        unlink(temp);
    }
    temp[0] = 0;
    rc = 0;
    goto done;
changed:
    snprintf(err, errlen, "file changed while editing; read it again: %s", path);
    goto done;
failed:
    snprintf(err, errlen, "replace %s: %s", path, strerror(errno));
done:
    if (fd >= 0) close(fd);
    if (src >= 0) close(src);
    if (temp[0]) unlink(temp);
    return rc;
}

static char *tool_write(const char *path, const char *content) {
    char ap[PATH_MAX], err[256];
    if (!absolute_path(path, ap)) return tool_err("write requires path", NULL);
    if (!content) return tool_err("write requires content", NULL);
    if (agent_replace_file(ap, content, strlen(content), NULL, 0, err, sizeof(err)))
        return tool_err(err, NULL);
    char result[PATH_MAX+80];
    snprintf(result, sizeof(result), "Wrote %zu bytes to %s\n", strlen(content), path);
    return xstrdup(result);
}

static char *tool_edit(const char *path, const char *old, const char *new, bool allow_upto) {
    char ap[PATH_MAX];
    if (!absolute_path(path, ap) || !old || !*old || !new)
        return tool_err("edit requires path, nonempty old, and new", NULL);
    FILE *f=open_regular_file(ap);
    if (!f) return tool_err("cannot open file: ", path);
    struct stat st;
    if (fstat(fileno(f), &st) || st.st_size<0 || st.st_size>8*1024*1024) {
        fclose(f); return tool_err("cannot edit file: ", path);
    }
    size_t n=(size_t)st.st_size;
    char *data=xmalloc(n+1);
    bool ok=fread(data, 1, n, f)==n && !ferror(f);
    fclose(f); data[n]=0;
    if (!ok || memchr(data, 0, n)) { free(data); return tool_err("cannot read text file: ", path); }
    char *head=NULL;
    const char *upto=allow_upto ? strstr(old, "[upto]") : NULL;
    if (upto) head=xstrndup(old, (size_t)(upto-old));
    const char *needle=head ? head : old;
    const char *first=*needle ? strstr(data, needle) : NULL;
    const char *last=first ? first+strlen(needle) : NULL;
    if (!first || strstr(first+1, needle)) ok=false;
    if (upto && ok) {
        const char *tail=upto+strlen("[upto]");
        const char *match=*tail ? strstr(last, tail) : NULL;
        if (!match || strstr(match+1, tail)) ok=false;
        else last=match+strlen(tail);
    }
    free(head);
    if (!ok) { free(data); return tool_err("edit fails: old text must match exactly once", NULL); }
    size_t before=(size_t)(first-data), replacement=strlen(new), after=n-(size_t)(last-data);
    char *result=xmalloc(before+replacement+after+1);
    memcpy(result, data, before);
    memcpy(result+before, new, replacement);
    memcpy(result+before+replacement, last, after);
    result[before+replacement+after]=0;
    char err[256];
    ok=!agent_replace_file(ap, result, before+replacement+after, data, n, err, sizeof(err));
    free(data); free(result);
    return ok ? xstrdup("Edited successfully.\n") : tool_err(err, NULL);
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

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    size_t skipped;
    char first_skip[PATH_MAX + 128];
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_puts(&ctx->out, data);
    agent_buf_puts(&ctx->out, "\n");
}

static void agent_search_skip(agent_search_ctx *ctx, const char *path, const char *why) {
    ctx->skipped++;
    if (!ctx->first_skip[0])
        snprintf(ctx->first_skip, sizeof(ctx->first_skip), "%s: %s", path, why);
}

/* A file can be arbitrarily large; only the current line and a small context
 * ring are resident. Oversized or binary lines are reported, never ignored. */
static int agent_search_read_line(FILE *fp, char **text) {
    agent_buf line = {0};
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!c || line.len == AGENT_TOOL_MAX_BYTES) {
            free(line.ptr);
            return -2;
        }
        if (c == '\n' || c == '\r') {
            if (c == '\r') {
                int next = fgetc(fp);
                if (next != '\n' && next != EOF) ungetc(next, fp);
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&line, &ch, 1);
    }
    if (ferror(fp)) { free(line.ptr); return -1; }
    if (c == EOF && !line.len) { free(line.ptr); return 0; }
    *text = agent_buf_take(&line);
    return 1;
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    FILE *fp = open_regular_file(path);
    if (!fp) { agent_search_skip(ctx, path, strerror(errno)); return; }
    char *ring[6] = {0};
    bool printed_file = false;
    int line = 0, last_emitted = 0, after = 0;
    while ((ctx->results < ctx->max_results || after) && !ctx->out.truncated) {
        char *text = NULL;
        int rc = agent_search_read_line(fp, &text);
        if (!rc) break;
        if (rc < 0) {
            agent_search_skip(ctx, path, rc == -2 ? "binary data or line exceeds 128 KiB" : strerror(errno));
            break;
        }
        if (line == INT_MAX) {
            free(text);
            agent_search_skip(ctx, path, "line number limit reached");
            break;
        }
        line++;
        free(ring[line % 6]);
        ring[line % 6] = text;
        bool match = ctx->results < ctx->max_results &&
                     agent_search_line_matches(ctx, text, strlen(text));
        if (match && !printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        if (match) {
            int from = line - ctx->context;
            if (from < 1) from=1;
            if (from <= last_emitted) from = last_emitted + 1;
            for (int j = from; j <= line; j++)
                agent_search_emit_line(ctx, ring[j % 6], j);
            last_emitted = line;
            after = ctx->context;
            ctx->results++;
        } else if (after) {
            agent_search_emit_line(ctx, text, line);
            last_emitted = line;
            after--;
        }
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    for (int i = 0; i < 6; i++) free(ring[i]);
    fclose(fp);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (depth > 24) { agent_search_skip(ctx, path, "directory depth limit reached"); return; }
    struct stat st;
    if ((depth == 0 ? stat(path, &st) : lstat(path, &st)) != 0) {
        agent_search_skip(ctx, path, strerror(errno));
        return;
    }
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        agent_search_skip(ctx, path, "not a regular file or directory (nested symlinks are not followed)");
        return;
    }
    DIR *dir = opendir(path);
    if (!dir) { agent_search_skip(ctx, path, strerror(errno)); return; }
    struct dirent *de;
    while (ctx->results < ctx->max_results && !ctx->out.truncated) {
        errno = 0;
        de = readdir(dir);
        if (!de) {
            if (errno) agent_search_skip(ctx, path, strerror(errno));
            break;
        }
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        if (snprintf(child, sizeof(child), "%s/%s", path, de->d_name) >= (int)sizeof(child)) {
            agent_search_skip(ctx, path, "path exceeds PATH_MAX");
            continue;
        }
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

static char *tool_search(search_args_t *a) {
    const char *query = a->query;
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = a->path;
    if (!path || !path[0]) path = ".";
    const char *mode = a->mode;
    if (mode && strcmp(mode, "literal") && strcmp(mode, "regex"))
        return xstrdup("Tool error: search mode must be literal or regex (POSIX extended)\n");
    struct stat root;
    if (stat(path, &root) != 0 || (!S_ISDIR(root.st_mode) && !S_ISREG(root.st_mode))) {
        agent_buf error = {0};
        agent_buf_puts(&error, "Tool error: search path is missing, unreadable, or not a regular file/directory: ");
        agent_buf_puts(&error, path);
        agent_buf_puts(&error, "\n");
        return agent_buf_take(&error);
    }
    agent_search_ctx ctx = {
        .query = query,
        .glob = a->glob,
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = a->case_sens,
        .context = (int)(a->context<0 ? 0 : a->context>5 ? 5 : a->context),
        .max_results = (int)(a->max_results<1 ? 1 : a->max_results>500 ? 500 : a->max_results),
        .out = {.limit = AGENT_TOOL_MAX_BYTES},
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, path, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches in searched text\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    ctx.out.limit = 0;
    if (ctx.skipped || ctx.results >= ctx.max_results) {
        bool truncated = ctx.out.truncated;
        ctx.out.truncated = false;
        char note[PATH_MAX + 256];
        snprintf(note, sizeof(note), "\nSearch incomplete: %zu skipped paths%s. %s\n",
                 ctx.skipped, ctx.results >= ctx.max_results ? "; match limit reached" : "", ctx.first_skip);
        agent_buf_puts(&ctx.out, note);
        ctx.out.truncated = truncated;
    }
    return agent_buf_take(&ctx.out);
}

static char *tool_list(const char *path) {
    if (!path || !path[0]) path = ".";
    DIR *dir = opendir(path);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {.limit = AGENT_TOOL_MAX_BYTES};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        if (snprintf(full, sizeof(full), "%s/%s", path, de->d_name) >= (int)sizeof(full)) continue;
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

/* ---------- native and legacy tool parsing ---------- */

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

static int parse_legacy_calls(const char *reply, tcalls_t *out) {
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
            ds4_tool_text_unescape(val, "</loina_param>");
            /* trim one surrounding newline (ds4 convention) */
            if (*val == '\n') memmove(val, val + 1, strlen(val));
            size_t vl = strlen(val);
            if (vl && val[vl - 1] == '\n') val[--vl] = 0;
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

static int parse_calls(const char *reply, tcalls_t *out) {
    const char *thinking = strstr(reply, "</think>");
    if (thinking) reply = thinking + 8;
    else if (strstr(reply, "<think>")) { memset(out, 0, sizeof(*out)); return 0; }
    if (!strstr(reply, "<tool_call>")) return parse_legacy_calls(reply, out);
    memset(out, 0, sizeof(*out));
    const char *p = reply;
    while ((p = strstr(p, "<tool_call>"))) {
        p += strlen("<tool_call>");
        while (isspace((unsigned char)*p)) p++;
        if (strncmp(p, "<function=", 10) || out->len == 4) goto malformed;
        const char *name = p+10, *gt = strchr(name, '>');
        if (!gt || gt == name) goto malformed;
        out->v = xrealloc(out->v, (size_t)(out->len+1)*sizeof(*out->v));
        tcall_t *call = &out->v[out->len++];
        memset(call, 0, sizeof(*call));
        call->name = xstrndup(name, (size_t)(gt-name));
        p = gt+1;
        while (1) {
            while (isspace((unsigned char)*p)) p++;
            if (!strncmp(p, "</function>", 11)) { p += 11; break; }
            if (strncmp(p, "<parameter=", 11) || call->argc == 20) goto malformed;
            const char *arg=p+11;
            gt=strchr(arg, '>');
            const char *close=gt ? strstr(gt+1, "</parameter>") : NULL;
            if (!gt || gt==arg || !close) goto malformed;
            call->args=xrealloc(call->args, (size_t)(call->argc+1)*sizeof(*call->args));
            targ_t *param=&call->args[call->argc++];
            param->name=xstrndup(arg, (size_t)(gt-arg));
            const char *value=gt+1, *last=close;
            if (*value == '\n') value++;
            if (last>value && last[-1]=='\n') last--;
            param->value=xstrndup(value, (size_t)(last-value));
            ds4_tool_text_unescape(param->value, "</parameter>");
            p=close+strlen("</parameter>");
        }
        while (isspace((unsigned char)*p)) p++;
        if (strncmp(p, "</tool_call>", 12)) goto malformed;
        p += 12;
    }
    return out->len;
malformed:
    tcalls_free(out);
    snprintf(out->error, sizeof(out->error), "Incomplete native tool call");
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
         "  --chdir PATH          Run local tools in PATH (default new /tmp/loina-agent-* directory)\n"
         "  -m FILE               Import catalog model file (siblings if needed)\n"
         "  --package ID          Use an immutable package\n"
         "  --import DIRECTORY    Install selected package from local artifacts\n"
         "  --session ID          Resume a saved conversation\n"
         "  --model-dir PATH      Model files (default gguf/ beside the executable)\n"
         "  --cache-dir PATH      Sessions and KV (default ~/.loina/kvcache)\n"
         "  --ctx N               Context size, max 32768 (default 32768)\n"
         "  -n N                  Max tokens per turn (default 8192 thinking, 4096 plain)\n"
         "  --threads N           CPU threads\n"
         "  --backend NAME        CPU, Vulkan, OpenCL, Metal or CUDA\n"
         "  --temp F --top-p F --min-p F  Sampling\n"
         "  --seed N              Sampling seed\n"
         "  --thinking on|off     Reasoning when the package allows it (default on)\n"
         "  --vision FILE         Attach an image to the first task\n"
         "REPL: /help /think /nothink /save /list /switch ID /del ID /new /read FILE /quit\n"
         "Tools: google_search visit_page bash bash_status bash_stop read more write edit search list view_image.");
}

static bool write_agent_template(const char *dir, bool edit_upto) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/system.jinja", dir);
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    const char *parts[] = {agent_system_jinja, edit_upto ? agent_edit_upto : agent_edit_exact, agent_rules_tail};
    bool ok = true;
    for (size_t i=0; i<sizeof(parts)/sizeof(*parts); i++) {
        size_t n = strlen(parts[i]);
        if (fwrite(parts[i], 1, n, f) != n) ok = false;
    }
    if (fclose(f)) ok = false;
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
    int streamed_tools;
} agent_t;

static char *run_turn(agent_t *a, const char *prompt, const char *vision_path, bool *cancelled) {
    loina_request r = loina_request_default(vision_path ? LOINA_VISION : LOINA_TEXT);
    r.session = a->session;
    r.prompt = prompt;
    r.input_path = vision_path;
    r.tools = false;
    r.tool_schema = agent_tool_schema;
    r.tool_response = !strncmp(prompt, "<tool_response>", 15);
    if (a->cfg->package_id) r.package = loina_catalog_find(a->cfg->package_id);
    r.thinking = a->cfg->thinking;
    r.max_tokens = a->cfg->n_set ? a->cfg->max_tokens : (a->cfg->thinking ? 8192 : 4096);
    uint32_t capacity = a->cfg->engine_cfg.context ? a->cfg->engine_cfg.context : 32768;
    if (!a->cfg->n_set && r.max_tokens >= capacity) r.max_tokens = capacity / 2;
    r.temperature = a->cfg->temp;
    r.top_p = a->cfg->top_p;
    r.min_p = a->cfg->min_p;
    if (a->cfg->seed_set)
        r.seed = a->cfg->seed;
    a->streamed_tools = 0;
    stream_t s = {.use_color = (loina_terminal_active() || isatty(fileno(stdout))), .last_nl = true, .context=capacity};
    loina_error err = {0};
    loina_job *job = loina_job_create(a->engine, &r, stream_event, &s, &err);
    if (!job) {
        fprintf(stderr, "%s\n", err.message);
        return NULL;
    }
    s.job = job;
    g_job = job;
    loina_terminal_job(job);
    loina_terminal_event(LOINA_PROGRESS, "", 0);
    loina_status st = loina_job_run(job, &err);
    loina_terminal_job(NULL);
    g_job = NULL;
    s_write(&s, NULL, 0, true);
    a->streamed_tools = s.streamed_tools;
    if (!s.last_nl) {
        putchar('\n');
        fflush(stdout);
    }
    bool interactive_ui = loina_terminal_active();
    if (s.have_metrics && !st && !interactive_ui) {
        bool ttyerr = isatty(fileno(stderr)) != 0;
        fprintf(stderr, "%sloina-agent: prefill: %.2f t/s, generation: %.2f t/s%s\n",
            ttyerr ? "\x1b[38;2;177;161;245m" : "", s.prefill, s.decode, ttyerr ? "\x1b[0m" : "");
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
    loina_renderer_free(s.renderer);
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
    loina_terminal_job(job);
    loina_status st = loina_job_run(job, &err);
    loina_terminal_job(NULL);
    g_job = NULL;
    char *res = NULL;
    if (!st)
        res = xstrdup(loina_job_result(job));
    else
        fprintf(stderr, "%s\n", err.message);
    loina_renderer_free(s.renderer);
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
        return loina_bash_run(arg_val(c, "command"), arg_long(arg_val(c, "timeout_sec"), 3600),
                              arg_long(arg_val(c, "refresh_sec"), 60), &g_interrupted);
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
    if (!strcmp(n, "bash_status") || !strcmp(n, "bash_stop"))
        return loina_bash_status((int)arg_long(arg_val(c, "job"), 0),
                                 (int)arg_long(arg_val(c, "pid"), 0),
                                 arg_long(arg_val(c, "refresh_sec"), !strcmp(n, "bash_stop") ? 1 : 0),
                                 !strcmp(n, "bash_stop"), &g_interrupted);
    return tool_err("unknown tool: ", n);
}

static void tool_body(const char *text, const char *prefix, const char *color, bool tty) {
    if (!text) return;
    while (*text) {
        const char *end = strchr(text, '\n');
        size_t n = end ? (size_t)(end - text) : strlen(text);
        fprintf(stderr, "%s%s%.*s%s\n", tty ? color : "", prefix, (int)n, text, tty ? "\x1b[0m" : "");
        text += n + (end != NULL);
    }
}

static void tool_announce(tcall_t *call) {
    bool tty = loina_terminal_active() || isatty(STDERR_FILENO);
    const char *name = call->name, *path = arg_val(call, "path");
    fprintf(stderr, "🛠️ %s", tty ? "\x1b[1;37m" : "");
    if (!strcmp(name, "bash")) {
        const char *command = arg_val(call, "command");
        fprintf(stderr, "%s$ %s", tty ? LOINA_ACCENT : "", command ? command : "");
    } else if (!strcmp(name, "read")) {
        fprintf(stderr, "Reading %s%s%s ", tty ? "\x1b[32m" : "", path ? path : "<unknown>", tty ? "\x1b[33m" : "");
        long start = arg_long(arg_val(call, "start_line"), 1);
        if (arg_bool(arg_val(call, "whole"), false)) {
            if (start == 1) fputs("(whole file)", stderr);
            else fprintf(stderr, "%ld:EOF", start);
        } else fprintf(stderr, "%ld:%ld", start, arg_long(arg_val(call, "max_lines"), 80));
        fputs("...", stderr);
    } else {
        const char *prefix = !strcmp(name, "web_search") || !strcmp(name, "google_search") ? "google" :
                             !strcmp(name, "web_visit") || !strcmp(name, "visit_page") ? "visit" : name;
        fprintf(stderr, "%s %s%s", prefix, tty ? "\x1b[32m" : "", path ? path : "");
        const char *query = arg_val(call, "query"), *url = arg_val(call, "url");
        if (query) fprintf(stderr, "%s%s", path ? " " : "", query);
        if (url) fputs(url, stderr);
    }
    fprintf(stderr, "%s\n", tty ? "\x1b[0m" : "");
    if (!strcmp(name, "write")) tool_body(arg_val(call, "content"), "", "\x1b[34m", tty);
    if (!strcmp(name, "edit")) {
        tool_body(arg_val(call, "old"), "- ", "\x1b[31m", tty);
        tool_body(arg_val(call, "new"), "+ ", "\x1b[32m", tty);
    }
}

static bool bash_result_ok(const char *result) {
    const char *end = strchr(result, '\n');
    const char *state = strstr(result, "status=done");
    const char *timeout = strstr(result, "timed_out=0");
    return end && state && state < end && timeout && timeout < end &&
           !strncmp(end + 1, "exit_status=0\n", strlen("exit_status=0\n"));
}

/* Keep host tools outside the app web loop and require checks after edits. */
static int agent_task(agent_t *a, const char *user_text, const char *vision_path) {
    char *prompt = xstrdup(user_text);
    const char *vision = vision_path;
    bool modified = false, checked = false;
    int check_reminders = 0;
    for (int round = 0; round < 32; round++) {
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
                prompt = xstrdup("Tool error: Incomplete call. Use <tool_call><function=NAME><parameter=NAME>VALUE</parameter></function></tool_call> with the supplied tools.");
                /* feed as tool_response so the transcript stays a tool turn */
                char *wrapped = xmalloc(strlen(prompt) + 64);
                sprintf(wrapped, "<tool_response>\n%s\n</tool_response>", prompt);
                free(prompt);
                prompt = wrapped;
                continue;
            }
            if (nc == 0) {
                if (!empty && modified && !checked && round < 31 && check_reminders++ < 2) {
                    const char *instruction = "\nThe files have changed, but no successful check was run afterward. "
                        "Use bash to run focused tests of the requested behavior and edge cases. "
                        "Fix failures before finishing. Do not merely describe how the user could test it.";
                    prompt = xmalloc(strlen(user_text) + strlen(instruction) + 32);
                    sprintf(prompt, "Original task: %s%s", user_text, instruction);
                    fprintf(stderr, "Verifying generated files before finishing.\n");
                    free(visible); free(reply);
                    continue;
                }
                if (modified && !checked) {
                    fprintf(stderr, "Files were written, but the model did not run successful checks.\n");
                    free(visible); free(reply);
                    return 1;
                }
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
            for (int i = a->streamed_tools; i < calls.len; i++) tool_announce(&calls.v[i]);
            free(visible);
        }
        if (round == 31) {
            fprintf(stderr, "Tool round limit reached before the task completed.\n");
            tcalls_free(&calls); free(reply);
            return 1;
        }
        /* Execute and wrap observations for the MiniCPM <tool_response> path. */
        char *obs = NULL;
        size_t ol = 0, oc = 0;
        for (int i = 0; i < calls.len; i++) {
            loina_terminal_event(LOINA_TOOL, "running tool", 0);
            char *r = dispatch_tool(a, &calls.v[i]);
            const char *name = calls.v[i].name;
            if ((!strcmp(name, "write") || !strcmp(name, "edit")) && strncmp(r, "Tool error:", 11)) {
                modified = true; checked = false;
            }
            if (!strcmp(name, "bash") || !strcmp(name, "bash_status"))
                checked = bash_result_ok(r);
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

typedef struct {
    agent_t *agent;
    const char *text, *vision;
} task_t;

static int task_run(void *pointer) {
    task_t *t = pointer;
    return agent_task(t->agent, t->text, t->vision);
}

static int run_task(agent_t *agent, const char *text, const char *vision) {
    g_interrupted=0;
    task_t task = {.agent=agent, .text=text, .vision=vision};
    return loina_terminal_run(task_run, &task);
}

static bool agent_prompt_yes_no(const char *prompt) {
    char line[80];
    for (;;) {
        fputs(prompt,stdout);
        fflush(stdout);
        if (!fgets(line,sizeof(line),stdin)) { clearerr(stdin); return false; }
        char *p=line;
        while (*p==' ' || *p=='\t') p++;
        if (*p=='y' || *p=='Y') return true;
        if (*p=='n' || *p=='N') return false;
    }
}

static bool agent_save_session(loina_session *session, size_t *saved_count) {
    loina_error error={0};
    if (loina_session_checkpoint(session,&error)) {
        printf("save failed: %s\n",error.message);
        return false;
    }
    *saved_count=loina_session_count(session);
    printf("saved session %s (chat + KV)\n",loina_session_id(session));
    return true;
}

static bool agent_maybe_save_before_exiting(loina_session *session, size_t *saved_count, bool tty) {
    if (!tty || loina_session_count(session)==*saved_count) return true;
    if (!agent_prompt_yes_no("Save current session? (y/n) ")) return true;
    if (agent_save_session(session,saved_count)) return true;
    return agent_prompt_yes_no("Continue anyway? (y/n) ");
}

int main(int argc, char **argv) {
    /* Worker output must reach the UI pipe without waiting for a flush. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    agent_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.thinking = true;
    cfg.temp = 0.1f;
    cfg.top_p = 1.0f;
    char store[PATH_MAX], model_store[PATH_MAX];
    const char *cache_dir = NULL, *model_dir = NULL;
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
        else if (!strcmp(a, "--model-dir"))
            model_dir = v;
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
    if (!cache_dir) {
        if (!loina_storage_directory(store,sizeof(store))) {
            fprintf(stderr,"Cannot locate the Loina data directory; use --cache-dir PATH\n");
            return 1;
        }
        cache_dir=store;
    }
    if (!model_dir) {
        if (!loina_storage_models(model_store,sizeof(model_store),argv[0])) {
            fprintf(stderr,"Cannot locate model files; use --model-dir PATH\n");
            return 1;
        }
        model_dir=model_store;
    }
    char launch[PATH_MAX], paths[7][PATH_MAX];
    if (!getcwd(launch, sizeof(launch))) { perror("loina-agent: getcwd"); return 1; }
    const char **arguments[] = {&cache_dir, &templates_override, &cfg.model_file,
                               &cfg.import, &cfg.prompt_file, &cfg.vision_path, &model_dir};
    for (size_t i = 0; i < sizeof(arguments)/sizeof(*arguments); i++) {
        const char *value = *arguments[i];
        if (!value || value[0] == '/') continue;
        int n = snprintf(paths[i], sizeof(paths[i]), "%s/%s", launch, value);
        if (n < 0 || (size_t)n >= sizeof(paths[i])) {
            fprintf(stderr, "loina-agent: path is too long: %s\n", value);
            return 1;
        }
        *arguments[i] = paths[i];
    }
    char workspace[] = "/tmp/loina-agent-XXXXXX";
    if (!cfg.chdir_path && (show_models || show_device)) cfg.chdir_path = launch;
    if (!cfg.chdir_path) {
        if (!mkdtemp(workspace)) { perror("loina-agent: workspace"); return 1; }
        cfg.chdir_path = workspace;
    }
    if (chdir(cfg.chdir_path)) {
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

    cfg.engine_cfg.context = ctx;
    loina_config ecfg = {.directory=cache_dir, .model_directory=model_dir,
        .session_directory=cache_dir, .cache_directory=cache_dir, .web=agent_web, .context=ctx};
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
        if (!write_agent_template(auto_tmpl, cfg.edit_upto)) {
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

    if (!cfg.non_interactive && loina_preload(engine, LOINA_TEXT, NULL, &err)) {
        fprintf(stderr, "%s\n", err.message);
        loina_close(engine);
        free(owned_prompt);
        return 1;
    }

    loina_session *session = cfg.session_id ? loina_session_load(engine, cfg.session_id, &err)
                                            : agent_new_session(engine, &err);
    if (!session) {
        fprintf(stderr, "%s\n", err.message);
        loina_close(engine);
        free(owned_prompt);
        return 1;
    }
    loina_session_set_autosave(session,cfg.non_interactive);
    size_t saved_count=cfg.session_id ? loina_session_count(session) : 0;
    more_state_t more = {.context=ctx};
    agent_t agent = {.engine = engine, .session = session, .cfg = &cfg, .more = &more};
    int rc = 0;
    if (cfg.non_interactive) {
        fprintf(stderr, "Working directory: %s\n", cfg.chdir_path);
        if (!cfg.prompt) {
            fprintf(stderr, "loina-agent: --non-interactive needs -p\n");
            rc = 2;
        } else {
            rc = run_task(&agent, cfg.prompt, cfg.vision_path);
        }
    } else {
        bool tty = isatty(fileno(stdin)) != 0;
        do {
            bool full_ui = loina_terminal_open("Agent", ctx ? ctx : 32768, &cfg.thinking, &g_interrupted);
            if (!full_ui) puts("Loina Agent. /help for commands.");
            char line[16384];
            while (full_ui ? loina_terminal_read(line, sizeof(line)) != NULL :
                   (fputs(tty ? "loina-agent> " : "", stdout), fflush(stdout), fgets(line, sizeof(line), stdin) != NULL)) {
                line[strcspn(line, "\r\n")] = 0;
                char *cmd = line;
                while (*cmd && isspace((unsigned char)*cmd))
                    cmd++;
                char *e = cmd + strlen(cmd);
                while (e > cmd && isspace((unsigned char)e[-1]))
                    *--e = 0;
                if (!*cmd)
                    continue;
                if (!full_ui) hist_append(cmd);
                if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit"))
                    break;
                if (loina_terminal_command(session, cmd)) continue;
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
                    agent_save_session(session,&saved_count);
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
                        loina_session_set_autosave(session,false);
                        saved_count=!strcmp(cmd,"/new") ? 0 : loina_session_count(session);
                        more = (more_state_t){.context=ctx};
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
            loina_terminal_close();
        } while (!agent_maybe_save_before_exiting(session,&saved_count,tty));
    }
    loina_bash_cleanup();
    loina_session_free(session);
    loina_close(engine);
    free(owned_prompt);
    return rc;
}

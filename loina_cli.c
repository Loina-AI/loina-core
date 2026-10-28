#include "loina.h"
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

/* Loina CLI, ds4 style.
 *
 * Interactive chat uses the same settings as the mobile apps: the default
 * request enables web tools for text/vision, uses 2048 tokens with thinking
 * and 1024 without (see loina_jni prepare), and resolves the same immutable
 * catalog packages through llama.cpp. Vision reuses the chat weights plus the
 * mmproj projector, exactly like the app: --vision or /read with an image
 * runs the vision layer, /read with text attaches a document.
 *
 * Browser tools run through a small Linux bridge (curl + tag stripping) that
 * returns the same "# Search results" / "# Visited page" shape as the app's
 * native WebView, so prompts and sources stay consistent across hosts. */

static volatile sig_atomic_t interrupted;
static void on_signal(int signal) { (void)signal; interrupted = 1; }

typedef struct {
    loina_job *job;
    FILE *out;
    bool use_color;
    bool in_think;
    bool color_open;
    bool last_newline;
    char pending[16];
    size_t pending_len;
    double prefill, decode;
    bool have_metrics;
    int last_progress;
} cli_stream;

static bool has_prefix(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n >= m && !memcmp(p, s, m);
}

static bool partial_prefix(const char *p, size_t n, const char *s) {
    size_t m = strlen(s);
    return n < m && !memcmp(s, p, n);
}

static void printer_grey(cli_stream *s) {
    if (s->use_color && !s->color_open) {
        fputs("\x1b[90m", s->out);
        s->color_open = true;
    }
}

static void printer_reset(cli_stream *s) {
    if (s->use_color && s->color_open) {
        fputs("\x1b[0m", s->out);
        s->color_open = false;
    }
}

static void printer_char(cli_stream *s, char c) {
    if (s->in_think)
        printer_grey(s);
    fputc((unsigned char)c, s->out);
    s->last_newline = (c == '\n');
}

static void printer_process(cli_stream *s, const char *text, size_t n, bool finish) {
    /* Strip <think> markers like ds4: reasoning shows grey, the answer plain.
     * Tokens may split a marker, so hold a trailing '<' run across calls. */
    size_t total = s->pending_len + n;
    char *buf = malloc(total ? total : 1);
    if (!buf)
        return;
    if (s->pending_len)
        memcpy(buf, s->pending, s->pending_len);
    if (n)
        memcpy(buf + s->pending_len, text, n);
    s->pending_len = 0;
    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        size_t rem = total - i;
        if (has_prefix(cur, rem, "<think>")) {
            s->in_think = true;
            i += 7;
            continue;
        }
        if (has_prefix(cur, rem, "</think>")) {
            s->in_think = false;
            printer_reset(s);
            if (!s->last_newline) {
                fputc('\n', s->out);
                s->last_newline = true;
            }
            i += 8;
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (partial_prefix(cur, rem, "<think>") || partial_prefix(cur, rem, "</think>"))) {
            if (rem < sizeof(s->pending)) {
                memcpy(s->pending, cur, rem);
                s->pending_len = rem;
            }
            break;
        }
        printer_char(s, cur[0]);
        i++;
    }
    free(buf);
}

static void event_cb(void *user, loina_event type, const char *text, size_t bytes, int progress) {
    cli_stream *s = user;
    if (interrupted && s->job)
        loina_job_cancel(s->job);
    if (type == LOINA_TOKEN) {
        printer_process(s, text, bytes, false);
        fflush(s->out);
    } else if (type == LOINA_LOADING) {
        fprintf(stderr, "Loading %.*s\n", (int)bytes, text);
    } else if (type == LOINA_PROGRESS) {
        /* Prefill / copy progress. Overwrite one stderr line on ttys. */
        if (isatty(fileno(stderr))) {
            fprintf(stderr, "\rprogress %d%%\x1b[K", progress);
            fflush(stderr);
            s->last_progress = progress;
            if (progress >= 100) {
                fputc('\n', stderr);
                s->last_progress = -1;
            }
        } else if (progress == 100 || progress % 25 == 0) {
            fprintf(stderr, "progress %d%%\n", progress);
        }
    } else if (type == LOINA_TOOL) {
        if (s->last_progress >= 0 && isatty(fileno(stderr))) {
            fputc('\n', stderr);
            s->last_progress = -1;
        }
        fprintf(stderr, "Reading the web...\n");
    } else if (type == LOINA_COMPACTED) {
        fprintf(stderr, "Earlier turns remain in the saved transcript.\n");
    } else if (type == LOINA_METRICS) {
        /* {"prefill":12.3,"decode":45.6,"backend":4,...} Keep the latest; the
         * runtime emits interim values, the CLI reports the final timing. */
        const char *p = strstr(text, "\"prefill\"");
        const char *d = strstr(text, "\"decode\"");
        if (p)
            s->prefill = strtod(strchr(p, ':') ? strchr(p, ':') + 1 : p, NULL);
        if (d)
            s->decode = strtod(strchr(d, ':') ? strchr(d, ':') + 1 : d, NULL);
        s->have_metrics = true;
    }
}

/* -- Linux web bridge (curl), same output shape as the app WebView. -- */

static void shell_quote(const char *s, char *out, size_t n) {
    size_t u = 0;
    if (u < n)
        out[u++] = '\'';
    for (; *s && u + 4 < n; s++) {
        if (*s == '\'') {
            memcpy(out + u, "'\\''", 4);
            u += 4;
        } else {
            out[u++] = *s;
        }
    }
    if (u < n)
        out[u++] = '\'';
    if (u < n)
        out[u] = 0;
    else
        out[n - 1] = 0;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

static void percent_decode_inplace(char *s) {
    char *w = s;
    for (char *r = s; *r; r++) {
        if (*r == '%' && hexval(r[1]) >= 0 && hexval(r[2]) >= 0) {
            *w++ = (char)(hexval(r[1]) * 16 + hexval(r[2]));
            r += 2;
        } else if (*r == '+') {
            *w++ = ' ';
        } else {
            *w++ = *r;
        }
    }
    *w = 0;
}

static char *fetch_https(const char *url, size_t *n_out) {
    char q[8192 + 16], cmd[8192 + 128];
    shell_quote(url, q, sizeof(q));
    /* --proto keeps this bridge on https like the app allow-list. 1 MiB cap
     * matches loina_web's document limit. */
    snprintf(cmd, sizeof(cmd),
        "curl -sL --max-time 25 --proto '=https' --user-agent 'Loina/1.0' %s 2>/dev/null | head -c 1048576",
        q);
    FILE *f = popen(cmd, "r");
    if (!f)
        return NULL;
    size_t cap = 1 << 20, len = 0;
    char *buf = malloc(cap + 1);
    if (!buf) {
        pclose(f);
        return NULL;
    }
    size_t n;
    while ((n = fread(buf + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap)
            break;
    }
    pclose(f);
    buf[len] = 0;
    if (n_out)
        *n_out = len;
    return buf;
}

static void append_text(char **b, size_t *len, size_t *cap, const char *s, size_t n) {
    if (!n)
        return;
    if (*len + n + 1 > *cap) {
        size_t nc = *cap ? *cap * 2 : 8192;
        while (nc < *len + n + 1)
            nc *= 2;
        char *nb = realloc(*b, nc);
        if (!nb)
            return;
        *b = nb;
        *cap = nc;
    }
    memcpy(*b + *len, s, n);
    *len += n;
    (*b)[*len] = 0;
}

static void decode_entities(const char *s, size_t n, char **b, size_t *len, size_t *cap) {
    size_t i = 0;
    while (i < n) {
        if (s[i] == '&') {
            size_t semi = i + 1;
            while (semi < n && semi - i < 12 && s[semi] != ';')
                semi++;
            if (semi < n && s[semi] == ';') {
                size_t el = semi - i - 1;
                char ent[12] = {0};
                if (el < sizeof(ent))
                    memcpy(ent, s + i + 1, el);
                const char *rep = NULL;
                char tmp[8] = {0};
                if (!strcmp(ent, "amp"))
                    rep = "&";
                else if (!strcmp(ent, "lt"))
                    rep = "<";
                else if (!strcmp(ent, "gt"))
                    rep = ">";
                else if (!strcmp(ent, "quot"))
                    rep = "\"";
                else if (!strcmp(ent, "apos"))
                    rep = "'";
                else if (!strcmp(ent, "nbsp"))
                    rep = " ";
                else if (ent[0] == '#') {
                    unsigned v = 0;
                    if (ent[1] == 'x' || ent[1] == 'X') {
                        for (size_t k = 2; ent[k]; k++) {
                            int h = hexval(ent[k]);
                            if (h < 0) {
                                v = 0;
                                break;
                            }
                            v = v * 16 + (unsigned)h;
                        }
                    } else {
                        for (size_t k = 1; ent[k]; k++) {
                            if (!isdigit((unsigned char)ent[k])) {
                                v = 0;
                                break;
                            }
                            v = v * 10 + (unsigned)(ent[k] - '0');
                        }
                    }
                    if (v && v < 0x80) {
                        tmp[0] = (char)v;
                        rep = tmp;
                    } else if (v) {
                        rep = " ";
                    }
                }
                if (rep) {
                    append_text(b, len, cap, rep, strlen(rep));
                    i = semi + 1;
                    continue;
                }
            }
        }
        append_text(b, len, cap, s + i, 1);
        i++;
    }
}

/* Strip tags, drop script/style/nav/header/footer/aside/noscript, collapse
 * whitespace. Returns malloc'd text. */
static char *html_plain(const char *html, size_t n, const char **title_out) {
    char *raw = NULL;
    size_t len = 0, cap = 0;
    static char title[256];
    title[0] = 0;
    size_t i = 0;
    int skip = 0;
    bool space = true;
    while (i < n) {
        if (html[i] == '<') {
            size_t j = i + 1;
            while (j < n && html[j] != '>')
                j++;
            if (j >= n)
                break;
            size_t tl = j - i - 1;
            char tag[32] = {0};
            size_t k = i + 1, u = 0;
            if (k < j && html[k] == '/')
                k++;
            while (k < j && u + 1 < sizeof(tag) && !isspace((unsigned char)html[k]) && html[k] != '/')
                tag[u++] = (char)tolower((unsigned char)html[k++]);
            bool close = (i + 1 < j && html[i + 1] == '/');
            bool drop = !strcmp(tag, "script") || !strcmp(tag, "style") || !strcmp(tag, "nav") ||
                !strcmp(tag, "header") || !strcmp(tag, "footer") || !strcmp(tag, "aside") ||
                !strcmp(tag, "noscript");
            if (drop)
                skip = close ? 0 : 1;
            else if (skip && !close) {
                /* stay inside dropped element */
            } else if (skip && close) {
                skip = 0;
            }
            if (!skip && (!strcmp(tag, "p") || !strcmp(tag, "br") || !strcmp(tag, "li") ||
                    tag[0] == 'h' || !strcmp(tag, "tr") || !strcmp(tag, "div") || !strcmp(tag, "article"))) {
                if (!space) {
                    append_text(&raw, &len, &cap, " ", 1);
                    space = true;
                }
            }
            if (!skip && !close && !strcmp(tag, "title")) {
                size_t t = j + 1, e = t;
                while (e < n && !(html[e] == '<' && e + 7 < n && !strncasecmp(html + e, "</title", 7)))
                    e++;
                char *tb = NULL;
                size_t tl2 = 0, tc = 0;
                decode_entities(html + t, e > t ? e - t : 0, &tb, &tl2, &tc);
                size_t c = 0;
                for (size_t q = 0; q < tl2 && c + 1 < sizeof(title); q++) {
                    if (isspace((unsigned char)tb[q])) {
                        if (c && title[c - 1] != ' ')
                            title[c++] = ' ';
                    } else {
                        title[c++] = tb[q];
                    }
                }
                while (c && title[c - 1] == ' ')
                    c--;
                title[c] = 0;
                free(tb);
            }
           (void)tl;
            i = j + 1;
            continue;
        }
        if (skip) {
            i++;
            continue;
        }
        if (isspace((unsigned char)html[i])) {
            if (!space) {
                append_text(&raw, &len, &cap, " ", 1);
                space = true;
            }
        } else {
            char tmp[8];
            size_t tl3 = 0;
            /* Decode one entity or copy one byte; full UTF-8 passes through. */
            if (html[i] == '&') {
                size_t semi = i + 1;
                while (semi < n && semi - i < 12 && html[semi] != ';' && html[semi] != '<')
                    semi++;
                if (semi < n && html[semi] == ';') {
                    char *db = NULL;
                    size_t dl = 0, dc = 0;
                    decode_entities(html + i, semi - i + 1, &db, &dl, &dc);
                    append_text(&raw, &len, &cap, db ? db : "", dl);
                    free(db);
                    i = semi + 1;
                    space = false;
                    continue;
                }
            }
            tmp[tl3++] = html[i];
            append_text(&raw, &len, &cap, tmp, tl3);
            space = false;
        }
        i++;
    }
    if (title_out)
        *title_out = title[0] ? title : NULL;
    if (!raw) {
        raw = malloc(1);
        if (raw)
            *raw = 0;
    }
    return raw;
}

static char *inner_text(const char *html, size_t a, size_t b) {
    char *raw = NULL;
    size_t len = 0, cap = 0;
    size_t i = a;
    bool space = true;
    while (i < b) {
        if (html[i] == '<') {
            size_t j = i + 1;
            while (j < b && html[j] != '>')
                j++;
            if (j >= b)
                break;
            i = j + 1;
            continue;
        }
        if (isspace((unsigned char)html[i])) {
            if (!space) {
                append_text(&raw, &len, &cap, " ", 1);
                space = true;
            }
        } else if (html[i] == '&') {
            size_t semi = i + 1;
            while (semi < b && semi - i < 12 && html[semi] != ';')
                semi++;
            if (semi < b && html[semi] == ';') {
                char *db = NULL;
                size_t dl = 0, dc = 0;
                decode_entities(html + i, semi - i + 1, &db, &dl, &dc);
                append_text(&raw, &len, &cap, db ? db : "", dl);
                free(db);
                i = semi + 1;
                space = false;
                continue;
            }
            append_text(&raw, &len, &cap, html + i, 1);
            space = false;
        } else {
            append_text(&raw, &len, &cap, html + i, 1);
            space = false;
        }
        i++;
    }
    if (!raw) {
        raw = malloc(1);
        if (raw)
            *raw = 0;
    }
    /* trim */
    if (raw) {
        char *s = raw;
        while (*s && isspace((unsigned char)*s))
            s++;
        size_t l = strlen(s);
        while (l && isspace((unsigned char)s[l - 1]))
            s[--l] = 0;
        if (s != raw) {
            memmove(raw, s, l + 1);
        }
    }
    return raw;
}

static bool host_blocked(const char *url) {
    const char *h = strstr(url, "://");
    h = h ? h + 3 : url;
    size_t n = strcspn(h, "/");
    if (!n || n >= 256)
        return true;
    char host[256];
    for (size_t i = 0; i < n; i++)
        host[i] = (char)tolower((unsigned char)h[i]);
    host[n] = 0;
    if (!strcmp(host, "localhost") || strstr(host, ".local") || strchr(host, ':') || strchr(host, '@'))
        return true;
    bool ip = true;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)host[i]) && host[i] != '.')
            ip = false;
    return ip;
}

static loina_status web_fail(loina_error *error, loina_status code, const char *message) {
    if (error) {
        error->code = code;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return code;
}

static loina_status cli_web(void *user, const char *url, char **text, loina_job *job, loina_error *error) {
    (void)user;
    if (text)
        *text = NULL;
    if (!url || strncmp(url, "https://", 8) || host_blocked(url))
        return web_fail(error, LOINA_INVALID, "Web tools require an HTTPS URL");
    if (loina_job_cancelled(job))
        return web_fail(error, LOINA_CANCELLED, "Web request cancelled");
    size_t n = 0;
    char *html = fetch_https(url, &n);
    if (!html || !n) {
        free(html);
        /* DuckDuckGo without JS often fails under curl; fall back to Bing like the app. */
        if (strstr(url, "html.duckduckgo.com")) {
            const char *q = strstr(url, "?q=");
            char bing[8448];
            if (q) {
                snprintf(bing, sizeof(bing), "https://www.bing.com/search?q=%s", q + 3);
                html = fetch_https(bing, &n);
            }
        }
        if (!html || !n) {
            free(html);
            return web_fail(error, LOINA_IO, "Tool error: Web page could not be loaded");
        }
    }
    bool is_search = strstr(url, "duckduckgo.com") || strstr(url, "bing.com/search");
    char *out = NULL;
    size_t olen = 0, ocap = 0;
    if (is_search) {
        append_text(&out, &olen, &ocap, "# Search results\n\nURL: ", 22);
        append_text(&out, &olen, &ocap, url, strlen(url));
        append_text(&out, &olen, &ocap, "\n", 1);
        int found = 0;
        size_t pos = 0;
        while (found < 6 && pos < n) {
            const char *a = strstr(html + pos, "<a");
            if (!a)
                break;
            pos = (size_t)(a - html) + 2;
            const char *tag_end = strchr(a, '>');
            if (!tag_end)
                break;
            size_t tag_len = (size_t)(tag_end - a);
            char *tag = malloc(tag_len + 1);
            if (!tag)
                break;
            memcpy(tag, a, tag_len);
            tag[tag_len] = 0;
            for (char *c = tag; *c; c++)
                *c = (char)tolower((unsigned char)*c);
            if (!strstr(tag, "result__a") && !strstr(tag, "b_algo") && !strstr(tag, "b_title")) {
                free(tag);
                continue;
            }
            const char *href = strstr(a, "href=");
            char link[4096] = {0};
            if (href) {
                href += 5;
                while (*href == ' ' || *href == '"' || *href == '\'')
                    href++;
                size_t li = 0;
                while (*href && *href != '"' && *href != '\'' && *href != ' ' && *href != '>' && li + 1 < sizeof(link))
                    link[li++] = *href++;
                /* duckduckgo redirect: /l/?...&uddg=https%3A... */
                char *ud = strstr(link, "uddg=");
                if (ud) {
                    memmove(link, ud + 5, strlen(ud + 5) + 1);
                    char *amp = strchr(link, '&');
                    if (amp)
                        *amp = 0;
                    percent_decode_inplace(link);
                } else if (!strncmp(link, "//", 2)) {
                    memmove(link + 6, link + 2, strlen(link + 2) + 1);
                    memcpy(link, "https:", 6);
                }
            }
            free(tag);
            const char *label_start = tag_end + 1;
            const char *label_end = strstr(label_start, "</a>");
            if (!label_end) {
                pos = (size_t)(tag_end - html) + 1;
                continue;
            }
            if (strncmp(link, "https://", 8) || host_blocked(link) || strstr(link, "duckduckgo.com") ||
                strstr(link, "bing.com")) {
                pos = (size_t)(label_end - html) + 4;
                continue;
            }
            char *label = inner_text(html, (size_t)(label_start - html), (size_t)(label_end - html));
            if (!label || strlen(label) < 3) {
                free(label);
                pos = (size_t)(label_end - html) + 4;
                continue;
            }
            if (strlen(label) > 160)
                label[160] = 0;
            /* Nearby snippet for duckduckgo. */
            char *snippet = NULL;
            const char *sn = strstr(label_end, "result__snippet");
            if (sn && (size_t)(sn - html) < (size_t)(label_end - html) + 4096) {
                const char *gt = strchr(sn, '>');
                const char *se = gt ? strstr(gt, "</") : NULL;
                if (gt && se && se > gt)
                    snippet = inner_text(html, (size_t)(gt - html) + 1, (size_t)(se - html));
            }
            append_text(&out, &olen, &ocap, "\n- ", 3);
            append_text(&out, &olen, &ocap, label, strlen(label));
            append_text(&out, &olen, &ocap, ": ", 2);
            append_text(&out, &olen, &ocap, link, strlen(link));
            append_text(&out, &olen, &ocap, "\n", 1);
            if (snippet && *snippet) {
                if (strlen(snippet) > 640)
                    snippet[640] = 0;
                append_text(&out, &olen, &ocap, snippet, strlen(snippet));
                append_text(&out, &olen, &ocap, "\n", 1);
            }
            free(label);
            free(snippet);
            found++;
            pos = (size_t)(label_end - html) + 4;
        }
        if (!found) {
            free(html);
            free(out);
            return web_fail(error, LOINA_IO,
                "Tool error: Search returned no readable results. Try a more specific query.");
        }
    } else {
        const char *title = NULL;
        char *plain = html_plain(html, n, &title);
        if (!plain) {
            free(html);
            free(out);
            return web_fail(error, LOINA_MEMORY, "Cannot retain web page");
        }
        if (strlen(plain) > 9000)
            plain[9000] = 0;
        append_text(&out, &olen, &ocap, "# ", 2);
        append_text(&out, &olen, &ocap, title && *title ? title : url, strlen(title && *title ? title : url));
        append_text(&out, &olen, &ocap, "\n\nURL: ", 7);
        append_text(&out, &olen, &ocap, url, strlen(url));
        append_text(&out, &olen, &ocap, "\n\n## Page text\n", 14);
        append_text(&out, &olen, &ocap, plain, strlen(plain));
        free(plain);
    }
    free(html);
    if (!out) {
        return web_fail(error, LOINA_MEMORY, "Cannot retain web page");
    }
    if (loina_job_cancelled(job)) {
        free(out);
        return web_fail(error, LOINA_CANCELLED, "Web request cancelled");
    }
    *text = out;
    return LOINA_OK;
}

/* -- CLI plumbing, same app request shape. -- */

static void help(void) {
    puts("Loina - local inference (ds4-style CLI, same settings as the app)\n"
         "Usage: loina [options]\n"
         "  -p TEXT               Answer once; otherwise start a conversation\n"
         "  --prompt-file FILE    Read the one-shot prompt from a file\n"
         "  -m FILE               Use a catalog model file (import siblings if needed)\n"
         "  --package ID          Use an immutable package from --models\n"
         "  --import DIRECTORY    Install selected package from local artifacts\n"
         "  --models              List packages, licenses and installation state\n"
         "  --device              Show core device selection capabilities\n"
         "  --ctx N               Context size, maximum 32768 (default 32768)\n"
         "  -n N                  Maximum generated tokens (default 2048 thinking, 1024 plain)\n"
         "  --threads N           CPU threads\n"
         "  --backend NAME        CPU, Vulkan, OpenCL, Metal or CUDA\n"
         "  --temp F              Temperature; zero selects greedy output (default 0.1)\n"
         "  --thinking on|off     Enable reasoning when the package allows it (default on)\n"
         "  --tools on|off        Enable web tools for text/vision (default on, like the app)\n"
         "  --top-p F --min-p F   Sampling filters\n"
         "  --seed N              Sampling seed\n"
         "  --session ID          Resume a saved conversation\n"
         "  --cache-dir PATH      Packages, transcripts and KV (default .loina)\n"
         "  --templates PATH      Override bundled Jinja templates and tools.json\n"
         "  --vision FILE         Read an image through the vision layer\n"
         "  --generate --out FILE Generate a Sana Sprint PNG (default 1024x1024, two steps)\n"
         "  --width N --height N  Image dimensions, 64-1024 in multiples of 32\n"
         "  --steps N             Sana Sprint sampling steps, 1-4\n"
         "  --img2img FILE        Generate from an image and prompt\n"
         "  --strength F          Img2img change strength, 0-1 (default 0.75)\n"
         "  --transcribe FILE     Transcribe PCM16 mono 16kHz WAV\n"
         "REPL: /help /think /nothink /ctx N /tools on|off /save /list /switch ID /del ID /read FILE /new /quit\n"
         "Vision uses the chat weights plus mmproj, like the app. Web tools use a Linux curl bridge.");
}

static bool integer(const char *s, uint32_t *n) {
    if (!s || !*s || *s == '-')
        return false;
    char *end;
    errno = 0;
    unsigned long value = strtoul(s, &end, 10);
    if (errno || *end || value > UINT32_MAX)
        return false;
    *n = (uint32_t)value;
    return true;
}

static bool real(const char *s, float *n) {
    char *end;
    errno = 0;
    *n = strtof(s, &end);
    return !errno && end != s && !*end;
}

static void list_session(void *user, const char *id) {
    (void)user;
    puts(id);
}

static loina_session *new_session(loina_engine *engine, loina_error *error) {
    static unsigned sequence;
    char id[64];
    snprintf(id, sizeof(id), "chat-%llu-%u", (unsigned long long)time(NULL), ++sequence);
    return loina_session_create(engine, id, error);
}

static bool is_image_path(const char *path) {
    const char *dot = strrchr(path, '.');
    if (dot) {
        char ext[16] = {0};
        size_t i = 0;
        for (dot++; *dot && i + 1 < sizeof(ext); dot++, i++)
            ext[i] = (char)tolower((unsigned char)*dot);
        if (!strcmp(ext, "png") || !strcmp(ext, "jpg") || !strcmp(ext, "jpeg") ||
            !strcmp(ext, "webp") || !strcmp(ext, "bmp"))
            return true;
    }
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;
    unsigned char magic[12] = {0};
    size_t n = fread(magic, 1, sizeof(magic), f);
    fclose(f);
    if (n >= 8 && !memcmp(magic, "\x89PNG\r\n\x1a\n", 8))
        return true;
    if (n >= 2 && magic[0] == 0xFF && magic[1] == 0xD8)
        return true;
    if (n >= 12 && !memcmp(magic, "RIFF", 4) && !memcmp(magic + 8, "WEBP", 4))
        return true;
    if (n >= 2 && !memcmp(magic, "BM", 2))
        return true;
    return false;
}

static char *read_text_file(const char *path, size_t limit) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    rewind(f);
    if (n < 0 || (size_t)n > limit) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (n && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    if (memchr(buf, 0, (size_t)n))
        return buf; /* binary; caller treats as invalid text */
    return buf;
}

static void history_append(const char *line) {
    const char *home = getenv("HOME");
    if (!home || !*home || !*line)
        return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/.loina_history", home);
    FILE *f = fopen(path, "a");
    if (!f)
        return;
    fprintf(f, "%s\n", line);
    fclose(f);
}

static int run(loina_engine *engine, loina_request *request, loina_error *error) {
    interrupted = 0;
    cli_stream stream = {.out = stdout, .use_color = isatty(fileno(stdout)) != 0, .last_newline = true, .last_progress = -1};
    /* ds4-style thinking starts grey when the package reasons. */
    stream.in_think = false;
    loina_job *job = loina_job_create(engine, request, event_cb, &stream, error);
    if (!job)
        return 1;
    stream.job = job;
    loina_status status = loina_job_run(job, error);
    printer_process(&stream, NULL, 0, true);
    printer_reset(&stream);
    if (stream.last_progress >= 0 && isatty(fileno(stderr)))
        fputc('\n', stderr);
    if (!status && request->feature == LOINA_IMAGE) {
        puts(loina_job_result(job));
    } else if (!status) {
        if (!stream.last_newline)
            putchar('\n');
        if (stream.have_metrics) {
            bool ttyerr = isatty(fileno(stderr)) != 0;
            fprintf(stderr, "%sloina: prefill: %.2f t/s, generation: %.2f t/s%s\n",
                ttyerr ? "\x1b[32m" : "", stream.prefill, stream.decode, ttyerr ? "\x1b[0m" : "");
        }
    }
    if (status == LOINA_CANCELLED) {
        interrupted = 0;
        fprintf(stderr, "Interrupted.\n");
    }
    loina_job_free(job);
    return status ? 1 : 0;
}

int main(int argc, char **argv) {
    loina_config config = {.directory = ".loina", .web = cli_web};
    loina_request request = loina_request_default(LOINA_TEXT);
    const char *package_id = NULL, *model_file = NULL, *import = NULL, *session_id = NULL, *prompt = NULL,
               *prompt_file = NULL;
    char *prompt_owned = NULL;
    bool models = false, device = false, n_set = false, tools_on = true;
    unsigned backend = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            help();
            return 0;
        }
        if (!strcmp(a, "--models")) {
            models = true;
            continue;
        }
        if (!strcmp(a, "--device")) {
            device = true;
            continue;
        }
        if (!strcmp(a, "--generate")) {
            request.feature = LOINA_IMAGE;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value: %s\n", a);
            return 1;
        }
        const char *v = argv[++i];
        bool ok = true;
        uint32_t n;
        if (!strcmp(a, "-p"))
            prompt = v;
        else if (!strcmp(a, "--prompt-file"))
            prompt_file = v;
        else if (!strcmp(a, "--package"))
            package_id = v;
        else if (!strcmp(a, "-m"))
            model_file = v;
        else if (!strcmp(a, "--import"))
            import = v;
        else if (!strcmp(a, "--session"))
            session_id = v;
        else if (!strcmp(a, "--cache-dir"))
            config.directory = v;
        else if (!strcmp(a, "--templates"))
            config.template_directory = v;
        else if (!strcmp(a, "--ctx")) {
            ok = integer(v, &n);
            if (ok)
                config.context = n;
        } else if (!strcmp(a, "--threads")) {
            ok = integer(v, &n) && n <= 64 && n > 0;
            if (ok)
                config.device.threads = (int)n;
        } else if (!strcmp(a, "--backend")) {
            backend = !strcmp(v, "CPU") ? LOINA_CPU : !strcmp(v, "Vulkan") ? LOINA_VULKAN : !strcmp(v, "OpenCL") ? LOINA_OPENCL : !strcmp(v, "Metal") ? LOINA_METAL : !strcmp(v, "CUDA") ? LOINA_CUDA : 0;
            ok = backend != 0;
        } else if (!strcmp(a, "-n")) {
            ok = integer(v, &request.max_tokens);
            n_set = ok;
        } else if (!strcmp(a, "--seed"))
            ok = integer(v, &request.seed);
        else if (!strcmp(a, "--temp"))
            ok = real(v, &request.temperature);
        else if (!strcmp(a, "--thinking")) {
            ok = !strcmp(v, "on") || !strcmp(v, "off");
            request.thinking = !strcmp(v, "on");
        } else if (!strcmp(a, "--tools")) {
            ok = !strcmp(v, "on") || !strcmp(v, "off");
            tools_on = !strcmp(v, "on");
        } else if (!strcmp(a, "--top-p"))
            ok = real(v, &request.top_p);
        else if (!strcmp(a, "--min-p"))
            ok = real(v, &request.min_p);
        else if (!strcmp(a, "--width")) {
            ok = integer(v, &n) && n <= 1024;
            if (ok)
                request.width = (int)n;
        } else if (!strcmp(a, "--height")) {
            ok = integer(v, &n) && n <= 1024;
            if (ok)
                request.height = (int)n;
        } else if (!strcmp(a, "--steps")) {
            ok = integer(v, &n) && n <= 4;
            if (ok)
                request.steps = (int)n;
        } else if (!strcmp(a, "--vision")) {
            request.feature = LOINA_VISION;
            request.input_path = v;
        } else if (!strcmp(a, "--img2img")) {
            request.feature = LOINA_IMAGE;
            request.input_path = v;
        } else if (!strcmp(a, "--strength"))
            ok = real(v, &request.strength);
        else if (!strcmp(a, "--transcribe")) {
            request.feature = LOINA_SPEECH;
            request.input_path = v;
        } else if (!strcmp(a, "--out"))
            request.output_path = v;
        else {
            fprintf(stderr, "Unknown option: %s\n", a);
            return 1;
        }
        if (!ok) {
            fprintf(stderr, "Invalid value for %s: %s\n", a, v);
            return 1;
        }
    }
    if (prompt && prompt_file) {
        fprintf(stderr, "Specify only one of -p and --prompt-file\n");
        return 1;
    }
    if (prompt_file) {
        prompt_owned = read_text_file(prompt_file, 16u * 1024u * 1024u);
        if (!prompt_owned) {
            fprintf(stderr, "Cannot read prompt file: %s\n", prompt_file);
            return 1;
        }
        prompt = prompt_owned;
    }
    /* Same app shape: web tools for chat/vision, longer answers with thinking. */
    if (!n_set)
        request.max_tokens = request.thinking ? 2048 : 1024;
    signal(SIGINT, on_signal);
    loina_error error = {0};
    loina_engine *engine = loina_open(&config, &error);
    if (!engine) {
        fprintf(stderr, "%s\n", error.message);
        free(prompt_owned);
        return 1;
    }
    if (backend && loina_set_backend(engine, backend, &error)) {
        fprintf(stderr, "%s\n", error.message);
        loina_close(engine);
        free(prompt_owned);
        return 1;
    }
    int exitcode = 0;
    if (package_id) {
        request.package = loina_catalog_find(package_id);
        if (!request.package) {
            fprintf(stderr, "Unknown package; see --models\n");
            exitcode = 1;
            goto done;
        }
    }
    char model_directory[4096];
    if (model_file) {
        const char *base = model_file;
        for (const char *c = model_file; *c; c++)
            if (*c == '/' || *c == '\\')
                base = c + 1;
        const loina_package *found = NULL;
        for (size_t i = 0; i < loina_catalog_count(); i++) {
            const loina_package *p = loina_catalog_get(i);
            if (p->feature == request.feature && !strcmp(p->files[0].name, base)) {
                found = p;
                break;
            }
        }
        if (!found || (request.package && request.package != found)) {
            fprintf(stderr, "Unknown or conflicting model file; see --models\n");
            exitcode = 1;
            goto done;
        }
        request.package = found;
        size_t length = (size_t)(base - model_file);
        if (length >= sizeof(model_directory)) {
            fprintf(stderr, "Model path too long\n");
            exitcode = 1;
            goto done;
        }
        if (length) {
            memcpy(model_directory, model_file, length);
            model_directory[length] = 0;
        } else
            strcpy(model_directory, ".");
        if (!loina_package_installed(engine, found))
            import = model_directory;
    }
    if (import) {
        const loina_package *p = request.package ? request.package : loina_resolve(engine, request.feature, &error);
        if (!p || loina_install(engine, p, import, NULL, &error)) {
            fprintf(stderr, "%s\n", error.message);
            exitcode = 1;
            goto done;
        }
        if (!prompt && !models && !device && !model_file && request.feature != LOINA_SPEECH)
            goto done;
    }
    if (device) {
        const loina_device *d = loina_get_device(engine);
        printf("RAM: %llu; available: %llu; free storage: %llu; threads: %d; backends: 0x%x\n",
            (unsigned long long)d->ram_bytes, (unsigned long long)d->available_ram_bytes,
            (unsigned long long)d->storage_bytes, d->threads, d->backends);
    }
    if (models)
        for (size_t i = 0; i < loina_catalog_count(); i++) {
            const loina_package *p = loina_catalog_get(i);
            printf("%s\n  %s | %s | %llu bytes | %s\n", p->id, p->name, p->license,
                (unsigned long long)loina_package_bytes(p),
                loina_package_installed(engine, p) ? "installed" : "not installed");
        }
    if (models || device)
        goto done;
    request.tools = tools_on && (request.feature == LOINA_TEXT || request.feature == LOINA_VISION);
    if (request.feature == LOINA_SPEECH) {
        request.prompt = prompt ? prompt : "";
        /* Speech jobs need a prompt field; the WAV path carries the audio. */
        if (!prompt)
            request.prompt = "transcribe";
        exitcode = run(engine, &request, &error);
        if (exitcode)
            fprintf(stderr, "%s\n", error.message);
        goto done;
    }
    request.session = session_id ? loina_session_load(engine, session_id, &error) : new_session(engine, &error);
    if (!request.session) {
        fprintf(stderr, "%s\n", error.message);
        exitcode = 1;
        goto done;
    }
    if (request.feature == LOINA_IMAGE && !request.output_path)
        request.output_path = "loina.png";
    if (prompt) {
        request.prompt = prompt;
        request.tools = tools_on && (request.feature == LOINA_TEXT || request.feature == LOINA_VISION);
        exitcode = run(engine, &request, &error);
        if (exitcode)
            fprintf(stderr, "%s\n", error.message);
    } else {
        char line[16384];
        bool tty = isatty(fileno(stdin)) != 0;
        puts("Loina. /help for commands.");
        while (fputs(tty ? "loina> " : "", stdout), fflush(stdout), fgets(line, sizeof(line), stdin)) {
            line[strcspn(line, "\r\n")] = 0;
            char *cmd = line;
            while (*cmd && isspace((unsigned char)*cmd))
                cmd++;
            char *end = cmd + strlen(cmd);
            while (end > cmd && isspace((unsigned char)end[-1]))
                *--end = 0;
            if (!*cmd)
                continue;
            history_append(cmd);
            if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit"))
                break;
            if (!strcmp(cmd, "/help")) {
                help();
                continue;
            }
            if (!strcmp(cmd, "/think")) {
                request.thinking = true;
                if (!n_set)
                    request.max_tokens = 2048;
                puts("Thinking on (when the package allows it).");
                continue;
            }
            if (!strcmp(cmd, "/nothink")) {
                request.thinking = false;
                if (!n_set)
                    request.max_tokens = 1024;
                puts("Thinking off.");
                continue;
            }
            if (!strncmp(cmd, "/ctx ", 5)) {
                uint32_t n = 0;
                if (integer(cmd + 5, &n) && n >= 512 && n <= 32768) {
                    /* Context needs an engine reopen; simplest is to report it
                     * for the next launch, keeping the session append-only. */
                    fprintf(stderr, "Context applies at startup; relaunch with --ctx %u\n", n);
                } else {
                    fprintf(stderr, "Invalid context (512-32768)\n");
                }
                continue;
            }
            if (!strncmp(cmd, "/tools ", 7)) {
                if (!strcmp(cmd + 7, "on"))
                    tools_on = true;
                else if (!strcmp(cmd + 7, "off"))
                    tools_on = false;
                else
                    fprintf(stderr, "Usage: /tools on|off\n");
                continue;
            }
            if (!strcmp(cmd, "/save")) {
                if (loina_session_save(request.session, &error))
                    fprintf(stderr, "%s\n", error.message);
                else
                    puts(loina_session_id(request.session));
                continue;
            }
            if (!strcmp(cmd, "/list")) {
                loina_session_list(engine, list_session, NULL, &error);
                continue;
            }
            if (!strncmp(cmd, "/del ", 5)) {
                if (loina_session_delete(engine, cmd + 5, &error))
                    fprintf(stderr, "%s\n", error.message);
                continue;
            }
            if (!strcmp(cmd, "/new") || !strncmp(cmd, "/switch ", 8)) {
                loina_session *next = !strcmp(cmd, "/new") ? new_session(engine, &error)
                                                           : loina_session_load(engine, cmd + 8, &error);
                if (next) {
                    loina_session_free(request.session);
                    request.session = next;
                } else
                    fprintf(stderr, "%s\n", error.message);
                continue;
            }
            if (!strncmp(cmd, "/read ", 6)) {
                char *path = cmd + 6;
                while (*path && isspace((unsigned char)*path))
                    path++;
                if (!*path) {
                    fprintf(stderr, "/read needs a file path\n");
                    continue;
                }
                if (is_image_path(path)) {
                    request.feature = LOINA_VISION;
                    request.input_path = path;
                    request.prompt = "Describe this image and extract its text.";
                    request.tools = tools_on;
                } else {
                    char *doc = read_text_file(path, 1024u * 1024u);
                    if (!doc) {
                        fprintf(stderr, "Cannot read text file (1 MB limit): %s\n", path);
                        continue;
                    }
                    if (memchr(doc, 0, strlen(doc))) {
                        fprintf(stderr, "Not a text file; use an image path for vision: %s\n", path);
                        free(doc);
                        continue;
                    }
                    const char *base = strrchr(path, '/');
                    base = base ? base + 1 : path;
                    size_t need = strlen(doc) + strlen(base) + 32;
                    char *wrapped = malloc(need);
                    if (!wrapped) {
                        free(doc);
                        fprintf(stderr, "Out of memory\n");
                        continue;
                    }
                    snprintf(wrapped, need, "[Document: %s]\n%s", base, doc);
                    free(doc);
                    request.feature = LOINA_TEXT;
                    request.input_path = NULL;
                    request.prompt = wrapped;
                    request.tools = tools_on;
                    if (run(engine, &request, &error))
                        fprintf(stderr, "%s\n", error.message);
                    free(wrapped);
                    request.feature = LOINA_TEXT;
                    request.input_path = NULL;
                    continue;
                }
            } else {
                request.prompt = cmd;
                if (request.feature != LOINA_VISION || !request.input_path) {
                    request.feature = LOINA_TEXT;
                    request.input_path = NULL;
                }
                request.tools = tools_on;
            }
            if (run(engine, &request, &error))
                fprintf(stderr, "%s\n", error.message);
            request.feature = LOINA_TEXT;
            request.input_path = NULL;
            if (interrupted)
                interrupted = 0;
        }
    }
    loina_session_free(request.session);
done:
    loina_close(engine);
    free(prompt_owned);
    return exitcode;
}

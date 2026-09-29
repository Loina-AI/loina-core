#include "loina_internal.h"
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

loina_status loina_fail(loina_error *error, loina_status code, const char *format, ...) {
    if (error) {
        error->code = code;
        va_list ap;
        va_start(ap, format);
        vsnprintf(error->message, sizeof(error->message), format, ap);
        va_end(ap);
    }
    return code;
}
void loina_free(void *pointer) { free(pointer); }
char *loina_strdup(const char *text) {
    if (!text) text = "";
    size_t n = strlen(text) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, text, n);
    return p;
}
bool loina_name_valid(const char *name) {
    if (!name || !*name || strlen(name) >= 128 || !strcmp(name, ".") || !strcmp(name, "..")) return false;
    for (; *name; name++) {
        unsigned char c = (unsigned char)*name;
        if (!isalnum(c) && c != '-' && c != '_' && c != '.') return false;
    }
    return true;
}
bool loina_buffer_add(loina_buffer *b, const void *data, size_t n) {
    if (n > LOINA_MAX_TEXT || b->length > LOINA_MAX_TEXT - n) return false;
    size_t need = b->length + n + 1;
    if (need > b->capacity) {
        size_t cap = b->capacity ? b->capacity * 2 : 1024;
        if (cap < need) cap = need;
        char *p = realloc(b->data, cap);
        if (!p) return false;
        b->data = p;
        b->capacity = cap;
    }
    if (n) memcpy(b->data + b->length, data, n);
    b->length += n;
    b->data[b->length] = 0;
    return true;
}
bool loina_buffer_puts(loina_buffer *b, const char *s) { return loina_buffer_add(b, s, strlen(s)); }
bool loina_buffer_json(loina_buffer *b, const char *s) {
    if (!loina_buffer_puts(b, "\"")) return false;
    for (; *s; s++) {
        char c[7];
        unsigned char x = (unsigned char)*s;
        if (x == '"' || x == '\\') { c[0] = '\\'; c[1] = (char)x; c[2] = 0; }
        else if (x < 32) snprintf(c, sizeof(c), "\\u%04x", x);
        else { c[0] = (char)x; c[1] = 0; }
        if (!loina_buffer_puts(b, c)) return false;
    }
    return loina_buffer_puts(b, "\"");
}
void loina_emit(loina_job *job, loina_event event, const char *text, size_t n, int progress) {
    if (job && job->event) job->event(job->user, event, text ? text : "", n, progress);
}
bool loina_path(char *out, const char *base, const char *a, const char *b) {
    int n = snprintf(out, LOINA_PATH, "%s/%s%s%s", base, a, b ? "/" : "", b ? b : "");
    return n >= 0 && n < LOINA_PATH;
}
bool loina_mkdir(const char *path) {
    char buf[LOINA_PATH];
    if (strlen(path) >= sizeof(buf)) return false;
    strcpy(buf, path);
    for (char *p = buf + 1; ; p++) {
        if (*p != '/' && *p != '\\' && *p) continue;
        char c = *p;
        *p = 0;
#ifdef _WIN32
        int rc = _mkdir(buf);
#else
        int rc = mkdir(buf, 0700);
#endif
        *p = c;
        if (rc && errno != EEXIST) return false;
        if (!c) break;
    }
    struct stat st;
    return !stat(path, &st) && (st.st_mode & S_IFDIR);
}
bool loina_replace(const char *from, const char *to) {
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(from, to) == 0;
#endif
}
uint64_t loina_file_bytes(const char *path) {
#ifdef _WIN32
    struct _stat64 st;
    return !_stat64(path, &st) && (st.st_mode & S_IFREG) ? (uint64_t)st.st_size : UINT64_MAX;
#else
    struct stat st;
    return !stat(path, &st) && S_ISREG(st.st_mode) ? (uint64_t)st.st_size : UINT64_MAX;
#endif
}
loina_status loina_copy(const char *from, const char *to, loina_job *job, uint64_t total, loina_error *error) {
    FILE *src = fopen(from, "rb"), *dst = NULL;
    if (src) dst = fopen(to, "wb");
    if (!src || !dst) {
        if (src) fclose(src);
        return loina_fail(error, LOINA_IO, "Cannot copy %s", from);
    }
    unsigned char buf[65536];
    size_t n;
    uint64_t done = 0;
    loina_status status = LOINA_OK;
    while ((n = fread(buf, 1, sizeof(buf), src))) {
        if (loina_job_cancelled(job)) { status = LOINA_CANCELLED; break; }
        if (fwrite(buf, 1, n, dst) != n) { status = LOINA_IO; break; }
        done += n;
        loina_job_progress(job, done, total);
    }
    if (ferror(src)) status = LOINA_IO;
    fclose(src);
    if (fclose(dst)) status = LOINA_IO;
    if (status) return loina_fail(error, status, "Copy interrupted or failed: %s", from);
    return LOINA_OK;
}

static const uint32_t sha_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
static void sha_block(loina_sha *s) {
    uint32_t w[64];
    for (int i=0; i<16; i++) {
        const unsigned char *p = s->block + i*4;
        w[i] = (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
    }
    for (int i=16; i<64; i++) {
        uint32_t a=w[i-15], b=w[i-2];
        w[i]=w[i-16]+(ror(a,7)^ror(a,18)^(a>>3))+w[i-7]+(ror(b,17)^ror(b,19)^(b>>10));
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (int i=0; i<64; i++) {
        uint32_t t=h+(ror(e,6)^ror(e,11)^ror(e,25))+((e&f)^(~e&g))+sha_k[i]+w[i];
        uint32_t u=(ror(a,2)^ror(a,13)^ror(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
void loina_sha_init(loina_sha *s) {
    *s=(loina_sha){.h={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
}
void loina_sha_update(loina_sha *s, const void *data, size_t n) {
    const unsigned char *p=data;
    s->bytes+=n;
    while (n) {
        size_t take=64-s->used;
        if (take>n) take=n;
        memcpy(s->block+s->used,p,take);
        s->used+=take;p+=take;n-=take;
        if (s->used==64) { sha_block(s);s->used=0; }
    }
}
void loina_sha_final(loina_sha *s, char hex[65]) {
    uint64_t bits=s->bytes*8;
    unsigned char one=0x80,zero=0;
    loina_sha_update(s,&one,1);
    while (s->used!=56) loina_sha_update(s,&zero,1);
    unsigned char len[8];
    for (int i=0;i<8;i++) len[7-i]=(unsigned char)(bits>>(i*8));
    loina_sha_update(s,len,8);
    for (int i=0;i<8;i++) snprintf(hex+i*8,9,"%08x",s->h[i]);
}
loina_status loina_hash_file(const char *path, char hex[65], loina_job *job, loina_error *error) {
    FILE *f=fopen(path,"rb");
    if (!f) return loina_fail(error,LOINA_IO,"Cannot read %s",path);
    loina_sha s;
    loina_sha_init(&s);
    unsigned char buf[65536];
    size_t n;
    while ((n=fread(buf,1,sizeof(buf),f))) {
        if (loina_job_cancelled(job)) { fclose(f);return loina_fail(error,LOINA_CANCELLED,"Verification cancelled"); }
        loina_sha_update(&s,buf,n);
    }
    bool ok=!ferror(f);
    fclose(f);
    if (!ok) return loina_fail(error,LOINA_IO,"Cannot hash %s",path);
    loina_sha_final(&s,hex);
    return LOINA_OK;
}

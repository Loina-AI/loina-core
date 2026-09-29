#include "loina_internal.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

static bool session_path(loina_engine *e, const char *id, char *path) {
    char name[160];
    if (!loina_name_valid(id)) return false;
    snprintf(name,sizeof(name),"%s.chat",id);
    return loina_path(path,e->directory,"sessions",name);
}
loina_session *loina_session_create(loina_engine *e, const char *id, loina_error *error) {
    if (!e || !loina_name_valid(id)) { loina_fail(error,LOINA_INVALID,"Invalid session ID");return NULL; }
    loina_session *s=calloc(1,sizeof(*s));
    if (!s) { loina_fail(error,LOINA_MEMORY,"Cannot allocate session");return NULL; }
    s->engine=e;strcpy(s->id,id);
    return s;
}
void loina_session_truncate(loina_session *s, size_t n) {
    while (s->count>n) free((void *)s->messages[--s->count].text);
    if (s->first>s->count) s->first=s->count;
}
void loina_session_free(loina_session *s) {
    if (!s) return;
    loina_session_truncate(s,0);free(s->messages);free(s);
}
const char *loina_session_id(const loina_session *s) { return s ? s->id : ""; }
size_t loina_session_count(const loina_session *s) { return s ? s->count : 0; }
loina_message loina_session_get(const loina_session *s, size_t i) {
    return s && i<s->count ? s->messages[i] : (loina_message){0};
}
bool loina_session_add(loina_session *s, loina_role role, loina_message_kind kind, const char *text) {
    if (s->count==s->capacity) {
        size_t cap=s->capacity ? s->capacity*2 : 16;
        if (cap>100000) return false;
        loina_message *m=realloc(s->messages,cap*sizeof(*m));
        if (!m) return false;
        s->messages=m;s->capacity=cap;
    }
    char *copy=loina_strdup(text);
    if (!copy) return false;
    s->messages[s->count++]=(loina_message){role,kind,copy};
    return true;
}
static bool put32(FILE *f, uint32_t x) {
    unsigned char b[4]={(unsigned char)x,(unsigned char)(x>>8),(unsigned char)(x>>16),(unsigned char)(x>>24)};
    return fwrite(b,1,4,f)==4;
}
static bool get32(FILE *f, uint32_t *x) {
    unsigned char b[4];
    if (fread(b,1,4,f)!=4) return false;
    *x=(uint32_t)b[0] | (uint32_t)b[1]<<8 | (uint32_t)b[2]<<16 | (uint32_t)b[3]<<24;
    return true;
}
loina_status loina_session_save(loina_session *s, loina_error *error) {
    if (!s) return loina_fail(error,LOINA_INVALID,"Missing session");
    char path[LOINA_PATH],temp[LOINA_PATH];
    if (!session_path(s->engine,s->id,path) || snprintf(temp,sizeof(temp),"%s.tmp",path)>=(int)sizeof(temp))
        return loina_fail(error,LOINA_INVALID,"Session path is too long");
    FILE *f=fopen(temp,"wb");
    if (!f) return loina_fail(error,LOINA_IO,"Cannot save conversation");
    uint32_t len=(uint32_t)strlen(s->package_id);
    bool ok=fwrite("LOINA01\n",1,8,f)==8 && put32(f,(uint32_t)s->first) && put32(f,(uint32_t)s->count) &&
        put32(f,len) && fwrite(s->package_id,1,len,f)==len;
    size_t total=0;
    for (size_t i=0;ok && i<s->count;i++) {
        const loina_message *m=&s->messages[i];
        size_t n=strlen(m->text);
        total+=n;
        if (total>LOINA_MAX_TEXT) { ok=false;break; }
        ok=put32(f,(uint32_t)m->role) && put32(f,(uint32_t)m->kind) && put32(f,(uint32_t)n) && fwrite(m->text,1,n,f)==n;
    }
    if (fclose(f)) ok=false;
    if (ok) ok=loina_replace(temp,path);
    if (!ok) { remove(temp);return loina_fail(error,LOINA_IO,"Cannot commit conversation"); }
    return LOINA_OK;
}
loina_session *loina_session_load(loina_engine *e, const char *id, loina_error *error) {
    loina_session *s=loina_session_create(e,id,error);
    if (!s) return NULL;
    char path[LOINA_PATH],magic[8];
    if (!session_path(e,id,path)) goto invalid;
    FILE *f=fopen(path,"rb");
    if (!f) { loina_session_free(s);loina_fail(error,LOINA_IO,"Conversation does not exist: %s",id);return NULL; }
    uint32_t first,count,len;
    bool ok=fread(magic,1,8,f)==8 && !memcmp(magic,"LOINA01\n",8) && get32(f,&first) && get32(f,&count) &&
        count<=100000 && first<=count && get32(f,&len) && len<sizeof(s->package_id) &&
        fread(s->package_id,1,len,f)==len;
    if (ok && len && (memchr(s->package_id,0,len) || !loina_name_valid(s->package_id))) ok=false;
    size_t total=0;
    for (uint32_t i=0;ok && i<count;i++) {
        uint32_t role,kind,n;
        ok=get32(f,&role) && role<=LOINA_ASSISTANT && get32(f,&kind) && kind<=LOINA_MESSAGE_IMAGE &&
            get32(f,&n) && n<=LOINA_MAX_TEXT && total<=LOINA_MAX_TEXT-n;
        if (!ok) break;
        total+=n;
        char *text=malloc((size_t)n+1);
        if (!text) { ok=false;break; }
        ok=fread(text,1,n,f)==n && !memchr(text,0,n);
        text[n]=0;
        if (ok) ok=loina_session_add(s,(loina_role)role,(loina_message_kind)kind,text);
        free(text);
    }
    if (ok && fgetc(f)!=EOF) ok=false;
    if (ferror(f)) ok=false;
    fclose(f);
    if (!ok) goto invalid;
    s->first=first;
    return s;
invalid:
    loina_session_free(s);loina_fail(error,LOINA_INTEGRITY,"Invalid conversation file");return NULL;
}
loina_status loina_session_delete(loina_engine *e, const char *id, loina_error *error) {
    char path[LOINA_PATH];
    if (!e || !session_path(e,id,path)) return loina_fail(error,LOINA_INVALID,"Invalid session ID");
    if (atomic_load(&e->busy)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    if (remove(path) && errno!=ENOENT) return loina_fail(error,LOINA_IO,"Cannot delete conversation");
    /* Cache files live under a build/package fingerprint. Delete the matching
     * session name in every cache directory, including retired model builds. */
#ifdef _WIN32
    char pattern[LOINA_PATH];
    loina_path(pattern,e->directory,"cache","*");
    WIN32_FIND_DATAA entry;
    HANDLE h=FindFirstFileA(pattern,&entry);
    if (h!=INVALID_HANDLE_VALUE) {
        do {
            if (!loina_name_valid(entry.cFileName) || !strcmp(entry.cFileName,".") || !strcmp(entry.cFileName,"..")) continue;
            char dir[LOINA_PATH],name[160];
            loina_path(dir,e->directory,"cache",entry.cFileName);
            snprintf(name,sizeof(name),"%s.kv",id);loina_path(path,dir,name,NULL);remove(path);
            snprintf(name,sizeof(name),"%s.stamp",id);loina_path(path,dir,name,NULL);remove(path);
        } while (FindNextFileA(h,&entry));
        FindClose(h);
    }
#else
    char root[LOINA_PATH];
    loina_path(root,e->directory,"cache",NULL);
    DIR *d=opendir(root);
    if (d) {
        struct dirent *entry;
        while ((entry=readdir(d))) {
            if (!loina_name_valid(entry->d_name)) continue;
            char dir[LOINA_PATH],name[160];
            loina_path(dir,root,entry->d_name,NULL);
            snprintf(name,sizeof(name),"%s.kv",id);loina_path(path,dir,name,NULL);remove(path);
            snprintf(name,sizeof(name),"%s.stamp",id);loina_path(path,dir,name,NULL);remove(path);
        }
        closedir(d);
    }
#endif
    return LOINA_OK;
}
static void list_entry(const char *name, loina_session_fn fn, void *user) {
    size_t n=strlen(name);
    if (n<6 || n>=133 || strcmp(name+n-5,".chat")) return;
    char id[128];memcpy(id,name,n-5);id[n-5]=0;
    if (loina_name_valid(id)) fn(user,id);
}
loina_status loina_session_list(loina_engine *e, loina_session_fn fn, void *user, loina_error *error) {
    if (!e || !fn) return loina_fail(error,LOINA_INVALID,"Missing session list arguments");
    char path[LOINA_PATH];
#ifdef _WIN32
    loina_path(path,e->directory,"sessions","*.chat");
    WIN32_FIND_DATAA entry;
    HANDLE h=FindFirstFileA(path,&entry);
    if (h==INVALID_HANDLE_VALUE) return LOINA_OK;
    do { if (!(entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) list_entry(entry.cFileName,fn,user); } while (FindNextFileA(h,&entry));
    FindClose(h);
#else
    loina_path(path,e->directory,"sessions",NULL);
    DIR *d=opendir(path);
    if (!d) return loina_fail(error,LOINA_IO,"Cannot list conversations");
    struct dirent *entry;
    while ((entry=readdir(d))) list_entry(entry->d_name,fn,user);
    closedir(d);
#endif
    return LOINA_OK;
}

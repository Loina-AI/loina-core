#include "loina_internal.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define rmdir _rmdir
#else
#include <unistd.h>
#endif

const loina_package *loina_catalog_find(const char *id) {
    if (!id) return NULL;
    for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *p=loina_catalog_get(i);
        if (!strcmp(p->id,id)) return p;
    }
    return NULL;
}
bool loina_package_valid(const loina_package *p) {
    if (!p || !loina_name_valid(p->id) || !p->name || !p->license || !p->runtime ||
        !p->file_count || p->file_count>LOINA_MAX_FILES || !p->runtime_abi ||
        p->feature<LOINA_TEXT || p->feature>LOINA_SPEECH) return false;
    for (size_t i=0;i<p->file_count;i++) {
        const loina_artifact *a=&p->files[i];
        if (!loina_name_valid(a->name) || !a->sha256 || strlen(a->sha256)!=64 || !a->bytes) return false;
        for (const char *s=a->sha256;*s;s++) if (!isxdigit((unsigned char)*s) || (*s>='A' && *s<='F')) return false;
        if (a->url && *a->url && strncmp(a->url,"https://",8)) return false;
        for (size_t j=0;j<i;j++) if (!strcmp(a->name,p->files[j].name)) return false;
    }
    return loina_package_bytes(p)!=UINT64_MAX;
}
uint64_t loina_package_bytes(const loina_package *p) {
    if (!p || p->file_count>LOINA_MAX_FILES) return UINT64_MAX;
    uint64_t total=0;
    for (size_t i=0;i<p->file_count;i++) {
        if (p->files[i].bytes>UINT64_MAX-total) return UINT64_MAX;
        total+=p->files[i].bytes;
    }
    return total;
}
void loina_package_identity(const loina_package *p, char hex[65]) {
    loina_sha s;
    loina_sha_init(&s);
    loina_sha_update(&s,p->id,strlen(p->id)+1);
    loina_sha_update(&s,p->runtime,strlen(p->runtime)+1);
    char fields[128];
    int n=snprintf(fields,sizeof(fields),"%u:%u:%zu",p->runtime_abi,(unsigned)p->feature,p->file_count);
    loina_sha_update(&s,fields,(size_t)n+1);
    for (size_t i=0;i<p->file_count;i++) {
        const loina_artifact *a=&p->files[i];
        loina_sha_update(&s,a->name,strlen(a->name)+1);
        loina_sha_update(&s,a->sha256,65);
        n=snprintf(fields,sizeof(fields),"%llu",(unsigned long long)a->bytes);
        loina_sha_update(&s,fields,(size_t)n+1);
    }
    loina_sha_final(&s,hex);
}
bool loina_artifact_path(loina_engine *e, const loina_package *p, size_t i, char path[LOINA_PATH]) {
    char dir[LOINA_PATH];
    return e && loina_package_valid(p) && i<p->file_count &&
        loina_path(dir,e->directory,"packages",p->id) && loina_path(path,dir,p->files[i].name,NULL);
}
bool loina_package_installed(loina_engine *e, const loina_package *p) {
    if (!e || !loina_package_valid(p)) return false;
    char dir[LOINA_PATH],path[LOINA_PATH],want[65],have[66]={0};
    if (!loina_path(dir,e->directory,"packages",p->id) || !loina_path(path,dir,"ready",NULL)) return false;
    FILE *f=fopen(path,"rb");
    if (!f) return false;
    size_t n=fread(have,1,sizeof(have),f);
    fclose(f);
    loina_package_identity(p,want);
    if (n!=64 || memcmp(want,have,64)) return false;
    for (size_t i=0;i<p->file_count;i++) {
        if (!loina_artifact_path(e,p,i,path) || loina_file_bytes(path)!=p->files[i].bytes) return false;
    }
    return true;
}

loina_status loina_install(loina_engine *e, const loina_package *p, const char *import,
                          loina_job *job, loina_error *error) {
    if (!e || !loina_package_valid(p)) return loina_fail(error,LOINA_INVALID,"Invalid package manifest");
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    loina_status status=LOINA_OK;
    char dir[LOINA_PATH],stage[LOINA_PATH],part[256],path[LOINA_PATH],source[LOINA_PATH],hash[65];
    snprintf(part,sizeof(part),"%s.partial",p->id);
    if (!loina_path(dir,e->directory,"packages",p->id) || !loina_path(stage,e->directory,"packages",part)) {
        status=loina_fail(error,LOINA_INVALID,"Package path is too long"); goto done;
    }
    if (loina_package_installed(e,p)) goto done;
    /* A failed or conflicting final directory is never reused. Partial files
     * may be resumed, but every artifact is rehashed before it becomes visible. */
    if (!loina_path(path,dir,"ready",NULL)) { status=LOINA_INVALID; goto done; }
    if (loina_file_bytes(path)!=UINT64_MAX) { status=loina_fail(error,LOINA_INTEGRITY,"Installed package identity conflicts: %s",p->id); goto done; }
    if (!loina_mkdir(stage)) { status=loina_fail(error,LOINA_IO,"Cannot create package staging directory"); goto done; }
    for (size_t i=0;i<p->file_count;i++) {
        const loina_artifact *a=&p->files[i];
        if (loina_job_cancelled(job)) { status=loina_fail(error,LOINA_CANCELLED,"Installation cancelled"); goto done; }
        if (!loina_path(path,stage,a->name,NULL)) { status=LOINA_INVALID; goto done; }
        if (import) {
            if (!loina_path(source,import,a->name,NULL)) { status=LOINA_INVALID; goto done; }
            status=loina_copy(source,path,job,a->bytes,error);
        } else if (loina_file_bytes(path)!=a->bytes) {
            if (!e->config.fetch || !a->url || !*a->url) { status=loina_fail(error,LOINA_UNSUPPORTED,"%s requires local import or a download bridge",a->name); goto done; }
            status=e->config.fetch(e->config.user,a,path,job,error);
        }
        if (status) goto done;
        if (loina_file_bytes(path)!=a->bytes) { status=loina_fail(error,LOINA_INTEGRITY,"Wrong artifact size: %s",a->name); goto done; }
        status=loina_hash_file(path,hash,job,error);
        if (status) goto done;
        if (strcmp(hash,a->sha256)) { remove(path); status=loina_fail(error,LOINA_INTEGRITY,"SHA-256 mismatch: %s",a->name); goto done; }
    }
    loina_package_identity(p,hash);
    if (!loina_path(path,stage,"ready",NULL)) { status=LOINA_INVALID; goto done; }
    FILE *f=fopen(path,"wb");
    bool ok=f && fwrite(hash,1,64,f)==64;
    if (f && fclose(f)) ok=false;
    if (!ok || rename(stage,dir)) status=loina_fail(error,LOINA_IO,"Cannot commit immutable package %s",p->id);
done:
    atomic_store(&e->busy,false);
    return status;
}
loina_status loina_uninstall(loina_engine *e, const loina_package *p, loina_error *error) {
    if (!e || !loina_package_valid(p)) return loina_fail(error,LOINA_INVALID,"Invalid package");
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    if (e->package && !strcmp(e->package->id,p->id)) loina_unload(e);
    char dir[LOINA_PATH],path[LOINA_PATH];
    loina_status status=LOINA_OK;
    if (!loina_path(dir,e->directory,"packages",p->id)) status=LOINA_INVALID;
    else {
        for (size_t i=0;i<p->file_count;i++) {
            if (loina_path(path,dir,p->files[i].name,NULL) && remove(path) && errno!=ENOENT) status=LOINA_IO;
        }
        if (loina_path(path,dir,"ready",NULL)) remove(path);
        if (rmdir(dir) && errno!=ENOENT) status=LOINA_IO;
    }
    atomic_store(&e->busy,false);
    return status ? loina_fail(error,status,"Cannot remove package %s",p->id) : LOINA_OK;
}

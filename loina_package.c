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

loina_status loina_packages_upgrade(loina_engine *e, loina_error *error) {
    if (e->config.model_directory) return LOINA_OK;
    static const struct { const char *id,*model,*projector; } retired[]={
        {"lfm2-5-vl-3b-q4_k_m-c815851c1e0a1361","LFM2.5-VL-3B-Q4_K_M.gguf",NULL},
        {"lfm2-5-vl-3b-q4_k_m-8f72ca4c841abd9c","LFM2.5-VL-3B-Q4_K_M.gguf","mmproj-BF16.gguf"},
        {"minicpm-v-4_7-1b-q8_0-b2ad48e4359d73b9","MiniCPM-V-4_7-1B-Q8_0.gguf",NULL},
        {"minicpm-v-4_7-1b-q8_0-0a441f54cc0b025b","MiniCPM-V-4_7-1B-Q8_0.gguf","mmproj-model-f16.gguf"},
        {"minicpm5-2b-q4_k_m-533e8c9b33ab3103","MiniCPM5-2B-Q4_K_M.gguf",NULL}
    };
    for (size_t i=0;i<sizeof(retired)/sizeof(*retired);i++) for (int partial=0;partial<2;partial++) {
        char name[256],dir[LOINA_PATH],path[LOINA_PATH];
        snprintf(name,sizeof(name),"%s%s",retired[i].id,partial ? ".partial" : "");
        if (!loina_path(dir,e->directory,"packages",name)) return loina_fail(error,LOINA_INVALID,"Retired package path is too long");
        const char *files[]={retired[i].model,retired[i].projector,"ready"};
        for (size_t j=0;j<3;j++) if (files[j]) for (int range=-1;range<4;range++) {
            if (range<0) snprintf(name,sizeof(name),"%s",files[j]);
            else snprintf(name,sizeof(name),"%s.range-%d",files[j],range);
            if (!loina_path(path,dir,name,NULL) || (remove(path) && errno!=ENOENT))
                return loina_fail(error,LOINA_IO,"Cannot remove retired weights: %s",name);
        }
        if (rmdir(dir) && errno!=ENOENT) return loina_fail(error,LOINA_IO,"Cannot remove retired package: %s",retired[i].id);
    }
    return LOINA_OK;
}

const loina_package *loina_catalog_find(const char *id) {
    if (!id) return NULL;
    for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *p=loina_catalog_get(i);
        if (!strcmp(p->id,id)) return p;
    }
    return NULL;
}
bool loina_package_valid(const loina_package *p) {
    if (p && p->chat_template && !loina_name_valid(p->chat_template)) return false;
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
    if (e && e->config.model_directory && loina_package_valid(p) && i<p->file_count)
        return loina_path(path,e->model_directory,p->files[i].name,NULL);
    char dir[LOINA_PATH];
    return e && loina_package_valid(p) && i<p->file_count &&
        loina_path(dir,e->directory,"packages",p->id) && loina_path(path,dir,p->files[i].name,NULL);
}
static bool flat_ready_path(loina_engine *e, const loina_package *p, char path[LOINA_PATH]) {
    char dir[LOINA_PATH],name[320];
    snprintf(name,sizeof(name),"%s.ready",p->id);
    return loina_path(dir,e->model_directory,".loina",NULL) && loina_path(path,dir,name,NULL);
}

bool loina_package_installed(loina_engine *e, const loina_package *p) {
    if (!e || !loina_package_valid(p)) return false;
    char dir[LOINA_PATH],path[LOINA_PATH],want[65],have[66]={0};
    if (e->config.model_directory) {
        if (!flat_ready_path(e,p,path)) return false;
    } else if (!loina_path(dir,e->directory,"packages",p->id) || !loina_path(path,dir,"ready",NULL)) return false;
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

static bool installed_artifact(loina_engine *e, const loina_artifact *a, char source[LOINA_PATH]) {
    for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *p=loina_catalog_get(i);
        for (size_t j=0;j<p->file_count;j++) {
            const loina_artifact *b=&p->files[j];
            if (a->bytes==b->bytes && !strcmp(a->sha256,b->sha256) && loina_package_installed(e,p))
                return loina_artifact_path(e,p,j,source);
        }
    }
    return false;
}
uint64_t loina_package_download_bytes(loina_engine *e, const loina_package *p) {
    if (!e || !loina_package_valid(p)) return UINT64_MAX;
    uint64_t total=0;
    char source[LOINA_PATH];
    for (size_t i=0;i<p->file_count;i++)
        if (!installed_artifact(e,&p->files[i],source)) total+=p->files[i].bytes;
    return total;
}

/* Terminal models live once in gguf/, including weights shared by chat and
 * vision. Publish the package marker only after every artifact is verified. */
static loina_status install_flat(loina_engine *e, const loina_package *p, const char *import,
                                  loina_job *job, loina_error *error) {
    if (loina_package_installed(e,p)) return LOINA_OK;
    char dir[LOINA_PATH],stage[LOINA_PATH],path[LOINA_PATH],source[LOINA_PATH],hash[65];
    if (!loina_path(dir,e->model_directory,".loina",NULL) || !loina_mkdir(dir))
        return loina_fail(error,LOINA_IO,"Cannot create model staging directory");
    if (!flat_ready_path(e,p,path)) return loina_fail(error,LOINA_INVALID,"Package path is too long");
    FILE *marker=fopen(path,"rb");
    if (marker) {
        char have[65];
        size_t n=fread(have,1,sizeof(have),marker);
        fclose(marker);
        loina_package_identity(p,hash);
        if (n!=64 || memcmp(have,hash,64))
            return loina_fail(error,LOINA_INTEGRITY,"Installed package identity conflicts: %s",p->id);
    }
    for (size_t i=0;i<p->file_count;i++) {
        const loina_artifact *file=&p->files[i];
        if (loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Installation cancelled");
        if (!loina_artifact_path(e,p,i,path) || !loina_path(stage,dir,file->name,NULL))
            return loina_fail(error,LOINA_INVALID,"Model path is too long");
        loina_status status=LOINA_OK;
        bool existing=loina_file_bytes(path)==file->bytes;
        if (!existing) {
            if (import) {
                if (!loina_path(source,import,file->name,NULL)) return loina_fail(error,LOINA_INVALID,"Import path is too long");
                status=loina_copy(source,stage,job,file->bytes,error);
            } else if (loina_file_bytes(stage)!=file->bytes) {
                if (!e->config.fetch) return loina_fail(error,LOINA_UNSUPPORTED,"Model requires a download bridge");
                status=e->config.fetch(e->config.user,file,stage,job,error);
            }
        }
        if (status) return status;
        const char *candidate=existing ? path : stage;
        if (loina_file_bytes(candidate)!=file->bytes)
            return loina_fail(error,LOINA_INTEGRITY,"Wrong artifact size: %s",file->name);
        status=loina_hash_file(candidate,hash,job,error);
        if (status) return status;
        if (strcmp(hash,file->sha256)) {
            if (!existing) remove(stage);
            return loina_fail(error,LOINA_INTEGRITY,"SHA-256 mismatch: %s",file->name);
        }
        if (!existing && !loina_replace(stage,path)) return loina_fail(error,LOINA_IO,"Cannot publish model: %s",file->name);
    }
    if (!flat_ready_path(e,p,path) || snprintf(stage,sizeof(stage),"%s.tmp",path)>=(int)sizeof(stage))
        return loina_fail(error,LOINA_INVALID,"Package marker path is too long");
    loina_package_identity(p,hash);
    FILE *f=fopen(stage,"wb");
    bool ok=f && fwrite(hash,1,64,f)==64;
    if (f && fclose(f)) ok=false;
    if (ok) ok=loina_replace(stage,path);
    if (!ok) { remove(stage);return loina_fail(error,LOINA_IO,"Cannot commit model package"); }
    return LOINA_OK;
}

loina_status loina_install(loina_engine *e, const loina_package *p, const char *import,
                          loina_job *job, loina_error *error) {
    if (!e || !loina_package_valid(p)) return loina_fail(error,LOINA_INVALID,"Invalid package manifest");
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    loina_status status=LOINA_OK;
    if (e->config.model_directory) {
        status=install_flat(e,p,import,job,error);
        atomic_store(&e->busy,false);
        return status;
    }
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
            /* Text and vision share immutable weights. Reuse the installed
             * artifact, then hash it again before publishing the new package. */
            if (installed_artifact(e,a,source)) status=loina_copy(source,path,job,a->bytes,error);
            else {
                if (!e->config.fetch || !a->url || !*a->url) { status=loina_fail(error,LOINA_UNSUPPORTED,"%s requires local import or a download bridge",a->name); goto done; }
                status=e->config.fetch(e->config.user,a,path,job,error);
            }
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
    char dir[LOINA_PATH],path[LOINA_PATH],source[LOINA_PATH];
    loina_status status=LOINA_OK;
    if (e->config.model_directory) {
        if (!flat_ready_path(e,p,path)) status=LOINA_INVALID;
        else if (remove(path) && errno!=ENOENT) status=LOINA_IO;
        for (size_t i=0;!status && i<p->file_count;i++) {
            if (!installed_artifact(e,&p->files[i],source) && loina_artifact_path(e,p,i,path) &&
                remove(path) && errno!=ENOENT) status=LOINA_IO;
        }
    } else if (!loina_path(dir,e->directory,"packages",p->id)) status=LOINA_INVALID;
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

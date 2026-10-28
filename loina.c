#include "loina_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const loina_runtime *runtimes[] = {
#ifdef LOINA_WITH_LLAMA
    &loina_llama,
#endif
#ifdef LOINA_WITH_DIFFUSION
    &loina_diffusion,
#endif
#ifdef LOINA_WITH_SPEECH
    &loina_speech,
#endif
#ifdef LOINA_TEST
    &loina_test_runtime,
#endif
    NULL
};
const loina_runtime *loina_runtime_find(const char *name) {
    if (!name) return NULL;
    for (size_t i=0;runtimes[i];i++) if (!strcmp(name,runtimes[i]->name)) return runtimes[i];
    return NULL;
}
unsigned loina_runtime_backends(void) {
    unsigned bits=0;
    for (size_t i=0;runtimes[i];i++) bits|=runtimes[i]->backends();
    return bits;
}
loina_engine *loina_open(const loina_config *c, loina_error *error) {
    if (!c || !c->directory || !*c->directory || strlen(c->directory)>LOINA_PATH-512 || c->context>LOINA_MAX_CONTEXT) {
        loina_fail(error,LOINA_INVALID,"Invalid storage directory or context (maximum 32768)"); return NULL;
    }
    loina_engine *e=calloc(1,sizeof(*e));
    if (!e) { loina_fail(error,LOINA_MEMORY,"Cannot allocate engine"); return NULL; }
    e->config=*c;
    strcpy(e->directory,c->directory);
    e->config.directory=e->directory;
    if (c->template_directory) {
        if (strlen(c->template_directory)>=sizeof(e->template_directory)) {
            free(e);loina_fail(error,LOINA_INVALID,"Template directory is too long");return NULL;
        }
        strcpy(e->template_directory,c->template_directory);
        e->config.template_directory=e->template_directory;
    }
    e->config.context=c->context ? c->context : LOINA_MAX_CONTEXT;
    if (e->config.context<512) { loina_fail(error,LOINA_INVALID,"Context must be at least 512 tokens"); free(e); return NULL; }
    atomic_init(&e->busy,false);
    if (!loina_mkdir(e->directory)) { loina_fail(error,LOINA_IO,"Cannot create storage directory"); free(e); return NULL; }
    const char *dirs[]={"packages","sessions","cache","outputs"};
    char path[LOINA_PATH];
    for (size_t i=0;i<sizeof(dirs)/sizeof(*dirs);i++) {
        if (!loina_path(path,e->directory,dirs[i],NULL) || !loina_mkdir(path)) {
            loina_fail(error,LOINA_IO,"Cannot create %s directory",dirs[i]); free(e); return NULL;
        }
    }
    if (loina_packages_upgrade(e,error)) { free(e);return NULL; }
    loina_device d;
    loina_device_probe(e->directory,&d,NULL);
    if (!e->config.device.ram_bytes) e->config.device.ram_bytes=d.ram_bytes;
    if (!e->config.device.available_ram_bytes) e->config.device.available_ram_bytes=d.available_ram_bytes;
    if (!e->config.device.storage_bytes) e->config.device.storage_bytes=d.storage_bytes;
    if (!e->config.device.threads) e->config.device.threads=d.threads;
    if (!e->config.device.backends) e->config.device.backends=d.backends;
    e->config.device.backends &= loina_runtime_backends();
    e->available_backends=e->config.device.backends;
    if (e->config.device.threads<1 || e->config.device.threads>64) {
        loina_fail(error,LOINA_INVALID,"Invalid CPU thread count"); free(e); return NULL;
    }
    if (error) *error=(loina_error){0};
    return e;
}
void loina_unload(loina_engine *e) {
    if (e->model) e->runtime->unload(e->model);
    e->model=NULL; e->runtime=NULL; e->package=NULL;
    e->cache_session[0]=e->cache_transcript[0]=0;
    e->chat_identity[0]=0;
}
void loina_close(loina_engine *e) {
    if (!e) return;
    loina_unload(e);
    free(e);
}
const loina_device *loina_get_device(const loina_engine *e) { return e ? &e->config.device : NULL; }
unsigned loina_available_backends(const loina_engine *e) { return e ? e->available_backends : 0; }
loina_status loina_set_backend(loina_engine *e, unsigned backend, loina_error *error) {
    if (!e || (backend && ((backend & (backend-1)) || !(backend & e->available_backends))))
        return loina_fail(error,LOINA_UNSUPPORTED,"Runtime is unavailable on this device");
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    if (backend!=e->selected_backend) {
        loina_unload(e);
        e->selected_backend=backend;
        e->config.device.backends=backend ? backend : e->available_backends;
    }
    atomic_store(&e->busy,false);
    return LOINA_OK;
}
loina_status loina_activate(loina_engine *e, const loina_package *p, loina_job *job, loina_error *error) {
    if (!loina_package_valid(p)) return loina_fail(error,LOINA_INVALID,"Invalid model package");
    const loina_runtime *r=loina_runtime_find(p->runtime);
    if (!r) return loina_fail(error,LOINA_UNSUPPORTED,"Runtime is not built: %s",p->runtime);
    if (r->abi!=p->runtime_abi) return loina_fail(error,LOINA_INCOMPATIBLE,"Package runtime ABI mismatch");
    if (!(r->backends() & p->backends & e->config.device.backends)) return loina_fail(error,LOINA_UNSUPPORTED,"No supported backend");
    if (!loina_package_installed(e,p)) return loina_fail(error,LOINA_NOT_INSTALLED,"Download or import %s first",p->name);
    if (e->package && !strcmp(e->package->id,p->id)) {
        char a[65],b[65];
        loina_package_identity(e->package,a);loina_package_identity(p,b);
        if (strcmp(a,b)) return loina_fail(error,LOINA_INCOMPATIBLE,"Loaded package identity changed");
        return LOINA_OK;
    }
    loina_unload(e);
    if (loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Loading cancelled");
    loina_emit(job,LOINA_LOADING,p->name,strlen(p->name),0);
    e->runtime=r;
    e->package=p;
    e->model=r->load(e,p,job,error);
    if (!e->model) {
        e->runtime=NULL;e->package=NULL;
        if (loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Loading cancelled");
        return error && error->code ? error->code : loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot load %s",p->name);
    }
    return LOINA_OK;
}
loina_status loina_preload(loina_engine *e, loina_feature feature, loina_job *job, loina_error *error) {
    if (!e) return loina_fail(error,LOINA_INVALID,"Missing engine");
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    const loina_package *p=loina_resolve(e,feature,error);
    loina_status status=p ? loina_activate(e,p,job,error) : (error ? error->code : LOINA_UNSUPPORTED);
    atomic_store(&e->busy,false);
    return status;
}
loina_request loina_request_default(loina_feature feature) {
    return (loina_request){.feature=feature,.max_tokens=512,.temperature=0.1f,
        .top_p=1.0f,.min_p=0.0f,.seed=UINT32_MAX,.width=1024,.height=1024,.steps=2,.thinking=true,.strength=0.75f};
}
loina_job *loina_job_create(loina_engine *e, const loina_request *r, loina_event_fn event, void *user, loina_error *error) {
    if (!e || !r || r->feature<LOINA_TEXT || r->feature>LOINA_SPEECH ||
        !isfinite(r->temperature) || r->temperature<0 || r->temperature>10 ||
        !isfinite(r->top_p) || r->top_p<=0 || r->top_p>1 ||
        !isfinite(r->min_p) || r->min_p<0 || r->min_p>1 ||
        !r->max_tokens || r->max_tokens>=e->config.context ||
        (r->session && r->session->engine!=e) ||
        ((r->feature==LOINA_TEXT || r->feature==LOINA_VISION || r->feature==LOINA_IMAGE) && (!r->prompt || !*r->prompt)) ||
        ((r->feature==LOINA_VISION || r->feature==LOINA_SPEECH) && (!r->input_path || !*r->input_path)) ||
        (r->feature==LOINA_IMAGE && (!r->output_path || r->width<64 || r->height<64 ||
            r->width>1024 || r->height>1024 || r->width%32 || r->height%32 || r->steps<1 || r->steps>4 ||
            !isfinite(r->strength) || r->strength<0 || r->strength>1))) {
        loina_fail(error,LOINA_INVALID,"Invalid inference request"); return NULL;
    }
    if (r->prompt && strlen(r->prompt)>LOINA_MAX_TEXT/2) { loina_fail(error,LOINA_INVALID,"Prompt is too large"); return NULL; }
    loina_job *j=calloc(1,sizeof(*j));
    if (!j) { loina_fail(error,LOINA_MEMORY,"Cannot allocate job"); return NULL; }
    j->engine=e;j->request=*r;j->event=event;j->user=user;
    j->prompt=loina_strdup(r->prompt);j->input=loina_strdup(r->input_path);j->output=loina_strdup(r->output_path);
    if (!j->prompt || !j->input || !j->output) { loina_job_free(j);loina_fail(error,LOINA_MEMORY,"Cannot copy request");return NULL; }
    j->request.prompt=j->prompt;j->request.input_path=j->input;j->request.output_path=j->output;
    atomic_init(&j->cancelled,false);
    return j;
}
void loina_job_cancel(loina_job *j) { if (j) atomic_store(&j->cancelled,true); }
bool loina_job_cancelled(const loina_job *j) { return j && (atomic_load(&j->cancelled) || loina_job_cancelled(j->parent)); }
void loina_job_progress(loina_job *j, uint64_t done, uint64_t total) {
    int progress=total ? (int)((long double)done*100/total) : 0;
    if (progress>100) progress=100;
    loina_emit(j,LOINA_PROGRESS,"",0,progress);
}
const char *loina_job_result(const loina_job *j) { return j && j->result ? j->result : ""; }
void loina_job_free(loina_job *j) {
    if (!j) return;
    free(j->prompt);free(j->input);free(j->output);free(j->result);free(j);
}
loina_status loina_job_run(loina_job *j, loina_error *error) {
    if (!j || j->ran) return loina_fail(error,LOINA_INVALID,"A job can run only once");
    loina_engine *e=j->engine;
    if (atomic_exchange(&e->busy,true)) return loina_fail(error,LOINA_BUSY,"Engine is busy");
    j->ran=true;
    loina_status status;
    if (error) *error=(loina_error){0};
    if (loina_job_cancelled(j)) { status=loina_fail(error,LOINA_CANCELLED,"Job cancelled");goto done; }
    if (j->request.feature==LOINA_TEXT || j->request.feature==LOINA_VISION) {
        status=loina_chat(j,error);goto done;
    }
    const loina_package *p=j->request.package ? j->request.package : loina_resolve(e,j->request.feature,error);
    if (!p) { status=error ? error->code : LOINA_UNSUPPORTED;goto done; }
    if (p->feature!=j->request.feature) { status=loina_fail(error,LOINA_INCOMPATIBLE,"Package does not implement the requested feature");goto done; }
    status=loina_activate(e,p,j,error);
    if (status) goto done;
    if (j->request.feature==LOINA_IMAGE && e->runtime->image) {
        status=e->runtime->image(e->model,j,error);
        if (!status) {
            j->result=loina_strdup(j->output);
            if (!j->result) status=loina_fail(error,LOINA_MEMORY,"Cannot retain image result");
            if (!status && j->request.session) {
                loina_session *s=j->request.session;
                size_t n=s->count;
                if (!loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,j->prompt) ||
                    !loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_IMAGE,j->output)) {
                    loina_session_truncate(s,n);status=loina_fail(error,LOINA_MEMORY,"Cannot retain image message");
                } else status=loina_session_save(s,error);
            }
        }
    } else if (j->request.feature==LOINA_SPEECH && e->runtime->speech) status=e->runtime->speech(e->model,j,&j->result,error);
    else status=loina_fail(error,LOINA_UNSUPPORTED,"Runtime lacks the requested operation");
done:
    if (loina_job_cancelled(j) && !status) status=loina_fail(error,LOINA_CANCELLED,"Job cancelled");
    /* Chat rolls back the transcript and resets inference state on cancellation.
     * Keep its weights and GPU kernels resident for the next request. */
    bool chat=j->request.feature==LOINA_TEXT || j->request.feature==LOINA_VISION;
    if (status==LOINA_RUNTIME_ERROR || (status==LOINA_CANCELLED &&
        (!chat || !e->runtime || !e->runtime->reset))) loina_unload(e);
    if (!status) loina_emit(j,LOINA_FINISHED,loina_job_result(j),strlen(loina_job_result(j)),100);
    atomic_store(&e->busy,false);
    return status;
}

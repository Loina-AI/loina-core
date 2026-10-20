/* Replay identical evidence through the production chat loop. No network or
 * judge is involved; retain answers, tool requests and metrics for review. */
#include "loina_internal.h"
#include "ggml-backend.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *directory;
    unsigned searches, visits;
    loina_buffer calls, metrics;
} evaluation;

static char *read_text(const char *path) {
    FILE *f=fopen(path,"rb");
    if (!f) return NULL;
    loina_buffer b={0};char data[4096];size_t n;
    while ((n=fread(data,1,sizeof(data),f))) {
        if (!loina_buffer_add(&b,data,n)) { free(b.data);fclose(f);return NULL; }
    }
    bool ok=!ferror(f);fclose(f);
    if (!ok) { free(b.data);return NULL; }
    return b.data ? b.data : loina_strdup("");
}
static loina_status web(void *user, const char *url, char **text, loina_job *job, loina_error *error) {
    (void)job;(void)error;
    evaluation *v=user;char path[LOINA_PATH],name[64];
    bool search=!strncmp(url,"https://html.duckduckgo.com/html/?q=",36);
    unsigned n=search ? ++v->searches : ++v->visits;
    snprintf(name,sizeof(name),"%s-%u.txt",search ? "search" : "page",n);
    if (!loina_path(path,v->directory,name,NULL)) return LOINA_INVALID;
    *text=read_text(path);
    if (!*text) {
        if (!loina_path(path,v->directory,search ? "search.txt" : "page.txt",NULL)) return LOINA_INVALID;
        *text=read_text(path);
    }
    if (!*text) *text=loina_strdup("Tool error: No evidence is available for this request.");
    if (v->calls.length) loina_buffer_puts(&v->calls,",");
    loina_buffer_puts(&v->calls,"{\"url\":");loina_buffer_json(&v->calls,url);
    loina_buffer_puts(&v->calls,",\"result\":");loina_buffer_json(&v->calls,*text ? *text : "");
    loina_buffer_puts(&v->calls,"}");
    return *text ? LOINA_OK : LOINA_MEMORY;
}
static void event(void *user, loina_event type, const char *text, size_t bytes, int progress) {
    (void)progress;evaluation *v=user;
    if (type==LOINA_METRICS) {
        fprintf(stderr,"%.*s\n",(int)bytes,text);
        if (v->metrics.length) loina_buffer_puts(&v->metrics,",");
        loina_buffer_add(&v->metrics,text,bytes);
    }
}
int main(int argc, char **argv) {
    if (argc!=6 && argc!=7) {
        fprintf(stderr,"Usage: loina-quality STORAGE LIBRARIES TEMPLATES MODEL_ID PLAN.tsv [CPU|Vulkan|OpenCL]\n");return 2;
    }
    const char *backend_name=argc==7 ? argv[6] : "CPU";
    unsigned backend=!strcmp(backend_name,"CPU") ? LOINA_CPU : !strcmp(backend_name,"Vulkan") ? LOINA_VULKAN : !strcmp(backend_name,"OpenCL") ? LOINA_OPENCL : 0;
    if (!backend) { fprintf(stderr,"Unknown backend: %s\n",backend_name);return 2; }
    setbuf(stdout,NULL);
    ggml_backend_load_all_from_path(argv[2]);
    evaluation v={0};loina_error error={0};
    loina_config config={.directory=argv[1],.template_directory=argv[3],.web=web,.user=&v,
        .device={.threads=4,.backends=backend}};
    loina_engine *e=loina_open(&config,&error);
    if (!e) { fprintf(stderr,"%s\n",error.message);return 1; }
    /* Pin the artifact across prompt variants. Installation still hashes and
     * commits the package; every supported chat model uses this same loop. */
    const loina_package *p=loina_catalog_find(argv[4]);
    if (!p || p->feature!=LOINA_TEXT) { fprintf(stderr,"Unknown text package: %s\n",argv[4]);loina_close(e);return 2; }
    if (loina_set_backend(e,backend,&error) || loina_install(e,p,NULL,NULL,&error)) {
        fprintf(stderr,"%s\n",error.message);loina_close(e);return 1;
    }
    FILE *plan=fopen(argv[5],"r");
    if (!plan) { perror(argv[5]);loina_close(e);return 1; }
    char *line=NULL;size_t capacity=0;int failed=0;
    while (getline(&line,&capacity,plan)>0) {
        line[strcspn(line,"\r\n")]=0;
        char *save=NULL,*id=strtok_r(line,"\t",&save),*tools=strtok_r(NULL,"\t",&save);
        char *limit=strtok_r(NULL,"\t",&save),*directory=strtok_r(NULL,"\t",&save);
        if (!id || !tools || !limit || !directory || !loina_name_valid(id)) { failed=1;break; }
        loina_session *s=loina_session_create(e,id,&error);
        if (!s) { failed=1;break; }
        for (unsigned turn=1;;turn++) {
            char path[LOINA_PATH],name[64];snprintf(name,sizeof(name),"prompt-%u.txt",turn);
            if (!loina_path(path,directory,name,NULL)) { failed=1;break; }
            char *prompt=read_text(path);
            if (!prompt) {
                if (turn==1) { fprintf(stderr,"Missing prompt: %s\n",path);failed=1; }
                break;
            }
            fprintf(stderr,"%s turn %u (%s)\n",id,turn,backend_name);
            v=(evaluation){.directory=directory};
            loina_request r=loina_request_default(LOINA_TEXT);
            r.package=p;r.session=s;r.prompt=prompt;r.tools=atoi(tools)!=0;
            r.thinking=false;r.temperature=0;r.seed=42;r.max_tokens=(uint32_t)atoi(limit);
            struct timespec start,end;clock_gettime(CLOCK_MONOTONIC,&start);
            loina_job *j=loina_job_create(e,&r,event,&v,&error);
            loina_status status=j ? loina_job_run(j,&error) : error.code;
            clock_gettime(CLOCK_MONOTONIC,&end);
            loina_buffer out={0};char numbers[256];
            loina_buffer_puts(&out,"{\"id\":");loina_buffer_json(&out,id);
            snprintf(numbers,sizeof(numbers),",\"turn\":%u,\"status\":%d,\"seconds\":%.3f,\"model\":",turn,status,
                (double)(end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)/1e9);
            loina_buffer_puts(&out,numbers);loina_buffer_json(&out,p->id);
            loina_buffer_puts(&out,",\"prompt\":");loina_buffer_json(&out,prompt);
            loina_buffer_puts(&out,",\"answer\":");loina_buffer_json(&out,j ? loina_job_result(j) : "");
            loina_buffer_puts(&out,",\"error\":");loina_buffer_json(&out,status ? error.message : "");
            loina_buffer_puts(&out,",\"calls\":[");loina_buffer_puts(&out,v.calls.data ? v.calls.data : "");
            loina_buffer_puts(&out,"],\"metrics\":[");loina_buffer_puts(&out,v.metrics.data ? v.metrics.data : "");
            loina_buffer_puts(&out,"]}");puts(out.data);
            free(out.data);free(v.calls.data);free(v.metrics.data);free(prompt);loina_job_free(j);
            if (status) { failed=1;break; }
        }
        loina_session_free(s);
    }
    free(line);fclose(plan);loina_close(e);return failed;
}

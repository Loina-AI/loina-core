#include "loina.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t interrupted;
static void on_signal(int signal) { (void)signal;interrupted=1; }
typedef struct { loina_job *job; } cli_stream;
static void event(void *user, loina_event type, const char *text, size_t bytes, int progress) {
    (void)progress;
    cli_stream *s=user;
    if (interrupted) loina_job_cancel(s->job);
    if (type==LOINA_TOKEN) { fwrite(text,1,bytes,stdout);fflush(stdout); }
    if (type==LOINA_LOADING) fprintf(stderr,"Loading %.*s\n",(int)bytes,text);
    if (type==LOINA_COMPACTED) fprintf(stderr,"Earlier turns remain in the saved transcript.\n");
}
static void help(void) {
    puts("Loina - local inference\n"
         "Usage: loina [options]\n"
         "  -p TEXT               Answer once; otherwise start a conversation\n"
         "  -m FILE               Use a catalog model file (import siblings if needed)\n"
         "  --package ID          Use an immutable package from --models\n"
         "  --import DIRECTORY    Install selected package from local artifacts\n"
         "  --models              List packages, licenses and installation state\n"
         "  --device              Show core device selection capabilities\n"
         "  --ctx N               Context size, maximum 32768\n"
         "  -n N                  Maximum generated tokens (default 512)\n"
         "  --threads N           CPU threads\n"
         "  --backend NAME        CPU, Vulkan, OpenCL, Metal or CUDA\n"
         "  --temp F              Temperature; zero selects greedy output\n"
         "  --top-p F --min-p F    Sampling filters\n"
         "  --seed N              Sampling seed\n"
         "  --session ID          Resume a saved conversation\n"
         "  --cache-dir PATH      Packages, transcripts and KV (default .loina)\n"
         "  --vision FILE         Read an image through the vision layer\n"
         "  --generate --out FILE Generate a Sana Sprint PNG (default 1024x1024, two steps)\n"
         "  --width N --height N  Image dimensions, 64-1024 in multiples of 32\n"
         "  --steps N             Sana Sprint sampling steps, 1-4\n"
         "  --img2img FILE        Generate from an image and prompt\n"
         "  --strength F          Img2img change strength, 0-1 (default 0.75)\n"
         "  --transcribe FILE     Transcribe PCM16 mono 16kHz WAV\n"
         "REPL: /help /save /list /switch ID /del ID /read FILE /new /quit\n"
         "Browser tools are provided by the embedding host's native bridge.");
}
static bool integer(const char *s, uint32_t *n) {
    if (!s || !*s || *s=='-') return false;
    char *end;
    errno=0;
    unsigned long value=strtoul(s,&end,10);
    if (errno || *end || value>UINT32_MAX) return false;
    *n=(uint32_t)value;return true;
}
static bool real(const char *s, float *n) {
    char *end;
    errno=0;*n=strtof(s,&end);
    return !errno && end!=s && !*end;
}
static void list_session(void *user, const char *id) { (void)user;puts(id); }
static loina_session *new_session(loina_engine *engine, loina_error *error) {
    static unsigned sequence;
    char id[64];
    snprintf(id,sizeof(id),"chat-%llu-%u",(unsigned long long)time(NULL),++sequence);
    return loina_session_create(engine,id,error);
}
static int run(loina_engine *engine, loina_request *request, loina_error *error) {
    interrupted=0;
    cli_stream stream={0};
    loina_job *job=loina_job_create(engine,request,event,&stream,error);
    if (!job) return 1;
    stream.job=job;
    loina_status status=loina_job_run(job,error);
    if (!status && request->feature==LOINA_IMAGE) puts(loina_job_result(job));
    else putchar('\n');
    loina_job_free(job);
    return status ? 1 : 0;
}
int main(int argc, char **argv) {
    loina_config config={.directory=".loina"};
    loina_request request=loina_request_default(LOINA_TEXT);
    const char *package_id=NULL,*model_file=NULL,*import=NULL,*session_id=NULL,*prompt=NULL;
    bool models=false,device=false;
    unsigned backend=0;
    for (int i=1;i<argc;i++) {
        const char *a=argv[i];
        if (!strcmp(a,"--help") || !strcmp(a,"-h")) { help();return 0; }
        if (!strcmp(a,"--models")) { models=true;continue; }
        if (!strcmp(a,"--device")) { device=true;continue; }
        if (!strcmp(a,"--generate")) { request.feature=LOINA_IMAGE;continue; }
        if (i+1>=argc) { fprintf(stderr,"Missing value: %s\n",a);return 1; }
        const char *v=argv[++i];
        bool ok=true;
        uint32_t n;
        if (!strcmp(a,"-p")) prompt=v;
        else if (!strcmp(a,"--package")) package_id=v;
        else if (!strcmp(a,"-m")) model_file=v;
        else if (!strcmp(a,"--import")) import=v;
        else if (!strcmp(a,"--session")) session_id=v;
        else if (!strcmp(a,"--cache-dir")) config.directory=v;
        else if (!strcmp(a,"--ctx")) { ok=integer(v,&n);if (ok) config.context=n; }
        else if (!strcmp(a,"--threads")) { ok=integer(v,&n) && n<=64 && n>0;if (ok) config.device.threads=(int)n; }
        else if (!strcmp(a,"--backend")) {
            backend=!strcmp(v,"CPU") ? LOINA_CPU : !strcmp(v,"Vulkan") ? LOINA_VULKAN :
                !strcmp(v,"OpenCL") ? LOINA_OPENCL : !strcmp(v,"Metal") ? LOINA_METAL :
                !strcmp(v,"CUDA") ? LOINA_CUDA : 0;
            ok=backend!=0;
        }
        else if (!strcmp(a,"-n")) ok=integer(v,&request.max_tokens);
        else if (!strcmp(a,"--seed")) ok=integer(v,&request.seed);
        else if (!strcmp(a,"--temp")) ok=real(v,&request.temperature);
        else if (!strcmp(a,"--top-p")) ok=real(v,&request.top_p);
        else if (!strcmp(a,"--min-p")) ok=real(v,&request.min_p);
        else if (!strcmp(a,"--width")) { ok=integer(v,&n) && n<=1024;if (ok) request.width=(int)n; }
        else if (!strcmp(a,"--height")) { ok=integer(v,&n) && n<=1024;if (ok) request.height=(int)n; }
        else if (!strcmp(a,"--steps")) { ok=integer(v,&n) && n<=4;if (ok) request.steps=(int)n; }
        else if (!strcmp(a,"--vision")) { request.feature=LOINA_VISION;request.input_path=v; }
        else if (!strcmp(a,"--img2img")) { request.feature=LOINA_IMAGE;request.input_path=v; }
        else if (!strcmp(a,"--strength")) ok=real(v,&request.strength);
        else if (!strcmp(a,"--transcribe")) { request.feature=LOINA_SPEECH;request.input_path=v; }
        else if (!strcmp(a,"--out")) request.output_path=v;
        else { fprintf(stderr,"Unknown option: %s\n",a);return 1; }
        if (!ok) { fprintf(stderr,"Invalid value for %s: %s\n",a,v);return 1; }
    }
    signal(SIGINT,on_signal);
    loina_error error={0};
    loina_engine *engine=loina_open(&config,&error);
    if (!engine) { fprintf(stderr,"%s\n",error.message);return 1; }
    if (backend && loina_set_backend(engine,backend,&error)) {
        fprintf(stderr,"%s\n",error.message);loina_close(engine);return 1;
    }
    int exitcode=0;
    if (package_id) {
        request.package=loina_catalog_find(package_id);
        if (!request.package) { fprintf(stderr,"Unknown package; see --models\n");exitcode=1;goto done; }
    }
    char model_directory[4096];
    if (model_file) {
        const char *base=model_file;
        for (const char *c=model_file;*c;c++) if (*c=='/' || *c=='\\') base=c+1;
        const loina_package *found=NULL;
        for (size_t i=0;i<loina_catalog_count();i++) {
            const loina_package *p=loina_catalog_get(i);
            if (p->feature==request.feature && !strcmp(p->files[0].name,base)) { found=p;break; }
        }
        if (!found || (request.package && request.package!=found)) {
            fprintf(stderr,"Unknown or conflicting model file; see --models\n");exitcode=1;goto done;
        }
        request.package=found;
        size_t length=(size_t)(base-model_file);
        if (length>=sizeof(model_directory)) { fprintf(stderr,"Model path too long\n");exitcode=1;goto done; }
        if (length) { memcpy(model_directory,model_file,length);model_directory[length]=0; }
        else strcpy(model_directory,".");
        if (!loina_package_installed(engine,found)) import=model_directory;
    }
    if (import) {
        const loina_package *p=request.package ? request.package : loina_resolve(engine,request.feature,&error);
        if (!p || loina_install(engine,p,import,NULL,&error)) { fprintf(stderr,"%s\n",error.message);exitcode=1;goto done; }
        if (!prompt && !models && !device && !model_file && request.feature!=LOINA_SPEECH) goto done;
    }
    if (device) {
        const loina_device *d=loina_get_device(engine);
        printf("RAM: %llu; available: %llu; free storage: %llu; threads: %d; backends: 0x%x\n",
            (unsigned long long)d->ram_bytes,(unsigned long long)d->available_ram_bytes,(unsigned long long)d->storage_bytes,d->threads,d->backends);
    }
    if (models) for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *p=loina_catalog_get(i);
        printf("%s\n  %s | %s | %llu bytes | %s\n",p->id,p->name,p->license,
            (unsigned long long)loina_package_bytes(p),loina_package_installed(engine,p) ? "installed" : "not installed");
    }
    if (models || device) goto done;
    if (request.feature==LOINA_SPEECH) { exitcode=run(engine,&request,&error);if (exitcode) fprintf(stderr,"%s\n",error.message);goto done; }
    request.session=session_id ? loina_session_load(engine,session_id,&error) : new_session(engine,&error);
    if (!request.session) { fprintf(stderr,"%s\n",error.message);exitcode=1;goto done; }
    if (request.feature==LOINA_IMAGE && !request.output_path) request.output_path="loina.png";
    if (prompt) {
        request.prompt=prompt;exitcode=run(engine,&request,&error);
        if (exitcode) fprintf(stderr,"%s\n",error.message);
    } else {
        char line[16384];
        puts("Loina. /help for commands.");
        while (fputs("you> ",stdout),fflush(stdout),fgets(line,sizeof(line),stdin)) {
            line[strcspn(line,"\r\n")]=0;
            if (!*line) continue;
            if (!strcmp(line,"/quit")) break;
            if (!strcmp(line,"/help")) { help();continue; }
            if (!strcmp(line,"/save")) { if (loina_session_save(request.session,&error)) fprintf(stderr,"%s\n",error.message);else puts(loina_session_id(request.session));continue; }
            if (!strcmp(line,"/list")) { loina_session_list(engine,list_session,NULL,&error);continue; }
            if (!strncmp(line,"/del ",5)) { if (loina_session_delete(engine,line+5,&error)) fprintf(stderr,"%s\n",error.message);continue; }
            if (!strcmp(line,"/new") || !strncmp(line,"/switch ",8)) {
                loina_session *next=!strcmp(line,"/new") ? new_session(engine,&error) : loina_session_load(engine,line+8,&error);
                if (next) { loina_session_free(request.session);request.session=next; }
                else fprintf(stderr,"%s\n",error.message);
                continue;
            }
            if (!strncmp(line,"/read ",6)) { request.feature=LOINA_VISION;request.input_path=line+6;request.prompt="Describe this image and extract its text."; }
            else request.prompt=line;
            if (run(engine,&request,&error)) fprintf(stderr,"%s\n",error.message);
            request.feature=LOINA_TEXT;request.input_path=NULL;
        }
    }
    loina_session_free(request.session);
done:
    loina_close(engine);
    return exitcode;
}

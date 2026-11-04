#include "loina_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define check(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while (0)
static int loaded,unloaded,restored,reset_count,web_calls,tool_chat,tool_step;
static bool thinking_live, migrating;
static bool thinking_requested, fail_save;
static unsigned backends(void) { return LOINA_CPU|LOINA_VULKAN|LOINA_OPENCL; }
static void *load(loina_engine *e, const loina_package *p, loina_job *j, loina_error *error) {
    (void)e;(void)p;(void)j;(void)error;loaded++;return malloc(1);
}
static void unload(void *p) { unloaded++;free(p); }
static loina_status chat(void *p, loina_session *s, loina_job *j, char **out, loina_error *error) {
    (void)p;(void)error;
    thinking_requested=j->request.thinking;
    if (j->request.tool_schema) {
        check(strstr(j->request.tool_schema, "write"));
        check(s->messages[s->count-1].role==(j->request.tool_response ? LOINA_TOOL_RESULT : LOINA_USER));
        *out=loina_strdup(j->request.tool_response ? "Created." :
            "<tool_call><function=write><parameter=path>probe.py</parameter><parameter=content>print(1)</parameter></function></tool_call>");
        check(*out);
        loina_emit(j,LOINA_TOKEN,*out,strlen(*out),0);
        return LOINA_OK;
    }
    if (migrating) {
        check(s->first==0 && s->count==3);
        check(!strcmp(s->messages[0].text,"Remember the codeword: turquoise."));
        check(!strcmp(s->messages[1].text,"I will remember turquoise."));
    }
    if (tool_chat) {
        const loina_message *last=&s->messages[s->count-1];
        static const char *search="<function name=\"web_search\"><param name=\"query\">test</param></function>";
        static const char *visit="<function name=\"web_visit\"><param name=\"url\">https://example.com/?a=1&amp;b=2</param></function>";
        int step=tool_step++;
        *out=loina_strdup(tool_chat==26 && step<4 ? search :
            !j->request.tools ? "Answer from the available sources." :
            tool_chat==6 ? (step==0 ? search : step==1 ? visit : "Answer using the first result despite the failed page.") :
            tool_chat==7 ? (step<2 ? search : "Answer using the second search result.") :
            tool_chat==8 ? "<think>First. Second. Third. Fourth.</think>\nAnswer." :
            tool_chat==9 ? "<think>Reasoning that reached the token limit" :
            tool_chat==15 ? "The attribute is name=\"example\"." :
            tool_chat==16 && step==0 ? "<|tool_call_start|>[web_search(query='test')]<|tool_call_end|>Searching." :
            tool_chat==17 && step==0 ? "Checking sources.\n<|tool_call_start|>[web_search(query='test'), web_visit(url='https://example.com/?a=1&b=2')]<|tool_call_end|>" :
            tool_chat==18 && step==0 ? "<|tool_call_start|>[web_search(query='test'), unknown(query='test')]<|tool_call_end|>" :
            tool_chat==19 && step==0 ? "<|tool_call_start|>[web_search(query='test', extra='no')]<|tool_call_end|>" :
            tool_chat==20 && step==0 ? "<|tool_call_start|>[web_search(query='test'), web_visit(url='https://example.com'" :
            tool_chat==21 && step==0 ? "<|tool_call_start|>[web_search(query='a b&c\\n\"quoted\"')]<|tool_call_end|>" :
            tool_chat==22 && step==0 ? "<|tool_call_start|>[web_search(query='test'), web_search(query='test'), web_search(query='test'), web_search(query='test'), web_search(query='test')]<|tool_call_end|>" :
            tool_chat==23 && step==0 ? "<|tool_call_start|>[web_search(query='')]<|tool_call_end|>" :
            tool_chat==24 && step==0 ? "<|tool_call_start|>[web_search(query=__import__('os'))]<|tool_call_end|>" :
            tool_chat==25 && step==0 ? "<|tool_call_start|>[web_search(query='test')]<|tool_call_end|><|tool_call_start|>[unknown()]<|tool_call_end|>" :
            tool_chat==10 && step==0 ? "<think>Find current facts.</think>\nI will check both sources.\n<tool_call>\n<function=web_search>\n<parameter=query>\ntest\n</parameter>\n</function>\n</tool_call>\n<tool_call><function=web_visit><parameter=url>https://example.com/?a=1&b=2</parameter></function></tool_call>" :
            tool_chat==11 && step==0 ? "<tool_call><function=web_search><parameter=wrong>test</parameter></function></tool_call>" :
            tool_chat==12 && step==0 ? "<tool_call><function=web_search><parameter=query>test</parameter></function></tool_call><tool_call><function=web_search><parameter=query>broken</parameter>" :
            tool_chat==13 && step==0 ? "<function name=\"web_search\"><param name=\"query\">test</param></function><function name=\"unknown\"><param name=\"query\">test</param></function>" :
            tool_chat==14 && step==0 ? "<function name=\"web_search\"><param name=\"query\"><![CDATA[a b&c\n\"quoted\"]]></param></function>" :
            tool_chat==3 && last->role!=LOINA_TOOL_RESULT ? "<function name=\"web_search\"><param name=\"query\">test" :
            (tool_chat==4 || tool_chat==5) ? search :
            last->role==LOINA_TOOL_RESULT ? "The page says so." : tool_chat==2 ? visit :
            "<tool_call><function=web_search><parameter=query>a b&c</parameter></function></tool_call>");
        if (*out) for (size_t i=0;(*out)[i];i++) loina_emit(j,LOINA_TOKEN,*out+i,1,0);
        return *out ? LOINA_OK : LOINA_MEMORY;
    }
    loina_emit(j,LOINA_TOKEN,"fixture",7,0);
    if (loina_job_cancelled(j)) return LOINA_CANCELLED;
    *out=loina_strdup("fixture");return *out ? LOINA_OK : LOINA_MEMORY;
}
static bool save(void *p, const char *path) { (void)p;if (fail_save) return false;FILE *f=fopen(path,"wb");if (!f) return false;bool ok=fwrite("fixture",1,7,f)==7;return fclose(f)==0 && ok; }
static bool restore(void *p, const char *path) { (void)p;(void)path;restored++;return true; }
static void reset(void *p) { (void)p;reset_count++; }
const loina_runtime loina_test_runtime={.name="llama",.revision="test-runtime",.abi=1,.backends=backends,
    .load=load,.unload=unload,.chat=chat,.save=save,.restore=restore,.reset=reset};
static void capture_event(void *user, loina_event type, const char *text, size_t n, int progress) {
    (void)progress;
    if (type==LOINA_TOKEN) {
        loina_buffer *b=user;
        check(loina_buffer_add(b,text,n));
        if (strstr(b->data,"Third.") && !strstr(b->data,"</think>")) thinking_live=true;
    }
    if (type==LOINA_TOOL) { loina_buffer *b=user;free(b->data);*b=(loina_buffer){0}; }
}
static void write_file(const char *path, const void *data, size_t n) { FILE *f=fopen(path,"wb");check(f);check(fwrite(data,1,n,f)==n);check(!fclose(f)); }
static void cancel_event(void *user, loina_event type, const char *text, size_t n, int progress) {
    (void)text;(void)n;(void)progress;
    if (type==LOINA_TOKEN) loina_job_cancel(*(loina_job **)user);
}
static void count_session(void *user, const char *id) { check(loina_name_valid(id));(*(int *)user)++; }
static loina_status web(void *user, const char *url, char **text, loina_job *job, loina_error *error) {
    (void)user;(void)job;(void)error;
    bool visit=strstr(url,"https://example.com/?a=1&b=2")!=NULL;
    check(((tool_chat==2 || tool_chat==6 || tool_chat==10 || tool_chat==17) && visit) ||
        strstr(url,tool_chat==14 || tool_chat==21 ? "html.duckduckgo.com/html/?q=a%20b%26c%0A%22quoted%22" : tool_chat>=4 ? "html.duckduckgo.com/html/?q=test" : "html.duckduckgo.com/html/?q=a%20b%26c"));
    web_calls++;
    *text=loina_strdup(tool_chat==4 || (tool_chat==6 && visit) ? "Tool error: Web page timed out" : "page");return LOINA_OK;
}
int main(int argc, char **argv) {
    check(argc==2);
    char hash[65];
    loina_sha sha;
    loina_sha_init(&sha);loina_sha_update(&sha,"abc",3);loina_sha_final(&sha,hash);
    check(!strcmp(hash,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    loina_sha_init(&sha);
    for (int i=0;i<1000000;i++) loina_sha_update(&sha,"a",1);
    loina_sha_final(&sha,hash);
    check(!strcmp(hash,"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    check(!loina_name_valid("../escape"));check(!loina_name_valid("a\\b"));check(loina_name_valid("chat-123"));
    loina_config config={.directory=argv[1],.context=1024,.web=web,
        .device={.ram_bytes=16000000000,.available_ram_bytes=10000000000,.storage_bytes=20000000000,.threads=2,.backends=LOINA_CPU}};
    loina_error error={0};
    loina_engine *e=loina_open(&config,&error);check(e);
    check(loina_available_backends(e)==LOINA_CPU);
    check(loina_set_backend(e,LOINA_CUDA,&error)==LOINA_UNSUPPORTED);
    check(loina_set_backend(e,LOINA_CPU|LOINA_VULKAN,&error)==LOINA_UNSUPPORTED);
    atomic_store(&e->busy,true);
    check(loina_set_backend(e,LOINA_CPU,&error)==LOINA_BUSY);
    atomic_store(&e->busy,false);
    check(loina_set_backend(e,LOINA_CPU,&error)==LOINA_OK);
    check(loina_set_backend(e,0,&error)==LOINA_OK);
    int text_packages=0;
    for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *model=loina_catalog_get(i);
        if (model->feature==LOINA_TEXT) { text_packages++;check(model->chat_template); }
    }
    check(text_packages==1);
    check(!strcmp(loina_resolve(e,LOINA_TEXT,&error)->chat_template,"lfm3-vl-2b.jinja"));
    check(loina_resolve(e,LOINA_TEXT,&error)->thinking);
    loina_request image=loina_request_default(LOINA_IMAGE);
    check(image.width==1024 && image.height==1024 && image.steps==2);
    image.prompt="A red fox";image.output_path="fox.png";
    loina_job *image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    image.width=96;image.height=64;image.steps=1;
    image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    image.steps=4;
    image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    image.steps=5;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.steps=0;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.steps=2;image.width=95;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.width=96;image.input_path="input.png";
    image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    check(image.strength==0.75f);
    image.strength=-0.1f;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.strength=1.1f;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.strength=NAN;check(!loina_job_create(e,&image,NULL,NULL,&error));
    image.strength=0;
    image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    image.strength=1;
    image_job=loina_job_create(e,&image,NULL,NULL,&error);check(image_job);loina_job_free(image_job);
    const loina_package *selected=loina_resolve(e,LOINA_TEXT,&error);check(selected && selected->quality==12);
    e->config.device.backends|=LOINA_VULKAN;
    selected=loina_resolve(e,LOINA_TEXT,&error);check(selected && selected->quality==12);
    e->config.device.backends=LOINA_OPENCL;
    check(loina_resolve(e,LOINA_TEXT,&error)==selected);
    e->config.device.backends=LOINA_CPU|LOINA_VULKAN;
    e->config.device.available_ram_bytes=1000;
    selected=loina_resolve(e,LOINA_TEXT,&error);check(selected && selected->quality==12);
    e->config.device.ram_bytes=1000;check(loina_resolve(e,LOINA_TEXT,&error)==selected);
    e->config.device.ram_bytes=16000000000;
    e->config.device.storage_bytes=1;check(loina_resolve(e,LOINA_TEXT,&error));
    e->config.device.storage_bytes=20000000000;
    e->config.device.available_ram_bytes=10000000000;
    loina_package p={.id="test-model",.name="Test",.license="MIT",.runtime="llama",.runtime_abi=1,
        .feature=LOINA_TEXT,.thinking=true,.quality=4,.backends=LOINA_CPU,.memory_bytes=100,.file_count=1,
        .files={{.name="model.gguf",.sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",.bytes=3}}};
    char import[LOINA_PATH],path[LOINA_PATH];
    check(loina_path(import,argv[1],"import",NULL));check(loina_mkdir(import));check(loina_path(path,import,"model.gguf",NULL));
    loina_uninstall(e,&p,NULL);
    write_file(path,"bad",3);check(loina_install(e,&p,import,NULL,&error)==LOINA_INTEGRITY);check(!loina_package_installed(e,&p));
    write_file(path,"abc",3);check(loina_install(e,&p,import,NULL,&error)==LOINA_OK);check(loina_package_installed(e,&p));
    loina_package incompatible=p;incompatible.runtime_abi=2;check(!loina_package_installed(e,&incompatible));
    check(loina_install(e,&incompatible,import,NULL,&error)==LOINA_INTEGRITY);
    char flat_models[LOINA_PATH],flat_cache[LOINA_PATH],flat_root[LOINA_PATH];
    check(loina_path(flat_models,argv[1],"gguf",NULL));
    check(loina_path(flat_cache,argv[1],"kvcache",NULL));
    check(loina_path(flat_root,argv[1],"terminal",NULL));
    loina_config flat_config=config;
    flat_config.directory=flat_root;flat_config.model_directory=flat_models;
    flat_config.session_directory=flat_cache;flat_config.cache_directory=flat_cache;
    int restores_before_flat=restored;
    loina_engine *flat=loina_open(&flat_config,&error);check(flat);
    check(!loina_uninstall(flat,&p,&error));
    check(loina_path(path,flat_models,"model.gguf",NULL));
    write_file(path,"bad",3);
    check(loina_install(flat,&p,import,NULL,&error)==LOINA_INTEGRITY && !loina_package_installed(flat,&p));
    write_file(path,"abc",3);
    struct stat before_stat,after_stat;
    check(!stat(path,&before_stat));
    check(!loina_install(flat,&p,flat_models,NULL,&error) && loina_package_installed(flat,&p));
    check(!stat(path,&after_stat) && before_stat.st_ino==after_stat.st_ino);
    check(loina_install(flat,&incompatible,import,NULL,&error)==LOINA_INTEGRITY);
    char model_path[LOINA_PATH];check(loina_artifact_path(flat,&p,0,model_path) && !strcmp(path,model_path));
    loina_session *flat_session=loina_session_create(flat,"flat-session",&error);check(flat_session);
    loina_request flat_request=loina_request_default(LOINA_TEXT);
    flat_request.session=flat_session;flat_request.package=&p;flat_request.prompt="Terminal storage fixture";
    loina_job *flat_job=loina_job_create(flat,&flat_request,NULL,NULL,&error);check(flat_job);
    check(!loina_job_run(flat_job,&error));loina_job_free(flat_job);
    check(loina_path(path,flat_cache,"flat-session.chat",NULL) && loina_file_bytes(path)>0);
    loina_unload(flat);check(!loina_activate(flat,&p,NULL,&error));
    check(loina_cache_restore(flat,flat_session));
    loina_session *flat_again=loina_session_load(flat,"flat-session",&error);check(flat_again && flat_again->count==2);
    loina_session_free(flat_again);
    int flat_count=0;check(!loina_session_list(flat,count_session,&flat_count,&error) && flat_count==1);
    check(!loina_session_delete(flat,"flat-session",&error));
    flat->cache_session[0]=flat->cache_transcript[0]=0;
    check(!loina_cache_restore(flat,flat_session));
    loina_session_free(flat_session);
    check(!loina_uninstall(flat,&p,&error) && !loina_package_installed(flat,&p));loina_close(flat);restored=restores_before_flat;
    loina_session *s=loina_session_create(e,"regression",&error);check(s);
    check(loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,"Hello"));
    check(loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,"Hi"));
    check(loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,"Again"));
    loina_session_truncate(s,0);
    /* A web round belongs to the user turn that requested it. Compaction
     * cannot leave the tool result behind without its assistant call. */
    check(loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,"Search"));
    check(loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,"<function name=\"web_search\"><param name=\"query\">test</param></function>"));
    check(loina_session_add(s,LOINA_TOOL_RESULT,LOINA_MESSAGE_TEXT,"page"));
    check(loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,"Answer"));
    check(loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,"Next question"));
    check(loina_chat_compact(s,NULL) && s->first==4);
    check(!loina_chat_compact(s,NULL));
    loina_session_truncate(s,0);s->first=0;
    loina_request request=loina_request_default(LOINA_TEXT);
    check(request.temperature==0.1f && request.top_p==1.0f && request.min_p==0.0f && request.thinking);
    request.package=&p;request.session=s;request.prompt="hello";
    loina_job *j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(!strcmp(loina_job_result(j),"fixture"));
    check(loina_job_run(j,&error)==LOINA_INVALID);loina_job_free(j);
    check(s->count==2 && !strcmp(s->package_id,p.id));
    loina_session *again=loina_session_load(e,"regression",&error);check(again && again->count==2);loina_session_free(again);
    check(loina_cache_restore(e,s));check(restored==0);
    loina_unload(e);
    check(loina_activate(e,&p,NULL,&error)==LOINA_OK);
    check(loina_cache_restore(e,s));check(restored==1);
    check(loina_set_backend(e,LOINA_CPU,&error)==LOINA_OK && !e->model);
    check(loina_activate(e,&p,NULL,&error)==LOINA_OK);
    check(!loina_cache_restore(e,s));
    check(loina_set_backend(e,0,&error)==LOINA_OK && !e->model);
    check(loina_activate(e,&p,NULL,&error)==LOINA_OK);
    check(loina_cache_restore(e,s));check(restored==2);
    /* A template edit must select a different disk cache after reloading. */
    e->cache_session[0]=e->cache_transcript[0]=0;
    strcpy(e->chat_identity,"custom-template");
    check(!loina_cache_restore(e,s));
    e->chat_identity[0]=0;
    check(loina_cache_restore(e,s));
    int resets=reset_count;
    check(loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,"different transcript"));check(!loina_cache_restore(e,s));
    check(reset_count==resets+1 && !e->cache_session[0]);
    loina_session_truncate(s,2);
    loina_session *fresh=loina_session_create(e,"fresh",&error);check(fresh);
    check(!loina_cache_restore(e,fresh));check(reset_count==resets+2);
    loina_session_free(fresh);
    check(loina_cache_restore(e,s));
    void *resident=e->model;
    int loads=loaded;resets=reset_count;
    j=loina_job_create(e,&request,cancel_event,&j,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_CANCELLED);check(s->count==2);
    check(e->model==resident && loaded==loads && reset_count==resets+1 && !e->cache_session[0]);
    loina_job_free(j);
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(s->count==4 && e->model==resident && loaded==loads);loina_job_free(j);
    loina_session_truncate(s,2);
    request.temperature=NAN;check(!loina_job_create(e,&request,NULL,NULL,&error));request.temperature=0.7f;
    char *text=NULL;
    check(loina_web_search(e,"a b&c",&text,NULL,&error)==LOINA_OK);check(!strcmp(text,"page"));free(text);
    check(loina_web_visit(e,"file:///secret",&text,NULL,&error)==LOINA_INVALID);
    request.tools=true;request.prompt="Find current information";tool_chat=1;
    loina_buffer visible={0};
    j=loina_job_create(e,&request,capture_event,&visible,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);
    check(strstr(loina_job_result(j),"The page says so"));check(web_calls==2);
    check(visible.data && !strcmp(visible.data,"The page says so."));free(visible.data);
    check(s->count==6);loina_job_free(j);tool_chat=0;
    request.prompt="Read this page";tool_chat=2;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(web_calls==3);
    check(s->count==10);loina_job_free(j);tool_chat=0;
    request.prompt="Search with a malformed tool call";tool_chat=3;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(strstr(loina_job_result(j),"The page says so"));
    check(s->count==14);loina_job_free(j);tool_chat=0;
    request.prompt="Search when the browser times out";tool_chat=4;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(strstr(loina_job_result(j),"available sources"));
    check(s->count==20 && web_calls==5);loina_job_free(j);tool_chat=0;
    request.prompt="Search without stopping";tool_chat=5;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);check(strstr(loina_job_result(j),"available sources"));
    check(s->count==28 && web_calls==7);loina_job_free(j);tool_chat=0;
    tool_chat=6;tool_step=0;request.prompt="Keep earlier evidence when a page fails";
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==9);
    check(strstr(loina_job_result(j),"first result") && strstr(s->messages[s->count-2].text,"timed out"));
    check(strstr(s->messages[s->count-4].text,"page"));loina_job_free(j);
    tool_chat=7;tool_step=0;request.prompt="Search again for missing details";
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==11);
    check(strstr(loina_job_result(j),"second search"));loina_job_free(j);
    tool_chat=8;tool_step=0;request.prompt="Stream reasoning before the answer";
    visible=(loina_buffer){0};
    j=loina_job_create(e,&request,capture_event,&visible,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && thinking_live);
    check(!strcmp(visible.data,loina_job_result(j)));free(visible.data);loina_job_free(j);
    size_t before=s->count;
    j=loina_job_create(e,&request,cancel_event,&j,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_CANCELLED && s->count==before);loina_job_free(j);
    tool_chat=9;tool_step=0;request.prompt="Finish an answer after a reasoning cutoff";
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);
    check(strstr(loina_job_result(j),"available sources") && strstr(s->messages[s->count-3].text,"</think>"));
    check(j->request.tools && j->request.thinking);loina_job_free(j);tool_chat=0;
    tool_chat=10;tool_step=0;request.prompt="A model reply containing two search calls";
    before=s->count;
    visible=(loina_buffer){0};
    j=loina_job_create(e,&request,capture_event,&visible,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==13);
    check(strstr(loina_job_result(j),"The page says so") && s->count==before+4);
    check(strstr(s->messages[before+1].text,"web_visit"));
    check(s->messages[before+2].role==LOINA_TOOL_RESULT);
    check(!strcmp(visible.data,loina_job_result(j)) && !strstr(visible.data,"<tool_call>"));
    free(visible.data);loina_job_free(j);
    tool_chat=11;tool_step=0;request.prompt="Reject unknown tool parameters";
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==13);
    check(strstr(s->messages[s->count-2].text,"Invalid function format"));loina_job_free(j);tool_chat=0;
    for (int mode=12;mode<=14;mode++) {
        tool_chat=mode;tool_step=0;request.prompt="Validate every call before dispatch";
        j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
        check(loina_job_run(j,&error)==LOINA_OK && web_calls==(mode==14 ? 14 : 13));
        check(strstr(loina_job_result(j),"The page says so"));loina_job_free(j);
    }
    tool_chat=15;tool_step=0;request.prompt="Explain an ordinary HTML attribute";
    visible=(loina_buffer){0};
    j=loina_job_create(e,&request,capture_event,&visible,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==14);
    check(!strcmp(visible.data,loina_job_result(j)) && strstr(visible.data,"name=\"example\""));
    free(visible.data);loina_job_free(j);
    tool_chat=0;
    for (int mode=16;mode<=25;mode++) {
        tool_chat=mode;tool_step=0;request.prompt="Parse native Python-style tool calls";
        int calls=web_calls,expected=mode==17 ? 2 : mode==16 || mode==21 ? 1 : 0;
        visible=(loina_buffer){0};
        j=loina_job_create(e,&request,capture_event,&visible,&error);check(j);
        check(loina_job_run(j,&error)==LOINA_OK && web_calls==calls+expected);
        check(strstr(loina_job_result(j),"The page says so"));
        check(!strstr(visible.data,"<|tool_call_start|>"));
        free(visible.data);loina_job_free(j);
    }
    tool_chat=26;tool_step=0;request.prompt="Finish after requesting a tool that is no longer available";
    int calls=web_calls;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==calls+2 && tool_step==5);
    check(strstr(loina_job_result(j),"available sources"));loina_job_free(j);
    request.tools=false;tool_step=0;calls=web_calls;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && web_calls==calls);
    loina_job_free(j);request.tools=true;
    tool_chat=0;p.thinking=false;
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && !thinking_requested && j->request.thinking);
    loina_job_free(j);p.thinking=true;
    check(loina_session_save(s,&error)==LOINA_OK);
    again=loina_session_load(e,"regression",&error);check(again && again->count==s->count);
    check(again->messages[again->count-4].role==LOINA_TOOL_RESULT);loina_session_free(again);
    loina_session *old=loina_session_create(e,"old-chat",&error);check(old);
    check(loina_session_add(old,LOINA_USER,LOINA_MESSAGE_TEXT,"Remember the codeword: turquoise."));
    check(loina_session_add(old,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,"I will remember turquoise."));
    const char *retired_id="lfm2-5-vl-3b-q4_k_m-c815851c1e0a1361";
    strcpy(old->package_id,retired_id);old->first=2;
    check(loina_session_save(old,&error)==LOINA_OK);loina_session_free(old);
    old=loina_session_load(e,"old-chat",&error);check(old);
    request.session=old;request.package=&p;request.tools=false;request.prompt="What is my codeword?";
    migrating=true;
    j=loina_job_create(e,&request,cancel_event,&j,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_CANCELLED);loina_job_free(j);
    check(old->count==2 && old->first==2 && !strcmp(old->package_id,retired_id));
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK);loina_job_free(j);migrating=false;
    check(old->count==4 && old->first==0 && !strcmp(old->package_id,p.id));
    loina_session_free(old);old=loina_session_load(e,"old-chat",&error);
    check(old && old->count==4 && old->first==0 && !strcmp(old->package_id,p.id));loina_session_free(old);
    check(loina_session_delete(e,"old-chat",&error)==LOINA_OK);
    char retired_dir[LOINA_PATH];check(loina_path(retired_dir,e->directory,"packages",retired_id) && loina_mkdir(retired_dir));
    check(loina_path(path,retired_dir,"LFM2.5-VL-3B-Q4_K_M.gguf",NULL));write_file(path,"old",3);
    check(loina_path(path,retired_dir,"ready",NULL));write_file(path,"old",3);
    check(loina_packages_upgrade(e,&error)==LOINA_OK);
    check(loina_file_bytes(path)==UINT64_MAX && loina_package_installed(e,&p));
    check(!loina_session_load(e,"../escape",&error));
    int sessions=0;check(!loina_session_list(e,count_session,&sessions,&error));check(sessions>=1);
    loina_session *ephemeral=loina_session_create(e,"temporary-test",&error);check(ephemeral);
    loina_session_set_temporary(ephemeral,true);
    request.session=ephemeral;request.prompt="Keep this conversation in memory";
    for (int turn=0;turn<2;turn++) {
        j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
        check(loina_job_run(j,&error)==LOINA_OK);loina_job_free(j);
        check(loina_session_count(ephemeral)==(size_t)(turn+1)*2);
        check(!loina_session_load(e,"temporary-test",NULL));
        int listed=0;check(!loina_session_list(e,count_session,&listed,&error));check(listed==sessions);
        check(loina_cache_restore(e,ephemeral));
    }
    e->cache_session[0]=e->cache_transcript[0]=0;
    check(!loina_cache_restore(e,ephemeral));
    loina_session_free(ephemeral);
    check(!loina_session_delete(e,"manual-save",&error));
    loina_session *manual=loina_session_create(e,"manual-save",&error);check(manual);
    loina_session_set_autosave(manual,false);
    request.session=manual;request.prompt="Save only on request";
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(!loina_job_run(j,&error));loina_job_free(j);
    check(manual->count==2 && !loina_session_load(e,"manual-save",NULL));
    check(loina_cache_restore(e,manual));
    atomic_store(&e->busy,true);
    check(loina_session_checkpoint(manual,&error)==LOINA_BUSY);
    atomic_store(&e->busy,false);
    check(!loina_session_checkpoint(manual,&error) && manual->manual_save);
    again=loina_session_load(e,"manual-save",&error);check(again && again->count==2);loina_session_free(again);
    e->cache_session[0]=e->cache_transcript[0]=0;
    int restores=restored;
    check(loina_cache_restore(e,manual) && restored==restores+1);
    j=loina_job_create(e,&request,NULL,NULL,&error);check(j);
    check(!loina_job_run(j,&error));loina_job_free(j);
    again=loina_session_load(e,"manual-save",&error);check(again && again->count==2);loina_session_free(again);
    fail_save=true;
    check(loina_session_checkpoint(manual,&error)==LOINA_IO && manual->manual_save && !atomic_load(&e->busy));
    fail_save=false;
    check(!loina_session_checkpoint(manual,&error));
    again=loina_session_load(e,"manual-save",&error);check(again && again->count==4);loina_session_free(again);
    loina_session_free(manual);
    loina_unload(e);
    check(!loina_activate(e,&p,NULL,&error));
    manual=loina_session_load(e,"manual-save",&error);check(manual);
    loina_session_set_autosave(manual,false);
    restores=restored;
    check(!loina_session_checkpoint(manual,&error) && restored==restores+1);
    loina_session_free(manual);check(!loina_session_delete(e,"manual-save",&error));
    loina_session *host=loina_session_create(e,"host-tools",&error);check(host);
    loina_session_set_temporary(host,true);
    loina_request external=request;
    char schema[]="[{\"name\":\"write\"}]";
    external.session=host;external.tool_schema=schema;external.tools=false;
    external.prompt="Create probe.py";
    loina_buffer raw={0};
    int before_web=web_calls;
    j=loina_job_create(e,&external,capture_event,&raw,&error);check(j);
    schema[0]=0;
    check(loina_job_run(j,&error)==LOINA_OK);
    check(raw.data && strstr(raw.data,"<function=write>") && !strcmp(raw.data,loina_job_result(j)));
    check(web_calls==before_web && loina_session_count(host)==2);
    loina_job_free(j);free(raw.data);
    external.tool_schema="[{\"name\":\"write\"}]";
    external.tool_response=true;external.prompt="File written.";
    j=loina_job_create(e,&external,NULL,NULL,&error);check(j);
    check(loina_job_run(j,&error)==LOINA_OK && !strcmp(loina_job_result(j),"Created."));
    check(loina_session_count(host)==4 && web_calls==before_web);
    loina_job_free(j);loina_session_free(host);
    check(loina_path(path,argv[1],"outputs","sample.png"));
    unsigned char pixels[12]={255,0,0,0,255,0,0,0,255,255,255,255};
    check(!loina_png(path,pixels,2,2,3,&error));check(loina_file_bytes(path)>50);
    check(loina_path(path,argv[1],"sessions","bad.chat"));write_file(path,"LOINA01\n\377\377\377\377",12);
    check(!loina_session_load(e,"bad",&error));
    check(!loina_session_delete(e,"bad",&error));check(!loina_session_delete(e,"regression",&error));
    loina_package other=p;other.id="second-local-model";
    check(!loina_install(e,&other,import,NULL,&error));
    check(loina_select(e,LOINA_VISION,&other,&error)==LOINA_INVALID);
    check(!loina_select(e,LOINA_TEXT,&p,&error) && loina_resolve(e,LOINA_TEXT,&error)==&p);
    loina_session *switched=loina_session_create(e,"model-switch",&error);check(switched);
    loina_request switching=loina_request_default(LOINA_TEXT);switching.session=switched;switching.prompt="First model";
    j=loina_job_create(e,&switching,NULL,NULL,&error);check(j && !loina_job_run(j,&error));loina_job_free(j);
    check(!strcmp(switched->package_id,p.id) && switched->count==2);
    check(!loina_select(e,LOINA_TEXT,&other,&error) && !e->model);
    switching.prompt="Continue on the other model";
    j=loina_job_create(e,&switching,NULL,NULL,&error);check(j && !loina_job_run(j,&error));loina_job_free(j);
    check(!strcmp(switched->package_id,other.id) && switched->count==4);
    check(!loina_select(e,LOINA_TEXT,NULL,&error) && loina_resolve(e,LOINA_TEXT,&error)!=&other);
    loina_session_free(switched);check(!loina_session_delete(e,"model-switch",&error));
    check(!loina_uninstall(e,&other,&error));
    check(!loina_uninstall(e,&p,&error));check(!loina_package_installed(e,&p));
    loina_session_free(s);loina_close(e);check(loaded==unloaded);
    puts("Core regression tests passed");
    return 0;
}

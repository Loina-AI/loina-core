#include "loina_internal.h"
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "jinja.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef LOINA_LLAMA_REVISION
#define LOINA_LLAMA_REVISION "unknown"
#endif
typedef struct {
    struct llama_model *model;
    struct llama_context *context;
    mtmd_context *vision;
    ggml_backend_dev_t devices[2];
    llama_token *tokens;
    size_t count,capacity;
    uint32_t context_size;
    unsigned backend;
    loina_job *job;
    llama_jinja *chat_template, *tool_template, *system_template;
    char *tools;
    const char *thinking_prefix;
    bool lfm3;
} llama_state;

static ggml_backend_dev_t gpu_device(unsigned bits, uint64_t minimum) {
    ggml_backend_load_all();
    ggml_backend_dev_t best=NULL;
    uint64_t best_free=0;
    for (size_t i=0;i<ggml_backend_dev_count();i++) {
        ggml_backend_dev_t d=ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type=ggml_backend_dev_type(d);
        if (type!=GGML_BACKEND_DEVICE_TYPE_GPU && type!=GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const char *name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        unsigned backend=strstr(name,"OpenCL") ? LOINA_OPENCL : strstr(name,"Vulkan") ? LOINA_VULKAN :
            strstr(name,"Metal") ? LOINA_METAL : strstr(name,"CUDA") ? LOINA_CUDA : 0;
        if (!(bits & backend)) continue;
        size_t free_bytes=0,total=0;
        ggml_backend_dev_memory(d,&free_bytes,&total);
        if (free_bytes>=minimum && free_bytes>best_free) { best=d;best_free=free_bytes; }
    }
    return best;
}
static void llama_quiet_default(void);
static unsigned llama_backends(void) {
    llama_quiet_default();
    ggml_backend_load_all();
    unsigned bits=LOINA_CPU;
#if defined(__aarch64__) || defined(_M_ARM64)
    bits|=LOINA_NEON;
#endif
    for (size_t i=0;i<ggml_backend_dev_count();i++) {
        ggml_backend_dev_t d=ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type=ggml_backend_dev_type(d);
        if (type!=GGML_BACKEND_DEVICE_TYPE_GPU && type!=GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const char *name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        if (strstr(name,"Vulkan")) bits|=LOINA_VULKAN;
        if (strstr(name,"OpenCL")) bits|=LOINA_OPENCL;
        if (strstr(name,"Metal")) bits|=LOINA_METAL;
        if (strstr(name,"CUDA")) bits|=LOINA_CUDA;
    }
    return bits;
}
static bool load_progress(float progress, void *user) {
    loina_job *job=user;
    loina_job_progress(job,(uint64_t)(progress*1000),1000);
    return !loina_job_cancelled(job);
}
static bool abort_decode(void *user) { return loina_job_cancelled(((llama_state *)user)->job); }
/* Reference hosts get quiet runtime logs (warnings/errors only). A host with
 * its own bridge keeps it: Android installs one before open. */
extern void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data);
static void llama_quiet(enum ggml_log_level level, const char *text, void *user) {
    (void)user;
    /* The full-size cache notice describes a supported configuration. */
    if (strstr(text, "llama_kv_cache_iswa: using full-size SWA cache")) return;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
        fflush(stderr);
    }
}
static void llama_quiet_default(void) {
    ggml_log_callback cur = NULL;
    void *ud = NULL;
    ggml_log_get(&cur, &ud);
    if (cur == ggml_log_callback_default)
        llama_log_set(llama_quiet, NULL);
}
static void llama_unload(void *pointer) {
    llama_state *s=pointer;
    if (!s) return;
    if (s->vision) mtmd_free(s->vision);
    if (s->context) llama_free(s->context);
    if (s->model) llama_model_free(s->model);
    llama_jinja_free(s->chat_template);llama_jinja_free(s->tool_template);llama_jinja_free(s->system_template);
    free(s->tools);free(s->tokens);free(s);
}
static void *llama_load(loina_engine *e, const loina_package *p, loina_job *job, loina_error *error) {
    char path[LOINA_PATH];
    if (!loina_artifact_path(e,p,0,path)) { loina_fail(error,LOINA_INVALID,"Invalid model path");return NULL; }
    llama_quiet_default();
    llama_backend_init();
    ggml_backend_dev_t candidates[5];
    int count=0;
    const unsigned order[]={LOINA_OPENCL,LOINA_VULKAN,LOINA_METAL,LOINA_CUDA};
    for (size_t i=0;i<sizeof(order)/sizeof(order[0]);i++) {
        /* The RAM estimate includes host work, and a projector may use CPU.
         * Check the language weights here; runtime allocation validates the rest. */
        ggml_backend_dev_t device=gpu_device(e->config.device.backends & order[i],p->files[0].bytes);
        if (device) candidates[count++]=device;
    }
    if (!e->selected_backend || (e->selected_backend & (LOINA_CPU|LOINA_NEON))) candidates[count++]=NULL;
    for (int attempt=0;attempt<count;attempt++) {
        ggml_backend_dev_t gpu=candidates[attempt];
        bool accelerated=gpu!=NULL;
        llama_state *s=calloc(1,sizeof(*s));
        if (!s) { loina_fail(error,LOINA_MEMORY,"Cannot allocate llama state");return NULL; }
        s->devices[0]=gpu;
        s->backend=accelerated ? 0 : LOINA_CPU;
        if (gpu) {
            const char *name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu));
            s->backend=strstr(name,"OpenCL") ? LOINA_OPENCL : strstr(name,"Vulkan") ? LOINA_VULKAN :
                strstr(name,"Metal") ? LOINA_METAL : LOINA_CUDA;
        }
        struct llama_model_params mp=llama_model_default_params();
        /* NULL discovers GPUs even with CPU weights. An empty list and
         * disabled operation offload keep an explicit CPU request on CPU. */
        mp.devices=s->devices;
        mp.split_mode=LLAMA_SPLIT_MODE_NONE;
        mp.n_gpu_layers=accelerated ? -1 : 0;
        mp.load_mode=accelerated ? LLAMA_LOAD_MODE_AUTO : LLAMA_LOAD_MODE_MMAP;
        mp.progress_callback=load_progress;mp.progress_callback_user_data=job;
        s->model=llama_model_load_from_file(path,mp);
        if (!s->model) goto retry;
        const char *embedded=llama_model_chat_template(s->model,NULL);
        const char *embedded_tools=llama_model_chat_template(s->model,"tool_use");
        if (!embedded || !strcmp(embedded,"chatml")) embedded=embedded_tools ? embedded_tools : llama_jinja_default_template();
        char *chat=loina_template_model(e,p->chat_template,embedded,error);
        char *tool=loina_template_model(e,p->chat_template,embedded_tools ? embedded_tools : embedded,error);
        char *system=loina_template_read(e,"system.jinja",error);
        s->tools=loina_template_read(e,"tools.json",error);
        char message[512]="";
        if (chat && tool && system && s->tools) {
            s->chat_template=llama_jinja_create(chat,message,sizeof(message));
            if (s->chat_template && strcmp(chat,tool)) s->tool_template=llama_jinja_create(tool,message,sizeof(message));
            if (s->chat_template && (!strcmp(chat,tool) || s->tool_template)) s->system_template=llama_jinja_create(system,message,sizeof(message));
        }
        if (!s->chat_template || !s->system_template || (chat && tool && strcmp(chat,tool) && !s->tool_template)) {
            free(chat);free(tool);free(system);llama_unload(s);
            if (*message) loina_fail(error,LOINA_UNSUPPORTED,"Invalid Jinja template: %s",message);
            return NULL;
        }
        loina_sha identity;loina_sha_init(&identity);
        loina_sha_update(&identity,chat,strlen(chat)+1);
        loina_sha_update(&identity,tool,strlen(tool)+1);
        loina_sha_update(&identity,system,strlen(system)+1);
        loina_sha_update(&identity,s->tools,strlen(s->tools)+1);
        loina_sha_final(&identity,e->chat_identity);
        free(chat);free(tool);free(system);
        char architecture[64]="";
        llama_model_meta_val_str(s->model,"general.architecture",architecture,sizeof(architecture));
        s->lfm3=!strcmp(architecture,"lfm3");
        struct llama_context_params cp=llama_context_default_params();
        cp.n_ctx=e->config.context;
        int32_t trained=llama_model_n_ctx_train(s->model);
        if (trained>0 && cp.n_ctx>(uint32_t)trained) cp.n_ctx=(uint32_t)trained;
        cp.n_batch=256;cp.n_ubatch=128;
        cp.n_threads=cp.n_threads_batch=e->config.device.threads;
        cp.offload_kqv=accelerated;cp.op_offload=accelerated;
        /* Quantized V requires flash attention. Keep the specified Q8 K / Q4 V
         * layout consistent across CPU and GPU; fail if a backend cannot run it. */
        cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.abort_callback=abort_decode;cp.abort_callback_data=s;
        s->job=job;
        s->context=llama_init_from_model(s->model,cp);
        if (!s->context) goto retry;
        s->context_size=llama_n_ctx(s->context);
        s->capacity=(size_t)s->context_size+1;
        s->tokens=malloc(s->capacity*sizeof(*s->tokens));
        if (!s->tokens) goto retry;
        if (p->feature==LOINA_VISION) {
            if (!loina_artifact_path(e,p,1,path)) goto retry;
            struct mtmd_context_params vp=mtmd_context_params_default();
            vp.use_gpu=accelerated;vp.device=gpu;vp.n_threads=e->config.device.threads;vp.warmup=false;
            vp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            vp.progress_callback=load_progress;vp.progress_callback_user_data=job;
            s->vision=mtmd_init_from_file(path,s->model,vp);
            if (!s->vision) goto retry;
        }
        s->job=NULL;
        return s;
retry:
        llama_unload(s);
        if (loina_job_cancelled(job)) break;
        if (accelerated) loina_emit(job,LOINA_LOADING,attempt+1<count-1 ? "GPU fallback" : "CPU / NEON",10,0);
    }
    loina_fail(error,loina_job_cancelled(job) ? LOINA_CANCELLED : LOINA_RUNTIME_ERROR,"Cannot initialize llama package %s",p->name);
    return NULL;
}
static char *special_token(const struct llama_vocab *vocab, llama_token token) {
    if (token<0) return loina_strdup("");
    int n=llama_token_to_piece(vocab,token,NULL,0,0,true);
    if (n<0) n=-n;
    char *text=malloc((size_t)n+1);
    if (!text) return NULL;
    int actual=llama_token_to_piece(vocab,token,text,n+1,0,true);
    if (actual<0 || actual>n) { free(text);return NULL; }
    text[actual]=0;return text;
}
static char *render(llama_state *state, loina_session *session, loina_job *job, loina_error *error) {
    char date[32]="",local_date[80]="",context[256],message[512]="";
    time_t now=time(NULL);
    struct tm *today=localtime(&now);
    if (today) {
        strftime(date,sizeof(date),"%Y-%m-%d",today);
        strftime(local_date,sizeof(local_date),"%Y-%m-%d (%A, UTC%z)",today);
    }
    snprintf(context,sizeof(context),"{\"date\":\"%s\",\"local_date\":\"%s\",\"tools\":%s",
        date,local_date,(job->request.tools || job->request.tool_schema) ? "true" : "false");
    loina_buffer variables={0};
    bool valid=loina_buffer_puts(&variables,context) &&
        loina_buffer_puts(&variables,job->request.tool_schema ? ",\"agent\":true" : ",\"agent\":false");
    if (job->request.tool_schema) {
        char directory[LOINA_PATH];
        valid=valid && getcwd(directory,sizeof(directory)) &&
            loina_buffer_puts(&variables,",\"working_directory\":") && loina_buffer_json(&variables,directory);
    }
    valid=valid && loina_buffer_puts(&variables,"}");
    if (!valid) { free(variables.data);loina_fail(error,LOINA_MEMORY,"Cannot retain system context");return NULL; }
    char *system=llama_jinja_render(state->system_template,variables.data,message,sizeof(message));
    free(variables.data);
    if (!system) { loina_fail(error,LOINA_UNSUPPORTED,"Cannot render system template: %s",message);return NULL; }
    const struct llama_vocab *vocab=llama_model_get_vocab(state->model);
    char *bos=special_token(vocab,llama_vocab_bos(vocab)),*eos=special_token(vocab,llama_vocab_eos(vocab));
    loina_buffer b={0};
    bool ok=bos && eos && loina_buffer_puts(&b,"{\"bos_token\":") && loina_buffer_json(&b,bos) &&
        loina_buffer_puts(&b,",\"eos_token\":") && loina_buffer_json(&b,eos) &&
        loina_buffer_puts(&b,",\"add_generation_prompt\":true,\"enable_thinking\":") &&
        loina_buffer_puts(&b,job->request.thinking ? "true" : "false") &&
        loina_buffer_puts(&b,",\"external_tools\":") && loina_buffer_puts(&b,job->request.tool_schema ? "true" : "false") &&
        loina_buffer_puts(&b,",\"tools\":") && loina_buffer_puts(&b,job->request.tool_schema ? job->request.tool_schema : job->request.tools ? state->tools : "[]") &&
        loina_buffer_puts(&b,",\"messages\":[{\"role\":\"system\",\"content\":") &&
        loina_buffer_json(&b,system) && loina_buffer_puts(&b,"}");
    free(system);free(bos);free(eos);
    for (size_t i=session->first;ok && i<session->count;i++) {
        const loina_message *m=&session->messages[i];
        const char *role=m->role==LOINA_USER ? "user" : m->role==LOINA_ASSISTANT ? "assistant" :
            m->role==LOINA_TOOL_RESULT ? "tool" : "system";
        const char *content=m->kind==LOINA_MESSAGE_IMAGE ? "[Generated image]" : m->text;
        loina_buffer image={0};
        if (state->vision && i==session->count-1) {
            ok=loina_buffer_puts(&image,mtmd_default_marker()) && loina_buffer_puts(&image,"\n") && loina_buffer_puts(&image,content);
            content=image.data;
        }
        ok=ok && loina_buffer_puts(&b,",{\"role\":") && loina_buffer_json(&b,role) &&
            loina_buffer_puts(&b,",\"content\":") && loina_buffer_json(&b,content) && loina_buffer_puts(&b,"}");
        free(image.data);
    }
    ok=ok && loina_buffer_puts(&b,"]}");
    if (!ok) { free(b.data);loina_fail(error,LOINA_MEMORY,"Cannot retain template context");return NULL; }
    llama_jinja *tmpl=(job->request.tools || job->request.tool_schema) && state->tool_template ? state->tool_template : state->chat_template;
    char *prompt=llama_jinja_render_chat(tmpl,b.data,message,sizeof(message));
    free(b.data);
    if (!prompt) { loina_fail(error,LOINA_UNSUPPORTED,"Cannot render chat template: %s",message);return NULL; }
    size_t n=strlen(prompt);
    if (!n || n>LOINA_MAX_TEXT) { free(prompt);loina_fail(error,LOINA_INVALID,"Rendered prompt is too large or empty");return NULL; }
    /* The opening thought marker is part of the generation prefill, so it
     * must also reach the transcript and the existing reasoning display. */
    state->thinking_prefix=n>=8 && !strcmp(prompt+n-8,"<think>\n") ? "<think>\n" :
        n>=7 && !strcmp(prompt+n-7,"<think>") ? "<think>" : NULL;
    return prompt;
}
static int decode(llama_state *s, const llama_token *tokens, int count, int position) {
    struct llama_batch b=llama_batch_init(count,0,1);
    if (!b.token) return -1;
    b.n_tokens=count;
    for (int i=0;i<count;i++) {
        b.token[i]=tokens[i];b.pos[i]=position+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;
        b.logits[i]=(i==count-1);
    }
    int rc=llama_decode(s->context,b);
    llama_batch_free(b);
    return rc;
}
static void llama_reset(void *pointer) {
    llama_state *s=pointer;
    llama_synchronize(s->context);
    llama_memory_clear(llama_get_memory(s->context),true);
    s->count=0;
}
static void metrics(llama_state *s, loina_job *job, size_t prompt_tokens, int64_t prompt_us,
                    uint32_t decoded, int64_t decode_us) {
    char text[224];
    int n=snprintf(text,sizeof(text),"{\"prefill\":%.2f,\"decode\":%.2f,\"backend\":%u,\"context\":%zu,\"capacity\":%u,\"generated\":%u}",
        prompt_us>0 ? prompt_tokens*1000000.0/prompt_us : 0,
        decode_us>0 ? decoded*1000000.0/decode_us : 0,s->backend,s->count,s->context_size,decoded);
    loina_emit(job,LOINA_METRICS,text,(size_t)n,0);
}
static loina_status llama_chat_run(void *pointer, loina_session *session, loina_job *job, char **result, loina_error *error) {
    llama_state *s=pointer;
    s->job=job;
    const struct llama_vocab *vocab=llama_model_get_vocab(s->model);
    char *prompt=NULL;
    llama_token *tokens=NULL;
    int32_t count=0;
    loina_status status=LOINA_OK;
    struct llama_sampler *sampler=NULL;
    loina_buffer output={0};
    size_t prompt_tokens=0;
    int64_t prompt_start=llama_time_us(),prompt_us=0,decode_start=0;
    uint32_t decoded=0;
    for (;;) {
        free(prompt);prompt=render(s,session,job,error);
        if (!prompt) { status=error ? error->code : LOINA_RUNTIME_ERROR;goto done; }
        /* Jinja already emits the model's special tokens, including BOS. */
        count=-llama_tokenize(vocab,prompt,(int32_t)strlen(prompt),NULL,0,false,true);
        if (count<=0) { status=loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot tokenize prompt");goto done; }
        if ((uint32_t)count+job->request.max_tokens+8<s->context_size || s->vision) break;
        if (!loina_chat_compact(session,job)) { status=loina_fail(error,LOINA_INVALID,"Message is too large for the model context");goto done; }
    }
    if (s->vision) {
        llama_memory_clear(llama_get_memory(s->context),true);s->count=0;
        struct mtmd_helper_bitmap_wrapper image=mtmd_helper_bitmap_init_from_file(s->vision,job->input,false,mtmd_helper_init_opt_default());
        mtmd_input_chunks *chunks=mtmd_input_chunks_init();
        mtmd_input_text input={prompt,strlen(prompt),false,true};
        const mtmd_bitmap *images[]={image.bitmap};
        if (!image.bitmap || !chunks || mtmd_tokenize(s->vision,chunks,&input,images,1)) {
            if (image.bitmap) mtmd_bitmap_free(image.bitmap);
            if (chunks) mtmd_input_chunks_free(chunks);
            status=loina_fail(error,LOINA_INVALID,"Cannot decode or tokenize the input image");goto done;
        }
        llama_pos position=0;
        if (mtmd_helper_get_n_pos(chunks)+(size_t)job->request.max_tokens+8>s->context_size) status=loina_fail(error,LOINA_INVALID,"Image exceeds the model context");
        else if (mtmd_helper_eval_chunks(s->vision,s->context,chunks,0,0,(int32_t)llama_n_batch(s->context),true,&position)) status=loina_fail(error,LOINA_RUNTIME_ERROR,"Image encoder failed");
        mtmd_bitmap_free(image.bitmap);mtmd_input_chunks_free(chunks);
        if (status) goto done;
        s->count=(size_t)position;
        prompt_tokens=s->count;
    } else {
        tokens=malloc((size_t)count*sizeof(*tokens));
        if (!tokens) { status=loina_fail(error,LOINA_MEMORY,"Cannot allocate prompt tokens");goto done; }
        if (llama_tokenize(vocab,prompt,(int32_t)strlen(prompt),tokens,count,false,true)!=count) { status=loina_fail(error,LOINA_RUNTIME_ERROR,"Tokenizer count changed");goto done; }
        size_t common=0;
        while (common<s->count && common<(size_t)count && s->tokens[common]==tokens[common]) common++;
        if (common==(size_t)count && common) common--; /* Recompute the last logits. */
        if (!llama_memory_seq_rm(llama_get_memory(s->context),0,(llama_pos)common,-1)) {
            llama_memory_clear(llama_get_memory(s->context),true);common=0;
        }
        s->count=common;
        prompt_tokens=(size_t)count-common;
        for (int32_t i=(int32_t)common;i<count;) {
            if (loina_job_cancelled(job)) { status=LOINA_CANCELLED;goto done; }
            int32_t n=count-i;
            if (n>256) n=256;
            if (decode(s,tokens+i,n,i)) { status=loina_fail(error,LOINA_RUNTIME_ERROR,"Prompt evaluation failed");goto done; }
            memcpy(s->tokens+i,tokens+i,(size_t)n*sizeof(*tokens));s->count+=(size_t)n;i+=n;
            loina_job_progress(job,(uint64_t)i,(uint64_t)count);
        }
    }
    llama_synchronize(s->context);
    prompt_us=llama_time_us()-prompt_start;
    metrics(s,job,prompt_tokens,prompt_us,0,0);
    sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) { status=loina_fail(error,LOINA_MEMORY,"Cannot allocate sampler");goto done; }
    if (job->request.temperature==0) llama_sampler_chain_add(sampler,llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(sampler,llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),s->lfm3 ? 64 : (int32_t)s->context_size,s->lfm3 ? 1.1f : 1.03f,0.0f,0.0f));
        llama_sampler_chain_add(sampler,llama_sampler_init_top_k(s->lfm3 ? 50 : 100));
        llama_sampler_chain_add(sampler,llama_sampler_init_top_p(job->request.top_p,1));
        if (job->request.min_p>0) llama_sampler_chain_add(sampler,llama_sampler_init_min_p(job->request.min_p,1));
        llama_sampler_chain_add(sampler,llama_sampler_init_temp(job->request.temperature));
        llama_sampler_chain_add(sampler,llama_sampler_init_dist(job->request.seed));
    }
    for (size_t i=0;!s->vision && i<s->count;i++) llama_sampler_accept(sampler,s->tokens[i]);
    if (s->thinking_prefix) {
        if (!loina_buffer_puts(&output,s->thinking_prefix)) { status=loina_fail(error,LOINA_MEMORY,"Cannot retain reasoning prefix");goto done; }
        loina_emit(job,LOINA_TOKEN,s->thinking_prefix,strlen(s->thinking_prefix),0);
    }
    decode_start=llama_time_us();
    for (uint32_t i=0;i<job->request.max_tokens;i++) {
        if (loina_job_cancelled(job)) { status=LOINA_CANCELLED;goto done; }
        llama_token token=llama_sampler_sample(sampler,s->context,-1);
        if (llama_vocab_is_eog(vocab,token)) break;
        /* Retain the model's special tool delimiters for dispatch. */
        int32_t n=llama_token_to_piece(vocab,token,NULL,0,0,true);
        if (n<0) n=-n;
        char *piece=malloc((size_t)n+1);
        if (!piece) { status=loina_fail(error,LOINA_MEMORY,"Cannot allocate token piece");goto done; }
        int32_t written=llama_token_to_piece(vocab,token,piece,n+1,0,true);
        if (written<0 || written>n || !loina_buffer_add(&output,piece,(size_t)written)) { free(piece);status=loina_fail(error,LOINA_MEMORY,"Cannot retain token output");goto done; }
        loina_emit(job,LOINA_TOKEN,piece,(size_t)written,0);free(piece);
        if (s->count>=s->context_size || decode(s,&token,1,(int)s->count)) { status=loina_fail(error,LOINA_RUNTIME_ERROR,"Token evaluation failed");goto done; }
        if (!s->vision) s->tokens[s->count]=token;
        s->count++;
        decoded++;
        if (!(decoded%8)) {
            llama_synchronize(s->context);
            metrics(s,job,prompt_tokens,prompt_us,decoded,llama_time_us()-decode_start);
        }
    }
    llama_synchronize(s->context);
    metrics(s,job,prompt_tokens,prompt_us,decoded,llama_time_us()-decode_start);
    if (!output.data) output.data=loina_strdup("");
    if (!output.data) { status=loina_fail(error,LOINA_MEMORY,"Cannot retain result");goto done; }
    *result=output.data;output.data=NULL;
done:
    free(prompt);free(tokens);free(output.data);
    if (sampler) llama_sampler_free(sampler);
    s->job=NULL;
    if (status==LOINA_CANCELLED || loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Generation cancelled");
    return status;
}
static bool llama_save(void *pointer, const char *path) {
    llama_state *s=pointer;
    if (s->vision) return false;
    llama_synchronize(s->context);
    return llama_state_save_file(s->context,path,s->tokens,s->count);
}
static bool llama_restore(void *pointer, const char *path) {
    llama_state *s=pointer;
    if (s->vision) return false;
    size_t count=0;
    if (!llama_state_load_file(s->context,path,s->tokens,s->capacity,&count) || count>s->context_size) {
        llama_memory_clear(llama_get_memory(s->context),true);s->count=0;return false;
    }
    s->count=count;
    return true;
}
const loina_runtime loina_llama={
    .name="llama",.revision=LOINA_LLAMA_REVISION,.abi=LOINA_RUNTIME_ABI,
    .backends=llama_backends,.load=llama_load,.unload=llama_unload,.chat=llama_chat_run,
    .save=llama_save,.restore=llama_restore,.reset=llama_reset
};

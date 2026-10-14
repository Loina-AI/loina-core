/* Compare a continued checkpoint against an independent replay, and exercise
 * restore after unrelated device work. */
#include "../loina_llama.c"
#include "ggml-alloc.h"
#include <ctype.h>
#include <math.h>

#define check(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while (0)
typedef struct { loina_job *job; loina_buffer text; bool cancel; int metrics; } stream;
static void log_message(enum ggml_log_level level, const char *text, void *user) {
    (void)user;
    if (level==GGML_LOG_LEVEL_WARN || level==GGML_LOG_LEVEL_ERROR) fputs(text,stderr);
}
static void capture(void *user, loina_event type, const char *text, size_t n, int progress) {
    (void)progress;
    stream *s=user;
    if (type==LOINA_TOKEN) {
        check(loina_buffer_add(&s->text,text,n));
        if (s->cancel) loina_job_cancel(s->job);
    }
    if (type==LOINA_METRICS) s->metrics++;
}
static char *run(loina_engine *e, loina_session *s, const char *prompt, bool cancel) {
    loina_request r=loina_request_default(LOINA_TEXT);
    r.session=s;r.prompt=prompt;r.temperature=0;r.max_tokens=192;r.seed=42;
    loina_error error={0};
    stream output={.cancel=cancel};
    output.job=loina_job_create(e,&r,capture,&output,&error);check(output.job);
    loina_status status=loina_job_run(output.job,&error);
    if (status && status!=LOINA_CANCELLED) fprintf(stderr,"%s\n",error.message);
    check(status==(cancel ? LOINA_CANCELLED : LOINA_OK));
    char *reply=cancel ? NULL : loina_strdup(loina_job_result(output.job));
    if (!cancel) {
        check(reply && output.text.data && !strcmp(reply,output.text.data));
        check(output.metrics>0);
    }
    free(output.text.data);loina_job_free(output.job);
    return reply;
}
static loina_session *copy_session(loina_engine *e, loina_session *source, const char *id) {
    loina_session *s=loina_session_create(e,id,NULL);check(s);
    strcpy(s->package_id,source->package_id);
    for (size_t i=0;i<source->count;i++) {
        loina_message m=source->messages[i];
        check(loina_session_add(s,m.role,m.kind,m.text));
    }
    s->first=source->first;
    return s;
}
static const char *answer(const char *text) {
    while (isspace((unsigned char)*text)) text++;
    return text;
}
static bool says(const char *text, const char *word) {
    text=answer(text);size_t n=strlen(word);
    if (strncmp(text,word,n)) return false;
    text+=n;if (*text=='.') text++;
    while (isspace((unsigned char)*text)) text++;
    return !*text;
}
static void signed_weights(const char *name) {
    ggml_backend_dev_t device=NULL;
    for (size_t i=0;i<ggml_backend_dev_count();i++) {
        ggml_backend_dev_t d=ggml_backend_dev_get(i);
        if (!strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)),name)) { device=d;break; }
    }
    check(device);
    ggml_backend_t backend=ggml_backend_dev_init(device,NULL);check(backend);
    const int cols=2048,rows=256;
    size_t bytes=ggml_row_size(GGML_TYPE_Q8_0,cols)*rows;
    unsigned char *weights=malloc(bytes);check(weights);
    /* Exercise every signed byte, both bytes of packed loads and block strides. */
    for (int row=0;row<rows;row++) for (int block=0;block<cols/32;block++) {
        unsigned char *p=weights+(row*cols/32+block)*34;
        ggml_fp16_t scale=ggml_fp32_to_fp16(1);memcpy(p,&scale,sizeof(scale));
        for (int col=0;col<32;col++) p[col+2]=(unsigned char)((row+block*32+col)%256-128);
    }
    for (int n=1;n<=41;n+=40) {
        struct ggml_init_params params={.mem_size=2*1024*1024,.no_alloc=true};
        struct ggml_context *ctx=ggml_init(params);check(ctx);
        struct ggml_tensor *a=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,cols,rows);
        struct ggml_tensor *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,n);
        struct ggml_tensor *y=ggml_mul_mat(ctx,a,x);
        struct ggml_cgraph *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer);
        float *input=calloc((size_t)cols*n,sizeof(float)),*output=malloc((size_t)rows*n*sizeof(float));
        check(input && output);
        for (int i=0;i<n;i++) input[i*cols+(i*53)%cols]=1;
        ggml_backend_tensor_set(a,weights,0,bytes);ggml_backend_tensor_set(x,input,0,ggml_nbytes(x));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y,output,0,ggml_nbytes(y));
        for (int i=0;i<n;i++) for (int row=0;row<rows;row++) {
            float want=(float)((row+(i*53)%cols)%256-128);
            check(isfinite(output[i*rows+row]) && fabsf(output[i*rows+row]-want)<0.02f);
        }
        free(input);free(output);ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    free(weights);ggml_backend_free(backend);puts("Signed Q8 weights pass prefill and decode");
    size_t available=0,total=0;ggml_backend_dev_memory(device,&available,&total);
    printf("%s memory after kernel test: free=%zu total=%zu\n",name,available,total);
}
static void same_logits(llama_state *s, const float *want, const float *got, int32_t count) {
    /* Bound GPU rounding as well as checking the selected token; a stable
     * greedy token alone can hide large errors in the rest of the logits. */
    float maximum=0;
    double error=0,scale=0;
    int32_t top_want=0,top_got=0;
    for (int32_t i=0;i<count;i++) {
        check(isfinite(want[i]) && isfinite(got[i]));
        float diff=want[i]-got[i];if (fabsf(diff)>maximum) maximum=fabsf(diff);
        error+=(double)diff*diff;scale+=(double)want[i]*want[i];
        if (want[i]>want[top_want]) top_want=i;
        if (got[i]>got[top_got]) top_got=i;
    }
    printf("Replay logits: max=%g relative_rms=%g greedy=%d/%d\n",maximum,sqrt(error/scale),top_want,top_got);
    if (s->backend==LOINA_CPU) check(maximum<1e-5f);
    else check(maximum<1e-2f && error/scale<1e-8);
    check(top_want==top_got);
}
static uint8_t *copy_kv(llama_state *s, size_t *size) {
    llama_synchronize(s->context);
    *size=llama_state_seq_get_size(s->context,0);check(*size);
    uint8_t *data=malloc(*size);check(data);
    check(llama_state_seq_get_data(s->context,data,*size,0)==*size);
    return data;
}
static void same_kv(llama_state *s, const uint8_t *want, size_t size) {
    size_t got_size=0;uint8_t *got=copy_kv(s,&got_size);
    check(size==got_size && !memcmp(want,got,size));free(got);
}
static void checkpoint_replay(loina_engine *e, const char *directory) {
    llama_state *s=e->model;
    const struct llama_vocab *vocab=llama_model_get_vocab(s->model);
    size_t count=s->count;
    llama_token *tokens=malloc(count*sizeof(*tokens));check(tokens);
    memcpy(tokens,s->tokens,count*sizeof(*tokens));
    char path[LOINA_PATH];check(loina_path(path,directory,"checkpoint.kv",NULL));
    size_t kv_size=0;uint8_t *kv=copy_kv(s,&kv_size);
    check(llama_save(s,path));
    /* A same-length but different history must not survive the restore. */
    llama_reset(s);
    llama_token other=tokens[count/2];
    check(!decode(s,&other,1,0));
    check(llama_restore(s,path));check(s->count==count);
    same_kv(s,kv,kv_size);
    check(!memcmp(tokens,s->tokens,count*sizeof(*tokens)));
    check(llama_memory_seq_pos_max(llama_get_memory(s->context),0)==(llama_pos)count-1);
    int32_t size=llama_vocab_n_tokens(vocab);
    float *want=malloc((size_t)size*sizeof(*want));check(want);
    llama_token next=tokens[count-1];
    check(!decode(s,&next,1,(int)count));llama_synchronize(s->context);
    memcpy(want,llama_get_logits_ith(s->context,-1),(size_t)size*sizeof(*want));
    check(llama_restore(s,path));
    same_kv(s,kv,kv_size);free(kv);
    check(!decode(s,&next,1,(int)count));llama_synchronize(s->context);
    const float *restored=llama_get_logits_ith(s->context,-1);
    same_logits(s,want,restored,size);
    free(tokens);
    loina_buffer text={0};
    for (int i=0;i<64;i++) check(loina_buffer_puts(&text,"The harbor records the weather and shipping schedules. "));
    int32_t available=-llama_tokenize(vocab,text.data,(int32_t)text.length,NULL,0,true,true);
    check(available>300);
    tokens=malloc((size_t)available*sizeof(*tokens));check(tokens);
    check(llama_tokenize(vocab,text.data,(int32_t)text.length,tokens,available,true,true)==available);
    free(text.data);count=300;
    /* Compare the same prefill/decode split. Replaying all tokens as one
     * prefill changes rounding in the quantized attention calculation. */
    const size_t boundaries[]={299,256,255,128,127,4,3,0};
    for (size_t test=0;test<sizeof(boundaries)/sizeof(*boundaries);test++) {
        size_t prefix=boundaries[test];
        llama_reset(s);
        for (size_t i=0;i<count;) {
            size_t end=i<prefix ? prefix : count,n=end-i;if (n>256) n=256;
            check(!decode(s,tokens+i,(int)n,(int)i));i+=n;
        }
        memcpy(s->tokens,tokens,count*sizeof(*tokens));s->count=count;
        kv=copy_kv(s,&kv_size);
        check(llama_save(s,path));
        check(llama_restore(s,path));
        same_kv(s,kv,kv_size);free(kv);
        size_t start=prefix;
        bool rewound=llama_memory_seq_rm(llama_get_memory(s->context),0,(llama_pos)prefix,-1);
        if (!rewound) {
            llama_reset(s);start=0;
        }
        for (size_t i=start;i<count;) {
            size_t end=i<prefix ? prefix : count,n=end-i;if (n>256) n=256;
            check(!decode(s,tokens+i,(int)n,(int)i));i+=n;
        }
        llama_synchronize(s->context);
        memcpy(want,llama_get_logits_ith(s->context,-1),(size_t)size*sizeof(*want));
        llama_reset(s);
        for (size_t i=0;i<count;) {
            size_t end=i<prefix ? prefix : count,n=end-i;if (n>256) n=256;
            check(!decode(s,tokens+i,(int)n,(int)i));i+=n;
        }
        llama_synchronize(s->context);
        const float *got=llama_get_logits_ith(s->context,-1);
        same_logits(s,want,got,size);
        printf("Rewind %zu matches replay\n",prefix);
        /* Stateful models reject partial rewinds; their recovery is one
         * full replay rather than a family of cache-boundary operations. */
        if (!rewound) break;
    }
    free(want);free(tokens);llama_reset(s);
    e->cache_session[0]=e->cache_transcript[0]=0;
    remove(path);
    puts("Checkpoint restore and rewind replay passed");
}
int main(int argc, char **argv) {
    setbuf(stdout,NULL);
    check(argc==3 || argc==4);
    llama_log_set(log_message,NULL);
    const char *name=argc==4 ? argv[3] : "CPU";
    unsigned backend=!strcmp(name,"CPU") ? LOINA_CPU : !strcmp(name,"Vulkan") ? LOINA_VULKAN : !strcmp(name,"OpenCL") ? LOINA_OPENCL : 0;
    check(backend);
    const char *libs=getenv("LOINA_TEST_BACKEND_DIR");
    if (libs) ggml_backend_load_all_from_path(libs);
    else ggml_backend_load_all();
    signed_weights(name);
    loina_config config={.directory=argv[1],.context=2048,.device={.threads=2,.backends=backend}};
    loina_error error={0};
    loina_engine *e=loina_open(&config,&error);check(e);
    check(loina_set_backend(e,backend,&error)==LOINA_OK);
    const loina_package *p=loina_resolve(e,LOINA_TEXT,&error);check(p);
    check(loina_install(e,p,argv[2],NULL,&error)==LOINA_OK);
    loina_session *s=loina_session_create(e,"live",&error);check(s);
    char *first=run(e,s,"Reply with OK.",false);
    check(says(first,"OK"));
    check(((llama_state *)e->model)->backend==backend);
    printf("First turn: %s\n",first);free(first);
    loina_session *reference=copy_session(e,s,"reference");
    char *continued=run(e,s,"How are you?",false);
    printf("Second turn: %s\n",continued);
    char *cold=run(e,reference,"How are you?",false);
    check(*answer(continued));
    check(*answer(cold));
    /* Prefill and decode use different GPU kernels and can choose different
     * words. Check exact cache bytes and numerical continuation below. */
    if (backend==LOINA_CPU) check(!strcmp(continued,cold));
    printf("Independent second turn: %s\n",cold);free(cold);free(continued);
    puts("Second turn completes from live context and independent replay");
    loina_session *reloaded=loina_session_load(e,"live",&error);check(reloaded);
    char *resumed=run(e,reloaded,"Reply with READY.",false);
    char *replayed=run(e,s,"Reply with READY.",false);
    printf("Restored turn: %s\nLive reference: %s\n",resumed,replayed);
    /* A full prefill and cached decode can choose different words
     * through floating-point rounding. Their requested answer must agree. */
    check(says(resumed,"READY") && says(replayed,"READY"));free(resumed);free(replayed);
    puts("Saved conversation restores after unrelated work");
    checkpoint_replay(e,argv[1]);
    loina_session *fresh=loina_session_create(e,"fresh",&error);check(fresh);
    char *new_chat=run(e,fresh,"Reply with OK.",false);
    loina_session *independent=loina_session_create(e,"independent",&error);check(independent);
    char *new_cold=run(e,independent,"Reply with OK.",false);
    check(says(new_chat,"OK") && says(new_cold,"OK"));free(new_chat);free(new_cold);
    size_t original=s->count;
    run(e,s,"Count to ten.",true);check(s->count==original && !e->model);
    resumed=run(e,s,"Reply with RECOVERED.",false);
    loina_session *cancel_reference=copy_session(e,s,"cancel-reference");
    loina_session_truncate(cancel_reference,original);
    replayed=run(e,cancel_reference,"Reply with RECOVERED.",false);
    check(says(resumed,"RECOVERED") && says(replayed,"RECOVERED"));free(resumed);free(replayed);
    puts("New chat isolation and cancellation recovery passed");
    loina_session_free(cancel_reference);loina_session_free(independent);loina_session_free(fresh);
    loina_session_free(reloaded);loina_session_free(reference);loina_session_free(s);loina_close(e);
    puts("Real model chat regression tests passed");
}

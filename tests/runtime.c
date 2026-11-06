#include "llama.h"
#include "ggml-backend.h"
#include "jinja.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr,"runtime-test: %s\n",message);
    exit(1);
}
static double milliseconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1000000.0;
}
static void json(const char *s) {
    putchar('"');
    for (const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if (*p=='"' || *p=='\\') printf("\\%c",*p);
        else if (*p<32) printf("\\u%04x",*p);
        else putchar(*p);
    }
    putchar('"');
}
static int tokenize(const struct llama_vocab *v, const char *s, llama_token *tokens, int capacity, bool special) {
    int n=llama_tokenize(v,s,(int)strlen(s),tokens,capacity,special,true);
    if (n<1 || n>capacity) fail("Tokenization failed");
    return n;
}
static llama_token top(const float *logits, int n) {
    int best=0;
    for (int i=0;i<n;i++) {
        if (!isfinite(logits[i])) fail("Non-finite logits");
        if (logits[i]>logits[best]) best=i;
    }
    return best;
}
static void decode(struct llama_context *ctx, llama_token *tokens, int n) {
    int batch=(int)llama_n_batch(ctx);
    for (int at=0;at<n;at+=batch) {
        int count=n-at<batch ? n-at : batch;
        if (llama_decode(ctx,llama_batch_get_one(tokens+at,count))) fail("Decode failed");
    }
    llama_synchronize(ctx);
}
struct tensor_trace { FILE *file; int32_t phase; int layer; bool active; };
static bool trace_tensor(struct ggml_tensor *tensor, bool ask, void *user) {
    struct tensor_trace *trace=user;
    if (!trace->active || (tensor->type!=GGML_TYPE_F32 && tensor->type!=GGML_TYPE_F16 && tensor->type!=GGML_TYPE_Q8_0 && tensor->type!=GGML_TYPE_Q4_0) || tensor->op==GGML_OP_NONE) return false;
    const char *name=tensor->name;
    size_t length=strlen(name);
    char suffix[24];snprintf(suffix,sizeof(suffix),"-%d",trace->layer);
    size_t suffix_length=strlen(suffix);
    if (!(length>=suffix_length && !strcmp(name+length-suffix_length,suffix)) &&
        strncmp(name,"l_out",5) && strncmp(name,"result_",7) && strcmp(name,"model.embed_tokens")) return false;
    if (ask) return true;
    uint32_t header[]={length,ggml_nelements(tensor),tensor->op};
    size_t count=ggml_nelements(tensor);
    float *values=malloc(count*sizeof(float));
    unsigned char *raw=malloc(ggml_nbytes(tensor));
    if (!values || !raw) fail("Cannot allocate tensor trace");
    ggml_backend_tensor_get(tensor,raw,0,ggml_nbytes(tensor));
    size_t pos=0;
    for (int64_t i3=0;i3<tensor->ne[3];i3++) for (int64_t i2=0;i2<tensor->ne[2];i2++)
    for (int64_t i1=0;i1<tensor->ne[1];i1++) for (int64_t i0=0;i0<tensor->ne[0];i0++) {
        unsigned char *row=raw+i3*tensor->nb[3]+i2*tensor->nb[2]+i1*tensor->nb[1];
        float value;
        if (tensor->type==GGML_TYPE_F32) memcpy(&value,row+i0*tensor->nb[0],4);
        else if (tensor->type==GGML_TYPE_F16) {
            ggml_fp16_t half;memcpy(&half,row+i0*tensor->nb[0],2);value=ggml_fp16_to_fp32(half);
        } else {
            bool q4=tensor->type==GGML_TYPE_Q4_0;
            unsigned char *block=row+(i0/32)*(q4 ? 18 : 34);
            ggml_fp16_t half;memcpy(&half,block,2);
            int j=i0%32;
            int q=q4 ? ((block[2+j%16]>>(j/16*4))&15)-8 : (int8_t)block[2+j];
            value=ggml_fp16_to_fp32(half)*q;
        }
        values[pos++]=value;
    }
    free(raw);
    if (fwrite(&trace->phase,sizeof(trace->phase),1,trace->file)!=1 ||
        fwrite(header,sizeof(header),1,trace->file)!=1 ||
        fwrite(tensor->ne,sizeof(tensor->ne),1,trace->file)!=1 ||
        fwrite(name,1,length,trace->file)!=length ||
        fwrite(values,count*sizeof(float),1,trace->file)!=1) fail("Cannot write tensor trace");
    free(values);
    return true;
}
int main(int argc, char **argv) {
    if (argc!=9) {
        fprintf(stderr,"Usage: %s MODEL CPU|DEVICE LOGITS PASSES TOKENS CONTEXT THREADS LIBRARIES\n",argv[0]);
        return 2;
    }
    int passes=atoi(argv[4]),generated=atoi(argv[5]),context=atoi(argv[6]),threads=atoi(argv[7]);
    if (passes<0 || generated<1 || generated>256 || context<512 || threads<1) fail("Invalid settings");
    ggml_backend_load_all_from_path(argv[8]);
    llama_backend_init();
    bool cpu=!strcmp(argv[2],"CPU");
    ggml_backend_dev_t devices[]={cpu ? NULL : ggml_backend_dev_by_name(argv[2]),NULL};
    if (!cpu && !devices[0]) fail("Requested device unavailable");
    struct llama_model_params mp=llama_model_default_params();
    mp.devices=devices;mp.n_gpu_layers=cpu ? 0 : -1;
    mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.load_mode=cpu ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_AUTO;
    const char *cpu_weights=getenv("LOINA_TEST_CPU_WEIGHTS");
    struct llama_model_tensor_buft_override overrides[]={
        {cpu_weights,ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))},{NULL,NULL}
    };
    if (!cpu && cpu_weights && *cpu_weights) mp.tensor_buft_overrides=overrides;
    double start=milliseconds();
    struct llama_model *model=llama_model_load_from_file(argv[1],mp);
    if (!model) fail("Model load failed");
    struct llama_context_params cp=llama_context_default_params();
    cp.n_ctx=context;cp.n_batch=256;cp.n_ubatch=128;
    cp.n_threads=cp.n_threads_batch=threads;
    cp.offload_kqv=!cpu;
    cp.op_offload=!cpu;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const char *kv=getenv("LOINA_TEST_KV_OFFLOAD");
    if (kv) cp.offload_kqv=!cpu && atoi(kv)!=0;
    const char *cache=getenv("LOINA_TEST_CACHE");
    if (cache && !strcmp(cache,"f16")) cp.type_k=cp.type_v=GGML_TYPE_F16;
    const char *trace_path=getenv("LOINA_TEST_TRACE");
    const char *trace_layer=getenv("LOINA_TEST_TRACE_LAYER");
    struct tensor_trace trace={.file=trace_path ? fopen(trace_path,"wb") : NULL,.layer=trace_layer ? atoi(trace_layer) : 0};
    if (trace_path && !trace.file) fail("Cannot open tensor trace");
    if (trace.file) { cp.cb_eval=trace_tensor;cp.cb_eval_user_data=&trace; }
    struct llama_context *ctx=llama_init_from_model(model,cp);
    if (!ctx) fail("Context creation failed");
    const struct llama_vocab *v=llama_model_get_vocab(model);
    int vocab=llama_vocab_n_tokens(v);
    printf("{\"kind\":\"settings\",\"device\":");json(argv[2]);
    printf(",\"vocab\":%d,\"context\":%u,\"threads\":%d,\"batch\":256,\"ubatch\":128,\"cache_k\":\"%s\",\"cache_v\":\"%s\",\"kv_offload\":%s,\"flash_attn\":true,\"load_ms\":%.3f}\n",vocab,llama_n_ctx(ctx),threads,ggml_type_name(cp.type_k),ggml_type_name(cp.type_v),cp.offload_kqv ? "true" : "false",milliseconds()-start);
    const char *questions[]={
        "What is 2 + 2? Answer with the number only.",
        "Rispondi in italiano: descrivi il mare in una frase.",
        "List three practical ways to save water at home."
    };
    const char *repeat=getenv("LOINA_TEST_PROMPT_REPEAT");
    int repeats=repeat ? atoi(repeat) : 0;
    if (repeats<0 || repeats>512) fail("Invalid prompt repeat count");
    char *padded=NULL;
    if (repeats) {
        const char *line="The quick brown fox jumps over the lazy dog. ";
        size_t width=strlen(line),size=width*repeats+strlen(questions[0])+1;
        padded=malloc(size);
        if (!padded) fail("Cannot allocate long prompt");
        for (int i=0;i<repeats;i++) memcpy(padded+i*width,line,width);
        strcpy(padded+repeats*width,questions[0]);questions[0]=padded;
    }
    const char *continuations[]={"4 is the answer.","Il mare è blu e calmo.","Fix leaks and take shorter showers."};
    char *prompts[3];
    char error[512]="";
    const char *source=llama_model_chat_template(model,NULL);
    if (!source) source=llama_jinja_default_template();
    llama_jinja *tmpl=llama_jinja_create(source,error,sizeof(error));
    if (!tmpl) fail(error);
    for (int i=0;i<3;i++) {
        size_t capacity=strlen(questions[i])+1024;
        char *variables=malloc(capacity);
        if (!variables) fail("Cannot allocate prompt variables");
        /* BOS is supplied by the tokenizer; all three fixture strings are JSON-safe. */
        snprintf(variables,capacity,"{\"bos_token\":\"\",\"eos_token\":\"\",\"add_generation_prompt\":true,\"enable_thinking\":false,\"tools\":[],\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"%s\"}]}",questions[i]);
        prompts[i]=llama_jinja_render_chat(tmpl,variables,error,sizeof(error));
        free(variables);
        if (!prompts[i]) fail(error);
    }
    llama_jinja_free(tmpl);free(padded);
    llama_token *prompt_tokens=malloc((size_t)context*sizeof(*prompt_tokens));
    if (!prompt_tokens) fail("Cannot allocate prompt tokens");
    const char *only_case=getenv("LOINA_TEST_CASE");
    for (int pass=passes ? -1 : 0;pass<passes;pass++) for (int i=0;i<3;i++) {
        if (only_case && i!=atoi(only_case)) continue;
        llama_memory_clear(llama_get_memory(ctx),true);
        llama_token *tokens=prompt_tokens,output[256];
        int n=tokenize(v,prompts[i],tokens,context,true);
        double begin=milliseconds();decode(ctx,tokens,n);
        double prefill=milliseconds()-begin;
        begin=milliseconds();
        int count=0;
        char answer[16384]="";size_t length=0;
        for (;count<generated;count++) {
            output[count]=top(llama_get_logits_ith(ctx,-1),vocab);
            decode(ctx,&output[count],1);
        }
        double elapsed=milliseconds()-begin;
        for (int j=0;j<count;j++) {
            char piece[512];
            int size=llama_token_to_piece(v,output[j],piece,sizeof(piece),0,true);
            if (size<0 || length+(size_t)size>=sizeof(answer)) fail("Output too long");
            memcpy(answer+length,piece,(size_t)size);length+=(size_t)size;
        }
        answer[length]=0;
        int completion=count;
        for (int j=0;j<count;j++) if (llama_vocab_is_eog(v,output[j])) { completion=j+1;break; }
        printf("{\"kind\":\"trial\",\"pass\":%d,\"case\":%d,\"prompt_tokens\":%d,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"completion_tokens\":%d,\"tokens\":[",pass,i,n,prefill,elapsed,completion);
        for (int j=0;j<count;j++) printf("%s%d",j ? "," : "",output[j]);
        printf("],\"answer\":");json(answer);puts("}");fflush(stdout);
    }
    FILE *file=fopen(argv[3],"wb");
    if (!file) fail("Cannot open logits output");
    for (int i=0;i<3;i++) {
        if (only_case && i!=atoi(only_case)) continue;
        trace.active=trace.file!=NULL;trace.phase=0;
        llama_token *tokens=prompt_tokens;
        int n=tokenize(v,prompts[i],tokens,context,true);
        llama_memory_clear(llama_get_memory(ctx),true);decode(ctx,tokens,n);
        int continuation=tokenize(v,continuations[i],tokens,context,false);
        printf("{\"kind\":\"parity\",\"case\":%d,\"input_tokens\":[",i);
        for (int j=0;j<continuation;j++) printf("%s%d",j ? "," : "",tokens[j]);
        puts("]}");
        /* Feed identical tokens on every backend so divergence never changes the input. */
        for (int j=0;j<=continuation;j++) {
            const float *logits=llama_get_logits_ith(ctx,-1);
            top(logits,vocab);
            if (fwrite(logits,sizeof(float),(size_t)vocab,file)!=(size_t)vocab) fail("Cannot write logits");
            if (j<continuation) { trace.phase=j+1;decode(ctx,&tokens[j],1); }
        }
    }
    if (fclose(file)) fail("Cannot close logits");
    if (trace.file && fclose(trace.file)) fail("Cannot close trace");
    free(prompt_tokens);
    for (int i=0;i<3;i++) free(prompts[i]);
    llama_free(ctx);llama_model_free(model);llama_backend_free();
    return 0;
}

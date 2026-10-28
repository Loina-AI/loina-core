#include "loina_internal.h"
#include "parakeet.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[160];
    int64_t source, bias;
    int32_t dims, ne[4], type;
    bool gates, subtract_bias;
} speech_tensor;
typedef struct {
    FILE *file;
    struct gguf_context *gguf;
    struct ggml_context *meta;
    loina_job *job;
    speech_tensor tensors[768];
    size_t count, next, offset;
    loina_buffer pending;
    bool failed;
} speech_reader;

static bool put32(loina_buffer *b, uint32_t value) {
    unsigned char bytes[4]={value,value>>8,value>>16,value>>24};
    return loina_buffer_add(b,bytes,4);
}
static bool metadata(speech_reader *r, const char *name, uint32_t value) {
    int64_t key=gguf_find_key(r->gguf,name);
    return key>=0 && gguf_get_kv_type(r->gguf,key)==GGUF_TYPE_UINT32 && gguf_get_val_u32(r->gguf,key)==value;
}
static void *tensor_data(speech_reader *r, int64_t index) {
    if (index<0 || loina_job_cancelled(r->job)) return NULL;
    size_t size=gguf_get_tensor_size(r->gguf,index);
    void *data=malloc(size);
    uint64_t offset=gguf_get_data_offset(r->gguf)+gguf_get_tensor_offset(r->gguf,index);
    if (!data || offset>INT64_MAX || fseek(r->file,(long)offset,SEEK_SET) || fread(data,1,size,r->file)!=size) {
        free(data);return NULL;
    }
    return data;
}
static bool floats(speech_reader *r, int64_t index, float *out, int64_t count) {
    const struct ggml_tensor *t=ggml_get_tensor(r->meta,gguf_get_tensor_name(r->gguf,index));
    if (!t || ggml_nelements(t)!=count) return false;
    void *data=tensor_data(r,index);
    if (!data) return false;
    bool ok=true;
    if (t->type==GGML_TYPE_F32) memcpy(out,data,(size_t)count*sizeof(float));
    else {
        const struct ggml_type_traits *traits=ggml_get_type_traits(t->type);
        if (!traits->to_float) ok=false;
        else traits->to_float(data,out,count);
    }
    free(data);return ok;
}
static bool add_tensor(speech_reader *r, const char *name, const char *source, int dims,
                       int a, int b, int c, int d, bool force_float, bool gates) {
    if (r->count==sizeof(r->tensors)/sizeof(r->tensors[0])) return false;
    speech_tensor t={.source=-1,.bias=-1,.dims=dims,.ne={a,b,c,d},.type=GGML_TYPE_I32,.gates=gates};
    if (snprintf(t.name,sizeof(t.name),"%s",name)>=(int)sizeof(t.name)) return false;
    if (source) {
        t.source=gguf_find_tensor(r->gguf,source);
        if (t.source<0) return false;
        const struct ggml_tensor *tensor=ggml_get_tensor(r->meta,source);
        if (!tensor || ggml_nelements(tensor)!=(int64_t)a*b*c*d) return false;
        /* Conv biases need broadcast dimensions; singleton axes in GGUF
         * kernels collapse without moving any values. Quantized rows stay intact. */
        int dimension=0;
        for (int i=0;i<4;i++) if (tensor->ne[i]!=1) {
            while (dimension<dims && t.ne[dimension]==1) dimension++;
            if (dimension==dims || tensor->ne[i]!=t.ne[dimension++]) return false;
        }
        t.type=force_float ? GGML_TYPE_F32 : tensor->type;
        if (ggml_is_quantized(t.type) && tensor->ne[0]!=a) return false;
    }
    r->tensors[r->count++]=t;
    return true;
}
static bool prepare(speech_reader *r) {
    static const struct { const char *key; uint32_t value; } fields[]={
        {"parakeet.sample_rate",16000},{"parakeet.n_mels",128},{"parakeet.n_fft",512},
        {"parakeet.win_length",400},{"parakeet.hop_length",160},{"parakeet.d_model",1024},
        {"parakeet.n_layers",24},{"parakeet.n_heads",8},{"parakeet.ff_dim",4096},
        {"parakeet.subsampling_factor",8},{"parakeet.subsampling_channels",256},
        {"parakeet.conv_kernel",9},{"parakeet.pred_hidden",640},{"parakeet.pred_layers",2},
        {"parakeet.joint_hidden",640},{"parakeet.vocab_size",8192},{"parakeet.blank_id",8192},
        {"parakeet.n_tdt_durations",5}
    };
    int64_t arch=gguf_find_key(r->gguf,"general.architecture");
    if (arch<0 || gguf_get_kv_type(r->gguf,arch)!=GGUF_TYPE_STRING ||
        strcmp(gguf_get_val_str(r->gguf,arch),"parakeet")) return false;
    for (size_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++)
        if (!metadata(r,fields[i].key,fields[i].value)) return false;
    static const uint32_t header[]={GGML_FILE_MAGIC,8192,5000,1024,8,24,128,1,512,8,256,9,640,2,5,10,128,257};
    for (size_t i=0;i<sizeof(header)/sizeof(header[0]);i++) if (!put32(&r->pending,header[i])) return false;
    float *filter=malloc(128*257*sizeof(float));
    int64_t index=gguf_find_tensor(r->gguf,"preprocessor.fb");
    bool ok=filter && index>=0 && floats(r,index,filter,128*257) && loina_buffer_add(&r->pending,filter,128*257*sizeof(float));
    free(filter);
    float window[400];
    index=gguf_find_tensor(r->gguf,"preprocessor.window");
    if (!ok || index<0 || !floats(r,index,window,400) || !put32(&r->pending,400) ||
        !loina_buffer_add(&r->pending,window,sizeof(window))) return false;
    int64_t durations=gguf_find_key(r->gguf,"parakeet.tdt_durations");
    if (durations<0 || gguf_get_kv_type(r->gguf,durations)!=GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(r->gguf,durations)!=GGUF_TYPE_INT32 || gguf_get_arr_n(r->gguf,durations)!=5) return false;
    const int32_t *values=gguf_get_arr_data(r->gguf,durations);
    for (uint32_t i=0;i<5;i++) if (values[i]!=(int32_t)i || !put32(&r->pending,i)) return false;
    int64_t vocab=gguf_find_key(r->gguf,"tokenizer.ggml.tokens");
    if (vocab<0 || gguf_get_kv_type(r->gguf,vocab)!=GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(r->gguf,vocab)!=GGUF_TYPE_STRING || gguf_get_arr_n(r->gguf,vocab)!=8192 ||
        !put32(&r->pending,8192)) return false;
    for (size_t i=0;i<8192;i++) {
        const char *word=gguf_get_arr_str(r->gguf,vocab,i);
        size_t n=strlen(word);
        if (n>UINT32_MAX || !put32(&r->pending,(uint32_t)n) || !loina_buffer_add(&r->pending,word,n)) return false;
    }
    const int convs[]={0,2,3,5,6};
    char name[160],source[160];
    for (int i=0;i<5;i++) {
        snprintf(name,sizeof(name),"encoder.pre_encode.conv.%d.weight",convs[i]);
        snprintf(source,sizeof(source),"encoder.pre.conv.%d.weight",convs[i]);
        bool pointwise=convs[i]==3 || convs[i]==6;
        if (!add_tensor(r,name,source,4,pointwise ? 1 : 3,pointwise ? 1 : 3,pointwise ? 256 : 1,256,true,false)) return false;
        snprintf(name,sizeof(name),"encoder.pre_encode.conv.%d.bias",convs[i]);
        snprintf(source,sizeof(source),"encoder.pre.conv.%d.bias",convs[i]);
        if (!add_tensor(r,name,source,4,1,1,256,1,true,false)) return false;
    }
    if (!add_tensor(r,"encoder.pre_encode.out.weight","encoder.pre.out.weight",2,4096,1024,1,1,false,false) ||
        !add_tensor(r,"encoder.pre_encode.out.bias","encoder.pre.out.bias",1,1024,1,1,1,true,false)) return false;
    static const struct { const char *name,*source; int a,b; bool fp32; } layers[]={
        {"norm_feed_forward1.weight","norm_ff1.weight",1024,1,true},
        {"norm_feed_forward1.bias","norm_ff1.bias",1024,1,true},
        {"feed_forward1.linear1.weight","ff1.linear1.weight",1024,4096,false},
        {"feed_forward1.linear2.weight","ff1.linear2.weight",4096,1024,false},
        {"norm_conv.weight","norm_conv.weight",1024,1,true},
        {"norm_conv.bias","norm_conv.bias",1024,1,true},
        {"conv.pointwise_conv1.weight","conv.pw1.weight",1024,2048,false},
        {"conv.depthwise_conv.weight","conv.dw.weight",9,1024,true},
        {"conv.batch_norm.weight","conv.bn.weight",1024,1,true},
        {"conv.batch_norm.bias","conv.bn.bias",1024,1,true},
        {"conv.batch_norm.running_mean","conv.bn.running_mean",1024,1,true},
        {"conv.batch_norm.running_var","conv.bn.running_var",1024,1,true},
        {"conv.batch_norm.num_batches_tracked",NULL,1,1,false},
        {"conv.pointwise_conv2.weight","conv.pw2.weight",1024,1024,false},
        {"norm_self_att.weight","norm_attn.weight",1024,1,true},
        {"norm_self_att.bias","norm_attn.bias",1024,1,true},
        {"self_attn.pos_bias_u","attn.pos_bias_u",128,8,true},
        {"self_attn.pos_bias_v","attn.pos_bias_v",128,8,true},
        {"self_attn.linear_q.weight","attn.q.weight",1024,1024,false},
        {"self_attn.linear_k.weight","attn.k.weight",1024,1024,false},
        {"self_attn.linear_v.weight","attn.v.weight",1024,1024,false},
        {"self_attn.linear_out.weight","attn.out.weight",1024,1024,false},
        {"self_attn.linear_pos.weight","attn.pos.weight",1024,1024,false},
        {"norm_feed_forward2.weight","norm_ff2.weight",1024,1,true},
        {"norm_feed_forward2.bias","norm_ff2.bias",1024,1,true},
        {"feed_forward2.linear1.weight","ff2.linear1.weight",1024,4096,false},
        {"feed_forward2.linear2.weight","ff2.linear2.weight",4096,1024,false},
        {"norm_out.weight","norm_out.weight",1024,1,true},
        {"norm_out.bias","norm_out.bias",1024,1,true}
    };
    for (int i=0;i<24;i++) for (size_t j=0;j<sizeof(layers)/sizeof(layers[0]);j++) {
        snprintf(name,sizeof(name),"encoder.layers.%d.%s",i,layers[j].name);
        snprintf(source,sizeof(source),"encoder.layers.%d.%s",i,layers[j].source ? layers[j].source : "");
        if (!add_tensor(r,name,layers[j].source ? source : NULL,layers[j].b==1 ? 1 : 2,
            layers[j].a,layers[j].b,1,1,layers[j].fp32,false)) return false;
        if (!strcmp(layers[j].name,"conv.batch_norm.running_mean")) {
            /* Fold the depthwise bias into the following batch norm:
             * (Wx + bias - mean) becomes Wx - (mean - bias). */
            snprintf(source,sizeof(source),"encoder.layers.%d.conv.dw.bias",i);
            speech_tensor *t=&r->tensors[r->count-1];
            t->bias=gguf_find_tensor(r->gguf,source);t->subtract_bias=true;
            if (t->bias<0) return false;
        }
    }
    if (!add_tensor(r,"decoder.prediction.embed.weight","decoder.embed.weight",2,640,8193,1,1,false,false)) return false;
    for (int i=0;i<2;i++) {
        const char *parts[]={"ih","hh"};
        for (int j=0;j<2;j++) {
            snprintf(name,sizeof(name),"decoder.prediction.dec_rnn.lstm.weight_%s_l%d",parts[j],i);
            snprintf(source,sizeof(source),"decoder.lstm.%d.w_%s",i,parts[j]);
            if (!add_tensor(r,name,source,2,640,2560,1,1,false,true)) return false;
        }
        snprintf(name,sizeof(name),"decoder.prediction.dec_rnn.lstm.bias_h_l%d",i);
        snprintf(source,sizeof(source),"decoder.lstm.%d.b_hh",i);
        if (!add_tensor(r,name,source,1,2560,1,1,1,true,true)) return false;
        snprintf(source,sizeof(source),"decoder.lstm.%d.b_ih",i);
        r->tensors[r->count-1].bias=gguf_find_tensor(r->gguf,source);
        if (r->tensors[r->count-1].bias<0) return false;
    }
    const char *parts[]={"enc","pred","out"};
    for (int i=0;i<3;i++) {
        const char *legacy=i==2 ? "joint_net.2" : parts[i];
        int in=i==0 ? 1024 : 640,out=i==2 ? 8198 : 640;
        snprintf(name,sizeof(name),"joint.%s.weight",legacy);snprintf(source,sizeof(source),"joint.%s.weight",parts[i]);
        if (!add_tensor(r,name,source,2,in,out,1,1,false,false)) return false;
        snprintf(name,sizeof(name),"joint.%s.bias",legacy);snprintf(source,sizeof(source),"joint.%s.bias",parts[i]);
        if (!add_tensor(r,name,source,1,out,1,1,1,true,false)) return false;
    }
    return true;
}
static bool next_tensor(speech_reader *r) {
    if (r->next==r->count || loina_job_cancelled(r->job)) return false;
    speech_tensor *t=&r->tensors[r->next++];
    int64_t elements=(int64_t)t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3];
    size_t bytes=ggml_row_size(t->type,t->ne[0])*(size_t)(elements/t->ne[0]);
    void *data=calloc(1,bytes);
    if (!data) return false;
    bool ok=true;
    if (t->source>=0) {
        if (t->type==GGML_TYPE_F32) ok=floats(r,t->source,data,elements);
        else {
            void *raw=tensor_data(r,t->source);
            ok=raw && gguf_get_tensor_size(r->gguf,t->source)==bytes;
            if (ok) memcpy(data,raw,bytes);
            free(raw);
        }
    }
    if (ok && t->bias>=0) {
        float *bias=malloc(bytes);
        ok=bias && floats(r,t->bias,bias,elements);
        if (ok) for (int64_t i=0;i<elements;i++) ((float *)data)[i]+=t->subtract_bias ? -bias[i] : bias[i];
        free(bias);
    }
    /* NeMo stores IFCO gates. The GGML decoder expects IFOC and one
     * combined input/hidden bias. Swap complete rows without requantizing. */
    if (ok && t->gates) {
        size_t quarter=bytes/4;
        unsigned char *p=data;
        for (size_t i=0;i<quarter;i++) {
            unsigned char c=p[2*quarter+i];p[2*quarter+i]=p[3*quarter+i];p[3*quarter+i]=c;
        }
    }
    free(r->pending.data);r->pending=(loina_buffer){0};r->offset=0;
    size_t length=strlen(t->name);
    ok=ok && put32(&r->pending,t->dims) && put32(&r->pending,(uint32_t)length) && put32(&r->pending,t->type);
    for (int i=0;ok && i<t->dims;i++) ok=put32(&r->pending,t->ne[i]);
    ok=ok && loina_buffer_add(&r->pending,t->name,length) && loina_buffer_add(&r->pending,data,bytes);
    free(data);return ok;
}
static size_t speech_read(void *pointer, void *out, size_t n) {
    speech_reader *r=pointer;
    size_t read=0;
    while (read<n && !r->failed) {
        if (r->offset==r->pending.length) {
            if (r->next==r->count) break;
            if (!next_tensor(r)) { r->failed=true;break; }
        }
        size_t available=r->pending.length-r->offset;
        if (available>n-read) available=n-read;
        memcpy((char *)out+read,r->pending.data+r->offset,available);
        r->offset+=available;read+=available;
    }
    if (read<n) memset((char *)out+read,0,n-read);
    return read;
}
static bool speech_eof(void *pointer) {
    speech_reader *r=pointer;
    return r->failed || (r->next==r->count && r->offset==r->pending.length);
}
static void speech_close(void *pointer) { (void)pointer; }
static int speech_type(void *pointer, const char *name) {
    speech_reader *r=pointer;
    for (size_t i=0;i<r->count;i++) if (!strcmp(r->tensors[i].name,name)) return r->tensors[i].type;
    return -1;
}
static struct parakeet_context_params speech_settings(unsigned backends) {
    struct parakeet_context_params settings=parakeet_context_default_params();
    settings.use_gpu=false;
    int gpu=0;
    for (size_t i=0;i<ggml_backend_dev_count();i++) {
        ggml_backend_dev_t device=ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type=ggml_backend_dev_type(device);
        if (type!=GGML_BACKEND_DEVICE_TYPE_GPU && type!=GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const char *name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        /* Parakeet's OpenCL path can return only blank tokens or crash before
         * backend initialization. Keep dictation on a supported GPU or CPU. */
        unsigned bit=strstr(name,"CUDA") ? LOINA_CUDA : strstr(name,"Metal") ? LOINA_METAL :
            strstr(name,"Vulkan") ? LOINA_VULKAN : 0;
        if (bit & backends) { settings.use_gpu=true;settings.gpu_device=gpu;break; }
        gpu++;
    }
    return settings;
}
struct parakeet_context *loina_parakeet_load(const char *path, unsigned backends, loina_job *job) {
    speech_reader *r=calloc(1,sizeof(*r));
    if (!r) return NULL;
    r->job=job;r->file=fopen(path,"rb");
    uint32_t magic=0;
    if (r->file && fread(&magic,sizeof(magic),1,r->file)==1 && magic==GGML_FILE_MAGIC) {
        fclose(r->file);free(r);
        struct parakeet_context_params settings=speech_settings(backends);
        return parakeet_init_from_file_with_params(path,settings);
    }
    if (r->file) rewind(r->file);
    struct gguf_init_params params={.no_alloc=true,.ctx=&r->meta};
    if (r->file) r->gguf=gguf_init_from_file_ptr(r->file,params);
    struct parakeet_context *ctx=NULL;
    if (r->gguf && prepare(r)) {
        struct parakeet_model_loader loader={.context=r,.read=speech_read,.eof=speech_eof,
            .close=speech_close,.tensor_type=speech_type};
        struct parakeet_context_params settings=speech_settings(backends);
        ctx=parakeet_init_with_params(&loader,settings);
        if (r->failed && ctx) { parakeet_free(ctx);ctx=NULL; }
    }
    if (r->file) fclose(r->file);
    if (r->gguf) gguf_free(r->gguf);
    if (r->meta) ggml_free(r->meta);
    free(r->pending.data);free(r);
    return ctx;
}

#define loina_parakeet_load test_parakeet_load
#include "../loina_parakeet.c"
#include <math.h>

#define check(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while (0)

static struct ggml_tensor *fixture(struct ggml_context *ctx, struct gguf_context *file,
                                  const char *name, enum ggml_type type, int a, int b) {
    struct ggml_tensor *t=ggml_new_tensor_2d(ctx,type,a,b);
    ggml_set_name(t,name);gguf_add_tensor(file,t);
    return t;
}
int main(int argc, char **argv) {
    check(argc==2);
    struct ggml_context *ctx=ggml_init((struct ggml_init_params){.mem_size=65536});check(ctx);
    struct gguf_context *file=gguf_init_empty();check(file);
    struct ggml_tensor *weights=fixture(ctx,file,"decoder.lstm.0.w_ih",GGML_TYPE_F16,4,16);
    for (int i=0;i<64;i++) ((ggml_fp16_t *)weights->data)[i]=ggml_fp32_to_fp16((float)i);
    struct ggml_tensor *bias=fixture(ctx,file,"decoder.lstm.0.b_hh",GGML_TYPE_F32,16,1);
    for (int i=0;i<16;i++) ((float *)bias->data)[i]=(float)i;
    bias=fixture(ctx,file,"decoder.lstm.0.b_ih",GGML_TYPE_F16,16,1);
    for (int i=0;i<16;i++) ((ggml_fp16_t *)bias->data)[i]=ggml_fp32_to_fp16(10);
    struct ggml_tensor *quant=fixture(ctx,file,"encoder.layers.0.conv.pw1.weight",GGML_TYPE_Q8_0,32,4);
    float input[128];for (int i=0;i<128;i++) input[i]=(float)i/10;
    ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(input,quant->data,128);
    check(gguf_write_to_file(file,argv[1],false));
    speech_reader r={0};r.file=fopen(argv[1],"rb");check(r.file);
    r.gguf=gguf_init_from_file_ptr(r.file,(struct gguf_init_params){.no_alloc=true,.ctx=&r.meta});check(r.gguf);
    check(add_tensor(&r,"weights","decoder.lstm.0.w_ih",2,4,16,1,1,false,true));
    check(!add_tensor(&r,"wrong","decoder.lstm.0.w_ih",2,8,8,1,1,false,true));
    check(add_tensor(&r,"bias","decoder.lstm.0.b_hh",1,16,1,1,1,true,true));
    r.tensors[1].bias=gguf_find_tensor(r.gguf,"decoder.lstm.0.b_ih");
    check(add_tensor(&r,"quant","encoder.layers.0.conv.pw1.weight",2,32,4,1,1,false,false));
    check(add_tensor(&r,"mean","decoder.lstm.0.b_hh",1,16,1,1,1,true,false));
    r.tensors[3].bias=gguf_find_tensor(r.gguf,"decoder.lstm.0.b_ih");
    r.tensors[3].subtract_bias=true;
    check(speech_type(&r,"quant")==GGML_TYPE_Q8_0);
    check(next_tensor(&r));
    size_t start=12+8+strlen("weights");
    for (int i=0;i<64;i++) {
        ggml_fp16_t value;memcpy(&value,r.pending.data+start+i*sizeof(value),sizeof(value));
        int source=i<32 ? i : i<48 ? i+16 : i-16;
        check(ggml_fp16_to_fp32(value)==source);
    }
    check(next_tensor(&r));start=12+4+strlen("bias");
    for (int i=0;i<16;i++) {
        float value;memcpy(&value,r.pending.data+start+i*sizeof(value),sizeof(value));
        int source=i<8 ? i : i<12 ? i+4 : i-4;
        check(value==source+10);
    }
    check(next_tensor(&r));start=12+8+strlen("quant");
    check(!memcmp(r.pending.data+start,quant->data,ggml_nbytes(quant)));
    check(next_tensor(&r));start=12+4+strlen("mean");
    for (int i=0;i<16;i++) {
        float value;memcpy(&value,r.pending.data+start+i*sizeof(value),sizeof(value));
        check(value==i-10);
    }
    r.offset=r.pending.length;check(speech_eof(&r));
    unsigned char end[12];check(speech_read(&r,end,sizeof(end))==0 && !r.failed);
    r.next=0;r.offset=r.pending.length;
    check(fclose(r.file)==0);r.file=fopen(argv[1],"wb+");check(r.file);
    check(speech_read(&r,end,sizeof(end))==0 && r.failed);
    fclose(r.file);gguf_free(r.gguf);ggml_free(r.meta);free(r.pending.data);
    gguf_free(file);ggml_free(ctx);remove(argv[1]);
    puts("Speech container regression tests passed");
    return 0;
}

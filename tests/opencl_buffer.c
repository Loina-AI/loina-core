#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#define check(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1; } } while(0)
int main(int argc,char **argv) {
    check(argc==2);ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev=ggml_backend_dev_by_name("GPUOpenCL");check(dev);
    ggml_backend_buffer_type_t buft=ggml_backend_dev_buffer_type(dev);check(buft);
    size_t alignment=ggml_backend_buft_get_alignment(buft),maximum=ggml_backend_buft_get_max_size(buft);
    check(alignment && maximum>alignment);
    struct ggml_init_params p={.mem_size=2*ggml_tensor_overhead(),.no_alloc=true};
    struct ggml_context *ctx=ggml_init(p);check(ctx);
    struct ggml_tensor *tensor=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,1536,39);
    check(ggml_backend_buft_get_alloc_size(buft,tensor)>=ggml_nbytes(tensor));
    printf("OpenCL buffer queries before backend creation pass: alignment=%zu maximum=%zu\n",alignment,maximum);
    ggml_free(ctx);
}

#include "loina_internal.h"
#include "stable-diffusion.h"
#include "ggml-backend.h"
#include <stdlib.h>
#include <sys/stat.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_MAX_DIMENSIONS 4096
#include "stb_image.h"

#ifndef LOINA_DIFFUSION_REVISION
#define LOINA_DIFFUSION_REVISION "unknown"
#endif
typedef struct { sd_ctx_t *context; loina_job *job; } diffusion_state;
static unsigned diffusion_backends(void) {
#ifdef LOINA_WITH_LLAMA
    return loina_llama.backends();
#else
    ggml_backend_load_all();
    return LOINA_CPU;
#endif
}
static void *diffusion_load(loina_engine *e, const loina_package *p, loina_job *job, loina_error *error) {
    (void)job;
    char path[4][LOINA_PATH];
    if (p->file_count != 4) { loina_fail(error,LOINA_INVALID,"Sana package needs denoiser, Gemma 2, DC-AE and tokenizer");return NULL; }
    for (int i=0;i<4;i++) if (!loina_artifact_path(e,p,i,path[i])) {
        loina_fail(error,LOINA_INVALID,"Invalid image package path");return NULL;
    }
    diffusion_state *s=calloc(1,sizeof(*s));
    if (!s) { loina_fail(error,LOINA_MEMORY,"Cannot allocate diffusion state");return NULL; }
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.diffusion_model_path=path[0];params.llm_path=path[1];params.vae_path=path[2];params.tokenizer=path[3];
    params.n_threads=e->config.device.threads;
    params.enable_mmap=true;
    params.flash_attn=true;params.diffusion_flash_attn=true;
    params.params_backend="CPU";
    if (e->selected_backend==LOINA_OPENCL) params.backend="OpenCL";
    else if (e->selected_backend==LOINA_VULKAN) params.backend="Vulkan";
    else if (e->selected_backend==LOINA_METAL) params.backend="Metal";
    else if (e->selected_backend==LOINA_CUDA) params.backend="CUDA";
    else if (!(diffusion_backends() & e->config.device.backends & (LOINA_OPENCL|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA))) params.backend="CPU";
    s->context=new_sd_ctx(&params);
    if (!s->context) { free(s);loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot initialize Sana Sprint package");return NULL; }
    return s;
}
static void diffusion_unload(void *pointer) {
    diffusion_state *s=pointer;
    free_sd_ctx(s->context);free(s);
}
static void diffusion_progress(int step, int steps, float time, void *pointer) {
    (void)time;
    diffusion_state *s=pointer;
    loina_job_progress(s->job,(uint64_t)step,(uint64_t)steps);
    if (loina_job_cancelled(s->job)) sd_cancel_generation(s->context,SD_CANCEL_ALL);
}
static loina_status diffusion_image(void *pointer, loina_job *job, loina_error *error) {
    diffusion_state *s=pointer;
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt=job->prompt;params.negative_prompt="";
    params.width=job->request.width;params.height=job->request.height;
    params.sample_params.sample_steps=job->request.steps;
    params.sample_params.sample_method=SCM_SAMPLE_METHOD;
    params.sample_params.guidance.txt_cfg=1.0f;
    params.sample_params.guidance.distilled_guidance=4.5f;
    params.seed=job->request.seed==UINT32_MAX ? -1 : (int64_t)job->request.seed;
    params.strength=job->request.strength;
    params.batch_count=1;
    params.vae_tiling_params.enabled=true;
    params.vae_tiling_params.tile_size_w=256;params.vae_tiling_params.tile_size_h=256;
    if (job->input && *job->input) {
        struct stat file;
        int width,height,channels;
        if (stat(job->input,&file) || file.st_size<=0 || file.st_size>20*1024*1024 ||
            !stbi_info(job->input,&width,&height,&channels) || width>4096 || height>4096 ||
            (uint64_t)width*height>16*1024*1024)
            return loina_fail(error,LOINA_INVALID,"Choose a PNG, JPEG or BMP image up to 20 MB and 4096 pixels per side");
        params.init_image.data=stbi_load(job->input,&width,&height,&channels,3);
        if (!params.init_image.data) return loina_fail(error,LOINA_INVALID,"Cannot decode input image");
        params.init_image.width=(uint32_t)width;params.init_image.height=(uint32_t)height;params.init_image.channel=3;
    }
    s->job=job;
    sd_cancel_generation(s->context,SD_CANCEL_RESET);
    sd_set_progress_callback(diffusion_progress,s);
    sd_image_t *images=NULL;
    int count=0;
    bool ok=generate_image(s->context,&params,&images,&count);
    stbi_image_free(params.init_image.data);
    sd_set_progress_callback(NULL,NULL);s->job=NULL;
    loina_status status;
    if (loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Image generation cancelled");
    else if (!ok || count<1) status=loina_fail(error,LOINA_RUNTIME_ERROR,"Image generation failed");
    else status=loina_png(job->output,images[0].data,(int)images[0].width,(int)images[0].height,(int)images[0].channel,error);
    if (images) free_sd_images(images,count);
    return status;
}
const loina_runtime loina_diffusion={
    .name="diffusion",.revision=LOINA_DIFFUSION_REVISION,.abi=LOINA_RUNTIME_ABI,
    .backends=diffusion_backends,.load=diffusion_load,.unload=diffusion_unload,.image=diffusion_image
};

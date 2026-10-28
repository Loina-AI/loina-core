#include "loina_internal.h"
#include "stable-diffusion.h"
#include "ggml-backend.h"
#include <stdlib.h>
#include <string.h>
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
typedef struct {
    sd_ctx_t *context;
    loina_job *job;
    ggml_log_callback log;
    void *log_user;
    int progress;
    bool decoding;
    char preview[LOINA_PATH];
} diffusion_state;
static void diffusion_log(enum sd_log_level_t level, const char *text, void *pointer) {
    diffusion_state *s=pointer;
    enum ggml_log_level severity=level==SD_LOG_ERROR ? GGML_LOG_LEVEL_ERROR :
        level==SD_LOG_WARN ? GGML_LOG_LEVEL_WARN : level==SD_LOG_INFO ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_DEBUG;
    if (s->log) s->log(severity,text,s->log_user);
}
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
    ggml_log_get(&s->log,&s->log_user);
    sd_set_log_callback(diffusion_log,s);
    s->context=new_sd_ctx(&params);
    /* Chat shares GGML's logger; SD messages keep their separate callback. */
    ggml_log_set(s->log,s->log_user);
    if (!s->context) {
        sd_set_log_callback(NULL,NULL);ggml_log_set(s->log,s->log_user);free(s);
        loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot initialize Sana Sprint package");return NULL;
    }
    return s;
}
static void diffusion_unload(void *pointer) {
    diffusion_state *s=pointer;
    free_sd_ctx(s->context);
    sd_set_log_callback(NULL,NULL);ggml_log_set(s->log,s->log_user);free(s);
}
static void diffusion_progress(int step, int steps, float time, void *pointer) {
    (void)time;
    diffusion_state *s=pointer;
    if (!s->decoding && steps==s->job->request.steps && step>0) {
        int progress=10+20*step/steps;
        if (progress>s->progress) {
            s->progress=progress;
            loina_emit(s->job,LOINA_PROGRESS,"Imagining",9,progress);
        }
    }
    if (loina_job_cancelled(s->job)) sd_cancel_generation(s->context,SD_CANCEL_ALL);
}
static bool diffusion_decode(int tile, int tiles, const sd_image_t *image, void *pointer) {
    diffusion_state *s=pointer;
    s->decoding=true;
    s->progress=30+(tiles ? 69*tile/tiles : 0);
    loina_emit(s->job,LOINA_PROGRESS,"Adding detail",13,s->progress);
    if (image && *s->preview && !loina_job_cancelled(s->job) &&
        !loina_png(s->preview,image->data,(int)image->width,(int)image->height,(int)image->channel,NULL))
        loina_emit(s->job,LOINA_IMAGE_PREVIEW,s->preview,strlen(s->preview),s->progress);
    if (loina_job_cancelled(s->job)) sd_cancel_generation(s->context,SD_CANCEL_ALL);
    return !loina_job_cancelled(s->job);
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
    /* Keep blend margins while avoiding the default half-tile overlap. */
    params.vae_tiling_params.target_overlap=0.25f;
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
    s->progress=0;s->decoding=false;
    if (snprintf(s->preview,sizeof(s->preview),"%s.preview.png",job->output)>=(int)sizeof(s->preview)) s->preview[0]=0;
    sd_cancel_generation(s->context,SD_CANCEL_RESET);
    sd_set_progress_callback(diffusion_progress,s);
    sd_set_decode_callback(diffusion_decode,s);
    loina_emit(job,LOINA_PROGRESS,"Preparing image",15,0);
    sd_image_generation_result_t *images=NULL;
    bool ok=generate_image(s->context,&params,&images);
    stbi_image_free(params.init_image.data);
    sd_set_progress_callback(NULL,NULL);sd_set_decode_callback(NULL,NULL);s->job=NULL;
    if (*s->preview) remove(s->preview);
    loina_status status;
    if (loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Image generation cancelled");
    else if (!ok || !images || images->count<1) status=loina_fail(error,LOINA_RUNTIME_ERROR,"Image generation failed");
    else {
        sd_image_t *image=&images->items[0].image;
        status=loina_png(job->output,image->data,(int)image->width,(int)image->height,(int)image->channel,error);
    }
    if (images) free_sd_image_generation_result(images);
    return status;
}
const loina_runtime loina_diffusion={
    .name="diffusion",.revision=LOINA_DIFFUSION_REVISION,.abi=LOINA_RUNTIME_ABI,
    .backends=diffusion_backends,.load=diffusion_load,.unload=diffusion_unload,.image=diffusion_image
};

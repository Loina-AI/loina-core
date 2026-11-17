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
    const char *backend=NULL;
    if (e->selected_backend==LOINA_OPENCL) backend="OpenCL";
    else if (e->selected_backend==LOINA_VULKAN) backend="Vulkan";
    else if (e->selected_backend==LOINA_METAL) backend="Metal";
    else if (e->selected_backend==LOINA_CUDA) backend="CUDA";
    else if (!(diffusion_backends() & e->config.device.backends & (LOINA_OPENCL|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA))) backend="CPU";
    sd_ctx_opt_t *params=sd_ctx_opt_create();
    if (!params) { free(s);loina_fail(error,LOINA_MEMORY,"Cannot allocate image options");return NULL; }
    bool configured=sd_opt_set_string(params,"diffusion_model_path",path[0])==SD_STATUS_OK &&
        sd_opt_set_string(params,"llm_path",path[1])==SD_STATUS_OK &&
        sd_opt_set_string(params,"vae_path",path[2])==SD_STATUS_OK &&
        sd_opt_set_string(params,"tokenizer",path[3])==SD_STATUS_OK &&
        sd_opt_set_int(params,"n_threads",e->config.device.threads)==SD_STATUS_OK &&
        sd_opt_set_bool(params,"enable_mmap",true)==SD_STATUS_OK &&
        sd_opt_set_bool(params,"flash_attn",true)==SD_STATUS_OK &&
        sd_opt_set_bool(params,"diffusion_flash_attn",true)==SD_STATUS_OK &&
        sd_opt_set_string(params,"params_backend","CPU")==SD_STATUS_OK &&
        (!backend || sd_opt_set_string(params,"backend",backend)==SD_STATUS_OK);
    ggml_log_get(&s->log,&s->log_user);
    sd_set_log_callback(diffusion_log,s);
    if (configured) sd_ctx_create(params,&s->context);
    sd_opt_free(params);
    /* Chat shares GGML's logger; SD messages keep their separate callback. */
    ggml_log_set(s->log,s->log_user);
    if (!s->context) {
        sd_set_log_callback(NULL,NULL);ggml_log_set(s->log,s->log_user);free(s);
        loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot initialize Sana Sprint package: %s",sd_get_last_error());return NULL;
    }
    return s;
}
static void diffusion_unload(void *pointer) {
    diffusion_state *s=pointer;
    sd_ctx_free(s->context);
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
static loina_status diffusion_png(const char *path,const sd_tensor_t *image,loina_error *error) {
    if (!image || sd_tensor_type(image)!=SD_TENSOR_U8 || sd_tensor_rank(image)!=3)
        return loina_fail(error,LOINA_RUNTIME_ERROR,"Invalid image tensor");
    int64_t channels=sd_tensor_dim(image,0),width=sd_tensor_dim(image,1),height=sd_tensor_dim(image,2);
    if ((channels!=3 && channels!=4) || width<1 || width>4096 || height<1 || height>4096 ||
        sd_tensor_byte_size(image)!=(size_t)(channels*width*height))
        return loina_fail(error,LOINA_RUNTIME_ERROR,"Invalid image dimensions");
    return loina_png(path,sd_tensor_data(image),(int)width,(int)height,(int)channels,error);
}
static bool diffusion_decode(int tile, int tiles, const sd_tensor_t *image, void *pointer) {
    diffusion_state *s=pointer;
    s->decoding=true;
    s->progress=30+(tiles ? 69*tile/tiles : 0);
    loina_emit(s->job,LOINA_PROGRESS,"Adding detail",13,s->progress);
    if (image && *s->preview && !loina_job_cancelled(s->job) &&
        !diffusion_png(s->preview,image,NULL))
        loina_emit(s->job,LOINA_IMAGE_PREVIEW,s->preview,strlen(s->preview),s->progress);
    if (loina_job_cancelled(s->job)) sd_cancel_generation(s->context,SD_CANCEL_ALL);
    return !loina_job_cancelled(s->job);
}
static loina_status diffusion_image(void *pointer, loina_job *job, loina_error *error) {
    diffusion_state *s=pointer;
    sd_tensor_t *input=NULL;
    if (job->input && *job->input) {
        struct stat file;
        int width,height,channels;
        if (stat(job->input,&file) || file.st_size<=0 || file.st_size>20*1024*1024 ||
            !stbi_info(job->input,&width,&height,&channels) || width>4096 || height>4096 ||
            (uint64_t)width*height>16*1024*1024)
            return loina_fail(error,LOINA_INVALID,"Choose a PNG, JPEG or BMP image up to 20 MB and 4096 pixels per side");
        unsigned char *pixels=stbi_load(job->input,&width,&height,&channels,3);
        if (!pixels) return loina_fail(error,LOINA_INVALID,"Cannot decode input image");
        int64_t shape[]={3,width,height};
        input=sd_tensor_create(SD_TENSOR_U8,3,shape,pixels,(size_t)width*height*3);
        stbi_image_free(pixels);
        if (!input) return loina_fail(error,LOINA_MEMORY,"Cannot allocate input image");
    }
    sd_img_gen_opt_t *params=sd_img_gen_opt_create();
    if (!params) { sd_tensor_free(input);return loina_fail(error,LOINA_MEMORY,"Cannot allocate image options"); }
    bool configured=sd_opt_set_string(params,"prompt",job->prompt)==SD_STATUS_OK &&
        sd_opt_set_string(params,"negative_prompt","")==SD_STATUS_OK &&
        sd_opt_set_int(params,"width",job->request.width)==SD_STATUS_OK &&
        sd_opt_set_int(params,"height",job->request.height)==SD_STATUS_OK &&
        sd_opt_set_int(params,"sample.steps",job->request.steps)==SD_STATUS_OK &&
        sd_opt_set_string(params,"sample.method","scm")==SD_STATUS_OK &&
        sd_opt_set_float(params,"sample.guidance.txt_cfg",1.0)==SD_STATUS_OK &&
        sd_opt_set_float(params,"sample.guidance.distilled_guidance",4.5)==SD_STATUS_OK &&
        sd_opt_set_int(params,"seed",job->request.seed==UINT32_MAX ? -1 : (int64_t)job->request.seed)==SD_STATUS_OK &&
        sd_opt_set_float(params,"strength",job->request.strength)==SD_STATUS_OK &&
        sd_opt_set_int(params,"batch_count",1)==SD_STATUS_OK &&
        sd_opt_set_bool(params,"vae_tiling.enabled",true)==SD_STATUS_OK &&
        sd_opt_set_int(params,"vae_tiling.tile_size_w",256)==SD_STATUS_OK &&
        sd_opt_set_int(params,"vae_tiling.tile_size_h",256)==SD_STATUS_OK &&
        sd_opt_set_float(params,"vae_tiling.target_overlap",0.25)==SD_STATUS_OK &&
        (!input || sd_opt_set_tensor(params,"init_image",input)==SD_STATUS_OK);
    sd_tensor_free(input);
    if (!configured) {
        loina_status status=loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot configure image generation: %s",sd_get_last_error());
        sd_opt_free(params);return status;
    }
    s->job=job;
    s->progress=0;s->decoding=false;
    if (snprintf(s->preview,sizeof(s->preview),"%s.preview.png",job->output)>=(int)sizeof(s->preview)) s->preview[0]=0;
    sd_cancel_generation(s->context,SD_CANCEL_RESET);
    sd_set_progress_callback(diffusion_progress,s);
    sd_set_decode_callback(diffusion_decode,s);
    loina_emit(job,LOINA_PROGRESS,"Preparing image",15,0);
    sd_result_t *images=NULL;
    sd_status_t generated=sd_generate_image(s->context,params,&images);
    sd_opt_free(params);
    sd_set_progress_callback(NULL,NULL);sd_set_decode_callback(NULL,NULL);s->job=NULL;
    if (*s->preview) remove(s->preview);
    loina_status status;
    if (loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Image generation cancelled");
    else if (generated!=SD_STATUS_OK || !images || sd_result_get_image_count(images)<1) status=loina_fail(error,LOINA_RUNTIME_ERROR,"Image generation failed");
    else {
        status=diffusion_png(job->output,sd_result_get_image(images,0),error);
    }
    sd_result_free(images);
    return status;
}
const loina_runtime loina_diffusion={
    .name="diffusion",.revision=LOINA_DIFFUSION_REVISION,.abi=LOINA_RUNTIME_ABI,
    .backends=diffusion_backends,.load=diffusion_load,.unload=diffusion_unload,.image=diffusion_image
};

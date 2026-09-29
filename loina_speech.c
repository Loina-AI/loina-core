#include "loina_internal.h"
#include "parakeet.h"
#include <stdlib.h>
#include <string.h>

#ifndef LOINA_SPEECH_REVISION
#define LOINA_SPEECH_REVISION "unknown"
#endif
static unsigned speech_backends(void) {
#ifdef LOINA_WITH_LLAMA
    return loina_llama.backends();
#else
    return LOINA_CPU;
#endif
}
static void *speech_load(loina_engine *e, const loina_package *p, loina_job *job, loina_error *error) {
    (void)job;
    char path[LOINA_PATH];
    if (!loina_artifact_path(e,p,0,path)) { loina_fail(error,LOINA_INVALID,"Invalid speech package path");return NULL; }
    struct parakeet_context_params params=parakeet_context_default_params();
    params.use_gpu=(speech_backends() & e->config.device.backends & (LOINA_OPENCL|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA))!=0;
    struct parakeet_context *ctx=parakeet_init_from_file_with_params(path,params);
    if (!ctx) loina_fail(error,LOINA_RUNTIME_ERROR,"Cannot initialize Parakeet speech package");
    return ctx;
}
static void speech_unload(void *pointer) { parakeet_free(pointer); }
static bool speech_abort(void *job) { return loina_job_cancelled(job); }
static void speech_progress(struct parakeet_context *ctx, struct parakeet_state *state, int progress, void *job) {
    (void)ctx;(void)state;loina_job_progress(job,(uint64_t)progress,100);
}
static loina_status speech_run(void *ctx, loina_job *job, char **result, loina_error *error) {
    float *samples=NULL;
    int count=0;
    loina_status status=loina_wav(job->input,&samples,&count,error);
    if (status) return status;
    struct parakeet_full_params params=parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    params.n_threads=job->engine->config.device.threads;params.no_context=true;
    params.abort_callback=speech_abort;params.abort_callback_user_data=job;
    params.progress_callback=speech_progress;params.progress_callback_user_data=job;
    int rc=parakeet_full(ctx,params,samples,count);
    free(samples);
    if (loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Transcription cancelled");
    if (rc) return loina_fail(error,LOINA_RUNTIME_ERROR,"Parakeet transcription failed");
    loina_buffer text={0};
    for (int i=0;i<parakeet_full_n_segments(ctx);i++) {
        const char *segment=parakeet_full_get_segment_text(ctx,i);
        if (!loina_buffer_puts(&text,segment)) { free(text.data);return loina_fail(error,LOINA_MEMORY,"Cannot retain transcription"); }
        loina_emit(job,LOINA_TOKEN,segment,strlen(segment),0);
    }
    *result=text.data ? text.data : loina_strdup("");
    return *result ? LOINA_OK : loina_fail(error,LOINA_MEMORY,"Cannot retain transcription");
}
const loina_runtime loina_speech={
    .name="parakeet",.revision=LOINA_SPEECH_REVISION,.abi=LOINA_RUNTIME_ABI,
    .backends=speech_backends,.load=speech_load,.unload=speech_unload,.speech=speech_run
};

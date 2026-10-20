#ifndef LOINA_H
#define LOINA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOINA_ABI 3
#define LOINA_MAX_FILES 4
#define LOINA_MAX_CONTEXT 32768

typedef enum { LOINA_TEXT = 1, LOINA_VISION, LOINA_IMAGE, LOINA_SPEECH } loina_feature;
enum { LOINA_CPU = 1, LOINA_NEON = 2, LOINA_VULKAN = 4, LOINA_METAL = 8, LOINA_CUDA = 16, LOINA_OPENCL = 32 };
typedef enum {
    LOINA_OK, LOINA_INVALID, LOINA_IO, LOINA_MEMORY, LOINA_UNSUPPORTED,
    LOINA_NOT_INSTALLED, LOINA_INCOMPATIBLE, LOINA_BUSY, LOINA_CANCELLED,
    LOINA_RUNTIME_ERROR, LOINA_INTEGRITY
} loina_status;
typedef struct { loina_status code; char message[512]; } loina_error;

typedef struct {
    const char *name, *sha256, *url;
    uint64_t bytes;
} loina_artifact;
typedef struct {
    const char *id, *name, *license, *runtime;
    uint32_t runtime_abi;
    loina_feature feature;
    int quality;
    unsigned backends;
    uint64_t memory_bytes;       /* Informational estimate, not a selection gate. */
    size_t file_count;
    loina_artifact files[LOINA_MAX_FILES];
    const char *chat_template;    /* Filename in templates; NULL uses GGUF metadata. */
    bool thinking;                /* Supports explicit reasoning mode. */
} loina_package;
typedef struct {
    uint64_t ram_bytes, available_ram_bytes, storage_bytes;
    unsigned backends;
    int threads;
} loina_device;

typedef struct loina_engine loina_engine;
typedef struct loina_session loina_session;
typedef struct loina_job loina_job;
typedef enum { LOINA_SYSTEM, LOINA_USER, LOINA_ASSISTANT, LOINA_TOOL_RESULT } loina_role;
typedef enum { LOINA_MESSAGE_TEXT, LOINA_MESSAGE_IMAGE } loina_message_kind;
typedef struct {
    loina_role role;
    loina_message_kind kind;
    const char *text;
} loina_message;
typedef enum {
    LOINA_LOADING = 1, LOINA_TOKEN, LOINA_PROGRESS, LOINA_COMPACTED,
    LOINA_TOOL, LOINA_FINISHED, LOINA_METRICS, LOINA_IMAGE_PREVIEW
} loina_event;
/* LOINA_IMAGE_PREVIEW carries the path of a temporary RGBA PNG thumbnail.
 * Read it during the callback; it is replaced by later previews and removed
 * when generation finishes. Progress is monotonic and remains below 100. */
/* LOINA_METRICS carries JSON: prefill/decode in tokens per second, the backend
 * bit actually used, and context/capacity in tokens. Measurements exclude
 * model loading and KV I/O. */
/* Callbacks run on the thread executing the job. Tokens are UTF-8 byte spans;
 * they need not end on a codepoint boundary. No engine reentry from callbacks. */
typedef void (*loina_event_fn)(void *user, loina_event event,
                             const char *text, size_t bytes, int progress);
typedef loina_status (*loina_fetch_fn)(void *user, const loina_artifact *file,
                                     const char *destination, loina_job *job,
                                     loina_error *error);
/* The host owns networking and its native browser. It returns UTF-8 text
 * allocated with malloc(), which the core releases with free(). */
typedef loina_status (*loina_web_fn)(void *user, const char *url, char **text,
                                   loina_job *job, loina_error *error);
typedef struct {
    const char *directory;
    loina_device device;           /* Zero fields are probed by the core. */
    uint32_t context;              /* 0 selects 32K, always capped at 32K. */
    loina_fetch_fn fetch;
    loina_web_fn web;
    void *user;
    const char *template_directory; /* Optional overrides for bundled templates. */
} loina_config;
typedef struct {
    loina_feature feature;
    const loina_package *package; /* NULL asks the core resolver. */
    loina_session *session;
    const char *prompt;
    const char *input_path;       /* Vision/img2img image or mono 16kHz PCM16 WAV. */
    const char *output_path;      /* Image PNG output. */
    uint32_t max_tokens;
    float temperature, top_p, min_p;
    uint32_t seed;
    int width, height, steps;
    bool tools;
    bool thinking;               /* Default on; available to model adapters. */
    float strength;              /* Img2img noise strength, 0-1; default 0.75. */
} loina_request;

size_t loina_catalog_count(void);
const loina_package *loina_catalog_get(size_t index);
const loina_package *loina_catalog_find(const char *id);
loina_status loina_device_probe(const char *directory, loina_device *device, loina_error *error);
loina_engine *loina_open(const loina_config *config, loina_error *error);
void loina_close(loina_engine *engine); /* Wait for jobs before closing. */
const loina_device *loina_get_device(const loina_engine *engine);
unsigned loina_available_backends(const loina_engine *engine);
/* Zero restores automatic selection. A nonzero choice must be one available
 * backend; changing it unloads the resident model and its context. */
loina_status loina_set_backend(loina_engine *engine, unsigned backend, loina_error *error);
const loina_package *loina_resolve(loina_engine *engine, loina_feature feature, loina_error *error);
bool loina_package_installed(loina_engine *engine, const loina_package *package);
bool loina_package_compatible(loina_engine *engine, const loina_package *package);
uint64_t loina_package_bytes(const loina_package *package);
uint64_t loina_package_download_bytes(loina_engine *engine, const loina_package *package);
/* Files are staged, size/hash checked, then committed by directory rename.
 * An installed identity is never overwritten. Import uses artifact filenames. */
loina_status loina_install(loina_engine *engine, const loina_package *package,
                          const char *import_directory, loina_job *job, loina_error *error);
loina_status loina_uninstall(loina_engine *engine, const loina_package *package, loina_error *error);

loina_session *loina_session_create(loina_engine *engine, const char *id, loina_error *error);
loina_session *loina_session_load(loina_engine *engine, const char *id, loina_error *error);
void loina_session_free(loina_session *session);
const char *loina_session_id(const loina_session *session);
size_t loina_session_count(const loina_session *session);
loina_message loina_session_get(const loina_session *session, size_t index);
loina_status loina_session_save(loina_session *session, loina_error *error);
loina_status loina_session_delete(loina_engine *engine, const char *id, loina_error *error);
typedef void (*loina_session_fn)(void *user, const char *id);
loina_status loina_session_list(loina_engine *engine, loina_session_fn callback, void *user, loina_error *error);

loina_request loina_request_default(loina_feature feature);
loina_job *loina_job_create(loina_engine *engine, const loina_request *request,
                          loina_event_fn event, void *user, loina_error *error);
loina_status loina_job_run(loina_job *job, loina_error *error);
void loina_job_cancel(loina_job *job); /* Thread safe; job must remain alive. */
bool loina_job_cancelled(const loina_job *job);
void loina_job_progress(loina_job *job, uint64_t done, uint64_t total);
const char *loina_job_result(const loina_job *job); /* Owned by the job. */
void loina_job_free(loina_job *job);
loina_status loina_preload(loina_engine *engine, loina_feature feature, loina_job *job, loina_error *error);
loina_status loina_web_search(loina_engine *engine, const char *query, char **text, loina_job *job, loina_error *error);
loina_status loina_web_visit(loina_engine *engine, const char *url, char **text, loina_job *job, loina_error *error);
void loina_free(void *pointer);

#ifdef __cplusplus
}
#endif
#endif

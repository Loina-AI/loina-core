#ifndef LOINA_INTERNAL_H
#define LOINA_INTERNAL_H
#include "loina.h"
#include <stdatomic.h>
#include <stdio.h>

#define LOINA_PATH 4096
#define LOINA_MAX_TEXT (16u * 1024u * 1024u)
#define LOINA_RUNTIME_ABI 1

typedef struct {
    const char *name, *revision;
    uint32_t abi;
    unsigned (*backends)(void);
    void *(*load)(loina_engine *, const loina_package *, loina_job *, loina_error *);
    void (*unload)(void *);
    loina_status (*chat)(void *, loina_session *, loina_job *, char **, loina_error *);
    loina_status (*image)(void *, loina_job *, loina_error *);
    loina_status (*speech)(void *, loina_job *, char **, loina_error *);
    bool (*save)(void *, const char *);
    bool (*restore)(void *, const char *);
    void (*reset)(void *);
} loina_runtime;
struct loina_engine {
    loina_config config;
    char directory[LOINA_PATH];
    char model_directory[LOINA_PATH], session_directory[LOINA_PATH], cache_directory[LOINA_PATH];
    char template_directory[LOINA_PATH], chat_identity[65];
    const loina_runtime *runtime;
    const loina_package *package;
    void *model;
    unsigned available_backends, selected_backend;
    char cache_session[128], cache_transcript[65];
    atomic_bool busy;
};
struct loina_session {
    loina_engine *engine;
    bool temporary, manual_save;
    char id[128];
    char package_id[256];
    loina_message *messages;
    size_t count, capacity, first;
};
struct loina_job {
    loina_engine *engine;
    loina_request request;
    loina_event_fn event;
    void *user;
    char *prompt, *input, *output, *result, *tool_schema;
    atomic_bool cancelled;
    struct loina_job *parent;
    bool ran;
};
typedef struct { char *data; size_t length, capacity; } loina_buffer;
typedef struct { uint32_t h[8]; uint64_t bytes; unsigned char block[64]; size_t used; } loina_sha;

loina_status loina_fail(loina_error *, loina_status, const char *, ...);
bool loina_name_valid(const char *);
char *loina_strdup(const char *);
bool loina_buffer_add(loina_buffer *, const void *, size_t);
bool loina_buffer_puts(loina_buffer *, const char *);
bool loina_buffer_json(loina_buffer *, const char *);
void loina_emit(loina_job *, loina_event, const char *, size_t, int);
bool loina_path(char *, const char *, const char *, const char *);
bool loina_mkdir(const char *);
bool loina_replace(const char *, const char *);
uint64_t loina_file_bytes(const char *);
loina_status loina_copy(const char *, const char *, loina_job *, uint64_t, loina_error *);
void loina_sha_init(loina_sha *);
void loina_sha_update(loina_sha *, const void *, size_t);
void loina_sha_final(loina_sha *, char hex[65]);
loina_status loina_hash_file(const char *, char hex[65], loina_job *, loina_error *);
bool loina_package_valid(const loina_package *);
void loina_package_identity(const loina_package *, char hex[65]);
loina_status loina_packages_upgrade(loina_engine *, loina_error *);
const loina_runtime *loina_runtime_find(const char *);
unsigned loina_runtime_backends(void);
loina_status loina_activate(loina_engine *, const loina_package *, loina_job *, loina_error *);
void loina_unload(loina_engine *);
bool loina_artifact_path(loina_engine *, const loina_package *, size_t, char path[LOINA_PATH]);
bool loina_session_add(loina_session *, loina_role, loina_message_kind, const char *);
void loina_session_truncate(loina_session *, size_t);
bool loina_chat_compact(loina_session *, loina_job *);
char *loina_template_read(loina_engine *, const char *, loina_error *);
char *loina_template_model(loina_engine *, const char *, const char *, loina_error *);
loina_status loina_chat(loina_job *, loina_error *);
bool loina_cache_restore(loina_engine *, loina_session *);
bool loina_cache_save(loina_engine *, loina_session *);
loina_status loina_png(const char *, const unsigned char *, int, int, int, loina_error *);
loina_status loina_wav(const char *, float **, int *, loina_error *);

#ifdef LOINA_WITH_LLAMA
extern const loina_runtime loina_llama;
#endif
#ifdef LOINA_WITH_DIFFUSION
extern const loina_runtime loina_diffusion;
#endif
#ifdef LOINA_WITH_SPEECH
extern const loina_runtime loina_speech;
struct parakeet_context *loina_parakeet_load(const char *, unsigned, loina_job *);
#endif
#ifdef LOINA_TEST
extern const loina_runtime loina_test_runtime;
#endif
#endif

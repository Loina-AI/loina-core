#ifndef LOINA_SERVER_H
#define LOINA_SERVER_H

typedef struct {
    const char *model, *projector, *host, *key, *cache;
    int port, threads, context;
    unsigned backend;
    int rpc;
} loina_server_config;

#ifdef __cplusplus
extern "C" {
#endif
int loina_server_run(const loina_server_config *);
int loina_server_main(int, char **);
const char *loina_server_device(unsigned);
int loina_server_rpc(const loina_server_config *);
int loina_llama_server(int, char **);
#ifdef __cplusplus
}
#endif
#endif

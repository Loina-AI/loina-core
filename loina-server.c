#include "loina_server.h"
#include <stdio.h>
#include <string.h>

int loina_server_main(int argc, char **argv) { return loina_llama_server(argc,argv); }

int loina_server_run(const loina_server_config *c) {
    if (!c || !c->host || c->port<1024 || c->port>65535 || c->threads<1 ||
        c->context<512 || c->context>32768) return 1;
    if (c->rpc) {
        if (strcmp(c->host,"127.0.0.1")) return 1;
        return loina_server_rpc(c);
    }
    if (!c->model || !*c->model || !c->key || strlen(c->key)<32) return 1;
    char port[16],threads[16],context[16];
    snprintf(port,sizeof(port),"%d",c->port);
    snprintf(threads,sizeof(threads),"%d",c->threads);
    snprintf(context,sizeof(context),"%d",c->context);
    const char *device=loina_server_device(c->backend);
    const char *args[]={"loina-server","--model",c->model,"--host",c->host,"--port",port,
        "--api-key",c->key,"--alias","loina-phone","--threads",threads,"--ctx-size",context,
        "--parallel","1","--jinja","--no-ui","--no-warmup","--load-mode","none",
        "--cors-origins","*","--cors-headers","Content-Type,Authorization",
        "--device",device,"--gpu-layers",strcmp(device,"none") ? "99" : "0",
        "--cache-type-k","q8_0","--cache-type-v","q4_0",
        "--flash-attn","on",NULL,NULL,NULL};
    int count=0;while(args[count])count++;
    if (c->projector && *c->projector) { args[count++]="--mmproj";args[count++]=c->projector; }
    else args[count++]="--no-mmproj";
    return loina_server_main(count,(char **)args);
}

#ifndef LOINA_SERVER_EMBEDDED
int main(int argc, char **argv) { return loina_server_main(argc,argv); }
#endif

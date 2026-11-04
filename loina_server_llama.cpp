#include "loina_server.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <cstdio>
#include <cstring>

int llama_server(int, char **);

extern "C" int loina_llama_server(int argc, char **argv) { return llama_server(argc,argv); }

static ggml_backend_dev_t server_device(unsigned backend) {
    const unsigned order[]={32,4,1};
    for (unsigned bit:order) {
        if (backend && backend!=bit && !(bit==1 && backend==2)) continue;
        for (size_t i=0;i<ggml_backend_dev_count();i++) {
            auto dev=ggml_backend_dev_get(i);
            const char *name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            if ((bit==32 && strstr(name,"OpenCL")) || (bit==4 && strstr(name,"Vulkan")) ||
                (bit==1 && ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU)) return dev;
        }
    }
    return nullptr;
}

extern "C" const char *loina_server_device(unsigned backend) {
    auto dev=server_device(backend);
    return !dev || ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU ? "none" : ggml_backend_dev_name(dev);
}

extern "C" int loina_server_rpc(const loina_server_config *c) {
    /* RPC recreates tensor metadata; OpenCL's packed weight metadata is local. */
    auto dev=server_device(!c->backend || c->backend==32 ? 4 : c->backend);
    if (!dev) dev=server_device(1);
    auto reg=ggml_backend_reg_by_name("RPC");
    if (!dev || !reg) return 1;
    auto start=(decltype(ggml_backend_rpc_start_server) *)ggml_backend_reg_get_proc_address(reg,"ggml_backend_rpc_start_server");
    if (!start) return 1;
    char address[64];snprintf(address,sizeof(address),"127.0.0.1:%d",c->port);
    start(address,c->cache,c->threads,1,&dev);
    return 0;
}

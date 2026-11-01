#include "loina_storage.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t interrupted;
static void stop(int signal) { (void)signal; interrupted=1; }
bool loina_job_cancelled(const loina_job *job) { (void)job; return false; }

int main(int argc, char **argv) {
    char path[4096];
    if (argc==2 && !strcmp(argv[1],"cache")) {
        if (!loina_storage_directory(path,sizeof(path))) return 1;
        puts(path); return 0;
    }
    if (argc==2 && !strcmp(argv[1],"models")) {
        if (!loina_storage_models(path,sizeof(path),argv[0])) return 1;
        puts(path); return 0;
    }
    if (argc<3) return 2;
    if (!strcmp(argv[1],"cancel")) { signal(SIGALRM,stop); alarm(1); }
    loina_artifact file={.url="https://example.invalid/model",.bytes=3,.name="model.gguf"};
    loina_error error={0};
    loina_status status=loina_storage_fetch((void *)&interrupted,&file,argv[2],NULL,&error);
    if (status) fprintf(stderr,"%s\n",error.message);
    return status;
}

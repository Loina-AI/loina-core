#include "loina_storage.h"
#include <errno.h>
#include <pwd.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

bool loina_storage_directory(char *path, size_t size) {
    const char *home=getenv("HOME");
    if (!home || home[0]!='/') {
        struct passwd *pw=getpwuid(getuid());
        home=pw ? pw->pw_dir : NULL;
    }
    if (!home || home[0]!='/') return false;
    int n=snprintf(path,size,"%s/.loina/kvcache",home);
    return n>=0 && (size_t)n<size;
}

bool loina_storage_models(char *path, size_t size, const char *program) {
    const char *override=getenv("LOINA_GGUF_DIR");
    char base[4096];
    if (override && *override) {
        if (*override=='/') return snprintf(path,size,"%s",override)<(int)size;
        if (!getcwd(base,sizeof(base))) return false;
        return snprintf(path,size,"%s/%s",base,override)<(int)size;
    }
    ssize_t n=readlink("/proc/self/exe",base,sizeof(base)-1);
    if (n>=0) base[n]=0;
    else if (!realpath(program,base)) return false;
    char *slash=strrchr(base,'/');
    if (!slash) return false;
    *slash=0;
    int length=snprintf(path,size,"%s/gguf",base);
    return length>=0 && (size_t)length<size;
}

static loina_status fetch_error(loina_error *error, loina_status code, const char *message) {
    if (error) { error->code=code; snprintf(error->message,sizeof(error->message),"%s",message); }
    return code;
}

extern char **environ;

/* Download into the core's staging file. The core checks size and SHA-256
 * before publishing a package, and partial files remain resumable. */
loina_status loina_storage_fetch(void *user, const loina_artifact *file, const char *path,
                                 loina_job *job, loina_error *error) {
    volatile sig_atomic_t *interrupted=user;
    if (!file || !file->url || strncmp(file->url,"https://",8))
        return fetch_error(error,LOINA_INVALID,"Model download requires an HTTPS URL");
    char *args[]={"curl","--fail","--location","--retry","3","--connect-timeout","15",
                  "--continue-at","-","--progress-bar","--output",(char *)path,"--",(char *)file->url,NULL};
    posix_spawnattr_t attr;
    int rc=posix_spawnattr_init(&attr);
    if (rc) return fetch_error(error,LOINA_IO,strerror(rc));
    rc=posix_spawnattr_setpgroup(&attr,0);
    if (!rc) rc=posix_spawnattr_setflags(&attr,POSIX_SPAWN_SETPGROUP);
    pid_t pid;
    if (!rc) rc=posix_spawnp(&pid,"curl",NULL,&attr,args,environ);
    posix_spawnattr_destroy(&attr);
    if (rc) return fetch_error(error,LOINA_IO,"Cannot start curl; install curl to download models");
    int status;
    for (;;) {
        pid_t result=waitpid(pid,&status,WNOHANG);
        if (result==pid) break;
        if (result<0 && errno!=EINTR) return fetch_error(error,LOINA_IO,"Cannot wait for model download");
        if ((interrupted && *interrupted) || loina_job_cancelled(job)) {
            kill(-pid,SIGKILL);
            while (waitpid(pid,&status,0)<0 && errno==EINTR) {}
            return fetch_error(error,LOINA_CANCELLED,"Model download cancelled");
        }
        struct timespec pause={0,100000000};
        nanosleep(&pause,NULL);
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status))
        return fetch_error(error,LOINA_IO,"Model download failed; rerun the command to resume");
    return LOINA_OK;
}

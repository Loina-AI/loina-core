#define main loina_agent_main
#include "../loina_agent.c"
#undef main
#include "loina_terminal.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdatomic.h>
/* This fixture exercises the UI without loading an inference engine. */
static atomic_bool backend_cancelled;
void loina_job_cancel(loina_job *job) { (void)job; atomic_store(&backend_cancelled,true); }
static size_t messages;
size_t loina_session_count(const loina_session *session) { (void)session; return messages; }
const char *loina_session_id(const loina_session *session) { (void)session; return "fixture"; }
loina_status loina_session_checkpoint(loina_session *session, loina_error *error) {
    (void)session;
    if (getenv("LOINA_TEST_SAVE_FAIL")) {
        strcpy(error->message,"cannot write KV fixture"); return LOINA_IO;
    }
    puts("CHECKPOINT_WRITTEN"); return LOINA_OK;
}
static volatile sig_atomic_t interrupted;
static int terminal_fd;
static int task(void *unused) {
    interrupted=0;
    messages+=2;
    if (fcntl(terminal_fd,F_GETFL) & O_NONBLOCK) { puts("TTY_NONBLOCK"); fflush(stdout); }
    if (!strcmp(unused, "prefill")) {
        atomic_store(&backend_cancelled,false);
        loina_terminal_job((loina_job *)&backend_cancelled);
        for (int i=0; i<100 && !atomic_load(&backend_cancelled); i++) {
            struct timespec pause={0,100000000}; nanosleep(&pause,NULL);
        }
        loina_terminal_job(NULL);
        if (atomic_load(&backend_cancelled)) puts("BACKEND_CANCELLED_BEFORE_TOKEN");
        return 0;
    }
    if (!strcmp(unused, "burst")) {
        for (int i=0; i<1200; i++) fprintf(stderr,"live tool output %04d\n",i);
        fprintf(stderr,"LIVE_BEFORE_CANCEL\n");
        for (int i=0; i<100 && !interrupted; i++) {
            struct timespec pause={0,100000000}; nanosleep(&pause,NULL);
        }
        if (interrupted) fprintf(stderr,"CANCELLED_WORKER\n");
        return 0;
    }
    if (!strcmp(unused, "think-tool")) {
        stream_t stream={.use_color=true, .last_nl=true};
        const char *thinking="<think>thinking text</think>";
        s_write(&stream,thinking,strlen(thinking),false);
        loina_terminal_event(LOINA_TOOL,"preparing tool call",0);
        struct timespec pause={0,500000000}; nanosleep(&pause,NULL);
        const char *prefix="<tool_call><function=write><parameter=path>mini_redis.py</parameter><parameter=content>\nCODE_VISIBLE\nif 1 < 2:\n    print('LIVE')";
        for (const char *p=prefix; *p; p++) s_write(&stream,p,1,false);
        for (int i=0; i<100 && !interrupted; i++) {
            struct timespec delay={0,100000000}; nanosleep(&delay,NULL);
        }
        s_write(&stream,NULL,0,true);
        loina_renderer_free(stream.renderer);
        if (interrupted) fprintf(stderr,"CANCELLED_WORKER\n");
        return 0;
    }
    for (int i=0; i<12 && !interrupted; i++) {
        struct timespec delay={0,100000000};
        loina_terminal_event(LOINA_METRICS,"{\"context\":321,\"generated\":3,\"decode\":12.5}",0);
        loina_terminal_event(LOINA_TOKEN,"",0);
        printf("output %d\n",i); fflush(stdout);
        nanosleep(&delay,NULL);
    }
    return 0;
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    setvbuf(stderr,NULL,_IONBF,0);
    terminal_fd=dup(STDOUT_FILENO);
    bool thinking=true;
    size_t saved_count=0;
    do {
        if (!loina_terminal_open("Agent",32768,&thinking,&interrupted)) return 2;
        char line[16384];
        while (loina_terminal_read(line,sizeof(line))) {
            if (!strcmp(line,"/quit")) break;
            if (!strcmp(line,"/save")) { agent_save_session(NULL,&saved_count); continue; }
            loina_terminal_run(task,line);
        }
        loina_terminal_close();
    } while (getenv("LOINA_TEST_EXIT_SAVE") && !agent_maybe_save_before_exiting(NULL,&saved_count,true));
    close(terminal_fd);
    return 0;
}

#ifndef LOINA_TERMINAL_H
#define LOINA_TERMINAL_H

#include "loina.h"
#include <signal.h>

#define LOINA_ACCENT "\x1b[38;2;177;161;245m"

bool loina_terminal_open(const char *mode, uint32_t context, bool *thinking,
                         volatile sig_atomic_t *interrupt);
void loina_terminal_close(void);
bool loina_terminal_active(void);
void loina_terminal_job(loina_job *);
char *loina_terminal_read(char *line, size_t size);
int loina_terminal_run(int (*run)(void *), void *user);
void loina_terminal_event(loina_event event, const char *text, int progress);
bool loina_terminal_command(loina_session *session, const char *command);

#endif

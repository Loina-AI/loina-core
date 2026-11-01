#ifndef LOINA_BASH_H
#define LOINA_BASH_H
#include <signal.h>
#include <stdbool.h>
char *loina_bash_run(const char *, long, long, volatile sig_atomic_t *);
char *loina_bash_status(int, int, long, bool, volatile sig_atomic_t *);
void loina_bash_cleanup(void);
#endif

#ifndef LOINA_STORAGE_H
#define LOINA_STORAGE_H
#include "loina.h"

bool loina_storage_directory(char *, size_t);
bool loina_storage_models(char *, size_t, const char *);
loina_status loina_storage_fetch(void *, const loina_artifact *, const char *, loina_job *, loina_error *);
#endif

#include "loina_internal.h"
#include "loina_templates.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

char *loina_template_model(loina_engine *e, const char *name, const char *fallback, loina_error *error) {
    if (name) {
        loina_error detail={0};
        char *source=loina_template_read(e,name,&detail);
        if (source) return source;
        if (detail.code!=LOINA_UNSUPPORTED) { if (error) *error=detail;return NULL; }
    }
    char *source=fallback ? loina_strdup(fallback) : NULL;
    if (!source) loina_fail(error,fallback ? LOINA_MEMORY : LOINA_UNSUPPORTED,"Cannot retain model chat template");
    return source;
}

char *loina_template_read(loina_engine *e, const char *name, loina_error *error) {
    if (!loina_name_valid(name)) { loina_fail(error,LOINA_INVALID,"Invalid template filename");return NULL; }
    if (e->template_directory[0]) {
        char path[LOINA_PATH];
        if (!loina_path(path,e->template_directory,name,NULL)) { loina_fail(error,LOINA_INVALID,"Template path is too long");return NULL; }
        FILE *f=fopen(path,"rb");
        if (f) {
            loina_buffer b={0};
            char data[4096];
            size_t n;
            while ((n=fread(data,1,sizeof(data),f))) {
                if (!loina_buffer_add(&b,data,n)) { fclose(f);free(b.data);loina_fail(error,LOINA_MEMORY,"Template is too large: %s",name);return NULL; }
            }
            bool ok=!ferror(f);
            fclose(f);
            if (ok && b.length && strlen(b.data)==b.length) return b.data;
            free(b.data);loina_fail(error,LOINA_IO,"Cannot read template: %s",name);return NULL;
        }
        if (errno!=ENOENT) { loina_fail(error,LOINA_IO,"Cannot open template: %s",name);return NULL; }
    }
    for (size_t i=0;i<sizeof(loina_templates)/sizeof(*loina_templates);i++) {
        if (!strcmp(name,loina_templates[i].name)) {
            char *source=loina_strdup(loina_templates[i].source);
            if (!source) loina_fail(error,LOINA_MEMORY,"Cannot retain template: %s",name);
            return source;
        }
    }
    loina_fail(error,LOINA_UNSUPPORTED,"Template is unavailable: %s",name);
    return NULL;
}

#ifndef LOINA_RENDERER_H
#define LOINA_RENDERER_H
#include <stdbool.h>
#include <stddef.h>
typedef struct loina_renderer loina_renderer;
loina_renderer *loina_renderer_new(bool color);
void loina_renderer_context(loina_renderer *, unsigned);
void loina_renderer_feed(loina_renderer *, const char *, size_t);
void loina_renderer_finish(loina_renderer *);
void loina_renderer_free(loina_renderer *);
bool loina_renderer_last_nl(loina_renderer *);
void loina_renderer_tool_start(loina_renderer *);
void loina_renderer_tool_name(loina_renderer *, const char *);
void loina_renderer_param_start(loina_renderer *, const char *);
void loina_renderer_param_byte(loina_renderer *, char);
void loina_renderer_param_end(loina_renderer *);
void loina_renderer_tool_end(loina_renderer *);
char *loina_renderer_safe_text(const char *, size_t);
#endif

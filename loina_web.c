#include "loina_internal.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

loina_status loina_web_visit(loina_engine *e, const char *url, char **text, loina_job *job, loina_error *error) {
    if (text) *text=NULL;
    if (!e || !text || !url || strncmp(url,"https://",8) || !url[8] || strlen(url)>8192)
        return loina_fail(error,LOINA_INVALID,"Web tools require an HTTPS URL");
    const char *end=strchr(url+8,'/');
    size_t host=end ? (size_t)(end-(url+8)) : strlen(url+8);
    if (!host || memchr(url+8,'@',host) || memchr(url+8,'\\',host)) return loina_fail(error,LOINA_INVALID,"Invalid web URL");
    for (const unsigned char *p=(const unsigned char *)url;*p;p++)
        if (*p<=32 || *p==127) return loina_fail(error,LOINA_INVALID,"Invalid character in web URL");
    if (!e->config.web) return loina_fail(error,LOINA_UNSUPPORTED,"No native browser bridge is configured");
    if (loina_job_cancelled(job)) return loina_fail(error,LOINA_CANCELLED,"Web request cancelled");
    loina_emit(job,LOINA_TOOL,url,strlen(url),0);
    loina_status status=e->config.web(e->config.user,url,text,job,error);
    if (!status && (!*text || strlen(*text)>1024*1024)) status=loina_fail(error,LOINA_INVALID,"Browser returned an invalid or oversized document");
    if (loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Web request cancelled");
    if (status) { free(*text);*text=NULL; }
    return status;
}
loina_status loina_web_search(loina_engine *e, const char *query, char **text, loina_job *job, loina_error *error) {
    if (!query || !*query || strlen(query)>2048) return loina_fail(error,LOINA_INVALID,"Invalid search query");
    loina_buffer url={0};
    bool ok=loina_buffer_puts(&url,"https://html.duckduckgo.com/html/?q=");
    for (const unsigned char *p=(const unsigned char *)query;ok && *p;p++) {
        char buf[4];
        if (isalnum(*p) && *p<128) { buf[0]=(char)*p;buf[1]=0; }
        else snprintf(buf,sizeof(buf),"%%%02X",*p);
        ok=loina_buffer_puts(&url,buf);
    }
    if (!ok) { free(url.data);return loina_fail(error,LOINA_MEMORY,"Cannot encode search URL"); }
    loina_status status=loina_web_visit(e,url.data,text,job,error);
    free(url.data);
    return status;
}

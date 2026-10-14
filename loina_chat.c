#include "loina_internal.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

bool loina_chat_compact(loina_session *s, loina_job *job) {
    size_t first=s->first;
    if (first<s->count && s->messages[first].role==LOINA_SYSTEM) first++;
    size_t next=first+1;
    while (next<s->count && s->messages[next].role!=LOINA_USER) next++;
    if (next>=s->count) return false;
    /* Only complete old turns leave the active prefix. The full transcript
     * remains on disk; removing half a turn would corrupt role alternation. */
    s->first=next;
    loina_emit(job,LOINA_COMPACTED,"Earlier turns saved outside the active context",46,0);
    return true;
}
static void extraction_event(void *user, loina_event event, const char *text, size_t n, int progress) {
    if (event!=LOINA_TOKEN && event!=LOINA_FINISHED) loina_emit(user,event,text,n,progress);
}
static const char *skip_space(const char *s) {
    while (isspace((unsigned char)*s)) s++;
    return s;
}
static bool take(const char **s, const char *text) {
    const char *start=skip_space(*s);
    size_t n=strlen(text);
    if (strncmp(start,text,n)) return false;
    *s=start+n;
    return true;
}
static char *tool_string(const char **s) {
    const char *p=skip_space(*s);
    char quote=*p++;
    if (quote!='\"' && quote!='\'') return NULL;
    loina_buffer b={0};
    while (*p && *p!=quote) {
        char c=*p++;
        if (c=='\\') {
            c=*p++;
            if (c=='n') c='\n';
            else if (c=='r') c='\r';
            else if (c=='t') c='\t';
            else if (c!='\\' && c!='\"' && c!='\'') goto fail;
        } else if ((unsigned char)c<32) goto fail;
        if (b.length>=8192 || !loina_buffer_add(&b,&c,1)) goto fail;
    }
    if (*p!=quote || !b.length) goto fail;
    *s=p+1;
    return b.data;
fail:
    free(b.data);
    return NULL;
}
typedef struct { bool search; char *argument; } tool_call;
static bool tool_like(const char *reply) {
    reply=skip_space(reply);
    const char *marker="<|tool_call_start|>";
    size_t n=strlen(reply),prefix=strlen(marker);
    return n && !strncmp(reply,marker,n<prefix ? n : prefix);
}
static int tool_requests(const char *reply, tool_call calls[4]) {
    int count=0;
    if (!take(&reply,"<|tool_call_start|>") || !take(&reply,"[")) return 0;
    do {
        if (count==4) goto fail;
        bool search=take(&reply,"web_search");
        if (!search && !take(&reply,"web_visit")) goto fail;
        if (!take(&reply,"(") || !take(&reply,search ? "query" : "url") || !take(&reply,"=")) goto fail;
        char *argument=tool_string(&reply);
        if (!argument) goto fail;
        calls[count++]=(tool_call){search,argument};
        if (!take(&reply,")")) goto fail;
    } while (take(&reply,","));
    if (!take(&reply,"]") || !take(&reply,"<|tool_call_end|>")) goto fail;
    return count;
fail:
    for (int i=0;i<count;i++) { free(calls[i].argument);calls[i].argument=NULL; }
    return 0;
}
typedef struct {
    loina_event_fn event;
    void *user;
    loina_buffer pending;
    bool visible, hidden, failed;
} chat_stream;
static void chat_event(void *user, loina_event event, const char *text, size_t n, int progress) {
    chat_stream *s=user;
    if (!s->event) return;
    if (event!=LOINA_TOKEN || s->visible) { s->event(s->user,event,text,n,progress);return; }
    if (s->hidden || s->failed) return;
    if (!loina_buffer_add(&s->pending,text,n)) { s->failed=true;return; }
    const char *answer=skip_space(s->pending.data);
    if (!*answer) return;
    const char *marker="<|tool_call_start|>";
    size_t have=strlen(answer),prefix=strlen(marker);
    if (have<prefix && !strncmp(answer,marker,have)) return;
    if (!strncmp(answer,marker,prefix)) { s->hidden=true;return; }
    s->visible=true;
    s->event(s->user,LOINA_TOKEN,s->pending.data,s->pending.length,0);
    free(s->pending.data);s->pending=(loina_buffer){0};
}
loina_status loina_chat(loina_job *job, loina_error *error) {
    loina_engine *e=job->engine;
    loina_request *r=&job->request;
    bool tools=r->tools;
    loina_session *s=r->session;
    bool temporary=!s;
    if (!s) s=loina_session_create(e,"transient",error);
    if (!s) return error ? error->code : LOINA_MEMORY;
    size_t original=s->count,old_first=s->first;
    char old_package[256];strcpy(old_package,s->package_id);
    char *augmented=NULL,*reply=NULL;
    const char *prompt=job->prompt;
    loina_status status=LOINA_OK;
    if (r->feature==LOINA_VISION && !temporary) {
        loina_request vr=*r;
        vr.session=NULL;vr.tools=false;
        vr.prompt="Describe this image and extract its visible text faithfully. Include details needed to answer questions about it.";
        loina_job *vision=loina_job_create(e,&vr,extraction_event,job,error);
        if (!vision) { status=error ? error->code : LOINA_MEMORY;goto done; }
        vision->parent=job;
        status=loina_chat(vision,error);
        if (!status) {
            loina_buffer b={0};
            bool ok=loina_buffer_puts(&b,prompt) && loina_buffer_puts(&b,"\n\n[Attached image]\n") && loina_buffer_puts(&b,loina_job_result(vision));
            if (ok) augmented=b.data;
            else { free(b.data);status=loina_fail(error,LOINA_MEMORY,"Cannot retain image context"); }
        }
        loina_job_free(vision);
        if (status) goto done;
        prompt=augmented;
    }
    loina_feature feature=r->feature==LOINA_VISION && temporary ? LOINA_VISION : LOINA_TEXT;
    const loina_package *p=NULL;
    if (r->package && r->package->feature==feature) p=r->package;
    else if (s->package_id[0]) p=loina_catalog_find(s->package_id);
    else p=loina_resolve(e,feature,error);
    if (!p) { status=loina_fail(error,LOINA_INCOMPATIBLE,"The original chat package is unavailable; start a new conversation to change models");goto done; }
    if (p->feature!=feature || (s->package_id[0] && strcmp(s->package_id,p->id))) {
        status=loina_fail(error,LOINA_INCOMPATIBLE,"Conversation belongs to another immutable package");goto done;
    }
    status=loina_activate(e,p,job,error);
    if (status) goto done;
    if (!e->runtime->chat) { status=loina_fail(error,LOINA_UNSUPPORTED,"Runtime does not implement chat");goto done; }
    if (!s->package_id[0]) strcpy(s->package_id,p->id);
    if (!temporary) loina_cache_restore(e,s);
    if (!loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,prompt)) { status=loina_fail(error,LOINA_MEMORY,"Cannot append message");goto done; }
    /* Keep useful evidence through errors and reserve a final inference pass
     * without tools. Repeated navigation must not replace the answer. */
    int requests=0,searches=0,failures=0;
    for (int round=0;round<6;round++) {
        if (round==5) r->tools=false;
        chat_stream stream={.event=job->event,.user=job->user};
        job->event=chat_event;job->user=&stream;
        status=e->runtime->chat(e->model,s,job,&reply,error);
        job->event=stream.event;job->user=stream.user;
        if (!stream.hidden && !stream.failed && stream.pending.length && !tool_like(stream.pending.data))
            loina_emit(job,LOINA_TOKEN,stream.pending.data,stream.pending.length,0);
        free(stream.pending.data);
        if (stream.failed && !status) status=loina_fail(error,LOINA_MEMORY,"Cannot buffer chat output");
        if (status) break;
        tool_call calls[4]={0};
        int count=tool_requests(reply,calls);
        bool malformed=!count && tool_like(reply);
        bool unfinished=!*skip_space(reply);
        if (!count && !malformed && !unfinished) break;
        if (!r->tools) {
            for (int i=0;i<count;i++) free(calls[i].argument);
            status=loina_fail(error,LOINA_RUNTIME_ERROR,"The model did not finish an answer");break;
        }
        loina_buffer results={0};
        loina_emit(job,LOINA_TOOL,"",0,0);
        if (!count) {
            const char *message=malformed ?
                "Tool error: Invalid function format. Use <|tool_call_start|>[web_search(query=\"search terms\")]<|tool_call_end|> or <|tool_call_start|>[web_visit(url=\"https://url\")]<|tool_call_end|>." :
                "Tool error: The response budget ended. Answer directly using the available evidence.";
            if (!loina_buffer_puts(&results,message)) status=LOINA_MEMORY;
            if (unfinished || ++failures>=2) r->tools=false;
        }
        for (int i=0;i<count && !status;i++) {
            char *document=NULL;
            if (requests>=4 || (calls[i].search && searches>=2)) {
                document=loina_strdup("Tool error: No more web requests are available. Answer from the sources already returned.");
                r->tools=false;
            } else {
                requests++;if (calls[i].search) searches++;
                status=calls[i].search ? loina_web_search(e,calls[i].argument,&document,job,error) :
                    loina_web_visit(e,calls[i].argument,&document,job,error);
            }
            if (status && status!=LOINA_CANCELLED && status!=LOINA_MEMORY) {
                free(document);
                document=loina_strdup("Tool error: This web request failed. Use earlier sources or try one different search. Do not invent current facts.");
                status=LOINA_OK;
                if (error) memset(error,0,sizeof(*error));
            }
            if (!status && !document) status=LOINA_MEMORY;
            if (!status) {
                if (!strncmp(document,"Tool error:",11) && ++failures>=2) r->tools=false;
                if ((i && !loina_buffer_puts(&results,"\n\n")) || !loina_buffer_puts(&results,document)) status=LOINA_MEMORY;
            }
            free(document);
        }
        for (int i=0;i<count;i++) free(calls[i].argument);
        if (!status && (!loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,reply) ||
            !loina_session_add(s,LOINA_TOOL_RESULT,LOINA_MESSAGE_TEXT,results.data))) status=LOINA_MEMORY;
        free(results.data);free(reply);reply=NULL;
        if (status) { if (status==LOINA_MEMORY) loina_fail(error,status,"Cannot retain web result");break; }
    }
    if (!status && loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Chat cancelled");
    if (!status && !loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,reply ? reply : "")) status=loina_fail(error,LOINA_MEMORY,"Cannot retain reply");
    if (!status && !temporary) {
        status=loina_session_save(s,error);
        if (!status && feature==LOINA_TEXT) loina_cache_save(e,s);
    }
    if (!status) { job->result=reply;reply=NULL; }
done:
    r->tools=tools;
    if (status) {
        loina_session_truncate(s,original);s->first=old_first;strcpy(s->package_id,old_package);
        e->cache_session[0]=e->cache_transcript[0]=0;
        if (e->model && e->runtime->reset) e->runtime->reset(e->model);
    }
    free(reply);free(augmented);
    if (temporary) loina_session_free(s);
    return status;
}

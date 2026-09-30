#include "loina_internal.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

char *loina_chat_prompt_minicpm5(loina_session *s, const char *system) {
    loina_buffer b={0};
    if (!loina_buffer_puts(&b,"<|im_start|>system\n") ||
        !loina_buffer_puts(&b,system) || !loina_buffer_puts(&b,"<|im_end|>\n")) goto fail;
    for (size_t i=s->first;i<s->count;i++) {
        const loina_message *m=&s->messages[i];
        const char *role=m->role==LOINA_USER ? "user" : m->role==LOINA_ASSISTANT ? "assistant" : "system";
        const char *content=m->kind==LOINA_MESSAGE_IMAGE ? "[Generated image]" : m->text;
        if (!loina_buffer_puts(&b,"<|im_start|>") || !loina_buffer_puts(&b,role) ||
            !loina_buffer_puts(&b,"\n")) goto fail;
        if (m->role==LOINA_ASSISTANT) {
            const char *end=strstr(content,"</think>");
            if (end) {
                const char *start=strstr(content,"<think>");
                start=start && start<end ? start+7 : content;
                while (start<end && *start=='\n') start++;
                while (end>start && end[-1]=='\n') end--;
                if (!loina_buffer_puts(&b,"<think>\n") ||
                    !loina_buffer_add(&b,start,(size_t)(end-start)) ||
                    !loina_buffer_puts(&b,"\n</think>\n\n")) goto fail;
                content=strstr(content,"</think>")+8;
                while (*content=='\n') content++;
            } else if (!strstr(content,"<think>")) {
                if (!loina_buffer_puts(&b,"<think>\n\n</think>\n\n")) goto fail;
                while (*content=='\n') content++;
            }
        }
        if (!loina_buffer_puts(&b,content) || !loina_buffer_puts(&b,"<|im_end|>\n")) goto fail;
    }
    if (!loina_buffer_puts(&b,"<|im_start|>assistant\n")) goto fail;
    return b.data;
fail:
    free(b.data);
    return NULL;
}

bool loina_chat_compact(loina_session *s, loina_job *job) {
    size_t first=s->first;
    if (first<s->count && s->messages[first].role==LOINA_SYSTEM) first++;
    size_t next=first+1;
    while (next<s->count && (s->messages[next].role!=LOINA_USER ||
        !strncmp(s->messages[next].text,"<tool_response>",15))) next++;
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
static bool tool_request(const char *reply, const char *name, const char *param, char **argument) {
    char open[64],field[64];
    snprintf(open,sizeof(open),"<function name=\"%s\">",name);
    snprintf(field,sizeof(field),"<param name=\"%s\">",param);
    while (isspace((unsigned char)*reply)) reply++;
    if (!strncmp(reply,"<think>",7)) {
        const char *end=strstr(reply+7,"</think>");
        if (!end) return false;
        reply=end+8;
        while (isspace((unsigned char)*reply)) reply++;
    }
    size_t n=strlen(open);
    if (strncmp(reply,open,n)) return false;
    reply+=n;
    while (isspace((unsigned char)*reply)) reply++;
    n=strlen(field);
    if (strncmp(reply,field,n)) return false;
    reply+=n;
    bool cdata=!strncmp(reply,"<![CDATA[",9);
    if (cdata) reply+=9;
    const char *end=strstr(reply,cdata ? "]]></param>" : "</param>");
    if (!end || end-reply>8192 || end==reply) return false;
    const char *tail=end+(cdata ? 11 : 8);
    while (isspace((unsigned char)*tail)) tail++;
    if (strncmp(tail,"</function>",11)) return false;
    tail+=11;
    while (isspace((unsigned char)*tail)) tail++;
    if (*tail) return false;
    loina_buffer b={0};
    while (reply<end) {
        const char *entity=NULL;
        char value=0;
        if (!cdata && *reply=='&') {
            if (end-reply>=5 && !strncmp(reply,"&amp;",5)) { entity=reply+5;value='&'; }
            else if (end-reply>=4 && !strncmp(reply,"&lt;",4)) { entity=reply+4;value='<'; }
            else if (end-reply>=4 && !strncmp(reply,"&gt;",4)) { entity=reply+4;value='>'; }
            else if (end-reply>=6 && !strncmp(reply,"&quot;",6)) { entity=reply+6;value='"'; }
            else if (end-reply>=6 && !strncmp(reply,"&apos;",6)) { entity=reply+6;value='\''; }
        }
        if (!loina_buffer_add(&b,entity ? &value : reply,1)) { free(b.data);return false; }
        reply=entity ? entity : reply+1;
    }
    *argument=b.data;
    return true;
}
static loina_status web_fallback(char **reply, loina_error *error) {
    free(*reply);
    *reply=loina_strdup("I couldn't complete the web search. Please try again.");
    if (!*reply) return loina_fail(error,LOINA_MEMORY,"Cannot retain reply");
    if (error) memset(error,0,sizeof(*error));
    return LOINA_OK;
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
    const char *answer=s->pending.data;
    while (isspace((unsigned char)*answer)) answer++;
    if (!strncmp(answer,"<think>",7)) {
        const char *end=strstr(answer+7,"</think>");
        if (!end) return;
        answer=end+8;
        while (isspace((unsigned char)*answer)) answer++;
    }
    if (!*answer) return;
    const char *prefixes[]={"<think>","<function", "<web_search>","<web_visit>","name=\"",NULL};
    for (int i=0;prefixes[i];i++) {
        size_t have=strlen(answer),prefix=strlen(prefixes[i]);
        if (have<prefix && !strncmp(answer,prefixes[i],have)) return;
        if (i && have>=prefix && !strncmp(answer,prefixes[i],prefix)) { s->hidden=true;return; }
    }
    /* Classify the response before releasing its preamble. A tool's XML and
     * reasoning must never flash on screen while it is being recognized. */
    s->visible=true;
    s->event(s->user,LOINA_TOKEN,s->pending.data,s->pending.length,0);
    free(s->pending.data);s->pending=(loina_buffer){0};
}
loina_status loina_chat(loina_job *job, loina_error *error) {
    loina_engine *e=job->engine;
    loina_request *r=&job->request;
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
    /* Native browser tools use a small model-independent envelope. Limit the
     * loop so a model cannot spend an unbounded time navigating the web. */
    for (int round=0;round<5;round++) {
        chat_stream stream={.event=job->event,.user=job->user};
        if (r->tools) { job->event=chat_event;job->user=&stream; }
        status=e->runtime->chat(e->model,s,job,&reply,error);
        if (r->tools) { job->event=stream.event;job->user=stream.user; }
        free(stream.pending.data);
        if (stream.failed && !status) status=loina_fail(error,LOINA_MEMORY,"Cannot buffer chat output");
        if (status) break;
        char *argument=NULL,*document=NULL;
        bool search=r->tools && tool_request(reply,"web_search","query",&argument);
        bool visit=!search && r->tools && tool_request(reply,"web_visit","url",&argument);
        if (!search && !visit) {
            if (r->tools && (strstr(reply,"name=\"web_search\"") || strstr(reply,"name=\"web_visit\"")))
                status=web_fallback(&reply,error);
            break;
        }
        if (round==4) { free(argument);status=web_fallback(&reply,error);break; }
        status=search ? loina_web_search(e,argument,&document,job,error) : loina_web_visit(e,argument,&document,job,error);
        free(argument);
        if (status || !strncmp(document,"Tool error:",11)) {
            free(document);
            if (status==LOINA_CANCELLED || status==LOINA_MEMORY) break;
            status=web_fallback(&reply,error);break;
        }
        loina_buffer b={0};
        bool ok=loina_buffer_puts(&b,"<tool_response>\n") && loina_buffer_puts(&b,document) &&
            loina_buffer_puts(&b,"\n</tool_response>");
        free(document);
        if (ok) ok=loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,reply) && loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,b.data);
        free(b.data);free(reply);reply=NULL;
        if (!ok) { status=loina_fail(error,LOINA_MEMORY,"Cannot retain web result");break; }
    }
    if (!status && loina_job_cancelled(job)) status=loina_fail(error,LOINA_CANCELLED,"Chat cancelled");
    if (!status && !loina_session_add(s,LOINA_ASSISTANT,LOINA_MESSAGE_TEXT,reply ? reply : "")) status=loina_fail(error,LOINA_MEMORY,"Cannot retain reply");
    if (!status && !temporary) {
        status=loina_session_save(s,error);
        if (!status && feature==LOINA_TEXT) loina_cache_save(e,s);
    }
    if (!status) { job->result=reply;reply=NULL; }
done:
    if (status) {
        loina_session_truncate(s,original);s->first=old_first;strcpy(s->package_id,old_package);
        e->cache_session[0]=e->cache_transcript[0]=0;
        if (e->model && e->runtime->reset) e->runtime->reset(e->model);
    }
    free(reply);free(augmented);
    if (temporary) loina_session_free(s);
    return status;
}

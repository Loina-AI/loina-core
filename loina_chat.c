#include "loina_internal.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

char *loina_chat_prompt_minicpm5(loina_session *s, const char *system, bool thinking) {
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
    if (!loina_buffer_puts(&b,thinking ? "<think>\n" : "<think>\n\n</think>\n\n")) goto fail;
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
static const char *chat_answer(const char *reply) {
    while (isspace((unsigned char)*reply)) reply++;
    if (!strncmp(reply,"<think>",7)) {
        const char *end=strstr(reply+7,"</think>");
        if (!end) return NULL;
        reply=end+8;
        while (isspace((unsigned char)*reply)) reply++;
    }
    return reply;
}
static bool tool_like(const char *answer) {
    if (!answer || !*answer) return false;
    const char *prefixes[]={"<function","<web_search>","<web_visit>","<tool_call>","name=\"",NULL};
    size_t have=strlen(answer);
    for (int i=0;prefixes[i];i++) {
        size_t n=strlen(prefixes[i]);
        if (have>=2 && !strncmp(answer,prefixes[i],have<n ? have : n)) return true;
    }
    return false;
}
static bool tool_request(const char *reply, const char *name, const char *param, char **argument, const char **next) {
    char open[64],field[64];
    snprintf(open,sizeof(open),"<function name=\"%s\">",name);
    snprintf(field,sizeof(field),"<param name=\"%s\">",param);
    reply=chat_answer(reply);
    if (!reply) return false;
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
    if (*tail && strncmp(tail,"<function ",10)) return false;
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
    *next=tail;
    return true;
}
typedef struct {
    loina_event_fn event;
    void *user;
    loina_buffer pending;
    size_t sent;
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
        /* Keep only the possible closing tag unshown. Reasoning is visible
         * while it is generated; tool XML still waits for classification. */
        size_t ready=end ? (size_t)(end+8-s->pending.data) : s->pending.length-7;
        if (ready>s->sent) {
            s->event(s->user,LOINA_TOKEN,s->pending.data+s->sent,ready-s->sent,0);
            s->sent=ready;
        }
        if (!end) return;
        answer=end+8;
        while (isspace((unsigned char)*answer)) answer++;
    }
    if (!*answer) return;
    const char *prefixes[]={"<think>","<function", "<web_search>","<web_visit>","<tool_call>","name=\"",NULL};
    for (int i=0;prefixes[i];i++) {
        size_t have=strlen(answer),prefix=strlen(prefixes[i]);
        if (have<prefix && !strncmp(answer,prefixes[i],have)) return;
        if (i && have>=prefix && !strncmp(answer,prefixes[i],prefix)) { s->hidden=true;return; }
    }
    s->visible=true;
    s->event(s->user,LOINA_TOKEN,s->pending.data+s->sent,s->pending.length-s->sent,0);
    free(s->pending.data);s->pending=(loina_buffer){0};
}
loina_status loina_chat(loina_job *job, loina_error *error) {
    loina_engine *e=job->engine;
    loina_request *r=&job->request;
    bool tools=r->tools,thinking=r->thinking;
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
        vr.session=NULL;vr.tools=false;vr.thinking=false;
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
    int calls=0,searches=0,failures=0;
    for (int round=0;round<6;round++) {
        if (round==5) { r->tools=false;r->thinking=false; }
        chat_stream stream={.event=job->event,.user=job->user};
        job->event=chat_event;job->user=&stream;
        status=e->runtime->chat(e->model,s,job,&reply,error);
        job->event=stream.event;job->user=stream.user;
        if (!stream.hidden && !stream.failed && stream.pending.length>stream.sent &&
            !tool_like(chat_answer(stream.pending.data)))
            loina_emit(job,LOINA_TOKEN,stream.pending.data+stream.sent,stream.pending.length-stream.sent,0);
        free(stream.pending.data);
        if (stream.failed && !status) status=loina_fail(error,LOINA_MEMORY,"Cannot buffer chat output");
        if (status) break;
        char *argument=NULL,*document=NULL;
        const char *next=NULL;
        bool search=tool_request(reply,"web_search","query",&argument,&next);
        bool visit=!search && tool_request(reply,"web_visit","url",&argument,&next);
        bool deferred=next && *next;
        /* MiniCPM can emit several calls despite a one-call instruction. Run
         * the first and let it inspect three sources before requesting more.
         * The transcript records only the call whose result follows it. */
        if (deferred) reply[next-reply]=0;
        const char *answer=chat_answer(reply);
        bool malformed=!search && !visit && tool_like(answer);
        bool unfinished=!answer || !*answer;
        if (!search && !visit) {
            if (!malformed && !unfinished) break;
        }
        if (round==5 || (!r->tools && !unfinished)) {
            free(argument);status=loina_fail(error,LOINA_RUNTIME_ERROR,"The model did not finish an answer; try with thinking disabled");break;
        }
        if (unfinished) {
            if (strstr(reply,"<think>") && !strstr(reply,"</think>")) {
                loina_buffer completed={0};
                if (!loina_buffer_puts(&completed,reply) || !loina_buffer_puts(&completed,"\n</think>")) {
                    free(completed.data);status=loina_fail(error,LOINA_MEMORY,"Cannot retain reasoning");break;
                }
                free(reply);reply=completed.data;
            }
            document=loina_strdup("Tool error: The response budget ended. Answer the user's question directly using the available evidence. Do not call tools.");
            r->tools=false;r->thinking=false;
            loina_emit(job,LOINA_TOOL,"",0,0);
        } else if (malformed) {
            document=loina_strdup("Tool error: Invalid function format. Use <function name=\"web_search\"><param name=\"query\">query</param></function> or answer from the available evidence.");
            if (++failures>=2) { r->tools=false;r->thinking=false; }
            loina_emit(job,LOINA_TOOL,"",0,0);
        } else if (calls>=4 || (search && searches>=2)) {
            document=loina_strdup("Tool error: No more web requests are available. Answer from the sources already returned, and explain any information you could not verify.");
            r->tools=false;r->thinking=false;
            loina_emit(job,LOINA_TOOL,"",0,0);
        } else {
            calls++;if (search) searches++;
            status=search ? loina_web_search(e,argument,&document,job,error) : loina_web_visit(e,argument,&document,job,error);
        }
        free(argument);
        if (status==LOINA_CANCELLED || status==LOINA_MEMORY) { free(document);break; }
        if (status) {
            free(document);
            document=loina_strdup("Tool error: This web request failed. Use the sources already returned, or try one different search. Do not invent current facts.");
            status=LOINA_OK;
            if (error) memset(error,0,sizeof(*error));
        }
        if (!document) { status=loina_fail(error,LOINA_MEMORY,"Cannot retain web result");break; }
        if (!strncmp(document,"Tool error:",11) && !malformed && !unfinished && ++failures>=2) { r->tools=false;r->thinking=false; }
        loina_buffer b={0};
        bool ok=loina_buffer_puts(&b,"<tool_response>\n") && loina_buffer_puts(&b,document) &&
            (!deferred || loina_buffer_puts(&b,"\n\nOnly the first requested function was executed. Inspect these results before requesting another function.")) &&
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
    r->tools=tools;r->thinking=thinking;
    if (status) {
        loina_session_truncate(s,original);s->first=old_first;strcpy(s->package_id,old_package);
        e->cache_session[0]=e->cache_transcript[0]=0;
        if (e->model && e->runtime->reset) e->runtime->reset(e->model);
    }
    free(reply);free(augmented);
    if (temporary) loina_session_free(s);
    return status;
}

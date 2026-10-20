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
typedef struct { bool search; char *argument; } tool_call;
static const char *tool_prefixes[]={"<function","<web_search>","<web_visit>","<tool_call>","<|tool_call_start|>","name=\"",NULL};
static const char *tool_start(const char *reply) {
    const char *first=NULL;
    const char *markers[]={"<function","<tool_call>","<|tool_call_start|>"};
    for (size_t i=0;i<sizeof(markers)/sizeof(*markers);i++) {
        const char *call=strstr(reply,markers[i]);
        if (call && (!first || call<first)) first=call;
    }
    return first;
}
static bool tool_like(const char *reply) {
    reply=reply ? chat_answer(reply) : NULL;
    if (!reply || !*reply) return false;
    if (tool_start(reply)) return true;
    size_t have=strlen(reply);
    for (int i=0;tool_prefixes[i];i++) {
        size_t n=strlen(tool_prefixes[i]);
        if (have>=2 && !strncmp(reply,tool_prefixes[i],have<n ? have : n)) return true;
    }
    return false;
}
static bool tool_request(const char **reply, tool_call *call) {
    bool modern=take(reply,"<tool_call>");
    const char *search=modern ? "<function=web_search>" : "<function name=\"web_search\">";
    const char *visit=modern ? "<function=web_visit>" : "<function name=\"web_visit\">";
    call->search=take(reply,search);
    if (!call->search && !take(reply,visit)) return false;
    const char *field=modern ? (call->search ? "<parameter=query>" : "<parameter=url>") :
        (call->search ? "<param name=\"query\">" : "<param name=\"url\">");
    if (!take(reply,field)) return false;
    const char *start=*reply;
    bool cdata=!modern && !strncmp(start,"<![CDATA[",9);
    if (cdata) start+=9;
    const char *close=modern ? "</parameter>" : cdata ? "]]></param>" : "</param>";
    const char *end=strstr(start,close);
    if (!end || end-start>8192) return false;
    *reply=end+strlen(close);
    if (!take(reply,"</function>") || (modern && !take(reply,"</tool_call>"))) return false;
    if (modern) {
        while (start<end && isspace((unsigned char)*start)) start++;
        while (end>start && isspace((unsigned char)end[-1])) end--;
    }
    if (start==end) return false;
    loina_buffer b={0};
    while (start<end) {
        const char *entity=NULL;
        char value=0;
        if (!modern && !cdata && *start=='&') {
            if (end-start>=5 && !strncmp(start,"&amp;",5)) { entity=start+5;value='&'; }
            else if (end-start>=4 && !strncmp(start,"&lt;",4)) { entity=start+4;value='<'; }
            else if (end-start>=4 && !strncmp(start,"&gt;",4)) { entity=start+4;value='>'; }
            else if (end-start>=6 && !strncmp(start,"&quot;",6)) { entity=start+6;value='\"'; }
            else if (end-start>=6 && !strncmp(start,"&apos;",6)) { entity=start+6;value='\''; }
        }
        if ((!cdata && !modern && *start=='<') || !loina_buffer_add(&b,entity ? &value : start,1)) { free(b.data);return false; }
        start=entity ? entity : start+1;
    }
    call->argument=b.data;
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
    *s=p+1;return b.data;
fail:
    free(b.data);return NULL;
}
static int python_requests(const char *reply, tool_call calls[4]) {
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
    if (strstr(reply,"<|tool_call_start|>")) goto fail;
    return count;
fail:
    for (int i=0;i<count;i++) { free(calls[i].argument);calls[i].argument=NULL; }
    return 0;
}
static int tool_requests(const char *reply, tool_call calls[4]) {
    int count=0;
    reply=chat_answer(reply);
    if (!reply || !(reply=tool_start(reply))) return 0;
    if (!strncmp(reply,"<|tool_call_start|>",19)) return python_requests(reply,calls);
    do {
        if (count==4 || !tool_request(&reply,&calls[count])) goto fail;
        count++;
        reply=skip_space(reply);
    } while (*reply);
    return count;
fail:
    for (int i=0;i<count;i++) { free(calls[i].argument);calls[i].argument=NULL; }
    return 0;
}
typedef struct {
    loina_event_fn event;
    void *user;
    loina_buffer pending;
    size_t sent;
    bool visible, hidden, failed;
} chat_stream;
static void stream_answer(chat_stream *s) {
    const char *answer=s->pending.data+s->sent,*call=tool_start(answer);
    size_t ready=s->pending.length;
    if (call) { ready=(size_t)(call-s->pending.data);s->hidden=true; }
    else {
        /* Hold a possible XML opener across token boundaries, even after a
         * natural-language preamble allowed by the model's tool template. */
        size_t available=ready-s->sent,hold=0;
        for (int i=0;tool_prefixes[i];i++) {
            size_t n=strlen(tool_prefixes[i]);
            for (size_t j=1;j<n && j<=available;j++)
                if (j>hold && !strncmp(s->pending.data+ready-j,tool_prefixes[i],j)) hold=j;
        }
        ready-=hold;
    }
    if (ready>s->sent) s->event(s->user,LOINA_TOKEN,s->pending.data+s->sent,ready-s->sent,0);
    size_t remaining=s->hidden ? 0 : s->pending.length-ready;
    memmove(s->pending.data,s->pending.data+ready,remaining);
    s->pending.data[remaining]=0;s->pending.length=remaining;s->sent=0;
}
static void chat_event(void *user, loina_event event, const char *text, size_t n, int progress) {
    chat_stream *s=user;
    if (!s->event) return;
    if (event!=LOINA_TOKEN) { s->event(s->user,event,text,n,progress);return; }
    if (s->hidden || s->failed) return;
    if (!loina_buffer_add(&s->pending,text,n)) { s->failed=true;return; }
    if (s->visible) { stream_answer(s);return; }
    const char *answer=s->pending.data;
    while (isspace((unsigned char)*answer)) answer++;
    if (!strncmp(answer,"<think>",7)) {
        const char *end=strstr(answer+7,"</think>");
        /* Keep only the possible closing tag unshown. Reasoning is visible
         * while it is generated; tool calls still wait for classification. */
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
    const char *prefixes[]={"<think>","<function","<web_search>","<web_visit>","<tool_call>","<|tool_call_start|>","name=\"",NULL};
    for (int i=0;prefixes[i];i++) {
        size_t have=strlen(answer),prefix=strlen(prefixes[i]);
        if (have<prefix && !strncmp(answer,prefixes[i],have)) return;
        if (i && have>=prefix && !strncmp(answer,prefixes[i],prefix)) { s->hidden=true;return; }
    }
    s->visible=true;
    stream_answer(s);
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
    r->thinking=thinking && p->thinking;
    if (!e->runtime->chat) { status=loina_fail(error,LOINA_UNSUPPORTED,"Runtime does not implement chat");goto done; }
    if (!s->package_id[0]) strcpy(s->package_id,p->id);
    if (!temporary) loina_cache_restore(e,s);
    if (!loina_session_add(s,LOINA_USER,LOINA_MESSAGE_TEXT,prompt)) { status=loina_fail(error,LOINA_MEMORY,"Cannot append message");goto done; }
    /* Keep useful evidence through errors and reserve a final inference pass
     * without tools. Repeated navigation must not replace the answer. */
    int requests=0,searches=0,failures=0;
    for (int round=0;round<6;round++) {
        if (round==5) { r->tools=false;r->thinking=false; }
        chat_stream stream={.event=job->event,.user=job->user};
        job->event=chat_event;job->user=&stream;
        status=e->runtime->chat(e->model,s,job,&reply,error);
        job->event=stream.event;job->user=stream.user;
        if (!stream.hidden && !stream.failed && stream.pending.length>stream.sent && !tool_like(stream.pending.data))
            loina_emit(job,LOINA_TOKEN,stream.pending.data+stream.sent,stream.pending.length-stream.sent,0);
        free(stream.pending.data);
        if (stream.failed && !status) status=loina_fail(error,LOINA_MEMORY,"Cannot buffer chat output");
        if (status) break;
        tool_call calls[4]={0};
        int count=tool_requests(reply,calls);
        bool malformed=!count && tool_like(reply);
        const char *answer=chat_answer(reply);
        bool unfinished=!answer || !*answer;
        if (!count && !malformed && !unfinished) break;
        if (round==5) {
            for (int i=0;i<count;i++) free(calls[i].argument);
            status=loina_fail(error,LOINA_RUNTIME_ERROR,"The model did not finish an answer; try with thinking disabled");break;
        }
        if (unfinished && strstr(reply,"<think>") && !strstr(reply,"</think>")) {
            loina_buffer completed={0};
            if (!loina_buffer_puts(&completed,reply) || !loina_buffer_puts(&completed,"\n</think>")) {
                free(completed.data);status=loina_fail(error,LOINA_MEMORY,"Cannot retain reasoning");break;
            }
            free(reply);reply=completed.data;
        }
        loina_buffer results={0};
        loina_emit(job,LOINA_TOOL,"",0,0);
        if (!count) {
            const char *message=malformed ?
                "Tool error: Invalid function format. Use the call format specified by the model template, with no extra parameters. Only web_search(query) and web_visit(url) are available." :
                "Tool error: The response budget ended. Answer the user's question directly using the available evidence. Do not call tools.";
            if (!loina_buffer_puts(&results,message)) status=LOINA_MEMORY;
            if (unfinished || ++failures>=2) { r->tools=false;r->thinking=false; }
        }
        for (int i=0;i<count && !status;i++) {
            char *document=NULL;
            if (!r->tools || requests>=4 || (calls[i].search && searches>=2)) {
                document=loina_strdup("Tool error: Web access is closed for this turn. No request was sent.");
                r->tools=false;r->thinking=false;
            } else {
                requests++;if (calls[i].search) searches++;
                status=calls[i].search ? loina_web_search(e,calls[i].argument,&document,job,error) :
                    loina_web_visit(e,calls[i].argument,&document,job,error);
            }
            if (status && status!=LOINA_CANCELLED && status!=LOINA_MEMORY) {
                free(document);
                /* Tool results report facts; the system prompt controls retries. */
                document=loina_strdup("Tool error: This web request failed. No facts were returned.");
                status=LOINA_OK;
                if (error) memset(error,0,sizeof(*error));
            }
            if (!status && !document) status=LOINA_MEMORY;
            if (!status) {
                if (!strncmp(document,"Tool error:",11) && ++failures>=2) { r->tools=false;r->thinking=false; }
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

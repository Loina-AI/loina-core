#include "loina_internal.h"
#include "jinja.h"
#include <stdlib.h>
#include <string.h>

#define check(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while (0)

static char *render(llama_jinja *tmpl, const char *tools, bool thinking, bool history, const char *question) {
    loina_buffer b={0};
    check(loina_buffer_puts(&b,"{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\",\"add_generation_prompt\":true,\"enable_thinking\":"));
    check(loina_buffer_puts(&b,thinking ? "true" : "false"));
    check(loina_buffer_puts(&b,",\"tools\":") && loina_buffer_puts(&b,tools));
    check(loina_buffer_puts(&b,",\"messages\":[{\"role\":\"system\",\"content\":\"Local assistant.\"}"));
    if (history) check(loina_buffer_puts(&b,",{\"role\":\"user\",\"content\":\"An earlier question.\"},{\"role\":\"assistant\",\"content\":\"An earlier answer.\"}"));
    check(loina_buffer_puts(&b,",{\"role\":\"user\",\"content\":") && loina_buffer_json(&b,question) && loina_buffer_puts(&b,"}"));
    if (history) check(loina_buffer_puts(&b,",{\"role\":\"assistant\",\"content\":\"<think>Check sources.</think>\",\"tool_calls\":[{\"function\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"test\"}}}]},{\"role\":\"tool\",\"content\":\"Source says 42.\"}"));
    check(loina_buffer_puts(&b,"]}"));
    char error[512]="";
    char *prompt=llama_jinja_render(tmpl,b.data,error,sizeof(error));
    if (!prompt) fprintf(stderr,"%s\n",error);
    check(prompt);free(b.data);return prompt;
}
int main(int argc, char **argv) {
    check(argc==2);
    loina_engine e={0};
    loina_error error={0};
    char *tools=loina_template_read(&e,"tools.json",&error);check(tools);
    const char *names[]={"minicpm-v4.7-1b.jinja","minicpm5-2b.jinja","lfm2.5-vl-3b.jinja"};
    for (size_t i=0;i<3;i++) {
        char *source=loina_template_read(&e,names[i],&error);check(source);
        char message[512]="";
        llama_jinja *tmpl=llama_jinja_create(source,message,sizeof(message));
        if (!tmpl) fprintf(stderr,"%s\n",message);
        check(tmpl);free(source);
        char *prompt=render(tmpl,"[]",true,false,"Search today.");
        check(i==2 ? !strstr(prompt,"<think>") : !strcmp(prompt+strlen(prompt)-8,"<think>\n"));
        check(strstr(prompt,"Local assistant."));
        check(!strstr(prompt,"# Tools"));free(prompt);
        prompt=render(tmpl,tools,false,false,"Search today.");
        if (i==2) {
            check(strstr(prompt,"List of tools: [") && !strstr(prompt,"\"function\":") && !strstr(prompt,"<think>"));
        } else {
            check(!strcmp(prompt+strlen(prompt)-19,"<think>\n\n</think>\n\n"));
            check(strstr(prompt,"# Tools") && strstr(prompt,"\"type\": \"function\""));
        }
        check(strstr(prompt,"web_search"));free(prompt);
        prompt=render(tmpl,tools,true,true,"Search today.");
        if (i==2) {
            check(strstr(prompt,"<|im_start|>tool\nSource says 42.<|im_end|>"));
            check(strstr(prompt,"<|tool_call_start|>[web_search(query='test')]<|tool_call_end|>"));
        } else {
            check(strstr(prompt,"<tool_response>\nSource says 42.\n</tool_response>"));
            check(strstr(prompt,i ? "<function name=\"web_search\">" : "<function=web_search>"));
            check(strstr(prompt,"<think>\nCheck sources.\n</think>"));
        }
        free(prompt);
        const char *questions[]={"What does the source say?","Что сообщает источник?","¿Qué dice la fuente?","来源说了什么？",
            "Come gestire la nausea in gravidanza?","¿Qué dice la fuente? Answer in English."};
        for (size_t q=0;q<sizeof(questions)/sizeof(*questions);q++) for (int thinking=0;thinking<2;thinking++) {
            prompt=render(tmpl,tools,thinking,true,questions[q]);
            if (i==2) {
                const char *question=strstr(prompt,questions[q]);check(question);
                check(!strstr(question+strlen(questions[q]),questions[q]));
                check(strstr(question,"<|im_start|>tool\nSource says 42."));
            } else {
                const char *evidence=strstr(prompt,"</tool_response>");check(evidence);
                loina_buffer turn={0};
                check(loina_buffer_puts(&turn,"</tool_response><|im_end|>\n<|im_start|>user\n") && loina_buffer_puts(&turn,questions[q]) &&
                    loina_buffer_puts(&turn,"<|im_end|>\n<|im_start|>assistant\n<think>"));
                if (!strstr(evidence,turn.data)) fprintf(stderr,"Template %s: %s\n",names[i],evidence);
                check(strstr(evidence,turn.data));free(turn.data);
            }
            free(prompt);
            prompt=render(tmpl,tools,thinking,false,questions[q]);
            const char *question=strstr(prompt,questions[q]);check(question && !strstr(question+strlen(questions[q]),questions[q]));free(prompt);
        }
        check(!llama_jinja_render(tmpl,"{broken",message,sizeof(message)) && *message);
        llama_jinja_free(tmpl);
    }
    free(tools);
    char *policy=loina_template_read(&e,"system.jinja",&error);check(policy);
    llama_jinja *system=llama_jinja_create(policy,error.message,sizeof(error.message));check(system);free(policy);
    char *online=llama_jinja_render(system,"{\"date\":\"2026-10-20\",\"local_date\":\"2026-10-20 (Tuesday, UTC+0200)\",\"tools\":true}",error.message,sizeof(error.message));
    char *offline=llama_jinja_render(system,"{\"date\":\"2026-10-20\",\"tools\":false}",error.message,sizeof(error.message));
    check(online && offline && strstr(online,"2026-10-20") && strstr(offline,"2026-10-20"));
    check(strstr(online,"Tuesday, UTC+0200") && strstr(offline,"Current device date: 2026-10-20."));
    check(strstr(online,"Use evidence to answer") && strstr(online,"For a news roundup"));
    check(!strstr(offline,"Use evidence to answer") && !strstr(offline,"For a news roundup"));
    check(strstr(offline,"Use supplied evidence") && strstr(offline,"Do not call tools."));
    free(online);free(offline);llama_jinja_free(system);
    char *fallback=loina_template_model(&e,"missing-model.jinja",llama_jinja_default_template(),&error);check(fallback);
    llama_jinja *generic=llama_jinja_create(fallback,error.message,sizeof(error.message));check(generic);free(fallback);
    char *generic_prompt=render(generic,"[]",false,false,"A generic question.");
    check(!strcmp(generic_prompt,"<|im_start|>system\nLocal assistant.<|im_end|>\n<|im_start|>user\nA generic question.<|im_end|>\n<|im_start|>assistant\n"));
    free(generic_prompt);llama_jinja_free(generic);
    check(!loina_template_model(&e,"../escape.jinja",llama_jinja_default_template(),&error));
    check(!loina_template_read(&e,"../escape.jinja",&error));
    check(loina_mkdir(argv[1]));
    strcpy(e.template_directory,argv[1]);
    char path[LOINA_PATH];check(loina_path(path,argv[1],"system.jinja",NULL));
    FILE *f=fopen(path,"wb");check(f);
    check(fputs("Custom {{ date }}",f)>=0 && !fclose(f));
    char *source=loina_template_model(&e,"system.jinja",llama_jinja_default_template(),&error);check(source);
    llama_jinja *tmpl=llama_jinja_create(source,error.message,sizeof(error.message));check(tmpl);free(source);
    char *prompt=llama_jinja_render(tmpl,"{\"date\":\"2026-10-14\"}",error.message,sizeof(error.message));
    check(prompt && !strcmp(prompt,"Custom 2026-10-14"));free(prompt);llama_jinja_free(tmpl);
    source=loina_template_read(&e,"tools.json",&error);check(source);free(source);
    check(!llama_jinja_create("{% if %}",error.message,sizeof(error.message)) && *error.message);
    remove(path);
    puts("Jinja templates and overrides passed");
    return 0;
}

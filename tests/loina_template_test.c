#include "loina_internal.h"
#include "jinja.h"
#include "loina_agent_tools.h"
#include <stdlib.h>
#include <string.h>

#define check(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while (0)

static char *render_mode(llama_jinja *tmpl, const char *tools, bool thinking, bool history, const char *question, bool external) {
    loina_buffer b={0};
    check(loina_buffer_puts(&b,"{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\",\"add_generation_prompt\":true,\"enable_thinking\":"));
    check(loina_buffer_puts(&b,thinking ? "true" : "false"));
    check(loina_buffer_puts(&b,",\"external_tools\":") && loina_buffer_puts(&b,external ? "true" : "false"));
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
static char *render(llama_jinja *tmpl, const char *tools, bool thinking, bool history, const char *question) {
    return render_mode(tmpl,tools,thinking,history,question,false);
}
int main(int argc, char **argv) {
    check(argc==2);
    loina_engine e={0};
    loina_error error={0};
    char *tools=loina_template_read(&e,"tools.json",&error);check(tools);
    char *lfm3=loina_template_read(&e,"lfm3-vl-2b.jinja",&error);check(lfm3);
    llama_jinja *liquid=llama_jinja_create(lfm3,error.message,sizeof(error.message));check(liquid);free(lfm3);
    for (int thinking=0;thinking<2;thinking++) {
        char *prompt=render(liquid,tools,thinking,true,"Search today.");
        const char *suffix=thinking ? "<think>" : "<think></think>";
        check(!strcmp(prompt+strlen(prompt)-strlen(suffix),suffix));
        check(!strncmp(prompt,"<s><|im_start|>system\n",22));
        check(strstr(prompt,"<tools>\n") && strstr(prompt,"<function=web_search>"));
        check(strstr(prompt,"<|im_start|>tool\n<tool_response>\nSource says 42.\n</tool_response>"));
        check(strstr(prompt,"<think>Check sources.</think>"));
        check(strstr(prompt,"<think></think>An earlier answer."));
        check(strstr(prompt,"</tool_response><|im_end|>\n<|im_start|>user\nSearch today.<|im_end|>"));
        free(prompt);
        prompt=render(liquid,tools,thinking,true,"Что сообщает источник?");
        check(strstr(prompt,"</tool_response><|im_end|>\n<|im_start|>user\nЧто сообщает источник?<|im_end|>"));free(prompt);
        prompt=render(liquid,"[]",thinking,false,"Что сообщает источник?");
        check(!strstr(prompt,"<tools>") && strstr(prompt,"Что сообщает источник?"));
        check(!strcmp(prompt+strlen(prompt)-strlen(suffix),suffix));free(prompt);
    }
    char *agent=render(liquid,agent_tool_schema,false,false,"Write a Python program.");
    check(strstr(agent,"<tools>\n") && strstr(agent,"\"name\": \"write\"") &&
          strstr(agent,"\"name\": \"bash\"") && strstr(agent,"\"content\""));
    free(agent);
    agent=render_mode(liquid,agent_tool_schema,false,true,"Write a Python program.",true);
    check(strstr(agent,"<|im_start|>tool\n<tool_response>") &&
          !strstr(agent,"</tool_response><|im_end|>\n<|im_start|>user"));
    free(agent);
    free(tools);llama_jinja_free(liquid);
    char *policy=loina_template_read(&e,"system.jinja",&error);check(policy);
    llama_jinja *system=llama_jinja_create(policy,error.message,sizeof(error.message));check(system);free(policy);
    char *online=llama_jinja_render(system,"{\"date\":\"2026-10-20\",\"local_date\":\"2026-10-20 (Tuesday, UTC+0200)\",\"tools\":true}",error.message,sizeof(error.message));
    char *offline=llama_jinja_render(system,"{\"date\":\"2026-10-20\",\"tools\":false}",error.message,sizeof(error.message));
    check(online && offline && strstr(online,"2026-10-20") && strstr(offline,"2026-10-20"));
    check(strstr(online,"Tuesday, UTC+0200") && strstr(offline,"Current device date: 2026-10-20."));
    check(strstr(online,"Call web_search before answering") && !strstr(offline,"Call web_search before answering"));
    const char *common[]={"Keep that language for short follow-ups", "cover several distinct relevant items", "Use sources as evidence"};
    for (size_t i=0;i<sizeof(common)/sizeof(*common);i++) check(strstr(online,common[i]) && strstr(offline,common[i]));
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

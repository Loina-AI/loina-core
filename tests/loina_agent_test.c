#define main loina_agent_main
#include "../loina_agent.c"
#undef main

#define check(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static int file_contracts(void) {
    more_state_t more={.context=32768};
    size_t size=300000;
    char *data=xmalloc(size+1);
    memset(data, 'a', size); data[size-2]='b'; data[size-1]='\n'; data[size]=0;
    char *result=tool_write("long.txt", data);
    check(result && !strstr(result, "Tool error")); free(result);
    size_t consumed=0;
    result=tool_read("long.txt", 1, 1, false, true, &more);
    while (1) {
        const char *note=strstr(result, "\n[Read truncated.");
        size_t n=note ? (size_t)(note-result) : strlen(result);
        check(consumed+n<=size && !memcmp(result, data+consumed, n));
        consumed+=n; free(result);
        if (!more.more_valid) break;
        result=tool_more(1, &more);
    }
    check(consumed==size);
    result=tool_read("long.txt", 1, 1, true, true, &more);
    check(strstr(result, "whole read exceeds") && !more.more_valid); free(result); free(data);
    result=tool_write("target.py", "print('before')\n"); free(result);
    check(!chmod("target.py", 0750) && !symlink("target.py", "alias.py"));
    result=tool_edit("alias.py", "before", "after", false);
    check(result && !strstr(result, "Tool error")); free(result);
    struct stat st;
    check(!lstat("alias.py", &st) && S_ISLNK(st.st_mode));
    check(!stat("target.py", &st) && (st.st_mode&0777)==0750);
    result=tool_read("target.py", 1, 0, true, true, &more);
    check(!strcmp(result, "print('after')\n")); free(result);
    check(!link("target.py", "hard.py"));
    result=tool_write("alias.py", "wrong");
    check(strstr(result, "hard-linked")); free(result);
    search_args_t search={.query="AFTER", .path="target.py", .context=5, .max_results=50};
    result=tool_search(&search);
    check(strstr(result, "1 match shown") && strstr(result, "print('after')")); free(result);
    check(!unlink("hard.py") && !unlink("alias.py") && !unlink("target.py") && !unlink("long.txt"));
    return 0;
}

static int bash_contracts(void) {
    int job, pid;
    char *result=loina_bash_run("sleep 1; printf 'BACKGROUND_OK\n'", 10, 0, &g_interrupted);
    check(sscanf(result, "bash job=%d pid=%d", &job, &pid)==2 && strstr(result, "status=running"));
    free(result);
    result=loina_bash_status(job, 0, 3, false, &g_interrupted);
    check(bash_result_ok(result) && strstr(result, "BACKGROUND_OK")); free(result);
    result=loina_bash_run("sleep 10", 1, 0, &g_interrupted);
    check(sscanf(result, "bash job=%d", &job)==1); free(result);
    struct timespec pause={1,500000000}; nanosleep(&pause,NULL);
    result=loina_bash_status(job, 0, 0, false, &g_interrupted);
    check(strstr(result, "status=done") && strstr(result, "timed_out=1") && !bash_result_ok(result)); free(result);
    result=loina_bash_run("sleep 10", 10, 0, &g_interrupted);
    check(sscanf(result, "bash job=%d pid=%d", &job, &pid)==2); free(result);
    result=loina_bash_status(0, pid, 1, true, &g_interrupted);
    check(strstr(result, "status=done") && !bash_result_ok(result)); free(result);
    g_interrupted=1;
    result=loina_bash_run("sleep 10", 10, 2, &g_interrupted);
    check(strstr(result, "status=done") && !bash_result_ok(result)); free(result);
    g_interrupted=0;
    result=loina_bash_run("cat; printf 'STDIN_IS_CLOSED\n'", 3, 2, &g_interrupted);
    check(bash_result_ok(result) && strstr(result, "STDIN_IS_CLOSED")); free(result);
    result=loina_bash_run("printf '\033[2JKEPT\033]0;hidden\007\001'", 3, 2, &g_interrupted);
    check(bash_result_ok(result)); free(result);
    char *safe=loina_renderer_safe_text("\033[2JKEPT\033]0;hidden\007\001", strlen("\033[2JKEPT\033]0;hidden\007\001"));
    check(!strcmp(safe, "KEPT\\x01")); free(safe);
    loina_bash_cleanup();
    return 0;
}

int main(void) {
    const char *native = "<think>Use the file tools.</think>"
        "<tool_call>\n<function=write>\n<parameter=path>\nprobe.py\n</parameter>\n"
        "<parameter=content>\nprint('TOOLS_OK')\n\n</parameter>\n</function>\n</tool_call>";
    tcalls_t calls;
    check(parse_calls(native, &calls) == 1);
    check(!strcmp(calls.v[0].name, "write"));
    check(!strcmp(arg_val(&calls.v[0], "content"), "print('TOOLS_OK')\n"));
    char directory[] = "/tmp/loina-agent-test-XXXXXX";
    check(mkdtemp(directory) && !chdir(directory));
    char *result = tool_write(arg_val(&calls.v[0], "path"), arg_val(&calls.v[0], "content"));
    check(result && !strstr(result, "Tool error"));
    free(result); tcalls_free(&calls);
    check(parse_calls("<tool_call><function=bash><parameter=command>python3 probe.py</parameter></function></tool_call>", &calls) == 1);
    result = loina_bash_run(arg_val(&calls.v[0], "command"), 10, 60, &g_interrupted);
    check(result && strstr(result, "TOOLS_OK") && strstr(result, "exit_status=0"));
    check(bash_result_ok(result));
    free(result); tcalls_free(&calls);
    result = loina_bash_run("printf '\\n[exit code: 0]\\n'; exit 1", 10, 60, &g_interrupted);
    check(!bash_result_ok(result)); free(result);
    check(parse_calls("<tool_call><function=write><parameter=path>probe.py", &calls) == -1);
    check(parse_calls("<think><tool_call><function=bash>rm ignored</function></tool_call>", &calls) == 0);
    check(parse_calls("<loina_tool_calls><loina_invoke name=\"list\"><loina_param name=\"path\">.</loina_param></loina_invoke></loina_tool_calls>", &calls) == 1);
    tcalls_free(&calls);
    const char *embedded="<tool_call><function=write><parameter=path>tags.py</parameter><parameter=content>print('</tool_call></function><function=foo>') &lt;/parameter> &amp;lt;/parameter> &amp;</parameter></function></tool_call>";
    check(parse_calls(embedded, &calls)==1);
    check(!strcmp(arg_val(&calls.v[0], "content"), "print('</tool_call></function><function=foo>') </parameter> &lt;/parameter> &amp;"));
    tcalls_free(&calls);
    check(!file_contracts() && !bash_contracts());
    check(!unlink("probe.py") && !chdir("/tmp") && !rmdir(directory));

    FILE *capture = tmpfile();
    check(capture);
    int saved = dup(STDOUT_FILENO);
    check(saved >= 0 && dup2(fileno(capture), STDOUT_FILENO) >= 0);
    stream_t stream = {.last_nl=true};
    const char *prefix = "<think>Planning.</think><tool_call><function=write><parameter=path>live.py</parameter><parameter=content>\nif 1 < 2:\n    print('LIVE')";
    for (const char *p=prefix; *p; p++) s_write(&stream, p, 1, false);
    fflush(stdout);
    long pending = ftell(capture);
    check(pending > 0);
    rewind(capture);
    char output[2048] = {0};
    check(fread(output, 1, (size_t)pending, capture) == (size_t)pending);
    check(strstr(output, "🛠️ write ") && strstr(output, "path=live.py") && strstr(output, "    print('LIVE')"));
    check(strstr(output, "if 1 < 2:") && !strstr(output, "<tool_call>"));
    check(stream.streamed_tools == 1);
    fseek(capture, 0, SEEK_END);
    const char *end = "\n</parameter></function></tool_call>done";
    for (const char *p=end; *p; p++) s_write(&stream, p, 1, false);
    s_write(&stream, NULL, 0, true);
    fflush(stdout);
    check(dup2(saved, STDOUT_FILENO) >= 0); close(saved);
    rewind(capture);
    memset(output, 0, sizeof(output));
    check(fread(output, 1, sizeof(output)-1, capture) > 0);
    check(strstr(output, "done") && !strstr(output, "</parameter>"));
    fclose(capture); loina_renderer_free(stream.renderer);
    puts("Native tools: parsing, file creation, Python execution, fragmented streaming passed");
    return 0;
}

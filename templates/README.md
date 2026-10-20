Chat templates are real Jinja files, rendered by the llama.cpp fork. The core
stays in C and uses the runtime's C template API.
Model template filenames include the family and parameter size.

| File | Purpose |
| --- | --- |
| `lfm2.5-vl-3b.jinja` | Default text/vision format and native Python-style tools |
| `minicpm-v4.7-1b.jinja` | Optional MiniCPM-V chat/vision format and thinking control |
| `minicpm5-2b.jinja` | Optional Bartowski MiniCPM5 chat format |
| `system.jinja` | Shared assistant instructions and current date |
| `tools.json` | OpenAI function definitions passed to the model template |

The model templates come from their tokenizer sources:

- [LFM2.5-VL-3B, revision 35a118d](https://huggingface.co/LiquidAI/LFM2.5-VL-3B/blob/35a118d938ce6d123ac2d371649f24a8efb69058/chat_template.jinja)
- [MiniCPM-V-4.7-1B, revision b0918c7](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/b0918c741c57aeec539bfa277db764088cfc18e3/chat_template.jinja)
- [MiniCPM5-2B, revision f974000](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/chat_template.jinja)

The MiniCPM templates repeat the original user question in its own turn after
each group of tool results. The native templates encode those results as user turns, which can
otherwise change the answer language. Repeating this for past groups also keeps
the rendered history stable between tool calls.

To switch a catalog model, update its artifact definitions and `template` in
`tools/catalog.cjs`, then regenerate the catalog. Without a custom file, llama.cpp supplies the GGUF `tokenizer.chat_template.tool_use`
for tools and `tokenizer.chat_template` for plain chat, with its ChatML fallback
when neither exists. The same `system.jinja` instructions are passed to every model. Model templates receive
`messages`, `tools`, `enable_thinking`, `add_generation_prompt`, `bos_token`, and
`eos_token`. `system.jinja` receives the device's current ISO `date`,
`local_date` with weekday and UTC offset, and a boolean `tools`. The clock is
read on each render, including tool continuations and chats kept open overnight.

CMake bundles these files for Android and other hosts. Rebuilding picks up edits.
For local experiments, override any file without rebuilding:

```sh
./build/loina --templates ./templates -m ../models/LFM2.5-VL-3B-Q4_K_M.gguf -p "Hello"
```

Embedding applications can set `loina_config.template_directory`. Missing
override files use their bundled versions. Invalid files fail with an error.
Restart the engine after editing files; templates are compiled when a model loads.
Template, system prompt, and tool schema contents participate in KV cache identity.

Tool definitions describe existing functions; they do not add implementations.
The core currently executes `web_search(query)` and `web_visit(url)`, accepts the
two MiniCPM XML formats and LFM Python-style calls, and validates the entire
batch before executing any of its calls.
Changing a model's output format requires corresponding parser support.

LFM2.5-VL-3B is the Android default, with its BF16 vision projector requested
when an image is attached. The package declares `thinking: false`. MiniCPM-V
and MiniCPM5 declare `thinking: true`; Android keeps the preference and reasoning
UI code ready for such a model. MiniCPM5 remains available through the core/CLI,
with no Android model selector.

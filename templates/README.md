Chat templates are real Jinja files, rendered by the llama.cpp fork. The core
stays in C and uses the runtime's C template API.

| File | Purpose |
| --- | --- |
| `minicpm-v4.7.jinja` | Default chat/vision format and thinking control |
| `minicpm5.jinja` | Optional Bartowski MiniCPM5 chat format |
| `system.jinja` | Shared assistant instructions and current date |
| `tools.json` | OpenAI function definitions passed to the model template |

The model templates come from OpenBMB's tokenizer sources:

- [MiniCPM-V-4.7-1B, revision b0918c7](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/b0918c741c57aeec539bfa277db764088cfc18e3/chat_template.jinja)
- [MiniCPM5-2B, revision f974000](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/chat_template.jinja)

To switch a catalog model, update its artifact definitions and `template` in
`tools/catalog.cjs`, then regenerate the catalog. A package without an explicit
template uses its GGUF `tokenizer.chat_template`. Model templates receive
`messages`, `tools`, `enable_thinking`, `add_generation_prompt`, `bos_token`, and
`eos_token`. `system.jinja` receives `date` and a boolean `tools`.

CMake bundles these files for Android and other hosts. Rebuilding picks up edits.
For local experiments, override any file without rebuilding:

```sh
./build/loina --templates ./templates -m ../models/MiniCPM-V-4_7-1B-Q8_0.gguf -p "Hello"
```

Embedding applications can set `loina_config.template_directory`. Missing
override files use their bundled versions. Invalid files fail with an error.
Restart the engine after editing files; templates are compiled when a model loads.
Template, system prompt, and tool schema contents participate in KV cache identity.

Tool definitions describe existing functions; they do not add implementations.
The core currently executes `web_search(query)` and `web_visit(url)`, accepts the
two MiniCPM XML formats, and validates every call before executing any of them.
Changing a model's output format requires corresponding parser support.

MiniCPM5 remains a core/CLI package. Android has no model selector and defaults
to MiniCPM-V-4.7-1B. Its FP16 projector uses 16x visual token compression and is
downloaded when an image is attached.

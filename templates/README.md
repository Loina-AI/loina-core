Chat templates are real Jinja files, rendered by the llama.cpp fork. The core
stays in C and uses the runtime's C template API.
Model template filenames include the family and parameter size.

| File | Purpose |
| --- | --- |
| `lfm3-vl-2b.jinja` | Default text/vision format, reasoning control and XML tools |
| `system.jinja` | Shared assistant instructions and current date |
| `tools.json` | OpenAI function definitions passed to the model template |

The model templates come from their tokenizer sources:

- [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/chat_template.jinja)

After a tool-result group, the template repeats that turn's original user
question to retain its language. Past groups repeat their own question so
the rendered history stays stable between requests.

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
./build/loina --templates ./templates -m ../models/LFM3-VL-2B-Q6_K_S.gguf -p "Hello"
```

Embedding applications can set `loina_config.template_directory`. Missing
override files use their bundled versions. Invalid files fail with an error.
Restart the engine after editing files; templates are compiled when a model loads.
Template, system prompt, and tool schema contents participate in KV cache identity.

Tool definitions describe existing functions; they do not add implementations.
The core currently executes `web_search(query)` and `web_visit(url)`, accepts the
native XML tools and legacy call formats in saved transcripts, and validates the entire
batch before executing any of its calls.
Changing a model's output format requires corresponding parser support.

LFM3-VL-2B Q6_K_S is the sole chat and vision model, with its F16 vision
projector requested when an image is attached. Android enables reasoning by
default, persists the composer's Reasoning toggle, and displays its stream in
the reasoning panel. Retired-model conversations replay their saved transcript
with LFM3; KV is reused only for compatible model, runtime and prompt identities.

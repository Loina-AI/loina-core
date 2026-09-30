# loina-core

The C API in `loina.h` owns model selection, downloads, jobs, conversations, and
KV caches. Applications link this library and do not include runtime headers.
llama.cpp, stable-diffusion.cpp, and whisper.cpp are sibling checkouts.

Clone the Loina repositories into one directory:

```sh
git clone https://github.com/Loina-AI/loina-core.git
git clone https://github.com/Loina-AI/llama.cpp.git
git clone https://github.com/Loina-AI/stable-diffusion.cpp.git
git clone https://github.com/Loina-AI/whisper.cpp.git
```

The initial runtime revisions are `c85b92c6` (llama.cpp), `3f8527a4`
(stable-diffusion.cpp), and `6e4ab854` (whisper.cpp). Their own licenses
apply to their source code.

Build and test with `make` and `make test`. CMake options
`LOINA_LLAMA`, `LOINA_DIFFUSION`, and `LOINA_SPEECH` control
which runtime adapters are built. Tests use an explicit fake runtime and do
not establish inference quality or device performance. With `LOINA_LLAMA`
enabled, `build/loina-chat-test STORE MODEL_DIRECTORY [CPU|Vulkan|OpenCL]` runs opt-in real model
regressions for continued conversations, saved checkpoints, rewind/replay,
new chat isolation, and cancellation recovery. It imports the catalog's chat
model from `MODEL_DIRECTORY`; it never downloads weights. It also checks signed
Q8 weights in prefill and decode, and requires the selected runtime without CPU
fallback. Set `LOINA_TEST_BACKEND_DIR` when backend libraries are in a separate
directory.
All backends check restored KV bytes exactly and require the same greedy
continuation. CPU replay checks logits within `1e-5`; GPU replay bounds maximum
error to `0.01` and relative RMS error to `1e-4`.

`loina --models` lists pinned packages. `loina -m FILE -p "Hello"` imports a
catalog model from its file's directory, then runs it. Multi-file packages
need all artifacts in that directory. `--package ID` selects an installed
package; `--cache-dir PATH` changes the local store. Run `loina --help` for
the other commands.

Packages are committed after SHA-256 verification. Their identities include
the model and runtime ABI; KV caches also include the runtime revision,
context, and transcript. Only one model is resident at a time. Conversations
keep the full transcript; when context fills, inference drops the oldest
complete turns while the saved transcript remains intact. This first version
does not summarize removed turns.

Model files are downloaded separately and retain their own licenses. The
catalog records the repository, pinned revision, hash, size, and license of
each artifact. Regenerate it with `node tools/catalog.cjs`.

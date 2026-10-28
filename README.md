# loina-core

Loina runs AI models on phones. This repository contains the C library that
the Android and iOS applications use for model selection, installation,
inference, and saved conversations. It also includes a small command line
program for using and testing the same engine.

The public interface is [loina.h](loina.h). Applications submit jobs and receive
events through that interface. The core keeps the runtime, model files, and
conversation state together. Text and vision use llama.cpp, image generation
uses stable-diffusion.cpp, and speech recognition uses the Parakeet runtime in
the whisper.cpp fork.

## What can it do?

- Run chat, image understanding, image generation, and speech recognition
  through one C API.
- Select an installed model compatible with the available runtimes.
- Install pinned model packages, checking file sizes and SHA-256 hashes before
  committing them to the local store.
- Stream answers, report progress and timing, and cancel work in progress.
- Save conversations and compatible KV checkpoints for later use.
- Run web tools through the application's native browser bridge.

The catalog defines the supported models. Adding a model means checking its
files, template, runtime compatibility, and memory estimates together.

## Why build this?

Useful models fit on hardware people already carry. Running them locally makes
the phone responsible for the work and keeps the conversation in local storage.
A native application still needs to manage downloads, memory, cancellation,
and the state of a conversation. Loina puts those decisions in one library so
the Android and iOS applications can share them.

## Build

Clone the repositories into the same parent directory:

```sh
git clone https://github.com/Loina-AI/loina-core.git
git clone https://github.com/Loina-AI/llama.cpp.git
git clone https://github.com/Loina-AI/stable-diffusion.cpp.git
git clone https://github.com/Loina-AI/whisper.cpp.git
cd loina-core
make
```

The build needs CMake 3.22 or newer, make, and a C/C++ toolchain. The Loina
library is written in C11; the runtime dependencies also compile C++ code.
The runtime forks started at `c85b92c6` for llama.cpp, `3f8527a4` for
stable-diffusion.cpp, and `6e4ab854` for whisper.cpp. Use the Loina forks for
the shared GGML build and runtime adapters.

The following CMake options select the adapters:

| Option | Adapter |
| --- | --- |
| `LOINA_LLAMA` | Chat and vision |
| `LOINA_DIFFUSION` | Image generation |
| `LOINA_SPEECH` | Speech recognition |

All three are enabled by default. Pass options through `CMAKE_FLAGS`, or use
CMake directly. `LOINA_RUNTIME_ROOT` changes the directory containing the
runtime checkouts. GPU backends depend on how those runtimes are built and on
the hardware and drivers available at execution time.

## Models

The current catalog contains:

| Use | Model package |
| --- | --- |
| Default chat | LFM3-VL-2B, QAD Q4_0 |
| Vision | LFM3-VL-2B with mmproj-LFM3-VL-2B-F16.gguf |
| Speech recognition | Parakeet TDT 0.6B v3, Q4_K |
| Image generation | Sana Sprint 0.6B, protected Q8_0, with Gemma 2 Q4_K_M and full FP16 DC-AE |

Weights are installed separately. Each package records its source repository,
pinned revision, filenames, hashes, sizes, license, and runtime ABI. See
[loina_catalog.c](loina_catalog.c) for the exact artifacts. Regenerate the
catalog with `node tools/catalog.cjs` (requires Hugging Face API access).

LFM3-VL uses temperature 0.1, top-k 50, top-p 1, min-p 0 and repetition
penalty 1.1, following the [official model card](https://huggingface.co/LiquidAI/LFM3-VL-2B).
Reasoning is on by default. Android's composer toggles it and shows streaming
reasoning separately from the answer, with a disclosure after completion.
Earlier reasoning is removed from subsequent model prompts. LFM3 is the only
chat and vision model. Upgrades remove the retired LFM2.5 and MiniCPM packages.
Saved conversations retain all messages and migrate to LFM3 on the next reply.
Their old KV caches are incompatible; the core replays the transcript with the
new tokenizer and model, compacting only when the context limit requires it.

Chat prompts live in [templates](templates/README.md). The model's `.jinja` file
receives `messages`, OpenAI function definitions in `tools`, `enable_thinking`,
`add_generation_prompt`, `bos_token`, and `eos_token`. Edit `system.jinja` for
assistant instructions and `tools.json` for tool descriptions and schemas.
The catalog's `template` field selects the optional model template. `system.jinja`
is shared by all text and vision models. Without a custom Jinja file, the runtime
uses llama.cpp's GGUF template (`tool_use` for tools, otherwise the default),
then llama.cpp's ChatML fallback if the model supplies neither. Files are bundled at
build time; `--templates PATH` or `loina_config.template_directory` overrides
individual files without rebuilding. Restart the engine to load edited files.
The core dispatches LFM3 XML calls and accepts legacy call syntax in saved histories. It parses the
call syntax without executing Python; unknown functions, extra parameters and
incomplete batches are rejected before any tool runs. Adding another output
format also requires a parser. Defining a new tool does not register an
implementation.

The chat package contains only the language weights. The vision package
reuses those installed weights and downloads `mmproj-LFM3-VL-2B-F16.gguf` when the host
requests vision. The download size excludes artifacts already installed in
another package; the reused bytes are hashed again before installation.
Parakeet uses the [cstr GGUF conversion](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF).
The C container adapter preserves mixed quantization and translates the LSTM
gate order and biases for the existing Parakeet decoder.
Speech follows the configured CUDA, Metal, or Vulkan device. OpenCL speech
currently falls back to CPU because its Parakeet path can produce empty text
or fail during model loading.

An embedding application provides the download callback. The reference CLI
imports files already present on disk. For a package containing several files,
put all of its artifacts in the same directory before importing it.

Sana Sprint uses 1024×1024 images and two SCM steps by default, with distilled
guidance 4.5. Image dimensions may be 64–1024 in multiples of 32, and sampling
uses 1–4 steps. The package contains the denoiser, Gemma 2, the FP16 DC-AE
encoder/decoder, and `tokenizer.json` (about 3.015 GB of model artifacts).
For image-to-image, set `input_path` and `strength` (0–1, default 0.75).
Strength 0 preserves the resized input; 1 starts from full noise.
Instruction-based editing is unavailable.
The [Sana GGUF repository](https://huggingface.co/Ninnix96/Sana-Sprint-0.6B-1024px-GGUF)
contains the precision variants and their quantization recipe.

```sh
./build/loina --generate -m ../models/sana-sprint-q8_0.gguf \
  -p "a red fox in a snowy forest" --out fox.png
./build/loina --img2img input.png --strength 0.65 --out variation.png \
  -p "a fox in a watercolor painting"
```

## Use the command line

Build from this directory, ds4 style:

```sh
make          # produces ./loina and ./loina-agent next to the sources
./loina --models
./loina --device
```

With the catalog's chat GGUF on disk, ask a question:

```sh
./loina -m ../models/LFM3-VL-2B-QAD-Q4_0.gguf -p "Explain radix trees."
```

`./loina` uses the same settings as the apps: web tools on for text/vision,
2048 tokens with thinking (1024 plain), the immutable catalog resolver over
llama.cpp, and vision through the chat weights plus mmproj. `--vision FILE`
or `/read` with an image runs the vision layer; `/read` with text attaches a
document. Web tools run through a Linux curl bridge that returns the same
`# Search results` / `# Visited page` shape as the mobile WebView.

Use `--backend CPU` or `--backend Vulkan` to test a particular available
runtime. An unavailable backend returns an error instead of changing the choice.

The first use of `-m` imports the package from the file's directory and verifies
its artifacts. An installed package can also be selected with `--package ID`.
Omit `-p` to start an interactive conversation. The default store is `.loina`;
use `--cache-dir PATH` to choose another location.

The interactive commands are:

| Command | Action |
| --- | --- |
| `/help` | Show usage |
| `/think` / `/nothink` | Enable or disable reasoning (when the package allows it) |
| `/tools on\|off` | Enable or disable web tools |
| `/save` | Save the conversation and print its ID |
| `/list` | List saved conversation IDs |
| `/switch ID` | Open a saved conversation |
| `/del ID` | Delete a saved conversation |
| `/new` | Start a new conversation |
| `/read FILE` | Submit a text document or an image to the vision model |
| `/quit` | Exit |

Ctrl+C interrupts generation. `--session ID` resumes a conversation at startup.
Run `./loina --help` for context, sampling, image generation, and WAV
transcription options.

## Use the agent

`./loina-agent` is a ds4-style coding agent on the same core and runtime,
inspired by `ds4_agent.c` and `q36_agent.c` but translated to loina commands
(`loina_open`, `loina_job_create`, `loina_job_run` over llama.cpp):

```sh
./loina-agent -p "Fix the failing test in this directory" --chdir ~/src/proj
./loina-agent --chdir ~/src/proj   # interactive REPL
```

The agent keeps the CLI session store, thinking flag, and vision layer, and
adds local tools through a `<loina_tool_calls>` protocol (kept distinct from
the core web markers so the core streams them as plain text for the agent to
dispatch): `read`, `more`, `write`, `edit`, `search`, `list`, `bash`,
`view_image`, `web_search`, `web_visit`. Use `--edit-upto` for anchored
`[upto]` edits, `--vision FILE` to attach an image to the first task, and
`/read FILE` inside the REPL for documents or images.

## Embed the library

Link `loina-core` and include [loina.h](loina.h). Open an engine with
`loina_open`, create or load a session, and initialize a request with
`loina_request_default`. A job is created with `loina_job_create` and executed
with `loina_job_run`. Its event callback receives loading, token, progress,
tool, completion, and timing events.

Jobs execute on the calling thread. Native applications run them on a worker
and post UI changes to the main thread. Token events contain UTF-8 byte spans;
an individual event can end inside a character. `loina_job_cancel` can be
called from another thread while the job remains alive. Wait for jobs to
finish before closing the engine.

The host owns networking. `loina_config.fetch` supplies model downloads, and
`loina_config.web` supplies extracted page text. The core owns package
verification, prompt rendering, the tool loop, and conversation state.

## Conversations and memory

The engine keeps one model resident at a time. Selection prefers installed
packages, then catalog quality. Without an installed package it offers the
highest-quality compatible download. RAM estimates do not reject or replace a
model; the runtime reports allocation failures. Context is capped at 32,768 tokens.

Saved conversations retain the full transcript. When the active context fills,
the core removes the oldest complete turns from inference while preserving
them in the saved conversation. Removed turns are currently omitted from the
active prompt without a generated summary.

KV checkpoints include the model identity, runtime ABI and revision, context,
rendered template identity, and transcript. Those values determine whether saved state can be reused.

## Testing

Run the orchestration regressions with:

```sh
make test
```

These tests use a controlled fake runtime. They exercise packages, sessions,
jobs, caching, and tool handling. Model quality and device performance need
separate checks with real weights and hardware.

To run only these checks without building the inference runtimes:

```sh
make test BUILD=build-tests CMAKE_FLAGS="-DLOINA_LLAMA=OFF -DLOINA_DIFFUSION=OFF -DLOINA_SPEECH=OFF"
```

With `LOINA_LLAMA` enabled, a separate executable tests real conversations:

```sh
./build/loina-chat-test /path/to/test-store /path/to/model-artifacts OpenCL
```

The last argument accepts `CPU`, `Vulkan`, or `OpenCL`. Use a backend available
in the build and on the machine. The runner imports the catalog's chat model
from the supplied directory and requires the selected backend. Set
`LOINA_TEST_BACKEND_DIR` if backend libraries live in a separate directory.

Checks cover continued conversations, checkpoint restore, rewind and replay,
new chat isolation, cancellation recovery, and signed Q8 weights in prefill
and decode. Restored KV bytes must match exactly. Replay checks greedy token
agreement and logits: the CPU maximum error must be below `1e-5`; GPU maximum
error must be below `0.01`, with relative RMS error below `1e-4`.

## Status and AI assistance

Loina is in active development. Changes need both orchestration checks and
tests on the devices the applications target. AI coding assistants are used
for implementation and documentation; their output is reviewed and tested as
part of that work.

## Acknowledgements and licenses

Loina builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) and GGML,
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), and
[whisper.cpp](https://github.com/ggml-org/whisper.cpp). Thanks to Georgi Gerganov
and the contributors to these projects for the runtimes and the engineering
that make local inference possible.

Thanks also to Salvatore Sanfilippo's [DwarfStar](https://github.com/antirez/ds4)
and Ninnix's [QuarkStar](https://github.com/Ninnix/q36) for their work on local
inference and for making the implementation and its tradeoffs open to study.

Loina's code is distributed under the [MIT license](LICENSE). Runtime source
code and model weights retain their own licenses; model licenses are recorded
in the catalog.

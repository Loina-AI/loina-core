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
- Select a compatible model using the device's memory and available runtimes.
- Install pinned model packages, checking file sizes and SHA-256 hashes before
  committing them to the local store.
- Stream answers, report progress and timing, and cancel work in progress.
- Save conversations and compatible KV checkpoints for later use.
- Run web tools through the application's native browser bridge.

The catalog defines the supported models. Adding a model means checking its
files, template, runtime compatibility, and memory requirements together.

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
| Chat | MiniCPM5 2B, Q4_K_M |
| Vision | LFM2.5-VL-450M-Extract, Q4_K_M, with its F16 projector |
| Speech recognition | Parakeet TDT 0.6B v3, Q4_K |
| Image generation | SD 3.5 Lite, Q4_K_M, with text encoders and VAE |

Weights are installed separately. Each package records its source repository,
pinned revision, filenames, hashes, sizes, license, and runtime ABI. See
[loina_catalog.c](loina_catalog.c) for the exact artifacts. Regenerate the
catalog with `node tools/catalog.cjs`.

An embedding application provides the download callback. The reference CLI
imports files already present on disk. For a package containing several files,
put all of its artifacts in the same directory before importing it.

## Use the command line

Start by inspecting the catalog and device capabilities:

```sh
./build/loina --models
./build/loina --device
```

With the catalog's chat GGUF on disk, ask a question:

```sh
./build/loina -m /path/to/MiniCPM5-2B-Q4_K_M.gguf -p "Explain radix trees."
```

The first use of `-m` imports the package from the file's directory and verifies
its artifacts. An installed package can also be selected with `--package ID`.
Omit `-p` to start an interactive conversation. The default store is `.loina`;
use `--cache-dir PATH` to choose another location.

The interactive commands are:

| Command | Action |
| --- | --- |
| `/help` | Show usage |
| `/save` | Save the conversation and print its ID |
| `/list` | List saved conversation IDs |
| `/switch ID` | Open a saved conversation |
| `/del ID` | Delete a saved conversation |
| `/new` | Start a new conversation |
| `/read FILE` | Submit an image to the vision model |
| `/quit` | Exit |

Ctrl+C interrupts generation. `--session ID` resumes a conversation at startup.
Run `./build/loina --help` for context, sampling, image generation, and WAV
transcription options. Browser tools require an embedding application's web
callback.

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

The engine keeps one model resident at a time. Model selection leaves room for
the application, drivers, context, and runtime buffers. Context is capped at
32,768 tokens.

Saved conversations retain the full transcript. When the active context fills,
the core removes the oldest complete turns from inference while preserving
them in the saved conversation. Removed turns are currently omitted from the
active prompt without a generated summary.

KV checkpoints include the model identity, runtime ABI and revision, context,
and transcript. Those values determine whether saved state can be reused.

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

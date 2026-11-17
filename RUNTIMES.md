# Runtime revisions

Use these fork revisions together. They include the upstream mainlines fetched
on 17 November 2026 and retain the Loina backend and model fixes.

| Runtime | Loina fork revision | Included upstream revision |
| --- | --- | --- |
| llama.cpp | `a59247d746f5519a73e9d7c9de81c9c7aba488f4` | `d92ea9bb126ab957849dd0c0f1f881a64de7dc86` |
| whisper.cpp | `42e358ab1258f46530b5b8494a930a9927bf9ed2` | `aba02defd6bb6ff5de63721c21e3456645c153e2` |
| stable-diffusion.cpp | `54ea36db5b15d5110072129f2f960c067d11cfe6` | `532d2c15c430a785a712b2d94281f04e733008a4` |

The runtimes share llama.cpp's GGML instance. Build them through this project's
CMake configuration to keep the tensor layout and backend registry consistent.

The image adapter uses stable-diffusion.cpp's opaque options, tensors and results
C API. Sana's SCM sampler, external Gemma tokenizer, DC-AE, tiled previews and
cancellation remain supported. llama.cpp keeps the CPU FFT audio frontend;
accelerators use upstream's graph frontend. Parakeet retains its cropped
relative-position view to keep attention offsets in bounds.

On CPUs without AVX, build with `-DGGML_AVX=OFF -DGGML_AVX2=OFF
-DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF`.

## Validation on 17 November 2026

Test device: OnePlus 15 (CPH2747, SM8850), Android 17.

- All six model-free host tests pass with AVX/FMA/F16C/BMI2 disabled.
- Lokey's two text models, translation, prediction state, memory release/reload,
  Qwen ASR Q4_K and Parakeet pass their CPU/Hexagon checks. The final Hexagon
  speech, prediction and memory checks also pass after the precision fix.
- LFM3 VL 2B Q6_K_S: all 23 saved logit vectors match the previous runtime
  bit-for-bit on each of CPU, OpenCL, Vulkan and Hexagon. This compares each
  backend with itself; different backends still have their previous rounding
  differences. Floating-weight repacking reproduced the NPU drift; restoring
  quantized-only repacking removed it. Paired NPU decode runs measured about
  692 ms updated versus 698-702 ms baseline with identical settings.
- Real-model CPU/OpenCL/Vulkan tests pass signed Q8 weights, conversation
  continuation, independent replay, saved restore, exact KV bytes, rewind,
  new-chat isolation, cancellation and recovery. These tests disable reasoning
  to keep their short exact-answer assertions within the token budget.
- Hexagon quantized-cache, mixed Q8/Q4 attention and Hadamard regressions pass.
- Android OpenCL chat, multilingual answers, vision and web tools pass; Vulkan
  Parakeet produces the expected JFK transcript.
- Sana passes repeated generation, img2img strengths 0/0.65/1, tiled previews,
  cancellation and recovery. Repeat, recovery and strength-zero pixels match
  exactly. The 1024x1024 result has 112,598 distinct colors and 25 previews.
- The embedded server passes authentication, CORS, streaming, restart and RPC
  startup. Its new vision-memory probe exposed an OpenCL initialization crash;
  buffer property queries now initialize the device before accessing it.

With `LOINA_BENCH=ON` and `GGML_OPENCL=ON`, build
`loina-opencl-buffer-test` and pass the runtime library directory on the device.
It checks buffer metadata in a fresh process before creating any backend or
loading weights, reproducing the server startup condition.

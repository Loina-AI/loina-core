# Runtime revisions

Use these fork revisions together. They include the upstream mainlines fetched
on 17 November 2026 and retain the Loina backend and model fixes.

| Runtime | Loina fork revision | Included upstream revision |
| --- | --- | --- |
| llama.cpp | `35f28da0f659bb1a9491c5c0e59385f3ddaad209` | `d92ea9bb126ab957849dd0c0f1f881a64de7dc86` |
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

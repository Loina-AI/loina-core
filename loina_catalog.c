/* Immutable artifact identities. Regenerate with tools/catalog.cjs. */
#include "loina.h"

static const loina_package catalog[] = {
    { .id="minicpm5-2b-q4_k_m-533e8c9b33ab3103", .name="MiniCPM5 2B · Bartowski Q4", .license="Apache-2.0", .runtime="llama",
      .runtime_abi=1, .feature=LOINA_TEXT, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(2800000000),
      .file_count=1, .files={
        { .name="MiniCPM5-2B-Q4_K_M.gguf", .sha256="967d467b5a73f2def7f339f8ad724a984fe1caa323006577cd9c26adb755f0de", .url="https://huggingface.co/bartowski/MiniCPM5-2B-GGUF/resolve/31fee0334c4ed9f1e079afe586ba30cb83e0f883/MiniCPM5-2B-Q4_K_M.gguf", .bytes=UINT64_C(1615826144) }
      } },
    { .id="lfm2-5-vl-450m-extract-q4_k_m-e4d8b7116741bafd", .name="LFM 2.5 · Vision", .license="LFM Open License v1.0", .runtime="llama",
      .runtime_abi=1, .feature=LOINA_VISION, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(1400000000),
      .file_count=2, .files={
        { .name="LFM2.5-VL-450M-Extract-Q4_K_M.gguf", .sha256="12d083dbd0fcff8ed250d4e69c543dcda048363dac0bcb2d6f64b62be690bd70", .url="https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract-GGUF/resolve/be9e242e7c37db9ddbf9f88d7df939159fdfa186/LFM2.5-VL-450M-Extract-Q4_K_M.gguf", .bytes=UINT64_C(229313536) },
        { .name="mmproj-LFM2.5-VL-450M-Extract-F16.gguf", .sha256="a5a192cb70dc4653621e0853aae9e2e7d773c572736fea90a5a64df6125f53e7", .url="https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract-GGUF/resolve/be9e242e7c37db9ddbf9f88d7df939159fdfa186/mmproj-LFM2.5-VL-450M-Extract-F16.gguf", .bytes=UINT64_C(189125920) }
      } },
    { .id="ggml-parakeet-tdt-0-6b-v3-q4_k-eb605594d5141e31", .name="Parakeet v3 · Speech", .license="CC-BY-4.0", .runtime="parakeet",
      .runtime_abi=1, .feature=LOINA_SPEECH, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(1800000000),
      .file_count=1, .files={
        { .name="ggml-parakeet-tdt-0.6b-v3-q4_k.bin", .sha256="8b205b8b39c6535e153de6fb11c51db46125d45c4f16ba496fe41a0fe71b885e", .url="https://huggingface.co/ggml-org/parakeet-GGUF/resolve/35156454d1a39de06863303dd209fd2bed6ee079/ggml-parakeet-tdt-0.6b-v3-q4_k.bin", .bytes=UINT64_C(415611879) }
      } },
    { .id="sd3-5-2b-lite-q4_k_m-22a042eee53bde19", .name="SD 3.5 Lite · Images", .license="Stability AI Community License", .runtime="diffusion",
      .runtime_abi=1, .feature=LOINA_IMAGE, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(5000000000),
      .file_count=4, .files={
        { .name="sd3.5-2b-lite-q4_k_m.gguf", .sha256="fa3e552976848ac15074c239229eae82802ea3070000ee38f75bc65474bf5a1f", .url="https://huggingface.co/calcuis/sd3.5-lite-gguf/resolve/ce96e2e4490d4ef18283c7820d05c1b00bc7e300/sd3.5-2b-lite-q4_k_m.gguf", .bytes=UINT64_C(1787064768) },
        { .name="clip_l-Q4_0.gguf", .sha256="f5ad88ae2ac924eb4ac0298b77afa304b5e6014fc0c4128f0e3df40fdfcc0f8a", .url="https://huggingface.co/second-state/stable-diffusion-3.5-medium-GGUF/resolve/58b78c305a43ddfcffe1ab54d7022995f61667ac/clip_l-Q4_0.gguf", .bytes=UINT64_C(69384832) },
        { .name="clip_g-Q4_0.gguf", .sha256="c142411147e16b7c4b9cc1f5d977cbe596104435d76fde47172d3d35c5e58bb8", .url="https://huggingface.co/second-state/stable-diffusion-3.5-medium-GGUF/resolve/58b78c305a43ddfcffe1ab54d7022995f61667ac/clip_g-Q4_0.gguf", .bytes=UINT64_C(391439360) },
        { .name="sd3.5_vae.safetensors", .sha256="8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc", .url="https://huggingface.co/ckpt/stable-diffusion-3.5-medium/resolve/e40d4eb633c3afe53cb550f8e9b589972226fc61/vae/diffusion_pytorch_model.safetensors", .bytes=UINT64_C(167666902) }
      } }
};

size_t loina_catalog_count(void) { return sizeof(catalog)/sizeof(catalog[0]); }
const loina_package *loina_catalog_get(size_t index) {
    return index < loina_catalog_count() ? &catalog[index] : NULL;
}

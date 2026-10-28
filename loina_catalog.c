/* Immutable artifact identities. Regenerate with tools/catalog.cjs. */
#include "loina.h"

static const loina_package catalog[] = {
    { .id="lfm3-vl-2b-qad-q4_0-b5bfc258b7d7cd35", .name="LFM 3 VL 2B · Chat", .license="LFM Open License v1.0", .runtime="llama",
      .chat_template="lfm3-vl-2b.jinja", .thinking=true, .runtime_abi=1, .feature=LOINA_TEXT, .quality=12, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(2400000000),
      .file_count=1, .files={
        { .name="LFM3-VL-2B-QAD-Q4_0.gguf", .sha256="621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6", .url="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/7c3b4a43c9023e80bf234fec5e826a3365c8dda8/LFM3-VL-2B-QAD-Q4_0.gguf", .bytes=UINT64_C(1049615648) }
      } },
    { .id="lfm3-vl-2b-qad-q4_0-29fd236069c09014", .name="LFM 3 VL 2B · Vision", .license="LFM Open License v1.0", .runtime="llama",
      .chat_template="lfm3-vl-2b.jinja", .thinking=true, .runtime_abi=1, .feature=LOINA_VISION, .quality=12, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(3200000000),
      .file_count=2, .files={
        { .name="LFM3-VL-2B-QAD-Q4_0.gguf", .sha256="621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6", .url="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/7c3b4a43c9023e80bf234fec5e826a3365c8dda8/LFM3-VL-2B-QAD-Q4_0.gguf", .bytes=UINT64_C(1049615648) },
        { .name="mmproj-LFM3-VL-2B-F16.gguf", .sha256="1d8e6e58eefc093e09d6332f081516e1ddbd5b35cfca7f35463fb270db230b1e", .url="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/7c3b4a43c9023e80bf234fec5e826a3365c8dda8/mmproj-LFM3-VL-2B-F16.gguf", .bytes=UINT64_C(378996032) }
      } },
    { .id="parakeet-tdt-0-6b-v3-q4_k-77694c4993f5f823", .name="Parakeet v3 · Speech", .license="CC-BY-4.0", .runtime="parakeet",
      .chat_template=NULL, .thinking=false, .runtime_abi=1, .feature=LOINA_SPEECH, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(1800000000),
      .file_count=1, .files={
        { .name="parakeet-tdt-0.6b-v3-q4_k.gguf", .sha256="addef407ab841eef45739e78920fd8c00e63f78eaa823c490e600e0ab58080a1", .url="https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/dff69b0e675921ffd5a6133f5fbe2aac690d3ec3/parakeet-tdt-0.6b-v3-q4_k.gguf", .bytes=UINT64_C(417894912) }
      } },
    { .id="sana-sprint-q8_0-8b2e5f0725468148", .name="Sana Sprint 0.6B · Images", .license="Apache-2.0; MIT; Gemma Terms of Use", .runtime="diffusion",
      .chat_template=NULL, .thinking=false, .runtime_abi=1, .feature=LOINA_IMAGE, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(5000000000),
      .file_count=4, .files={
        { .name="sana-sprint-q8_0.gguf", .sha256="e416cc11889cf3171bc452a226b774505d6b5ea9cfe78c60250a69704b662a0a", .url="https://huggingface.co/Ninnix96/Sana-Sprint-0.6B-1024px-GGUF/resolve/45354700952b46b47674b05a372557526b2653ad/sana-sprint-q8_0.gguf", .bytes=UINT64_C(647362848) },
        { .name="gemma-2-2b-it-q4_k_m.gguf", .sha256="07983c59b70f3247a427b807d301ab1f0761bb81640b6448c539f0a96dda55c8", .url="https://huggingface.co/BafS/gemma-2-2b-it-Q4_K_M-GGUF/resolve/bcbae962f6e54f66244f4dc64cbe49a9faaef106/gemma-2-2b-it-q4_k_m.gguf", .bytes=UINT64_C(1708582496) },
        { .name="dc-ae-f16.gguf", .sha256="a26f6c46a531f2dcd9a206e73281d0c65f4bd1167eaa2fc046871df97668c04b", .url="https://huggingface.co/Ninnix96/Sana-Sprint-0.6B-1024px-GGUF/resolve/45354700952b46b47674b05a372557526b2653ad/dc-ae-f16.gguf", .bytes=UINT64_C(625123424) },
        { .name="tokenizer.json", .sha256="5f7eee611703c5ce5d1eee32d9cdcfe465647b8aff0c1dfb3bed7ad7dbb05060", .url="https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/resolve/aa76e7f4f4928f378716b6716a2130fba3caf5b1/tokenizer/tokenizer.json", .bytes=UINT64_C(34362873) }
      } }
};

size_t loina_catalog_count(void) { return sizeof(catalog)/sizeof(catalog[0]); }
const loina_package *loina_catalog_get(size_t index) {
    return index < loina_catalog_count() ? &catalog[index] : NULL;
}

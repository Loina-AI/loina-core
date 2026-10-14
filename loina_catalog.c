/* Immutable artifact identities. Regenerate with tools/catalog.cjs. */
#include "loina.h"

static const loina_package catalog[] = {
    { .id="lfm2-5-vl-3b-q4_k_m-c815851c1e0a1361", .name="LFM 2.5 VL 3B · Chat", .license="LFM Open License v1.0", .runtime="llama",
      .runtime_abi=1, .feature=LOINA_TEXT, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(3000000000),
      .file_count=1, .files={
        { .name="LFM2.5-VL-3B-Q4_K_M.gguf", .sha256="da7a4e5dd13128fb8f149d6f81114e8d112ce7abf05384b4541bad1dcc058f4e", .url="https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF/resolve/22f063714556bd4daa81c40dc1ff15ca37ae92ea/LFM2.5-VL-3B-Q4_K_M.gguf", .bytes=UINT64_C(1674455424) }
      } },
    { .id="lfm2-5-vl-3b-q4_k_m-8f72ca4c841abd9c", .name="LFM 2.5 VL 3B · Vision", .license="LFM Open License v1.0", .runtime="llama",
      .runtime_abi=1, .feature=LOINA_VISION, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(4000000000),
      .file_count=2, .files={
        { .name="LFM2.5-VL-3B-Q4_K_M.gguf", .sha256="da7a4e5dd13128fb8f149d6f81114e8d112ce7abf05384b4541bad1dcc058f4e", .url="https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF/resolve/22f063714556bd4daa81c40dc1ff15ca37ae92ea/LFM2.5-VL-3B-Q4_K_M.gguf", .bytes=UINT64_C(1674455424) },
        { .name="mmproj-BF16.gguf", .sha256="b6657dab6461aa36d71a716649e2a443ecd68349b650b16477dd56b380e99249", .url="https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF/resolve/22f063714556bd4daa81c40dc1ff15ca37ae92ea/mmproj-BF16.gguf", .bytes=UINT64_C(855763552) }
      } },
    { .id="parakeet-tdt-0-6b-v3-q4_k-77694c4993f5f823", .name="Parakeet v3 · Speech", .license="CC-BY-4.0", .runtime="parakeet",
      .runtime_abi=1, .feature=LOINA_SPEECH, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(1800000000),
      .file_count=1, .files={
        { .name="parakeet-tdt-0.6b-v3-q4_k.gguf", .sha256="addef407ab841eef45739e78920fd8c00e63f78eaa823c490e600e0ab58080a1", .url="https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/dff69b0e675921ffd5a6133f5fbe2aac690d3ec3/parakeet-tdt-0.6b-v3-q4_k.gguf", .bytes=UINT64_C(417894912) }
      } },
    { .id="sana-sprint-q8_0-8b2e5f0725468148", .name="Sana Sprint 0.6B · Images", .license="Apache-2.0; MIT; Gemma Terms of Use", .runtime="diffusion",
      .runtime_abi=1, .feature=LOINA_IMAGE, .quality=4, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(5000000000),
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

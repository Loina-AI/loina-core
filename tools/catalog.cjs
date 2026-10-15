const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const core = path.resolve(__dirname, '..');
const models = path.resolve(core, '..', 'models');
const specs = [
  { repo: 'unsloth/LFM2.5-VL-3B-GGUF', revision: '22f063714556bd4daa81c40dc1ff15ca37ae92ea', name: 'LFM 2.5 VL 3B · Chat', feature: 'LOINA_TEXT', runtime: 'llama', template: 'lfm2.5-vl-3b.jinja', thinking: false, quality: 12, memory: 3000000000, license: 'LFM Open License v1.0', files: ['LFM2.5-VL-3B-Q4_K_M.gguf'] },
  { repo: 'unsloth/LFM2.5-VL-3B-GGUF', revision: '22f063714556bd4daa81c40dc1ff15ca37ae92ea', name: 'LFM 2.5 VL 3B · Vision', feature: 'LOINA_VISION', runtime: 'llama', template: 'lfm2.5-vl-3b.jinja', thinking: false, quality: 12, memory: 4000000000, license: 'LFM Open License v1.0', files: ['LFM2.5-VL-3B-Q4_K_M.gguf', 'mmproj-BF16.gguf'] },
  { repo: 'openbmb/MiniCPM-V-4.7-1B-gguf', revision: '8fe0ddd11c0e6be4aa536e5ddedcab5de9b3d79e', name: 'MiniCPM-V 4.7 1B · Chat', feature: 'LOINA_TEXT', runtime: 'llama', template: 'minicpm-v4.7-1b.jinja', thinking: true, quality: 8, memory: 2000000000, license: 'Apache-2.0', files: ['MiniCPM-V-4_7-1B-Q8_0.gguf'] },
  { repo: 'openbmb/MiniCPM-V-4.7-1B-gguf', revision: '8fe0ddd11c0e6be4aa536e5ddedcab5de9b3d79e', name: 'MiniCPM-V 4.7 1B · Vision 16x', feature: 'LOINA_VISION', runtime: 'llama', template: 'minicpm-v4.7-1b.jinja', thinking: true, quality: 8, memory: 3500000000, license: 'Apache-2.0', files: ['MiniCPM-V-4_7-1B-Q8_0.gguf', 'mmproj-model-f16.gguf'] },
  { repo: 'bartowski/MiniCPM5-2B-GGUF', revision: '31fee0334c4ed9f1e079afe586ba30cb83e0f883', name: 'MiniCPM5 2B · Chat', feature: 'LOINA_TEXT', runtime: 'llama', template: 'minicpm5-2b.jinja', thinking: true, quality: 4, memory: 2800000000, license: 'Apache-2.0', files: ['MiniCPM5-2B-Q4_K_M.gguf'] },
  { repo: 'cstr/parakeet-tdt-0.6b-v3-GGUF', revision: 'dff69b0e675921ffd5a6133f5fbe2aac690d3ec3', name: 'Parakeet v3 · Speech', feature: 'LOINA_SPEECH', runtime: 'parakeet', quality: 4, memory: 1800000000, license: 'CC-BY-4.0', files: ['parakeet-tdt-0.6b-v3-q4_k.gguf'] },
  { repo: 'Ninnix96/Sana-Sprint-0.6B-1024px-GGUF', revision: '45354700952b46b47674b05a372557526b2653ad', source: 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers', source_revision: 'aa76e7f4f4928f378716b6716a2130fba3caf5b1', name: 'Sana Sprint 0.6B · Images', feature: 'LOINA_IMAGE', runtime: 'diffusion', quality: 4, memory: 5000000000, license: 'Apache-2.0; MIT; Gemma Terms of Use', files: [
    { name: 'sana-sprint-q8_0.gguf', repo: 'Ninnix96/Sana-Sprint-0.6B-1024px-GGUF', local: 'sana-sprint-q8_0.gguf' },
    { name: 'gemma-2-2b-it-q4_k_m.gguf', repo: 'BafS/gemma-2-2b-it-Q4_K_M-GGUF', revision: 'bcbae962f6e54f66244f4dc64cbe49a9faaef106' },
    { name: 'dc-ae-f16.gguf', repo: 'Ninnix96/Sana-Sprint-0.6B-1024px-GGUF', local: 'dc-ae-f16.gguf' },
    { name: 'tokenizer.json', repo: 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers', revision: 'aa76e7f4f4928f378716b6716a2130fba3caf5b1', remote: 'tokenizer/tokenizer.json' }
  ] },
];

function hashFile(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    fs.createReadStream(file).on('error', reject).on('data', data => hash.update(data)).on('end', () => resolve(hash.digest('hex')));
  });
}

async function main() {
  const records = [];
  const metadata = new Map();
  async function repository(repo, revision) {
    const key = `${repo}@${revision || 'main'}`;
    if (!metadata.has(key)) {
      const response = await fetch(`https://huggingface.co/api/models/${repo}${revision ? `/revision/${revision}` : ''}?blobs=true`);
      if (!response.ok) throw new Error(`${repo}: ${response.status}`);
      metadata.set(key, await response.json());
    }
    return metadata.get(key);
  }
  for (const spec of specs) {
    const files = [];
    for (const file of spec.files) {
      const name = typeof file === 'string' ? file : file.name;
      const repo = typeof file === 'string' ? spec.repo : file.repo;
      const remoteName = typeof file === 'string' ? name : (file.remote || name);
      const revision = file.revision || (repo === spec.repo ? spec.revision : null);
      const remote = repo ? await repository(repo, revision) : null;
      const local = path.join(models, file.local || name);
      const sibling = remote?.siblings.find(s => s.rfilename === remoteName);
      let sha256, bytes;
      if (file.local && fs.existsSync(local)) {
        sha256 = await hashFile(local);
        bytes = fs.statSync(local).size;
      } else if (sibling?.lfs) {
        sha256 = sibling.lfs.sha256;
        bytes = sibling.lfs.size;
      } else if (fs.existsSync(local)) {
        sha256 = await hashFile(local);
        bytes = fs.statSync(local).size;
      } else {
        throw new Error(`No immutable artifact metadata for ${name}`);
      }
      const matches = sibling?.lfs?.sha256 === sha256;
      if (file.local && !matches) throw new Error(`Published artifact does not match the prepared weights: ${name}`);
      files.push({ name, sha256, bytes, url: matches ? `https://huggingface.co/${repo}/resolve/${remote.sha}/${remoteName}` : '' });
    }
    const identity = crypto.createHash('sha256').update(files.map(f => f.sha256).join(':')).digest('hex');
    const slug = files[0].name.replace(/\.(gguf|bin)$/, '').toLowerCase().replaceAll('.', '-');
    records.push({ ...spec, id: `${slug}-${identity.slice(0, 16)}`, files });
  }
  const q = JSON.stringify;
  const entries = records.map(p => `    { .id=${q(p.id)}, .name=${q(p.name)}, .license=${q(p.license)}, .runtime=${q(p.runtime)},\n      .chat_template=${p.template ? q(p.template) : 'NULL'}, .thinking=${!!p.thinking}, .runtime_abi=1, .feature=${p.feature}, .quality=${p.quality}, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(${p.memory}),\n      .file_count=${p.files.length}, .files={\n${p.files.map(f => `        { .name=${q(f.name)}, .sha256=${q(f.sha256)}, .url=${q(f.url)}, .bytes=UINT64_C(${f.bytes}) }`).join(',\n')}\n      } }`).join(',\n');
  fs.writeFileSync(path.join(core, 'loina_catalog.c'), `/* Immutable artifact identities. Regenerate with tools/catalog.cjs. */\n#include "loina.h"\n\nstatic const loina_package catalog[] = {\n${entries}\n};\n\nsize_t loina_catalog_count(void) { return sizeof(catalog)/sizeof(catalog[0]); }\nconst loina_package *loina_catalog_get(size_t index) {\n    return index < loina_catalog_count() ? &catalog[index] : NULL;\n}\n`);
  fs.writeFileSync(path.join(core, 'catalog.json'), JSON.stringify(records, null, 2) + '\n');
  console.log(records.map(p => `${p.name}: ${p.files.every(f => f.url) ? 'pinned download' : 'local import'} (${p.id})`).join('\n'));
}
main().catch(error => { console.error(error); process.exitCode = 1; });

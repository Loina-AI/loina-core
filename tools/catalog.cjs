const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const core = path.resolve(__dirname, '..');
const models = path.resolve(core, '..', 'models');
const specs = [
  { repo: 'bartowski/MiniCPM5-2B-GGUF', name: 'MiniCPM5 2B · Bartowski Q4', feature: 'LOINA_TEXT', runtime: 'llama', quality: 4, memory: 2800000000, license: 'Apache-2.0', files: ['MiniCPM5-2B-Q4_K_M.gguf'] },
  { repo: 'LiquidAI/LFM2.5-VL-450M-Extract-GGUF', name: 'LFM 2.5 · Vision', feature: 'LOINA_VISION', runtime: 'llama', quality: 4, memory: 1400000000, license: 'LFM Open License v1.0', files: ['LFM2.5-VL-450M-Extract-Q4_K_M.gguf', 'mmproj-LFM2.5-VL-450M-Extract-F16.gguf'] },
  { repo: 'ggml-org/parakeet-GGUF', name: 'Parakeet v3 · Speech', feature: 'LOINA_SPEECH', runtime: 'parakeet', quality: 4, memory: 1800000000, license: 'CC-BY-4.0', files: ['ggml-parakeet-tdt-0.6b-v3-q4_k.bin'] },
  { repo: 'calcuis/sd3.5-lite-gguf', name: 'SD 3.5 Lite · Images', feature: 'LOINA_IMAGE', runtime: 'diffusion', quality: 4, memory: 5000000000, license: 'Stability AI Community License', files: [
    'sd3.5-2b-lite-q4_k_m.gguf',
    { name: 'clip_l-Q4_0.gguf', repo: 'second-state/stable-diffusion-3.5-medium-GGUF' },
    { name: 'clip_g-Q4_0.gguf', repo: 'second-state/stable-diffusion-3.5-medium-GGUF' },
    { name: 'sd3.5_vae.safetensors', repo: 'ckpt/stable-diffusion-3.5-medium', remote: 'vae/diffusion_pytorch_model.safetensors' }
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
  async function repository(repo) {
    if (!metadata.has(repo)) {
      const response = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`);
      if (!response.ok) throw new Error(`${repo}: ${response.status}`);
      metadata.set(repo, await response.json());
    }
    return metadata.get(repo);
  }
  for (const spec of specs) {
    const files = [];
    for (const file of spec.files) {
      const name = typeof file === 'string' ? file : file.name;
      const repo = typeof file === 'string' ? spec.repo : file.repo;
      const remoteName = typeof file === 'string' ? name : (file.remote || name);
      const remote = repo ? await repository(repo) : null;
      const local = path.join(models, name);
      const sibling = remote?.siblings.find(s => s.rfilename === remoteName);
      let sha256, bytes;
      if (sibling?.lfs) {
        sha256 = sibling.lfs.sha256;
        bytes = sibling.lfs.size;
      } else if (fs.existsSync(local)) {
        sha256 = await hashFile(local);
        bytes = fs.statSync(local).size;
      } else {
        throw new Error(`No immutable artifact metadata for ${name}`);
      }
      const matches = sibling?.lfs?.sha256 === sha256;
      files.push({ name, sha256, bytes, url: matches ? `https://huggingface.co/${repo}/resolve/${remote.sha}/${remoteName}` : '' });
    }
    const identity = crypto.createHash('sha256').update(files.map(f => f.sha256).join(':')).digest('hex');
    const slug = spec.files[0].replace(/\.(gguf|bin)$/, '').toLowerCase().replaceAll('.', '-');
    records.push({ ...spec, id: `${slug}-${identity.slice(0, 16)}`, files });
  }
  const q = JSON.stringify;
  const entries = records.map(p => `    { .id=${q(p.id)}, .name=${q(p.name)}, .license=${q(p.license)}, .runtime=${q(p.runtime)},\n      .runtime_abi=1, .feature=${p.feature}, .quality=${p.quality}, .backends=LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL, .memory_bytes=UINT64_C(${p.memory}),\n      .file_count=${p.files.length}, .files={\n${p.files.map(f => `        { .name=${q(f.name)}, .sha256=${q(f.sha256)}, .url=${q(f.url)}, .bytes=UINT64_C(${f.bytes}) }`).join(',\n')}\n      } }`).join(',\n');
  fs.writeFileSync(path.join(core, 'loina_catalog.c'), `/* Immutable artifact identities. Regenerate with tools/catalog.cjs. */\n#include "loina.h"\n\nstatic const loina_package catalog[] = {\n${entries}\n};\n\nsize_t loina_catalog_count(void) { return sizeof(catalog)/sizeof(catalog[0]); }\nconst loina_package *loina_catalog_get(size_t index) {\n    return index < loina_catalog_count() ? &catalog[index] : NULL;\n}\n`);
  fs.writeFileSync(path.join(core, 'catalog.json'), JSON.stringify(records, null, 2) + '\n');
  console.log(records.map(p => `${p.name}: ${p.files.every(f => f.url) ? 'pinned download' : 'local import'} (${p.id})`).join('\n'));
}
main().catch(error => { console.error(error); process.exitCode = 1; });

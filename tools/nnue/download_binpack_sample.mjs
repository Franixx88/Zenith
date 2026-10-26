// Fetch exactly one complete BINP chunk, not a multi-gigabyte dataset.
import * as fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.resolve(process.argv[2] ?? path.join(root, 'target/binpack-samples'));
const sources = JSON.parse(await fs.readFile(path.join(root, 'tools/nnue/dataset-sources.json'), 'utf8'));
const source = sources.sources.find(s => s.id === 'adamtwiss/t91-binpacks-filtered');
const file = source.files.find(f => f.name.includes('2026-04-apr'));
const base = `https://huggingface.co/datasets/${source.id}/resolve/${source.revision}/${file.name}`;
async function range(start, end) {
    const response = await fetch(`${base}?download=true&zenith_range=${start}-${end}`, {
        headers: { Range: `bytes=${start}-${end}` }, signal: AbortSignal.timeout(60000),
    });
    if (response.status !== 206 || response.headers.get('content-range') !== `bytes ${start}-${end}/${file.bytes}`) {
        await response.body?.cancel();
        throw Error('Server did not honor the bounded Range request');
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes.length !== end - start + 1) throw Error('Truncated range');
    return bytes;
}
const header = await range(0, 7);
const size = header.readUInt32LE(4);
if (header.toString('ascii', 0, 4) !== 'BINP' || size < 34 || size > 8 * 1024 * 1024) {
    throw Error('Unexpected first chunk header/size');
}
const bytes = await range(0, size + 7);
if (!header.equals(bytes.subarray(0, 8))) throw Error('Chunk header changed between requests');
await fs.mkdir(out, { recursive: true });
const target = path.join(out, 't91-april-first.binpack');
try {
    const existing = await fs.readFile(target);
    if (!existing.equals(bytes)) throw Error('Refusing to replace a different existing sample');
} catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const temporary = `${target}.part`;
    await fs.writeFile(temporary, bytes, { flag: 'wx' });
    await fs.rename(temporary, target);
}
const metadata = { source: source.id, revision: source.revision, file: file.name,
    range: [0, size + 7], bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
    full_source_sha256: file.sha256, full_source_hash_verified: false, partial_source: true };
await fs.writeFile(path.join(out, 'source.json'), JSON.stringify(metadata, null, 2) + '\n');
const recipe = { format: 'zenith-binpack-stream-v1', files: ['t91-april-first.binpack'],
    source: `${source.id}@${source.revision}; first complete chunk; range 0-${size + 7}`,
    license: 'ODbL', score_num: 1, score_den: 1,
    score_unit: 'raw teacher score; identity scale for compatibility test, not calibrated centipawns',
    val_mod: 20, max_cp: 3000, quiet: true, validation_seed: 1 };
await fs.writeFile(path.join(out, 't91-smoke.json'), JSON.stringify(recipe, null, 2) + '\n');
console.log(`Saved complete first chunk: ${bytes.length} bytes; sample SHA256 ${metadata.sha256}`);
console.log('This is a bounded sample; the full source checksum was not checked.');

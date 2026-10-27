// Pinned public binpacks: bounded HTTP ranges, resume, exact size and SHA-256.
// The compressed files stay compressed. Node.js 20+, no npm dependencies.
import * as fs from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(await fs.readFile(path.resolve(process.argv[2] ?? path.join(here, 'corpora/first-50gb.json')), 'utf8'));
const out = path.resolve(process.argv[3] ?? path.join(here, '../../data/nnue/sf-first-50gb'));
if (manifest.format !== 'zenith-binpack-corpus-v1'
    || !/^[\w.-]+\/[\w.-]+$/.test(manifest.source)
    || !/^[a-f0-9]{40}$/.test(manifest.revision) || !manifest.license
    || !Array.isArray(manifest.files) || !manifest.files.length) throw Error('Invalid source manifest');
for (const f of manifest.files) {
    if (!/^[\w.-]+\.binpack$/.test(f.name) || !Number.isSafeInteger(f.bytes) || f.bytes <= 0
        || !/^[a-f0-9]{64}$/.test(f.sha256)) throw Error('Invalid pinned file');
}
if (new Set(manifest.files.map(f => f.name)).size !== manifest.files.length) throw Error('Duplicate files');
await fs.mkdir(out, { recursive: true });
const size = async file => (await fs.stat(file).catch(e => { if (e.code === 'ENOENT') return { size: 0 }; throw e; })).size;
const digest = async file => {
    const h = createHash('sha256');
    for await (const bytes of createReadStream(file)) h.update(bytes);
    return h.digest('hex');
};
let required = 0;
for (const file of manifest.files) {
    const target = path.join(out, file.name);
    const current = await size(target);
    if (current && current !== file.bytes) throw Error(`Refusing to replace existing ${target}`);
    if (!current) required += file.bytes - await size(target + '.part');
}
const disk = await fs.statfs(out, { bigint: true });
if (disk.bavail * disk.bsize < BigInt(Math.max(0, required)) + 10n * 1024n ** 3n) throw Error('Need corpus space plus 10 GiB reserve');

const total = manifest.files.reduce((sum, file) => sum + file.bytes, 0);
const verified = [];
const start = Date.now();
let current = '', received = 0, phase = 'starting', last = Date.now(), previous = 0;
let progressWrite = Promise.resolve();
function progress() {
    const now = Date.now(), rate = (received - previous) / Math.max(0.001, (now - last) / 1000);
    const state = { file: current, phase, received, total_bytes: total, verified_files: verified.length,
        elapsed_seconds: (now - start) / 1000, bytes_per_second: Math.max(0, rate) };
    console.log(JSON.stringify(state));
    progressWrite = progressWrite.then(async () => {
        await fs.writeFile(path.join(out, 'progress.json.tmp'), JSON.stringify(state));
        await fs.rename(path.join(out, 'progress.json.tmp'), path.join(out, 'progress.json'));
    });
    last = now; previous = received;
}
const timer = setInterval(progress, 10000);
try {
    for (const file of manifest.files) {
        current = file.name;
        const target = path.join(out, file.name), partial = target + '.part';
        if (await size(target)) {
            phase = 'verifying existing file';
            if (await digest(target) !== file.sha256) throw Error(`Existing file SHA256 mismatch: ${target}`);
            received += file.bytes;
        } else {
            let offset = await size(partial);
            if (offset > file.bytes) throw Error('Oversized partial file; inspect before resuming');
            received += offset;
            let failures = 0;
            while (offset < file.bytes) {
                const end = Math.min(file.bytes - 1, offset + 128 * 1024 * 1024 - 1);
                const base = `https://huggingface.co/datasets/${manifest.source}/resolve/${manifest.revision}/${encodeURIComponent(file.name)}`;
                phase = 'downloading';
                try {
                    const response = await fetch(`${base}?download=true&zenith_range=${offset}-${end}`, {
                        headers: { Range: `bytes=${offset}-${end}`, 'Accept-Encoding': 'identity' },
                        signal: AbortSignal.timeout(180000),
                    });
                    const expected = `bytes ${offset}-${end}/${file.bytes}`;
                    if (response.status !== 206 || response.headers.get('content-range') !== expected) {
                        await response.body?.cancel();
                        throw Error(`Expected bounded ${expected}, HTTP ${response.status}`);
                    }
                    let count = 0;
                    const meter = new Transform({ transform(bytes, encoding, done) {
                        count += bytes.length;
                        if (count > end - offset + 1) return done(Error('Response exceeded requested range'));
                        received += bytes.length;
                        done(null, bytes);
                    } });
                    await pipeline(Readable.fromWeb(response.body), meter, createWriteStream(partial, { flags: 'a' }));
                    if (count !== end - offset + 1) throw Error('Truncated range');
                    offset = end + 1;
                    failures = 0;
                } catch (error) {
                    offset = await size(partial);
                    received = verified.reduce((sum, f) => sum + f.bytes, 0) + offset;
                    if (++failures >= 8) throw error;
                    console.error(`Retry ${failures}: ${file.name}: ${error.message}`);
                    await new Promise(resolve => setTimeout(resolve, Math.min(30000, failures * 2000)));
                }
            }
            phase = 'verifying SHA256';
            if (await size(partial) !== file.bytes || await digest(partial) !== file.sha256) throw Error(`Downloaded SHA256 mismatch: ${file.name}`);
            // Existing final files were checked above and are never overwritten.
            await fs.link(partial, target);
            await fs.unlink(partial);
        }
        verified.push(file);
        await fs.writeFile(path.join(out, 'verified.json.tmp'), JSON.stringify({ ...manifest, verified_at: new Date().toISOString(), verified }, null, 2));
        await fs.rename(path.join(out, 'verified.json.tmp'), path.join(out, 'verified.json'));
        console.log(`VERIFIED ${file.name} ${file.sha256}`);
        progress();
    }
    phase = 'complete'; progress();
} finally {
    clearInterval(timer);
    await progressWrite;
}

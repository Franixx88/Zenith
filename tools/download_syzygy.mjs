// Full 3–5 piece WDL+DTZ set. Node.js 20+, no npm dependencies.
// Downloads are bounded, resumable, streamed to disk and SHA256-checked.
import * as fs from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const destination = path.resolve(process.argv[2] ?? path.join(here, '../syzygy/3-4-5'));
const { source, files } = JSON.parse(await fs.readFile(path.join(here, 'syzygy-3-4-5-manifest.json'), 'utf8'));
for (const file of files) {
    if (!/^[KQRBNPv]+\.rtb[wz]$/.test(file.file) || !/^[a-f0-9]{64}$/.test(file.sha256)
        || !Number.isSafeInteger(file.bytes) || file.bytes <= 0
        || file.url !== `https://tablebase.lichess.ovh/tables/standard/3-4-5-${file.file.endsWith('w') ? 'wdl' : 'dtz'}/${file.file}`) {
        throw Error('Invalid pinned manifest');
    }
}
await fs.mkdir(destination, { recursive: true });
const size = async file => (await fs.stat(file).catch(e => { if (e.code === 'ENOENT') return { size: 0 }; throw e; })).size;
const digest = async file => {
    const hash = createHash('sha256');
    for await (const data of createReadStream(file)) hash.update(data);
    return hash.digest('hex');
};
let remaining = 0;
for (const file of files) {
    if (await size(path.join(destination, file.file)) !== file.bytes) remaining += file.bytes;
}
const disk = await fs.statfs(destination, { bigint: true });
if (disk.bavail * disk.bsize < BigInt(remaining) + 5n * 1024n ** 3n) {
    throw Error('Not enough free space for the set plus a 5 GiB reserve');
}
const total = files.reduce((sum, f) => sum + f.bytes, 0);
console.log(`Syzygy: ${files.length} files, ${total} bytes -> ${destination}`);
let next = 0, complete = 0, verifiedBytes = 0;
const start = Date.now();
const progress = () => console.log(`Verified ${complete}/${files.length}, ${(verifiedBytes / 1048576).toFixed(1)}/${(total / 1048576).toFixed(1)} MiB, elapsed ${((Date.now() - start) / 1000).toFixed(0)}s`);
const timer = setInterval(progress, 10000);
async function download(file) {
    const target = path.join(destination, file.file);
    if (await size(target) === file.bytes && await digest(target) === file.sha256) return;
    const partial = `${target}.part`;
    for (let attempt = 1; attempt <= 5; attempt++) {
        try {
            let offset = await size(partial);
            if (offset > file.bytes) { await fs.truncate(partial, 0); offset = 0; }
            if (offset < file.bytes) {
                const response = await fetch(file.url, {
                    headers: offset ? { Range: `bytes=${offset}-` } : {},
                    signal: AbortSignal.timeout(180000),
                });
                if (response.status !== 200 && response.status !== 206) throw Error(`HTTP ${response.status}`);
                if (response.status === 206) {
                    const range = response.headers.get('content-range');
                    if (!range?.startsWith(`bytes ${offset}-`) || !range.endsWith(`/${file.bytes}`)) {
                        await response.body.cancel();
                        throw Error(`Unexpected Content-Range: ${range}`);
                    }
                } else { offset = 0; }
                await pipeline(Readable.fromWeb(response.body), createWriteStream(partial, { flags: offset ? 'a' : 'w' }));
            }
            if (await size(partial) !== file.bytes) throw Error('Downloaded length mismatch');
            if (await digest(partial) !== file.sha256) {
                await fs.truncate(partial, 0);
                throw Error('SHA256 mismatch');
            }
            await fs.rename(partial, target);
            return;
        } catch (error) {
            if (attempt === 5) throw Error(`${file.file}: ${error.message}`, { cause: error });
            console.error(`Retry ${attempt}/5 ${file.file}: ${error.message}`);
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
    }
}
try {
    // Four network transfers; this does not limit the engine's eight CPU threads.
    const workers = Array.from({ length: 4 }, async () => {
        while (next < files.length) {
            const file = files[next++];
            await download(file);
            complete++;
            verifiedBytes += file.bytes;
        }
    });
    const outcomes = await Promise.allSettled(workers);
    const failed = outcomes.find(r => r.status === 'rejected');
    if (failed) throw failed.reason;
    const record = { source, verified_at: new Date().toISOString(), bytes: total, files };
    const staging = path.join(destination, 'verified.json.part');
    await fs.writeFile(staging, JSON.stringify(record, null, 2) + '\n');
    await fs.rename(staging, path.join(destination, 'verified.json'));
    progress();
    console.log('Complete: all SHA256 checks passed.');
} finally {
    clearInterval(timer);
}

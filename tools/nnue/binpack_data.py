"""Bounded IPC to the process-isolated Rust binpack decoder (no expanded files)."""
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
from records import RECORD, check_records

MAGIC = b"ZBSTRM1\0"


def read_exact(stream, size):
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        read = stream.readinto(view[offset:])
        if not read:
            raise ValueError("binpack decoder ended without a complete success frame")
        offset += read
    return chunks


class BinpackSource:
    def __init__(self, recipe, reader=None):
        self.path = Path(recipe).resolve()
        text = self.path.read_text(encoding="utf-8")
        self.config = json.loads(text)
        cfg = self.config
        allowed = {"format", "files", "source", "license", "score_num", "score_den", "score_unit",
                   "val_mod", "max_cp", "quiet", "validation_seed"}
        if set(cfg) - allowed or cfg.get("format") != "zenith-binpack-stream-v1":
            raise ValueError("unsupported binpack recipe/fields")
        if not isinstance(cfg.get("files"), list) or not cfg["files"]:
            raise ValueError("binpack recipe needs files")
        for key in ["source", "license", "score_unit"]:
            if not isinstance(cfg.get(key), str) or not cfg[key].strip():
                raise ValueError(f"binpack recipe needs explicit {key}")
        for key in ["score_num", "score_den"]:
            if type(cfg.get(key)) is not int or not 1 <= cfg[key] <= 1_000_000:
                raise ValueError("explicit positive integer score_num/score_den required")
        cfg.setdefault("val_mod", 20)
        cfg.setdefault("max_cp", 3000)
        cfg.setdefault("quiet", True)
        cfg.setdefault("validation_seed", 1)
        if type(cfg["quiet"]) is not bool or type(cfg["val_mod"]) is not int or not 2 <= cfg["val_mod"] < 2**64:
            raise ValueError("invalid split/quiet policy")
        if type(cfg["max_cp"]) is not int or not 1 <= cfg["max_cp"] <= 30000:
            raise ValueError("max_cp must be 1..30000")
        if type(cfg["validation_seed"]) is not int or not 0 <= cfg["validation_seed"] < 2**64:
            raise ValueError("validation_seed must fit u64")
        self.files = []
        for name in cfg["files"]:
            if not isinstance(name, str):
                raise ValueError("files must contain paths")
            file = (self.path.parent / name).resolve()
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError(f"empty or missing binpack: {file}")
            self.files.append(file)
        if len(set(self.files)) != len(self.files):
            raise ValueError("duplicate binpack files")
        default = Path(__file__).resolve().parents[2] / "target/binpack/release" / ("zenith-binpack.exe" if os.name == "nt" else "zenith-binpack")
        self.reader = Path(reader or default).resolve()
        if not self.reader.is_file():
            raise ValueError("build tools/binpack first (see docs/binpack-stream.md)")
        entries = [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in self.files]
        identity = {"recipe": cfg, "files": entries,
                    "reader_sha256": hashlib.sha256(self.reader.read_bytes()).hexdigest(),
                    "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.manifest = json.dumps({"format": cfg["format"], "source": cfg["source"], "license": cfg["license"],
                                    "score_pov": "side_to_move", "score_unit": cfg["score_unit"],
                                    "score_scale": [cfg["score_num"], cfg["score_den"]], "identity": identity}, sort_keys=True)
        self.last_stats = {}

    def validation(self, rows):
        stream = NativeEpoch(self, "val", self.config["validation_seed"], rows, 4096, 4096, None, False)
        try:
            parts = list(stream)
        finally:
            stream.close()
        if not parts:
            raise ValueError("binpack validation split has no usable records")
        return np.concatenate(parts)

    def epoch(self, rows, batch, block_rows, rng):
        seed = int(rng.integers(0, 2**64 - 1, dtype=np.uint64))
        return NativeEpoch(self, "train", seed, rows, batch, block_rows, rng, True)


class NativeEpoch:
    def __init__(self, source, split, seed, rows, batch, block_rows, rng, as_torch):
        if rows <= 0 or batch <= 0 or block_rows < batch:
            raise ValueError("invalid native epoch limits")
        self.source, self.split, self.seed = source, split, seed
        self.rows, self.batch, self.block_rows, self.rng, self.as_torch = rows, batch, block_rows, rng, as_torch
        self.process = None
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.iterator = self._generate()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()

    def close(self):
        self.cancel()
        self.iterator.close()

    def _batches(self, records):
        if not self.as_torch:
            yield records
            return
        order = self.rng.permutation(len(records))
        for i in range(0, len(order), self.batch):
            sample = records[np.sort(order[i:i + self.batch])]
            yield torch.from_numpy(np.ascontiguousarray(sample["bits"])), torch.from_numpy(sample["score"].astype(np.float32))

    def _generate(self):
        cfg = self.source.config
        command = [str(self.source.reader), "stream", "--split", self.split, "--val-mod", str(cfg["val_mod"]),
                   "--seed", str(self.seed), "--limit", str(self.rows), "--batch-rows", str(min(4096, self.block_rows)),
                   "--max-cp", str(cfg["max_cp"]), "--score-num", str(cfg["score_num"]), "--score-den", str(cfg["score_den"])]
        for file in self.source.files:
            command.extend(["--input", str(file)])
        if not cfg["quiet"]:
            command.append("--keep-tactical")
        with tempfile.TemporaryFile() as diagnostics:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=diagnostics)
            with self.lock:
                self.process = process
                if self.cancelled.is_set():
                    process.kill()
            try:
                if read_exact(process.stdout, 8) != MAGIC:
                    raise ValueError("unsupported binpack stream protocol")
                parts, count, received = [], 0, 0
                while True:
                    size = struct.unpack("<I", read_exact(process.stdout, 4))[0]
                    if size == 0:
                        break
                    if size > 65536 or received + size > self.rows:
                        raise ValueError("binpack frame exceeds configured bounds")
                    records = np.frombuffer(read_exact(process.stdout, size * RECORD.itemsize), dtype=RECORD)
                    check_records(records)
                    received += size
                    # Frames never exceed block_rows, but a partial block can straddle frames.
                    while len(records):
                        take = min(self.block_rows - count, len(records))
                        parts.append(records[:take])
                        count += take
                        records = records[take:]
                        if count == self.block_rows:
                            yield from self._batches(np.concatenate(parts))
                            parts, count = [], 0
                footer_size = struct.unpack("<I", read_exact(process.stdout, 4))[0]
                if footer_size > 65536:
                    raise ValueError("binpack footer too large")
                stats = json.loads(read_exact(process.stdout, footer_size))
                if stats.get("emitted") != received or received == 0:
                    raise ValueError("binpack footer count mismatch or empty split")
                if process.stdout.read(1) != b"" or process.wait(timeout=10) != 0:
                    raise ValueError("binpack decoder failed after emitting records")
                if self.as_torch and received != self.rows:
                    raise ValueError(f"binpack has only {received} usable training records; reduce epoch-rows")
                self.source.last_stats[self.split] = stats
                if parts:
                    yield from self._batches(np.concatenate(parts))
            except Exception as error:
                diagnostics.seek(0)
                detail = diagnostics.read(16384).decode("utf-8", errors="replace")
                raise ValueError(f"binpack stream failed: {error}\n{detail}") from error
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                process.stdout.close()
                with self.lock:
                    self.process = None

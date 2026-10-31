"""Bounded readers for Zenith's explicit 100-byte, little-endian records."""
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

RECORD = np.dtype([("bits", "u1", (96,)), ("score", "<i2"), ("depth", "u1"), ("flags", "u1")])


def record_count(path, allow_empty=False):
    size = Path(path).stat().st_size
    if size % RECORD.itemsize:
        raise ValueError(f"{path}: truncated record (size must be a multiple of 100)")
    count = size // RECORD.itemsize
    if not count and not allow_empty:
        raise ValueError(f"{path}: no records")
    return count


@contextmanager
def mapped_records(path):
    count = record_count(path)
    mapping = np.memmap(path, dtype=RECORD, mode="r", shape=(count,))
    try:
        yield mapping
    finally:
        mapping._mmap.close()


def check_records(records):
    flags = records["flags"]
    if np.any((flags != 0) & ((flags < 128) | (flags > 130))):
        raise ValueError("unsupported record flags")
    if np.any(np.abs(records["score"].astype(np.int32)) > 30000):
        raise ValueError("record score outside supported range; VALUE_NONE/mates must be filtered")


def outcomes(records):
    """Known side-to-move results as probabilities. Never invent missing draws."""
    check_records(records)
    if np.any(records["flags"] == 0):
        raise ValueError("WDL training requires known game results for every record")
    return (records["flags"] & 3).astype(np.float32) / 2


def sample_records(path, rows, seed=1):
    if rows <= 0:
        raise ValueError("sample size must be positive")
    with mapped_records(path) as mapping:
        # random.sample(range) uses memory bounded by the sample size, not the file.
        indices = sorted(random.Random(seed).sample(range(len(mapping)), min(rows, len(mapping))))
        sample = np.array(mapping[indices], copy=True)
    check_records(sample)
    return sample


def shuffled_batches(files, batch, block_rows, rng):
    if batch <= 0 or block_rows < batch:
        raise ValueError("batch must be positive and block-rows must be >= batch")
    for index in rng.permutation(len(files)):
        file = files[index]
        if record_count(file, allow_empty=True) == 0:
            continue
        with mapped_records(file) as mapping:
            blocks = (len(mapping) + block_rows - 1) // block_rows
            for block in rng.permutation(blocks):
                start = int(block) * block_rows
                records = np.array(mapping[start:start + block_rows], copy=True)
                check_records(records)
                order = rng.permutation(len(records))
                for offset in range(0, len(order), batch):
                    yield records[np.sort(order[offset:offset + batch])]


def dataset_identity(directory, files):
    directory = Path(directory)
    if (directory / "INCOMPLETE").exists():
        raise ValueError("dataset is INCOMPLETE")
    manifest = (directory / "manifest.txt").read_text(encoding="utf-8")
    fields = dict(line.split("=", 1) for line in manifest.splitlines() if "=" in line)
    for key, expected in {"format": "zenith-nnue-records-v1", "record_bytes": "100",
                          "perspective": "side_to_move", "score": "centipawns_side_to_move"}.items():
        if fields.get(key) != expected:
            raise ValueError(f"unsupported dataset manifest: {key}")
    entries = []
    for file in [*files, directory / "val.bin"]:
        stat = file.stat()
        entries.append((file.name, stat.st_size, stat.st_mtime_ns))
    # Fast local identity, not a full-content checksum or authentication.
    identity = hashlib.sha256((manifest + json.dumps(entries)).encode()).hexdigest()
    return manifest, identity

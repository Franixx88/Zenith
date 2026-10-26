#!/usr/bin/env python3
"""Copy complete BINP chunks from a growing download for compatibility tests.

This is a prefix sample, not a verified full corpus or representative training
set. The manifest records that distinction and hashes the immutable sample.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunks", type=int, default=64)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    name = args.source.name.removesuffix(".part")
    entry = next(f for f in manifest["files"] if f["name"] == name)
    if not 1 <= args.chunks <= 1024:
        ap.error("chunks must be 1..1024")
    # Snapshot the available length; ignore the final incomplete chunk.
    available = args.source.stat().st_size
    chunks = []
    with args.source.open("rb") as source:
        for _ in range(args.chunks):
            offset = source.tell()
            if offset + 8 > available:
                break
            header = source.read(8)
            length = struct.unpack("<I", header[4:])[0]
            if header[:4] != b"BINP" or not 34 <= length <= 100 * 1024 * 1024:
                raise ValueError(f"invalid BINP header at {offset}")
            if offset + 8 + length > available:
                break
            chunks.append((offset, length + 8))
            source.seek(length, 1)
    if not chunks:
        raise ValueError("no complete chunks available")
    args.out.mkdir(parents=True, exist_ok=False)
    sample = args.out / "prefix.binpack"
    digest = hashlib.sha256()
    with args.source.open("rb") as source, sample.open("xb") as dest:
        for offset, length in chunks:
            source.seek(offset)
            remaining = length
            while remaining:
                block = source.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("source was truncated while sampling")
                dest.write(block)
                digest.update(block)
                remaining -= len(block)
    metadata = dict(source=manifest["source"], revision=manifest["revision"],
                    file=entry, range=[0, sample.stat().st_size - 1],
                    chunks=len(chunks), sample_sha256=digest.hexdigest(),
                    full_source_hash_verified=False, partial_source=True)
    (args.out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    recipe = dict(format="zenith-binpack-stream-v1", files=[sample.name],
                  source=f"{manifest['source']}@{manifest['revision']}; {name}; prefix sample only",
                  license=manifest["license"], score_num=1, score_den=1,
                  score_unit="raw legacy Stockfish/Lc0 teacher units; identity-scale compatibility pilot",
                  val_mod=20, max_cp=3000, quiet=True, validation_seed=1)
    (args.out / "pilot.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

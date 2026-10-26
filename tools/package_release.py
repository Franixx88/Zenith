"""Package a verified native UCI executable with source identity and licenses.

Standard library only. Existing release archives are never overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(binary: Path, target: str, output: Path, commit: str):
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Release version must be X.Y.Z")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full Git commit SHA is required")
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("This release target requires a native x86-64 build")
    windows = target == "windows-x86_64"
    if sys.platform != ("win32" if windows else "linux"):
        raise ValueError("Package the executable on its native operating system")
    binary = binary.resolve(strict=True)
    env = dict(os.environ, ZENITH_SYZYGY_PATH="<empty>")
    probe = subprocess.run([str(binary)], input="uci\nquit\n", text=True,
                           capture_output=True, env=env, check=True, timeout=15)
    if f"id name Zenith {version}" not in probe.stdout.splitlines():
        raise ValueError("Executable UCI version does not match Cargo.toml")
    name = "zenith.exe" if windows else "zenith"
    binary_bytes = binary.read_bytes()
    digest = hashlib.sha256(binary_bytes).hexdigest()
    folder = f"zenith-v{version}-{target}"
    source = f"https://github.com/Franixx88/Zenith/tree/{commit}"
    manifest = {"version": version, "commit": commit, "platform": target,
                "binary": name, "sha256": digest, "source": source,
                "license": "GPL-3.0-or-later", "trained_network_included": False}
    readme = (
        f"Zenith {version} - native UCI chess engine\n"
        f"Source snapshot: {source}\n"
        f"Executable: {name}\n\n"
        "Select the executable in a UCI-compatible chess GUI.\n"
        "Send 'uci' to inspect the version and supported options.\n"
        "No trained NNUE weights or Syzygy tables are bundled.\n"
        "Without EvalFile the engine uses classical evaluation.\n"
        "AVX2 is selected at runtime with a scalar fallback.\n"
        "The optional browser analysis board is available in the source tree.\n\n"
        "BUILD.json records the exact source commit and executable SHA-256.\n"
        "License: GPL-3.0-or-later; see LICENSE and THIRD_PARTY.md.\n"
    ).encode("utf-8")
    entries = [(name, binary_bytes, 0o755),
               ("BUILD.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o644),
               ("README.txt", readme, 0o644)]
    for path in [ROOT / "LICENSE", ROOT / "THIRD_PARTY.md", ROOT / "docs/stockfish-authors.txt",
                 *sorted((ROOT / "docs/licenses").rglob("*"))]:
        if path.is_file():
            entries.append((path.relative_to(ROOT).as_posix(), path.read_bytes(), 0o644))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (folder + (".zip" if windows else ".tar.gz"))
    # Exclusive creation protects a previous published/staged archive.
    if windows:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as writer:
            for path, content, mode in entries:
                entry = zipfile.ZipInfo(f"{folder}/{path}", date_time=(2026, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = (0o100000 | mode) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                writer.writestr(entry, content)
    else:
        with archive.open("xb") as file:
            with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as writer:
                    for path, content, mode in entries:
                        entry = tarfile.TarInfo(f"{folder}/{path}")
                        entry.size, entry.mode, entry.mtime = len(content), mode, 0
                        writer.addfile(entry, io.BytesIO(content))
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(archive.name + ".sha256")
    with checksum.open("x", encoding="ascii", newline="\n") as file:
        file.write(f"{archive_hash}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "archive_sha256": archive_hash, "build": manifest}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--platform", choices=["windows-x86_64", "linux-x86_64"], required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    commit = args.commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    package(args.binary, args.platform, args.out, commit)


if __name__ == "__main__":
    main()

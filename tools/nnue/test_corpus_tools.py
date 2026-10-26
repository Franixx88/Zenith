"""Prefix extraction keeps only complete chunks and never replaces samples."""
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_first_training as job

SCRIPT = Path(__file__).with_name("extract_binpack_prefix.py")


class PrefixTests(unittest.TestCase):
    def run_sample(self, contents, success=True, chunks=64):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "fixture.binpack.part"
        source.write_bytes(contents)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(source="test/corpus", revision="a" * 40,
            license="test", files=[dict(name="fixture.binpack", bytes=9999, sha256="b" * 64)])))
        command = [sys.executable, str(SCRIPT), "--manifest", str(manifest),
                   "--source", str(source), "--out", str(root / "sample"), "--chunks", str(chunks)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        self.assertEqual(source.read_bytes(), contents)
        return root / "sample", command

    def test_complete_chunks_only(self):
        chunk = b"BINP" + struct.pack("<I", 34) + bytes(range(34))
        out, command = self.run_sample(chunk * 2 + chunk[:21])
        self.assertEqual((out / "prefix.binpack").read_bytes(), chunk * 2)
        metadata = json.loads((out / "source.json").read_text())
        self.assertEqual(metadata["sample_sha256"], hashlib.sha256(chunk * 2).hexdigest())
        self.assertEqual(metadata["range"], [0, 83])
        self.assertFalse(metadata["full_source_hash_verified"])
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual((out / "prefix.binpack").read_bytes(), chunk * 2)

    def test_missing_complete_chunk(self):
        self.run_sample(b"BINP" + struct.pack("<I", 34) + b"incomplete", success=False)

    def test_bad_magic_and_bound(self):
        self.run_sample(b"NOPE" + struct.pack("<I", 34) + bytes(34), success=False)
        self.run_sample(b"BINP" + struct.pack("<I", 100 * 1024 * 1024 + 1), success=False)


class JobFailureTests(unittest.TestCase):
    def test_failed_download_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text('{}')
            out = root / "job"
            argv = [str(job.__file__), "--out", str(out), "--manifest", str(fixture),
                    "--recipe", str(fixture), "--engine", str(fixture),
                    "--reader", str(fixture), "--nnue-harness", str(fixture)]
            success = SimpleNamespace(pid=100, wait=lambda timeout=None: 0)
            failure = SimpleNamespace(pid=101, wait=lambda timeout=None: 2)
            with patch.object(sys, "argv", argv), patch.object(sys, "stdout", io.StringIO()), \
                    patch.object(job.subprocess, "Popen", side_effect=[success, failure]) as spawn:
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    job.main()
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["stage"], "failed")
            self.assertIn("download failed", status["error"])
            self.assertEqual(spawn.call_count, 2)
            self.assertFalse((out / "network").exists())


if __name__ == "__main__":
    unittest.main()

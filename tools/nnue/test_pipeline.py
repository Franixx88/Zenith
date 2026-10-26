"""Run after cargo build --release --example prepare_nnue --example nnue."""
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import records as R
import train_nnue as T

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "target/release/examples/prepare_nnue.exe"
NNUE = ROOT / "target/release/examples/nnue.exe"
TOOLS = ROOT / "tools/nnue"


def command(args, success=True, **kwargs):
    run = subprocess.run([str(arg) for arg in args], capture_output=True, text=True, timeout=120, **kwargs)
    if (run.returncode == 0) != success:
        raise AssertionError(f"{args}: exit {run.returncode}\n{run.stdout}\n{run.stderr}")
    return run


def fixture_rows(count=512):
    rng = random.Random(7)
    board = chess.Board()
    rows = []
    while len(rows) < count:
        if board.is_game_over() or board.fullmove_number > 80:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        if not board.is_check() and any(board.legal_moves):
            # Arithmetic/smoke labels only: never represent this as a trained teacher.
            values = [0, 100, 320, 335, 500, 950, 0]
            cp = sum(values[p.piece_type] * (1 if p.color else -1) for p in board.piece_map().values())
            rows.append(f"{board.fen()}\t24\t{max(-3000, min(3000, cp))}\t\n")
    return rows


class RecordsTests(unittest.TestCase):
    def test_bounded_reader_visits_all_rows_and_samples_without_fromfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "records.bin"
            arr = np.zeros(37, dtype=R.RECORD)
            arr["score"] = np.arange(len(arr))
            arr.tofile(file)
            with patch.object(np, "fromfile", side_effect=AssertionError("unbounded reader")):
                sample = R.sample_records(file, 5)
                seen = np.concatenate([batch["score"] for batch in R.shuffled_batches([file], 3, 7, np.random.default_rng(1))])
            self.assertEqual(len(sample), 5)
            np.testing.assert_array_equal(np.sort(seen), arr["score"])

    def test_empty_truncated_invalid_flags_and_empty_validation_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "data.bin"
            for data in [b"", b"x"]:
                file.write_bytes(data)
                with self.assertRaises(ValueError):
                    R.sample_records(file, 1)
            with self.assertRaises(ValueError):
                R.sample_records(file, 0)
            arr = np.zeros(1, dtype=R.RECORD)
            arr["flags"] = 1
            arr.tofile(file)
            with self.assertRaises(ValueError):
                R.sample_records(file, 1)
            with self.assertRaises(ValueError):
                T.validate(None, [], 400, None, None)

    def test_prefetch_failure_and_early_close_join_producer(self):
        def failing():
            yield 1
            raise ValueError("source failed")
        stream = T.prefetch(failing(), depth=1)
        self.assertEqual(next(stream), 1)
        with self.assertRaisesRegex(ValueError, "source failed"):
            next(stream)
        closed = []
        def source():
            try:
                yield from range(1000)
            finally:
                closed.append(True)
        stream = T.prefetch(source(), depth=1)
        next(stream)
        stream.close()
        self.assertEqual(closed, [True])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PREPARE.exists() or not NNUE.exists():
            raise RuntimeError("build the prepare_nnue and nnue release examples first")
        cls.tmp = tempfile.TemporaryDirectory(prefix="zenith-nnue-", dir=ROOT / "target")
        cls.base = Path(cls.tmp.name)
        cls.data = cls.base / "dataset"
        cls.rows = fixture_rows()
        command([PREPARE, "--out", cls.data, "--shards", 2, "--val-mod", 5, "--keep-tactical",
                 "--require-complete", "--source", "synthetic-smoke-only"],
                input="".join(cls.rows) + f"#zenith-tsv-complete\t{len(cls.rows)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_preparer_preserves_partial_output_and_rejects_failed_producer(self):
        directory = self.base / "partial"
        directory.mkdir()
        existing = directory / "train-000.bin"
        existing.write_bytes(b"preserve me")
        command([PREPARE, "--out", directory, "--input", self.base / "missing.tsv"], success=False)
        self.assertEqual(existing.read_bytes(), b"preserve me")
        command([PREPARE, "--out", directory], input="".join(self.rows), success=False)
        self.assertEqual(existing.read_bytes(), b"preserve me")
        failed = self.base / "failed-producer"
        command([PREPARE, "--out", failed, "--require-complete"], input="".join(self.rows), success=False)
        self.assertTrue((failed / "INCOMPLETE").exists())
        self.assertFalse((failed / "manifest.txt").exists())

    def test_original_fens_labels_and_feature_split_are_consistent(self):
        expected = {}
        for row in self.rows:
            fen, _, cp, _ = row.split("\t")
            board = chess.Board(fen)
            bits = bytearray()
            for color in [board.turn, not board.turn]:
                for piece in range(1, 7):
                    mask = 0
                    for square in board.pieces(piece, color):
                        mask |= 1 << (square ^ (56 if board.turn == chess.BLACK else 0))
                    bits += mask.to_bytes(8, "little")
            expected[bytes(bits)] = int(cp) * (1 if board.turn else -1)
        partitions = []
        for files in [[self.data / "val.bin"], sorted(self.data.glob("train-*.bin"))]:
            keys = set()
            for file in files:
                if R.record_count(file, allow_empty=True):
                    with R.mapped_records(file) as mapping:
                        for record in mapping:
                            key = bytes(record["bits"])
                            self.assertEqual(int(record["score"]), expected[key])
                            keys.add(key)
            partitions.append(keys)
        self.assertFalse(partitions[0] & partitions[1])
        self.assertEqual(partitions[0] | partitions[1], set(expected))

    def test_parquet_glob_nulls_and_completion_marker(self):
        file = self.base / "fixture.parquet"
        fen = chess.STARTING_FEN
        pq.write_table(pa.table({"fen": [fen, fen], "depth": [24, 25], "cp": [100, None], "mate": [None, 2]}), file)
        run = command([sys.executable, TOOLS / "parquet_to_tsv.py", self.base / "fixture*.parquet"])
        self.assertEqual(run.stdout, f"{fen}\t24\t100\t\n{fen}\t25\t\t2\n#zenith-tsv-complete\t2\n")
        command([sys.executable, TOOLS / "parquet_to_tsv.py", file, "--stride", 0], success=False)

    def test_training_resume_export_parity_and_rejection_guards(self):
        full, split = self.base / "full", self.base / "split"
        common = [sys.executable, TOOLS / "train_nnue.py", "--data", self.data,
                  "--epochs", 2, "--batch-size", 32, "--block-rows", 128,
                  "--val-rows", 64, "--val-chunk", 32, "--threads", 1, "--warmup-steps", 0]
        command([*common, "--out", full])
        command([*common, "--out", split, "--stop-after-epochs", 1])
        command([*common, "--out", split, "--resume", split / "last.pt"])
        self.assertEqual((full / "last.znnue").read_bytes(), (split / "last.znnue").read_bytes())
        a = torch.load(full / "last.pt", weights_only=True)
        b = torch.load(split / "last.pt", weights_only=True)
        self.assertEqual(a["history"], b["history"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]))
        metadata = json.loads((split / "provenance.json").read_text())
        self.assertEqual(metadata["network_sha256_best"], hashlib.sha256((split / "best.znnue").read_bytes()).hexdigest())
        parity = command([sys.executable, TOOLS / "check_parity.py", "--net", split / "best.znnue",
                          "--records", self.data / "val.bin", "--zenith-nnue", NNUE, "--count", 32])
        self.assertIn("mismatches 0", parity.stdout)
        command([sys.executable, TOOLS / "check_parity.py", "--net", split / "best.znnue",
                 "--records", self.data / "val.bin", "--zenith-nnue", NNUE, "--count", 0], success=False)
        before = (split / "last.znnue").read_bytes()
        command([*common, "--out", split], success=False)
        command([*common, "--out", split, "--resume", split / "last.pt", "--lr", "0.02"], success=False)
        self.assertEqual(before, (split / "last.znnue").read_bytes())
        empty = self.base / "empty-validation"
        empty.mkdir()
        (empty / "manifest.txt").write_bytes((self.data / "manifest.txt").read_bytes())
        (empty / "train-000.bin").write_bytes(bytes(100))
        (empty / "val.bin").write_bytes(b"")
        failed = command([*common, "--data", empty, "--out", self.base / "must-not-train"], success=False)
        self.assertIn("no records", failed.stderr)
        self.assertFalse((self.base / "must-not-train").exists())
        # Persistent smoke artifacts for inspection; explicitly synthetic labels.
        artifact = ROOT / "target/nnue-pipeline-smoke"
        artifact.mkdir(exist_ok=True)
        for name in ["best.znnue", "provenance.json", "latest.json"]:
            (artifact / name).write_bytes((split / name).read_bytes())
        (artifact / "parity.txt").write_text(parity.stdout + parity.stderr, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

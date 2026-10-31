"""Native binpack transport, canonical labels/splits, training and exact resume."""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch
import records as R
import train_nnue as T

from binpack_data import BinpackSource, NativeEpoch
from test_pipeline import ROOT, TOOLS, NNUE, command

READER = ROOT / "target/binpack/release" / ("zenith-binpack.exe" if os.name == "nt" else "zenith-binpack")


def native_record(board, score):
    output = bytearray()
    for color in [board.turn, not board.turn]:
        for piece in range(1, 7):
            bits = sum(1 << (sq ^ (56 if not board.turn else 0)) for sq in board.pieces(piece, color))
            output.extend(bits.to_bytes(8, "little"))
    return bytes(output) + struct.pack("<hBB", score, 0, 0)


def key(record):
    value = 0xcbf29ce484222325
    for byte in record[:96]:
        value = ((value ^ byte) * 0x100000001b3) & (2**64 - 1)
    return value


class BinpackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not READER.exists():
            raise RuntimeError("build tools/binpack with --target-dir target/binpack first")
        cls.tmp = tempfile.TemporaryDirectory(prefix="zenith-binpack-", dir=ROOT / "target")
        cls.base = Path(cls.tmp.name)
        cls.file = cls.base / "fixture.binpack"
        cls.entries = []
        cls.results = []
        rng = random.Random(12)
        for game in range(5):
            board = chess.Board()
            for i in range(100):
                if board.is_game_over():
                    break
                mv = rng.choice(list(board.legal_moves))
                material = sum([0, 100, 320, 335, 500, 950, 0][p.piece_type] * (1 if p.color == board.turn else -1) for p in board.piece_map().values())
                score = 32002 if i % 29 == 0 else max(-2500, min(2500, material))
                cls.entries.append((board.copy(), mv.uci(), score))
                cls.results.append((1 if board.turn == (game == 0) else -1) if game < 2 else 0)
                board.push(mv)
        for fen, move, score in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 123),
            ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 12),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", -88),
            (chess.STARTING_FEN.replace(" w ", " b "), "e7e5", -123),
        ]:
            cls.entries.append((chess.Board(fen), move, score))
            cls.results.append(0)
        cls.text = "".join(f"{board.fen(en_passant='fen')}\t{move}\t{score}\t{result}\n" for (board, move, score), result in zip(cls.entries, cls.results))
        command([READER, "pack-fixture", cls.file], input=cls.text)
        cls.recipe = cls.base / "recipe.json"
        cls.config = {"format": "zenith-binpack-stream-v1", "files": [cls.file.name],
                      "source": "synthetic compressed games, regression only", "license": "synthetic",
                      "score_num": 1, "score_den": 1, "score_unit": "synthetic teacher units",
                      "quiet": False, "val_mod": 5, "max_cp": 3000, "validation_seed": 1}
        cls.recipe.write_text(json.dumps(cls.config))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_decoded_chains_and_special_moves_match_originals(self):
        run = command([READER, "inspect", "--input", self.file, "--no-shuffle"])
        decoded = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual(len(decoded), len(self.entries))
        for actual, (board, move, score), result in zip(decoded, self.entries, self.results):
            self.assertEqual(actual["fen"], board.fen(en_passant="fen"))
            self.assertEqual(actual["move"], move)
            self.assertEqual(actual["score_stm"], score)
            self.assertEqual(actual["result_stm"], result)

    def test_upstream_fixed_fixture_and_scores(self):
        run = command([READER, "inspect", "--input", ROOT / "tools/binpack/tests/data/ep1.binpack", "--no-shuffle"])
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual([r["score_stm"] for r in rows], [-201, 254, -220])
        self.assertEqual([r["move"] for r in rows], ["c2c4", "d4d3", "g2b7"])
        self.assertEqual([r["ply"] for r in rows], [68, 69, 70])

    def test_canonical_records_split_sentinels_and_no_disk_expansion(self):
        source = BinpackSource(self.recipe)
        expected = set()
        for board, move, score in self.entries:
            if abs(score) < 32000 and not board.is_check() and not board.is_insufficient_material() and board.halfmove_clock < 100:
                expected.add(native_record(board, score))
        partitions = []
        for split in ["train", "val"]:
            stream = NativeEpoch(source, split, 2, 10000, 32, 64, None, False)
            try:
                records = np.concatenate(list(stream))
            finally:
                stream.close()
            actual = {record.tobytes() for record in records}
            self.assertEqual(actual, {r for r in expected if (key(r) % 5 == 0) == (split == "val")})
            self.assertGreater(source.last_stats[split]["rejected"]["sentinel_or_mate"], 0)
            self.assertTrue(np.all(records["depth"] == 0))
            partitions.append({r[:96] for r in actual})
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(list(self.base.glob("*.bin")))

    def test_bad_headers_truncation_and_decoder_failure_are_not_eof(self):
        for i, data in enumerate([b"BIN", b"NOPE" + struct.pack("<I", 34) + bytes(34),
                                  b"BINP" + struct.pack("<I", 1000) + bytes(34),
                                  b"BINP" + struct.pack("<I", 101 * 1024 * 1024)]):
            file = self.base / f"bad-{i}.binpack"
            file.write_bytes(data)
            recipe = self.base / f"bad-{i}.json"
            recipe.write_text(json.dumps({**self.config, "files": [file.name]}))
            with self.assertRaises(ValueError):
                BinpackSource(recipe).validation(16)

    def test_cancellation_does_not_leave_decoder_running(self):
        source = BinpackSource(self.recipe)
        stream = source.epoch(128, 8, 16, np.random.default_rng(7))
        next(stream)
        process = stream.process
        stream.close()
        self.assertIsNotNone(process.poll())

    def test_results_survive_stream_v2_for_both_colors_and_missing_is_not_draw(self):
        recipe=self.base/"results.json"
        recipe.write_text(json.dumps({**self.config,"format":"zenith-binpack-stream-v2"}))
        source=BinpackSource(recipe)
        stream=NativeEpoch(source,"all",2,10000,32,64,None,False)
        try:
            records=np.concatenate(list(stream))
        finally:
            stream.close()
        labels={native_record(board,score)[:98]:(result+1)/2
                for (board,mv,score),result in zip(self.entries,self.results)
                if abs(score)<=3000 and not board.is_check() and not board.is_game_over()}
        actual=R.outcomes(records)
        for row,result in zip(records,actual):
            self.assertEqual(result,labels[row.tobytes()[:98]])
        self.assertEqual(set(actual),{0,0.5,1})
        with self.assertRaisesRegex(ValueError,"known game results"):
            R.outcomes(BinpackSource(self.recipe).validation(8))
        score=torch.tensor([-100.,0.,100.]);wdl=torch.tensor([1.,.5,0.])
        torch.testing.assert_close(T.training_target(score,200,wdl,.25),
                                   .75*torch.sigmoid(score/200)+.25*wdl)
        torch.testing.assert_close(T.training_target(score,200),torch.sigmoid(score/200),rtol=0,atol=0)
        with self.assertRaises(ValueError):T.training_target(score,200,None,.25)
        with self.assertRaises(ValueError):T.training_target(score,200,torch.tensor([0.,.2,1.]),.25)

    def test_wdl_mixture_training_resumes_exactly_and_keeps_native_export(self):
        recipe=self.base/"wdl-training.json"
        recipe.write_text(json.dumps({**self.config,"format":"zenith-binpack-stream-v2"}))
        full,split=self.base/"wdl-full",self.base/"wdl-split"
        common=[sys.executable,TOOLS/"train_nnue.py","--binpack-config",recipe,
                "--epoch-rows",64,"--epochs",2,"--batch-size",16,"--block-rows",32,
                "--threads",1,"--val-rows",16,"--val-chunk",16,"--warmup-steps",0,"--wdl-weight",.25]
        command([*common,"--out",full])
        command([*common,"--out",split,"--stop-after-epochs",1])
        command([*common,"--out",split,"--resume",split/"last.pt"])
        self.assertEqual((full/"last.znnue").read_bytes(),(split/"last.znnue").read_bytes())
        meta=json.loads((full/"provenance.json").read_text())
        self.assertEqual(meta["contract"]["wdl_weight"],.25)
        sample=self.base/"wdl-validation.bin"
        BinpackSource(recipe).validation(16).tofile(sample)
        run=command([sys.executable,TOOLS/"check_parity.py","--net",full/"best.znnue",
                     "--records",sample,"--zenith-nnue",NNUE,"--count",16])
        self.assertIn("mismatches 0",run.stdout)

    def test_stream_training_resume_export_and_validation_parity(self):
        full, split = self.base / "full", self.base / "split"
        common = [sys.executable, TOOLS / "train_nnue.py", "--binpack-config", self.recipe,
                  "--epoch-rows", 128, "--epochs", 2, "--batch-size", 16, "--block-rows", 32,
                  "--threads", 1, "--val-rows", 32, "--val-chunk", 16, "--warmup-steps", 0]
        command([*common, "--out", full])
        command([*common, "--out", split, "--stop-after-epochs", 1])
        command([*common, "--out", split, "--resume", split / "last.pt"])
        self.assertEqual((full / "last.znnue").read_bytes(), (split / "last.znnue").read_bytes())
        source = BinpackSource(self.recipe)
        sample = self.base / "validation-only.bin"
        source.validation(32).tofile(sample)
        run = command([sys.executable, TOOLS / "check_parity.py", "--net", split / "best.znnue",
                       "--records", sample, "--zenith-nnue", NNUE, "--count", 32])
        self.assertIn("mismatches 0", run.stdout)
        meta = json.loads((split / "provenance.json").read_text())
        self.assertEqual(meta["contract"]["input_format"], "binpack")
        self.assertEqual(meta["native_reader_stats"]["train"]["emitted"], 128)


if __name__ == "__main__":
    unittest.main()

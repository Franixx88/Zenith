"""V2 banking, warm starts, independent integer oracle and exact CPU resume."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import chess
import numpy as np
import torch

import train_nnue as T
import check_parity as P
from records import RECORD
from test_pipeline import ROOT, PREPARE, NNUE, TOOLS, fixture_rows, command


def positions():
    boards = []
    for white, black in [(chess.A1, chess.H8), (chess.H1, chess.A8),
                         (chess.A5, chess.H1), (chess.H5, chess.A1)]:
        board = chess.Board.empty()
        board.set_piece_at(white, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(black, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.C3, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.F6, chess.Piece(chess.PAWN, chess.BLACK))
        assert board.is_valid()
        boards.extend([board, board.mirror()])
    return boards


def records_for(boards):
    rows = np.zeros(len(boards), dtype=RECORD)
    for row, board in zip(rows, boards):
        bits = bytearray()
        for color in [board.turn, not board.turn]:
            for piece in range(1, 7):
                mask = sum(1 << (sq ^ (56 if not board.turn else 0)) for sq in board.pieces(piece, color))
                bits.extend(mask.to_bytes(8, "little"))
        row["bits"] = np.frombuffer(bits, dtype=np.uint8)
    return rows


class V2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.rows = records_for(positions())
        self.x, self.xo = T.unpack(torch.from_numpy(self.rows["bits"].copy()), torch.device("cpu"),
                                   torch.as_tensor(T.opponent_inverse_permutation()))

    def test_v1_transfer_is_integer_exact_and_gradients_only_touch_selected_bank(self):
        old, new = T.Net(), T.Net(4)
        T.initialize_from_checkpoint(new, {"model": old.state_dict()})
        torch.testing.assert_close(old(self.x, self.xo), new(self.x, self.xo), atol=1e-6, rtol=1e-6)
        for a, b in zip(T.quantized_cp(T.quantize(old), self.x, self.xo),
                        T.quantized_cp(T.quantize(new), self.x, self.xo)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            T.initialize_from_checkpoint(T.Net(), {"model": new.state_dict()})
        self.assertEqual(set(T.king_bucket_indices(self.x).tolist()), set(range(4)))
        selected = self.x[T.king_bucket_indices(self.x) == 0]
        T.banked_linear(selected, new.ft.weight, new.ft.bias, 4).sum().backward()
        self.assertGreater(new.ft.weight.grad[:, :768].abs().sum().item(), 0)
        self.assertEqual(new.ft.weight.grad[:, 768:].abs().sum().item(), 0)

    def test_four_distinct_banks_match_numpy_rust_scalar_and_avx2(self):
        model = T.Net(4)
        with torch.no_grad():
            for bucket in range(4):
                model.ft.weight[:, bucket * 768:(bucket + 1) * 768].add_(bucket / 128)
        q = T.quantize(model)
        raw, cp = T.quantized_cp(q, self.x, self.xo)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as tmp:
            base = Path(tmp)
            net = base / "four bank model.znnue"
            net.write_bytes(T.to_bytes(q))
            self.assertEqual(net.stat().st_size, T.V2_FILE_BYTES)
            spec = P.read_net(net)
            for i, row in enumerate(self.rows):
                self.assertEqual((int(raw[i]), int(cp[i])), P.spec_forward(spec, P.record_to_fen(row)))
            records = base / "parity.bin"
            self.rows.tofile(records)
            run = command([sys.executable, TOOLS / "check_parity.py", "--net", net,
                           "--records", records, "--zenith-nnue", NNUE, "--count", len(self.rows)])
            self.assertIn("mismatches 0", run.stdout)
            data = bytearray(net.read_bytes())
            data[36:40] = struct.pack("<I", 3)
            data[48:56] = bytes(8)
            struct.pack_into("<Q", data, 48, T.fnv1a64(data))
            net.write_bytes(data)
            with self.assertRaises(ValueError):
                P.read_net(net)

    def test_v2_training_and_resume_preserve_schedule_weights_and_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as tmp:
            base = Path(tmp)
            data = base / "data"
            rows = fixture_rows(256)
            command([PREPARE, "--out", data, "--shards", 2, "--val-mod", 5, "--keep-tactical",
                     "--require-complete", "--source", "synthetic-v2-smoke-only"],
                    input="".join(rows) + f"#zenith-tsv-complete\t{len(rows)}\n")
            initial = base / "v1.pt"
            torch.save({"model": T.Net().state_dict()}, initial)
            common = [sys.executable, TOOLS / "train_nnue.py", "--data", data, "--epochs", 2,
                      "--king-buckets", 4, "--batch-size", 32, "--block-rows", 128,
                      "--val-rows", 64, "--val-chunk", 32, "--threads", 1, "--warmup-steps", 0]
            full, split = base / "full", base / "split"
            command([*common, "--out", full, "--init-model", initial])
            command([*common, "--out", split, "--init-model", initial, "--stop-after-epochs", 1])
            command([*common, "--out", split, "--resume", split / "last.pt"])
            self.assertEqual((full / "last.znnue").read_bytes(), (split / "last.znnue").read_bytes())
            a = torch.load(full / "last.pt", weights_only=True)
            b = torch.load(split / "last.pt", weights_only=True)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["contract"]["king_buckets"], 4)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertIn("v2", json.loads((split / "provenance.json").read_text())["architecture"])
            self.assertEqual((split / "last.znnue").stat().st_size, T.V2_FILE_BYTES)


if __name__ == "__main__":
    unittest.main()

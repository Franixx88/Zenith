#!/usr/bin/env python3
"""End-to-end export check: trainer arithmetic vs. the spec vs. Zenith's Rust code.

Takes a bounded sample from val.bin, rebuilds white-to-move and color/rank-mirrored
black-to-move FENs (the same relative network input), then compares:
  1. the trainer's quantized forward pass (train_nnue.quantized_cp) on the records,
  2. an independent numpy implementation written from docs/nnue-v1.md that reads
     only the exported .znnue file and the FEN (python-chess),
  3. `examples/nnue verify` with scalar and automatic SIMD selection.
Every raw sum and centipawn value must be identical.

    cargo build --release --example nnue
    python tools/nnue/check_parity.py --net run1/best.znnue --records ds/val.bin \
        --zenith-nnue Zenith/target/release/examples/nnue --count 500
"""
import argparse
import struct
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import torch

import train_nnue as T
import v3_model
from records import sample_records


def read_net(path):
    with Path(path).open("rb") as file:
        data = file.read(v3_model.FILE_BYTES + 1)
    if len(data) < 64 or data[:8] != b"ZNNUE\0\0\0":
        raise ValueError("not a Zenith NNUE net")
    fields = struct.unpack_from("<8I", data, 8)
    version = fields[0]
    if version == 3:
        return v3_model.read_net(path)
    if version not in (1, 2):
        raise ValueError("unsupported version")
    buckets = 1 if version == 1 else 4
    payload = T.PAYLOAD if version == 1 else T.V2_PAYLOAD
    if len(data) != 64 + payload:
        raise ValueError("invalid file size")
    if fields != (version, 768, 256, 1, 255, 64, 400, buckets):
        raise ValueError("unsupported architecture")
    if struct.unpack_from("<Q", data, 40)[0] != payload or struct.unpack_from("<Q", data, 56)[0] != 0:
        raise ValueError("invalid payload size/reserved header")
    if struct.unpack_from("<Q", data, 48)[0] != T.fnv1a64(data[:48] + bytes(8) + data[56:]):
        raise ValueError("checksum")
    body = np.frombuffer(data, dtype="<i2", offset=64, count=(payload - 4) // 2)
    bias, rest = body[:256].astype(np.int64), body[256:]
    features = buckets * 768
    weights = rest[:features * 256].reshape(features, 256).astype(np.int64)
    out_w = rest[features * 256:].astype(np.int64)
    out_b = struct.unpack_from("<i", data, len(data) - 4)[0]
    return bias, weights, out_w, out_b


def record_to_fen(record):
    """Own pieces become White, enemy pieces Black, White to move (relative frame)."""
    board = chess.Board.empty()
    types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for index in range(12):
        color = chess.WHITE if index < 6 else chess.BLACK
        bits = int.from_bytes(bytes(record["bits"][index * 8:index * 8 + 8]), "little")
        for square in chess.scan_forward(bits):
            board.set_piece_at(square, chess.Piece(types[index % 6], color))
    board.turn = chess.WHITE
    return board


def spec_forward(net, board):
    if isinstance(net, dict):
        return v3_model.spec_forward(net, board)
    bias, weights, out_w, out_b = net
    raw = out_b
    for half, perspective in enumerate((board.turn, not board.turn)):  # stm first
        bucket = 0
        if len(weights) == 4 * 768:
            king = board.king(perspective) ^ (56 if perspective == chess.BLACK else 0)
            bucket = int(chess.square_file(king) >= 4) + 2 * int(chess.square_rank(king) >= 4)
        total = bias.copy()
        for square, piece in board.piece_map().items():
            relative = square ^ (56 if perspective == chess.BLACK else 0)
            feature = ((0 if piece.color == perspective else 1) * 6
                       + piece.piece_type - 1) * 64 + relative
            total += weights[bucket * 768 + feature]
        raw += int(np.clip(total, 0, 255) @ out_w[half * 256:(half + 1) * 256])
    cp = int(np.trunc(raw * 400 / (255 * 64)))
    return raw, max(-28000, min(28000, cp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--zenith-nnue", required=True)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--checkpoint", type=Path, help="v3: also verify QAT checkpoint weights against export")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="QAT checkpoint verification device")
    args = ap.parse_args()

    if args.count <= 0:
        ap.error("count must be positive")
    records = sample_records(args.records, args.count, seed=3)
    net = read_net(args.net)
    boards = [board for r in records for board in (record_to_fen(r), record_to_fen(r).mirror())]
    fens = [b.fen() for b in boards]

    # 1. trainer arithmetic, built from the .znnue file the same way a user would reload it
    q = net if isinstance(net, dict) else {"ft_b": net[0], "ft_w": net[1], "out_w": net[2], "out_b": net[3]}
    inv = torch.as_tensor(T.opponent_inverse_permutation())
    x, xo = T.unpack(torch.from_numpy(np.ascontiguousarray(records["bits"])),
                     torch.device("cpu"), inv)
    raw_t, cp_t = T.quantized_cp(q, x, xo)
    if args.checkpoint:
        if q.get("version") != 3:
            ap.error("checkpoint verification currently requires v3")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("contract", {}).get("architecture") != "v3":
            raise ValueError("checkpoint is not v3")
        model = v3_model.Net().to(args.device)
        model.load_state_dict(checkpoint["model"])
        if T.to_bytes(T.quantize(model)) != Path(args.net).read_bytes():
            raise ValueError("checkpoint export differs from the selected network")
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            prediction = model.forward_cp(x.to(args.device), xo.to(args.device)).cpu().double()
        torch.testing.assert_close(prediction, cp_t.cpu(), atol=0, rtol=0)
        print(f"checkpoint QAT == integer export on {args.device}: {len(x)} records")
    raw_t, cp_t = raw_t.repeat_interleave(2), cp_t.repeat_interleave(2)

    # 2. independent spec implementation
    spec = [spec_forward(net, b) for b in boards]

    # 3. Rust
    backends = []
    for flags in (["--scalar"], []):
        run = subprocess.run([args.zenith_nnue, "verify", args.net, *flags], input="\n".join(fens) + "\n",
                             capture_output=True, text=True, check=True)
        rust = [tuple(map(int, line.rsplit("\t", 2)[1:])) for line in run.stdout.strip().splitlines()]
        if len(rust) != len(fens):
            raise ValueError(f"Rust returned wrong count: {run.stderr}")
        backends.append(rust)
        print(run.stderr.strip())

    bad = 0
    for i, fen in enumerate(fens):
        trainer = (int(raw_t[i].item()), int(cp_t[i].item()))
        if not (trainer == spec[i] == backends[0][i] == backends[1][i]):
            bad += 1
            if bad <= 5:
                print("MISMATCH", fen, "trainer", trainer, "spec", spec[i], "rust", [b[i] for b in backends])
    print(f"{len(fens)} positions: trainer == spec == rust for {len(fens) - bad}, mismatches {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

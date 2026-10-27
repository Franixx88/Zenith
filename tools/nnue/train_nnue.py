#!/usr/bin/env python3
"""Zenith NNUE v1/v2 trainer: 1/4 king banks -> 256 x 2 -> 1 CReLU.

Contract (docs/nnue-v1.md):
  A[p]   = clamp(bias + sum(W[feature]), 0, 1)           p in {side to move, opponent}
  output = out_bias + dot(concat(A[stm], A[opp]), out_w)  (units of 400 cp)
  engine cp = trunc(raw * 400 / (255 * 64)) on the quantized integers

Calibration is built in: the sigmoid scale S used for the loss is separate from the
400 cp unit of the file format, and the logit is `output * 400 / S`. So whatever S
you pick, the exported network speaks the same centipawns as the labels.

Data: the directory written by examples/prepare_nnue.rs (train-*.bin, val.bin).

    python tools/nnue/train_nnue.py --data data/nnue --out runs/v1 --epochs 10

Epoch checkpoints include RNG state/history. Resume requires the same data and
schedule; --init-model starts a new optimizer/schedule for fine tuning on new data.
"""
import argparse
import hashlib
import json
import math
import queue
import struct
import threading
import time
import os
import platform
import io
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from torch import nn
import v3_model
from awake_time import awake_seconds
from records import RECORD, check_records, record_count, sample_records, shuffled_batches, dataset_identity

INPUTS, HIDDEN = 768, 256
QA, QB, CP_SCALE, MAX_SCORE = 255, 64, 400, 28_000
MAX_FT_W, MAX_FT_B, MAX_OUT_W, MAX_OUT_B = 512, 4096, 8192, 16_777_216
# Float clamps that keep every parameter exportable after rounding.
LIMIT_FT_W, LIMIT_FT_B, LIMIT_OUT_W = 2.0, 16.0, 127.0
assert RECORD.itemsize == 100
PAYLOAD = HIDDEN * 2 + INPUTS * HIDDEN * 2 + 2 * HIDDEN * 2 + 4
FILE_BYTES = 64 + PAYLOAD
KING_BUCKETS = 4
V2_PAYLOAD = PAYLOAD + (KING_BUCKETS - 1) * INPUTS * HIDDEN * 2
V2_FILE_BYTES = 64 + V2_PAYLOAD


def opponent_inverse_permutation():
    """x_opp = x_stm[:, inv]: same bits seen from the other side of the board."""
    inv = np.empty(INPUTS, dtype=np.int64)
    for f in range(INPUTS):
        color, piece = divmod(f // 64, 6)
        inv[((1 - color) * 6 + piece) * 64 + ((f % 64) ^ 56)] = f
    return inv


def king_bucket_indices(x):
    """Relative own-king plane, four quadrants, no horizontal mirroring."""
    square = x[:, 5 * 64:6 * 64].argmax(dim=1)
    return ((square % 8) >= 4).long() + 2 * ((square // 8) >= 4).long()


def banked_linear(x, weight, bias, buckets):
    """One GEMM per populated bank; never materialize per-position tensors.

    weight is [hidden, buckets*768]. Grouping avoids computing all four
    banks for every position. Activations/gradients remain bounded by batch.
    """
    if buckets == 1:
        return nn.functional.linear(x, weight, bias)
    indices = king_bucket_indices(x)
    out = x.new_zeros((len(x), HIDDEN))
    for bucket in range(buckets):
        positions = (indices == bucket).nonzero(as_tuple=True)[0]
        if positions.numel():
            values = nn.functional.linear(x.index_select(0, positions),
                weight[:, bucket * INPUTS:(bucket + 1) * INPUTS], bias)
            out.index_copy_(0, positions, values)
    return out


class Net(nn.Module):
    def __init__(self, king_buckets=1):
        super().__init__()
        if king_buckets not in (1, KING_BUCKETS):
            raise ValueError("only one or four king banks are supported")
        self.king_buckets = king_buckets
        self.ft = nn.Linear(INPUTS * king_buckets, HIDDEN)
        if king_buckets != 1:
            # Only 768 features feed any position, regardless of bank count.
            nn.init.uniform_(self.ft.weight, -1 / math.sqrt(INPUTS), 1 / math.sqrt(INPUTS))
            nn.init.uniform_(self.ft.bias, -1 / math.sqrt(INPUTS), 1 / math.sqrt(INPUTS))
        self.out = nn.Linear(2 * HIDDEN, 1)

    def forward(self, x_stm, x_opp):
        n = x_stm.shape[0]
        h = torch.clamp(banked_linear(torch.cat([x_stm, x_opp], 0),
                       self.ft.weight, self.ft.bias, self.king_buckets), 0.0, 1.0)
        return self.out(torch.cat([h[:n], h[n:]], 1)).squeeze(1)  # units of 400 cp

    @torch.no_grad()
    def constrain(self):
        self.ft.weight.clamp_(-LIMIT_FT_W, LIMIT_FT_W)
        self.ft.bias.clamp_(-LIMIT_FT_B, LIMIT_FT_B)
        self.out.weight.clamp_(-LIMIT_OUT_W, LIMIT_OUT_W)
        self.out.bias.clamp_(-1000.0, 1000.0)


def initialize_from_checkpoint(model, initial):
    """Copy v1 banks exactly into v2; optimizer/schedule intentionally reset."""
    if getattr(model, "architecture", None) == "v3":
        if initial.get("contract", {}).get("architecture") != "v3":
            raise ValueError("v3 requires v3 weights; legacy mirroring/pairwise cannot be warm-started exactly")
        model.load_state_dict(initial["model"])
        return
    state = dict(initial["model"])
    shape = tuple(state["ft.weight"].shape)
    if shape == (HIDDEN, INPUTS) and model.king_buckets == KING_BUCKETS:
        state["ft.weight"] = state["ft.weight"].repeat(1, KING_BUCKETS)
    elif shape != (HIDDEN, INPUTS * model.king_buckets):
        raise ValueError("incompatible init-model architecture (v2 cannot be collapsed into v1)")
    model.load_state_dict(state)


def unpack(bits_u8, device, inv):
    """uint8[B, 96] -> float32 [B, 768] for both perspectives (bit order = feature order)."""
    shifts = torch.arange(8, dtype=torch.int16, device=device)
    bits = bits_u8.to(device).to(torch.int16)
    x = ((bits.unsqueeze(-1) >> shifts) & 1).reshape(bits.shape[0], INPUTS).to(torch.float32)
    return x, x.index_select(1, inv)


# ---------------------------------------------------------------- quantization
def _round_away(values, scale, bound, name):
    scaled = np.asarray(values, dtype=np.float64) * scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)  # ties away from zero
    if not np.all(np.isfinite(rounded)) or np.abs(rounded).max() > bound:
        raise ValueError(f"{name}: non-finite or out of range for NNUE v1")
    return rounded.astype(np.int64)


def quantize(model):
    if getattr(model, "architecture", None) == "v3":
        return v3_model.quantize(model)
    ft_w = model.ft.weight.detach().cpu().numpy().astype(np.float32).T  # feature-major
    return {
        "ft_b": _round_away(model.ft.bias.detach().cpu().numpy(), QA, MAX_FT_B, "ft bias"),
        "ft_w": _round_away(np.ascontiguousarray(ft_w), QA, MAX_FT_W, "ft weights"),
        "out_w": _round_away(model.out.weight.detach().cpu().numpy().reshape(-1), QB, MAX_OUT_W,
                             "output weights"),
        "out_b": int(_round_away(model.out.bias.detach().cpu().numpy().reshape(-1)[0],
                                 QA * QB, MAX_OUT_B, "output bias")),
    }


def quantized_cp(q, x_stm, x_opp):
    """Exact integer arithmetic of the Rust engine (float64 is exact at these magnitudes)."""
    if q.get("version") == 3:
        return v3_model.quantized_cp(q, x_stm, x_opp)
    dev = x_stm.device
    ft_w = torch.as_tensor(q["ft_w"], dtype=torch.float64, device=dev)
    buckets = ft_w.shape[0] // INPUTS
    if buckets not in (1, KING_BUCKETS) or ft_w.shape != (buckets * INPUTS, HIDDEN):
        raise ValueError("invalid quantized feature tensor")
    ft_b = torch.as_tensor(q["ft_b"], dtype=torch.float64, device=dev)
    out_w = torch.as_tensor(q["out_w"], dtype=torch.float64, device=dev)
    raw = torch.full((x_stm.shape[0],), float(q["out_b"]), dtype=torch.float64, device=dev)
    for half, x in enumerate((x_stm, x_opp)):
        a = torch.clamp(banked_linear(x.double(), ft_w.T, ft_b, buckets), 0, QA)
        raw += a @ out_w[half * HIDDEN:(half + 1) * HIDDEN]
    cp = torch.trunc(raw * CP_SCALE / (QA * QB))
    return raw, torch.clamp(cp, -MAX_SCORE, MAX_SCORE)


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def to_bytes(q):
    if q.get("version") == 3:
        return v3_model.to_bytes(q)
    buckets = q["ft_w"].shape[0] // INPUTS
    if buckets not in (1, KING_BUCKETS) or q["ft_w"].shape != (buckets * INPUTS, HIDDEN):
        raise ValueError("unsupported NNUE feature dimensions")
    version = 1 if buckets == 1 else 2
    payload = PAYLOAD if buckets == 1 else V2_PAYLOAD
    header = bytearray(64)
    header[0:8] = b"ZNNUE\0\0\0"
    struct.pack_into("<8I", header, 8, version, INPUTS, HIDDEN, 1, QA, QB, CP_SCALE, buckets)
    struct.pack_into("<Q", header, 40, payload)
    data = bytearray(header)
    data += q["ft_b"].astype("<i2").tobytes()
    data += np.ascontiguousarray(q["ft_w"]).astype("<i2").tobytes()
    data += q["out_w"].astype("<i2").tobytes()
    data += struct.pack("<i", q["out_b"])
    assert len(data) == 64 + payload
    struct.pack_into("<Q", data, 48, fnv1a64(bytes(data)))  # field is zero while hashing
    return bytes(data)


# ----------------------------------------------------------------------- data
def load_val(path, rows, seed=1):
    return sample_records(path, rows, seed)


def shard_batches(files, batch, rng, block_rows=65536):
    for sub in shuffled_batches(files, batch, block_rows, rng):
        yield torch.from_numpy(np.ascontiguousarray(sub["bits"])), \
            torch.from_numpy(sub["score"].astype(np.float32))


def prefetch(generator, depth=6):
    q, done, errors = queue.Queue(depth), object(), []
    cancel = threading.Event()

    def put(item):
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def work():
        try:
            for item in generator:
                if not put(item):
                    break
        except BaseException as error:  # surfaced in the consumer thread
            errors.append(error)
        finally:
            generator.close()
            put(done)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        while (item := q.get()) is not done:
            yield item
        if errors:
            raise errors[0]
    finally:
        cancel.set()
        if hasattr(generator, "cancel"):
            generator.cancel()
        worker.join()


# ----------------------------------------------------------------- evaluation
def prediction(model, x, xo, scale):
    # Preserve the exact v1/v2 operation order for existing resume contracts.
    logits = (model.forward_cp(x, xo) / scale if getattr(model, "architecture", None) == "v3"
              else model(x, xo) * (CP_SCALE / scale))
    return torch.sigmoid(logits)


@torch.no_grad()
def validate(model, val, scale, device, inv, chunk=4096):
    if len(val) == 0:
        raise ValueError("validation must contain records")
    model.eval()
    qdev = device if device.type == "cuda" else torch.device("cpu")
    q = quantize(model)
    sums = dict(float=0.0, quant=0.0, base=0.0, abs_cp=0.0)
    for i in range(0, len(val), chunk):
        part = val[i:i + chunk]
        x, xo = unpack(torch.from_numpy(np.ascontiguousarray(part["bits"])), device, inv)
        score = torch.from_numpy(part["score"].astype(np.float32)).to(device)
        target = torch.sigmoid(score / scale)
        pred = prediction(model, x, xo, scale)
        _, cp = quantized_cp(q, x.to(qdev), xo.to(qdev))
        cp = cp.to(device).float()
        sums["float"] += ((pred - target) ** 2).sum().item()
        sums["quant"] += ((torch.sigmoid(cp / scale) - target) ** 2).sum().item()
        sums["base"] += ((0.5 - target) ** 2).sum().item()
        sums["abs_cp"] += (cp - score).abs().clamp(max=1000).sum().item()
    n = len(val)
    model.train()
    return {"val_float": sums["float"] / n, "val_quant": sums["quant"] / n,
            "val_baseline": sums["base"] / n, "mean_abs_cp_error_capped": sums["abs_cp"] / n}


def atomic_bytes(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path)
    inputs.add_argument("--binpack-config", type=Path)
    ap.add_argument("--binpack-reader", type=Path)
    ap.add_argument("--epoch-rows", type=int, default=0, help="binpack: fixed accepted training positions per epoch")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--sigmoid-scale", type=float, default=400.0,
                    help="S in sigmoid(cp/S) for loss; the file's cp unit stays 400")
    ap.add_argument("--val-rows", type=int, default=500_000)
    ap.add_argument("--train-shards", type=int, default=0, help="use only the first N shards (0=all); shards are uniform random subsets")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--init-model", type=Path, help="warm start weights on new data; reset optimizer/schedule")
    ap.add_argument("--king-buckets", type=int, choices=[1, KING_BUCKETS], default=1,
                    help="1=v1, 4=v2 relative king quadrants")
    ap.add_argument("--architecture", choices=["legacy", "v3"], default="legacy",
                    help="v3: mirrored 16 banks, 512 FT, pairwise, 8 x 512-16-32-1 heads, QAT")
    ap.add_argument("--block-rows", type=int, default=65536)
    ap.add_argument("--val-chunk", type=int, default=4096)
    ap.add_argument("--stop-after-epochs", type=int, help="stop at this epoch boundary; keep the full --epochs schedule")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--max-train-seconds", type=float,
                    help="per invocation time budget; finish and checkpoint the current epoch before stopping")
    args = ap.parse_args()

    for name in ["epochs", "batch_size", "val_rows", "val_chunk", "threads", "log_every"]:
        if getattr(args, name) <= 0:
            ap.error(f"{name.replace('_', '-')} must be positive")
    if args.block_rows < args.batch_size or args.train_shards < 0 or args.warmup_steps < 0:
        ap.error("block-rows must be >= batch-size; train-shards/warmup-steps must be nonnegative")
    if args.stop_after_epochs is not None and not 1 <= args.stop_after_epochs <= args.epochs:
        ap.error("stop-after-epochs must be within 1..epochs")
    if args.resume and args.init_model:
        ap.error("resume and init-model are mutually exclusive")
    if args.architecture == "v3" and args.king_buckets != 1:
        ap.error("v3 fixes its own 16 mirrored banks; omit --king-buckets")
    if args.binpack_config and not 0 < args.epoch_rows < 2**64:
        ap.error("binpack input requires positive --epoch-rows")
    if args.data and (args.epoch_rows or args.binpack_reader):
        ap.error("epoch-rows/binpack-reader apply only to binpack input")
    if any(not math.isfinite(v) or v <= 0 for v in [args.lr, args.lr_final, args.sigmoid_scale]) or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        ap.error("learning rates/scale must be finite and positive; weight-decay nonnegative")

    if args.max_train_seconds is not None and (not math.isfinite(args.max_train_seconds) or args.max_train_seconds <= 0):
        ap.error("max-train-seconds must be finite and positive")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.architecture == "v3":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    native = None
    if args.binpack_config:
        if args.train_shards:
            ap.error("train-shards does not apply to binpack input")
        from binpack_data import BinpackSource
        native = BinpackSource(args.binpack_config, args.binpack_reader)
        rows = [args.epoch_rows]
        manifest, data_fingerprint = native.manifest, native.fingerprint
        val = native.validation(args.val_rows)
    else:
        files = sorted(args.data.glob("train-*.bin"))
        if args.train_shards:
            files = files[:args.train_shards]
        rows = [record_count(f, allow_empty=True) for f in files]
        if not files or sum(rows) == 0:
            raise ValueError("training data must contain records")
        manifest, data_fingerprint = dataset_identity(args.data, files)
        val = load_val(args.data / "val.bin", args.val_rows, args.seed)
    steps_per_epoch = sum((r // args.block_rows) * math.ceil(args.block_rows / args.batch_size)
                          + math.ceil((r % args.block_rows) / args.batch_size) for r in rows)
    total_steps = steps_per_epoch * args.epochs
    contract_names = ["epochs", "batch_size", "block_rows", "val_rows", "val_chunk", "train_shards",
                      "seed", "threads", "lr", "lr_final", "warmup_steps", "sigmoid_scale", "weight_decay"]
    contract = {name: getattr(args, name) for name in contract_names}
    contract.update(data_fingerprint=data_fingerprint, device=str(device),
                    torch=str(torch.__version__), numpy=np.__version__)
    if native is not None:
        contract.update(input_format="binpack", epoch_rows=args.epoch_rows)
    if args.king_buckets != 1:
        contract["king_buckets"] = args.king_buckets
    if args.architecture == "v3":
        contract.update(architecture="v3", king_buckets=16, factorizer=True, quantization="integer-ste-v1")
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or args.resume.resolve().parent != args.out.resolve():
            raise ValueError("output run exists; use a fresh directory or resume its checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train_rows={sum(rows):,} val_rows={len(val):,} "
          f"steps/epoch={steps_per_epoch:,}", flush=True)

    inv = torch.as_tensor(opponent_inverse_permutation(), device=device)
    model = (v3_model.Net() if args.architecture == "v3" else Net(args.king_buckets)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch0, step, best = 0, 0, float("inf")
    history, best_blob, best_epoch = [], None, None
    initialization = None
    if args.init_model:
        initial = torch.load(args.init_model, map_location="cpu", weights_only=True)
        initialize_from_checkpoint(model, initial)
        initialization = {"source_sha256": hashlib.sha256(args.init_model.read_bytes()).hexdigest(),
                          "validation": validate(model, val, args.sigmoid_scale, device, inv, args.val_chunk)}
        print("INIT " + json.dumps(initialization), flush=True)
        atomic_bytes(args.out / "initial.znnue", to_bytes(quantize(model)))
        atomic_json(args.out / "initial-validation.json", initialization)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("checkpoint_version") != 1 or state.get("contract") != contract:
            raise ValueError("resume contract changed (data/settings/runtime); use init-model for a new training run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch0, step, best = state["epoch"], state["step"], state["best"]
        history, best_blob, best_epoch = state["history"], state["best_blob"], state["best_epoch"]
        initialization = state.get("initialization")
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if epoch0 >= args.epochs:
            raise ValueError("checkpoint has already completed the requested schedule")
        if args.stop_after_epochs is not None and args.stop_after_epochs <= epoch0:
            raise ValueError("stop-after-epochs must be after the resumed epoch")

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        t = (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * min(t, 1)))

    training_started = awake_seconds()
    training_wall_started = time.monotonic()
    for epoch in range(epoch0, args.epochs):
        start, seen, running = awake_seconds(), 0, 0.0
        logged = 0
        iterator = native.epoch(args.epoch_rows, args.batch_size, args.block_rows, rng) if native else shard_batches(files, args.batch_size, rng, args.block_rows)
        with closing(prefetch(iterator)) as batches:
            for bits, score in batches:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                x, xo = unpack(bits, device, inv)
                target = torch.sigmoid(score.to(device) / args.sigmoid_scale)
                pred = prediction(model, x, xo, args.sigmoid_scale)
                loss = ((pred - target) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.constrain()
                step += 1
                seen += len(score)
                running += loss.item()
                logged += 1
                if step % args.log_every == 0:
                    print(f"epoch {epoch + 1} step {step:,}/{total_steps:,} loss {running / logged:.6f} "
                          f"lr {lr_at(step):.2e} {seen / max(awake_seconds() - start, 1e-9):,.0f} pos/s", flush=True)
                    running = 0.0
                    logged = 0
        if seen != sum(rows):
            raise ValueError("epoch did not visit every training record")
        metrics = validate(model, val, args.sigmoid_scale, device, inv, args.val_chunk)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite validation metrics")
        metrics.update(epoch=epoch + 1, step=step,
                       explained=1 - metrics["val_quant"] / max(metrics["val_baseline"], 1e-12))
        history.append(metrics)
        print("VAL " + json.dumps(metrics), flush=True)

        q = quantize(model)
        blob = to_bytes(q)
        improved = metrics["val_quant"] < best
        if improved:
            best_blob, best_epoch = blob, epoch + 1
        best = min(best, metrics["val_quant"])
        state = {"checkpoint_version": 1, "contract": contract,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "step": step, "best": best,
                 "best_blob": best_blob, "best_epoch": best_epoch, "history": history,
                 "initialization": initialization,
                 "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        packed = io.BytesIO()
        torch.save(state, packed)
        checkpoint = packed.getvalue()
        epoch_tag = f"epoch-{epoch + 1:04}"
        atomic_bytes(args.out / f"{epoch_tag}.pt", checkpoint)
        atomic_bytes(args.out / f"{epoch_tag}.znnue", blob)
        provenance = {
            "architecture": ("zenith-nnue-v3 mirrored 16x768->512 pairwise, 8x512->16->32->1 QAT" if args.architecture == "v3"
                             else "zenith-nnue-v1 768->256x2->1 CReLU" if args.king_buckets == 1
                             else "zenith-nnue-v2 4 king quadrants x 768->256x2->1 CReLU"),
            "network_sha256_last": hashlib.sha256(blob).hexdigest(),
            "network_sha256_best": hashlib.sha256(best_blob).hexdigest(),
            "best_epoch": best_epoch, "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            "contract": contract, "runtime_python": platform.python_version(),
            "epoch_seconds": awake_seconds() - start,
            "training_seconds": awake_seconds() - training_started,
            "training_wall_seconds": time.monotonic() - training_wall_started,
            "time_budget_clock": "system-awake; excludes suspend on Windows",
            "args": {k: str(v) for k, v in vars(args).items()},
            "data_manifest": manifest, "history": history,
            "native_reader_stats": native.last_stats if native else None,
            "initialization": initialization,
            "note": "Keep source and license from data_manifest with released nets; do not infer licensing from the file format.",
        }
        atomic_json(args.out / f"{epoch_tag}.json", provenance)
        # latest.json is the commit marker for the complete epoch triplet.
        atomic_json(args.out / "latest.json", {"epoch": epoch + 1, "checkpoint": f"{epoch_tag}.pt",
                    "network": f"{epoch_tag}.znnue", "provenance": f"{epoch_tag}.json",
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "network_sha256": provenance["network_sha256_last"]})
        atomic_bytes(args.out / "last.pt", checkpoint)
        atomic_bytes(args.out / "last.znnue", blob)
        atomic_bytes(args.out / "best.znnue", best_blob)
        if improved:
            atomic_bytes(args.out / "best.pt", checkpoint)
        atomic_json(args.out / "provenance.json", provenance)
        if args.stop_after_epochs is not None and epoch + 1 >= args.stop_after_epochs:
            break
        if args.max_train_seconds is not None and awake_seconds() - training_started >= args.max_train_seconds:
            print("time budget reached; completed epoch checkpoint saved", flush=True)
            break
    print(f"done; best quantized val loss {best:.6f}; nets in {args.out}")


if __name__ == "__main__":
    main()

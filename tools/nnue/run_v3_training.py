#!/usr/bin/env python3
"""Finite v3 experiment: verified existing corpus, scratch CUDA training, parity and matches.

No downloads or automatic promotion. Run in the CUDA venv with a fresh output directory.
The stable v1 opponent has a different training history: this is a deployment check,
not an equal-exposure experiment proving that architecture alone is better.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_first_training import ROOT, atomic_json
from run_king_bucket_training import corpus_snapshot, digest
from awake_time import wait_process


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--epoch-rows", type=int, default=8_388_608)
    ap.add_argument("--seconds", type=float, default=5400)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--recipe", type=Path, default=ROOT / "tools/nnue/corpora/first-50gb-training.json")
    ap.add_argument("--integrity", type=Path, default=ROOT / "runs/first-50gb-control-20261026/post-outage-integrity.json")
    ap.add_argument("--baseline", type=Path, default=ROOT / "networks/zenith-v1-1c7c79756c6e.znnue")
    ap.add_argument("--resume-run", type=Path, help="fork a completed checkpoint into the fresh --out directory")
    args = ap.parse_args()
    if min(args.epochs, args.epoch_rows, args.pairs) <= 0 or not math.isfinite(args.seconds) or not 0 < args.seconds <= 7200:
        ap.error("positive counts and at most two hours of training required")
    engine = ROOT / "target/release/zenith.exe"
    harness = ROOT / "target/release/examples/nnue.exe"
    reader = ROOT / "target/binpack/release/zenith-binpack.exe"
    for path in [engine, harness, reader, args.baseline]:
        if not path.is_file():
            ap.error(f"missing required file: {path}")
    recipe, snapshot = corpus_snapshot(args.recipe, args.integrity)
    recovery = None
    if args.resume_run:
        previous = args.resume_run.resolve()
        previous_net = previous / "network"
        latest = json.loads((previous_net / "latest.json").read_text())
        provenance = json.loads((previous_net / "provenance.json").read_text())
        selected = json.loads((previous_net / f"epoch-{provenance['best_epoch']:04}.json").read_text())
        for name, expected in [(latest["checkpoint"], latest["checkpoint_sha256"]),
                               ("best.pt", selected["checkpoint_sha256"]),
                               ("best.znnue", provenance["network_sha256_best"])]:
            if digest(previous_net / name) != expected:
                raise ValueError(f"resume source hash mismatch: {name}")
        contract = provenance["contract"]
        if (contract.get("architecture") != "v3" or contract["epochs"] != args.epochs
                or contract["epoch_rows"] != args.epoch_rows or latest["epoch"] >= args.epochs):
            ap.error("resume requires an unfinished v3 schedule with identical epochs/epoch-rows")
        for name in ["v3_model.py", "records.py", "binpack_data.py"]:
            if digest(previous / "scripts" / name) != digest(ROOT / "tools/nnue" / name):
                raise ValueError(f"resume arithmetic/data source changed: {name}")
        recovery = dict(source=str(previous), checkpoint=latest["checkpoint"],
                        checkpoint_sha256=latest["checkpoint_sha256"], epoch=latest["epoch"])
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    recipe_path = args.out / "recipe.json"
    atomic_json(recipe_path, recipe)
    # Freeze native binaries and the opponent before a long unattended run.
    for source, name in [(engine, "zenith.exe"), (harness, "nnue.exe"), (args.baseline, "baseline.znnue")]:
        (args.out / name).write_bytes(source.read_bytes())
    engine, harness, baseline = [args.out / name for name in ["zenith.exe", "nnue.exe", "baseline.znnue"]]
    source_files = sorted((ROOT / "src").rglob("*.rs")) + [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    source_files += sorted((ROOT / "tools/nnue").glob("*.py"))
    plan = dict(args={k: str(v) for k,v in vars(args).items()}, corpus_snapshot=snapshot,
        architecture="v3", initialization=recovery or "scratch; separate pilot excluded",
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        binaries_sha256={p.name: digest(p) for p in [engine,harness,reader,baseline]})
    atomic_json(args.out / "plan.json", plan)
    scripts = args.out / "scripts"
    scripts.mkdir()
    for name in ["train_nnue.py", "v3_model.py", "records.py", "binpack_data.py", "check_parity.py", "awake_time.py"]:
        (scripts / name).write_bytes((ROOT / "tools/nnue" / name).read_bytes())
    state = dict(pid=os.getpid(), started_at=time.time(), stage="starting")

    def update(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        atomic_json(args.out / "status.json", state)
        print(json.dumps(state), flush=True)

    def run(stage, command, timeout):
        command = [str(item) for item in command]
        update(stage, command=command, log=str(args.out / f"{stage}.log"))
        with (args.out / f"{stage}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(stage, child_pid=child.pid)
            try:
                code = wait_process(child, timeout)
            except BaseException:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                else:
                    child.kill()
                child.wait()
                raise
        if code:
            raise RuntimeError(f"{stage} failed ({code}); inspect {stage}.log")

    try:
        run("cuda", [sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
             "print(torch.__version__,torch.cuda.get_device_name()); "
             "x=torch.ones(32,32,device='cuda'); assert (x@x).sum().item()==32768"], 120)
        netdir = args.out / "network"
        resume_options = []
        if recovery:
            netdir.mkdir()
            metadata = ["latest.json", "provenance.json", f"epoch-{provenance['best_epoch']:04}.json"]
            for name in [recovery["checkpoint"], "best.pt", "best.znnue", *metadata]:
                (netdir / name).write_bytes((previous_net / name).read_bytes())
            resume_options = ["--resume", netdir / recovery["checkpoint"]]
        run("training", [sys.executable, "-u", scripts / "train_nnue.py", "--architecture", "v3",
            "--binpack-config", recipe_path, "--binpack-reader", reader, "--out", netdir,
            "--epochs", args.epochs, "--epoch-rows", args.epoch_rows, "--batch-size", 8192,
            "--block-rows", 65536, "--val-rows", 131072, "--val-chunk", 4096, "--threads", 2,
            "--device", "cuda", "--seed", 3, "--sigmoid-scale", 200, "--lr", 1e-3,
            "--lr-final", 1e-5, "--warmup-steps", 128, "--log-every", 128,
            "--max-train-seconds", args.seconds, *resume_options], args.seconds + 1200)
        provenance = json.loads((netdir / "provenance.json").read_text())
        if digest(netdir / "best.znnue") != provenance["network_sha256_best"]:
            raise ValueError("selected network differs from provenance")
        for item in snapshot:
            stat = Path(item["path"]).stat()
            if (stat.st_size,stat.st_mtime_ns) != (item["bytes"],item["mtime_ns"]):
                raise ValueError("corpus mutated during training")
        parity_records = args.out / "parity.bin"
        run("parity-data", [sys.executable, "-c", "import sys;sys.path.insert(0,sys.argv[4]); "
            "from binpack_data import BinpackSource; "
            "BinpackSource(sys.argv[1],sys.argv[2]).validation(2048).tofile(sys.argv[3])",
            recipe_path, reader, parity_records, scripts], 300)
        for device in ["cpu", "cuda"]:
            run(f"parity-{device}", [sys.executable, scripts / "check_parity.py", "--net", netdir / "best.znnue",
                "--checkpoint", netdir / "best.pt", "--device", device, "--records", parity_records,
                "--zenith-nnue", harness, "--count", 512], 300)
        match = args.out / "stable-v1-match"
        run("match", [sys.executable, ROOT / "tools/engine_test.py", engine, engine, match,
            "--pairs", args.pairs, "--split", "holdout", "--threads", 1, "--concurrency", 2,
            "--hash", 32, "--tc", "0:05+0.05", "--timeout", 2400,
            "--candidate-option", f"EvalFile={netdir / 'best.znnue'}", "--candidate-option", "Use NNUE=true",
            "--baseline-option", f"EvalFile={baseline}", "--baseline-option", "Use NNUE=true",
            "--candidate-option", "Context Histories=false", "--baseline-option", "Context Histories=false"], 2460)
        result = json.loads((match / "summary.json").read_text())
        update("complete", epochs=len(provenance["history"]), best_epoch=provenance["best_epoch"],
            presentations=len(provenance["history"]) * args.epoch_rows, val_quant=min(p["val_quant"] for p in provenance["history"]),
            match_wdl=result["pgn"]["wdl"], note="Experimental network; no automatic promotion or Elo claim.")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

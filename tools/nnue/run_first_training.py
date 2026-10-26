#!/usr/bin/env python3
"""Finite local job: verified corpus -> CUDA training -> parity -> UCI/match checks.

Run with the CUDA virtualenv. A new job directory is required. Training finishes
at an epoch boundary; its time budget excludes download and data preparation.
It never installs a candidate into the user's analysis board.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "tools/nnue/corpora/first-50gb.json")
    ap.add_argument("--recipe", type=Path, default=ROOT / "tools/nnue/corpora/first-50gb-training.json")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/nnue/sf-first-50gb")
    ap.add_argument("--engine", type=Path, default=ROOT / "target/release/zenith.exe")
    ap.add_argument("--reader", type=Path, default=ROOT / "target/binpack/release/zenith-binpack.exe")
    ap.add_argument("--nnue-harness", type=Path, default=ROOT / "target/release/examples/nnue.exe")
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--epoch-rows", type=int, default=8_388_608)
    ap.add_argument("--max-train-seconds", type=float, default=5400)
    ap.add_argument("--val-rows", type=int, default=131072)
    ap.add_argument("--match-pairs", type=int, default=16)
    args = ap.parse_args()
    if not 0 < args.max_train_seconds <= 7200 or not math.isfinite(args.max_train_seconds):
        ap.error("training budget must be positive and at most two hours")
    if min(args.epochs, args.epoch_rows, args.val_rows, args.match_pairs) <= 0:
        ap.error("counts must be positive")
    for name in ["manifest", "recipe", "engine", "reader", "nnue_harness"]:
        if not getattr(args, name).is_file():
            ap.error(f"missing {name}: {getattr(args, name)}")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = dict(pid=os.getpid(), started_at=started, stage="starting", job=str(args.out))

    def update(stage, **extra):
        status.update(stage=stage, updated_at=time.time(), **extra)
        atomic_json(args.out / "status.json", status)
        print(json.dumps(status), flush=True)

    def run(stage, command, timeout=None):
        command = [str(item) for item in command]
        update(stage, command=command, log=str(args.out / f"{stage}.log"))
        with (args.out / f"{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(stage, child_pid=process.pid)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                process.kill()
                process.wait()
                raise
        if code:
            raise RuntimeError(f"{stage} failed with exit {code}; see {stage}.log")

    try:
        run("cuda-check", [sys.executable, "-c",
            "import json,torch; assert torch.cuda.is_available(); "
            "x=torch.randn(32,32,device='cuda'); y=x@x; torch.cuda.synchronize(); "
            "assert bool(y.isfinite().all()); print(json.dumps(dict(torch=torch.__version__, "
            "cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())))"], 180)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        atomic_json(args.out / "corpus-manifest.json", manifest)
        run("download", ["node", ROOT / "tools/nnue/download_corpus.mjs", args.manifest, args.corpus])
        verified = json.loads((args.corpus / "verified.json").read_text(encoding="utf-8"))
        if (verified["source"], verified["revision"], verified["verified"]) != (manifest["source"], manifest["revision"], manifest["files"]):
            raise ValueError("corpus verification does not match the pinned manifest")
        atomic_json(args.out / "verified-corpus.json", verified)
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        paths = [(args.recipe.parent / name).resolve() for name in recipe["files"]]
        expected = [(args.corpus / file["name"]).resolve() for file in manifest["files"]]
        if paths != expected:
            raise ValueError("recipe and verified corpus differ")
        recipe["files"] = [str(path) for path in paths]
        recipe_path = args.out / "training-recipe.json"
        atomic_json(recipe_path, recipe)
        # Inspect each source independently rather than only the first global block.
        from binpack_data import BinpackSource
        checks = []
        for index, path in enumerate(paths):
            single = dict(recipe, files=[str(path)], validation_seed=731 + index)
            single_path = args.out / f"source-{index}-recipe.json"
            atomic_json(single_path, single)
            source = BinpackSource(single_path, args.reader)
            records = source.validation(8192)
            if len(records) != 8192:
                raise ValueError("source sample has too few accepted positions")
            checks.append(dict(file=path.name, accepted=len(records), stats=source.last_stats,
                               min_score=int(records["score"].min()), max_score=int(records["score"].max())))
        atomic_json(args.out / "source-checks.json", checks)
        network_run = args.out / "network"
        run("training", [sys.executable, "-u", ROOT / "tools/nnue/train_nnue.py",
            "--binpack-config", recipe_path, "--binpack-reader", args.reader,
            "--out", network_run, "--device", "cuda", "--threads", 2,
            "--epochs", args.epochs, "--epoch-rows", args.epoch_rows,
            "--batch-size", 8192, "--block-rows", 65536, "--val-rows", args.val_rows,
            "--sigmoid-scale", 200, "--warmup-steps", 200, "--log-every", 128,
            "--max-train-seconds", args.max_train_seconds], 7200)
        provenance = json.loads((network_run / "provenance.json").read_text())
        best = network_run / "best.znnue"
        if hashlib.sha256(best.read_bytes()).hexdigest() != provenance["network_sha256_best"]:
            raise ValueError("best network provenance mismatch")
        history = provenance["history"]
        if not all(math.isfinite(row["val_quant"]) for row in history):
            raise ValueError("non-finite validation loss")
        if min(row["val_quant"] for row in history) >= history[0]["val_baseline"]:
            raise ValueError("candidate does not improve over zero-score validation baseline")
        parity_records = args.out / "parity.bin"
        run("parity-data", [sys.executable, "-c",
            "import sys; sys.path.insert(0,'tools/nnue'); from binpack_data import BinpackSource; "
            "BinpackSource(sys.argv[1],sys.argv[2]).validation(2048).tofile(sys.argv[3])",
            recipe_path, args.reader, parity_records], 180)
        run("parity", [sys.executable, ROOT / "tools/nnue/check_parity.py", "--net", best,
            "--records", parity_records, "--zenith-nnue", args.nnue_harness, "--count", 500], 180)
        # A finite paired match; the runner checks PGN legality and forfeits.
        run("match", [sys.executable, ROOT / "tools/engine_test.py", args.engine, args.engine,
            args.out / "match", "--pairs", args.match_pairs, "--split", "calibration",
            "--threads", 1, "--concurrency", 2, "--tc", "0:02+0.05", "--timeout", 1200,
            "--candidate-option", f"EvalFile={best}", "--candidate-option", "Use NNUE=true",
            "--baseline-option", "Use NNUE=false"], 1260)
        update("complete", best_network=str(best), network_sha256=provenance["network_sha256_best"],
               completed_epochs=len(history), trained_positions=len(history) * args.epoch_rows,
               training_seconds=provenance["training_seconds"],
               best_validation=min(row["val_quant"] for row in history),
               note="Candidate only. A small paired match does not establish Elo or authorize replacing the default evaluator.")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

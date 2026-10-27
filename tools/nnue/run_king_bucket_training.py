#!/usr/bin/env python3
"""Finite matched experiment: cloned v2 vs v1 fine tuning on identical data.

Both start from the same v1 checkpoint, use the same full LR schedule, seed,
positions, batch size and validation. V2's completed epoch count bounds the
control run. Existing verified data is reused; no downloads or model promotion.
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

from run_first_training import ROOT, atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_snapshot(recipe_path, integrity_path):
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "tools/nnue/corpora/first-50gb.json").read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("ok") or integrity["revision"] != manifest["revision"]:
        raise ValueError("full corpus verification report is missing or differs from pinned source")
    paths = [(recipe_path.parent / name).resolve() for name in recipe["files"]]
    expected = {file["name"]: file for file in manifest["files"]}
    checked = {file["name"]: file for file in integrity["files"]}
    if len(paths) != len(expected) or {p.name for p in paths} != set(expected):
        raise ValueError("recipe does not describe the pinned corpus")
    snapshot = []
    for path in paths:
        if path != (ROOT / "data/nnue/sf-first-50gb" / path.name).resolve():
            raise ValueError("cached verification belongs to the pinned corpus directory")
        entry, check, stat = expected[path.name], checked[path.name], path.stat()
        if (not check["ok"] or check["sha256"] != entry["sha256"]
                or check["bytes"] != entry["bytes"] or stat.st_size != entry["bytes"]
                or stat.st_mtime > integrity["checked_at_unix"]):
            raise ValueError(f"corpus changed since full verification: {path}; recheck its SHA-256")
        snapshot.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                             verified_sha256=entry["sha256"]))
    recipe["files"] = [str(path) for path in paths]
    return recipe, snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init-model", type=Path, default=ROOT / "runs/first-50gb-control-20261026/network/best.pt")
    ap.add_argument("--recipe", type=Path, default=ROOT / "tools/nnue/corpora/first-50gb-training.json")
    ap.add_argument("--integrity", type=Path, default=ROOT / "runs/first-50gb-control-20261026/post-outage-integrity.json")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--epoch-rows", type=int, default=8_388_608)
    ap.add_argument("--seconds-per-net", type=float, default=1800)
    ap.add_argument("--pairs", type=int, default=64)
    args = ap.parse_args()
    if min(args.epochs, args.epoch_rows, args.pairs) <= 0 or not math.isfinite(args.seconds_per_net) or not 0 < args.seconds_per_net <= 3600:
        ap.error("positive counts and at most one hour per network required")
    engine = ROOT / "target/release/zenith.exe"
    harness = ROOT / "target/release/examples/nnue.exe"
    reader = ROOT / "target/binpack/release/zenith-binpack.exe"
    for path in [engine, harness, reader, args.init_model, args.recipe, args.integrity]:
        if not path.is_file():
            ap.error(f"missing required file: {path}")
    recipe, snapshot = corpus_snapshot(args.recipe, args.integrity)
    base = json.loads((args.init_model.parent / "provenance.json").read_text())
    selected = json.loads((args.init_model.parent / f"epoch-{base['best_epoch']:04}.json").read_text())
    if digest(args.init_model) != selected["checkpoint_sha256"]:
        ap.error("initial checkpoint does not match its selected-epoch provenance")
    original_net = args.init_model.parent / "best.znnue"
    if digest(original_net) != base["network_sha256_best"]:
        ap.error("initial v1 network does not match provenance")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    recipe_file = args.out / "recipe.json"
    atomic_json(recipe_file, recipe)
    plan = {"args": {key: str(value) for key, value in vars(args).items()},
            "source_checkpoint_sha256": digest(args.init_model), "initial_network_sha256": digest(original_net),
            "corpus_snapshot": snapshot,
            "files_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
                engine, harness, reader, Path(__file__), ROOT / "tools/nnue/train_nnue.py",
                ROOT / "tools/nnue/check_parity.py", ROOT / "tools/nnue/binpack_data.py"]}}
    atomic_json(args.out / "plan.json", plan)
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
                code = child.wait(timeout=timeout)
            except BaseException:
                child.kill(); child.wait()
                raise
        if code:
            raise RuntimeError(f"{stage} failed ({code}); inspect {stage}.log")

    try:
        common = [sys.executable, "-u", ROOT / "tools/nnue/train_nnue.py", "--binpack-config", recipe_file,
                  "--binpack-reader", reader, "--init-model", args.init_model, "--epochs", args.epochs,
                  "--epoch-rows", args.epoch_rows, "--batch-size", 8192, "--block-rows", 65536,
                  "--val-rows", 131072, "--val-chunk", 4096, "--device", "cuda", "--threads", 2,
                  "--seed", 2, "--sigmoid-scale", 200, "--lr", 3e-4, "--lr-final", 1e-5,
                  "--warmup-steps", 128, "--log-every", 128, "--max-train-seconds", args.seconds_per_net]
        v2 = args.out / "v2"
        run("train-v2", [*common, "--king-buckets", 4, "--out", v2], args.seconds_per_net + 900)
        p2 = json.loads((v2 / "provenance.json").read_text())
        epochs = len(p2["history"])
        v1 = args.out / "v1-control"
        run("train-v1-control", [*common, "--king-buckets", 1, "--out", v1,
                                 "--stop-after-epochs", epochs], args.seconds_per_net + 900)
        p1 = json.loads((v1 / "provenance.json").read_text())
        if len(p1["history"]) != epochs:
            raise ValueError("time budget stopped control early; unequal exposures cannot be compared")
        if not (p1["initialization"]["source_sha256"] == p2["initialization"]["source_sha256"]
                == plan["source_checkpoint_sha256"]):
            raise ValueError("initial checkpoint changed between training runs")
        if digest(v1 / "initial.znnue") != base["network_sha256_best"]:
            raise ValueError("control did not start from the exact original v1 export")
        contract2 = dict(p2["contract"]); contract2.pop("king_buckets")
        if contract2 != p1["contract"]:
            raise ValueError("control training contract differs beyond king buckets")
        for item in snapshot:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise ValueError("corpus mutated during training")
        parity_records = args.out / "parity.bin"
        run("parity-data", [sys.executable, "-c",
            "import sys; sys.path.insert(0,'tools/nnue'); from binpack_data import BinpackSource; "
            "BinpackSource(sys.argv[1],sys.argv[2]).validation(2048).tofile(sys.argv[3])",
            recipe_file, reader, parity_records], 300)
        for tag, directory, provenance in [("v2", v2, p2), ("v1", v1, p1)]:
            if digest(directory / "best.znnue") != provenance["network_sha256_best"]:
                raise ValueError(f"{tag} export SHA256 differs from provenance")
            run(f"parity-{tag}", [sys.executable, ROOT / "tools/nnue/check_parity.py", "--net", directory / "best.znnue",
                "--records", parity_records, "--zenith-nnue", harness, "--count", 500], 180)
        match_common = [sys.executable, ROOT / "tools/engine_test.py", engine, engine]
        match_options = ["--pairs", args.pairs, "--split", "holdout", "--threads", 1, "--concurrency", 4,
                         "--hash", 32, "--tc", "0:05+0.05", "--timeout", 1200,
                         "--candidate-option", f"EvalFile={v2 / 'best.znnue'}", "--candidate-option", "Use NNUE=true",
                         "--baseline-option", "Use NNUE=true"]
        run("matched-match", [*match_common, args.out / "matched-match", *match_options,
            "--baseline-option", f"EvalFile={v1 / 'best.znnue'}"], 1260)
        run("original-match", [*match_common, args.out / "original-match", *match_options,
            "--baseline-option", f"EvalFile={original_net}"], 1260)
        matched = json.loads((args.out / "matched-match/summary.json").read_text())
        original = json.loads((args.out / "original-match/summary.json").read_text())
        update("complete", completed_epochs=epochs, additional_presentations_per_network=epochs * args.epoch_rows,
               networks={"v2": str(v2 / "best.znnue"), "v1_control": str(v1 / "best.znnue")},
               matched_wdl=matched["pgn"]["wdl"], original_wdl=original["pgn"]["wdl"],
               note="Experiment only: inspect matched games and uncertainty before choosing a new recommended network.")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

"""Alternate two builds on identical NNUE inference and fixed-depth search work.

Only timings may differ. Scores, moves, work counters and inference checksums
must agree before a performance comparison is emitted. Run without other jobs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, check=True, timeout=120)
    return result.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ["baseline", "candidate", "baseline-nnue", "candidate-nnue", "network", "out"]:
        ap.add_argument("--" + flag, required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--positions", type=int, default=500000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()
    if min(args.repeats, args.positions, args.depth) <= 0:
        ap.error("positive repeats, positions and depth required")
    paths = {name: getattr(args, name).resolve(strict=True) for name in
             ["baseline", "candidate", "baseline_nnue", "candidate_nnue", "network"]}
    runs = {name: [] for name in ["baseline", "candidate"]}
    signature, checksum = None, None
    for repeat in range(args.repeats):
        for name in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
            micro = run([paths[name + "_nnue"], "bench", paths["network"], args.positions])
            matches = re.findall(r"^(refresh\+head|delta\+head):.*? ([\d.]+) ns/position, checksum (-?\d+)$", micro, re.M)
            if len(matches) != 2 or matches[0][2] != matches[1][2]:
                raise ValueError("inference benchmark did not return two matching checksums")
            current = matches[0][2]
            if checksum is not None and current != checksum:
                raise ValueError("inference sums differ across builds/runs")
            checksum = current
            search = run([paths[name], "bench", args.depth, "--eval-file", paths["network"]])
            # Timing fields are the only permitted differences in this workload.
            stable = re.sub(r"\b(?:time_us|nps)=\d+", "TIMING", search)
            if signature is not None and stable != signature:
                raise ValueError("search scores/moves/nodes/counters differ across builds/runs")
            signature = stable
            total = re.search(r"^total nodes=(\d+) time_us=(\d+) nps=(\d+)$", search, re.M)
            if total is None:
                raise ValueError("missing search totals")
            item = dict(repeat=repeat, refresh_ns=float(matches[0][1]), delta_ns=float(matches[1][1]),
                        search_us=int(total[2]), nodes=int(total[1]), micro=micro, search=search)
            runs[name].append(item)
            print(json.dumps({k: v for k, v in item.items() if k not in ["micro", "search"]} | {"build": name}), flush=True)
    medians = {name: {key: statistics.median(item[key] for item in items)
                     for key in ["refresh_ns", "delta_ns", "search_us"]} for name, items in runs.items()}
    reduction = {key: (1 - medians["candidate"][key] / medians["baseline"][key]) * 100
                 for key in medians["baseline"]}
    report = dict(args={k: str(v) for k, v in vars(args).items()},
                  sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                  runs=runs, medians=medians, reduction_percent=reduction, work_identical=True,
                  note="Local timings and fixed workload; not a playing-strength measurement.")
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"medians": medians, "reduction_percent": reduction, "work_identical": True}))


if __name__ == "__main__":
    main()

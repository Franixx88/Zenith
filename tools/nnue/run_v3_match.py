#!/usr/bin/env python3
"""Freeze an engine and two local networks for a longer, finite paired match."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from awake_time import wait_process
from run_first_training import ROOT, atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", type=Path, default=ROOT / "target/release/zenith.exe")
    ap.add_argument("--network", type=Path, default=ROOT / "networks/zenith-v3-576258170751.znnue")
    ap.add_argument("--baseline-net", type=Path, default=ROOT / "networks/zenith-v1-1c7c79756c6e.znnue")
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tc", default="0:15+0.15")
    ap.add_argument("--seconds", type=int, default=7200)
    args = ap.parse_args()
    if args.pairs <= 0 or not 1 <= args.concurrency <= (os.cpu_count() or 1) or args.seconds <= 0:
        ap.error("positive pairs/seconds and concurrency within logical CPU count required")
    sources = [p.resolve(strict=True) for p in [args.engine,args.network,args.baseline_net]]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source, name in zip(sources,["zenith.exe","candidate.znnue","baseline.znnue"]):
        data=source.read_bytes()
        (args.out / name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    command=[sys.executable,"-u",ROOT / "tools/engine_test.py",args.out / "zenith.exe",args.out / "zenith.exe",args.out / "match",
             "--pairs",args.pairs,"--concurrency",args.concurrency,"--threads",1,"--hash",32,
             "--tc",args.tc,"--overhead",50,"--timeout",args.seconds,"--seed",20261027,"--split","holdout",
             "--candidate-option",f"EvalFile={args.out / 'candidate.znnue'}","--candidate-option","Use NNUE=true",
             "--baseline-option",f"EvalFile={args.out / 'baseline.znnue'}","--baseline-option","Use NNUE=true",
             "--candidate-option","Context Histories=false","--baseline-option","Context Histories=false"]
    command=[str(x) for x in command]
    atomic_json(args.out / "plan.json",dict(args={k:str(v) for k,v in vars(args).items()},
        commit=commit,sha256=hashes,command=command,source_files=[str(p) for p in sources]))
    state=dict(pid=os.getpid(),stage="starting",started_at=time.time())
    def update(stage,**extra):
        state.update(stage=stage,updated_at=time.time(),**extra)
        atomic_json(args.out / "status.json",state)
        print(json.dumps(state),flush=True)
    child=None
    try:
        with (args.out / "match.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            update("match",child_pid=child.pid)
            code=wait_process(child,args.seconds+60)
        if code:
            raise RuntimeError(f"match failed ({code}); inspect match.log and match/summary.json")
        summary=json.loads((args.out / "match/summary.json").read_text())
        if not summary["valid"]:
            raise ValueError("match did not pass PGN/protocol validation")
        update("complete",wdl=summary["pgn"]["wdl"],games=summary["pgn"]["games"],
               note="Fixed-length relative comparison; inspect uncertainty before promotion.")
    except BaseException as error:
        if child is not None and child.poll() is None:
            if os.name=="nt":
                subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True)
            else:
                child.kill()
            child.wait()
        update("failed",error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

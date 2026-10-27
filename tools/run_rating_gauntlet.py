#!/usr/bin/env python3
"""Finite external rating experiment: scout, fresh holdout, cross games, Ordo.

Uses published CCRL versions as reference anchors, not official CCRL testing.
Snapshots native binaries and weights; validates every full match before fitting.
Scout games select opponents but never enter the rating PGN.
"""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import chess.pgn
import engine_test as E
from nnue.awake_time import awake_seconds

ROOT = Path(__file__).resolve().parents[1]
ZENITH = "Zenith v3 12ce81e"


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_ratings(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, skipinitialspace=True))
    header = [value.strip().lower() for value in rows[0]]
    name, rating = header.index("player"), header.index("rating")
    return {row[name].strip(): float(row[rating]) for row in rows[1:] if row}


def opening_families(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        family, moves = line.split("|", 1)
        result[tuple(moves.split())] = family
    return result


def combine(matches, output, family_map):
    """Preserve color pairs and family membership for cluster resampling."""
    clusters = {}
    games = 0
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for directory, white_name, black_name in matches:
            with (directory / "games.pgn").open(encoding="utf-8-sig") as source:
                while (game := chess.pgn.read_game(source)) is not None:
                    names = {"candidate": white_name, "baseline": black_name}
                    game.headers["White"] = names[game.headers["White"]]
                    game.headers["Black"] = names[game.headers["Black"]]
                    moves = tuple(move.uci() for move in itertools.islice(game.mainline_moves(), 8))
                    family = family_map[moves]
                    game.headers["OpeningFamily"] = family
                    game.headers["Event"] = directory.name
                    target.write(str(game) + "\n\n")
                    # Ordo reads results, not board evaluations. Minimal PGNs
                    # keep the resampling small; the full evidence is above.
                    record = f'[White "{game.headers["White"]}"]\n[Black "{game.headers["Black"]}"]\n[Result "{game.headers["Result"]}"]\n\n{game.headers["Result"]}\n\n'
                    clusters.setdefault(family, []).append(record)
                    games += 1
    return clusters, games


def run_ordo(binary, pgn, csv_path, text_path, anchor, anchor_rating, simulations=0):
    command = [str(binary), "-q", "-p", str(pgn), "-A", anchor, "-a", str(anchor_rating),
               "-W", "-D", "-N", "2,2", "-c", str(csv_path), "-o", str(text_path)]
    if simulations:
        command += ["-s", str(simulations), "-F", "95", "-n", "1"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Ordo failed: {result.stdout[-2000:]} {result.stderr[-2000:]}")
    ratings = read_ratings(csv_path)
    if ZENITH not in ratings or not math.isfinite(ratings[ZENITH]):
        raise ValueError("Ordo did not produce a finite Zenith rating")
    return ratings


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def bootstrap(binary, clusters, folder, anchor, anchor_rating, samples, deadline=None):
    """Resample opening families across ALL pairings, retaining both colors.

    This covers opening/sample uncertainty conditional on one reference rating.
    It does not quantify a different clock/hardware/tablebase policy from CCRL.
    """
    rng = random.Random(20261028)
    keys = sorted(clusters)
    values, errors = [], []
    pgn, csv_path, text = folder / "sample.pgn", folder / "sample.csv", folder / "sample.txt"
    for _ in range(samples):
        if deadline is not None and awake_seconds() >= deadline:
            raise TimeoutError("rating bootstrap exceeded awake budget; completed matches can be resumed")
        pgn.write_text("".join(record for key in rng.choices(keys, k=len(keys)) for record in clusters[key]), encoding="utf-8")
        try:
            ratings = run_ordo(binary, pgn, csv_path, text, anchor, anchor_rating)
            values.append(ratings[ZENITH])
        except (RuntimeError, ValueError) as error:
            errors.append(str(error)[:1000])
    if len(values) < max(1, math.ceil(samples * 0.95)):
        raise ValueError("too many disconnected/uninformative bootstrap samples; collect more games")
    return dict(samples=samples, valid_samples=len(values), opening_families=len(keys),
                percentile95=[percentile(values, 0.025), percentile(values, 0.975)],
                ratings=values, errors=errors,
                interpretation="Opening-family cluster bootstrap, conditional on fixed anchor; excludes reference/systematic uncertainty.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=ROOT / "tools/rating-pool.json")
    ap.add_argument("--scout-pairs", type=int, default=16)
    ap.add_argument("--main-pairs", type=int, default=96)
    ap.add_argument("--cross-pairs", type=int, default=16)
    ap.add_argument("--select", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=10800)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if min(args.scout_pairs,args.main_pairs,args.cross_pairs,args.select,args.seconds,args.bootstrap) <= 0:
        ap.error("counts and budget must be positive")
    if not 1 <= args.concurrency <= (os.cpu_count() or 1):
        ap.error("invalid concurrency")
    pool = json.loads(args.pool.read_text())
    if not 2 <= args.select <= len(pool["opponents"]):
        ap.error("select must be 2..pool size")
    args.out = args.out.resolve()
    contract = {k:v for k,v in vars(args).items() if k not in ["out","pool","resume","seconds"]}
    if args.resume:
        plan = json.loads((args.out / "plan.json").read_text())
        if plan["contract"] != contract:
            raise ValueError("resume contract changed")
        for file, digest in plan["frozen_sha256"].items():
            if E.sha256(args.out / file) != digest:
                raise ValueError(f"frozen file changed: {file}")
        pool = plan["pool"]
        if E.sha256(Path(__file__)) != plan["runner_sha256"]:
            raise ValueError("orchestrator source changed; review resume compatibility")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "binaries").mkdir()
        frozen = {}
        zenith_manifest=json.loads((ROOT / "networks/layered-control-20261027.json").read_text())
        sources = [(ROOT / "target/release/zenith.exe", "binaries/zenith.exe", zenith_manifest["speed"]["candidate_sha256"]),
                   (ROOT / "target/fastchess/fastchess-windows-x86-64/fastchess.exe", "binaries/fastchess.exe", None),
                   (ROOT / "data/openings.txt", "openings-source.txt", None),
                   (ROOT / "runs/ccrl-blitz-20261027.html", "reference.html", pool["reference_sha256"]),
                   (ROOT / "networks/zenith-v3-576258170751.znnue", "zenith-v3.znnue", "576258170751e54bbf6c0e7729cbe432168fdbc474a27a4f67e1214a7f4891d9"),
                   (ROOT / pool["ordo"]["binary"], "binaries/ordo.exe", pool["ordo"]["sha256"])]
        sources += [(ROOT / p["binary"], f'binaries/{p["id"]}.exe', p["sha256"]) for p in pool["opponents"]]
        for source, name, expected in sources:
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if expected is not None and digest != expected:
                raise ValueError(f"source SHA-256 mismatch: {source}")
            (args.out / name).write_bytes(data)
            frozen[name] = digest
        plan = dict(contract=contract, pool=pool, frozen_sha256=frozen,
                    runner_sha256=E.sha256(Path(__file__)), engine_test_sha256=E.sha256(Path(E.__file__)),
                    source_commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    policy="Local estimate linked to CCRL Blitz references. Scout excluded. No automatic network promotion.")
        save(args.out / "plan.json", plan)
    state = dict(pid=os.getpid(),stage="starting",started_utc=datetime.now(timezone.utc).isoformat())
    started = awake_seconds()
    def update(stage, **extra):
        state.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        save(args.out / "status.json",state)
        print(json.dumps(state),flush=True)
    zenith = args.out / "binaries/zenith.exe"
    settings = [f'EvalFile={args.out / "zenith-v3.znnue"}',"Use NNUE=true","Context Histories=false"]
    book = args.out / "openings-source.txt"
    counter = 0
    def match(label, candidate, baseline, c_options, b_options, pairs, split, tc):
        nonlocal counter
        counter += 1
        # A previous incomplete attempt is retained, never overwritten or rated.
        destination = args.out / label
        for attempt in range(100):
            destination = args.out / (label if attempt == 0 else f"{label}-retry{attempt}")
            if not destination.exists():
                break
            summary_file = destination / "summary.json"
            if summary_file.is_file():
                summary = json.loads(summary_file.read_text())
                if summary.get("valid"):
                    return destination, summary
        else:
            raise ValueError("too many incomplete attempts")
        if E.sha256(Path(E.__file__)) != plan["engine_test_sha256"]:
            raise ValueError("match runner source changed; resume after reviewing changes")
        remaining = args.seconds - (awake_seconds() - started)
        if remaining < 60:
            raise TimeoutError("overall awake budget exhausted; completed matches can be resumed")
        update(label,match_directory=str(destination),completed_steps=counter-1)
        job = argparse.Namespace(candidate=str(candidate),baseline=str(baseline),output=str(destination),
            fastchess=str(args.out / "binaries/fastchess.exe"),book=str(book),
            pairs=pairs,threads=1,concurrency=args.concurrency,hash=128,overhead=50,timeout=int(min(remaining,3600)),
            nodes=None,sprt=None,engine_log=False,external_engines=True,seed=20261028+counter,split=split,tc=tc,
            candidate_option=c_options,baseline_option=b_options)
        with (args.out / f"{label}.log").open("a",encoding="utf-8") as log:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(log), redirect_stderr(log):
                code=E.run(job)
        summary=json.loads((destination / "summary.json").read_text())
        if code or not summary["valid"]:
            raise RuntimeError(f"{label} invalid; inspect {destination / 'summary.json'}")
        return destination,summary
    try:
        scouts=[]
        for p in pool["opponents"]:
            _,summary=match("scout-"+p["id"],zenith,args.out / f'binaries/{p["id"]}.exe',settings,
                            [f"{k}={v}" for k,v in p["options"].items()],args.scout_pairs,"calibration","0:05+0.05")
            wdl=summary["pgn"]["wdl"]
            score=(wdl.get("wins",0)+wdl.get("draws",0)/2)/summary["pgn"]["games"]
            scouts.append(dict(opponent=p,score=score,wdl=wdl))
            save(args.out / "scouts.json",scouts)
        if all(s["score"]==0 for s in scouts) or all(s["score"]==1 for s in scouts):
            raise ValueError("pool does not bracket Zenith; expand weaker/stronger opponents before rating")
        selected=sorted(scouts,key=lambda s:(abs(s["score"]-0.5),s["opponent"]["ccrl_error95"]))[:args.select]
        save(args.out / "selected.json",selected)
        completed=[]
        for s in selected:
            p=s["opponent"]
            directory,_=match("main-"+p["id"],zenith,args.out / f'binaries/{p["id"]}.exe',settings,
                              [f"{k}={v}" for k,v in p["options"].items()],args.main_pairs,"holdout","0:10+0.10")
            completed.append((directory,ZENITH,p["name"]))
        for left,right in itertools.combinations(selected,2):
            a,b=left["opponent"],right["opponent"]
            directory,_=match(f'cross-{a["id"]}-{b["id"]}',args.out / f'binaries/{a["id"]}.exe',args.out / f'binaries/{b["id"]}.exe',
                [f"{k}={v}" for k,v in a["options"].items()],[f"{k}={v}" for k,v in b["options"].items()],args.cross_pairs,"holdout","0:10+0.10")
            completed.append((directory,a["name"],b["name"]))
        update("rating")
        clusters,games=combine(completed,args.out / "rated-games.pgn",opening_families(book))
        anchor=selected[0]["opponent"]
        ordo=args.out / "binaries/ordo.exe"
        ratings=run_ordo(ordo,args.out / "rated-games.pgn",args.out / "ratings.csv",args.out / "ratings.txt",anchor["name"],anchor["ccrl_blitz_rating"],2000)
        sensitivity={s["opponent"]["name"]:ratings[ZENITH]+s["opponent"]["ccrl_blitz_rating"]-ratings[s["opponent"]["name"]] for s in selected}
        update("bootstrap",point_estimate=ratings[ZENITH],anchor=anchor["name"])
        folder=args.out / "bootstrap";folder.mkdir(exist_ok=True)
        boot=bootstrap(ordo,clusters,folder,anchor["name"],anchor["ccrl_blitz_rating"],args.bootstrap,started+args.seconds)
        save(args.out / "bootstrap.json",boot)
        report=dict(point_estimate=ratings[ZENITH],cluster_interval95_conditional=boot["percentile95"],
                    anchor=anchor,anchor_sensitivity=sensitivity,reference_anchor_range=[min(sensitivity.values()),max(sensitivity.values())],
                    games=games,zenith_games=args.select*args.main_pairs*2,cross_games=games-args.select*args.main_pairs*2,
                    scout_games_excluded=len(scouts)*args.scout_pairs*2,ratings=ratings,
                    conditions=dict(tc="10+0.10",threads=1,hash_mib=128,concurrency=args.concurrency,syzygy=False),
                    warning="Approximate CCRL-linked local pool estimate; interval excludes published-anchor and clock/hardware/tablebase systematic uncertainty.")
        save(args.out / "rating.json",report)
        update("complete",rating=ratings[ZENITH],interval95=boot["percentile95"],games=games,reference_anchor_range=report["reference_anchor_range"])
    except BaseException as error:
        update("failed",error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

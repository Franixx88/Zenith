"""Reproducible paired matches; statistics/SPRT are delegated to Fastchess.

Requires python-chess (the NNUE venv already contains it). No engine Elo claim
is inferred from a small match. Every completed PGN is independently replayed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
FAULT = re.compile(r"^.*(?:loses on time|disconnect|illegal move|stalls|crash|connection stalls|does not respond).*$", re.M | re.I)


def terminate_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        process.terminate()
    process.wait()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sprt_decision(log_text, enabled):
    decisions = re.findall(r"\b(H[01]) was accepted\b", log_text)
    # The sequential test stops at the FIRST boundary crossing. Later games
    # already in flight do not restart the test or change its decision.
    return decisions[0] if decisions else "inconclusive" if enabled else "fixed_length"


def options(items):
    result = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or key in result:
            raise ValueError(f"expected a unique Name=value: {item!r}")
        if key.lower() in {"threads", "hash", "syzygypath", "move overhead"}:
            raise ValueError(f"use the dedicated argument for {key}")
        result[key] = value
    return result


def preflight(binary, settings):
    with chess.engine.SimpleEngine.popen_uci(str(binary), timeout=15) as engine:
        parsed = {}
        for key, value in settings.items():
            if key not in engine.options:
                raise ValueError(f"{binary.name}: unknown option {key}")
            option = engine.options[key]
            # python-chess Option.parse treats a nonempty string 'false' as true.
            if option.type == "check":
                if str(value).lower() not in {"true", "false"}:
                    raise ValueError(f"{key}: expected true or false")
                value = str(value).lower() == "true"
            parsed[key] = option.parse(value)
        engine.configure(parsed)
        engine.ping()
        return {"id": engine.id, "options": parsed}


def opening_book(source, output, split):
    seen = set()
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            family, moves = line.split("|", 1)
            number = int(family.removeprefix("F"))
            if (split == "calibration" and number > 16) or (split == "holdout" and number <= 16):
                continue
            game = chess.pgn.Game()
            game.headers["Event"] = family
            node = game
            board = game.board()
            for token in moves.split():
                move = board.parse_uci(token)
                board.push(move)
                node = node.add_variation(move)
            if board.ply() != 8 or board.is_game_over(claim_draw=True):
                raise ValueError("opening must have exactly eight nonterminal legal plies")
            key = board.fen(en_passant="legal").rsplit(" ", 2)[0]
            if key in seen:
                raise ValueError("duplicate opening root")
            seen.add(key)
            target.write(str(game) + "\n\n")
            count += 1
    if not count:
        raise ValueError("empty opening book")
    return count


def validate_pgn(path, allow_singletons=False):
    """Fail closed: faults, incomplete/color-mismatched pairs are not evidence."""
    pairs = defaultdict(list)
    reasons = Counter()
    wdl = Counter()
    errors = []
    games = 0
    with Path(path).open(encoding="utf-8-sig") as source:
        while (game := chess.pgn.read_game(source)) is not None:
            games += 1
            if game.errors:
                errors.append(f"game {games}: PGN errors {game.errors}")
                continue
            board = game.board()
            opening = []
            for ply, move in enumerate(game.mainline_moves()):
                if move not in board.legal_moves:
                    errors.append(f"game {games}: illegal move {move}")
                    break
                board.push(move)
                if ply < 8:
                    opening.append(move.uci())
            headers = game.headers
            reason = headers.get("Termination", "missing")
            reasons[reason] += 1
            outcome = board.outcome(claim_draw=True)
            result = headers.get("Result")
            # No resignation/adjudication enabled: terminal board must explain result.
            if outcome is None or outcome.result() != result:
                errors.append(f"game {games}: unexplained result {result}, termination={reason}")
            white, black = headers.get("White"), headers.get("Black")
            if {white, black} != {"candidate", "baseline"} or result not in {"1-0", "0-1", "1/2-1/2"}:
                errors.append(f"game {games}: bad players/result")
                continue
            score = 1 if result == "1/2-1/2" else 2 * int((result == "1-0") == (white == "candidate"))
            wdl[{2: "wins", 1: "draws", 0: "losses"}[score]] += 1
            # Fastchess Round is the pair ID; arrival order is concurrent.
            pairs[headers.get("Round")].append((white, tuple(opening), score))
    penta = [0] * 5
    unpaired = []
    paired_wdl = Counter()
    for round_id, pair in pairs.items():
        # A normal SPRT stop cancels games already in flight. Keep their
        # completed partners visible, but only COMPLETE pairs enter evidence.
        if allow_singletons and len(pair) == 1 and len(pair[0][1]) == 8:
            unpaired.append({"round": round_id, "candidate_white": pair[0][0] == "candidate", "score_halfpoints": pair[0][2]})
            continue
        if len(pair) != 2 or {g[0] for g in pair} != {"candidate", "baseline"} or pair[0][1] != pair[1][1] or len(pair[0][1]) != 8:
            errors.append(f"round {round_id}: missing or mismatched color pair")
            continue
        penta[pair[0][2] + pair[1][2]] += 1
        for game in pair:
            paired_wdl[{2: "wins", 1: "draws", 0: "losses"}[game[2]]] += 1
    if not games:
        errors.append("no completed games")
    return {"games": games, "pairs": sum(penta), "wdl": dict(wdl), "paired_wdl": dict(paired_wdl),
            "unpaired_games": unpaired, "pentanomial": penta,
            "terminations": dict(reasons), "errors": errors, "valid": not errors}


def run(args):
    if args.pairs < 1 or args.threads < 1 or args.concurrency < 1 or args.hash < 1:
        raise ValueError("pairs/threads/concurrency/hash must be positive")
    if args.timeout < 1 or (args.nodes is not None and args.nodes < 1):
        raise ValueError("timeout/nodes must be positive")
    if args.threads * args.concurrency > (os.cpu_count() or 1):
        raise ValueError("Threads * concurrency exceeds logical CPU count")
    if args.sprt and (not all(math.isfinite(n) for n in args.sprt) or args.sprt[0] >= args.sprt[1]):
        raise ValueError("SPRT elo0 must be less than elo1")
    binaries = [Path(p).resolve(strict=True) for p in (args.candidate, args.baseline, args.fastchess)]
    book = Path(args.book).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Same tablebase policy even when Zenith discovers its local installation.
    os.environ["ZENITH_SYZYGY_PATH"] = "<empty>"
    common = {"Threads": args.threads, "Hash": args.hash, "Move Overhead": args.overhead,
              "SyzygyPath": "<empty>"}
    settings = [common | options(args.candidate_option), common | options(args.baseline_option)]
    preflights = [preflight(binary, setting) for binary, setting in zip(binaries, settings)]
    count = opening_book(book, output / "openings.pgn", args.split)
    argv = [str(binaries[2])]
    for name, binary, setting in zip(("candidate", "baseline"), binaries, settings):
        argv += ["-engine", f"cmd={binary}", f"name={name}"]
        argv += [f"option.{key}={value}" for key, value in setting.items()]
    argv += ["-each", f"tc={args.tc}", "timemargin=100", "proto=uci",
             "-rounds", str(args.pairs), "-repeat", "-concurrency", str(args.concurrency),
             "-openings", f"file={output / 'openings.pgn'}", "format=pgn", "order=random", "plies=8",
             "-srand", str(args.seed), "-output", "format=fastchess", "-report", "penta=true",
             "-pgnout", f"file={output / 'games.pgn'}", "notation=san", "nodes=true", "append=false",
             "-config", f"outname={output / 'fastchess.json'}", "-scoreinterval", "8", "-ratinginterval", "8"]
    if args.nodes:
        argv += ["-each", f"nodes={args.nodes}"]
    if args.sprt:
        argv += ["-sprt", f"elo0={args.sprt[0]}", f"elo1={args.sprt[1]}", "alpha=0.05", "beta=0.05", "model=normalized"]
    if args.engine_log:
        argv += ["-log", f"file={output / 'protocol.log'}", "level=trace", "engine=true", "append=false"]
    source_files = sorted((ROOT / "src").rglob("*.rs")) + [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
                "args": vars(args), "engines": preflights, "logical_cpus": os.cpu_count(),
                "platform": platform.platform(), "processor": platform.processor(),
                "python_version": platform.python_version(),
                "sha256": {str(p): sha256(p) for p in [*binaries, book, output / "openings.pgn"]},
                "openings": count, "python_chess": chess.__version__,
                "runner_sha256": sha256(__file__), "tablebases": "disabled",
                "workspace_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                "source_note": "Workspace snapshot for provenance; binary hashes identify the actual tested engines."}
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    (output / "openings-source.txt").write_bytes(book.read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Running {args.pairs} pairs, Threads={args.threads}, concurrency={args.concurrency}: {output}", flush=True)
    start = time.monotonic()
    aborted = None
    with (output / "fastchess.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            while process.poll() is None:
                time.sleep(1)
                elapsed = int(time.monotonic() - start)
                current_log = (output / "fastchess.log").read_text(encoding="utf-8", errors="replace")
                if FAULT.search(current_log) or elapsed >= args.timeout:
                    aborted = "engine/protocol fault" if FAULT.search(current_log) else "wall-clock limit"
                    terminate_tree(process)
                    break
                if elapsed and elapsed % 30 == 0:
                    print(f"Match running: {elapsed}s", flush=True)
        except BaseException:
            # Terminate the tournament and both engine children on Windows.
            terminate_tree(process)
            raise
    log_text = (output / "fastchess.log").read_text(encoding="utf-8", errors="replace")
    decision = sprt_decision(log_text, bool(args.sprt))
    faults = FAULT.findall(log_text)
    normal_sprt_stop = bool(args.sprt) and decision in {"H0", "H1"} and process.returncode == 0 and not faults and aborted is None
    validation = validate_pgn(output / "games.pgn", allow_singletons=normal_sprt_stop) if (output / "games.pgn").exists() else {"valid": False, "errors": ["no PGN"]}
    valid = validation["valid"] and process.returncode == 0 and not faults and aborted is None
    reported_penta = re.findall(r"Ptnml\(0-2\): \[([\d, ]+)\]", log_text)
    if not reported_penta or [int(n) for n in reported_penta[-1].split(",")] != validation.get("pentanomial"):
        valid = False
        validation["errors"].append("PGN pentanomial disagrees with Fastchess final statistics")
    if not normal_sprt_stop and validation.get("pairs") != args.pairs:
        valid = False
        validation["errors"].append("match did not finish its requested pairs or accept a hypothesis")
    validation["valid"] = not validation["errors"]
    summary = {"valid": valid, "exit_code": process.returncode, "aborted": aborted, "seconds": time.monotonic() - start,
               "sprt_decision": decision if valid else "invalid", "faults": faults, "pgn": validation}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("output")
    parser.add_argument("--fastchess", default=str(ROOT / "target/fastchess/fastchess-windows-x86-64/fastchess.exe"))
    parser.add_argument("--book", default=str(ROOT / "data/openings.txt"))
    parser.add_argument("--pairs", type=int, default=128, help="hard cap, two games per pair")
    parser.add_argument("--tc", default="0:02+0.05")
    parser.add_argument("--nodes", type=int, help="optional fixed nodes per move in addition to clocks")
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--hash", type=int, choices=range(1, 1025), default=16)
    parser.add_argument("--overhead", type=int, choices=range(0, 5001), default=30)
    parser.add_argument("--timeout", type=int, default=3600, help="hard wall-clock cap in seconds; interrupted runs are invalid")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--split", choices=["all", "calibration", "holdout"], default="all")
    parser.add_argument("--candidate-option", action="append", default=[])
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--sprt", type=float, nargs=2, metavar=("ELO0", "ELO1"))
    parser.add_argument("--engine-log", action="store_true", help="save full UCI trace (can be large)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

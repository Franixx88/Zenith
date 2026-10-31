"""Measure actual go-to-bestmove latency on tiny clocks, with legal-move checks.

Run independently of builds/training. This is a bounded protocol stress test,
not an Elo benchmark. Each process is warmed by UCI/ready, not by a search.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import chess
import chess.pgn

FENS = [chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnb1kbnr/pppp1ppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"]


def worker(args, index):
    env={**os.environ,"ZENITH_SYZYGY_PATH":"<empty>"}
    proc=subprocess.Popen([str(args.engine)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL,text=True,bufsize=1,env=env)
    lines=queue.Queue()
    def read():
        for line in proc.stdout:lines.put(line.strip())
        lines.put(None)
    reader=threading.Thread(target=read,daemon=True);reader.start()
    def send(command):
        proc.stdin.write(command+"\n");proc.stdin.flush()
    def until(prefix):
        transcript=[]
        deadline=time.monotonic()+20
        while True:
            line=lines.get(timeout=max(.001,deadline-time.monotonic()))
            if line is None:raise RuntimeError("engine disconnected")
            transcript.append(line)
            if line.startswith(prefix):return line,transcript
    samples=[]
    try:
        send("uci");until("uciok")
        for name,value in [("Threads",args.threads),("Hash",32),("Move Overhead",50),
                           ("SyzygyPath","<empty>"),("EvalFile",args.network),("Context Histories","false")]:
            send(f"setoption name {name} value {value}")
        send("isready");until("readyok")
        for i in range(args.rounds):
            fen=args.fens[(i+index)%len(args.fens)]
            clock=[250,50,100,1000][i%4]
            send("position fen "+fen);send("isready");until("readyok")
            command=f"go wtime {clock} btime {clock} winc 10 binc 10"
            start=time.perf_counter();send(command)
            best,transcript=until("bestmove ")
            elapsed=1000*(time.perf_counter()-start)
            move=chess.Move.from_uci(best.split()[1])
            if move not in chess.Board(fen).legal_moves:raise RuntimeError(f"illegal {move}: {fen}")
            available=max(0,clock-50)
            soft=max(1,min(available//30+7,max(1,available//2)))
            hard=min(4*soft,max(1,available//2))
            samples.append(dict(worker=index,round=i,fen=fen,command=command,bestmove=move.uci(),
                                elapsed_ms=elapsed,hard_budget_ms=hard,
                                over_hard_plus_25ms=elapsed>hard+25,
                                over_clock=elapsed>clock,transcript=transcript))
    finally:
        try:send("quit");proc.wait(timeout=5)
        except (OSError,subprocess.TimeoutExpired):proc.kill();proc.wait(timeout=5)
        reader.join(timeout=5)
        proc.stdin.close();proc.stdout.close()
    return samples


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine",type=Path,default=Path("target/release/zenith.exe"))
    ap.add_argument("--network",type=Path,default=Path("networks/zenith-v3-576258170751.znnue"))
    ap.add_argument("--rounds",type=int,default=16)
    ap.add_argument("--threads",type=int,default=1)
    ap.add_argument("--concurrency",type=int,default=4)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--pgn",type=Path,help="also replay positions from saved time forfeits")
    args=ap.parse_args()
    if min(args.rounds,args.threads,args.concurrency)<1 or args.threads*args.concurrency>(os.cpu_count() or 1):
        ap.error("positive counts within available logical CPUs required")
    args.engine=args.engine.resolve(strict=True);args.network=args.network.resolve(strict=True)
    args.fens=list(FENS)
    if args.pgn:
        with args.pgn.open(encoding="utf-8") as file:
            while (game:=chess.pgn.read_game(file)) is not None:
                if game.headers.get("Termination")!="time forfeit":continue
                node=game.end();board=node.board()
                loser_white=game.headers["Result"]=="0-1"
                if board.turn!=loser_white and node.parent:board=node.parent.board()
                if not board.is_game_over():args.fens.append(board.fen())
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples=sum(pool.map(lambda i:worker(args,i),range(args.concurrency)),[])
    report=dict(engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest(),
                network_sha256=hashlib.sha256(args.network.read_bytes()).hexdigest(),
                threads=args.threads,concurrency=args.concurrency,
                count=len(samples),late_to_clock=sum(s["over_clock"] for s in samples),
                over_hard_plus_25ms=sum(s["over_hard_plus_25ms"] for s in samples),
                max_elapsed_ms=max(s["elapsed_ms"] for s in samples),samples=samples)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="samples"}))
    return int(bool(report["late_to_clock"]))


if __name__=="__main__":raise SystemExit(main())

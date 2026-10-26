#!/usr/bin/env python3
"""Stream Parquet shards of the Lichess/Stockfish evaluation dataset as TSV rows
`fen<TAB>depth<TAB>cp<TAB>mate` for examples/prepare_nnue.rs.

    python tools/nnue/parquet_to_tsv.py "part-*.parquet" | prepare_nnue --out data/nnue --min-depth 20 --require-complete

The cp column is white-perspective in this source; prepare_nnue converts it to the
side to move. Null cp/mate become empty fields. --stride N keeps every N-th row
(parts are sorted by FEN, so prefer a stride over taking only the first rows).
The final #zenith-tsv-complete marker lets the consumer detect producer failure.
"""
import argparse
import sys
import glob
import os
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--columns", default="fen,depth,cp,mate", help="names of the four columns")
    args = ap.parse_args()
    names = [name.strip() for name in args.columns.split(",")]
    if args.stride <= 0 or len(names) != 4 or len(set(names)) != 4 or not all(names):
        ap.error("stride must be positive; columns must give four distinct names in fen,depth,cp,mate order")
    files = []
    for pattern in args.files:
        matches = [pattern] if Path(pattern).is_file() else sorted(glob.glob(pattern))
        if not matches:
            ap.error(f"no files match {pattern}")
        files.extend(matches)
    files = sorted(set(str(Path(file).resolve()) for file in files))
    # Check every schema before emitting the first row.
    for file in files:
        if not set(names).issubset(pq.ParquetFile(file).schema_arrow.names):
            ap.error(f"{file}: missing required columns")
    out, counter, emitted = sys.stdout, 0, 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=names):
            cols = [batch.column(n).to_pylist() for n in names]
            lines = []
            for fen, depth, cp, mate in zip(*cols):
                counter += 1
                if counter % args.stride:
                    continue
                lines.append(f"{fen}\t{'' if depth is None else depth}\t"
                             f"{'' if cp is None else cp}\t{'' if mate is None else mate}\n")
                emitted += 1
            out.writelines(lines)
    out.write(f"#zenith-tsv-complete\t{emitted}\n")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, sys.stdout.fileno())
        os.close(sink)
        sys.exit(0)

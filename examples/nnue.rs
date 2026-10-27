//! Standalone NNUE arithmetic harness; no model is loaded into UCI automatically.
use std::{
    io::{self, BufRead},
    process::ExitCode,
    time::Instant,
};
use zenith::{
    board::Board,
    nnue::{Backend, Network, parse_training_fen},
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let scalar = args.iter().any(|s| s == "--scalar");
    let avx2 = args.iter().any(|s| s == "--avx2");
    if scalar && avx2 {
        return Err("choose either --scalar or --avx2".into());
    }
    args.retain(|s| s != "--scalar" && s != "--avx2");
    let usage = "usage: nnue inspect NETWORK | eval NETWORK [\"FEN\"] [MOVES...] | verify NETWORK < fens.txt | walk NETWORK [PLIES=10000] | bench NETWORK [POSITIONS=100000]; optional backend: --scalar | --avx2";
    let command = args.first().ok_or(usage)?;
    let mut net = Network::load(args.get(1).ok_or(usage)?)?;
    if scalar {
        net = net.with_backend(Backend::Scalar)?;
    }
    if avx2 {
        net = net.with_backend(Backend::Avx2)?;
    }
    eprintln!("NNUE backend: {:?}; accumulator lanes: i16", net.backend());
    match command.as_str() {
        "inspect" if args.len() == 2 => println!(
            "Zenith NNUE v{}: {} king banks x 768 -> 256 x 2 -> 1; CReLU; {} bytes; QA=255 QB=64 CP=400; validated (training provenance must be supplied separately)",
            net.format_version(),
            net.king_buckets(),
            net.file_bytes()
        ),
        "eval" => {
            let board = match args.get(2) {
                Some(fen) => parse_training_fen(fen)?,
                None => Board::startpos(),
            };
            let mut accumulator = net.accumulator(&board);
            print_position(&accumulator);
            for text in args.iter().skip(3) {
                let next = accumulator
                    .board()
                    .make_move(accumulator.board().parse_uci_move(text)?);
                accumulator.update(&next);
                verify(&net, &accumulator)?;
                print_position(&accumulator);
            }
        }
        "verify" if args.len() == 2 => {
            let mut count = 0;
            let mut a = net.accumulator(&Board::startpos());
            for (index, line) in io::stdin().lock().lines().enumerate() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() || line.starts_with('#') {
                    continue;
                }
                let board = parse_training_fen(line.trim())
                    .map_err(|e| format!("line {}: {e}", index + 1))?;
                a.update(&board);
                verify(&net, &a)?;
                print_position(&a);
                count += 1;
            }
            eprintln!("verified {count} positions: delta = refresh = dense");
        }
        "walk" if args.len() <= 3 => {
            let count = args
                .get(2)
                .map_or(Ok(10000), |s| s.parse::<usize>())
                .map_err(|_| usage)?;
            let mut seed = 239u64;
            let mut a = net.accumulator(&Board::startpos());
            let started = Instant::now();
            for ply in 0..count {
                let moves = a.board().legal_moves();
                if moves.is_empty() || ply % 256 == 255 {
                    a.update(&Board::startpos());
                } else {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let next = a
                        .board()
                        .make_move(moves[(seed >> 32) as usize % moves.len()]);
                    a.update(&next);
                }
                verify(&net, &a)?;
                print_position(&a);
            }
            eprintln!(
                "verified {count} walk positions in {:.3}s (includes movegen, dense reference and output; not inference benchmark)",
                started.elapsed().as_secs_f64()
            );
        }
        "bench" if args.len() <= 3 => {
            let count = args
                .get(2)
                .map_or(Ok(100000), |s| s.parse::<usize>())
                .map_err(|_| usage)?;
            if count == 0 {
                return Err("benchmark count must be positive".into());
            }
            let mut boards = Vec::with_capacity(1024);
            let mut board = Board::startpos();
            let mut seed = 239u64;
            for ply in 0..1024 {
                boards.push(board);
                let moves = board.legal_moves();
                if moves.is_empty() || ply % 256 == 255 {
                    board = Board::startpos();
                } else {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    board = board.make_move(moves[(seed >> 32) as usize % moves.len()]);
                }
            }
            let mut checksums = [0i64; 2];
            let mut elapsed = [std::time::Duration::ZERO; 2];
            for incremental in [false, true] {
                let mut accumulator = net.accumulator(&Board::startpos());
                let started = Instant::now();
                for i in 0..count {
                    let board = std::hint::black_box(&boards[i % boards.len()]);
                    let score = if incremental {
                        accumulator.update(board);
                        accumulator.evaluate()
                    } else {
                        net.accumulator(board).evaluate()
                    };
                    checksums[usize::from(incremental)] += i64::from(std::hint::black_box(score));
                }
                elapsed[usize::from(incremental)] = started.elapsed();
            }
            if checksums[0] != checksums[1] {
                return Err("benchmark checksum mismatch".into());
            }
            for (index, name) in ["refresh+head", "delta+head"].into_iter().enumerate() {
                println!(
                    "{name}: {count} positions, {:.3} ms, {:.1} ns/position, checksum {}",
                    elapsed[index].as_secs_f64() * 1000.,
                    elapsed[index].as_secs_f64() * 1e9 / count as f64,
                    checksums[index]
                );
            }
            println!(
                "{:?} microbenchmark; includes periodic FEN resets, excludes movegen and search",
                net.backend()
            );
        }
        _ => return Err(usage.into()),
    }
    Ok(())
}

fn verify(net: &Network, a: &zenith::nnue::Accumulator<'_>) -> Result<(), String> {
    let fresh = net.accumulator(a.board());
    for color in [zenith::board::Color::White, zenith::board::Color::Black] {
        if a.lanes(color) != fresh.lanes(color) {
            return Err(format!("accumulator mismatch: {}", a.board().to_fen()));
        }
    }
    if a.raw_score() != net.raw_reference(a.board()) {
        return Err(format!("dense mismatch: {}", a.board().to_fen()));
    }
    Ok(())
}

fn print_position(a: &zenith::nnue::Accumulator<'_>) {
    println!(
        "{}\t{}\t{}",
        a.board().to_fen(),
        a.raw_score(),
        a.evaluate()
    );
}

use std::process::ExitCode;

use zenith::board::{Board, perft_divide};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => zenith::uci::run().map_err(|error| error.to_string()),
        Some("bench") => {
            let mut depth = 6;
            let mut lmr = true;
            let mut see_ordering = true;
            let mut advanced_history = false;
            let mut null_pruning = zenith::search::DEFAULT_NULL_PRUNING;
            let mut pruning = true;
            let mut enhancements = true;
            let mut modern = zenith::search::DEFAULT_MODERN_SEARCH;
            let mut network = None;
            let mut context_histories = false;
            let mut arguments = args[1..].iter();
            while let Some(argument) = arguments.next() {
                if argument == "--context-history" {
                    context_histories = true;
                } else if argument == "--eval-file" {
                    let file = arguments
                        .next()
                        .ok_or("--eval-file requires a model path")?;
                    if network.is_some() {
                        return Err("--eval-file may only be specified once".into());
                    }
                    network = Some(std::sync::Arc::new(zenith::nnue::Network::load(file)?));
                } else if argument == "--no-lmr" {
                    lmr = false;
                } else if argument == "--no-see" {
                    see_ordering = false;
                } else if argument == "--no-history" {
                    advanced_history = false;
                } else if argument == "--history" {
                    advanced_history = true;
                } else if argument == "--no-nmp" {
                    null_pruning = false;
                } else if argument == "--nmp" {
                    null_pruning = true;
                } else if argument == "--no-pruning" {
                    pruning = false;
                } else if argument == "--no-enhancements" {
                    enhancements = false;
                } else if argument == "--modern" {
                    modern = true;
                    null_pruning = true;
                } else if argument == "--classic" {
                    modern = false;
                    null_pruning = false;
                } else {
                    depth = argument.parse::<u8>().map_err(
                        |_| "usage: zenith bench [DEPTH] [--eval-file PATH] [--no-lmr] [--no-see] [--history] [--context-history] [--nmp|--no-nmp] [--no-pruning] [--no-enhancements] [--modern|--classic]",
                    )?;
                }
            }
            if !(1..=120).contains(&depth) {
                return Err("depth must be from 1 to 120".into());
            }
            zenith::bench::run(
                depth,
                lmr,
                see_ordering,
                advanced_history,
                null_pruning,
                pruning,
                enhancements,
                modern,
                network,
                context_histories,
            );
            Ok(())
        }
        Some("perft") => {
            let depth = args
                .get(1)
                .ok_or("usage: zenith perft DEPTH [\"FEN\"]")?
                .parse::<u8>()
                .map_err(|_| "depth must be an integer from 0 to 255")?;
            let board = if args.len() > 2 {
                Board::from_fen(&args[2..].join(" "))?
            } else {
                Board::startpos()
            };
            let started = std::time::Instant::now();
            let total = if depth == 0 {
                1
            } else {
                let rows = perft_divide(&board, depth);
                for (mv, nodes) in &rows {
                    println!("{mv}: {nodes}");
                }
                rows.iter().map(|(_, nodes)| nodes).sum::<u64>()
            };
            let elapsed = started.elapsed();
            println!("nodes {total}");
            println!("time_ms {}", elapsed.as_millis());
            println!(
                "nps {}",
                (total as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64
            );
            Ok(())
        }
        Some("--help" | "-h") => {
            println!(
                "Zenith {}\n\nzenith                     Run UCI on stdin/stdout\nzenith perft DEPTH [\"FEN\"]  Count legal move trees\nzenith bench [DEPTH] [--eval-file PATH] [--no-lmr] [--no-see] [--history] [--context-history] [--nmp|--no-nmp] [--no-pruning] [--no-enhancements] [--modern|--classic]  Run fixed search benchmark\nzenith --version           Print version",
                env!("CARGO_PKG_VERSION")
            );
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("Zenith {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(argument) => Err(format!("unknown argument: {argument}; run zenith --help")),
    }
}

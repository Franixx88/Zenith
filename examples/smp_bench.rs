//! Timed Lazy SMP work/scaling check, not an Elo measurement.
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};
use zenith::{
    bench::POSITIONS,
    board::Board,
    search::{Limits, Search},
};
fn main() {
    let millis = std::env::args()
        .nth(1)
        .map_or(1000, |v| v.parse::<u64>().expect("milliseconds"));
    let requested: Vec<usize> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "1,2,4".into())
        .split(',')
        .map(|s| s.parse::<usize>().expect("thread count"))
        .collect();
    assert!(!requested.is_empty() && requested.iter().all(|t| (1..=64).contains(t)));
    println!("round,threads,position,nodes,helper_nodes,depth,elapsed_us,nps,bestmove,score");
    for round in 0..3 {
        // Rotate the order to reduce temperature/order bias.
        let mut order = requested.clone();
        order.rotate_right(round % requested.len());
        for threads in order {
            let mut search = Search::new(16);
            search.set_threads(threads);
            for (name, fen) in POSITIONS {
                search.clear();
                let board = Board::from_fen(fen).unwrap();
                let start = Instant::now();
                let result = search.search(
                    board,
                    &[board.hash()],
                    Limits {
                        movetime: Some(Duration::from_millis(millis)),
                        ..Limits::default()
                    },
                    Arc::new(AtomicBool::new(false)),
                    |_| {},
                );
                let elapsed = start.elapsed();
                assert_eq!(result.stats.helper_failures, 0);
                let mut pos = board;
                for &mv in &result.pv {
                    assert!(pos.legal_moves().contains(&mv));
                    pos = pos.make_move(mv);
                }
                println!(
                    "{round},{threads},{name},{},{},{},{},{},{},{}",
                    result.nodes,
                    result.stats.helper_nodes,
                    result.depth,
                    elapsed.as_micros(),
                    (result.nodes as f64 / elapsed.as_secs_f64()) as u64,
                    result.best_move.unwrap(),
                    result.score
                );
            }
        }
    }
}

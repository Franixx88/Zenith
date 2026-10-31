//! Fixed, reproducible search workload. Counts measure work, not playing Elo.
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

use crate::{
    board::Board,
    search::{Limits, Search},
};

pub const POSITIONS: [(&str, &str); 5] = [
    (
        "start",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    (
        "tactical",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ),
    (
        "middlegame",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ),
];

#[allow(clippy::too_many_arguments)] // Independent switches for reproducible ablations.
pub fn run(
    depth: u8,
    lmr: bool,
    see_ordering: bool,
    advanced_history: bool,
    null_pruning: bool,
    pruning: bool,
    enhancements: bool,
    modern: bool,
    network: Option<Arc<crate::nnue::Network>>,
    context_histories: bool,
) {
    let mut search = Search::new(16);
    search.set_lmr(lmr);
    search.set_see_ordering(see_ordering);
    search.set_advanced_history(advanced_history);
    search.set_null_pruning(null_pruning);
    search.set_pruning(pruning);
    search.set_enhancements(enhancements);
    search.set_modern_search(modern);
    search.set_context_histories(context_histories);
    let evaluation = network.as_ref().map_or_else(
        || "classical".to_owned(),
        |model| {
            format!(
                "nnue-v{} backend={:?}",
                model.format_version(),
                model.backend()
            )
        },
    );
    search.set_network(network);
    let mut total_nodes = 0;
    let mut total_time = std::time::Duration::ZERO;
    println!(
        "bench version={} depth={depth} hash_mib=16 evaluation={evaluation} lmr={lmr} see_ordering={see_ordering} advanced_history={advanced_history} null_pruning={null_pruning} pruning={pruning} enhancements={enhancements} modern={modern} context_histories={context_histories}",
        env!("CARGO_PKG_VERSION")
    );
    for (name, fen) in POSITIONS {
        search.clear();
        let board = Board::from_fen(fen).expect("benchmark FEN");
        let start = Instant::now();
        let result = search.search(
            board,
            &[board.hash()],
            Limits {
                depth: Some(depth),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        let elapsed = start.elapsed();
        println!(
            "nnue refreshes={} updates={} bucket_refreshes={} evaluations={}",
            result.stats.nnue_refreshes,
            result.stats.nnue_updates,
            result.stats.nnue_bucket_refreshes,
            result.stats.nnue_evaluations,
        );
        println!(
            "cache hits={} misses={} probcut probes={} cutoffs={}",
            result.stats.nnue_cache_hits,
            result.stats.nnue_cache_misses,
            result.stats.probcut_probes,
            result.stats.probcut_cutoffs
        );
        println!(
            "singular probes={} extensions={} doubles={} verification_nodes={}",
            result.stats.singular_probes,
            result.stats.singular_extensions,
            result.stats.singular_double_extensions,
            result.stats.singular_verification_nodes
        );
        println!(
            "modern lmp={} futility={} history={} quiet_see={} iir={} corrections={}",
            result.stats.lmp_pruned,
            result.stats.futility_pruned,
            result.stats.history_pruned,
            result.stats.quiet_see_pruned,
            result.stats.iir,
            result.stats.correction_updates
        );
        println!(
            "tt probes={} keys={} contexts={} cutoffs={} collisions={} replaced={} extensions={} lmr={} researches={}",
            result.stats.tt_probes,
            result.stats.tt_key_hits,
            result.stats.tt_context_hits,
            result.stats.tt_cutoffs,
            result.stats.tt_collisions,
            result.stats.tt_replacements,
            result.stats.extensions,
            result.stats.reductions,
            result.stats.reduction_researches
        );
        total_time += elapsed;
        total_nodes += result.nodes;
        println!(
            "position={name} depth={} score={} bestmove={} nodes={} time_us={} null_probes={} null_cutoffs={} qnodes={} q_pruned={} rfp={} eval_hits={}/{}",
            result.depth,
            result.score,
            result
                .best_move
                .map_or_else(|| "0000".into(), |mv| mv.to_uci()),
            result.nodes,
            elapsed.as_micros(),
            result.null_probes,
            result.null_cutoffs,
            result.stats.qnodes,
            result.stats.q_pruned,
            result.stats.reverse_cutoffs,
            result.stats.eval_hits,
            result.stats.eval_calls
        );
    }
    println!(
        "total nodes={total_nodes} time_us={} nps={}",
        total_time.as_micros(),
        (total_nodes as f64 / total_time.as_secs_f64().max(0.000_001)) as u64
    );
}

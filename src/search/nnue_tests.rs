//! Search-level regressions use synthetic weights, not a playing network.
use super::*;
use crate::{
    board::Color,
    nnue::{self, Backend, Parameters},
};

fn model() -> Network {
    let mut p = Parameters::default();
    p.feature_bias.fill(128);
    for perspective in [Color::White, Color::Black] {
        for color in [Color::White, Color::Black] {
            for piece in [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                for square in 0..64 {
                    let index = nnue::feature_index(perspective, color, piece, square);
                    p.feature_weights[index * nnue::HIDDEN + piece.index()] =
                        if color == perspective { 8 } else { -8 };
                }
            }
        }
    }
    for (piece, value) in [100, 320, 335, 500, 950, 0].into_iter().enumerate() {
        p.output_weights[piece] = value * 4;
        p.output_weights[nnue::HIDDEN + piece] = -value * 4;
    }
    // Spatial features are essential: material-only fixtures would fail to
    // detect a stale accumulator after an ordinary non-capturing move.
    for feature in 0..nnue::INPUTS {
        for lane in 6..22 {
            p.feature_weights[feature * nnue::HIDDEN + lane] =
                (((feature * 17 + lane * 31) ^ (feature >> 2)) % 9) as i16 - 4;
        }
    }
    for lane in 6..22 {
        p.output_weights[lane] = ((lane * 13) % 97) as i16 - 48;
        p.output_weights[nnue::HIDDEN + lane] = ((lane * 17) % 89) as i16 - 44;
    }
    Network::new(p).unwrap()
}

fn constant(score: i32) -> Arc<Network> {
    Arc::new(
        Network::new(Parameters {
            output_bias: score * (nnue::QA * nnue::QB) / nnue::CP_SCALE,
            ..Parameters::default()
        })
        .unwrap(),
    )
}

fn run(search: &mut Search, board: Board, depth: u8, budget: u64) -> SearchResult {
    search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(depth),
            nodes: Some(budget),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    )
}

#[test]
fn incremental_and_full_refresh_search_have_identical_scores_pvs_and_work() {
    let network = Arc::new(model());
    for (index, (_, fen)) in crate::bench::POSITIONS.into_iter().enumerate() {
        let board = Board::from_fen(fen).unwrap();
        let mut delta = Search::new(1);
        delta.set_network(Some(network.clone()));
        let mut refresh = Search::new(1);
        refresh.set_network(Some(network.clone()));
        refresh.reference_nnue = true;
        if index == 0 {
            delta.set_multi_pv(3);
            refresh.set_multi_pv(3);
        }
        let a = run(&mut delta, board, 5, 12_000);
        let b = run(&mut refresh, board, 5, 12_000);
        assert_eq!(
            (a.score, a.best_move, a.depth, a.nodes, &a.pv),
            (b.score, b.best_move, b.depth, b.nodes, &b.pv),
            "{fen}"
        );
        assert_eq!(
            a.variations
                .iter()
                .map(|v| (v.score, &v.pv))
                .collect::<Vec<_>>(),
            b.variations
                .iter()
                .map(|v| (v.score, &v.pv))
                .collect::<Vec<_>>()
        );
        assert_eq!(a.stats.nnue_refreshes, 1);
        assert!(a.stats.nnue_evaluations > 1 && a.stats.nnue_updates > 0);
        assert!(b.stats.nnue_refreshes > a.stats.nnue_refreshes);
    }
}

#[test]
fn null_exclusion_and_cancelled_branches_leave_parent_accumulators_exact() {
    let board = Board::startpos();
    let network = model();
    for budget in [None, Some(1)] {
        let mut search = Search::new(1);
        let limits = Limits {
            nodes: budget,
            ..Limits::default()
        };
        let mut ctx = super::tests::null_context(&mut search, &limits, board);
        ctx.nnue = Some(NnueStack::new(&network, &board));
        let beta = network.evaluate_reference(&board);
        let result = ctx.node(
            &board,
            5,
            1,
            beta - 1,
            beta,
            NodeState {
                history: 0,
                extensions_left: 0,
            },
        );
        assert_eq!(result.is_some(), budget.is_none());
        assert!(ctx.null_probes > 0);
        assert!(!ctx.null_active && !ctx.pruning_suppressed);
        assert_eq!(
            ctx.raw_evaluate(&board, 1),
            network.evaluate_reference(&board)
        );
    }
    let mut search = Search::new(1);
    let limits = Limits::default();
    let mut ctx = super::tests::null_context(&mut search, &limits, board);
    ctx.nnue = Some(NnueStack::new(&network, &board));
    let excluded = board.parse_uci_move("e2e4").unwrap();
    assert!(
        ctx.verify_excluding(
            &board,
            3,
            1,
            100,
            NodeState {
                history: 0,
                extensions_left: 0
            },
            excluded
        )
        .is_some()
    );
    assert_eq!(
        ctx.raw_evaluate(&board, 1),
        network.evaluate_reference(&board)
    );
    assert!(ctx.excluded[1].is_none());
}

#[test]
fn model_switches_clear_caches_and_smp_uses_only_current_weights() {
    let board = Board::startpos();
    let mut search = Search::new(1);
    let first = constant(100);
    search.set_network(Some(first.clone()));
    let _ = run(&mut search, board, 3, 10_000);
    assert!(search.table.iter().any(|e| e.valid));
    search.set_network(Some(constant(-200)));
    assert!(search.table.iter().all(|e| !e.valid));
    assert!(search.eval_cache.iter().all(|e| !e.valid));
    assert!(search.correction.iter().all(|&v| v == 0));
    assert_eq!(run(&mut search, board, 3, 0).score, -200);
    search.set_network(None);
    assert_eq!(run(&mut search, board, 3, 0).score, evaluate(&board));
    search.set_threads(8);
    search.set_network(Some(first.clone()));
    let result = run(&mut search, board, 4, 4000);
    assert!(result.nodes <= 4000 && result.stats.helper_failures == 0);
    assert!(result.stats.nnue_refreshes >= 1);
    assert!(
        search
            .workers
            .iter()
            .all(|w| Arc::ptr_eq(w.network.as_ref().unwrap(), &first))
    );
    search.set_network(Some(constant(200)));
    assert!(search.workers.is_empty());
    assert_eq!(run(&mut search, board, 3, 0).score, 200);
    search.clear();
    assert_eq!(run(&mut search, board, 3, 0).score, 200);
}

#[test]
fn neural_scores_do_not_override_mates_draws_or_legal_cancelled_fallbacks() {
    let mut search = Search::new(1);
    search.set_network(Some(constant(28_000)));
    for (fen, expected) in [
        ("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1", -MATE_SCORE),
        ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", 0),
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", 0),
        ("7k/8/8/8/8/8/P7/K7 w - - 100 1", 0),
    ] {
        assert_eq!(
            run(&mut search, Board::from_fen(fen).unwrap(), 2, 1000).score,
            expected
        );
    }
    let board = Board::startpos();
    let result = search.search(
        board,
        &[board.hash()],
        Limits::default(),
        Arc::new(AtomicBool::new(true)),
        |_| {},
    );
    assert_eq!(result.score, 28_000);
    assert!(board.legal_moves().contains(&result.best_move.unwrap()));
}

#[test]
fn scalar_and_selected_simd_backend_search_the_same_tree() {
    let network = model();
    let scalar = Arc::new(
        Network::from_bytes(&network.to_bytes())
            .unwrap()
            .with_backend(Backend::Scalar)
            .unwrap(),
    );
    let mut a = Search::new(1);
    let mut b = Search::new(1);
    a.set_network(Some(Arc::new(network)));
    b.set_network(Some(scalar));
    let board = Board::from_fen(crate::bench::POSITIONS[1].1).unwrap();
    let a = run(&mut a, board, 4, 4000);
    let b = run(&mut b, board, 4, 4000);
    assert_eq!(
        (a.score, a.best_move, a.nodes, a.pv),
        (b.score, b.best_move, b.nodes, b.pv)
    );
}

#[test]
fn lazy_frames_handle_special_moves_skipped_parents_and_sibling_reuse() {
    let network = model();
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
        "7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1",
        "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/p7/1R5K b - - 0 1",
    ] {
        let root = Board::from_fen(fen).unwrap();
        let mut stack = NnueStack::new(&network, &root);
        for mv in root.legal_moves() {
            let child = root.make_move(mv);
            stack.enter(&child, 1);
            // The parent may never ask for NNUE (TT hit / in check). A child
            // evaluation must still reconstruct every skipped delta correctly.
            for reply in child.legal_moves().into_iter().take(3) {
                let next = child.make_move(reply);
                stack.enter(&next, 2);
                assert_eq!(stack.evaluate(2), network.evaluate_reference(&next));
            }
            assert_eq!(stack.evaluate(1), network.evaluate_reference(&child));
            if let Some(null) = child.null_move() {
                stack.enter(&null, 2);
                assert_eq!(stack.evaluate(2), network.evaluate_reference(&null));
            }
            assert_eq!(stack.evaluate(0), network.evaluate_reference(&root));
        }
    }
}

#[test]
fn v3_lazy_frames_match_refresh_tree_with_context_histories_and_mirror_changes() {
    let mut p = nnue::v3::Parameters::default();
    p.feature_bias.fill(128);
    for (i, w) in p.feature_weights.iter_mut().enumerate() {
        *w = (((i * 17) ^ (i >> 9)) % 17) as i16 - 8;
    }
    for h in &mut p.heads {
        for (i, w) in h.l1_weights.iter_mut().enumerate() {
            *w = (i % 19) as i16 - 9;
        }
        h.l1_bias.fill(6000);
        h.l2_weights[0] = 64;
        h.output_weights[0] = 64;
    }
    let network = Arc::new(Network::new_v3(p).unwrap());
    for fen in [
        crate::bench::POSITIONS[1].1,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        let mut a = Search::new(1);
        let mut b = Search::new(1);
        for search in [&mut a, &mut b] {
            search.set_network(Some(network.clone()));
            search.set_context_histories(true);
        }
        b.reference_nnue = true;
        let a = run(&mut a, board, 4, 4000);
        let b = run(&mut b, board, 4, 4000);
        assert_eq!(
            (a.score, a.best_move, a.nodes, a.pv),
            (b.score, b.best_move, b.nodes, b.pv)
        );
    }
    let root = Board::from_fen("7k/8/8/8/8/8/8/3K4 w - - 0 1").unwrap();
    let child = root.make_move(root.parse_uci_move("d1e1").unwrap());
    let mut stack = NnueStack::new(&network, &root);
    stack.enter(&child, 1);
    if let Some(null) = child.null_move() {
        stack.enter(&null, 2);
        assert_eq!(stack.evaluate(2), network.evaluate_reference(&null));
    }
    assert_eq!(stack.evaluate(1), network.evaluate_reference(&child));
    assert_eq!(stack.evaluate(0), network.evaluate_reference(&root));
}

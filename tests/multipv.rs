use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::AtomicBool},
};
use zenith::{
    board::Board,
    search::{Limits, MATE_SCORE, Search, SearchResult},
};

fn run(search: &mut Search, board: Board, limits: Limits) -> SearchResult {
    search.search(
        board,
        &[board.hash()],
        limits,
        Arc::new(AtomicBool::new(false)),
        |_| {},
    )
}
fn legal_lines(board: Board, result: &SearchResult, count: usize) {
    assert_eq!(result.variations.len(), count);
    let mut roots = HashSet::new();
    for line in &result.variations {
        assert!(roots.insert(line.pv[0]));
        let mut pos = board;
        for &mv in &line.pv {
            assert!(pos.legal_moves().contains(&mv), "{mv} in {}", pos.to_fen());
            pos = pos.make_move(mv);
        }
    }
    assert!(
        result
            .variations
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score)
    );
    assert_eq!(result.best_move, result.variations[0].pv.first().copied());
    assert_eq!(result.score, result.variations[0].score);
    assert_eq!(result.pv, result.variations[0].pv);
}

fn reference() -> Search {
    let mut search = Search::new(1);
    search.set_modern_search(false);
    search.set_null_pruning(false);
    search.set_pruning(false);
    search.set_lmr(false);
    search.set_enhancements(false);
    search
}

#[test]
fn top_lines_match_separate_full_window_root_searches() {
    let board = Board::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
    let mut scores = HashMap::new();
    for mv in board.legal_moves() {
        let result = run(
            &mut reference(),
            board,
            Limits {
                depth: Some(3),
                searchmoves: Some(vec![mv]),
                ..Limits::default()
            },
        );
        scores.insert(mv, result.score);
    }
    let mut ranked: Vec<_> = scores.values().copied().collect();
    ranked.sort_unstable_by_key(|score| std::cmp::Reverse(*score));
    let mut search = reference();
    search.set_multi_pv(3);
    let mut infos = vec![];
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(3),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |info| infos.push(info),
    );
    legal_lines(board, &result, 3);
    assert_eq!(result.depth, 3);
    for (index, line) in result.variations.iter().enumerate() {
        assert_eq!(line.score, scores[&line.pv[0]]);
        assert_eq!(line.score, ranked[index]);
    }
    for batch in infos.chunks(3) {
        assert_eq!(batch.len(), 3);
        for (rank, info) in batch.iter().enumerate() {
            assert_eq!(info.multipv, rank + 1);
            assert_eq!(info.depth, batch[0].depth);
            assert_eq!(info.nodes, batch[0].nodes);
        }
    }
}

#[test]
fn budgets_restrictions_and_mode_changes_do_not_publish_partial_batches() {
    let board = Board::startpos();
    let allowed: Vec<_> = ["e2e4", "d2d4", "g1f3"]
        .iter()
        .map(|m| board.parse_uci_move(m).unwrap())
        .collect();
    let mut search = Search::new(2);
    search.set_smp_voting(true);
    for threads in [1, 8, 2, 1] {
        search.set_threads(threads);
        search.set_multi_pv(8);
        for budget in [0, 1, 100, 10_000] {
            let result = run(
                &mut search,
                board,
                Limits {
                    nodes: Some(budget),
                    searchmoves: Some(allowed.clone()),
                    ..Limits::default()
                },
            );
            assert_eq!(result.nodes, budget);
            assert!(allowed.contains(&result.best_move.unwrap()));
            assert_eq!(result.stats.helper_failures, 0);
            assert_eq!(result.stats.smp_selected_worker, 0);
            if result.depth > 0 {
                legal_lines(board, &result, 3);
            } else {
                assert!(result.variations.is_empty());
            }
            for line in result.variations {
                assert!(allowed.contains(&line.pv[0]));
            }
        }
    }
    search.set_multi_pv(1);
    search.set_threads(1);
    search.set_smp_voting(false);
    let limits = Limits {
        depth: Some(5),
        ..Limits::default()
    };
    let reused = run(&mut search, board, limits.clone());
    let fresh = run(&mut Search::new(2), board, limits);
    assert!(reused.variations.is_empty());
    assert_eq!(
        (reused.nodes, reused.score, reused.pv),
        (fresh.nodes, fresh.score, fresh.pv)
    );
}

#[test]
fn mates_forced_moves_draws_and_empty_roots_remain_exact() {
    let mut search = Search::new(1);
    search.set_multi_pv(4);
    let mate = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let result = run(
        &mut search,
        mate,
        Limits {
            depth: Some(4),
            ..Limits::default()
        },
    );
    legal_lines(mate, &result, 4);
    assert_eq!(result.score, MATE_SCORE - 1);
    let forced = Board::from_fen("r6k/8/8/8/8/8/1P6/KB6 w - - 0 1").unwrap();
    assert_eq!(forced.legal_moves().len(), 1);
    legal_lines(
        forced,
        &run(
            &mut search,
            forced,
            Limits {
                depth: Some(3),
                ..Limits::default()
            },
        ),
        1,
    );
    let draw = Board::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let result = run(&mut search, draw, Limits::default());
    legal_lines(draw, &result, draw.legal_moves().len().min(4));
    assert!(result.variations.iter().all(|line| line.score == 0));
    let empty = run(
        &mut search,
        Board::startpos(),
        Limits {
            searchmoves: Some(vec![]),
            ..Limits::default()
        },
    );
    assert!(empty.best_move.is_none() && empty.variations.is_empty());
}

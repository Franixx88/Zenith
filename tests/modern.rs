use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::Board,
    search::{Limits, MATE_SCORE, Search},
};

#[test]
fn modern_search_preserves_exact_mates_and_restricted_roots() {
    for fen in [
        "7k/8/5K2/8/8/8/8/3Q4 w - - 0 1",
        "k7/8/2K5/8/8/8/8/3R4 w - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        let mut search = Search::new(1);
        search.set_modern_search(true);
        let result = search.search(
            board,
            &[],
            Limits {
                depth: Some(4),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.score, MATE_SCORE - 3);
        let mut next = board;
        for mv in result.pv {
            assert!(next.legal_moves().contains(&mv));
            next = next.make_move(mv);
        }
    }
    let board = Board::startpos();
    let mv = board.parse_uci_move("a2a3").unwrap();
    let mut search = Search::new(1);
    search.set_modern_search(true);
    let result = search.search(
        board,
        &[],
        Limits {
            depth: Some(5),
            nodes: Some(20000),
            searchmoves: Some(vec![mv]),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.best_move, Some(mv));
    assert!(result.nodes <= 20000);
}

#[test]
fn modern_group_is_active_switchable_and_keeps_legal_pv() {
    let board = Board::startpos();
    let mut search = Search::new(1);
    search.set_modern_search(true);
    search.set_null_pruning(true);
    let result = search.search(
        board,
        &[],
        Limits {
            depth: Some(6),
            nodes: Some(150000),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert!(
        result.stats.lmp_pruned + result.stats.futility_pruned + result.stats.quiet_see_pruned > 0
    );
    assert!(result.stats.correction_updates > 0);
    assert!(result.null_probes > 0);
    let mut next = board;
    for mv in result.pv {
        assert!(next.legal_moves().contains(&mv));
        next = next.make_move(mv);
    }
    search.set_modern_search(false);
    search.set_null_pruning(false);
    let old = search.search(
        board,
        &[],
        Limits {
            depth: Some(4),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(
        old.stats.lmp_pruned
            + old.stats.futility_pruned
            + old.stats.quiet_see_pruned
            + old.stats.iir
            + old.stats.correction_updates,
        0
    );
    assert_eq!(old.null_probes, 0);
}

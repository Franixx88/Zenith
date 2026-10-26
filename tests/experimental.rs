use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::Board,
    search::{Limits, Search, SearchParameters},
};

#[test]
fn singular_activates_and_experimental_search_preserves_shared_limits_and_legal_pv() {
    let board = Board::startpos();
    let mut search = Search::new(4);
    search.set_singular_extensions(true);
    search.set_smp_voting(true);
    let deep = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(9),
            nodes: Some(300_000),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert!(deep.stats.singular_probes > 0);
    assert!(deep.stats.singular_extensions > 0);
    for threads in [1, 8, 2, 1] {
        search.set_threads(threads);
        let mv = board.parse_uci_move("a2a3").unwrap();
        let result = search.search(
            board,
            &[board.hash()],
            Limits {
                nodes: Some(10_000),
                searchmoves: Some(vec![mv]),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.nodes, 10_000);
        assert_eq!(result.best_move, Some(mv));
        assert_eq!(result.stats.helper_failures, 0);
        assert!(result.stats.smp_selected_worker < threads);
        let mut position = board;
        for mv in result.pv {
            assert!(position.legal_moves().contains(&mv));
            position = position.make_move(mv);
        }
    }
}

#[test]
fn changing_parameters_invalidates_scores_but_rejected_parameters_leave_search_unchanged() {
    let mut search = Search::new(1);
    let board = Board::startpos();
    let run = |search: &mut Search| {
        search.search(
            board,
            &[board.hash()],
            Limits {
                depth: Some(5),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
    };
    run(&mut search);
    let params = SearchParameters {
        lmr_divisor: 250,
        ..SearchParameters::default()
    };
    search.set_parameters(params).unwrap();
    assert!(
        search
            .set_parameters(SearchParameters {
                lmr_divisor: 0,
                ..params
            })
            .is_err()
    );
    let actual = run(&mut search);
    let mut fresh = Search::new(1);
    fresh.set_parameters(params).unwrap();
    let expected = run(&mut fresh);
    assert_eq!(
        (actual.nodes, actual.score, actual.pv),
        (expected.nodes, expected.score, expected.pv)
    );
}

use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;
use zenith::{
    board::Board,
    search::{Limits, MATE_SCORE, Search},
};

fn legal_pv(mut board: Board, pv: &[zenith::board::Move]) {
    for &mv in pv {
        assert!(
            board.legal_moves().contains(&mv),
            "illegal PV move {mv} in {}",
            board.to_fen()
        );
        board = board.make_move(mv);
    }
}

#[test]
fn shared_node_budget_repeated_search_and_thread_changes() {
    let board = Board::startpos();
    let mut search = Search::new(2);
    for threads in [2, 8, 4, 1, 3, 1] {
        search.set_threads(threads);
        for budget in [0, 1, 100, 5000] {
            let result = search.search(
                board,
                &[board.hash()],
                Limits {
                    nodes: Some(budget),
                    ..Limits::default()
                },
                Arc::new(AtomicBool::new(false)),
                |info| {
                    assert!(info.nodes <= budget);
                },
            );
            assert_eq!(result.nodes, budget);
            assert_eq!(result.stats.helper_failures, 0);
            assert!(result.stats.helper_nodes <= result.nodes);
            assert!(board.legal_moves().contains(&result.best_move.unwrap()));
            legal_pv(board, &result.pv);
        }
    }
}

#[test]
fn parallel_mate_and_restricted_roots() {
    let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let mate = board.parse_uci_move("f7g7").unwrap();
    let mut search = Search::new(2);
    search.set_threads(4);
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(4),
            searchmoves: Some(vec![mate]),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.best_move, Some(mate));
    assert_eq!(result.score, MATE_SCORE - 1);
    legal_pv(board, &result.pv);
}

#[test]
fn cancellation_joins_helpers_and_keeps_state_reusable() {
    let mut search = Search::new(1);
    search.set_threads(4);
    let board = Board::startpos();
    let stop = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        let flag = stop.clone();
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let start = std::time::Instant::now();
        let result = search.search(board, &[board.hash()], Limits::default(), stop, |_| {});
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_eq!(result.stats.helper_failures, 0);
        legal_pv(board, &result.pv);
    });
    search.clear();
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            nodes: Some(1000),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.nodes, 1000);
}

#[test]
fn callback_unwind_cancels_and_joins_scoped_helpers() {
    let board = Board::startpos();
    let mut search = Search::new(1);
    search.set_threads(4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        search.search(
            board,
            &[board.hash()],
            Limits::default(),
            Arc::new(AtomicBool::new(false)),
            |_| panic!("test callback failure"),
        );
    }));
    assert!(result.is_err());
    search.set_threads(1);
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            nodes: Some(1000),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.nodes, 1000);
}

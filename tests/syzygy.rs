//! Run with: cargo test --release --test syzygy -- --ignored
//! First run tools/download_syzygy_test.ps1; override ZENITH_SYZYGY_PATH as needed.
use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::Board,
    search::{Limits, Search, is_mate_score},
    syzygy::Syzygy,
};

fn searcher() -> Search {
    let path = std::env::var("ZENITH_SYZYGY_PATH").unwrap_or_else(|_| "target/syzygy-tests".into());
    let tables = Syzygy::load(&path).expect("download the tiny test tables first");
    assert_eq!(tables.as_ref().unwrap().max_pieces(), 3);
    let mut search = Search::new(1);
    search.set_syzygy(tables, 7, 1);
    search
}
fn run(search: &mut Search, fen: &str) -> zenith::search::SearchResult {
    let board = Board::from_fen(fen).unwrap();
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(3),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    if let Some(mv) = result.best_move {
        assert!(board.legal_moves().contains(&mv));
    }
    result
}

#[test]
#[ignore = "requires tiny downloaded Syzygy files"]
fn real_wdl_dtz_rule50_and_promotion() {
    let mut search = searcher();
    for fen in [
        "7k/8/6K1/8/8/8/Q7/8 w - - 0 1",
        "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
        "8/k1P5/2K5/8/8/8/8/8 w - - 0 1",
        "8/k1P5/2K5/8/8/8/8/8 w - - 75 1",
    ] {
        let result = run(&mut search, fen);
        assert_eq!(result.stats.tb_hits, 1, "{fen}: {result:?}");
        assert_eq!(result.stats.tb_errors, 0);
        assert_eq!(result.score, 29000);
        assert_eq!(result.depth, 0);
        assert!(!is_mate_score(result.score));
    }
    // A rook cannot mate from here before the hundredth reversible ply.
    let result = run(&mut search, "7k/8/8/8/8/2K5/8/R7 w - - 99 1");
    assert_eq!(result.score, 0);
    assert_eq!(result.stats.tb_hits, 1);
    // Mate overrides the 50-move claim, including in the tablebase shortcut.
    let result = run(&mut search, "7k/5Q2/6K1/8/8/8/8/8 w - - 99 1");
    assert!(result.score > 28000);
    let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 99 1").unwrap();
    let child = board.make_move(result.best_move.unwrap());
    assert!(child.in_check(child.side_to_move) && child.legal_moves().is_empty());
}

#[test]
#[ignore = "requires tiny downloaded Syzygy files"]
fn root_repetition_restrictions_and_disabled_probes() {
    let board = Board::from_fen("7k/8/6K1/8/8/8/Q7/8 w - - 8 1").unwrap();
    let mut search = searcher();
    search.set_threads(2);
    let result = search.search(
        board,
        &[board.hash(), board.hash(), board.hash()],
        Limits {
            depth: Some(3),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.score, 0);
    assert_eq!(result.stats.tb_hits, 0);
    let only = board.parse_uci_move("a2b2").unwrap();
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(2),
            searchmoves: Some(vec![only]),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.best_move, Some(only));
    // Duplicate reversible history must disable the DTZ root shortcut.
    let result = search.search(
        board,
        &[board.hash(), board.hash()],
        Limits {
            depth: Some(2),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert!(result.depth > 0);
    search.set_syzygy(None, 0, 1);
    let result = run(&mut search, &board.to_fen());
    assert_eq!(result.stats.tb_probes, 0);
}

#[test]
#[ignore = "requires tiny downloaded Syzygy files"]
fn internal_zeroing_wdl_and_lazy_corruption_fallback() {
    let mut search = searcher();
    let board = Board::from_fen("7k/8/8/8/p7/2K5/2Q5/8 w - - 0 1").unwrap();
    let capture = board.parse_uci_move("c2a4").unwrap();
    let result = search.search(
        board,
        &[board.hash()],
        Limits {
            depth: Some(3),
            searchmoves: Some(vec![capture]),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert_eq!(result.best_move, Some(capture));
    assert!(result.stats.tb_hits > 0);
    assert_eq!(result.stats.tb_errors, 0);
    assert!(result.score >= 28800 && !is_mate_score(result.score));
    // Registration validates file size; an invalid header is detected lazily.
    let dir = std::env::temp_dir().join(format!(
        "zenith-corrupt-tb-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let file = dir.join("KQvK.rtbw");
    std::fs::write(&file, [0u8; 272]).unwrap();
    let tables = Syzygy::load(dir.to_str().unwrap()).unwrap();
    search.set_syzygy(tables, 7, 1);
    let result = run(&mut search, "7k/8/6K1/8/8/8/Q7/8 w - - 0 1");
    assert!(result.stats.tb_errors > 0);
    assert!(result.depth > 0);
    drop(search);
    std::fs::remove_file(file).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
#[ignore = "requires tiny downloaded Syzygy files"]
fn tablebase_mainlines_finish_without_repetition_or_rule50_draws() {
    for fen in [
        "7k/8/8/8/8/2K5/8/R7 w - - 0 1",
        "7k/8/8/8/8/2K5/Q7/8 w - - 0 1",
        "8/k1P5/2K5/8/8/8/8/8 w - - 0 1",
    ] {
        let mut search = searcher();
        search.set_threads(2);
        let mut board = Board::from_fen(fen).unwrap();
        let mut history = vec![board.hash()];
        let mut mated = false;
        for _ in 0..100 {
            if !board.legal_moves().is_empty() {
                assert!(board.halfmove_clock < 100);
                assert!(history.iter().filter(|&&key| key == board.hash()).count() < 3);
                let result = search.search(
                    board,
                    &history,
                    Limits {
                        depth: Some(3),
                        ..Limits::default()
                    },
                    Arc::new(AtomicBool::new(false)),
                    |_| {},
                );
                assert_eq!(result.stats.tb_errors, 0);
                board = board.make_move(result.best_move.unwrap());
                history.push(board.hash());
            } else {
                assert!(
                    board.in_check(board.side_to_move),
                    "tablebase mainline stalemated: {fen}"
                );
                mated = true;
                break;
            }
        }
        assert!(mated, "tablebase mainline failed to mate: {fen}");
    }
}

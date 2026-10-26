//! Full downloaded 3–5 piece set: cargo test --release --test syzygy_full -- --ignored
use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::Board,
    search::{Limits, Search, is_mate_score},
    syzygy::Syzygy,
};

#[test]
#[ignore = "requires the full local Syzygy set"]
fn multipv_certifies_each_child_including_stalemate_and_rule50_boundaries() {
    let directory = std::env::var("ZENITH_FULL_SYZYGY_PATH")
        .unwrap_or_else(|_| format!("{}/syzygy/3-4-5", env!("CARGO_MANIFEST_DIR")));
    let mut cases = std::collections::BTreeMap::<&str, Vec<(&str, i32)>>::new();
    for line in include_str!("data/multipv-syzygy.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('|').collect();
        cases
            .entry(fields[0])
            .or_default()
            .push((fields[1], fields[2].parse().unwrap()));
    }
    let mut search = Search::new(2);
    search.set_syzygy(Syzygy::load(&directory).unwrap(), 5, 1);
    search.set_multi_pv(32);
    search.set_threads(8);
    search.set_smp_voting(true);
    for (fen, expected) in cases {
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
        assert_eq!(result.depth, 0);
        assert_eq!(result.stats.tb_errors, 0);
        assert!(result.stats.tb_hits > 0);
        assert_eq!(result.variations.len(), expected.len());
        assert!(
            result
                .variations
                .windows(2)
                .all(|pair| pair[0].score >= pair[1].score)
        );
        for line in result.variations {
            let text = line.pv[0].to_string();
            let score = expected.iter().find(|(mv, _)| *mv == text).unwrap().1;
            assert_eq!(line.score, score, "{fen} {text}");
        }
    }
}

#[test]
#[ignore = "requires node tools/download_syzygy.mjs (938 MiB)"]
fn full_set_matches_independent_lichess_outcomes_on_eight_threads() {
    let directory = std::env::var("ZENITH_FULL_SYZYGY_PATH")
        .unwrap_or_else(|_| format!("{}/syzygy/3-4-5", env!("CARGO_MANIFEST_DIR")));
    let tables = Syzygy::load(&directory).unwrap().unwrap();
    assert_eq!(tables.files(), 290);
    assert_eq!(tables.max_pieces(), 5);
    let mut search = Search::new(2);
    search.set_threads(8);
    search.set_singular_extensions(true);
    search.set_smp_voting(true);
    search.set_syzygy(Some(tables), 5, 1);
    // Outcomes independently fetched from https://tablebase.lichess.ovh/standard
    // on 2026-10-25. DTZ values: 0, 51, 21, 1, -6 respectively.
    for (fen, expected) in [
        ("8/8/5k2/5r2/8/3K4/4P3/4R3 w - - 0 1", 0),
        ("7k/8/8/8/8/2K5/1NB5/8 w - - 0 1", 29000),
        ("7k/8/8/8/8/2K5/1BB5/8 w - - 0 1", 29000),
        ("7k/8/8/8/8/2K5/1PP5/8 w - - 0 1", 29000),
        ("7k/8/8/8/8/2K5/1PP5/7q w - - 0 1", -29000),
    ] {
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
        assert_eq!(result.score, expected, "{fen}: {result:?}");
        assert_eq!(result.depth, 0, "root DTZ shortcut: {fen}");
        assert!(result.stats.tb_hits > 0);
        assert_eq!(result.stats.tb_errors, 0);
        assert_eq!(result.stats.helper_failures, 0);
        assert!(!is_mate_score(result.score));
        assert!(board.legal_moves().contains(&result.best_move.unwrap()));
    }
}

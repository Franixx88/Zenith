//! Independent move-generation reference tests.
//!
//! Counts and FENs: https://chessprogramming.org/Perft_Results
//! The same fixtures are used by Stockfish's own tests:
//! https://github.com/official-stockfish/Stockfish/blob/master/tests/perft.sh

use zenith::board::{Board, Color, perft};

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const ENDGAME: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const PROMOTIONS: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const PROMOTIONS_MIRRORED: &str =
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";
const TACTICAL: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const MIDDLEGAME: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

fn check_counts(fen: &str, counts: &[u64]) {
    let board = Board::from_fen(fen).expect("valid reference FEN");
    let original_fen = board.to_fen();
    let original_hash = board.hash();
    for (depth, &expected) in counts.iter().enumerate() {
        assert_eq!(
            perft(&board, depth as u8),
            expected,
            "perft mismatch at depth {depth}: {fen}"
        );
    }
    assert_eq!(board.to_fen(), original_fen, "perft changed its input");
    assert_eq!(board.hash(), original_hash, "perft changed its input hash");
}

#[test]
fn starting_position_perft() {
    check_counts(&Board::startpos().to_fen(), &[1, 20, 400, 8_902, 197_281]);
}

#[test]
fn kiwipete_castling_and_pins() {
    check_counts(KIWIPETE, &[1, 48, 2_039, 97_862]);
}

#[test]
fn endgame_en_passant_and_checks() {
    check_counts(ENDGAME, &[1, 14, 191, 2_812, 43_238]);
}

#[test]
fn promotions_and_check_evasions() {
    check_counts(PROMOTIONS, &[1, 6, 264, 9_467]);
    check_counts(PROMOTIONS_MIRRORED, &[1, 6, 264, 9_467]);
}

#[test]
fn tactical_castling_and_promotions() {
    check_counts(TACTICAL, &[1, 44, 1_486, 62_379]);
}

#[test]
fn middlegame_sliding_attacks() {
    check_counts(MIDDLEGAME, &[1, 46, 2_079, 89_890]);
}

#[test]
#[ignore = "deeper reference suite; run cargo test --release --test perft -- --ignored"]
fn deeper_reference_suite() {
    let starting_fen = Board::startpos().to_fen();
    for (fen, depth, expected) in [
        (starting_fen.as_str(), 5, 4_865_609),
        (KIWIPETE, 4, 4_085_603),
        (ENDGAME, 5, 674_624),
        (PROMOTIONS, 4, 422_333),
        (PROMOTIONS_MIRRORED, 4, 422_333),
        (TACTICAL, 4, 2_103_487),
        (MIDDLEGAME, 4, 3_894_594),
    ] {
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(perft(&board, depth), expected, "depth {depth}: {fen}");
    }
}

#[test]
fn en_passant_cannot_expose_own_king() {
    let board = Board::from_fen("8/8/8/r4pPK/8/8/8/4k3 w - f6 0 1").unwrap();
    assert!(!board.in_check(Color::White));
    assert!(board.parse_uci_move("g5f6").is_err());
    assert!(board.parse_uci_move("g5g6").is_ok());
}

#[test]
fn castling_cannot_cross_an_attacked_square() {
    let board = Board::from_fen("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1").unwrap();
    assert!(!board.in_check(Color::White));
    assert!(board.parse_uci_move("e1g1").is_err());
    assert!(board.parse_uci_move("e1c1").is_ok());
}

#[test]
fn all_four_promotions_are_available() {
    let board = Board::from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
    for suffix in ['q', 'r', 'b', 'n'] {
        assert!(board.parse_uci_move(&format!("a7a8{suffix}")).is_ok());
    }
    assert!(board.parse_uci_move("a7a8").is_err());
    assert!(board.parse_uci_move("a7a8k").is_err());
}

#[test]
fn special_moves_update_all_fen_fields() {
    for (fen, played, expected) in [
        (
            "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
            "e5d6",
            "7k/8/3P4/8/8/8/8/K7 b - - 0 1",
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "e1g1",
            "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1",
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "a1a8",
            "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 1",
        ),
        (
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "a7a8n",
            "N6k/8/8/8/8/8/8/7K b - - 0 1",
        ),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let next = board.make_move(board.parse_uci_move(played).unwrap());
        assert_eq!(next.to_fen(), expected, "after {played} from {fen}");
        assert_eq!(next.hash(), Board::from_fen(expected).unwrap().hash());
        assert_eq!(board.to_fen(), fen, "make_move changed its input");
    }
}

#[test]
fn repetition_hash_ignores_move_counters() {
    let initial = Board::startpos();
    let mut board = initial;
    for played in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        board = board.make_move(board.parse_uci_move(played).unwrap());
    }
    assert_ne!(initial.to_fen(), board.to_fen());
    assert_eq!(initial.hash(), board.hash());
}

#[test]
fn terminal_positions_have_no_children_but_one_depth_zero_node() {
    for (fen, is_check) in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", true),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", false),
    ] {
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.in_check(Color::Black), is_check);
        assert!(board.legal_moves().is_empty());
        assert_eq!(perft(&board, 0), 1);
        assert_eq!(perft(&board, 1), 0);
        assert_eq!(perft(&board, 2), 0);
    }
}

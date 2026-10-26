//! Small exact regressions. Mate-in-two fixtures were independently checked by
//! exhaustive legal minimax through 3 plies using chess.js 1.4.0.
use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::{Board, Piece},
    search::{Limits, MATE_SCORE, Search},
};

#[test]
fn exact_mate_in_two_distances_survive_search_changes() {
    for fen in [
        "7k/8/5K2/8/8/8/8/3Q4 w - - 0 1",
        "7k/8/4K3/8/8/8/8/3Q4 w - - 0 1",
        "8/7k/5K2/8/8/8/8/3Q4 w - - 0 1",
        "k7/8/2K5/8/8/8/8/3Q4 w - - 0 1",
        "8/k7/2K5/8/8/8/8/3Q4 w - - 0 1",
        "7k/8/5K2/8/8/8/8/3R4 w - - 0 1",
        "k7/8/2K5/8/8/8/8/3R4 w - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        let result = Search::new(1).search(
            board,
            &[],
            Limits {
                depth: Some(4),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.score, MATE_SCORE - 3, "{fen}");
        let mut position = board;
        for mv in result.pv {
            assert!(position.legal_moves().contains(&mv));
            position = position.make_move(mv);
        }
    }
}

#[test]
fn underpromotion_avoids_an_immediate_stalemate() {
    let board = Board::from_fen("8/6P1/5K1k/8/5N2/8/8/8 w - - 0 1").unwrap();
    let promotions: Vec<_> = ["g7g8q", "g7g8r", "g7g8b", "g7g8n"]
        .map(|s| board.parse_uci_move(s).unwrap())
        .into();
    assert!(board.make_move(promotions[0]).legal_moves().is_empty());
    let result = Search::new(1).search(
        board,
        &[],
        Limits {
            depth: Some(4),
            searchmoves: Some(promotions),
            ..Limits::default()
        },
        Arc::new(AtomicBool::new(false)),
        |_| {},
    );
    assert!(matches!(
        result.best_move.unwrap().promotion,
        Some(Piece::Rook | Piece::Knight)
    ));
    assert!(
        !board
            .make_move(result.best_move.unwrap())
            .legal_moves()
            .is_empty()
    );
    assert!(result.score > 0);
}

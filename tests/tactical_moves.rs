use zenith::board::{Board, Piece};

fn compare(board: Board, depth: u8) {
    let full = board.legal_moves();
    let expected: Vec<_> = full
        .iter()
        .copied()
        .filter(|mv| {
            mv.promotion.is_some()
                || board.piece_at(mv.to).is_some()
                || (board.piece_at(mv.from).unwrap().1 == Piece::Pawn && mv.from % 8 != mv.to % 8)
        })
        .collect();
    assert_eq!(board.legal_tactical_moves(), expected, "{}", board.to_fen());
    assert_eq!(board.has_legal_move(), !full.is_empty());
    if depth > 0 {
        for mv in full {
            compare(board.make_move(mv), depth - 1);
        }
    }
}

#[test]
fn tactical_generator_matches_filtered_full_generator_in_reference_trees() {
    for (_, fen) in zenith::bench::POSITIONS {
        compare(Board::from_fen(fen).unwrap(), 2);
    }
    for fen in [
        "7k/P7/8/8/8/8/7p/1K6 w - - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
    ] {
        compare(Board::from_fen(fen).unwrap(), 2);
    }
}

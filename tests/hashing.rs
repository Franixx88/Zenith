//! Incremental state must agree with independent reconstruction from FEN.
use zenith::board::Board;

#[test]
fn public_side_field_is_always_reflected_in_key() {
    let mut board = Board::startpos();
    let white = board.hash();
    board.side_to_move = zenith::board::Color::Black;
    assert_ne!(board.hash(), white);
    check_tree(board, 1);
}

fn check_tree(board: Board, depth: u8) {
    let rebuilt = Board::from_fen(&board.to_fen()).unwrap();
    assert_eq!(board.hash(), rebuilt.hash(), "{}", board.to_fen());
    if depth > 0 {
        for mv in board.legal_moves() {
            check_tree(board.make_move(mv), depth - 1);
        }
    }
}

#[test]
fn incremental_keys_match_rebuilt_positions_in_move_trees() {
    check_tree(Board::startpos(), 3);
    for (_, fen) in zenith::bench::POSITIONS {
        check_tree(Board::from_fen(fen).unwrap(), 2);
    }
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
        "7k/P7/8/8/8/8/7p/1K6 w - - 0 1",
    ] {
        check_tree(Board::from_fen(fen).unwrap(), 3);
    }
}

#[test]
fn incremental_keys_survive_deterministic_playouts() {
    let mut random = 0x7f25_4e31_d9a8_b607_u64;
    for _ in 0..16 {
        let mut board = Board::startpos();
        for _ in 0..100 {
            check_tree(board, 0);
            let moves = board.legal_moves();
            if moves.is_empty() {
                break;
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            board = board.make_move(moves[random as usize % moves.len()]);
        }
        check_tree(board, 0);
    }
}

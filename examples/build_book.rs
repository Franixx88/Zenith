//! Deterministic, legal opening variations for testing, not a playing book.
use std::collections::HashSet;
use std::sync::{Arc, atomic::AtomicBool};
use zenith::{
    board::{Board, Color, Piece},
    eval::evaluate,
    search::{Limits, Search},
};

const ROOTS: [&str; 32] = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 e7e5 g1f3 g8f6 d2d3 d7d6",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4",
    "e2e4 e7e5 b1c3 g8f6 f1c4 b8c6",
    "e2e4 e7e5 f2f4 e5f4 g1f3 d7d5",
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6",
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6",
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "d2d4 d7d5 c2c4 c7c6 b1c3 e7e6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6",
    "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6",
    "f2f4 d7d5 g1f3 g8f6 e2e3 g7g6",
    "g2g3 d7d5 f1g2 g8f6 d2d3 e7e5",
    "c2c4 c7c5 b1c3 b8c6 g1f3 g8f6",
];
fn next(rng: &mut u64) -> usize {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng as usize
}
fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: build_book OUTPUT")?;
    let mut rng = 0x5e17_2026_u64;
    let mut seen = HashSet::new();
    let mut filter = Search::new(1);
    filter.set_modern_search(false);
    filter.set_null_pruning(false);
    filter.set_enhancements(false);
    filter.set_pruning(false);
    filter.set_lmr(false);
    let mut out = String::from(
        "# Zenith test openings v1: family|UCI moves; 32 families, 4 variations each.\n",
    );
    for _ in 0..4 {
        for (family, root) in ROOTS.iter().enumerate() {
            let mut initial = Board::startpos();
            for text in root.split_whitespace() {
                initial = initial.make_move(initial.parse_uci_move(text)?);
            }
            let mut accepted = None;
            for _ in 0..2000 {
                let mut board = initial;
                let mut moves = root.to_string();
                for _ in 0..2 {
                    let enemy = board.side_to_move.other();
                    let pawns = board.bitboard(enemy, Piece::Pawn);
                    let file_a = 0x0101_0101_0101_0101;
                    let pawn_attacks = if enemy == Color::White {
                        ((pawns & !file_a) << 7) | ((pawns & !(file_a << 7)) << 9)
                    } else {
                        ((pawns & !file_a) >> 9) | ((pawns & !(file_a << 7)) >> 7)
                    };
                    let choices: Vec<_> = board
                        .legal_moves()
                        .into_iter()
                        .filter(|mv| {
                            let piece = board.piece_at(mv.from).unwrap().1;
                            let from_rank = if board.side_to_move == Color::White {
                                mv.from / 8
                            } else {
                                7 - mv.from / 8
                            };
                            let to_rank = if board.side_to_move == Color::White {
                                mv.to / 8
                            } else {
                                7 - mv.to / 8
                            };
                            let developing = match piece {
                                Piece::Pawn => ![0, 5, 7].contains(&(mv.from % 8)) && to_rank <= 3,
                                Piece::Knight | Piece::Bishop => {
                                    to_rank >= from_rank && (1..=6).contains(&(mv.to % 8))
                                }
                                Piece::King => mv.from.abs_diff(mv.to) == 2,
                                _ => false,
                            };
                            developing
                                && board.piece_at(mv.to).is_none()
                                && mv.promotion.is_none()
                                && !(piece == Piece::Pawn && mv.from % 8 != mv.to % 8)
                                && pawn_attacks & (1u64 << mv.to) == 0
                                && !board.make_move(*mv).in_check(enemy)
                        })
                        .collect();
                    if choices.is_empty() {
                        break;
                    }
                    let mv = choices[next(&mut rng) % choices.len()];
                    board = board.make_move(mv);
                    moves.push(' ');
                    moves.push_str(&mv.to_uci());
                }
                if moves.split_whitespace().count() == 8
                    && evaluate(&board).abs() <= 120
                    && !seen.contains(&board.hash())
                {
                    filter.clear();
                    let checked = filter.search(
                        board,
                        &[board.hash()],
                        Limits {
                            depth: Some(2),
                            ..Limits::default()
                        },
                        Arc::new(AtomicBool::new(false)),
                        |_| {},
                    );
                    if checked.score.abs() > 130 {
                        continue;
                    }
                    seen.insert(board.hash());
                    accepted = Some(moves);
                    break;
                }
            }
            let moves = accepted.ok_or_else(|| format!("cannot fill family {}", family + 1))?;
            out.push_str(&format!("F{:02}|{moves}\n", family + 1));
        }
    }
    std::fs::write(path, out).map_err(|e| e.to_string())?;
    println!("128 unique legal test positions generated; development, static and depth-2 filters");
    Ok(())
}

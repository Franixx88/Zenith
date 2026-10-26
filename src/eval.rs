//! Classical tapered evaluation: PSQT, safe mobility, king pressure and pawns.
mod psqt;
use crate::board::{Board, Color, Piece};
use psqt::piece_square;
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];
const MG_VALUE: [i32; 6] = [100, 320, 335, 500, 950, 0];
const EG_VALUE: [i32; 6] = [120, 305, 330, 530, 930, 0];
const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];
const FULL_PHASE: i32 = 24;
const FILE_A: u64 = 0x0101_0101_0101_0101;

fn pawn_attacks(pawns: u64, color: Color) -> u64 {
    if color == Color::White {
        ((pawns & !FILE_A) << 7) | ((pawns & !(FILE_A << 7)) << 9)
    } else {
        ((pawns & !FILE_A) >> 9) | ((pawns & !(FILE_A << 7)) >> 7)
    }
}
fn distance(a: u8, b: u8) -> i32 {
    i32::from((a % 8).abs_diff(b % 8).max((a / 8).abs_diff(b / 8)))
}

/// Centipawns from the side to move. Mate/draw adjudication belongs to search.
pub fn evaluate(board: &Board) -> i32 {
    let mut mg = [0; 2];
    let mut eg = [0; 2];
    let mut phase = 0;
    let pawns = [
        board.bitboard(Color::White, Piece::Pawn),
        board.bitboard(Color::Black, Piece::Pawn),
    ];
    let pawn_control = [
        pawn_attacks(pawns[0], Color::White),
        pawn_attacks(pawns[1], Color::Black),
    ];
    let kings = [
        board.bitboard(Color::White, Piece::King).trailing_zeros() as u8,
        board.bitboard(Color::Black, Piece::King).trailing_zeros() as u8,
    ];
    for color in [Color::White, Color::Black] {
        let us = color.index();
        let them = color.other().index();
        let own = board.occupied_by(color);
        let king_zone =
            board.piece_attacks(Piece::King, color.other(), kings[them]) | (1 << kings[them]);
        let safe_area = !own & !pawn_control[them];
        let mut attack_units = 0;
        let mut attackers = 0;
        for piece in PIECES {
            let mut bits = board.bitboard(color, piece);
            let count = bits.count_ones() as i32;
            mg[us] += count * MG_VALUE[piece.index()];
            eg[us] += count * EG_VALUE[piece.index()];
            phase += count * PHASE[piece.index()];
            while bits != 0 {
                let square = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                let relative = if color == Color::White {
                    square
                } else {
                    square ^ 56
                };
                let (middle, end) = piece_square(piece, relative);
                mg[us] += middle;
                eg[us] += end;
                if matches!(piece, Piece::Pawn | Piece::King) {
                    continue;
                }
                let attacks = board.piece_attacks(piece, color, square);
                let mobility = (attacks & safe_area).count_ones() as i32;
                let (base, wm, we) = match piece {
                    Piece::Knight => (4, 4, 4),
                    Piece::Bishop => (6, 4, 5),
                    Piece::Rook => (7, 2, 3),
                    _ => (12, 1, 2),
                };
                mg[us] += (mobility - base) * wm;
                eg[us] += (mobility - base) * we;
                let hits = (attacks & king_zone).count_ones() as i32;
                if hits > 0 {
                    attackers += 1;
                    attack_units += hits
                        * match piece {
                            Piece::Queen => 5,
                            Piece::Rook => 3,
                            _ => 2,
                        };
                }
                // A minor supported by a pawn is useful when enemy pawns cannot chase it.
                if matches!(piece, Piece::Knight | Piece::Bishop)
                    && (3..=5).contains(&(relative / 8))
                    && pawn_control[us] & (1 << square) != 0
                {
                    let file = square % 8;
                    let adjacent = (if file > 0 { FILE_A << (file - 1) } else { 0 })
                        | (if file < 7 { FILE_A << (file + 1) } else { 0 });
                    let forward = if color == Color::White {
                        u64::MAX << ((square / 8 + 1) * 8)
                    } else {
                        (1_u64 << (square / 8 * 8)) - 1
                    };
                    if pawns[them] & adjacent & forward == 0 {
                        mg[us] += 18;
                        eg[us] += 10;
                    }
                }
                if piece == Piece::Rook {
                    let file = FILE_A << (square % 8);
                    if pawns[us] & file == 0 {
                        let open = pawns[them] & file == 0;
                        mg[us] += if open { 24 } else { 12 };
                        eg[us] += if open { 14 } else { 7 };
                    }
                    let enemy_relative_king = if color == Color::White {
                        kings[them]
                    } else {
                        kings[them] ^ 56
                    };
                    if relative / 8 == 6 && enemy_relative_king / 8 >= 6 {
                        mg[us] += 18;
                        eg[us] += 25;
                    }
                }
            }
        }
        if board.bitboard(color, Piece::Bishop).count_ones() >= 2 {
            mg[us] += 28;
            eg[us] += 42;
        }
        let (pm, pe) = pawn_structure(board, color);
        mg[us] += pm;
        eg[us] += pe;
        // Coordinated attacks matter more with a queen still on the board.
        if attackers >= 2 {
            let danger = (attack_units * attack_units / 4).min(240);
            mg[us] += if board.bitboard(color, Piece::Queen) != 0 {
                danger
            } else {
                danger / 3
            };
        }
        mg[us] += king_shelter(board, color, kings[us]);
        // Encourage clean conversion against a bare king, especially KRK/KQK.
        if board.occupied_by(color.other()).count_ones() == 1
            && (board.bitboard(color, Piece::Rook) | board.bitboard(color, Piece::Queen)) != 0
        {
            let k = kings[them];
            let edge = (k % 8).min(7 - k % 8).min(k / 8).min(7 - k / 8);
            eg[us] += 25 * (3 - i32::from(edge)) + 12 * (7 - distance(kings[us], k));
        }
    }
    phase = phase.min(FULL_PHASE);
    let score = ((mg[0] - mg[1]) * phase + (eg[0] - eg[1]) * (FULL_PHASE - phase)) / FULL_PHASE;
    let relative = if board.side_to_move == Color::White {
        score
    } else {
        -score
    };
    (relative + 10).clamp(-20_000, 20_000)
}

fn king_shelter(board: &Board, color: Color, king: u8) -> i32 {
    let own_pawns = board.bitboard(color, Piece::Pawn);
    let enemy = board.bitboard(color.other(), Piece::Pawn);
    let file = i32::from(king % 8);
    let rank = i32::from(king / 8);
    let direction = if color == Color::White { 1 } else { -1 };
    let mut score = 0;
    for f in (file - 1).max(0)..=(file + 1).min(7) {
        let mask = FILE_A << f;
        if own_pawns & mask == 0 {
            score -= if enemy & mask == 0 { 24 } else { 16 };
        }
        for steps in 1..=2 {
            let r = rank + direction * steps;
            if (0..8).contains(&r) && own_pawns & (1 << (r * 8 + f)) != 0 {
                score += if steps == 1 { 14 } else { 7 };
            }
        }
    }
    score
}

fn pawn_structure(board: &Board, color: Color) -> (i32, i32) {
    let own = board.bitboard(color, Piece::Pawn);
    let enemy = board.bitboard(color.other(), Piece::Pawn);
    let mut mg = 0;
    let mut eg = 0;
    for file in 0..8 {
        let file_mask = FILE_A << file;
        let mut pawns = own & file_mask;
        let count = pawns.count_ones() as i32;
        if count == 0 {
            continue;
        }
        if count > 1 {
            mg -= (count - 1) * 13;
            eg -= (count - 1) * 20;
        }
        let adjacent_files = if file > 0 { FILE_A << (file - 1) } else { 0 }
            | if file < 7 { FILE_A << (file + 1) } else { 0 };
        if own & adjacent_files == 0 {
            mg -= count * 12;
            eg -= count * 10;
        }
        while pawns != 0 {
            let square = pawns.trailing_zeros();
            pawns &= pawns - 1;
            let rank = square / 8;
            let relative_rank = if color == Color::White {
                rank
            } else {
                7 - rank
            };
            let front = if color == Color::White {
                if rank == 7 {
                    0
                } else {
                    u64::MAX << ((rank + 1) * 8)
                }
            } else if rank == 0 {
                0
            } else {
                (1_u64 << (rank * 8)) - 1
            };
            if enemy & (file_mask | adjacent_files) & front == 0 && own & file_mask & front == 0 {
                let advancement = relative_rank.saturating_sub(1) as i32;
                let next = if color == Color::White {
                    square + 8
                } else {
                    square - 8
                };
                let blocked = board.occupied() & (1 << next) != 0;
                let divisor = if blocked { 2 } else { 1 };
                mg += 4 * advancement * advancement / divisor;
                eg += 8 * advancement * advancement / divisor;
                let own_king = board.bitboard(color, Piece::King).trailing_zeros() as u8;
                let enemy_king = board.bitboard(color.other(), Piece::King).trailing_zeros() as u8;
                eg += 3
                    * advancement
                    * (distance(enemy_king, next as u8) - distance(own_king, next as u8));
            }
            let neighboring_ranks = (0xFF_u64 << (rank * 8))
                | if rank > 0 {
                    0xFF_u64 << ((rank - 1) * 8)
                } else {
                    0
                }
                | if rank < 7 {
                    0xFF_u64 << ((rank + 1) * 8)
                } else {
                    0
                };
            if own & adjacent_files & neighboring_ranks != 0 {
                mg += 5;
                eg += 8;
            }
        }
    }
    (mg, eg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_and_rank_reflection_preserves_relative_score() {
        for (_, fen) in crate::bench::POSITIONS {
            let board = Board::from_fen(fen).unwrap();
            let fields: Vec<_> = fen.split_whitespace().collect();
            let rows: Vec<String> = fields[0]
                .split('/')
                .rev()
                .map(|row| {
                    row.chars()
                        .map(|c| {
                            if c.is_ascii_uppercase() {
                                c.to_ascii_lowercase()
                            } else {
                                c.to_ascii_uppercase()
                            }
                        })
                        .collect()
                })
                .collect();
            let side = if board.side_to_move == Color::White {
                "b"
            } else {
                "w"
            };
            let mirrored = Board::from_fen(&format!("{} {side} - - 0 1", rows.join("/"))).unwrap();
            assert_eq!(evaluate(&board), evaluate(&mirrored), "{fen}");
        }
    }

    #[test]
    fn shelter_and_blocked_passers_have_the_expected_direction() {
        let sheltered = Board::from_fen("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1").unwrap();
        let exposed = Board::from_fen("6k1/8/8/8/5PPP/8/8/6K1 w - - 0 1").unwrap();
        assert!(
            king_shelter(&sheltered, Color::White, 6) > king_shelter(&exposed, Color::White, 6)
        );
        let blocked = Board::from_fen("8/3k4/3P4/8/2K5/8/8/8 w - - 0 1").unwrap();
        let free = Board::from_fen("7k/8/3P4/8/2K5/8/8/8 w - - 0 1").unwrap();
        assert!(pawn_structure(&free, Color::White).1 > pawn_structure(&blocked, Color::White).1);
    }

    #[test]
    fn mobility_and_king_approach_help() {
        let blocked = Board::from_fen("6k1/8/8/8/8/1P1P4/2B5/6K1 w - - 0 1").unwrap();
        let open = Board::from_fen("6k1/8/8/8/8/8/1PBP4/6K1 w - - 0 1").unwrap();
        assert!(evaluate(&open) > evaluate(&blocked));
        let near = Board::from_fen("7k/8/5K2/8/8/8/8/3Q4 w - - 0 1").unwrap();
        let far = Board::from_fen("7k/8/8/8/8/8/8/K2Q4 w - - 0 1").unwrap();
        assert!(evaluate(&near) > evaluate(&far));
    }

    #[test]
    fn starting_position_is_balanced() {
        assert_eq!(evaluate(&Board::startpos()), 10);
        assert_eq!(piece_square(Piece::Pawn, 8), (1, -4));
        assert_eq!(piece_square(Piece::Knight, 1), (-46, -32));
        assert_eq!(piece_square(Piece::Knight, 27), (24, 14));
    }

    #[test]
    fn material_advantage_has_correct_sign() {
        let white = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let black = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1").unwrap();
        assert!(evaluate(&white) > 800);
        assert!(evaluate(&black) < -800);
        assert_eq!(evaluate(&white) + evaluate(&black), 20);
    }
}

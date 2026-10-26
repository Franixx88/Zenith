//! Legal least-valuable-attacker exchanges on a virtual occupied board.
//! Only the target occupant moves: all other live pieces keep their original
//! squares. Removing origins is enough to expose x-rays and recheck king safety.
use super::*;

const VALUES: [i32; 6] = [100, 320, 335, 500, 950, 20_000];

impl Board {
    /// Material gain after legal least-valuable recaptures on the destination.
    /// Either side may stop. This local heuristic is not a tactical search.
    pub fn see(&self, mv: Move) -> i32 {
        let mover = self
            .piece_at(mv.from)
            .expect("SEE expects a generated move")
            .1;
        if mover == Piece::King && mv.from.abs_diff(mv.to) == 2 {
            return 0;
        }
        let mut gains = [0; 32];
        gains[0] = self.exchange_gain(mv, mover);
        let mut exchange = Exchange::new(self, mv, mover);
        let mut depth = 0;
        while depth + 1 < gains.len() {
            let Some(gain) = exchange.capture() else {
                break;
            };
            depth += 1;
            gains[depth] = gain - gains[depth - 1];
        }
        while depth > 0 {
            gains[depth - 1] = gains[depth - 1].min(-gains[depth]);
            depth -= 1;
        }
        gains[0]
    }

    /// Exactly `see(mv) >= threshold`, with early material bounds and a null
    /// window through the exchange. Accepts generated pseudo-legal moves too;
    /// callers must still check the initial move's legality before searching it.
    pub fn see_ge(&self, mv: Move, threshold: i32) -> bool {
        let mover = self
            .piece_at(mv.from)
            .expect("SEE expects a generated move")
            .1;
        if mover == Piece::King && mv.from.abs_diff(mv.to) == 2 {
            return 0 >= threshold;
        }
        let gain = self.exchange_gain(mv, mover);
        let victim = mv.promotion.unwrap_or(mover);
        // A king on the target cannot be recaptured in a legal exchange.
        if victim == Piece::King {
            return gain >= threshold;
        }
        if gain < threshold {
            return false;
        }
        // The opponent can win at most this occupant plus a promotion bonus.
        // Include that bonus on EITHER back rank, including quiet first moves
        // and underpromotions; a pawn's later recapture may promote to a queen.
        let promotion = if matches!(mv.to / 8, 0 | 7) {
            VALUES[4] - VALUES[0]
        } else {
            0
        };
        if gain - VALUES[victim.index()] - promotion >= threshold {
            return true;
        }
        // These bounds also keep subtraction safe for every i32 threshold.
        let mut balance = gain - threshold;
        let mut exchange = Exchange::new(self, mv, mover);
        let mut ours = false;
        while let Some(gain) = exchange.capture() {
            balance = gain - balance;
            // At their turn, <= equality suffices for us. At our turn we must
            // strictly fail the threshold to stop: equality is a successful SEE.
            if ours && balance < 0 {
                return false;
            }
            if !ours && balance <= 0 {
                return true;
            }
            ours = !ours;
        }
        !ours
    }

    fn exchange_gain(&self, mv: Move, mover: Piece) -> i32 {
        self.piece_at(mv.to).map_or_else(
            || {
                if mover == Piece::Pawn && mv.from % 8 != mv.to % 8 {
                    VALUES[0]
                } else {
                    0
                }
            },
            |(_, piece)| VALUES[piece.index()],
        ) + mv
            .promotion
            .map_or(0, |piece| VALUES[piece.index()] - VALUES[0])
    }

    fn exchange_attackers(&self, square: u8, color: Color, occupied: u64, live: u64) -> u64 {
        (PAWN_ATTACKS[color.other().index()][square as usize] & self.bitboard(color, Piece::Pawn)
            | KNIGHT_ATTACKS[square as usize] & self.bitboard(color, Piece::Knight)
            | KING_ATTACKS[square as usize] & self.bitboard(color, Piece::King)
            | sliding_attacks(square, occupied, &DIAGONALS)
                & (self.bitboard(color, Piece::Bishop) | self.bitboard(color, Piece::Queen))
            | sliding_attacks(square, occupied, &ORTHOGONALS)
                & (self.bitboard(color, Piece::Rook) | self.bitboard(color, Piece::Queen)))
            & live
    }

    fn exchange_attacked(&self, square: u8, color: Color, occupied: u64, live: u64) -> bool {
        if (PAWN_ATTACKS[color.other().index()][square as usize]
            & self.bitboard(color, Piece::Pawn)
            | KNIGHT_ATTACKS[square as usize] & self.bitboard(color, Piece::Knight)
            | KING_ATTACKS[square as usize] & self.bitboard(color, Piece::King))
            & live
            != 0
        {
            return true;
        }
        let queens = self.bitboard(color, Piece::Queen) & live;
        sliding_attacks(square, occupied, &DIAGONALS)
            & (queens | (self.bitboard(color, Piece::Bishop) & live))
            != 0
            || sliding_attacks(square, occupied, &ORTHOGONALS)
                & (queens | (self.bitboard(color, Piece::Rook) & live))
                != 0
    }
}

struct Exchange<'a> {
    board: &'a Board,
    target: u8,
    // Original pieces still away from the exchange square. The target always
    // blocks rays, but its ORIGINAL occupant must never reappear as an attacker.
    live: u64,
    side: Color,
    victim: Piece,
}

impl<'a> Exchange<'a> {
    fn new(board: &'a Board, mv: Move, mover: Piece) -> Self {
        let mut live = board.occupied() & !(bit(mv.from) | bit(mv.to));
        if mover == Piece::Pawn
            && board.en_passant == Some(mv.to)
            && board.piece_at(mv.to).is_none()
        {
            live &= !bit(mv.to ^ 8);
        }
        Self {
            board,
            target: mv.to,
            live,
            side: board.side_to_move.other(),
            victim: mv.promotion.unwrap_or(mover),
        }
    }

    fn capture(&mut self) -> Option<i32> {
        if self.victim == Piece::King {
            return None;
        }
        let occupied = self.live | bit(self.target);
        let attackers = self
            .board
            .exchange_attackers(self.target, self.side, occupied, self.live);
        for piece in PIECES {
            let mut candidates = attackers & self.board.bitboard(self.side, piece);
            while candidates != 0 {
                let from = pop_lsb(&mut candidates);
                let live = self.live & !bit(from);
                let king = if piece == Piece::King {
                    self.target
                } else {
                    self.board.bitboard(self.side, Piece::King).trailing_zeros() as u8
                };
                // Re-evaluate after EACH removal: pins may appear/disappear as
                // the exchange opens lines, and a king may capture only safely.
                if self.board.exchange_attacked(
                    king,
                    self.side.other(),
                    live | bit(self.target),
                    live,
                ) {
                    continue;
                }
                let promoted = piece == Piece::Pawn && matches!(self.target / 8, 0 | 7);
                let gain =
                    VALUES[self.victim.index()] + if promoted { VALUES[4] - VALUES[0] } else { 0 };
                self.live = live;
                self.side = self.side.other();
                self.victim = if promoted { Piece::Queen } else { piece };
                return Some(gain);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen 0.12 implementation: real temporary Boards and king checks. This
    // oracle deliberately does not share the virtual occupancy or early exits.
    impl Board {
        /// Material gain of a legal move followed by least-valuable legal
        /// recaptures on its destination. Either side may stop the exchange.
        /// This is a local heuristic, not a tactical search or proof of safety.
        fn see_reference(&self, mv: Move) -> i32 {
            let (_, mover) = self.piece_at(mv.from).expect("SEE expects a legal move");
            if mover == Piece::King && mv.from.abs_diff(mv.to) == 2 {
                return 0;
            }
            let mut gains = [0; 32];
            gains[0] = self.piece_at(mv.to).map_or_else(
                || {
                    if mover == Piece::Pawn && mv.from % 8 != mv.to % 8 {
                        VALUES[0]
                    } else {
                        0
                    }
                },
                |(_, piece)| VALUES[piece.index()],
            ) + mv
                .promotion
                .map_or(0, |piece| VALUES[piece.index()] - VALUES[0]);
            // Temporary positions only feed geometry/check tests, never hashing.
            let mut position = self.apply_move(mv, false);
            let mut depth = 0;
            while depth + 1 < gains.len() {
                let Some((_, victim)) = position.piece_at(mv.to) else {
                    break;
                };
                if victim == Piece::King {
                    break;
                }
                let attackers = position.attackers_to(mv.to, position.side_to_move);
                let mut chosen = None;
                'pieces: for piece in PIECES {
                    let mut candidates =
                        attackers & position.bitboard(position.side_to_move, piece);
                    while candidates != 0 {
                        let from = pop_lsb(&mut candidates);
                        let promotion = (piece == Piece::Pawn && matches!(mv.to / 8, 0 | 7))
                            .then_some(Piece::Queen);
                        let capture = Move {
                            from,
                            to: mv.to,
                            promotion,
                        };
                        let after = position.apply_move(capture, false);
                        if !after.in_check(position.side_to_move) {
                            chosen = Some((after, promotion));
                            break 'pieces;
                        }
                    }
                }
                let Some((after, promotion)) = chosen else {
                    break;
                };
                depth += 1;
                gains[depth] = VALUES[victim.index()]
                    + promotion.map_or(0, |piece| VALUES[piece.index()] - VALUES[0])
                    - gains[depth - 1];
                position = after;
            }
            while depth > 0 {
                gains[depth - 1] = gains[depth - 1].min(-gains[depth]);
                depth -= 1;
            }
            gains[0]
        }

        fn attackers_to(&self, square: u8, color: Color) -> u64 {
            let occupied = self.occupied();
            PAWN_ATTACKS[color.other().index()][square as usize] & self.bitboard(color, Piece::Pawn)
                | KNIGHT_ATTACKS[square as usize] & self.bitboard(color, Piece::Knight)
                | KING_ATTACKS[square as usize] & self.bitboard(color, Piece::King)
                | sliding_attacks(square, occupied, &DIAGONALS)
                    & (self.bitboard(color, Piece::Bishop) | self.bitboard(color, Piece::Queen))
                | sliding_attacks(square, occupied, &ORTHOGONALS)
                    & (self.bitboard(color, Piece::Rook) | self.bitboard(color, Piece::Queen))
        }
    }

    #[test]
    fn virtual_exchange_and_threshold_match_board_copy_oracle() {
        let fixtures = [
            "4k3/4n3/8/3p4/2B5/8/8/4R1K1 w - - 0 1",
            "8/8/4k3/3p4/2B5/8/8/3R2K1 w - - 0 1",
            "3rk3/8/8/3p4/3R4/8/8/3RK3 w - - 0 1",
            "3r3k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
            "8/8/8/r4pPK/8/8/8/4k3 w - f6 0 1",
            "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
            "7k/8/8/8/8/7K/pR6/1b6 w - - 0 1",
            "7k/8/8/8/8/7K/pR6/1r6 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        ];
        let mut seed = 0x8af1_0d91_268e_5bc3u64;
        let mut checked = 0;
        for (_, fen) in crate::bench::POSITIONS
            .into_iter()
            .chain(fixtures.map(|fen| ("fixture", fen)))
        {
            let mut board = Board::from_fen(fen).unwrap();
            for _ in 0..96 {
                let before = board;
                // The move picker calls SEE BEFORE initial king safety. Include
                // pseudo-legal pinned moves/evasions, not just legal exchanges.
                for &mv in &board.generate(GenMode::All) {
                    let expected = board.see_reference(mv);
                    assert_eq!(board.see(mv), expected, "{mv}: {}", board.to_fen());
                    for threshold in [
                        i32::MIN,
                        -1000,
                        -850,
                        -480,
                        -120,
                        -80,
                        -30,
                        -1,
                        0,
                        1,
                        100,
                        500,
                        i32::MAX,
                        expected - 1,
                        expected,
                        expected + 1,
                    ] {
                        assert_eq!(
                            board.see_ge(mv, threshold),
                            expected >= threshold,
                            "{mv}, threshold={threshold}, expected={expected}: {}",
                            board.to_fen()
                        );
                    }
                    checked += 1;
                }
                assert_eq!(board, before);
                let moves = board.legal_moves();
                if moves.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                board = board.make_move(moves[seed as usize % moves.len()]);
            }
        }
        assert!(checked > 20_000, "insufficient coverage: {checked}");
        eprintln!(
            "SEE oracle: {checked} moves, {} threshold comparisons",
            checked * 16
        );
    }
}

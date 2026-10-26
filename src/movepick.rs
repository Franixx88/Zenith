//! Lazy pseudo-legal stages with fixed storage. Legality is checked only when a
//! move is selected; its child board is reused by search. No per-node allocation.
use crate::board::{Board, GenMode, MAX_MOVES, Move, Piece};
use arrayvec::ArrayVec;
const VALUES: [i32; 6] = [100, 320, 335, 500, 950, 20_000];
#[derive(Clone, Copy)]
struct Scored {
    mv: Move,
    score: i32,
    order: u16,
    bad: bool,
    needs_see: bool,
}

pub(crate) struct MovePicker {
    board: Board,
    hash_move: Option<Move>,
    hash_child: Option<Board>,
    entries: ArrayVec<Scored, MAX_MOVES>,
    bad_end: usize,
    stage: u8,
    tactical_only: bool,
    see_ordering: bool,
}
impl MovePicker {
    pub fn new(
        board: &Board,
        hash: Option<(Move, Board)>,
        tactical_only: bool,
        see_ordering: bool,
    ) -> Self {
        Self {
            board: *board,
            hash_move: hash.map(|(mv, _)| mv),
            hash_child: hash.map(|(_, child)| child),
            entries: ArrayVec::new(),
            bad_end: 0,
            stage: 0,
            tactical_only,
            see_ordering,
        }
    }
    fn push(&mut self, entry: Scored) {
        self.entries.push(entry);
    }
    fn pop(&mut self, from: usize, good_only: bool) -> Option<Scored> {
        let index = (from..self.entries.len())
            .filter(|&i| !good_only || !self.entries[i].bad)
            .max_by_key(|&i| {
                (
                    self.entries[i].score,
                    std::cmp::Reverse(self.entries[i].order),
                )
            })?;
        Some(self.entries.swap_remove(index))
    }
    pub fn next(
        &mut self,
        quiet_score: impl Fn(Move) -> i32,
        capture_history: impl Fn(Move) -> i32,
    ) -> Option<(Move, Board)> {
        loop {
            let candidate = match self.stage {
                0 => {
                    self.stage = 1;
                    if let Some(child) = self.hash_child.take() {
                        return Some((self.hash_move.unwrap(), child));
                    }
                    continue;
                }
                1 => {
                    let mode = if self.see_ordering || self.tactical_only {
                        GenMode::Tactical
                    } else {
                        GenMode::All
                    };
                    let generated = self.board.generate(mode);
                    for (order, &mv) in generated.iter().enumerate() {
                        if Some(mv) == self.hash_move {
                            continue;
                        }
                        if !self.see_ordering {
                            self.push(Scored {
                                mv,
                                score: quiet_score(mv),
                                order: order as u16,
                                bad: false,
                                needs_see: false,
                            });
                            continue;
                        }
                        let captured = self.board.piece_at(mv.to).map_or_else(
                            || {
                                if is_capture(&self.board, mv) {
                                    VALUES[0]
                                } else {
                                    0
                                }
                            },
                            |(_, p)| VALUES[p.index()],
                        );
                        let attacker = self
                            .board
                            .piece_at(mv.from)
                            .map_or(0, |(_, p)| VALUES[p.index()]);
                        let promotion = mv.promotion.map_or(0, |p| VALUES[p.index()] - VALUES[0]);
                        let good = mv.promotion.is_none()
                            && (1..7).contains(&(mv.to / 8))
                            && captured >= attacker;
                        self.push(Scored {
                            mv,
                            score: 16 * captured - attacker + promotion + capture_history(mv),
                            order: order as u16,
                            bad: false,
                            needs_see: !good,
                        });
                    }
                    self.stage = if self.see_ordering { 2 } else { 4 };
                    continue;
                }
                2 => match self.pop(0, true) {
                    Some(mut entry) => {
                        // Only classify the highest-scoring remaining tactic.
                        // A cutoff on this move avoids SEE for all later ones.
                        // Explicit order keys preserve the eager picker's ties.
                        if entry.needs_see && !self.board.see_ge(entry.mv, 0) {
                            entry.bad = true;
                            entry.needs_see = false;
                            self.push(entry);
                            continue;
                        }
                        Some(entry.mv)
                    }
                    None => {
                        self.stage = 3;
                        continue;
                    }
                },
                3 => {
                    self.bad_end = self.entries.len();
                    if !self.tactical_only {
                        let generated = self.board.generate(GenMode::Quiet);
                        for (order, &mv) in generated.iter().enumerate() {
                            if Some(mv) != self.hash_move {
                                self.push(Scored {
                                    mv,
                                    score: quiet_score(mv),
                                    order: order as u16,
                                    bad: false,
                                    needs_see: false,
                                });
                            }
                        }
                    }
                    self.stage = 4;
                    continue;
                }
                4 => match self.pop(if self.see_ordering { self.bad_end } else { 0 }, false) {
                    Some(entry) => Some(entry.mv),
                    None => {
                        self.stage = 5;
                        continue;
                    }
                },
                5 => match self.pop(0, false) {
                    Some(entry) => Some(entry.mv),
                    None => {
                        self.stage = 6;
                        continue;
                    }
                },
                _ => return None,
            };
            if let Some(mv) = candidate
                && let Some(child) = self.board.generated_child(mv)
            {
                return Some((mv, child));
            }
        }
    }
}
pub(crate) fn is_capture(board: &Board, mv: Move) -> bool {
    board.piece_at(mv.to).is_some()
        || (mv.from % 8 != mv.to % 8
            && board
                .piece_at(mv.from)
                .is_some_and(|(_, p)| p == Piece::Pawn))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    fn collect(
        board: &Board,
        hash: Option<Move>,
        tactical: bool,
        score: impl Fn(Move) -> i32,
    ) -> Vec<Move> {
        let mut picker = MovePicker::new(
            board,
            hash.and_then(|mv| board.hash_child(mv).map(|child| (mv, child))),
            tactical,
            true,
        );
        let mut moves = vec![];
        while let Some((mv, child)) = picker.next(&score, |_| 0) {
            assert_eq!(child, board.make_move(mv));
            moves.push(mv);
        }
        moves
    }
    #[test]
    fn hash_yields_before_any_generation_and_invalid_hash_is_rejected() {
        let board = Board::startpos();
        let mv = board.parse_uci_move("e2e4").unwrap();
        let mut picker = MovePicker::new(&board, Some((mv, board.make_move(mv))), false, true);
        assert_eq!(
            picker
                .next(
                    |_| panic!("quiets scored before hash"),
                    |_| panic!("captures scored before hash")
                )
                .unwrap()
                .0,
            mv
        );
        assert!(picker.entries.is_empty());
        assert_eq!(picker.stage, 1);
        assert_eq!(
            collect(
                &board,
                Some(Move {
                    from: 0,
                    to: 63,
                    promotion: None
                }),
                false,
                |_| 0
            )
            .len(),
            20
        );
    }
    #[test]
    fn capture_cutoff_does_not_classify_remaining_exchanges() {
        let board = Board::from_fen("4k3/8/4p3/3p3r/8/8/2B5/3QK3 w - - 0 1").unwrap();
        let winning = board.parse_uci_move("d1h5").unwrap();
        let losing = board.parse_uci_move("d1d5").unwrap();
        assert!(board.see(losing) < 0);
        let mut picker = MovePicker::new(&board, None, false, true);
        assert_eq!(
            picker
                .next(|_| panic!("premature quiet generation"), |_| 0)
                .unwrap()
                .0,
            winning
        );
        let pending = picker
            .entries
            .iter()
            .find(|entry| entry.mv == losing)
            .unwrap();
        assert!(
            pending.needs_see && !pending.bad,
            "remaining SEE must still be unevaluated"
        );
        assert_eq!(picker.stage, 2);
    }

    #[test]
    fn lazy_exchange_stages_preserve_the_complete_eager_order() {
        let quiet_score = |mv: Move| (i32::from(mv.from) * 7 + i32::from(mv.to) * 3) % 11;
        let history = |mv: Move| (i32::from(mv.from) * 17 + i32::from(mv.to)) % 5;
        for (_, fen) in crate::bench::POSITIONS.into_iter().chain([
            ("evasion", "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"),
            ("promotion", "1r5k/P7/8/8/8/8/8/7K w - - 0 1"),
            ("ep-pin", "8/8/8/r4pPK/8/8/8/4k3 w - f6 0 1"),
        ]) {
            let board = Board::from_fen(fen).unwrap();
            for tactical in [false, true] {
                // Frozen eager stage semantics: classify every pseudo-tactic,
                // sort each stage, THEN check legality. Small score moduli
                // deliberately exercise tie stability after swap_remove.
                let mut expected = Vec::new();
                for (order, &mv) in board.generate(GenMode::Tactical).iter().enumerate() {
                    let captured = board.piece_at(mv.to).map_or_else(
                        || i32::from(is_capture(&board, mv)) * 100,
                        |(_, p)| VALUES[p.index()],
                    );
                    let attacker = VALUES[board.piece_at(mv.from).unwrap().1.index()];
                    let promotion = mv.promotion.map_or(0, |p| VALUES[p.index()] - 100);
                    let good = (mv.promotion.is_none()
                        && (1..7).contains(&(mv.to / 8))
                        && captured >= attacker)
                        || board.see(mv) >= 0;
                    expected.push((
                        (
                            if good { 0 } else { 2 },
                            std::cmp::Reverse(16 * captured - attacker + promotion + history(mv)),
                            order,
                        ),
                        mv,
                    ));
                }
                if !tactical {
                    for (order, &mv) in board.generate(GenMode::Quiet).iter().enumerate() {
                        expected.push(((1, std::cmp::Reverse(quiet_score(mv)), order), mv));
                    }
                }
                expected.sort_by_key(|&(key, _)| key);
                let expected: Vec<_> = expected
                    .into_iter()
                    .map(|(_, mv)| mv)
                    .filter(|&mv| board.generated_child(mv).is_some())
                    .collect();
                let mut picker = MovePicker::new(&board, None, tactical, true);
                let mut actual = Vec::new();
                while let Some((mv, _)) = picker.next(quiet_score, history) {
                    actual.push(mv);
                }
                assert_eq!(actual, expected, "tactical={tactical}: {fen}");
            }
        }
    }
    #[test]
    fn stages_preserve_legal_set_evasions_promotions_and_bad_capture_order() {
        for (_, fen) in crate::bench::POSITIONS.into_iter().chain([
            ("evasion", "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"),
            ("promotion", "7k/P7/8/8/8/8/8/7K w - - 0 1"),
            ("ep-pin", "8/8/8/r4pPK/8/8/8/4k3 w - f6 0 1"),
        ]) {
            let board = Board::from_fen(fen).unwrap();
            for tactical in [false, true] {
                let reference = if tactical {
                    board.legal_tactical_moves()
                } else {
                    board.legal_moves()
                };
                let picked = collect(&board, None, tactical, |_| 0);
                assert_eq!(picked.len(), reference.len());
                assert_eq!(
                    picked.into_iter().collect::<HashSet<_>>(),
                    reference.into_iter().collect()
                );
            }
        }
        let board = Board::from_fen("4k3/8/4p3/3p3p/8/8/8/3QK3 w - - 0 1").unwrap();
        let good = board.parse_uci_move("d1h5").unwrap();
        let bad = board.parse_uci_move("d1d5").unwrap();
        let ordered = collect(&board, None, false, |_| 0);
        let at = |mv| ordered.iter().position(|&m| m == mv).unwrap();
        for &mv in &ordered {
            if !is_capture(&board, mv) {
                assert!(at(good) < at(mv));
                assert!(at(mv) < at(bad));
            }
        }
    }
    #[test]
    fn hash_geometry_matches_generated_legality_exhaustively() {
        for (_, fen) in crate::bench::POSITIONS.into_iter().chain([
            (
                "black-start",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            ),
            ("white-ep", "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1"),
            ("black-ep", "7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1"),
            ("black-promote", "7k/8/8/8/8/8/p7/1R5K b - - 0 1"),
            ("black-castle", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        ]) {
            let board = Board::from_fen(fen).unwrap();
            let legal = board.legal_moves();
            for from in 0..64 {
                for to in 0..64 {
                    for promotion in [
                        None,
                        Some(Piece::Queen),
                        Some(Piece::Rook),
                        Some(Piece::Bishop),
                        Some(Piece::Knight),
                        Some(Piece::King),
                    ] {
                        let mv = Move {
                            from,
                            to,
                            promotion,
                        };
                        assert_eq!(
                            board.hash_child(mv).is_some(),
                            legal.contains(&mv),
                            "{fen}: {mv}"
                        );
                    }
                }
            }
        }
    }
}

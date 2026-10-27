//! Experimental position-context histories. Private to each SMP worker.
use super::update_history;
use crate::board::{Board, Color, Move, Piece};

const CORR_SIZE: usize = 16384;
const PAWN_BUCKETS: usize = 256;
const PIECES: [Piece; 5] = [
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

pub(super) struct ContextHistory {
    threats: Vec<i32>,
    pawn_moves: Vec<i32>,
    non_pawn: [Vec<i16>; 2],
}
impl ContextHistory {
    pub fn new() -> Self {
        Self {
            threats: vec![0; 2 * 4 * 64 * 64],
            pawn_moves: vec![0; PAWN_BUCKETS * 2 * 6 * 64],
            non_pawn: std::array::from_fn(|_| vec![0; 2 * CORR_SIZE]),
        }
    }
    pub fn clear(&mut self) {
        self.threats.fill(0);
        self.pawn_moves.fill(0);
        for table in &mut self.non_pawn {
            table.fill(0);
        }
    }
    fn pawn_key(board: &Board) -> u64 {
        board.bitboard(Color::White, Piece::Pawn).rotate_left(13)
            ^ board.bitboard(Color::Black, Piece::Pawn).rotate_right(7)
    }
    fn pawn_index(board: &Board, mv: Move) -> usize {
        let bucket = ((Self::pawn_key(board).wrapping_mul(0x9e3779b97f4a7c15)) >> 56) as usize;
        let piece = board.piece_at(mv.from).expect("history move").1.index();
        ((bucket * 2 + board.side_to_move.index()) * 6 + piece) * 64 + usize::from(mv.to)
    }
    fn threat_index(board: &Board, mv: Move, attacks: u64) -> usize {
        let from = usize::from(attacks & (1u64 << mv.from) != 0);
        let to = usize::from(attacks & (1u64 << mv.to) != 0);
        (((board.side_to_move.index() * 2 + from) * 2 + to) * 64 + usize::from(mv.from)) * 64
            + usize::from(mv.to)
    }
    pub fn score(&self, board: &Board, mv: Move, attacks: u64) -> i32 {
        (self.threats[Self::threat_index(board, mv, attacks)]
            + self.pawn_moves[Self::pawn_index(board, mv)])
            / 2
    }
    pub fn reward(&mut self, board: &Board, mv: Move, attacks: u64, bonus: i32) {
        update_history(
            &mut self.threats[Self::threat_index(board, mv, attacks)],
            bonus,
        );
        update_history(&mut self.pawn_moves[Self::pawn_index(board, mv)], bonus);
    }
    fn non_pawn_index(board: &Board, color: Color) -> usize {
        let mut key = 0x517cc1b727220a95u64;
        for (i, piece) in PIECES.into_iter().enumerate() {
            key = key.rotate_left(11)
                ^ board
                    .bitboard(color, piece)
                    .wrapping_mul(0x9e3779b97f4a7c15u64.wrapping_add(2 * i as u64));
        }
        board.side_to_move.index() * CORR_SIZE
            + ((key.wrapping_mul(0x94d049bb133111eb) >> 50) as usize)
    }
    pub fn corrected(&self, board: &Board, pawn: i32) -> i32 {
        let white = self.non_pawn[0][Self::non_pawn_index(board, Color::White)];
        let black = self.non_pawn[1][Self::non_pawn_index(board, Color::Black)];
        // Each component learns the complete residual. Averaging keeps its
        // fixed point and avoids summing the same residual three times.
        (2 * pawn + i32::from(white) + i32::from(black)) / 4
    }
    pub fn learn_correction(&mut self, board: &Board, target: i32, weight: i32) {
        for color in [Color::White, Color::Black] {
            let entry = &mut self.non_pawn[color.index()][Self::non_pawn_index(board, color)];
            *entry = ((i32::from(*entry) * (256 - weight) + target * weight) / 256) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn threats_and_pawn_contexts_are_distinct_and_clearable() {
        let b = Board::startpos();
        let mv = b.parse_uci_move("e2e4").unwrap();
        let mut h = ContextHistory::new();
        h.reward(&b, mv, 1 << mv.from, 1000);
        assert!(h.score(&b, mv, 1 << mv.from) > h.score(&b, mv, 0));
        let next = b.make_move(b.parse_uci_move("d2d4").unwrap());
        assert_ne!(
            ContextHistory::pawn_index(&b, mv),
            ContextHistory::pawn_index(&next, mv)
        );
        h.learn_correction(&b, 4096, 16);
        assert!(h.corrected(&b, 0) > 0);
        h.clear();
        assert_eq!(h.score(&b, mv, 0), 0);
        assert_eq!(h.corrected(&b, 0), 0);
    }
    #[test]
    fn nonpawn_key_tracks_each_color_separately() {
        let a = Board::startpos();
        let b = a.make_move(a.parse_uci_move("g1f3").unwrap());
        // side-to-move is deliberately part of the table index; compare keys
        // after undoing that high-bit side partition.
        assert_ne!(
            ContextHistory::non_pawn_index(&a, Color::White) % CORR_SIZE,
            ContextHistory::non_pawn_index(&b, Color::White) % CORR_SIZE
        );
        assert_eq!(
            ContextHistory::non_pawn_index(&a, Color::Black) % CORR_SIZE,
            ContextHistory::non_pawn_index(&b, Color::Black) % CORR_SIZE
        );
    }
}

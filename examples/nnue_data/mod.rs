//! Canonical 100-byte NNUE record shared by the TSV and binpack adapters.
use zenith::board::{Board, Color, Piece};
pub const RECORD_BYTES: usize = 100;
pub fn split_key(record: &[u8; RECORD_BYTES]) -> u64 {
    record[..96]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
}
pub fn encode(board: &Board, score_stm: i16, depth: u8) -> [u8; RECORD_BYTES] {
    let side = board.side_to_move;
    let flip = |bits: u64| {
        if side == Color::Black {
            bits.swap_bytes()
        } else {
            bits
        }
    };
    let mut record = [0; RECORD_BYTES];
    for (i, piece) in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ]
    .into_iter()
    .enumerate()
    {
        record[i * 8..i * 8 + 8].copy_from_slice(&flip(board.bitboard(side, piece)).to_le_bytes());
        record[(6 + i) * 8..(6 + i) * 8 + 8]
            .copy_from_slice(&flip(board.bitboard(side.other(), piece)).to_le_bytes());
    }
    record[96..98].copy_from_slice(&score_stm.to_le_bytes());
    record[98] = depth;
    record
}

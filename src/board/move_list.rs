use super::Move;

// <=16 men per color, each with at most 27 queen destinations. A promoting
// pawn has at most 12 destinations counting all four piece choices.
pub(crate) const MAX_MOVES: usize = 512;
pub(crate) const EMPTY_MOVE: Move = Move {
    from: 0,
    to: 0,
    promotion: None,
};
// ArrayVec keeps only the written prefix initialized: no heap allocation and
// no clearing hundreds of unused entries on every search node.
pub(crate) type MoveList = arrayvec::ArrayVec<Move, MAX_MOVES>;

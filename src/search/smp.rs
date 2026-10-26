//! Root consensus over completed iterations. Decisive main-thread results and
//! root tablebase answers stay authoritative. No scores from unfinished depths.
use super::{MATE_SCORE, MAX_PLY, SearchResult, is_mate_score};
use crate::board::{Board, Move};

fn valid(board: Board, allowed: Option<&[Move]>, result: &SearchResult) -> bool {
    let Some(first) = result.best_move else {
        return false;
    };
    if result.depth == 0
        || result.pv.first() != Some(&first)
        || allowed.is_some_and(|moves| !moves.contains(&first))
    {
        return false;
    }
    let mut position = board;
    for &mv in &result.pv {
        let Some(child) = position.hash_child(mv) else {
            return false;
        };
        position = child;
    }
    true
}
fn decisive(score: i32) -> bool {
    score.abs() >= crate::syzygy::TB_WIN_SCORE - MAX_PLY as i32
}
fn winning_mate(board: Board, result: &SearchResult) -> bool {
    if result.score <= 0
        || !is_mate_score(result.score)
        || MATE_SCORE - result.score != result.pv.len() as i32
    {
        return false;
    }
    let mut position = board;
    for &mv in &result.pv {
        let Some(child) = position.hash_child(mv) else {
            return false;
        };
        position = child;
    }
    position.side_to_move != board.side_to_move
        && position.in_check(position.side_to_move)
        && !position.has_legal_move()
}
pub(super) fn select(board: Board, allowed: Option<&[Move]>, results: &[SearchResult]) -> usize {
    let main = &results[0];
    if decisive(main.score) || (main.depth == 0 && main.stats.tb_hits != 0) {
        return 0;
    }
    let eligible: Vec<usize> = results
        .iter()
        .enumerate()
        .filter_map(|(id, result)| {
            (result.depth.saturating_add(2) >= main.depth && valid(board, allowed, result))
                .then_some(id)
        })
        .collect();
    if let Some(id) = eligible
        .iter()
        .copied()
        .filter(|&id| winning_mate(board, &results[id]))
        .max_by_key(|&id| results[id].score)
    {
        return id;
    }
    let ordinary: Vec<_> = eligible
        .into_iter()
        .filter(|&id| !decisive(results[id].score))
        .collect();
    let Some(minimum) = ordinary.iter().map(|&id| results[id].score).min() else {
        return 0;
    };
    let vote = |mv: Option<Move>| -> i64 {
        ordinary
            .iter()
            .filter(|&&id| results[id].best_move == mv)
            .map(|&id| {
                i64::from((results[id].score - minimum + 16).clamp(1, 216))
                    * i64::from(results[id].depth)
            })
            .sum()
    };
    ordinary
        .iter()
        .copied()
        .max_by_key(|&id| {
            let r = &results[id];
            (
                vote(r.best_move),
                r.depth,
                r.score,
                r.pv.len(),
                std::cmp::Reverse(id),
            )
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchStats;
    fn result(board: Board, mv: &str, score: i32, depth: u8) -> SearchResult {
        let mv = board.parse_uci_move(mv).unwrap();
        SearchResult {
            variations: Vec::new(),
            best_move: Some(mv),
            pv: vec![mv],
            score,
            depth,
            nodes: 1,
            null_probes: 0,
            null_cutoffs: 0,
            stats: SearchStats::default(),
        }
    }
    #[test]
    fn consensus_rejects_shallow_outliers_invalid_pvs_and_disallowed_roots() {
        let board = Board::startpos();
        let mut results = vec![
            result(board, "e2e4", 10, 8),
            result(board, "d2d4", 20, 8),
            result(board, "d2d4", 18, 8),
            result(board, "g2g4", 2000, 2),
        ];
        assert_eq!(select(board, None, &results), 1);
        let allowed = [board.parse_uci_move("e2e4").unwrap()];
        assert_eq!(select(board, Some(&allowed), &results), 0);
        let first = results[1].pv[0];
        results[1].pv.push(first);
        assert_eq!(select(board, None, &results), 2);
        results[0].score = MATE_SCORE - 5;
        assert_eq!(select(board, None, &results), 0);
    }
    #[test]
    fn legal_completed_mate_can_replace_an_ordinary_main_result() {
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let results = [
            result(board, "f7f6", 100, 3),
            result(board, "f7g7", MATE_SCORE - 1, 2),
        ];
        assert_eq!(select(board, None, &results), 1);
    }
}

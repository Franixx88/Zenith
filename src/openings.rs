//! Versioned test openings with family IDs for paired games and dataset splits.
use crate::board::Board;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Opening {
    pub family: String,
    pub moves: Vec<String>,
    pub key: u64,
}

pub fn parse(text: &str) -> Result<Vec<Opening>, String> {
    let mut result = Vec::new();
    let mut keys = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (family, moves) = line
            .split_once('|')
            .ok_or_else(|| format!("opening line {}: expected family|moves", index + 1))?;
        if family.is_empty()
            || !family
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
        {
            return Err("opening family must be an ASCII identifier".into());
        }
        let moves: Vec<_> = moves.split_whitespace().map(str::to_owned).collect();
        if moves.is_empty() || moves.len() > 64 {
            return Err("opening needs 1..64 plies".into());
        }
        let mut board = Board::startpos();
        for mv in &moves {
            board = board.make_move(board.parse_uci_move(mv)?);
        }
        if board.legal_moves().is_empty()
            || board.is_insufficient_material()
            || board.halfmove_clock >= 100
        {
            return Err("opening is terminal".into());
        }
        if !keys.insert(board.hash()) {
            return Err("duplicate opening position".into());
        }
        result.push(Opening {
            family: family.to_string(),
            moves,
            key: board.hash(),
        });
    }
    if result.is_empty() {
        return Err("opening book is empty".into());
    }
    Ok(result)
}

pub fn test_book() -> Vec<Opening> {
    parse(include_str!("../data/openings.txt")).expect("validated built-in opening book")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn book_has_128_legal_unique_positions_in_32_families() {
        let book = test_book();
        assert_eq!(book.len(), 128);
        let families: HashSet<_> = book.iter().map(|o| &o.family).collect();
        assert_eq!(families.len(), 32);
        for family in families {
            assert_eq!(book.iter().filter(|o| &o.family == family).count(), 4);
        }
        for opening in book {
            assert_eq!(opening.moves.len(), 8);
            let mut board = Board::startpos();
            for mv in opening.moves {
                board = board.make_move(board.parse_uci_move(&mv).unwrap());
            }
            assert_eq!(board.hash(), opening.key);
            assert!(crate::eval::evaluate(&board).abs() <= 180);
        }
    }
    #[test]
    fn malformed_illegal_and_duplicate_books_are_rejected() {
        for book in ["", "bad", "F|e2e5", "bad name|e2e4", "A|e2e4\nB|e2e4"] {
            assert!(parse(book).is_err());
        }
    }
}

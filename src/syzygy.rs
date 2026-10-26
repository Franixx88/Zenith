//! Safe file-backed Syzygy adapter. No mmap and no process-global table state.
//! WDL cutoffs are restricted to real zeroing positions: the tables do not know
//! the game's reversible history. Root DTZ moves additionally check the actual
//! halfmove counter and reject rounded/ambiguous results.
use crate::board::{Board, Move};
use shakmaty::{CastlingMode, Chess, fen::Fen, uci::UciMove};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::sync::Arc;

pub const TB_WIN_SCORE: i32 = 29_000;

pub struct Syzygy {
    tables: Tablebase<Chess>,
    files: usize,
}
impl Syzygy {
    /// An explicit environment value (including <empty>) overrides discovery.
    /// Only a completed, verified local download is enabled automatically.
    pub(crate) fn default_path() -> String {
        if let Ok(path) = std::env::var("ZENITH_SYZYGY_PATH") {
            return path;
        }
        if let Ok(executable) = std::env::current_exe() {
            for directory in executable
                .parent()
                .into_iter()
                .flat_map(|p| p.ancestors())
                .take(3)
            {
                let path = directory.join("syzygy/3-4-5");
                if path.join("verified.json").is_file() {
                    return path.to_string_lossy().into_owned();
                }
            }
        }
        "<empty>".into()
    }

    /// Directory lists use the platform separator (`;` on Windows).
    /// Construct before replacing an active configuration: errors are atomic.
    pub fn load(paths: &str) -> Result<Option<Arc<Self>>, String> {
        if paths.trim().is_empty() || paths.trim() == "<empty>" {
            return Ok(None);
        }
        let mut tables = Tablebase::new();
        let mut files = 0;
        for path in std::env::split_paths(paths.trim_matches('"')) {
            files += tables
                .add_directory(&path)
                .map_err(|e| format!("Syzygy {}: {e}", path.display()))?;
        }
        if files == 0 {
            return Err("SyzygyPath contains no table files".into());
        }
        Ok(Some(Arc::new(Self { tables, files })))
    }
    pub fn files(&self) -> usize {
        self.files
    }
    pub fn max_pieces(&self) -> usize {
        self.tables.max_pieces()
    }
    pub(crate) fn eligible(&self, board: &Board, limit: usize) -> bool {
        board.occupied().count_ones() as usize <= limit.min(self.max_pieces())
            && !board.has_castling_rights()
    }
    fn position(board: &Board) -> Result<Chess, String> {
        board
            .to_fen()
            .parse::<Fen>()
            .map_err(|e| e.to_string())?
            .into_position(CastlingMode::Standard)
            .map_err(|e| e.to_string())
    }
    pub(crate) fn wdl_zeroing(&self, board: &Board, ply: usize) -> Result<i32, String> {
        debug_assert_eq!(board.halfmove_clock, 0);
        self.tables
            .probe_wdl_after_zeroing(&Self::position(board)?)
            .map(|wdl| score(wdl, ply))
            .map_err(|e| e.to_string())
    }
    pub(crate) fn root(
        &self,
        board: &Board,
        allowed: &[Move],
    ) -> Result<Option<(Move, i32)>, String> {
        let pos = Self::position(board)?;
        let Some(wdl) = self
            .tables
            .probe_wdl(&pos)
            .map_err(|e| e.to_string())?
            .unambiguous()
        else {
            return Ok(None);
        };
        let Some((mv, _dtz)) = self.tables.best_move(&pos).map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let text = UciMove::from_move(mv, CastlingMode::Standard).to_string();
        let Some(&mv) = allowed.iter().find(|m| m.to_string() == text) else {
            return Ok(None);
        };
        let child = board.make_move(mv);
        let child_value = if !child.has_legal_move() {
            if child.in_check(child.side_to_move) {
                TB_WIN_SCORE
            } else {
                0
            }
        } else if child.halfmove_clock >= 100 || child.is_insufficient_material() {
            0
        } else {
            let Some(child_wdl) = self
                .tables
                .probe_wdl(&Self::position(&child)?)
                .map_err(|e| e.to_string())?
                .unambiguous()
            else {
                return Ok(None);
            };
            score(-child_wdl, 0)
        };
        // Arbitrary FENs can land at a DTZ rounding boundary. Never certify a
        // move whose actual child fails to preserve the root's 50-move result.
        Ok((child_value == score(wdl, 0)).then_some((mv, child_value)))
    }

    /// Certify each actual child. A winning root does not imply every move wins.
    /// Rounded rule-50 ambiguity falls back to search; DTZ only breaks ties.
    pub(crate) fn root_multiple(
        &self,
        board: &Board,
        allowed: &[Move],
        count: usize,
        mut stopped: impl FnMut() -> bool,
    ) -> Result<Option<Vec<(Move, i32)>>, String> {
        let position = Self::position(board)?;
        let preferred = self
            .tables
            .best_move(&position)
            .map_err(|e| e.to_string())?
            .map(|(mv, _)| UciMove::from_move(mv, CastlingMode::Standard).to_string());
        let mut ranked = Vec::with_capacity(allowed.len());
        for &mv in allowed {
            if stopped() {
                return Ok(None);
            }
            let child = board.make_move(mv);
            let terminal = !child.has_legal_move();
            let (value, distance) = if terminal {
                (
                    if child.in_check(child.side_to_move) {
                        TB_WIN_SCORE
                    } else {
                        0
                    },
                    0,
                )
            } else if child.halfmove_clock >= 100 || child.is_insufficient_material() {
                (0, 0)
            } else {
                let pos = Self::position(&child)?;
                let Some(wdl) = self
                    .tables
                    .probe_wdl(&pos)
                    .map_err(|e| e.to_string())?
                    .unambiguous()
                else {
                    return Ok(None);
                };
                let value = score(-wdl, 0);
                let distance = if child.halfmove_clock == 0 || value == 0 {
                    0
                } else {
                    self.tables
                        .probe_dtz(&pos)
                        .map_err(|e| e.to_string())?
                        .ignore_rounding()
                        .0
                        .abs()
                };
                (value, distance)
            };
            let dtz_order = if value > 0 {
                -distance
            } else if value < 0 {
                distance
            } else {
                0
            };
            let preferred = preferred
                .as_deref()
                .is_some_and(|text| text == mv.to_string());
            ranked.push((mv, value, preferred, dtz_order));
        }
        ranked.sort_by_key(|entry| std::cmp::Reverse((entry.1, entry.2, entry.3)));
        Ok(Some(
            ranked
                .into_iter()
                .take(count)
                .map(|(mv, value, _, _)| (mv, value))
                .collect(),
        ))
    }
}

fn score(wdl: Wdl, ply: usize) -> i32 {
    match wdl {
        Wdl::Win => TB_WIN_SCORE - ply as i32,
        Wdl::Loss => -TB_WIN_SCORE + ply as i32,
        Wdl::BlessedLoss | Wdl::Draw | Wdl::CursedWin => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fifty_move_draws_are_not_wins_or_mates() {
        for wdl in [Wdl::BlessedLoss, Wdl::Draw, Wdl::CursedWin] {
            assert_eq!(score(wdl, 8), 0);
        }
        assert_eq!(score(Wdl::Win, 5), 28995);
        assert_eq!(score(Wdl::Loss, 5), -28995);
        assert!(!crate::search::is_mate_score(score(Wdl::Win, 0)));
        assert!(Syzygy::load("<empty>").unwrap().is_none());
        assert!(Syzygy::load("target/does-not-exist-syzygy").is_err());
    }
}

//! NNUE preparation stage 1: outcome-labelled quiet positions from validated logs.
//! No network or teacher centipawn labels are produced here.
use crate::board::{Board, Color};
use std::collections::{BTreeMap, HashSet};

pub fn validation_family(family: &str) -> bool {
    let hash = family.bytes().fold(0xcbf29ce484222325u64, |h, c| {
        (h ^ u64::from(c)).wrapping_mul(0x100000001b3)
    });
    hash.is_multiple_of(5)
}

struct Sample {
    fen: String,
    family: String,
    validation: bool,
    outcome_sum: f64,
    count: u64,
}
#[derive(Default)]
pub struct Dataset {
    samples: BTreeMap<u64, Sample>,
    banned: HashSet<u64>,
    pub games: usize,
    pub excluded_adjudications: usize,
    pub cross_split_positions: usize,
}
impl Dataset {
    fn insert(&mut self, board: &Board, family: &str, outcome: f64) {
        let key = board.hash();
        if self.banned.contains(&key) {
            return;
        }
        let validation = validation_family(family);
        if let Some(old) = self.samples.get_mut(&key) {
            if old.validation != validation {
                self.samples.remove(&key);
                self.banned.insert(key);
                self.cross_split_positions += 1;
            } else {
                old.outcome_sum += outcome;
                old.count += 1;
            }
        } else {
            self.samples.insert(
                key,
                Sample {
                    fen: board.to_fen(),
                    family: family.into(),
                    validation,
                    outcome_sum: outcome,
                    count: 1,
                },
            );
        }
    }

    pub fn add_log(&mut self, log: &str) -> Result<(), String> {
        // Build a temporary validated batch so a bad game cannot partly emit rows.
        let mut batch = Vec::new();
        let mut excluded = 0;
        let mut games = 0;
        for line in log.lines().filter(|s| s.starts_with("game=")) {
            let (meta, moves) = line.split_once(" moves=").ok_or("missing moves")?;
            let field = |name: &str| meta.split_whitespace().find_map(|s| s.strip_prefix(name));
            let reason = field("reason=").ok_or("missing termination")?;
            let winner = field("winner=").ok_or("missing winner")?;
            let outcome = match winner {
                "Some(White)" => 1.0,
                "Some(Black)" => 0.0,
                "None" => 0.5,
                _ => return Err("invalid winner".into()),
            };
            let opening_plies = field("opening_plies=")
                .unwrap_or("6")
                .parse::<usize>()
                .map_err(|_| "invalid opening length")?;
            let mut board = Board::startpos();
            let mut hashes = vec![board.hash()];
            let mut positions = Vec::new();
            let mut opening_key = None;
            for (i, text) in moves.split_whitespace().enumerate() {
                board = board.make_move(board.parse_uci_move(text)?);
                hashes.push(board.hash());
                let ply = i + 1;
                if ply == opening_plies {
                    opening_key = Some(board.hash());
                }
                if ply >= 16.max(opening_plies + 8)
                    && ply.is_multiple_of(4)
                    && board.halfmove_clock < 90
                    && !board.in_check(board.side_to_move)
                    && !board.is_insufficient_material()
                    && board.legal_tactical_moves().is_empty()
                    && board.has_legal_move()
                {
                    positions.push(board);
                }
            }
            let plies = hashes.len() - 1;
            if field("plies=")
                .ok_or("missing ply count")?
                .parse::<usize>()
                .map_err(|_| "invalid ply count")?
                != plies
            {
                return Err("ply count mismatch".into());
            }
            let terminal = board.legal_moves().is_empty();
            match reason {
                "checkmate"
                    if terminal
                        && board.in_check(board.side_to_move)
                        && ((board.side_to_move == Color::Black && outcome == 1.0)
                            || (board.side_to_move == Color::White && outcome == 0.0)) => {}
                "stalemate"
                    if terminal && !board.in_check(board.side_to_move) && winner == "None" => {}
                "draw_claim"
                    if winner == "None"
                        && (board.halfmove_clock >= 100
                            || board.is_insufficient_material()
                            || hashes.iter().filter(|&&key| key == board.hash()).count() >= 3) => {}
                "ply_limit" if winner == "None" && plies == opening_plies + 300 => {
                    excluded += 1;
                    games += 1;
                    continue;
                }
                _ => return Err("unverified game result".into()),
            }
            let opening_key = opening_key.ok_or("opening longer than game")?;
            if let Some(id) = field("opening_id=")
                && u64::from_str_radix(id, 16).map_err(|_| "invalid opening ID")? != opening_key
            {
                return Err("opening ID mismatch".into());
            }
            let family = field("family=")
                .map(str::to_owned)
                .unwrap_or_else(|| format!("legacy-{opening_key:016x}"));
            if !family
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
            {
                return Err("invalid family ID".into());
            }
            batch.push((family, outcome, positions));
            games += 1;
        }
        if games == 0 {
            return Err("log contains no games".into());
        }
        for (family, outcome, positions) in batch {
            for board in positions {
                self.insert(&board, &family, outcome);
            }
        }
        self.games += games;
        self.excluded_adjudications += excluded;
        Ok(())
    }

    pub fn csv(&self, validation: bool) -> String {
        let mut out = String::from("position_key,fen,opening_family,white_result_mean,samples\n");
        for (key, row) in &self.samples {
            if row.validation == validation {
                out.push_str(&format!(
                    "{key:016x},\"{}\",{},{:.6},{}\n",
                    row.fen,
                    row.family,
                    row.outcome_sum / row.count as f64,
                    row.count
                ));
            }
        }
        out
    }
    pub fn count(&self, validation: bool) -> usize {
        self.samples
            .values()
            .filter(|s| s.validation == validation)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_transpositions_cannot_leak_across_splits() {
        let train = (0..100)
            .map(|i| format!("F{i:02}"))
            .find(|s| !validation_family(s))
            .unwrap();
        let valid = (0..100)
            .map(|i| format!("F{i:02}"))
            .find(|s| validation_family(s))
            .unwrap();
        let board = Board::startpos();
        let mut data = Dataset::default();
        data.insert(&board, &train, 1.0);
        data.insert(&board, &train, 0.0);
        assert_eq!(data.count(false), 1);
        assert!(data.csv(false).contains("0.500000,2"));
        data.insert(&board, &valid, 0.5);
        data.insert(&board, &train, 1.0);
        assert_eq!(data.count(false) + data.count(true), 0);
        assert_eq!(data.cross_split_positions, 1);
    }
    #[test]
    fn invalid_game_results_do_not_partially_modify_dataset() {
        let moves = "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7";
        let good = format!("game=1 winner=Some(White) reason=checkmate plies=7 moves={moves}");
        let bad = format!("game=2 winner=Some(Black) reason=checkmate plies=7 moves={moves}");
        let mut data = Dataset::default();
        assert!(data.add_log(&format!("{good}\n{bad}")).is_err());
        assert_eq!(data.games, 0);
        data.add_log(&good).unwrap();
        assert_eq!(data.games, 1);
    }
}

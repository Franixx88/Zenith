//! Convert validated UCI match logs into portable PGN with standard SAN moves.
use std::fmt::Write as _;
use zenith::board::{Board, Color, Move, Piece};

fn piece_letter(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}

fn san(board: &Board, mv: Move) -> Result<String, String> {
    let legal = board.legal_moves();
    if !legal.contains(&mv) {
        return Err(format!("illegal move {mv}"));
    }
    let (_, piece) = board.piece_at(mv.from).ok_or("empty origin")?;
    let capture =
        board.piece_at(mv.to).is_some() || (piece == Piece::Pawn && mv.from % 8 != mv.to % 8);
    let mut text = String::new();
    if piece == Piece::King && mv.from.abs_diff(mv.to) == 2 {
        text.push_str(if mv.to > mv.from { "O-O" } else { "O-O-O" });
    } else {
        if piece != Piece::Pawn {
            text.push(piece_letter(piece));
            let rivals: Vec<_> = legal
                .iter()
                .filter(|other| {
                    other.from != mv.from
                        && other.to == mv.to
                        && board.piece_at(other.from) == Some((board.side_to_move, piece))
                })
                .collect();
            if !rivals.is_empty() {
                let same_file = rivals.iter().any(|other| other.from % 8 == mv.from % 8);
                let same_rank = rivals.iter().any(|other| other.from / 8 == mv.from / 8);
                if !same_file || same_rank {
                    text.push((b'a' + mv.from % 8) as char);
                }
                if same_file {
                    text.push((b'1' + mv.from / 8) as char);
                }
            }
        } else if capture {
            text.push((b'a' + mv.from % 8) as char);
        }
        if capture {
            text.push('x');
        }
        text.push((b'a' + mv.to % 8) as char);
        text.push((b'1' + mv.to / 8) as char);
        if let Some(promotion) = mv.promotion {
            text.push('=');
            text.push(piece_letter(promotion));
        }
    }
    let next = board.make_move(mv);
    if next.in_check(next.side_to_move) {
        text.push(if next.legal_moves().is_empty() {
            '#'
        } else {
            '+'
        });
    }
    Ok(text)
}

fn tag(out: &mut String, name: &str, value: &str) {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace(['\r', '\n'], " ");
    writeln!(out, "[{name} \"{escaped}\"]").unwrap();
}

fn export(log: &str, a: &str, b: &str, date: &str) -> Result<(String, usize), String> {
    let summary = log
        .lines()
        .find(|line| line.starts_with("A wins="))
        .ok_or("missing limit summary")?;
    let (budget, budget_tag, description) =
        if let Some((_, value)) = summary.split_once("nodes_per_move=") {
            (value.trim(), "NodesPerMove", "node limit")
        } else if let Some((_, value)) = summary.split_once("movetime_ms=") {
            (value.trim(), "MoveTimeMilliseconds", "fixed time per move")
        } else {
            return Err("missing search budget".into());
        };
    if budget.parse::<u64>().map_err(|_| "invalid search budget")? == 0 {
        return Err("zero search budget".into());
    }
    let mut out = String::new();
    let mut count = 0;
    for line in log.lines().filter(|line| line.starts_with("game=")) {
        let (meta, moves) = line.split_once(" moves=").ok_or("missing moves")?;
        let field = |name: &str| {
            meta.split_whitespace()
                .find_map(|word| word.strip_prefix(name))
                .ok_or_else(|| format!("missing {name}"))
        };
        let round = field("game=")?;
        let opening_plies = meta
            .split_whitespace()
            .find_map(|word| word.strip_prefix("opening_plies="))
            .unwrap_or("6")
            .parse::<usize>()
            .map_err(|_| "invalid opening length")?;
        let a_white = field("a_white=")?
            .parse::<bool>()
            .map_err(|_| "invalid colour")?;
        let reason = field("reason=")?;
        let winner = field("winner=")?;
        let result = match winner {
            "Some(White)" => "1-0",
            "Some(Black)" => "0-1",
            "None" => "1/2-1/2",
            _ => return Err("invalid winner".into()),
        };
        let mut board = Board::startpos();
        let mut hashes = vec![board.hash()];
        let mut words = Vec::new();
        for (ply, text) in moves.split_whitespace().enumerate() {
            if ply % 2 == 0 {
                words.push(format!("{}.", ply / 2 + 1));
            }
            let mv = board
                .parse_uci_move(text)
                .map_err(|e| format!("round {round}, ply {}: {e}", ply + 1))?;
            words.push(san(&board, mv)?);
            board = board.make_move(mv);
            hashes.push(board.hash());
        }
        let plies = hashes.len() - 1;
        if plies
            != field("plies=")?
                .parse::<usize>()
                .map_err(|_| "invalid ply count")?
        {
            return Err("ply count mismatch".into());
        }
        let terminal = board.legal_moves().is_empty();
        match reason {
            "checkmate" => {
                let expected = if board.side_to_move == Color::White {
                    "0-1"
                } else {
                    "1-0"
                };
                if !terminal || !board.in_check(board.side_to_move) || expected != result {
                    return Err(format!("round {round}: incorrect mate result"));
                }
            }
            "stalemate" if terminal && !board.in_check(board.side_to_move) && winner == "None" => {}
            "draw_claim"
                if winner == "None"
                    && (board.halfmove_clock >= 100
                        || board.is_insufficient_material()
                        || hashes.iter().filter(|&&key| key == board.hash()).count() >= 3) => {}
            "ply_limit" if winner == "None" && plies == opening_plies + 300 => {}
            _ => return Err(format!("round {round}: unverified termination {reason}")),
        }
        tag(&mut out, "Event", "Zenith paired development match");
        tag(&mut out, "Site", "Local");
        tag(&mut out, "Date", date);
        tag(&mut out, "Round", round);
        tag(&mut out, "White", if a_white { a } else { b });
        tag(&mut out, "Black", if a_white { b } else { a });
        tag(&mut out, "Result", result);
        tag(&mut out, "TimeControl", "-");
        tag(&mut out, budget_tag, budget);
        tag(&mut out, "OpeningPlies", &opening_plies.to_string());
        if let Ok(family) = field("family=") {
            tag(&mut out, "OpeningFamily", family);
        }
        if let Ok(id) = field("opening_id=") {
            tag(&mut out, "OpeningId", id);
        }
        tag(&mut out, "PlyCount", &plies.to_string());
        tag(
            &mut out,
            "Termination",
            if reason == "ply_limit" {
                "adjudication"
            } else {
                "normal"
            },
        );
        tag(&mut out, "TerminationDetails", reason);
        writeln!(
            out,
            "\n{{Fixed opening: first {opening_plies} plies. Engines use a {description}.}}"
        )
        .unwrap();
        words.push(result.into());
        let mut column = 0;
        for word in words {
            if column > 0 && column + 1 + word.len() > 80 {
                out.push('\n');
                column = 0;
            }
            if column > 0 {
                out.push(' ');
                column += 1;
            }
            out.push_str(&word);
            column += word.len();
        }
        out.push_str("\n\n");
        count += 1;
    }
    if count == 0 {
        return Err("log has no games".into());
    }
    Ok((out, count))
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: export_pgn LOG OUTPUT ENGINE_A_NAME ENGINE_B_NAME YYYY.MM.DD".into());
    }
    let log = std::fs::read_to_string(&args[0]).map_err(|e| e.to_string())?;
    let (pgn, count) = export(&log, &args[2], &args[3], &args[4])?;
    std::fs::write(&args[1], pgn).map_err(|e| e.to_string())?;
    println!("Exported and verified {count} games to {}", args[1]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn san_special_cases_and_disambiguation() {
        for (fen, text, expected) in [
            ("4k3/8/8/8/8/8/8/1N3N1K w - - 0 1", "b1d2", "Nbd2"),
            ("7k/8/8/8/8/R7/8/R6K w - - 0 1", "a1a2", "R1a2"),
            ("7k/8/8/8/8/1N6/8/1N3N1K w - - 0 1", "b1d2", "Nb1d2"),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", "a8=Q+"),
            ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", "exd6"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "O-O"),
            ("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", "g6g7", "Qg7#"),
        ] {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(
                san(&board, board.parse_uci_move(text).unwrap()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn rejects_fake_results_and_escapes_headers() {
        let fake = "game=1 a_white=true winner=Some(White) reason=checkmate plies=1 moves=e2e4\nA wins=1 nodes_per_move=5000";
        assert!(export(fake, "A", "B", "2026.10.24").is_err());
        let mut text = String::new();
        tag(&mut text, "White", "A\"\\\nB");
        assert_eq!(text, "[White \"A\\\"\\\\ B\"]\n");
    }

    #[test]
    fn fixed_time_budget_is_exported_without_a_node_tag() {
        let log = "game=1 a_white=true winner=Some(White) reason=checkmate plies=7 moves=e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7\nA wins=1 draws=0 losses=0 movetime_ms=50";
        let (pgn, count) = export(log, "A", "B", "2026.10.24").unwrap();
        assert_eq!(count, 1);
        assert!(pgn.contains("[MoveTimeMilliseconds \"50\"]"));
        assert!(!pgn.contains("NodesPerMove"));
        assert!(pgn.contains("Qxf7#"));
    }
}

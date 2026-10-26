//! Small paired UCI match runner. Intended for smoke matches, not Elo estimates.
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};
use zenith::board::{Board, Color};

struct Engine {
    child: Child,
    input: ChildStdin,
    output: Receiver<String>,
}

impl Engine {
    fn open(path: &str) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("launch {path}: {e}"))?;
        let input = child.stdin.take().ok_or("missing stdin")?;
        let stdout = child.stdout.take().ok_or("missing stdout")?;
        let (sender, output) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });
        let mut engine = Self {
            child,
            input,
            output,
        };
        engine.send("uci")?;
        engine.wait("uciok")?;
        engine.send("setoption name Hash value 16")?;
        engine.ready()?;
        Ok(engine)
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.input, "{line}")
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())
    }

    fn wait(&self, prefix: &str) -> Result<String, String> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let line = self
                .output
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|e| format!("waiting for {prefix}: {e}"))?;
            if line == prefix || line.starts_with(&format!("{prefix} ")) {
                return Ok(line);
            }
            if Instant::now() >= deadline {
                return Err(format!("timeout waiting for {prefix}"));
            }
        }
    }

    fn ready(&mut self) -> Result<(), String> {
        self.send("isready")?;
        self.wait("readyok")?;
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut threads = [None, None];
    for (index, flag) in ["--a-threads=", "--b-threads="].into_iter().enumerate() {
        if let Some(value) = args.iter().find_map(|s| s.strip_prefix(flag)) {
            let count = value.parse::<usize>().map_err(|e| e.to_string())?;
            if !(1..=64).contains(&count) {
                return Err("threads must be 1..64".into());
            }
            threads[index] = Some(count);
        }
    }
    args.retain(|s| !s.starts_with("--a-threads=") && !s.starts_with("--b-threads="));
    let disable_a_enhancements = args.iter().any(|s| s == "--a-no-enhancements");
    let a_modern = args.iter().any(|s| s == "--a-modern");
    let b_modern = args.iter().any(|s| s == "--b-modern");
    let a_classic = args.iter().any(|s| s == "--a-classic");
    let b_classic = args.iter().any(|s| s == "--b-classic");
    if (a_modern && a_classic) || (b_modern && b_classic) {
        return Err("choose one search mode per engine".into());
    }
    args.retain(|s| s != "--a-classic" && s != "--b-classic");
    args.retain(|s| s != "--a-modern" && s != "--b-modern");
    args.retain(|s| s != "--a-no-enhancements");
    if !(2..=7).contains(&args.len()) {
        return Err(
            "usage: match ENGINE_A ENGINE_B [PAIRS=4] [BUDGET=5000] [nodes|movetime] [BOOK] [OFFSET=0]".into(),
        );
    }
    let pairs = args
        .get(2)
        .map_or(Ok(4), |v| v.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let nodes = args
        .get(3)
        .map_or(Ok(5000), |v| v.parse::<u64>())
        .map_err(|e| e.to_string())?;
    let mode = args.get(4).map_or("nodes", String::as_str);
    if !matches!(mode, "nodes" | "movetime") {
        return Err("mode must be nodes or movetime".into());
    }
    if pairs == 0 || nodes == 0 {
        return Err("pairs and nodes must be positive".into());
    }
    let openings = if let Some(path) = args.get(5) {
        zenith::openings::parse(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)?
    } else {
        zenith::openings::test_book()
    };
    let offset = args
        .get(6)
        .map_or(Ok(0), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let mut engines = [Engine::open(&args[0])?, Engine::open(&args[1])?];
    for (index, count) in threads.into_iter().enumerate() {
        if let Some(count) = count {
            engines[index].send(&format!("setoption name Threads value {count}"))?;
            engines[index].ready()?;
        }
    }
    for (index, enabled) in [a_classic, b_classic].into_iter().enumerate() {
        if enabled {
            engines[index].send("setoption name Modern Search value false")?;
            engines[index].send("setoption name Null Move Pruning value false")?;
            engines[index].ready()?;
        }
    }
    for (index, enabled) in [a_modern, b_modern].into_iter().enumerate() {
        if enabled {
            engines[index].send("setoption name Modern Search value true")?;
            engines[index].send("setoption name Null Move Pruning value true")?;
            engines[index].ready()?;
        }
    }
    if disable_a_enhancements {
        engines[0].send("setoption name Search Enhancements value false")?;
        engines[0].ready()?;
    }
    let (mut wins, mut draws, mut losses, mut capped) = (0, 0, 0, 0);
    for game in 0..pairs.checked_mul(2).ok_or("too many pairs")? {
        let a_white = game % 2 == 0;
        let opening = &openings[(game / 2 + offset % openings.len()) % openings.len()];
        let opening_plies = opening.moves.len();
        let mut moves = opening.moves.clone();
        let mut board = Board::startpos();
        let mut hashes = vec![board.hash()];
        for mv in &moves {
            board = board.make_move(board.parse_uci_move(mv)?);
            hashes.push(board.hash());
        }
        for engine in &mut engines {
            engine.send("ucinewgame")?;
            engine.ready()?;
        }
        let mut winner = None;
        let mut reason = "ply_limit";
        for ply in 0..=300 {
            let legal = board.legal_moves();
            if legal.is_empty() {
                reason = if board.in_check(board.side_to_move) {
                    winner = Some(board.side_to_move.other());
                    "checkmate"
                } else {
                    "stalemate"
                };
                break;
            }
            if board.halfmove_clock >= 100
                || board.is_insufficient_material()
                || hashes.iter().filter(|&&h| h == board.hash()).count() >= 3
            {
                reason = "draw_claim";
                break;
            }
            if ply == 300 {
                break;
            }
            let a_turn = (board.side_to_move == Color::White) == a_white;
            let engine = &mut engines[usize::from(!a_turn)];
            engine.send(&format!("position startpos moves {}", moves.join(" ")))?;
            engine.send(&format!("go {mode} {nodes}"))?;
            let reply = engine.wait("bestmove")?;
            let text = reply.split_whitespace().nth(1).ok_or("missing bestmove")?;
            let mv = board
                .parse_uci_move(text)
                .map_err(|e| format!("game {}: {e}", game + 1))?;
            board = board.make_move(mv);
            hashes.push(board.hash());
            moves.push(text.to_owned());
        }
        match winner {
            Some(color) if (color == Color::White) == a_white => wins += 1,
            Some(_) => losses += 1,
            None => draws += 1,
        }
        capped += usize::from(reason == "ply_limit");
        println!(
            "game={} a_white={a_white} winner={winner:?} reason={reason} family={} opening_id={:016x} opening_plies={opening_plies} plies={} moves={}",
            game + 1,
            opening.family,
            opening.key,
            moves.len(),
            moves.join(" ")
        );
    }
    let budget_name = if mode == "nodes" {
        "nodes_per_move"
    } else {
        "movetime_ms"
    };
    println!(
        "A wins={wins} draws={draws} losses={losses} ply_limit_draws={capped} {budget_name}={nodes}"
    );
    Ok(())
}

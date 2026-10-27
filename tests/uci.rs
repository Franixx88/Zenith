//! Process-level UCI tests. Synchronization is based on actual protocol replies.
//!
//! Original April 2006 specification, faithfully reproduced at:
//! https://backscattering.de/chess/uci/

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use zenith::board::Board;

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);

struct NeuralFiles {
    model: std::path::PathBuf,
    corrupt: std::path::PathBuf,
}
impl NeuralFiles {
    fn new() -> Self {
        Self::with_buckets(1)
    }
    fn with_buckets(buckets: usize) -> Self {
        use zenith::{
            board::{Color, Piece},
            nnue::{self, Network, Parameters},
        };
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let model = std::env::temp_dir().join(format!("zenith NNUE  test {unique}.znnue"));
        let corrupt = model.with_extension("corrupt.znnue");
        // Synthetic positional fixture: at depth 1 only e2e4 activates the
        // rewarded feature. This proves search uses the loaded weights.
        let mut p = Parameters::default();
        p.feature_weights
            .resize(buckets * nnue::INPUTS * nnue::HIDDEN, 0);
        for bucket in 0..buckets {
            p.feature_weights[(bucket * nnue::INPUTS
                + nnue::feature_index(Color::White, Color::White, Piece::Pawn, 28))
                * nnue::HIDDEN] = 200;
        }
        p.output_weights[0] = 64;
        p.output_weights[nnue::HIDDEN] = -64;
        let network = if buckets == 16 {
            let mut p = nnue::v3::Parameters::default();
            p.feature_bias.fill(128);
            for (i, w) in p.feature_weights.iter_mut().enumerate() {
                *w = (i % 7) as i16 - 3;
            }
            for head in &mut p.heads {
                head.l1_weights[0] = 64;
                head.l2_weights[0] = 64;
                head.output_weights[0] = 64;
            }
            Network::new_v3(p).unwrap()
        } else {
            Network::new(p).unwrap()
        };
        network.save(&model).unwrap();
        let mut bytes = network.to_bytes();
        bytes[64] ^= 1;
        std::fs::write(&corrupt, bytes).unwrap();
        Self { model, corrupt }
    }
}
impl Drop for NeuralFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.model);
        let _ = std::fs::remove_file(&self.corrupt);
    }
}

struct Engine {
    child: Child,
    stdin: Option<ChildStdin>,
    output: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    transcript: Vec<String>,
}

impl Engine {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_zenith"))
            // Protocol regressions must not depend on locally installed tables.
            .env("ZENITH_SYZYGY_PATH", "<empty>")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("launch zenith binary");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let (sender, output) = mpsc::channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            child,
            stdin,
            output,
            reader: Some(reader),
            transcript: Vec::new(),
        }
    }

    fn send(&mut self, command: &str) {
        self.transcript.push(format!("> {command}"));
        let stdin = self.stdin.as_mut().expect("stdin is open");
        writeln!(stdin, "{command}").expect("send UCI command");
        stdin.flush().expect("flush UCI command");
    }

    fn wait_for(&mut self, description: &str, matches: impl Fn(&str) -> bool) -> String {
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {description}\n{}",
                self.transcript.join("\n")
            );
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = self.output.recv_timeout(remaining).unwrap_or_else(|error| {
                panic!(
                    "waiting for {description}: {error}\n{}",
                    self.transcript.join("\n")
                )
            });
            self.transcript.push(format!("< {line}"));
            if matches(&line) {
                return line;
            }
        }
    }

    fn assert_no_bestmove_for(&mut self, duration: Duration) {
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.output.recv_timeout(remaining) {
                Ok(line) => {
                    self.transcript.push(format!("< {line}"));
                    assert!(
                        !line.starts_with("bestmove "),
                        "infinite search completed before stop\n{}",
                        self.transcript.join("\n")
                    );
                }
                Err(RecvTimeoutError::Timeout) => return,
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("engine exited while an infinite search was active");
                }
            }
        }
    }

    fn handshake(&mut self) {
        self.send("uci");
        self.wait_for("uciok", |line| line == "uciok");
        assert!(
            self.transcript
                .iter()
                .any(|line| line.starts_with("< id name ")),
            "engine must identify itself: {:?}",
            self.transcript
        );
        self.ready();
    }

    fn ready(&mut self) {
        self.send("isready");
        self.wait_for("readyok", |line| line == "readyok");
    }

    fn bestmove(&mut self) -> String {
        let line = self.wait_for("bestmove", |line| line.starts_with("bestmove "));
        line.split_whitespace().nth(1).unwrap().to_owned()
    }

    fn assert_legal_bestmove(&mut self, board: &Board) -> String {
        let bestmove = self.bestmove();
        assert!(
            board.parse_uci_move(&bestmove).is_ok(),
            "illegal bestmove {bestmove} in {}\n{}",
            board.to_fen(),
            self.transcript.join("\n")
        );
        bestmove
    }

    fn wait_for_exit(&mut self) {
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().expect("read child exit status") {
                assert!(status.success(), "engine exited with {status}");
                return;
            }
            assert!(
                Instant::now() < deadline,
                "engine failed to exit\n{}",
                self.transcript.join("\n")
            );
            // This polls process termination, never assumes search has progressed.
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn quit(&mut self) {
        self.send("quit");
        self.wait_for_exit();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[test]
fn handshake_options_and_new_game() {
    let mut engine = Engine::spawn();
    engine.handshake();
    assert!(
        engine
            .transcript
            .iter()
            .any(|line| line == "< option name Threads type spin default 8 min 1 max 64")
    );
    assert!(
        engine
            .transcript
            .iter()
            .any(|line| { line.starts_with("< option name Hash type spin ") })
    );
    engine.send("setoption name Hash value 1");
    engine.send("ucinewgame");
    engine.ready();
    engine.send("unknown_command");
    engine.send("stop");
    engine.ready();
    engine.quit();
}

#[test]
fn depth_search_respects_startpos_move_history() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position startpos moves e2e4 e7e5 g1f3");
    engine.send("go depth 3");
    let board =
        Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2").unwrap();
    engine.assert_legal_bestmove(&board);
    assert!(
        engine
            .transcript
            .iter()
            .any(|line| { line.starts_with("< info depth 3 ") })
    );
    engine.quit();
}

#[test]
fn fen_position_accepts_promotion_move_history() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position fen 7k/P7/8/8/8/8/8/7K w - - 0 1 moves a7a8q");
    engine.send("go depth 2");
    let board = Board::from_fen("Q6k/8/8/8/8/8/8/7K b - - 0 1").unwrap();
    engine.assert_legal_bestmove(&board);
    engine.quit();
}

#[test]
fn infinite_search_answers_isready_and_stop() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position startpos");
    engine.send("go infinite");
    engine.wait_for("search progress", |line| line.starts_with("info depth "));
    let start = engine.transcript.len();
    engine.ready();
    assert!(
        !engine.transcript[start..]
            .iter()
            .any(|line| line.starts_with("< bestmove ")),
        "isready must not stop an infinite search"
    );
    engine.send("stop");
    engine.assert_legal_bestmove(&Board::startpos());
    engine.ready();
    assert_eq!(
        engine
            .transcript
            .iter()
            .filter(|line| line.starts_with("< bestmove "))
            .count(),
        1,
        "each search must emit exactly one bestmove"
    );
    engine.quit();
}

#[test]
fn immediate_stop_still_returns_a_legal_move() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position startpos");
    engine.send("go infinite");
    engine.send("stop");
    engine.assert_legal_bestmove(&Board::startpos());
    engine.quit();
}

#[test]
fn node_and_clock_limits_finish_without_stop() {
    let mut engine = Engine::spawn();
    engine.handshake();
    for go in [
        "go nodes 500",
        "go movetime 30",
        "go wtime 300 btime 300 winc 1 binc 1 movestogo 20",
    ] {
        engine.send("position startpos");
        engine.send(go);
        engine.assert_legal_bestmove(&Board::startpos());
        engine.ready();
    }
    engine.quit();
}

#[test]
fn mate_and_stalemate_report_null_move() {
    let mut engine = Engine::spawn();
    engine.handshake();
    for fen in [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    ] {
        engine.send(&format!("position fen {fen}"));
        engine.send("go depth 2");
        assert_eq!(engine.bestmove(), "0000", "terminal position: {fen}");
    }
    engine.quit();
}

#[test]
fn restricted_root_search_plays_only_the_requested_move() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position startpos");
    engine.send("go depth 2 searchmoves e2e4");
    assert_eq!(engine.bestmove(), "e2e4");
    engine.quit();
}

#[test]
fn malformed_position_does_not_partially_apply_moves() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position startpos moves d2d4");
    engine.ready();
    engine.send("d");
    let original = engine.wait_for("position display", |line| {
        line.starts_with("info string fen ")
    });
    engine.send("position startpos moves e2e4 e7e5 g1g5");
    engine.wait_for("invalid position diagnostic", |line| {
        line.starts_with("info string error:")
    });
    engine.send("d");
    let after_error = engine.wait_for("position display after invalid input", |line| {
        line.starts_with("info string fen ")
    });
    assert_eq!(after_error, original);
    engine.quit();
}

#[test]
fn infinite_search_in_a_terminal_position_waits_for_stop() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    engine.send("go infinite");
    engine.ready();
    assert!(
        !engine
            .transcript
            .iter()
            .any(|line| line.starts_with("< bestmove ")),
        "even a terminal infinite search must wait for stop"
    );
    // Verify intentional silence after synchronization; this is not a readiness sleep.
    engine.assert_no_bestmove_for(Duration::from_millis(200));
    engine.send("stop");
    assert_eq!(engine.bestmove(), "0000");
    engine.ready();
    assert_eq!(
        engine
            .transcript
            .iter()
            .filter(|line| line.starts_with("< bestmove "))
            .count(),
        1
    );
    engine.quit();
}

#[test]
fn eof_during_search_exits_cleanly() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("setoption name Threads value 4");
    engine.send("position startpos");
    engine.send("go infinite");
    engine.wait_for("search progress", |line| line.starts_with("info depth "));
    engine.stdin.take();
    engine.wait_for_exit();
}

#[test]
fn quit_during_search_exits_cleanly() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("setoption name Threads value 4");
    engine.send("position startpos");
    engine.send("go infinite");
    engine.wait_for("search progress", |line| line.starts_with("info depth "));
    engine.quit();
}

#[test]
fn smp_stop_reconfigure_and_global_node_budget() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("setoption name Singular Extensions value true");
    engine.send("setoption name SMP Voting value true");
    engine.send("setoption name LMR Divisor value 275");
    engine.send("setoption name Threads value 8");
    engine.send("position startpos");
    engine.send("go infinite");
    engine.wait_for("parallel progress", |line| line.starts_with("info depth "));
    engine.ready();
    engine.send("stop");
    let mv = engine.bestmove();
    assert!(Board::startpos().parse_uci_move(&mv).is_ok());
    engine.send("setoption name Threads value 2");
    engine.send("setoption name Hash value 1");
    engine.send("go nodes 5000");
    let complete = engine.wait_for("summed node budget", |line| {
        line.starts_with("info string search complete")
    });
    assert!(complete.contains("nodes 5000 "), "{complete}");
    assert!(complete.contains("helper_failures 0"));
    engine.bestmove();
    engine.send("setoption name Threads value 1");
    engine.send("go depth 2 searchmoves e2e4");
    assert_eq!(engine.bestmove(), "e2e4");
    engine.quit();
}

#[test]
fn multipv_protocol_reports_unique_lines_and_responds_to_stop() {
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("setoption name Threads value 2");
    engine.send("setoption name Hash value 2");
    engine.send("setoption name MultiPV value 3");
    engine.send("position startpos");
    engine.send("go depth 4");
    engine.wait_for("MultiPV search completion", |line| {
        line.starts_with("info string search complete")
    });
    let mut lines = std::collections::BTreeMap::new();
    for line in &engine.transcript {
        let words: Vec<_> = line.split_whitespace().collect();
        if let Some(index) = words.iter().position(|&word| word == "multipv") {
            let rank: usize = words[index + 1].parse().unwrap();
            let depth: u8 = words[3].parse().unwrap(); // '< info depth N ...'
            let pv = words.iter().position(|&word| word == "pv").unwrap();
            lines.insert(rank, (depth, words[pv + 1].to_owned()));
        }
    }
    assert_eq!(lines.len(), 3);
    assert!(lines.values().all(|line| line.0 == lines[&1].0));
    let unique: std::collections::HashSet<_> = lines.values().map(|line| &line.1).collect();
    assert_eq!(unique.len(), 3);
    assert_eq!(engine.bestmove(), lines[&1].1);
    engine.send("go infinite");
    engine.wait_for("third PV", |line| {
        line.starts_with("info depth ") && line.contains("multipv 3 ")
    });
    engine.ready();
    engine.send("stop");
    engine.assert_legal_bestmove(&Board::startpos());
    engine.send("setoption name MultiPV value 1");
    engine.send("go depth 2 searchmoves d2d4");
    assert_eq!(engine.bestmove(), "d2d4");
    engine.quit();
}

#[test]
fn nnue_file_loading_search_switches_and_hash_resets_are_atomic() {
    let files = NeuralFiles::new();
    let mut engine = Engine::spawn();
    engine.handshake();
    assert!(
        engine
            .transcript
            .iter()
            .any(|line| line == "< option name EvalFile type string default <empty>")
    );
    assert!(
        engine
            .transcript
            .iter()
            .any(|line| line == "< option name Use NNUE type check default true")
    );
    engine.send("setoption name Threads value 1");
    engine.send(&format!(
        "setoption name EvalFile value {}",
        files.model.display()
    ));
    engine.wait_for("loaded NNUE status", |line| {
        line.starts_with("info string evaluation NNUE v1")
    });
    for command in ["position startpos", "go depth 1"] {
        engine.send(command);
    }
    let completed = engine.wait_for("neural search", |line| {
        line.starts_with("info string search complete")
    });
    assert!(completed.contains("nnue_refreshes 1") && !completed.contains("nnue_updates 0"));
    assert_eq!(engine.bestmove(), "e2e4");
    // Keep the validated allocation alive across fresh Search construction;
    // Hash/newgame must not reopen or silently lose the model file.
    std::fs::remove_file(&files.model).unwrap();
    engine.send("setoption name Hash value 1");
    engine.send("ucinewgame");
    engine.send("go depth 1");
    assert_eq!(engine.bestmove(), "e2e4");
    for path in [&files.model, &files.corrupt] {
        engine.send(&format!("setoption name EvalFile value {}", path.display()));
        engine.wait_for("rejected NNUE file", |line| {
            line.starts_with("info string error: NNUE")
        });
        engine.send("ucinewgame");
        engine.send("go depth 1");
        assert_eq!(
            engine.bestmove(),
            "e2e4",
            "bad file must preserve the loaded model"
        );
    }
    engine.send("setoption name Use NNUE value false");
    engine.send("go depth 1");
    let completed = engine.wait_for("classical search", |line| {
        line.starts_with("info string search complete")
    });
    assert!(completed.contains("nnue_evaluations 0"));
    let _ = engine.bestmove();
    engine.send("setoption name Use NNUE value true");
    engine.send("go depth 1");
    assert_eq!(engine.bestmove(), "e2e4");
    engine.send("setoption name EvalFile value <empty>");
    engine.send("go depth 1");
    let completed = engine.wait_for("unloaded neural search", |line| {
        line.starts_with("info string search complete")
    });
    assert!(completed.contains("nnue_evaluations 0"));
    let _ = engine.bestmove();
    engine.quit();
}

#[test]
fn nnue_smp_multipv_and_failed_load_during_search_remain_responsive() {
    check_neural_smp(1);
    check_neural_smp(4);
    check_neural_smp(16);
}

fn check_neural_smp(buckets: usize) {
    let files = NeuralFiles::with_buckets(buckets);
    let mut engine = Engine::spawn();
    engine.handshake();
    engine.send("setoption name Threads value 8");
    engine.send("setoption name Hash value 2");
    engine.send("setoption name MultiPV value 3");
    engine.send(&format!(
        "setoption name EvalFile value {}",
        files.model.display()
    ));
    engine.send("go depth 3");
    let completed = engine.wait_for("SMP neural search", |line| {
        line.starts_with("info string search complete")
    });
    assert!(completed.contains("helper_failures 0") && !completed.contains("nnue_updates 0"));
    let mut roots = std::collections::BTreeMap::new();
    for line in &engine.transcript {
        let words: Vec<_> = line.split_whitespace().collect();
        if words.get(1..4) == Some(&["info", "depth", "3"]) {
            let rank = words.iter().position(|&word| word == "multipv").unwrap();
            let pv = words.iter().position(|&word| word == "pv").unwrap();
            roots.insert(words[rank + 1].to_owned(), words[pv + 1].to_owned());
            let mut board = Board::startpos();
            for token in &words[pv + 1..] {
                board = board.make_move(board.parse_uci_move(token).unwrap());
            }
        }
    }
    assert_eq!(roots.len(), 3);
    assert_eq!(
        roots
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert_eq!(engine.bestmove(), roots["1"]);
    engine.send("go infinite");
    engine.wait_for("active neural iteration", |line| {
        line.starts_with("info depth ")
    });
    engine.send(&format!(
        "setoption name EvalFile value {}",
        files.corrupt.display()
    ));
    engine.wait_for("bad load while searching", |line| {
        line.starts_with("info string error: NNUE")
    });
    engine.ready();
    engine.assert_no_bestmove_for(Duration::from_millis(50));
    engine.send("stop");
    let best = engine.bestmove();
    assert!(Board::startpos().parse_uci_move(&best).is_ok());
    engine.send("go nodes 1000");
    let completed = engine.wait_for("neural node budget", |line| {
        line.starts_with("info string search complete")
    });
    let words: Vec<_> = completed.split_whitespace().collect();
    let nodes = words.iter().position(|&word| word == "nodes").unwrap();
    assert!(words[nodes + 1].parse::<u64>().unwrap() <= 1000);
    let _ = engine.bestmove();
    engine.send("setoption name Threads value 1");
    engine.send("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    engine.send("go depth 2");
    let completed = engine.wait_for("king-bank crossing search", |line| {
        line.starts_with("info string search complete")
    });
    let words: Vec<_> = completed.split_whitespace().collect();
    let count = words
        .iter()
        .position(|&word| word == "nnue_bucket_refreshes")
        .unwrap();
    assert_eq!(words[count + 1].parse::<u64>().unwrap() > 0, buckets > 1);
    let _ = engine.bestmove();
    engine.quit();
}

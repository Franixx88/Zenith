//! Line-oriented UCI with a separate, cancellable search worker.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::{Board, Color};
use crate::search::{
    DEFAULT_MODERN_SEARCH, DEFAULT_NULL_PRUNING, DEFAULT_SINGULAR, DEFAULT_SMP_VOTING, Limits,
    MATE_SCORE, MAX_MULTI_PV, Search, SearchInfo, SearchParameters, is_mate_score,
};

const DEFAULT_HASH_MB: usize = 16;
const DEFAULT_THREADS: usize = 8;

fn emit(message: &str) {
    let mut output = io::stdout().lock();
    let _ = writeln!(output, "{message}");
    let _ = output.flush();
}

fn info(info: SearchInfo) {
    let score = if is_mate_score(info.score) {
        let moves = (MATE_SCORE - info.score.abs() + 1) / 2;
        format!("mate {}", moves * info.score.signum())
    } else {
        format!("cp {}", info.score)
    };
    let milliseconds = info.elapsed.as_millis();
    let nps = (u128::from(info.nodes) * 1000 / milliseconds.max(1)).min(u128::from(u64::MAX));
    let pv = info
        .pv
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    emit(&format!(
        "info depth {} multipv {} score {score} nodes {} time {milliseconds} nps {nps} tbhits {} pv {pv}",
        info.depth, info.multipv, info.nodes, info.tbhits
    ));
}

struct Engine {
    network: Option<Arc<crate::nnue::Network>>,
    eval_file: String,
    use_nnue: bool,
    board: Board,
    history: Vec<u64>,
    search: Option<Search>,
    worker: Option<JoinHandle<Search>>,
    stop: Arc<AtomicBool>,
    hash_mb: usize,
    overhead_ms: u64,
    null_pruning: bool,
    pruning: bool,
    enhancements: bool,
    modern: bool,
    singular: bool,
    smp_voting: bool,
    parameters: SearchParameters,
    threads: usize,
    multi_pv: usize,
    syzygy: Option<Arc<crate::syzygy::Syzygy>>,
    default_syzygy_path: String,
    probe_limit: usize,
    probe_depth: i16,
}

impl Engine {
    fn new() -> Self {
        let board = Board::startpos();
        let mut default_syzygy_path = crate::syzygy::Syzygy::default_path();
        let syzygy = match crate::syzygy::Syzygy::load(&default_syzygy_path) {
            Ok(tables) => tables,
            Err(error) => {
                emit(&format!(
                    "info string automatic Syzygy load failed: {error}"
                ));
                default_syzygy_path = "<empty>".into();
                None
            }
        };
        let mut search = Search::new(DEFAULT_HASH_MB);
        search.set_threads(DEFAULT_THREADS);
        search.set_syzygy(syzygy.clone(), 7, 1);
        Self {
            network: None,
            eval_file: "<empty>".into(),
            use_nnue: true,
            board,
            history: vec![board.hash()],
            search: Some(search),
            worker: None,
            stop: Arc::new(AtomicBool::new(false)),
            hash_mb: DEFAULT_HASH_MB,
            overhead_ms: 10,
            null_pruning: DEFAULT_NULL_PRUNING,
            pruning: true,
            enhancements: true,
            modern: DEFAULT_MODERN_SEARCH,
            singular: DEFAULT_SINGULAR,
            smp_voting: DEFAULT_SMP_VOTING,
            parameters: SearchParameters::default(),
            threads: DEFAULT_THREADS,
            multi_pv: 1,
            syzygy,
            default_syzygy_path,
            probe_limit: 7,
            probe_depth: 1,
        }
    }

    fn fresh_search(&self) -> Search {
        let mut search = Search::new(self.hash_mb);
        search.set_null_pruning(self.null_pruning);
        search.set_pruning(self.pruning);
        search.set_enhancements(self.enhancements);
        search.set_modern_search(self.modern);
        search.set_singular_extensions(self.singular);
        search.set_smp_voting(self.smp_voting);
        search
            .set_parameters(self.parameters)
            .expect("validated parameters");
        search.set_threads(self.threads);
        search.set_multi_pv(self.multi_pv);
        search.set_syzygy(self.syzygy.clone(), self.probe_limit, self.probe_depth);
        search.set_network(self.active_network());
        search
    }

    fn active_network(&self) -> Option<Arc<crate::nnue::Network>> {
        if self.use_nnue {
            self.network.clone()
        } else {
            None
        }
    }

    fn evaluation_status(&self) {
        if let Some(network) = self.active_network() {
            emit(&format!(
                "info string evaluation NNUE v{} backend {:?} EvalFile {}",
                network.format_version(),
                network.backend(),
                self.eval_file
            ));
        } else {
            emit("info string evaluation classical");
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            self.search = Some(match worker.join() {
                Ok(search) => search,
                Err(_) => {
                    emit("info string search worker failed; resetting search state");
                    self.fresh_search()
                }
            });
        }
    }

    fn set_position(&mut self, tokens: &[&str]) -> Result<(), String> {
        let (board, history) = parse_position(tokens)?;
        self.stop_search();
        self.board = board;
        self.history = history;
        Ok(())
    }

    fn set_option(&mut self, tokens: &[&str], line: &str) -> Result<(), String> {
        if tokens.first() != Some(&"name") {
            return Err("setoption expects name".into());
        }
        let split = tokens.iter().position(|token| *token == "value");
        let name = tokens[1..split.unwrap_or(tokens.len())]
            .join(" ")
            .to_ascii_lowercase();
        // Keep interior whitespace in file names. Token joining would turn
        // e.g. "my  network.znnue" into a different path.
        let value = split
            .map(|index| {
                let mut rest = line;
                for _ in 0..index + 2 {
                    rest = rest.trim_start();
                    rest = rest
                        .find(char::is_whitespace)
                        .map_or("", |offset| &rest[offset..]);
                }
                rest.trim().to_owned()
            })
            .unwrap_or_default();
        match name.as_str() {
            "evalfile" => {
                // Validate and read BEFORE touching the current search/model.
                // A missing, corrupt or incompatible file is an atomic failure.
                let network = if value.is_empty() || value.eq_ignore_ascii_case("<empty>") {
                    None
                } else {
                    Some(Arc::new(crate::nnue::Network::load(&value)?))
                };
                self.stop_search();
                self.eval_file = if network.is_some() {
                    value
                } else {
                    "<empty>".into()
                };
                self.network = network;
                let active = self.active_network();
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_network(active);
                self.evaluation_status();
            }
            "use nnue" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "Use NNUE must be true or false")?;
                self.stop_search();
                self.use_nnue = enabled;
                let active = self.active_network();
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_network(active);
                self.evaluation_status();
            }
            "multipv" => {
                let count = value
                    .parse::<usize>()
                    .map_err(|_| "MultiPV must be 1..32")?;
                if !(1..=MAX_MULTI_PV).contains(&count) {
                    return Err("MultiPV must be 1..32".into());
                }
                self.stop_search();
                self.multi_pv = count;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_multi_pv(count);
            }
            "threads" => {
                let threads = value
                    .parse::<usize>()
                    .map_err(|_| "Threads must be from 1 to 64")?;
                if !(1..=64).contains(&threads) {
                    return Err("Threads must be from 1 to 64".into());
                }
                self.stop_search();
                self.threads = threads;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_threads(threads);
            }
            "syzygypath" => {
                self.stop_search();
                let tables = crate::syzygy::Syzygy::load(&value)?;
                if let Some(tb) = &tables {
                    emit(&format!(
                        "info string Syzygy registered {} files, up to {} pieces",
                        tb.files(),
                        tb.max_pieces()
                    ));
                } else {
                    emit("info string Syzygy disabled");
                }
                self.syzygy = tables;
                self.search.as_mut().expect("idle search").set_syzygy(
                    self.syzygy.clone(),
                    self.probe_limit,
                    self.probe_depth,
                );
            }
            "syzygyprobelimit" | "syzygyprobedepth" => {
                let number = value
                    .parse::<usize>()
                    .map_err(|_| "Syzygy probe option requires an integer")?;
                if (name == "syzygyprobelimit" && number > 7)
                    || (name == "syzygyprobedepth" && !(1..=120).contains(&number))
                {
                    return Err(
                        "SyzygyProbeLimit must be 0..7; SyzygyProbeDepth must be 1..120".into(),
                    );
                }
                self.stop_search();
                if name == "syzygyprobelimit" {
                    self.probe_limit = number;
                } else {
                    self.probe_depth = number as i16;
                }
                self.search.as_mut().expect("idle search").set_syzygy(
                    self.syzygy.clone(),
                    self.probe_limit,
                    self.probe_depth,
                );
            }
            "hash" => {
                let size = value
                    .parse::<usize>()
                    .map_err(|_| "Hash must be an integer from 1 to 1024")?;
                if !(1..=1024).contains(&size) {
                    return Err("Hash must be an integer from 1 to 1024".into());
                }
                self.stop_search();
                // Release the previous allocation before allocating its replacement.
                self.search = None;
                self.hash_mb = size;
                self.search = Some(self.fresh_search());
            }
            "clear hash" => {
                self.stop_search();
                self.search.as_mut().expect("idle search").clear();
            }
            "move overhead" => {
                let overhead = value
                    .parse::<u64>()
                    .map_err(|_| "Move Overhead must be an integer from 0 to 5000")?;
                if overhead > 5000 {
                    return Err("Move Overhead must be an integer from 0 to 5000".into());
                }
                self.overhead_ms = overhead;
            }
            "null move pruning" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "Null Move Pruning must be true or false")?;
                self.stop_search();
                self.null_pruning = enabled;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_null_pruning(enabled);
            }
            "selective pruning" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "Selective Pruning must be true or false")?;
                self.stop_search();
                self.pruning = enabled;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_pruning(enabled);
            }
            "search enhancements" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "Search Enhancements must be true or false")?;
                self.stop_search();
                self.enhancements = enabled;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_enhancements(enabled);
            }
            "modern search" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "Modern Search must be true or false")?;
                self.stop_search();
                self.modern = enabled;
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_modern_search(enabled);
            }
            "singular extensions" | "smp voting" => {
                let enabled = value
                    .to_ascii_lowercase()
                    .parse::<bool>()
                    .map_err(|_| "search switch must be true or false")?;
                self.stop_search();
                if name == "singular extensions" {
                    self.singular = enabled;
                    self.search
                        .as_mut()
                        .expect("idle search")
                        .set_singular_extensions(enabled);
                } else {
                    self.smp_voting = enabled;
                    self.search
                        .as_mut()
                        .expect("idle search")
                        .set_smp_voting(enabled);
                }
            }
            "rfp margin" | "futility margin" | "nmp base reduction" | "lmr divisor" => {
                let number = value
                    .parse::<u16>()
                    .map_err(|_| "search parameter requires a nonnegative integer")?;
                let mut parameters = self.parameters;
                match name.as_str() {
                    "rfp margin" => parameters.rfp_margin = i32::from(number),
                    "futility margin" => parameters.futility_margin = i32::from(number),
                    "nmp base reduction" => {
                        parameters.nmp_base_reduction =
                            i16::try_from(number).map_err(|_| "NMP Base Reduction must be 2..5")?
                    }
                    _ => parameters.lmr_divisor = number,
                }
                parameters.validate()?;
                self.stop_search();
                self.search
                    .as_mut()
                    .expect("idle search")
                    .set_parameters(parameters)?;
                self.parameters = parameters;
            }
            _ => return Err(format!("unknown option: {name}")),
        }
        Ok(())
    }

    fn go(&mut self, tokens: &[&str]) -> Result<(), String> {
        let limits = parse_go(tokens, &self.board, self.overhead_ms)?;
        self.stop_search();
        self.evaluation_status();
        self.stop = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&self.stop);
        let board = self.board;
        let history = self.history.clone();
        let mut search = self.search.take().expect("idle search");
        let worker = thread::Builder::new()
            .name("zenith-search".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let infinite = limits.infinite;
                let start = std::time::Instant::now();
                let result = search.search(board, &history, limits, Arc::clone(&stop), info);
                if result.variations.is_empty() { info(SearchInfo { multipv: 1, depth: result.depth, score: result.score,
                    nodes: result.nodes, elapsed: start.elapsed(), pv: result.pv.clone(),
                    tbhits: result.stats.tb_hits }); } else {
                    let elapsed = start.elapsed();
                    for (rank, line) in result.variations.iter().enumerate() {
                        info(SearchInfo { multipv: rank + 1, depth: result.depth, score: line.score,
                            nodes: result.nodes, elapsed, pv: line.pv.clone(), tbhits: result.stats.tb_hits });
                    }
                }
                emit(&format!("info string search complete nodes {} tbhits {} tb_errors {} helper_nodes {} helper_failures {} singular_probes {} singular_extensions {} selected_worker {} nnue_refreshes {} nnue_updates {} nnue_evaluations {} nnue_bucket_refreshes {}",
                    result.nodes, result.stats.tb_hits, result.stats.tb_errors,
                    result.stats.helper_nodes, result.stats.helper_failures,
                    result.stats.singular_probes, result.stats.singular_extensions, result.stats.smp_selected_worker,
                    result.stats.nnue_refreshes, result.stats.nnue_updates, result.stats.nnue_evaluations, result.stats.nnue_bucket_refreshes));
                // UCI infinite searches must wait for stop, including terminal positions.
                while infinite && !stop.load(Ordering::Relaxed) {
                    thread::park_timeout(Duration::from_millis(10));
                }
                let best_move = result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "0000".into());
                emit(&format!("bestmove {best_move}"));
                search
            });
        match worker {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                self.search = Some(self.fresh_search());
                return Err(format!("cannot start search worker: {error}"));
            }
        }
        Ok(())
    }

    fn command(&mut self, line: &str) -> Result<bool, String> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(command) = tokens.first() else {
            return Ok(true);
        };
        match *command {
            "uci" => {
                emit(&format!("id name Zenith {}", env!("CARGO_PKG_VERSION")));
                emit("id author Zenith contributors");
                emit("option name Hash type spin default 16 min 1 max 1024");
                emit("option name EvalFile type string default <empty>");
                emit("option name Use NNUE type check default true");
                emit(&format!(
                    "option name MultiPV type spin default 1 min 1 max {MAX_MULTI_PV}"
                ));
                emit(&format!(
                    "option name Threads type spin default {DEFAULT_THREADS} min 1 max 64"
                ));
                emit(&format!(
                    "option name SyzygyPath type string default {}",
                    self.default_syzygy_path
                ));
                emit("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
                emit("option name SyzygyProbeDepth type spin default 1 min 1 max 120");
                emit("option name Clear Hash type button");
                emit("option name Move Overhead type spin default 10 min 0 max 5000");
                emit(&format!(
                    "option name Null Move Pruning type check default {DEFAULT_NULL_PRUNING}"
                ));
                emit("option name Selective Pruning type check default true");
                emit("option name Search Enhancements type check default true");
                emit(&format!(
                    "option name Modern Search type check default {DEFAULT_MODERN_SEARCH}"
                ));
                emit(&format!(
                    "option name Singular Extensions type check default {DEFAULT_SINGULAR}"
                ));
                emit(&format!(
                    "option name SMP Voting type check default {DEFAULT_SMP_VOTING}"
                ));
                let parameters = SearchParameters::default();
                emit(&format!(
                    "option name RFP Margin type spin default {} min 40 max 300",
                    parameters.rfp_margin
                ));
                emit(&format!(
                    "option name Futility Margin type spin default {} min 40 max 300",
                    parameters.futility_margin
                ));
                emit(&format!(
                    "option name NMP Base Reduction type spin default {} min 2 max 5",
                    parameters.nmp_base_reduction
                ));
                emit(&format!(
                    "option name LMR Divisor type spin default {} min 150 max 600",
                    parameters.lmr_divisor
                ));
                emit("uciok");
            }
            "isready" => emit("readyok"),
            "ucinewgame" => {
                self.stop_search();
                self.search.as_mut().expect("idle search").clear();
                self.board = Board::startpos();
                self.history = vec![self.board.hash()];
            }
            "position" => self.set_position(&tokens[1..])?,
            "setoption" => self.set_option(&tokens[1..], line)?,
            "go" => self.go(&tokens[1..])?,
            "stop" => self.stop_search(),
            "quit" => {
                self.stop_search();
                return Ok(false);
            }
            "d" => emit(&format!("info string fen {}", self.board.to_fen())),
            "debug" => {}
            _ => {} // UCI requires tolerating unknown commands.
        }
        Ok(true)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop_search();
    }
}

fn parse_position(tokens: &[&str]) -> Result<(Board, Vec<u64>), String> {
    let (mut board, mut cursor) = match tokens.first().copied() {
        Some("startpos") => (Board::startpos(), 1),
        Some("fen") if tokens.len() >= 7 => (Board::from_fen(&tokens[1..7].join(" "))?, 7),
        _ => return Err("position expects startpos or fen with six fields".into()),
    };
    let mut history = vec![board.hash()];
    if cursor < tokens.len() {
        if tokens[cursor] != "moves" {
            return Err("expected moves after position".into());
        }
        cursor += 1;
        for token in &tokens[cursor..] {
            let mv = board.parse_uci_move(token)?;
            board = board.make_move(mv);
            history.push(board.hash());
        }
    }
    Ok((board, history))
}

fn clock_budget(remaining: u64, increment: u64, moves_to_go: u64, overhead: u64) -> Duration {
    let available = remaining.saturating_sub(overhead);
    let target = (available / moves_to_go.max(1)).saturating_add(increment.saturating_mul(3) / 4);
    Duration::from_millis(target.min((available / 2).max(1)).max(1))
}

fn parse_go(tokens: &[&str], board: &Board, overhead: u64) -> Result<Limits, String> {
    let mut limits = Limits::default();
    let (mut white_time, mut black_time) = (None, None);
    let (mut white_increment, mut black_increment) = (0, 0);
    let mut moves_to_go = 30;
    let mut cursor = 0;
    while cursor < tokens.len() {
        let key = tokens[cursor];
        cursor += 1;
        match key {
            "infinite" => limits.infinite = true,
            "searchmoves" => {
                let mut moves = Vec::new();
                while cursor < tokens.len() && !is_go_keyword(tokens[cursor]) {
                    let mv = board.parse_uci_move(tokens[cursor])?;
                    if !moves.contains(&mv) {
                        moves.push(mv);
                    }
                    cursor += 1;
                }
                if moves.is_empty() {
                    return Err("searchmoves requires at least one legal move".into());
                }
                limits.searchmoves = Some(moves);
            }
            "ponder" | "mate" => return Err(format!("go {key} is not supported")),
            "depth" | "nodes" | "movetime" | "wtime" | "btime" | "winc" | "binc" | "movestogo" => {
                let value = tokens
                    .get(cursor)
                    .ok_or_else(|| format!("missing value after {key}"))?
                    .parse::<u64>()
                    .map_err(|_| format!("invalid nonnegative integer after {key}"))?;
                cursor += 1;
                match key {
                    "depth" => {
                        if !(1..=120).contains(&value) {
                            return Err("depth must be from 1 to 120".into());
                        }
                        limits.depth = Some(value as u8);
                    }
                    "nodes" => {
                        if value == 0 {
                            return Err("nodes must be positive".into());
                        }
                        limits.nodes = Some(value);
                    }
                    "movetime" => limits.movetime = Some(Duration::from_millis(value.max(1))),
                    "wtime" => white_time = Some(value),
                    "btime" => black_time = Some(value),
                    "winc" => white_increment = value,
                    "binc" => black_increment = value,
                    "movestogo" => {
                        if value == 0 {
                            return Err("movestogo must be positive".into());
                        }
                        moves_to_go = value;
                    }
                    _ => unreachable!(),
                }
            }
            _ => {}
        }
    }
    if limits.infinite {
        limits.movetime = None;
        limits.soft_time = None;
    } else if limits.movetime.is_none() {
        let (remaining, increment) = match board.side_to_move {
            Color::White => (white_time, white_increment),
            Color::Black => (black_time, black_increment),
        };
        if let Some(remaining) = remaining {
            let soft = clock_budget(remaining, increment, moves_to_go, overhead);
            let available = remaining.saturating_sub(overhead);
            limits.soft_time = Some(soft);
            limits.movetime = Some(
                soft.saturating_mul(4)
                    .min(Duration::from_millis((available / 2).max(1))),
            );
        }
    }
    if limits.depth.is_none() && limits.nodes.is_none() && limits.movetime.is_none() {
        limits.infinite = true;
    }
    Ok(limits)
}

fn is_go_keyword(token: &str) -> bool {
    matches!(
        token,
        "infinite"
            | "searchmoves"
            | "ponder"
            | "mate"
            | "depth"
            | "nodes"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
    )
}

/// Serve UCI until `quit` or EOF. Standard output contains protocol messages only.
pub fn run() -> io::Result<()> {
    let mut engine = Engine::new();
    for line in io::stdin().lock().lines() {
        match engine.command(&line?) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => emit(&format!("info string error: {error}")),
        }
    }
    engine.stop_search();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_option_survives_hash_resize_and_new_game() {
        let mut engine = Engine::new();
        let board = Board::from_fen(crate::bench::POSITIONS[3].1).unwrap();
        for enabled in [true, false] {
            engine
                .command(&format!(
                    "setoption name Search Enhancements value {enabled}"
                ))
                .unwrap();
            engine
                .command(&format!("setoption name Selective Pruning value {enabled}"))
                .unwrap();
            engine
                .command(&format!("setoption name Null Move Pruning value {enabled}"))
                .unwrap();
            engine
                .command(&format!("setoption name Modern Search value {enabled}"))
                .unwrap();
            engine.command("setoption name Hash value 1").unwrap();
            engine.command("ucinewgame").unwrap();
            let result = engine.search.as_mut().unwrap().search(
                board,
                &[board.hash()],
                Limits {
                    depth: Some(6),
                    ..Limits::default()
                },
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            assert_eq!(result.null_probes > 0, enabled);
            assert_eq!(result.stats.q_pruned > 0, enabled);
            assert_eq!(result.stats.extensions > 0, enabled);
            assert_eq!(result.stats.correction_updates > 0, enabled);
        }
        assert!(
            engine
                .command("setoption name Null Move Pruning value maybe")
                .is_err()
        );
        assert!(!engine.null_pruning);
        assert!(
            engine
                .command("setoption name Modern Search value maybe")
                .is_err()
        );
        assert!(!engine.modern);
    }

    #[test]
    fn time_allocation_respects_clock_and_side() {
        assert_eq!(clock_budget(30_010, 0, 30, 10), Duration::from_millis(1000));
        assert_eq!(clock_budget(0, 1000, 30, 10), Duration::from_millis(1));
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let limits = parse_go(
            &["wtime", "300010", "btime", "3010", "binc", "0"],
            &board,
            10,
        )
        .unwrap();
        assert_eq!(limits.soft_time, Some(Duration::from_millis(100)));
        assert_eq!(limits.movetime, Some(Duration::from_millis(400)));
    }

    #[test]
    fn experimental_controls_validate_and_survive_hash_and_newgame() {
        let mut engine = Engine::new();
        for command in [
            "setoption name Threads value 1",
            "setoption name Singular Extensions value true",
            "setoption name SMP Voting value true",
            "setoption name RFP Margin value 100",
            "setoption name Futility Margin value 140",
            "setoption name NMP Base Reduction value 4",
            "setoption name LMR Divisor value 275",
            "setoption name Hash value 1",
            "ucinewgame",
        ] {
            engine.command(command).unwrap();
        }
        for command in [
            "setoption name Singular Extensions value maybe",
            "setoption name SMP Voting value yes",
            "setoption name RFP Margin value 0",
            "setoption name Futility Margin value 301",
            "setoption name NMP Base Reduction value 65535",
            "setoption name LMR Divisor value 149",
        ] {
            assert!(engine.command(command).is_err(), "{command}");
        }
        assert!(engine.singular && engine.smp_voting);
        assert_eq!(
            engine.parameters,
            SearchParameters {
                rfp_margin: 100,
                futility_margin: 140,
                nmp_base_reduction: 4,
                lmr_divisor: 275
            }
        );
        let mut reference = Search::new(1);
        reference.set_singular_extensions(true);
        reference.set_smp_voting(true);
        reference.set_parameters(engine.parameters).unwrap();
        let limits = Limits {
            depth: Some(5),
            ..Limits::default()
        };
        let expected = reference.search(
            engine.board,
            &engine.history,
            limits.clone(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        let actual = engine.search.as_mut().unwrap().search(
            engine.board,
            &engine.history,
            limits,
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(
            (actual.nodes, actual.score, actual.best_move),
            (expected.nodes, expected.score, expected.best_move)
        );
    }

    #[test]
    fn multipv_is_validated_and_preserved_across_hash_and_newgame() {
        let mut engine = Engine::new();
        for command in [
            "setoption name Threads value 1",
            "setoption name MultiPV value 4",
            "setoption name Hash value 1",
            "ucinewgame",
        ] {
            engine.command(command).unwrap();
        }
        for value in ["0", "33", "-1", "yes"] {
            assert!(
                engine
                    .command(&format!("setoption name MultiPV value {value}"))
                    .is_err()
            );
        }
        assert_eq!(engine.multi_pv, 4);
        let result = engine.search.as_mut().unwrap().search(
            engine.board,
            &engine.history,
            Limits {
                depth: Some(3),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.variations.len(), 4);
        engine.command("setoption name MultiPV value 1").unwrap();
        assert_eq!(engine.multi_pv, 1);
    }

    #[test]
    fn position_is_legal_and_preserves_history() {
        let (board, history) = parse_position(&["startpos", "moves", "e2e4", "e7e5"]).unwrap();
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last(), Some(&board.hash()));
        assert!(parse_position(&["startpos", "moves", "e2e5"]).is_err());
    }

    #[test]
    fn go_validates_values_and_root_moves() {
        let board = Board::startpos();
        for command in [
            vec!["depth", "0"],
            vec!["nodes", "0"],
            vec!["wtime", "-1"],
            vec!["movetime"],
            vec!["searchmoves", "e2e5"],
        ] {
            assert!(parse_go(&command, &board, 10).is_err());
        }
        let limits = parse_go(&["searchmoves", "e2e4", "d2d4", "depth", "3"], &board, 10).unwrap();
        assert_eq!(limits.searchmoves.unwrap().len(), 2);
        assert_eq!(limits.depth, Some(3));
        let clocks = parse_go(&["wtime", "1000", "winc", "10000"], &board, 10).unwrap();
        assert!(clocks.soft_time <= clocks.movetime);
        assert!(clocks.movetime.unwrap() <= Duration::from_millis(495));
        let explicit = parse_go(&["wtime", "10000", "movetime", "25"], &board, 10).unwrap();
        assert_eq!(explicit.movetime, Some(Duration::from_millis(25)));
        assert_eq!(explicit.soft_time, None);
        let infinite = parse_go(&["wtime", "10000", "infinite"], &board, 10).unwrap();
        assert_eq!(infinite.movetime, None);
        assert_eq!(infinite.soft_time, None);
    }

    #[test]
    fn smp_and_syzygy_options_validate_and_survive_reset() {
        let mut engine = Engine::new();
        engine.command("setoption name Threads value 3").unwrap();
        engine
            .command("setoption name SyzygyProbeLimit value 5")
            .unwrap();
        engine
            .command("setoption name SyzygyProbeDepth value 4")
            .unwrap();
        for command in [
            "setoption name Threads value 0",
            "setoption name Threads value 65",
            "setoption name SyzygyProbeLimit value 8",
            "setoption name SyzygyProbeDepth value 0",
            "setoption name SyzygyPath value target/no-such-tables",
        ] {
            assert!(engine.command(command).is_err());
        }
        engine.command("setoption name Hash value 1").unwrap();
        engine.command("ucinewgame").unwrap();
        assert_eq!(engine.threads, 3);
        assert_eq!(engine.probe_limit, 5);
        assert_eq!(engine.probe_depth, 4);
        engine
            .command("setoption name SyzygyPath value <empty>")
            .unwrap();
        assert!(engine.syzygy.is_none());
    }
}

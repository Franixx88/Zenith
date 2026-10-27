//! Iterative deepening, PVS, quiescence, and a bounded transposition table.
//!
//! TT scores require a matching reversible-history fingerprint and halfmove
//! clock. Board hashes alone cannot safely cache repetition or 50-move results.
//! Hash moves can still order moves across different histories. Like ordinary
//! Zobrist keys, the context fingerprint has a theoretical collision risk.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::board::{Board, EMPTY_MOVE, Move, MoveList, Piece};
use crate::eval::evaluate;
use crate::movepick::{MovePicker, is_capture};
use crate::nnue::Network;
mod evaluation;
use evaluation::NnueStack;
mod multipv;
#[cfg(test)]
mod nnue_tests;
mod params;
mod smp;
mod tt;
pub use params::SearchParameters;
use tt::Table;

pub const MATE_SCORE: i32 = 30_000;
pub const DEFAULT_NULL_PRUNING: bool = true;
pub const DEFAULT_MODERN_SEARCH: bool = true;
pub const DEFAULT_SINGULAR: bool = false;
pub const DEFAULT_SMP_VOTING: bool = false;
pub const MAX_MULTI_PV: usize = 32;
const INFINITY: i32 = 32_000;
const MAX_PLY: usize = 128;
const MAX_DEPTH: u8 = 120;
const ORDER_VALUES: [i32; 6] = [100, 320, 335, 500, 950, 20_000];
const HISTORY_MAX: i32 = 16_384;
const MAX_EXTENSIONS: u8 = 2;
const PIECE_SQUARES: usize = 6 * 64;

#[derive(Clone)]
struct PvLine {
    moves: [Move; MAX_PLY],
    len: usize,
}
impl Default for PvLine {
    fn default() -> Self {
        Self {
            moves: [EMPTY_MOVE; MAX_PLY],
            len: 0,
        }
    }
}
impl PvLine {
    fn clear(&mut self) {
        self.len = 0;
    }
    fn push(&mut self, mv: Move) {
        assert!(self.len < MAX_PLY);
        self.moves[self.len] = mv;
        self.len += 1;
    }
    fn extend_from_slice(&mut self, moves: &[Move]) {
        for &mv in moves.iter().take(MAX_PLY - self.len) {
            self.push(mv);
        }
    }
}
impl std::ops::Deref for PvLine {
    type Target = [Move];
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}
impl<'a> IntoIterator for &'a PvLine {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone, Copy)]
struct NodeState {
    history: u64,
    extensions_left: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Played {
    piece: Piece,
    to: u8,
}
impl Played {
    fn index(self) -> usize {
        self.piece.index() * 64 + self.to as usize
    }
    fn new(board: &Board, mv: Move) -> Self {
        Self {
            piece: mv
                .promotion
                .unwrap_or(board.piece_at(mv.from).expect("legal origin").1),
            to: mv.to,
        }
    }
}

fn update_history(entry: &mut i32, bonus: i32) {
    let bonus = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    *entry += bonus - *entry * bonus.abs() / HISTORY_MAX;
}

pub fn is_mate_score(score: i32) -> bool {
    score.unsigned_abs() >= (MATE_SCORE - MAX_PLY as i32) as u32
}

#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub depth: Option<u8>,
    pub nodes: Option<u64>,
    pub movetime: Option<Duration>,
    /// Clock allocation: stop between iterations; movetime remains the hard cap.
    pub soft_time: Option<Duration>,
    pub infinite: bool,
    /// `Some` restricts the root to these legal moves, including an empty list.
    pub searchmoves: Option<Vec<Move>>,
}

#[derive(Clone, Debug)]
pub struct SearchInfo {
    /// UCI rank, starting at one. A batch contains only one completed depth.
    pub multipv: usize,
    pub depth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
    pub tbhits: u64,
}

#[derive(Clone, Debug)]
pub struct RootVariation {
    pub score: i32,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Completed MultiPV batch. Empty in the allocation-free single-PV mode,
    /// or when cancellation precedes the first completed batch.
    pub variations: Vec<RootVariation>,
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub null_probes: u64,
    pub null_cutoffs: u64,
    pub stats: SearchStats,
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub nnue_refreshes: u64,
    pub nnue_bucket_refreshes: u64,
    pub nnue_updates: u64,
    pub nnue_evaluations: u64,
    pub singular_probes: u64,
    pub singular_extensions: u64,
    pub smp_selected_worker: usize,
    pub tb_probes: u64,
    pub tb_hits: u64,
    pub tb_errors: u64,
    pub helper_nodes: u64,
    pub helper_failures: u64,
    pub qnodes: u64,
    pub q_pruned: u64,
    pub reverse_cutoffs: u64,
    pub eval_calls: u64,
    pub eval_hits: u64,
    pub tt_probes: u64,
    pub tt_key_hits: u64,
    pub tt_context_hits: u64,
    pub tt_cutoffs: u64,
    pub tt_collisions: u64,
    pub tt_replacements: u64,
    pub extensions: u64,
    pub reductions: u64,
    pub reduction_researches: u64,
    pub lmp_pruned: u64,
    pub futility_pruned: u64,
    pub history_pruned: u64,
    pub quiet_see_pruned: u64,
    pub iir: u64,
    pub correction_updates: u64,
    pub max_main_ply: usize,
}

#[derive(Clone, Copy, Default)]
struct EvalEntry {
    key: u64,
    score: i32,
    valid: bool,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    context: u64,
    halfmove: u16,
    depth: i16,
    score: i32,
    best: Option<Move>,
    bound: Bound,
    generation: u8,
    valid: bool,
    extensions_left: u8,
}

pub struct Search {
    network: Option<Arc<Network>>,
    #[cfg(test)]
    reference_nnue: bool,
    multi_pv: usize,
    singular: bool,
    smp_voting: bool,
    parameters: SearchParameters,
    threads: usize,
    worker_id: usize,
    workers: Vec<Search>,
    control: Option<Arc<Control>>,
    syzygy: Option<Arc<crate::syzygy::Syzygy>>,
    probe_limit: usize,
    probe_depth: i16,
    lmr: bool,
    see_ordering: bool,
    advanced_history: bool,
    null_pruning: bool,
    pruning: bool,
    enhancements: bool,
    modern: bool,
    correction: Vec<i16>,
    continuation: [Vec<i16>; 2],
    counters: [[Option<Move>; PIECE_SQUARES]; 2],
    reductions: Vec<i16>,
    eval_cache: Vec<EvalEntry>,
    table: Table,
    generation: u8,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    captures: [[[[i32; 6]; 64]; 6]; 2],
}

impl Search {
    pub fn new(hash_mb: usize) -> Self {
        let bytes = hash_mb.clamp(1, 1024) * 1024 * 1024;
        let eval_length = ((bytes / 8) / std::mem::size_of::<EvalEntry>())
            .min(65536)
            .next_power_of_two();
        let table_bytes = bytes - eval_length * std::mem::size_of::<EvalEntry>();
        let parameters = SearchParameters::default();
        let reductions = parameters.reductions();
        Self {
            network: None,
            #[cfg(test)]
            reference_nnue: false,
            multi_pv: 1,
            singular: DEFAULT_SINGULAR,
            smp_voting: DEFAULT_SMP_VOTING,
            parameters,
            threads: 1,
            worker_id: 0,
            workers: Vec::new(),
            control: None,
            syzygy: None,
            probe_limit: 7,
            probe_depth: 1,
            lmr: true,
            see_ordering: true,
            advanced_history: false,
            null_pruning: DEFAULT_NULL_PRUNING,
            pruning: true,
            enhancements: true,
            modern: DEFAULT_MODERN_SEARCH,
            correction: vec![0; 2 * 16384],
            continuation: std::array::from_fn(|_| vec![0; 2 * PIECE_SQUARES * PIECE_SQUARES]),
            counters: [[None; PIECE_SQUARES]; 2],
            reductions,
            eval_cache: vec![EvalEntry::default(); eval_length],
            table: Table::new(table_bytes),
            generation: 0,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            captures: [[[[0; 6]; 64]; 6]; 2],
        }
    }

    /// Select validated immutable weights, or None for classical evaluation.
    /// Model changes invalidate every score cache and learned search history.
    pub fn set_network(&mut self, network: Option<Arc<Network>>) {
        let unchanged = match (&self.network, &network) {
            (None, None) => true,
            (Some(old), Some(new)) => Arc::ptr_eq(old, new),
            _ => false,
        };
        if !unchanged {
            self.clear();
            self.network = network;
        }
    }

    pub fn clear(&mut self) {
        self.control = None;
        self.workers.clear();
        self.correction.fill(0);
        self.eval_cache.fill(EvalEntry::default());
        self.table.clear();
        for history in &mut self.continuation {
            history.fill(0);
        }
        self.counters = [[None; PIECE_SQUARES]; 2];
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.captures = [[[[0; 6]; 64]; 6]; 2];
        self.generation = 0;
    }

    pub fn set_singular_extensions(&mut self, enabled: bool) {
        if self.singular != enabled {
            self.clear();
            self.singular = enabled;
        }
    }

    pub fn set_multi_pv(&mut self, count: usize) {
        let count = count.clamp(1, MAX_MULTI_PV);
        if self.multi_pv != count {
            self.clear();
            self.multi_pv = count;
        }
    }
    pub fn set_smp_voting(&mut self, enabled: bool) {
        if self.smp_voting != enabled {
            self.clear();
            self.smp_voting = enabled;
        }
    }
    pub fn set_parameters(&mut self, parameters: SearchParameters) -> Result<(), String> {
        let parameters = parameters.validate()?;
        if self.parameters != parameters {
            self.clear();
            self.reductions = parameters.reductions();
            self.parameters = parameters;
        }
        Ok(())
    }

    /// Experimental selectivity/correction group; cache and learning isolation
    /// make the switch reproducible independently of move generation and SIMD.
    pub fn set_modern_search(&mut self, enabled: bool) {
        if self.modern != enabled {
            self.clear();
            self.modern = enabled;
        }
    }

    /// Toggle selective search for reproducible comparisons. Cached scores must
    /// not cross configurations because reductions change the searched tree.
    pub fn set_lmr(&mut self, enabled: bool) {
        if self.lmr != enabled {
            self.clear();
            self.lmr = enabled;
        }
    }

    /// Switch ordering for ablation. Ordering also affects selective reductions.
    pub fn set_see_ordering(&mut self, enabled: bool) {
        if self.see_ordering != enabled {
            self.clear();
            self.see_ordering = enabled;
        }
    }

    pub fn set_advanced_history(&mut self, enabled: bool) {
        if self.advanced_history != enabled {
            self.clear();
            self.advanced_history = enabled;
        }
    }

    pub fn set_null_pruning(&mut self, enabled: bool) {
        if self.null_pruning != enabled {
            self.clear();
            self.null_pruning = enabled;
        }
    }

    pub fn set_pruning(&mut self, enabled: bool) {
        if self.pruning != enabled {
            self.clear();
            self.pruning = enabled;
        }
    }

    /// Continuation/countermove ordering, adaptive LMR and bounded extensions.
    pub fn set_enhancements(&mut self, enabled: bool) {
        if self.enhancements != enabled {
            self.clear();
            self.enhancements = enabled;
        }
    }

    fn capture_entry(&self, board: &Board, mv: Move) -> i32 {
        let attacker = board.piece_at(mv.from).expect("legal origin").1.index();
        let victim = board
            .piece_at(mv.to)
            .map_or(Piece::Pawn, |(_, p)| p)
            .index();
        self.captures[board.side_to_move.index()][attacker][mv.to as usize][victim]
    }

    fn update_capture(&mut self, board: &Board, mv: Move, bonus: i32) {
        let attacker = board.piece_at(mv.from).expect("legal origin").1.index();
        let victim = board
            .piece_at(mv.to)
            .map_or(Piece::Pawn, |(_, p)| p)
            .index();
        update_history(
            &mut self.captures[board.side_to_move.index()][attacker][mv.to as usize][victim],
            bonus,
        );
    }

    fn reward_cutoff(
        &mut self,
        board: &Board,
        mv: Move,
        depth: i16,
        quiets: &[Move],
        captures: &[Move],
    ) {
        let bonus = (i32::from(depth).pow(2) * 16).min(2048);
        let penalty = bonus / 2;
        let color = board.side_to_move.index();
        if is_capture(board, mv) {
            if self.advanced_history {
                self.update_capture(board, mv, bonus);
            }
        } else if mv.promotion.is_none() {
            update_history(
                &mut self.history[color][mv.from as usize][mv.to as usize],
                bonus,
            );
            if self.advanced_history {
                for tried in quiets {
                    update_history(
                        &mut self.history[color][tried.from as usize][tried.to as usize],
                        -penalty,
                    );
                }
            }
        }
        if self.advanced_history {
            for tried in captures {
                self.update_capture(board, *tried, -penalty);
            }
        }
    }

    /// Search a board. `history` contains position hashes through the root.
    /// Cancellation always leaves a legal fallback if an allowed move exists.
    pub fn search(
        &mut self,
        board: Board,
        history: &[u64],
        limits: Limits,
        stop: Arc<AtomicBool>,
        on_info: impl FnMut(SearchInfo),
    ) -> SearchResult {
        if self.threads == 1 {
            return self.search_single(board, history, limits, stop, on_info);
        }
        let start = Instant::now();
        let control = Arc::new(Control {
            done: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            tbhits: AtomicU64::new(0),
            limit: limits.nodes,
            start,
            deadline: limits.movetime.and_then(|time| start.checked_add(time)),
        });
        self.control = Some(Arc::clone(&control));
        let mut idle = std::mem::take(&mut self.workers).into_iter();
        // A voted single line cannot replace one member of a coherent MultiPV
        // batch: it might duplicate another root or belong to a different depth.
        let voting = self.smp_voting && self.multi_pv == 1;
        std::thread::scope(|scope| {
            let guard = FinishGuard(Arc::clone(&control));
            let mut handles = Vec::new();
            let mut failures = 0;
            for id in 1..self.threads {
                let mut worker = idle.next().unwrap_or_else(|| Self::new(1));
                worker.table = self.table.shared_clone();
                worker.network = self.network.clone();
                #[cfg(test)]
                {
                    worker.reference_nnue = self.reference_nnue;
                }
                worker.worker_id = id;
                worker.multi_pv = self.multi_pv;
                worker.generation = self.generation;
                worker.lmr = self.lmr;
                worker.see_ordering = self.see_ordering;
                worker.advanced_history = self.advanced_history;
                worker.null_pruning = self.null_pruning;
                worker.pruning = self.pruning;
                worker.enhancements = self.enhancements;
                worker.modern = self.modern;
                worker.singular = self.singular;
                worker.smp_voting = self.smp_voting;
                if worker.parameters != self.parameters {
                    worker.parameters = self.parameters;
                    worker.reductions = self.reductions.clone();
                }
                worker.syzygy = self.syzygy.clone();
                worker.probe_limit = self.probe_limit;
                worker.probe_depth = self.probe_depth;
                worker.control = Some(Arc::clone(&control));
                let history = history.to_vec();
                let mut limits = limits.clone();
                limits.soft_time = None;
                let stop = Arc::clone(&stop);
                match std::thread::Builder::new()
                    .name(format!("zenith-helper-{id}"))
                    .stack_size(16 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let result = worker.search_single(board, &history, limits, stop, |_| {});
                        worker.control = None;
                        (worker, result)
                    }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => failures += 1,
                }
            }
            // The guard also cancels helpers if a caller's info callback unwinds.
            let allowed = limits.searchmoves.clone();
            let mut result = self.search_single(board, history, limits, stop, on_info);
            let mut choices = if voting { vec![result.clone()] } else { vec![] };
            let mut choice_ids = if voting { vec![0] } else { vec![] };
            drop(guard);
            for handle in handles {
                match handle.join() {
                    Ok((worker, helper)) => {
                        if voting {
                            choices.push(helper.clone());
                            choice_ids.push(worker.worker_id);
                        }
                        result.stats.helper_nodes += helper.nodes;
                        result.stats.add(&helper.stats);
                        result.null_probes += helper.null_probes;
                        result.null_cutoffs += helper.null_cutoffs;
                        self.workers.push(worker);
                    }
                    Err(_) => failures += 1,
                }
            }
            result.nodes = control.nodes.load(Ordering::Relaxed);
            result.stats.helper_failures += failures;
            if voting {
                let selected = smp::select(board, allowed.as_deref(), &choices);
                result.stats.smp_selected_worker = choice_ids[selected];
                if selected != 0 {
                    let chosen = &choices[selected];
                    result.best_move = chosen.best_move;
                    result.score = chosen.score;
                    result.depth = chosen.depth;
                    result.pv = chosen.pv.clone();
                }
            }
            self.control = None;
            result
        })
    }

    pub fn set_threads(&mut self, threads: usize) {
        let threads = threads.clamp(1, 64);
        if self.threads != threads {
            self.clear();
            self.table.make_local();
            self.threads = threads;
        }
    }

    pub fn set_syzygy(
        &mut self,
        tables: Option<Arc<crate::syzygy::Syzygy>>,
        limit: usize,
        depth: i16,
    ) {
        self.clear();
        self.syzygy = tables;
        self.probe_limit = limit.min(7);
        self.probe_depth = depth.clamp(1, 120);
    }

    fn search_single(
        &mut self,
        board: Board,
        history: &[u64],
        limits: Limits,
        stop: Arc<AtomicBool>,
        mut on_info: impl FnMut(SearchInfo),
    ) -> SearchResult {
        let start = self.control.as_ref().map_or_else(Instant::now, |c| c.start);
        let deadline = self.control.as_ref().map_or_else(
            || limits.movetime.and_then(|time| start.checked_add(time)),
            |c| c.deadline,
        );
        self.generation = self.generation.wrapping_add(1);
        if self.generation & 15 == 0 {
            for history in &mut self.continuation {
                for score in history {
                    *score /= 2;
                }
            }
        }
        self.killers = [[None; 2]; MAX_PLY];
        for color in &mut self.history {
            for from in color {
                for value in from {
                    *value /= 2;
                }
            }
        }
        for color in &mut self.captures {
            for piece in color {
                for target in piece {
                    for entry in target {
                        *entry /= 2;
                    }
                }
            }
        }
        let mut root_moves = board.legal_moves();
        let terminal = root_moves.is_empty();
        if let Some(allowed) = &limits.searchmoves {
            root_moves.retain(|mv| allowed.contains(mv));
        }
        if self.worker_id != 0 && !root_moves.is_empty() {
            let offset = self.worker_id % root_moves.len();
            root_moves.rotate_left(offset);
        }
        let fallback = root_moves.first().copied();
        // Clone the Arc before borrowing Search mutably into Context. Every
        // worker owns its frames; only the validated model allocation is shared.
        let network = self.network.clone();
        let mut nnue = network
            .as_deref()
            .filter(|_| !root_moves.is_empty())
            .map(|network| NnueStack::new(network, &board));
        #[cfg(test)]
        if let Some(stack) = &mut nnue {
            stack.refresh_on_evaluate = self.reference_nnue;
        }
        let mut result = SearchResult {
            variations: Vec::new(),
            best_move: fallback,
            score: if terminal && board.in_check(board.side_to_move) {
                -MATE_SCORE
            } else if root_moves.is_empty() {
                0
            } else {
                nnue.as_mut()
                    .map_or_else(|| evaluate(&board), |stack| stack.evaluate(0))
            },
            depth: 0,
            nodes: 0,
            pv: fallback.into_iter().collect(),
            null_probes: 0,
            null_cutoffs: 0,
            stats: SearchStats::default(),
        };
        if root_moves.is_empty() {
            return result;
        }
        let root_key = board.hash();
        let mut path = history.to_vec();
        path.reserve(MAX_PLY + 1);
        if path.last() != Some(&root_key) {
            path.push(root_key);
        }
        let root_context = path
            .iter()
            .rev()
            .take(usize::from(board.halfmove_clock) + 1)
            .fold(0_u64, |value, hash| {
                value.wrapping_add(context_component(*hash))
            });
        let mut context = Context {
            nnue,
            null_active: false,
            pruning_suppressed: false,
            repetition_floor: 0,
            null_probes: 0,
            null_cutoffs: 0,
            stats: SearchStats::default(),
            search: self,
            limits: &limits,
            stop,
            start,
            deadline,
            nodes: 0,
            pending_nodes: 0,
            path,
            root_moves,
            pv: (0..MAX_PLY).map(|_| PvLine::default()).collect(),
            prior: [None; MAX_PLY],
            excluded: [None; MAX_PLY],
            static_evals: [None; MAX_PLY],
        };
        if context.is_draw(&board, root_key) {
            result.score = 0;
            if context.search.multi_pv > 1 {
                result.variations = context
                    .root_moves
                    .iter()
                    .take(context.search.multi_pv)
                    .map(|&mv| RootVariation {
                        score: 0,
                        pv: vec![mv],
                    })
                    .collect();
            }
            result.stats = context.snapshot_stats();
            return result;
        }
        // DTZ has no knowledge of old positions. Fall back to normal search if
        // a reversible position already repeated, even before the third visit.
        let reversible: Vec<_> = context
            .path
            .iter()
            .rev()
            .take(usize::from(board.halfmove_clock) + 1)
            .copied()
            .collect();
        let repeated = reversible
            .iter()
            .enumerate()
            .any(|(i, key)| reversible[..i].contains(key));
        if context.search.worker_id == 0
            && context.search.multi_pv == 1
            && !repeated
            && !context.should_stop(true)
            && let Some(tb) = context.search.syzygy.clone()
            && tb.eligible(&board, context.search.probe_limit)
        {
            context.stats.tb_probes += 1;
            match tb.root(&board, &context.root_moves) {
                Ok(Some((mv, score))) => {
                    context.record_tb_hit();
                    result.best_move = Some(mv);
                    result.pv = vec![mv];
                    result.score = score;
                    on_info(SearchInfo {
                        multipv: 1,
                        depth: 0,
                        score,
                        nodes: context.total_nodes(),
                        elapsed: start.elapsed(),
                        pv: result.pv.clone(),
                        tbhits: context.total_tbhits(),
                    });
                    result.stats = context.snapshot_stats();
                    return result;
                }
                Err(_) => context.stats.tb_errors += 1,
                Ok(None) => {}
            }
        }
        if context.search.multi_pv > 1 {
            result = context.search_multiple(&board, root_context, repeated, result, &mut on_info);
            context.flush_nodes();
            result.nodes = context.nodes;
            result.null_probes = context.null_probes;
            result.null_cutoffs = context.null_cutoffs;
            result.stats = context.snapshot_stats();
            return result;
        }
        let max_depth = limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        let mut stable_iterations = 0;
        for depth in (1 + (context.search.worker_id % 2) as u8).min(max_depth)..=max_depth {
            if context.should_stop(true) {
                break;
            }
            let mut window = if depth >= 4 { 35 } else { INFINITY };
            let mut alpha = if window == INFINITY {
                -INFINITY
            } else {
                result.score - window
            };
            let mut beta = if window == INFINITY {
                INFINITY
            } else {
                result.score + window
            };
            let completed_score = loop {
                let Some(score) =
                    context.negamax(&board, i16::from(depth), 0, alpha, beta, root_context)
                else {
                    break None;
                };
                if score <= alpha || score >= beta {
                    window = (window * 2).min(INFINITY);
                    if window == INFINITY {
                        alpha = -INFINITY;
                        beta = INFINITY;
                    } else {
                        alpha = score - window;
                        beta = score + window;
                    }
                    continue;
                }
                break Some(score);
            };
            let Some(score) = completed_score else { break };
            let pv = context.pv[0].to_vec();
            let best = pv.first().copied().or(fallback);
            stable_iterations = if best == result.best_move {
                stable_iterations + 1
            } else {
                0
            };
            let score_drop = result.score - score;
            result = SearchResult {
                variations: Vec::new(),
                best_move: best,
                score,
                depth,
                nodes: context.nodes,
                pv,
                null_probes: context.null_probes,
                null_cutoffs: context.null_cutoffs,
                stats: context.snapshot_stats(),
            };
            context.flush_nodes();
            on_info(SearchInfo {
                multipv: 1,
                depth,
                score,
                nodes: context.total_nodes(),
                elapsed: context.start.elapsed(),
                pv: result.pv.clone(),
                tbhits: context.total_tbhits(),
            });
            if is_mate_score(score) && (MATE_SCORE - score.abs()) <= i32::from(depth) {
                break;
            }
            if depth >= 3
                && let Some(soft) = limits.soft_time
            {
                let adjusted = soft_budget(
                    soft,
                    stable_iterations,
                    score_drop,
                    context.root_moves.len() == 1,
                );
                if context.start.elapsed() >= adjusted {
                    break;
                }
            }
        }
        context.flush_nodes();
        result.nodes = context.nodes;
        result.null_probes = context.null_probes;
        result.null_cutoffs = context.null_cutoffs;
        result.stats = context.snapshot_stats();
        result
    }
}

struct Control {
    done: AtomicBool,
    nodes: AtomicU64,
    tbhits: AtomicU64,
    limit: Option<u64>,
    start: Instant,
    deadline: Option<Instant>,
}
struct FinishGuard(Arc<Control>);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.0.done.store(true, Ordering::Relaxed);
    }
}

impl SearchStats {
    fn add(&mut self, other: &Self) {
        macro_rules! sum { ($($field:ident),*) => {$(self.$field += other.$field;)*}; }
        sum!(
            nnue_refreshes,
            nnue_bucket_refreshes,
            nnue_updates,
            nnue_evaluations,
            singular_probes,
            singular_extensions,
            tb_probes,
            tb_hits,
            tb_errors,
            qnodes,
            q_pruned,
            reverse_cutoffs,
            eval_calls,
            eval_hits,
            tt_probes,
            tt_key_hits,
            tt_context_hits,
            tt_cutoffs,
            tt_collisions,
            tt_replacements,
            extensions,
            reductions,
            reduction_researches,
            lmp_pruned,
            futility_pruned,
            history_pruned,
            quiet_see_pruned,
            iir,
            correction_updates
        );
        self.max_main_ply = self.max_main_ply.max(other.max_main_ply);
    }
}

struct Context<'a> {
    nnue: Option<NnueStack<'a>>,
    excluded: [Option<Move>; MAX_PLY],
    null_active: bool,
    // Null probes and verification must not recursively prune or publish
    // artificial scores/history into the normal search caches.
    pruning_suppressed: bool,
    repetition_floor: usize,
    null_probes: u64,
    null_cutoffs: u64,
    stats: SearchStats,
    search: &'a mut Search,
    limits: &'a Limits,
    stop: Arc<AtomicBool>,
    start: Instant,
    deadline: Option<Instant>,
    nodes: u64,
    pending_nodes: u64,
    path: Vec<u64>,
    root_moves: Vec<Move>,
    pv: Vec<PvLine>,
    prior: [Option<Played>; MAX_PLY],
    static_evals: [Option<i32>; MAX_PLY],
}

impl Context<'_> {
    fn snapshot_stats(&self) -> SearchStats {
        let mut stats = self.stats.clone();
        if let Some(nnue) = &self.nnue {
            nnue.record(&mut stats);
        }
        stats
    }

    fn enter_evaluation(&mut self, board: &Board, ply: usize) {
        if let Some(nnue) = &mut self.nnue {
            nnue.enter(board, ply);
        }
    }

    fn raw_evaluate(&mut self, board: &Board, ply: usize) -> i32 {
        if let Some(nnue) = &mut self.nnue {
            nnue.enter(board, ply);
            nnue.evaluate(ply)
        } else {
            evaluate(board)
        }
    }
    fn is_countermove(&self, board: &Board, mv: Move, ply: usize) -> bool {
        self.search.enhancements
            && !self.pruning_suppressed
            && ply
                .checked_sub(1)
                .and_then(|i| self.prior[i])
                .is_some_and(|previous| {
                    self.search.counters[board.side_to_move.index()][previous.index()] == Some(mv)
                })
    }
    fn continuation_index(
        &self,
        board: &Board,
        mv: Move,
        ply: usize,
        back: usize,
    ) -> Option<usize> {
        let previous = ply.checked_sub(back).and_then(|i| self.prior[i])?;
        Some(
            (board.side_to_move.index() * PIECE_SQUARES + previous.index()) * PIECE_SQUARES
                + Played::new(board, mv).index(),
        )
    }

    fn quiet_history(&self, board: &Board, mv: Move, ply: usize) -> i32 {
        let mut value =
            self.search.history[board.side_to_move.index()][mv.from as usize][mv.to as usize];
        if self.search.enhancements && !self.pruning_suppressed {
            let current = Played::new(board, mv).index();
            let base = board.side_to_move.index() * PIECE_SQUARES * PIECE_SQUARES;
            for back in 1..=2 {
                if let Some(previous) = ply.checked_sub(back).and_then(|i| self.prior[i]) {
                    let index = base + previous.index() * PIECE_SQUARES + current;
                    value += i32::from(self.search.continuation[back - 1][index]);
                }
            }
        }
        value
    }

    fn reward_continuation(
        &mut self,
        board: &Board,
        mv: Move,
        depth: i16,
        quiets: &[Move],
        ply: usize,
    ) {
        if is_capture(board, mv) || mv.promotion.is_some() {
            return;
        }
        let bonus = (i32::from(depth).pow(2) * 8).min(1024);
        for (candidate, change) in
            std::iter::once((mv, bonus)).chain(quiets.iter().map(|&m| (m, -bonus / 2)))
        {
            for back in 1..=2 {
                if let Some(index) = self.continuation_index(board, candidate, ply, back) {
                    let entry = &mut self.search.continuation[back - 1][index];
                    let mut value = i32::from(*entry);
                    update_history(&mut value, change);
                    *entry = value as i16;
                }
            }
        }
        if let Some(previous) = ply.checked_sub(1).and_then(|i| self.prior[i]) {
            self.search.counters[board.side_to_move.index()][previous.index()] = Some(mv);
        }
    }

    fn evaluate(&mut self, board: &Board, ply: usize) -> i32 {
        self.stats.eval_calls += 1;
        let key = board.hash();
        let slot = key as usize & (self.search.eval_cache.len() - 1);
        let entry = self.search.eval_cache[slot];
        if entry.valid && entry.key == key {
            self.stats.eval_hits += 1;
            return entry.score + self.correction_value(board);
        }
        let score = self.raw_evaluate(board, ply);
        self.search.eval_cache[slot] = EvalEntry {
            key,
            score,
            valid: true,
        };
        score + self.correction_value(board)
    }

    fn correction_index(board: &Board) -> usize {
        let pawns = board
            .bitboard(crate::board::Color::White, Piece::Pawn)
            .rotate_left(13)
            ^ board
                .bitboard(crate::board::Color::Black, Piece::Pawn)
                .rotate_right(7);
        board.side_to_move.index() * 16384
            + ((pawns.wrapping_mul(0x9e3779b97f4a7c15) >> 50) as usize)
    }
    fn correction_value(&self, board: &Board) -> i32 {
        if self.search.modern {
            i32::from(self.search.correction[Self::correction_index(board)]) / 64
        } else {
            0
        }
    }
    fn dangerous_pawn(board: &Board, mv: Move) -> bool {
        if board
            .piece_at(mv.from)
            .is_none_or(|(_, p)| p != Piece::Pawn)
        {
            return false;
        }
        let relative = if board.side_to_move == crate::board::Color::White {
            mv.to
        } else {
            mv.to ^ 56
        };
        if relative / 8 >= 5 {
            return true;
        }
        let mut enemy = board.bitboard(board.side_to_move.other(), Piece::Pawn);
        while enemy != 0 {
            let square = enemy.trailing_zeros() as u8;
            enemy &= enemy - 1;
            if (square % 8).abs_diff(mv.to % 8) <= 1
                && if board.side_to_move == crate::board::Color::White {
                    square / 8 > mv.to / 8
                } else {
                    square / 8 < mv.to / 8
                }
            {
                return false;
            }
        }
        true
    }
    fn should_stop(&self, check_time: bool) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self.search.control.as_ref().is_some_and(|c| {
                c.done.load(Ordering::Relaxed)
                    || c.limit
                        .is_some_and(|limit| c.nodes.load(Ordering::Relaxed) >= limit)
            })
            || (self.search.control.is_none()
                && self.limits.nodes.is_some_and(|limit| self.nodes >= limit))
            || (check_time && self.deadline.is_some_and(|end| Instant::now() >= end))
    }

    fn visit(&mut self, ply: usize) -> Option<()> {
        if self.should_stop(self.nodes & 63 == 0) {
            return None;
        }
        if let Some(control) = &self.search.control {
            if let Some(limit) = control.limit {
                control
                    .nodes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |nodes| {
                        (nodes < limit).then(|| nodes + 1)
                    })
                    .ok()?;
            } else {
                self.pending_nodes += 1;
                if self.pending_nodes == 64 {
                    control.nodes.fetch_add(64, Ordering::Relaxed);
                    self.pending_nodes = 0;
                }
            }
        }
        self.nodes += 1;
        self.pv[ply].clear();
        Some(())
    }

    fn flush_nodes(&mut self) {
        if self.pending_nodes != 0 {
            if let Some(c) = &self.search.control {
                c.nodes.fetch_add(self.pending_nodes, Ordering::Relaxed);
            }
            self.pending_nodes = 0;
        }
    }
    fn total_nodes(&self) -> u64 {
        self.search
            .control
            .as_ref()
            .map_or(self.nodes, |c| c.nodes.load(Ordering::Relaxed))
    }
    fn total_tbhits(&self) -> u64 {
        self.search
            .control
            .as_ref()
            .map_or(self.stats.tb_hits, |c| c.tbhits.load(Ordering::Relaxed))
    }
    fn record_tb_hit(&mut self) {
        self.stats.tb_hits += 1;
        if let Some(control) = &self.search.control {
            control.tbhits.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn is_draw(&self, board: &Board, key: u64) -> bool {
        board.halfmove_clock >= 100
            || board.is_insufficient_material()
            || self.path[self.repetition_floor..]
                .iter()
                .rev()
                .take(usize::from(board.halfmove_clock) + 1)
                .filter(|&&hash| hash == key)
                .take(3)
                .count()
                >= 3
    }

    fn negamax(
        &mut self,
        board: &Board,
        depth: i16,
        ply: usize,
        alpha: i32,
        beta: i32,
        history: u64,
    ) -> Option<i32> {
        let extensions_left = if self.search.enhancements {
            MAX_EXTENSIONS
        } else {
            0
        };
        self.node(
            board,
            depth,
            ply,
            alpha,
            beta,
            NodeState {
                history,
                extensions_left,
            },
        )
    }

    fn node(
        &mut self,
        board: &Board,
        mut depth: i16,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        mut state: NodeState,
    ) -> Option<i32> {
        if depth <= 0 {
            return self.quiescence(board, ply, alpha, beta);
        }
        self.visit(ply)?;
        self.enter_evaluation(board, ply);
        self.stats.max_main_ply = self.stats.max_main_ply.max(ply);
        let key = board.hash();
        let in_check = board.in_check(board.side_to_move);
        if self.is_draw(board, key) {
            // Checkmate takes precedence over a draw claim at the boundary.
            return Some(if in_check && !board.has_legal_move() {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        alpha = alpha.max(-MATE_SCORE + ply as i32);
        beta = beta.min(MATE_SCORE - ply as i32 - 1);
        if alpha >= beta {
            return Some(alpha);
        }
        let original_alpha = alpha;
        let requested_depth = depth;
        let incoming_extensions = state.extensions_left;
        self.stats.tt_probes += 1;
        let probe = self.search.table.probe(
            key,
            state.history,
            board.halfmove_clock,
            incoming_extensions,
        );
        self.stats.tt_key_hits += u64::from(probe.key_hit);
        self.stats.tt_context_hits += u64::from(probe.entry.is_some());
        self.stats.tt_collisions += u64::from(probe.collision);
        let hash_move = if ply == 0 && self.search.worker_id % 2 == 1 {
            None
        } else {
            probe.best
        };
        if ply != 0
            && !self.pruning_suppressed
            && !self.null_active
            && let Some(entry) = probe.entry
            && entry.depth >= requested_depth
        {
            let score = score_from_tt(entry.score, ply);
            if entry.bound == Bound::Exact
                || (entry.bound == Bound::Lower && score >= beta)
                || (entry.bound == Bound::Upper && score <= alpha)
            {
                if let Some(mv) = entry.best {
                    self.pv[ply].push(mv);
                }
                self.stats.tt_cutoffs += 1;
                return Some(score);
            }
        }
        let hash = hash_move.and_then(|mv| board.hash_child(mv).map(|child| (mv, child)));
        if hash.is_none() && !board.has_legal_move() {
            return Some(if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        if ply > 0
            && board.halfmove_clock == 0
            && depth >= self.search.probe_depth
            && !self.null_active
            && !self.pruning_suppressed
            && let Some(tb) = self.search.syzygy.clone()
            && tb.eligible(board, self.search.probe_limit)
        {
            self.stats.tb_probes += 1;
            match tb.wdl_zeroing(board, ply) {
                Ok(score) => {
                    self.record_tb_hit();
                    return Some(score);
                }
                Err(_) => self.stats.tb_errors += 1,
            }
        }
        if !self.pruning_suppressed
            && ply > 0
            && state.extensions_left > 0
            && in_check
            && board.legal_count_up_to(4) <= 3
        {
            depth += 1;
            state.extensions_left -= 1;
            self.stats.extensions += 1;
        }
        self.static_evals[ply] = if in_check {
            None
        } else {
            Some(self.evaluate(board, ply))
        };
        let improving = ply >= 2
            && self.static_evals[ply]
                .zip(self.static_evals[ply - 2])
                .is_some_and(|(current, old)| current > old);
        let mut stored_depth = requested_depth;
        if self.search.modern
            && self.search.pruning
            && !self.pruning_suppressed
            && !self.null_active
            && ply > 0
            && depth >= 5
            && hash.is_none()
            && !in_check
        {
            depth -= 1;
            stored_depth -= 1;
            self.stats.iir += 1;
        }
        // The ply cap is a safety bound for pathological checking sequences.
        // Legal terminal positions above are still scored exactly at the cap.
        if ply >= MAX_PLY - 1 {
            return Some(self.evaluate(board, ply));
        }
        // A conservative static fail-high in middlegames. Never skip terminal
        // detection, checks, PV nodes, mate windows, or null verification.
        if self.search.pruning
            && !self.pruning_suppressed
            && ply > 0
            && depth <= 3
            && beta - alpha == 1
            && !in_check
            && !is_mate_score(alpha)
            && !is_mate_score(beta)
            && board.halfmove_clock < 80
            && self.has_pruning_material(board)
            && self.static_evals[ply].is_some_and(|eval| {
                eval - (self.search.parameters.rfp_margin * i32::from(depth) + 70) >= beta
            })
        {
            self.stats.reverse_cutoffs += 1;
            return Some(beta);
        }
        if self.can_try_null(board, depth, ply, alpha, beta) {
            self.null_probes += 1;
            let passed = board.null_move().expect("guard excludes check");
            let saved_floor = self.repetition_floor;
            self.repetition_floor = self.path.len();
            self.path.push(passed.hash());
            self.pruning_suppressed = !self.search.modern;
            self.null_active = true;
            let reduced = (depth - self.search.parameters.nmp_base_reduction - depth / 6).max(1);
            let saved_prior = self.prior[ply];
            self.prior[ply] = None;
            let probe = self.node(
                &passed,
                reduced,
                ply + 1,
                -beta,
                -beta + 1,
                NodeState {
                    history: 0,
                    extensions_left: 0,
                },
            );
            self.prior[ply] = saved_prior;
            self.path.pop();
            self.repetition_floor = saved_floor;
            self.pruning_suppressed = false;
            self.null_active = false;
            let score = -probe?;
            if score >= beta && !is_mate_score(score) {
                if self.search.modern && depth < 10 {
                    self.null_cutoffs += 1;
                    self.pv[ply].clear();
                    return Some(beta);
                }
                // Verify using actual legal moves with NMP, LMR and TT score
                // reuse disabled. This is still selective depth, not a proof
                // against every possible zugzwang.
                self.pruning_suppressed = true;
                let verified = self.node(
                    board,
                    (depth - self.search.parameters.nmp_base_reduction).max(1),
                    ply,
                    beta - 1,
                    beta,
                    state,
                );
                self.pruning_suppressed = false;
                let verified = verified?;
                if verified >= beta {
                    self.null_cutoffs += 1;
                    self.pv[ply].clear();
                    return Some(beta);
                }
                self.pv[ply].clear();
            }
        }
        let mut singular_move = None;
        if self.search.singular
            && !self.null_active
            && !self.pruning_suppressed
            && ply > 0
            && depth >= 6
            && state.extensions_left > 0
            && !in_check
            && let Some(entry) = probe.entry
            && entry.depth >= requested_depth - 3
            && entry.bound != Bound::Upper
            && let Some((mv, _)) = hash
            && entry.best == Some(mv)
        {
            let score = score_from_tt(entry.score, ply);
            if score.abs() < crate::syzygy::TB_WIN_SCORE - MAX_PLY as i32 {
                let threshold = score - 2 * i32::from(depth);
                self.stats.singular_probes += 1;
                let alternative =
                    self.verify_excluding(board, (depth - 1) / 2, ply, threshold, state, mv)?;
                if alternative < threshold {
                    singular_move = Some(mv);
                }
            }
        }
        let mut moves = MovePicker::new(board, hash, false, self.search.see_ordering);
        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut quiets_tried = MoveList::default();
        let mut captures_tried = MoveList::default();
        let mut number = 0;
        let mut quiet_count = 0;
        while let Some((mv, next)) = self.next_move(&mut moves, board, ply) {
            if self.excluded[ply] == Some(mv) || (ply == 0 && !self.root_moves.contains(&mv)) {
                continue;
            }
            let quiet = !is_capture(board, mv) && mv.promotion.is_none();
            quiet_count += usize::from(quiet);
            let history = self.quiet_history(board, mv, ply);
            if self.search.modern
                && self.search.pruning
                && !self.pruning_suppressed
                && ply > 0
                && number > 0
                && beta - alpha == 1
                && depth <= 4
                && !in_check
                && !is_mate_score(alpha)
                && !is_mate_score(beta)
                && !is_mate_score(best_score)
                && board.halfmove_clock < 80
                && board.occupied().count_ones() > 8
                && quiet
                && !Self::dangerous_pawn(board, mv)
                && !next.in_check(next.side_to_move)
                && !self.search.killers[ply].contains(&Some(mv))
                && !self.is_countermove(board, mv, ply)
            {
                if history <= 0
                    && quiet_count > usize::from(improving) * 2 + 2 + (depth * depth) as usize
                {
                    self.stats.lmp_pruned += 1;
                    continue;
                }
                if depth <= 3
                    && history < 2000
                    && self.static_evals[ply].is_some_and(|value| {
                        value + 100 + self.search.parameters.futility_margin * i32::from(depth)
                            <= alpha
                    })
                {
                    self.stats.futility_pruned += 1;
                    continue;
                }
                if depth <= 2 && history < -1200 * i32::from(depth) {
                    self.stats.history_pruned += 1;
                    continue;
                }
                if history < 1000 && !board.see_ge(mv, -30 * i32::from(depth).pow(2)) {
                    self.stats.quiet_see_pruned += 1;
                    continue;
                }
            }
            let next_key = next.hash();
            let next_context = if next.halfmove_clock == 0 {
                context_component(next_key)
            } else {
                state.history.wrapping_add(context_component(next_key))
            };
            let extension = i16::from(singular_move == Some(mv));
            self.stats.singular_extensions += u64::from(extension != 0);
            self.stats.extensions += u64::from(extension != 0);
            let child_depth = depth - 1 + extension;
            let next_state = NodeState {
                history: next_context,
                extensions_left: state.extensions_left - extension as u8,
            };
            self.prior[ply] = Some(Played::new(board, mv));
            self.path.push(next_key);
            let child = if number == 0 {
                self.node(&next, child_depth, ply + 1, -beta, -alpha, next_state)
                    .map(|score| -score)
            } else {
                // The modern group also reduces ordinary pawn moves and late
                // PV moves. Checks, evasions, advanced/passed pawns, killers
                // and promising history remain protected. Alpha improvements
                // ALWAYS receive a full-depth scout before PVS re-search.
                let reduction = if self.search.lmr
                    && !self.pruning_suppressed
                    && ply > 0
                    && (beta - alpha == 1 || (self.search.modern && depth >= 4 && number >= 5))
                    && !is_mate_score(alpha)
                    && !is_mate_score(beta)
                    && depth >= 3
                    && number >= 3
                    && !in_check
                    && !is_capture(board, mv)
                    && mv.promotion.is_none()
                    && (if self.search.modern {
                        !Self::dangerous_pawn(board, mv)
                    } else {
                        board
                            .piece_at(mv.from)
                            .is_some_and(|(_, p)| p != Piece::Pawn)
                    })
                    && !next.in_check(next.side_to_move)
                    && !self.search.killers[ply].contains(&Some(mv))
                    && !self.is_countermove(board, mv, ply)
                    && self.quiet_history(board, mv, ply) < 4000
                {
                    if self.search.modern {
                        (self.search.reductions[(depth as usize).min(127) * 256 + number.min(255)]
                            - i16::from(improving)
                            - i16::from(beta - alpha > 1)
                            - (history / 4000) as i16)
                            .clamp(1, depth - 2)
                    } else if self.search.enhancements {
                        let base = self.search.reductions
                            [(depth as usize).min(127) * 256 + number.min(255)];
                        (base
                            - i16::from(improving)
                            - (self.quiet_history(board, mv, ply) / 4000) as i16)
                            .clamp(
                                1,
                                (1 + i16::from(depth >= 6) + i16::from(depth >= 9)).min(depth - 2),
                            )
                    } else {
                        1 + i16::from(depth >= 6 && number >= 8)
                    }
                } else {
                    0
                };
                self.stats.reductions += u64::from(reduction > 0);
                let mut scout = self
                    .node(
                        &next,
                        child_depth - reduction,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        next_state,
                    )
                    .map(|score| -score);
                if reduction > 0 && scout.is_some_and(|score| score > alpha) {
                    self.stats.reduction_researches += 1;
                    scout = self
                        .node(&next, child_depth, ply + 1, -alpha - 1, -alpha, next_state)
                        .map(|score| -score);
                }
                match scout {
                    Some(score) if score > alpha && score < beta => self
                        .node(&next, child_depth, ply + 1, -beta, -alpha, next_state)
                        .map(|score| -score),
                    other => other,
                }
            };
            self.path.pop();
            let score = child?;
            number += 1;
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
                self.update_pv(ply, mv);
            }
            if alpha >= beta {
                if !self.pruning_suppressed && !self.null_active {
                    if !is_capture(board, mv)
                        && mv.promotion.is_none()
                        && self.search.killers[ply][0] != Some(mv)
                    {
                        self.search.killers[ply][1] = self.search.killers[ply][0];
                        self.search.killers[ply][0] = Some(mv);
                    }
                    if self.search.enhancements {
                        self.reward_continuation(board, mv, depth, &quiets_tried, ply);
                    }
                    self.search
                        .reward_cutoff(board, mv, depth, &quiets_tried, &captures_tried);
                }
                break;
            }
            if self.search.advanced_history || self.search.enhancements {
                if is_capture(board, mv) {
                    captures_tried.push(mv);
                } else if mv.promotion.is_none() {
                    quiets_tried.push(mv);
                }
            }
        }
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // A root restricted by searchmoves does not describe the whole position.
        if !self.pruning_suppressed
            && !self.null_active
            && !(ply == 0 && (self.limits.searchmoves.is_some() || self.search.multi_pv > 1))
        {
            self.stats.tt_replacements += u64::from(self.search.table.write(Entry {
                key,
                context: state.history,
                halfmove: board.halfmove_clock,
                depth: stored_depth,
                extensions_left: incoming_extensions,
                score: score_to_tt(best_score, ply),
                best: best_move,
                bound,
                generation: self.search.generation,
                valid: true,
            }));
        }
        if self.search.modern
            && !self.pruning_suppressed
            && !self.null_active
            && !in_check
            && !(ply == 0 && (self.limits.searchmoves.is_some() || self.search.multi_pv > 1))
            && !is_mate_score(best_score)
            && board.halfmove_clock < 80
            && board.occupied().count_ones() > 8
            && best_move.is_some_and(|mv| !is_capture(board, mv) && mv.promotion.is_none())
            && self.static_evals[ply].is_some_and(|value| {
                bound == Bound::Exact
                    || (bound == Bound::Lower && best_score > value)
                    || (bound == Bound::Upper && best_score < value)
            })
        {
            let raw = self.raw_evaluate(board, ply);
            let target = ((best_score - raw) * 64).clamp(-8192, 8192);
            let weight = (i32::from(depth) + 1).clamp(1, 16);
            let entry = &mut self.search.correction[Self::correction_index(board)];
            *entry = ((i32::from(*entry) * (256 - weight) + target * weight) / 256) as i16;
            self.stats.correction_updates += 1;
        }
        Some(best_score)
    }

    /// Same-position excluded-move search. No TT scores/writes, pruning or
    /// learning can turn a reduced alternatives search into a cached full-node
    /// result. Restore state before propagating cancellation.
    #[allow(clippy::too_many_arguments)]
    fn verify_excluding(
        &mut self,
        board: &Board,
        depth: i16,
        ply: usize,
        beta: i32,
        state: NodeState,
        mv: Move,
    ) -> Option<i32> {
        let old_excluded = self.excluded[ply];
        let old_suppressed = self.pruning_suppressed;
        let old_prior = self.prior[ply];
        let old_eval = self.static_evals[ply];
        let old_pv = self.pv[ply].clone();
        self.excluded[ply] = Some(mv);
        self.pruning_suppressed = true;
        let score = self.node(
            board,
            depth,
            ply,
            beta - 1,
            beta,
            NodeState {
                extensions_left: 0,
                ..state
            },
        );
        self.excluded[ply] = old_excluded;
        self.pruning_suppressed = old_suppressed;
        self.prior[ply] = old_prior;
        self.static_evals[ply] = old_eval;
        self.pv[ply] = old_pv;
        score
    }

    fn can_try_null(
        &mut self,
        board: &Board,
        depth: i16,
        ply: usize,
        alpha: i32,
        beta: i32,
    ) -> bool {
        if !self.search.null_pruning
            || self.pruning_suppressed
            || self.null_active
            || ply == 0
            || depth < 4
            || beta - alpha != 1
            || is_mate_score(alpha)
            || is_mate_score(beta)
            || board.halfmove_clock >= 80
            || board.in_check(board.side_to_move)
            || board.bitboard(board.side_to_move, Piece::Pawn) == 0
        {
            return false;
        }
        self.has_pruning_material(board)
            && self.evaluate(board, ply) >= beta + if self.search.modern { 0 } else { 50 }
    }

    fn has_pruning_material(&self, board: &Board) -> bool {
        if board.bitboard(board.side_to_move, Piece::Pawn) == 0 {
            return false;
        }
        let mut material = 0;
        let mut pieces = 0;
        for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
            let count = board.bitboard(board.side_to_move, piece).count_ones() as i32;
            pieces += count;
            material += count * ORDER_VALUES[piece.index()];
        }
        pieces >= 2 && material >= 950
    }

    fn quiescence(&mut self, board: &Board, ply: usize, mut alpha: i32, beta: i32) -> Option<i32> {
        self.visit(ply)?;
        self.enter_evaluation(board, ply);
        self.stats.qnodes += 1;
        let in_check = board.in_check(board.side_to_move);
        // Check evasions remain complete. Outside check, generate tactics only;
        // if none exist, distinguish a quiet position from stalemate exactly.
        if !board.has_legal_move() {
            return Some(if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        if self.is_draw(board, board.hash()) {
            return Some(0);
        }
        if ply >= MAX_PLY - 1 {
            return Some(self.evaluate(board, ply));
        }
        let stand_pat = if in_check {
            -INFINITY
        } else {
            self.evaluate(board, ply)
        };
        let mut best_score = stand_pat;
        if !in_check {
            if best_score >= beta {
                return Some(best_score);
            }
            alpha = alpha.max(best_score);
        }
        let mut moves = MovePicker::new(board, None, !in_check, self.search.see_ordering);
        while let Some((mv, next)) = self.next_move(&mut moves, board, ply) {
            if self.search.pruning
                && !self.pruning_suppressed
                && !in_check
                && mv.promotion.is_none()
                && !is_mate_score(alpha)
                && !is_mate_score(beta)
                && board.halfmove_clock < 80
                && board.occupied().count_ones() > 6
                && !next.in_check(next.side_to_move)
                && !(board
                    .piece_at(mv.from)
                    .is_some_and(|(_, p)| p == Piece::Pawn)
                    && if board.side_to_move == crate::board::Color::White {
                        mv.to / 8 >= 5
                    } else {
                        mv.to / 8 <= 2
                    })
            {
                let gain = board
                    .piece_at(mv.to)
                    .map_or(100, |(_, p)| ORDER_VALUES[p.index()]);
                if stand_pat + gain + 180 < alpha || !board.see_ge(mv, -80) {
                    self.stats.q_pruned += 1;
                    continue;
                }
            }
            self.prior[ply] = Some(Played::new(board, mv));
            self.path.push(next.hash());
            let score = self
                .quiescence(&next, ply + 1, -beta, -alpha)
                .map(|score| -score);
            self.path.pop();
            let score = score?;
            best_score = best_score.max(score);
            if score > alpha {
                alpha = score;
                self.update_pv(ply, mv);
            }
            if alpha >= beta {
                break;
            }
        }
        Some(best_score)
    }

    fn update_pv(&mut self, ply: usize, mv: Move) {
        let (ancestors, descendants) = self.pv.split_at_mut(ply + 1);
        let line = &mut ancestors[ply];
        line.clear();
        line.push(mv);
        line.extend_from_slice(&descendants[0]);
    }

    fn next_move(
        &self,
        picker: &mut MovePicker,
        board: &Board,
        ply: usize,
    ) -> Option<(Move, Board)> {
        picker.next(
            |mv| self.move_priority(board, mv, None, ply),
            |mv| {
                if self.search.advanced_history && is_capture(board, mv) {
                    self.search.capture_entry(board, mv) / 4
                } else {
                    0
                }
            },
        )
    }

    fn move_priority(&self, board: &Board, mv: Move, hash_move: Option<Move>, ply: usize) -> i32 {
        if hash_move == Some(mv) {
            return 2_000_000;
        }
        let promotion = mv.promotion.map_or(0, |piece| ORDER_VALUES[piece.index()]);
        if is_capture(board, mv) {
            let victim = board
                .piece_at(mv.to)
                .map_or(Piece::Pawn, |(_, piece)| piece);
            let attacker = board
                .piece_at(mv.from)
                .map_or(Piece::Pawn, |(_, piece)| piece);
            return 1_000_000 + 16 * ORDER_VALUES[victim.index()] - ORDER_VALUES[attacker.index()]
                + promotion;
        }
        if promotion != 0 {
            return 900_000 + promotion;
        }
        if self.search.killers[ply][0] == Some(mv) {
            return 800_000;
        }
        if self.search.killers[ply][1] == Some(mv) {
            return 700_000;
        }
        if self.is_countermove(board, mv, ply) {
            return 650_000;
        }
        let jitter = if ply == 0 && self.search.worker_id != 0 {
            (context_component(
                u64::from(mv.from) * 64 + u64::from(mv.to) + (self.search.worker_id as u64) * 4096,
            ) % 128) as i32
        } else {
            0
        };
        self.quiet_history(board, mv, ply) + jitter
    }
}

fn context_component(mut key: u64) -> u64 {
    key ^= key >> 30;
    key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key ^= key >> 27;
    key = key.wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

fn soft_budget(base: Duration, stable: u32, score_drop: i32, forced: bool) -> Duration {
    if forced {
        base / 4
    } else if score_drop > 70 {
        base.saturating_mul(2)
    } else if stable == 0 {
        base.saturating_mul(3) / 2
    } else if stable >= 3 {
        base.saturating_mul(3) / 4
    } else {
        base
    }
}

fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= crate::syzygy::TB_WIN_SCORE - MAX_PLY as i32 {
        score + ply as i32
    } else if score <= -crate::syzygy::TB_WIN_SCORE + MAX_PLY as i32 {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= crate::syzygy::TB_WIN_SCORE - MAX_PLY as i32 {
        score - ply as i32
    } else if score <= -crate::syzygy::TB_WIN_SCORE + MAX_PLY as i32 {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // These regressions explicitly retain the pre-modern control configuration.
    // Production defaults are exercised by integration/tactics and modern tests.
    fn classic_search(hash_mb: usize) -> Search {
        let mut search = Search::new(hash_mb);
        search.set_modern_search(false);
        search.set_null_pruning(false);
        search
    }

    #[test]
    fn iir_records_only_the_depth_actually_searched() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        search.set_modern_search(true);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        ctx.search.set_null_pruning(false);
        ctx.node(
            &board,
            5,
            1,
            -INFINITY,
            INFINITY,
            NodeState {
                history: 0,
                extensions_left: 0,
            },
        )
        .unwrap();
        assert!(ctx.stats.iir > 0);
        let entry = ctx
            .search
            .table
            .probe(board.hash(), 0, board.halfmove_clock, 0)
            .entry
            .unwrap();
        assert_eq!(
            entry.depth, 4,
            "IIR must not publish an unsearched extra ply"
        );
    }

    #[test]
    fn correction_does_not_contaminate_raw_eval_cache_and_clears_on_toggle() {
        let board = Board::startpos();
        let raw = evaluate(&board);
        let mut search = classic_search(1);
        search.set_modern_search(true);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        let index = Context::correction_index(&board);
        ctx.search.correction[index] = 640;
        assert_eq!(ctx.evaluate(&board, 0), raw + 10);
        ctx.search.correction[index] = -1280;
        assert_eq!(ctx.evaluate(&board, 0), raw - 20);
        let slot = board.hash() as usize & (ctx.search.eval_cache.len() - 1);
        assert_eq!(ctx.search.eval_cache[slot].score, raw);
        ctx.search.set_modern_search(false);
        assert_eq!(ctx.evaluate(&board, 0), raw);
        assert!(ctx.search.correction.iter().all(|&v| v == 0));
    }

    #[test]
    fn modern_null_cancellation_restores_all_context_and_cannot_publish_scores() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        search.set_modern_search(true);
        let limits = Limits {
            nodes: Some(2),
            ..Limits::default()
        };
        let mut ctx = null_context(&mut search, &limits, board);
        assert!(
            ctx.node(
                &board,
                6,
                1,
                -101,
                -100,
                NodeState {
                    history: 0,
                    extensions_left: 0
                }
            )
            .is_none()
        );
        assert!(ctx.null_probes > 0);
        assert!(!ctx.null_active);
        assert!(!ctx.pruning_suppressed);
        assert_eq!(ctx.repetition_floor, 0);
        assert_eq!(ctx.path, vec![board.hash()]);
        assert!(ctx.search.table.iter().all(|e| !e.valid));
        assert!(ctx.search.correction.iter().all(|&v| v == 0));
    }

    #[test]
    fn signed_history_remains_bounded_and_recovers_from_saturation() {
        let mut entry = 0;
        for _ in 0..10_000 {
            update_history(&mut entry, 2048);
            assert!((-HISTORY_MAX..=HISTORY_MAX).contains(&entry));
        }
        assert!(entry > 15_000);
        for _ in 0..100 {
            update_history(&mut entry, -2048);
        }
        assert!(entry < -15_000);
        update_history(&mut entry, i32::MAX);
        assert_eq!(entry, HISTORY_MAX);
        update_history(&mut entry, i32::MIN);
        assert_eq!(entry, -HISTORY_MAX);
    }

    #[test]
    fn cutoff_history_rewards_winner_and_only_penalizes_tried_alternatives() {
        let board = Board::from_fen("4k3/8/4p3/3p3p/8/8/8/3QK3 w - - 0 1").unwrap();
        let winner = board.parse_uci_move("d1e2").unwrap();
        let failed = board.parse_uci_move("d1c2").unwrap();
        let untouched = board.parse_uci_move("d1a4").unwrap();
        let capture = board.parse_uci_move("d1d5").unwrap();
        let winning_capture = board.parse_uci_move("d1h5").unwrap();
        let mut search = classic_search(1);
        search.set_advanced_history(true);
        search.history[0][untouched.from as usize][untouched.to as usize] = 777;
        search.reward_cutoff(&board, winner, 4, &[failed], &[capture]);
        assert!(search.history[0][winner.from as usize][winner.to as usize] > 0);
        assert!(search.history[0][failed.from as usize][failed.to as usize] < 0);
        assert_eq!(
            search.history[0][untouched.from as usize][untouched.to as usize],
            777
        );
        assert!(search.capture_entry(&board, capture) < 0);
        let quiet_before = search.history[0][failed.from as usize][failed.to as usize];
        search.reward_cutoff(&board, winning_capture, 4, &[failed], &[capture]);
        assert!(search.capture_entry(&board, winning_capture) > 0);
        assert_eq!(
            search.history[0][failed.from as usize][failed.to as usize],
            quiet_before
        );
        search.clear();
        assert_eq!(search.capture_entry(&board, capture), 0);
        assert_eq!(
            search.history[0][failed.from as usize][failed.to as usize],
            0
        );
    }

    #[test]
    fn history_changes_order_but_preserves_unreduced_search_scores() {
        for (_, fen) in crate::bench::POSITIONS {
            let board = Board::from_fen(fen).unwrap();
            let mut old = classic_search(1);
            old.set_advanced_history(false);
            old.set_lmr(false);
            old.set_pruning(false);
            let mut new = classic_search(1);
            new.set_advanced_history(true);
            new.set_lmr(false);
            new.set_pruning(false);
            let limits = Limits {
                depth: Some(3),
                ..Limits::default()
            };
            let expected = old.search(
                board,
                &[],
                limits.clone(),
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            let actual = new.search(board, &[], limits, Arc::new(AtomicBool::new(false)), |_| {});
            assert_eq!(actual.score, expected.score, "{fen}");
        }
    }

    fn search_fen(fen: &str, depth: u8) -> (Board, SearchResult) {
        let board = Board::from_fen(fen).unwrap();
        let result = classic_search(1).search(
            board,
            &[board.hash()],
            Limits {
                depth: Some(depth),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        (board, result)
    }

    fn quiescence_fen(fen: &str) -> (i32, u64, Vec<Move>) {
        let board = Board::from_fen(fen).unwrap();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut context = Context {
            nnue: None,
            pruning_suppressed: false,
            null_active: false,
            repetition_floor: 0,
            null_probes: 0,
            null_cutoffs: 0,
            stats: SearchStats::default(),
            search: &mut search,
            limits: &limits,
            stop: Arc::new(AtomicBool::new(false)),
            start: Instant::now(),
            deadline: None,
            nodes: 0,
            pending_nodes: 0,
            path: vec![board.hash()],
            root_moves: board.legal_moves(),
            pv: (0..MAX_PLY).map(|_| PvLine::default()).collect(),
            prior: [None; MAX_PLY],
            excluded: [None; MAX_PLY],
            static_evals: [None; MAX_PLY],
        };
        let score = context.quiescence(&board, 0, -INFINITY, INFINITY).unwrap();
        (score, context.nodes, context.pv[0].to_vec())
    }

    pub(super) fn null_context<'a>(
        search: &'a mut Search,
        limits: &'a Limits,
        board: Board,
    ) -> Context<'a> {
        search.set_null_pruning(true);
        Context {
            nnue: None,
            search,
            null_active: false,
            limits,
            stop: Arc::new(AtomicBool::new(false)),
            start: Instant::now(),
            deadline: None,
            nodes: 0,
            pending_nodes: 0,
            path: vec![board.hash()],
            root_moves: board.legal_moves(),
            pv: (0..MAX_PLY).map(|_| PvLine::default()).collect(),
            prior: [None; MAX_PLY],
            excluded: [None; MAX_PLY],
            static_evals: [None; MAX_PLY],
            pruning_suppressed: false,
            repetition_floor: 0,
            null_probes: 0,
            null_cutoffs: 0,
            stats: SearchStats::default(),
        }
    }

    #[test]
    fn excluded_verification_is_isolated_and_restores_on_cancellation() {
        let board = Board::startpos();
        for budget in [Some(1), None] {
            let mut search = Search::new(1);
            let limits = Limits {
                nodes: budget,
                ..Limits::default()
            };
            let mut ctx = null_context(&mut search, &limits, board);
            let mv = board.parse_uci_move("e2e4").unwrap();
            ctx.pv[1].push(mv);
            ctx.prior[1] = Some(Played::new(&board, mv));
            ctx.static_evals[1] = Some(123);
            let score = ctx.verify_excluding(
                &board,
                3,
                1,
                10,
                NodeState {
                    history: 0,
                    extensions_left: 2,
                },
                mv,
            );
            assert_eq!(score.is_none(), budget.is_some());
            assert_eq!(ctx.excluded[1], None);
            assert!(!ctx.pruning_suppressed);
            assert_eq!(ctx.prior[1], Some(Played::new(&board, mv)));
            assert_eq!(ctx.static_evals[1], Some(123));
            assert_eq!(ctx.pv[1].as_ref(), &[mv]);
            assert_eq!(ctx.path, vec![board.hash()]);
            assert!(ctx.search.table.iter().all(|e| !e.valid));
            assert!(
                ctx.search
                    .history
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|&v| v == 0)
            );
            assert!(ctx.search.killers.iter().flatten().all(Option::is_none));
            assert!(ctx.search.correction.iter().all(|&v| v == 0));
            assert!(ctx.search.continuation.iter().flatten().all(|&v| v == 0));
            assert_eq!(ctx.null_probes, 0);
        }
    }

    #[test]
    fn singular_extends_only_a_legal_tt_move_with_remaining_budget() {
        // Qxd5 is the only move that wins the hanging queen. An intentionally
        // shallow lower bound is eligible for verification, not a TT cutoff.
        let board = Board::from_fen("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1").unwrap();
        let mv = board.parse_uci_move("d1d5").unwrap();
        for budget in [0, 1, 2] {
            let mut search = Search::new(1);
            search.set_singular_extensions(true);
            search.set_null_pruning(false);
            search.set_pruning(false);
            let limits = Limits::default();
            let mut ctx = null_context(&mut search, &limits, board);
            ctx.search.set_null_pruning(false);
            ctx.search.table.write(Entry {
                key: board.hash(),
                context: 0,
                halfmove: 0,
                extensions_left: budget,
                depth: 3,
                score: score_to_tt(500, 1),
                best: Some(mv),
                bound: Bound::Lower,
                generation: 0,
                valid: true,
            });
            ctx.node(
                &board,
                6,
                1,
                -INFINITY,
                INFINITY,
                NodeState {
                    history: 0,
                    extensions_left: budget,
                },
            )
            .unwrap();
            assert_eq!(ctx.stats.singular_probes > 0, budget > 0);
            assert_eq!(ctx.stats.singular_extensions > 0, budget > 0);
            assert!(ctx.stats.max_main_ply <= 6 + usize::from(budget));
            assert_eq!(ctx.excluded[1], None);
            assert_eq!(ctx.pv[1].first(), Some(&mv));
        }
    }

    #[test]
    fn singular_never_uses_other_score_contexts_upper_bounds_or_decisive_scores() {
        let board = Board::from_fen("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1").unwrap();
        let mv = board.parse_uci_move("d1d5").unwrap();
        for case in 0..6 {
            let mut search = Search::new(1);
            search.set_singular_extensions(true);
            search.set_pruning(false);
            let limits = Limits::default();
            let mut ctx = null_context(&mut search, &limits, board);
            ctx.search.set_null_pruning(false);
            let mut entry = Entry {
                key: board.hash(),
                context: 0,
                halfmove: 0,
                extensions_left: 2,
                depth: 3,
                score: 500,
                best: Some(mv),
                bound: Bound::Lower,
                generation: 0,
                valid: true,
            };
            match case {
                0 => entry.context = 1,
                1 => entry.halfmove = 1,
                2 => entry.extensions_left = 1,
                3 => entry.bound = Bound::Upper,
                4 => entry.score = crate::syzygy::TB_WIN_SCORE,
                _ => entry.best = Board::startpos().parse_uci_move("e2e4").ok(),
            }
            ctx.search.table.write(entry);
            ctx.node(
                &board,
                6,
                1,
                -INFINITY,
                INFINITY,
                NodeState {
                    history: 0,
                    extensions_left: 2,
                },
            )
            .unwrap();
            assert_eq!(ctx.stats.singular_probes, 0, "case {case}");
        }
    }

    #[test]
    fn null_guards_exclude_check_sparse_material_clocks_and_mate_windows() {
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut board = Board::startpos();
        let mut ctx = null_context(&mut search, &limits, board);
        assert!(ctx.can_try_null(&board, 6, 1, -101, -100));
        assert!(!ctx.can_try_null(&board, 6, 0, -101, -100));
        assert!(!ctx.can_try_null(&board, 3, 1, -101, -100));
        assert!(!ctx.can_try_null(&board, 6, 1, -200, 100));
        assert!(!ctx.can_try_null(&board, 6, 1, 100, 101));
        assert!(!ctx.can_try_null(&board, 6, 1, -MATE_SCORE, -MATE_SCORE + 1));
        board.halfmove_clock = 80;
        assert!(!ctx.can_try_null(&board, 6, 1, -101, -100));
        ctx.pruning_suppressed = true;
        assert!(!ctx.can_try_null(&Board::startpos(), 6, 1, -101, -100));
        ctx.pruning_suppressed = false;
        for fen in [
            "4r1k1/8/8/8/8/8/PP6/R2QK2R w KQ - 0 1",
            "8/8/8/8/2k5/8/2P5/2K5 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            assert!(!ctx.can_try_null(&board, 6, 1, -1001, -1000), "{fen}");
        }
    }

    #[test]
    fn static_cache_is_separate_from_draw_history_and_side() {
        let mut board = Board::startpos();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        assert_eq!(ctx.evaluate(&board, 0), evaluate(&board));
        board.halfmove_clock = 99;
        assert_eq!(ctx.evaluate(&board, 0), evaluate(&board));
        assert_eq!(ctx.stats.eval_hits, 1);
        board.side_to_move = board.side_to_move.other();
        assert_eq!(ctx.evaluate(&board, 0), evaluate(&board));
        assert_eq!(ctx.stats.eval_hits, 1, "side is part of the cache key");
        ctx.search.clear();
        ctx.evaluate(&board, 0);
        assert_eq!(ctx.stats.eval_hits, 1);
    }

    #[test]
    fn continuation_and_countermove_depend_on_previous_move_and_only_tried_quiets() {
        let start = Board::startpos();
        let played = start.parse_uci_move("e2e4").unwrap();
        let board = start.make_move(played);
        let winner = board.parse_uci_move("e7e5").unwrap();
        let failed = board.parse_uci_move("d7d5").unwrap();
        let untried = board.parse_uci_move("c7c5").unwrap();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        ctx.prior[0] = Some(Played::new(&start, played));
        ctx.reward_continuation(&board, winner, 6, &[failed], 1);
        assert!(ctx.quiet_history(&board, winner, 1) > 0);
        assert!(ctx.quiet_history(&board, failed, 1) < 0);
        assert_eq!(ctx.quiet_history(&board, untried, 1), 0);
        assert_eq!(ctx.move_priority(&board, winner, None, 1), 650_000);
        ctx.prior[0] = Some(Played {
            piece: Piece::Pawn,
            to: 29,
        });
        assert_eq!(ctx.quiet_history(&board, winner, 1), 0);
        assert_eq!(ctx.move_priority(&board, winner, None, 1), 0);
        ctx.search.clear();
        assert!(ctx.search.continuation.iter().flatten().all(|&v| v == 0));
        assert!(ctx.search.counters.iter().flatten().all(Option::is_none));
    }

    #[test]
    fn check_extensions_consume_a_shared_line_budget() {
        let board = Board::from_fen("r6k/8/8/8/8/8/1P6/KB6 w - - 0 1").unwrap();
        for budget in [0, 1, MAX_EXTENSIONS] {
            let mut search = classic_search(1);
            search.set_lmr(false);
            search.set_pruning(false);
            let limits = Limits::default();
            let mut ctx = null_context(&mut search, &limits, board);
            ctx.node(
                &board,
                2,
                1,
                -INFINITY,
                INFINITY,
                NodeState {
                    history: 0,
                    extensions_left: budget,
                },
            )
            .unwrap();
            assert_eq!(ctx.stats.extensions > 0, budget > 0);
            assert!(ctx.stats.max_main_ply <= 2 + usize::from(budget));
            assert_eq!(ctx.path, vec![board.hash()]);
        }
    }

    #[test]
    fn qsearch_skips_poisoned_capture_but_preserves_score_and_legal_pv() {
        let board = Board::from_fen("4k3/p7/4p3/3p3p/8/8/P7/3QK3 w - - 0 1").unwrap();
        let mut outcomes = Vec::new();
        for enabled in [false, true] {
            let mut search = classic_search(1);
            let limits = Limits::default();
            let mut ctx = null_context(&mut search, &limits, board);
            ctx.search.set_pruning(enabled);
            let score = ctx.quiescence(&board, 0, -2000, 2000).unwrap();
            let mut position = board;
            for &mv in &ctx.pv[0] {
                assert!(position.legal_moves().contains(&mv));
                position = position.make_move(mv);
            }
            outcomes.push((score, ctx.nodes, ctx.stats.q_pruned));
        }
        assert_eq!(outcomes[0].0, outcomes[1].0);
        assert!(outcomes[1].1 < outcomes[0].1);
        assert!(outcomes[1].2 > 0);
    }

    #[test]
    fn reverse_futility_only_cuts_eligible_non_pv_nodes() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        assert_eq!(ctx.negamax(&board, 1, 1, -1000, -999, 0), Some(-999));
        assert_eq!(ctx.stats.reverse_cutoffs, 1);
        ctx.stats.reverse_cutoffs = 0;
        ctx.negamax(&board, 1, 0, -1000, -999, 0).unwrap();
        assert_eq!(ctx.stats.reverse_cutoffs, 0);
        ctx.negamax(&board, 1, 1, -1000, 1000, 0).unwrap();
        assert_eq!(ctx.stats.reverse_cutoffs, 0);
        let sparse = Board::from_fen("8/8/8/8/2k5/8/2P5/2K5 w - - 0 1").unwrap();
        ctx.negamax(&sparse, 1, 1, -1000, -999, 0).unwrap();
        assert_eq!(ctx.stats.reverse_cutoffs, 0);
    }

    #[test]
    fn soft_time_stops_at_completed_iteration_and_keeps_hard_cap() {
        let board = Board::startpos();
        let result = classic_search(1).search(
            board,
            &[],
            Limits {
                depth: Some(9),
                soft_time: Some(Duration::ZERO),
                movetime: Some(Duration::from_secs(5)),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.depth, 3);
        assert!(board.legal_moves().contains(&result.best_move.unwrap()));
        let base = Duration::from_millis(100);
        assert!(soft_budget(base, 3, 0, false) < base);
        assert!(soft_budget(base, 0, 100, false) > base);
        assert!(soft_budget(base, 0, 0, true) < base);
    }

    #[test]
    fn null_probe_and_verification_do_not_pollute_real_history_or_tt() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        assert_eq!(ctx.negamax(&board, 4, 1, -101, -100, 0), Some(-100));
        assert_eq!(ctx.null_cutoffs, 1);
        assert_eq!(ctx.null_probes, 1, "nested null moves must be suppressed");
        assert_eq!(ctx.path, vec![board.hash()]);
        assert_eq!(ctx.repetition_floor, 0);
        assert!(!ctx.pruning_suppressed);
        assert!(ctx.search.table.iter().all(|e| !e.valid));
        assert!(
            ctx.search
                .history
                .iter()
                .flatten()
                .flatten()
                .all(|&v| v == 0)
        );
        assert!(ctx.search.killers.iter().flatten().all(Option::is_none));
    }

    #[test]
    fn cancelled_null_probe_restores_search_context() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        let limits = Limits {
            nodes: Some(1),
            ..Limits::default()
        };
        let mut ctx = null_context(&mut search, &limits, board);
        assert_eq!(ctx.negamax(&board, 4, 1, -101, -100, 0), None);
        assert_eq!(ctx.null_probes, 1);
        assert_eq!(ctx.path, vec![board.hash()]);
        assert_eq!(ctx.repetition_floor, 0);
        assert!(!ctx.pruning_suppressed);
        assert!(ctx.search.table.iter().all(|e| !e.valid));
    }

    #[test]
    fn cancelled_null_verification_restores_search_context() {
        let board = Board::startpos();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut probe = null_context(&mut search, &limits, board);
        probe.pruning_suppressed = true;
        let passed = board.null_move().unwrap();
        probe.path = vec![passed.hash()];
        assert!(probe.negamax(&passed, 1, 2, 100, 101, 0).unwrap() <= 100);
        let budget = 1 + probe.nodes;
        let limits = Limits {
            nodes: Some(budget),
            ..Limits::default()
        };
        // Exhaust the budget after the probe completes, before verification.
        let mut ctx = null_context(&mut search, &limits, board);
        assert_eq!(ctx.negamax(&board, 4, 1, -101, -100, 0), None);
        assert_eq!(ctx.nodes, budget);
        assert_eq!(ctx.null_probes, 1);
        assert_eq!(ctx.null_cutoffs, 0);
        assert_eq!(ctx.path, vec![board.hash()]);
        assert_eq!(ctx.repetition_floor, 0);
        assert!(!ctx.pruning_suppressed);
        assert!(ctx.search.table.iter().all(|e| !e.valid));
    }

    #[test]
    fn null_move_does_not_create_repetition_from_real_game_history() {
        let board =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5").unwrap();
        let passed = board.null_move().unwrap().hash();
        let mut search = classic_search(1);
        let limits = Limits::default();
        let mut ctx = null_context(&mut search, &limits, board);
        ctx.path = vec![passed, 1, passed, 2, board.hash()];
        let before = ctx.path.clone();
        assert_eq!(ctx.negamax(&board, 4, 1, 199, 200, 0), Some(200));
        assert_eq!(ctx.null_cutoffs, 1);
        assert_eq!(ctx.path, before);
    }

    #[test]
    fn null_pruning_is_inactive_in_pawn_and_rook_endgames() {
        for fen in [
            "8/8/8/8/2k5/8/2P5/2K5 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let limits = Limits {
                depth: Some(5),
                ..Limits::default()
            };
            let mut old = classic_search(1);
            old.set_null_pruning(false);
            let expected = old.search(
                board,
                &[],
                limits.clone(),
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            let mut enabled = classic_search(1);
            enabled.set_null_pruning(true);
            let actual =
                enabled.search(board, &[], limits, Arc::new(AtomicBool::new(false)), |_| {});
            assert_eq!(actual.score, expected.score);
            assert_eq!(actual.best_move, expected.best_move);
            assert_eq!(actual.nodes, expected.nodes);
            assert_eq!(actual.null_probes, 0);
        }
    }

    #[test]
    fn finds_mate_in_one_at_depth_one() {
        let (board, result) = search_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", 1);
        let after = board.make_move(result.best_move.unwrap());
        assert!(after.in_check(after.side_to_move));
        assert!(after.legal_moves().is_empty());
        assert_eq!(result.score, MATE_SCORE - 1);
    }

    #[test]
    fn recognizes_terminal_mate_and_stalemate() {
        let (_, mate) = search_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(mate.best_move, None);
        assert_eq!(mate.score, -MATE_SCORE);
        let (_, stalemate) = search_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(stalemate.best_move, None);
        assert_eq!(stalemate.score, 0);
    }

    #[test]
    fn takes_a_hanging_queen() {
        let (_, result) = search_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3);
        assert_eq!(result.best_move.unwrap().to_uci(), "d1d5");
        assert!(result.score > 300);
    }

    #[test]
    fn quiescence_searches_quiet_check_evasions_and_promotions() {
        let (score, nodes, pv) = quiescence_fen("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1");
        assert!(!is_mate_score(score));
        assert!(nodes > 1, "a checked king must make a quiet evasion");
        assert!(!pv.is_empty());
        let (score, _, pv) = quiescence_fen("7k/P7/8/8/8/8/8/4K3 w - - 0 1");
        assert!(score > 800);
        assert_eq!(pv[0].promotion, Some(Piece::Queen));
    }

    #[test]
    fn quiescence_detects_stalemate_before_stand_pat() {
        let (score, nodes, _) = quiescence_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
        assert_eq!(score, 0);
        assert_eq!(nodes, 1);
    }

    #[test]
    fn cancelled_search_has_legal_fallback() {
        let board = Board::startpos();
        let result = classic_search(1).search(
            board,
            &[],
            Limits::default(),
            Arc::new(AtomicBool::new(true)),
            |_| {},
        );
        assert!(board.legal_moves().contains(&result.best_move.unwrap()));
        assert_eq!(result.nodes, 0);
        assert_eq!(result.depth, 0);
    }

    #[test]
    fn node_limit_is_respected() {
        let board = Board::startpos();
        let result = classic_search(1).search(
            board,
            &[],
            Limits {
                nodes: Some(100),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert!(result.nodes <= 100);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn repetition_and_fifty_move_are_draws() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 8 5").unwrap();
        let key = board.hash();
        let result = classic_search(1).search(
            board,
            &[key, 1, key, 1, key],
            Limits {
                depth: Some(2),
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(result.score, 0);
        let (_, fifty) = search_fen("4k3/8/8/8/8/8/8/3QK3 w - - 100 51", 2);
        assert_eq!(fifty.score, 0);
        let (_, mate) = search_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 51", 2);
        assert_eq!(mate.score, -MATE_SCORE);
    }

    #[test]
    fn cached_search_respects_changed_fifty_move_counter() {
        let mut search = classic_search(1);
        let near_draw = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 98 50").unwrap();
        let fresh = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(near_draw.hash(), fresh.hash());
        let limits = Limits {
            depth: Some(2),
            ..Limits::default()
        };
        let drawn = search.search(
            near_draw,
            &[],
            limits.clone(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(drawn.score, 0);
        let winning = search.search(fresh, &[], limits, Arc::new(AtomicBool::new(false)), |_| {});
        assert!(winning.score > 700);
    }

    #[test]
    fn cached_search_respects_repetition_context_of_descendants() {
        let mut search = classic_search(1);
        let root = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 8 5").unwrap();
        let forced = root.parse_uci_move("e1f1").unwrap();
        let child = root.make_move(forced);
        let repeated = child
            .make_move(child.parse_uci_move("e8f8").unwrap())
            .hash();
        let limits = Limits {
            depth: Some(2),
            searchmoves: Some(vec![forced]),
            ..Limits::default()
        };
        let winning = search.search(
            root,
            &[root.hash()],
            limits.clone(),
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert!(winning.score > 700);
        // The same root and clock, but Black can now claim a third occurrence
        // of the grandchild. A cached child score from the first run is invalid.
        let drawn = search.search(
            root,
            &[repeated, 1, repeated, 2, root.hash()],
            limits,
            Arc::new(AtomicBool::new(false)),
            |_| {},
        );
        assert_eq!(drawn.score, 0);
    }

    #[test]
    fn searchmoves_restricts_root_even_after_cancellation() {
        let board = Board::startpos();
        let allowed = board.parse_uci_move("a2a3").unwrap();
        for cancelled in [false, true] {
            let result = classic_search(1).search(
                board,
                &[],
                Limits {
                    depth: Some(2),
                    searchmoves: Some(vec![allowed]),
                    ..Limits::default()
                },
                Arc::new(AtomicBool::new(cancelled)),
                |_| {},
            );
            assert_eq!(result.best_move, Some(allowed));
        }
    }

    #[test]
    fn mate_scores_round_trip_through_table() {
        for score in [0, 100, -500, MATE_SCORE - 5, -MATE_SCORE + 7] {
            assert_eq!(score_from_tt(score_to_tt(score, 3), 3), score);
        }
        assert_eq!(
            score_from_tt(score_to_tt(MATE_SCORE - 5, 3), 1),
            MATE_SCORE - 3
        );
    }

    #[test]
    fn selective_search_keeps_pv_legal_and_finishes_requested_depth() {
        for (_, fen) in crate::bench::POSITIONS {
            let board = Board::from_fen(fen).unwrap();
            let result = classic_search(1).search(
                board,
                &[board.hash()],
                Limits {
                    depth: Some(4),
                    ..Limits::default()
                },
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            assert!(result.depth == 4 || is_mate_score(result.score));
            assert_eq!(result.best_move, result.pv.first().copied());
            let mut position = board;
            for mv in result.pv {
                assert!(position.legal_moves().contains(&mv), "illegal PV: {fen}");
                position = position.make_move(mv);
            }
        }
    }

    #[test]
    fn selective_search_preserves_tactics_against_full_depth_reference() {
        for fen in [
            "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
            "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let mut reference = classic_search(1);
            reference.set_lmr(false);
            reference.set_pruning(false);
            let limits = Limits {
                depth: Some(4),
                ..Limits::default()
            };
            let full = reference.search(
                board,
                &[],
                limits.clone(),
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            let reduced = classic_search(1).search(
                board,
                &[],
                limits,
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            assert_eq!(reduced.best_move, full.best_move, "{fen}");
            assert_eq!(reduced.score, full.score, "{fen}");
        }
    }

    #[test]
    fn staged_ordering_preserves_full_depth_scores_without_reductions() {
        for (_, fen) in crate::bench::POSITIONS {
            let board = Board::from_fen(fen).unwrap();
            let mut legacy = classic_search(1);
            legacy.set_lmr(false);
            legacy.set_pruning(false);
            legacy.set_see_ordering(false);
            let mut staged = classic_search(1);
            staged.set_lmr(false);
            staged.set_pruning(false);
            let limits = Limits {
                depth: Some(3),
                ..Limits::default()
            };
            let old = legacy.search(
                board,
                &[],
                limits.clone(),
                Arc::new(AtomicBool::new(false)),
                |_| {},
            );
            let new = staged.search(board, &[], limits, Arc::new(AtomicBool::new(false)), |_| {});
            assert_eq!(new.score, old.score, "{fen}");
            assert!(board.legal_moves().contains(&new.best_move.unwrap()));
        }
    }
}

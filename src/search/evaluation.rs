//! Per-search lazy NNUE frames. The borrowed model is immutable and kept alive
//! by the search's local Arc; no accumulator can outlive or change its weights.
use super::{MAX_PLY, SearchStats};
use crate::{
    board::Board,
    nnue::{Accumulator, Network},
};

pub(super) struct NnueStack<'net> {
    network: &'net Network,
    boards: Vec<Board>,
    frames: Vec<Option<Accumulator<'net>>>,
    refreshes: u64,
    updates: u64,
    evaluations: u64,
    #[cfg(test)]
    pub refresh_on_evaluate: bool,
}

impl<'net> NnueStack<'net> {
    pub fn new(network: &'net Network, root: &Board) -> Self {
        let mut frames = vec![None; MAX_PLY];
        frames[0] = Some(network.accumulator(root));
        Self {
            network,
            boards: vec![*root; MAX_PLY],
            frames,
            refreshes: 1,
            updates: 0,
            evaluations: 0,
            #[cfg(test)]
            refresh_on_evaluate: false,
        }
    }

    pub fn enter(&mut self, board: &Board, ply: usize) {
        // Same-position verification and PVS re-search can reuse a valid frame.
        // A sibling with another board invalidates this ply, never its parent.
        if self.boards[ply] != *board {
            self.frames[ply] = None;
            self.boards[ply] = *board;
        }
    }

    pub fn evaluate(&mut self, ply: usize) -> i32 {
        self.evaluations += 1;
        #[cfg(test)]
        if self.refresh_on_evaluate {
            self.refreshes += 1;
            return self.network.accumulator(&self.boards[ply]).evaluate();
        }
        if self.frames[ply].is_none() {
            let ancestor = (0..ply).rev().find(|&i| self.frames[i].is_some());
            if let Some(ancestor) = ancestor {
                for i in ancestor + 1..=ply {
                    let mut frame = self.frames[i - 1].as_ref().unwrap().clone();
                    frame.update(&self.boards[i]);
                    self.frames[i] = Some(frame);
                    self.updates += 1;
                }
            } else {
                // Also handles a changed root in internal diagnostic callers.
                self.frames[ply] = Some(self.network.accumulator(&self.boards[ply]));
                self.refreshes += 1;
            }
        }
        let frame = self.frames[ply].as_ref().unwrap();
        #[cfg(test)]
        {
            let fresh = self.network.accumulator(&self.boards[ply]);
            for color in [crate::board::Color::White, crate::board::Color::Black] {
                assert_eq!(
                    frame.lanes(color),
                    fresh.lanes(color),
                    "NNUE ply={ply}: {}",
                    self.boards[ply].to_fen()
                );
            }
            assert_eq!(frame.raw_score(), fresh.raw_score());
        }
        frame.evaluate()
    }

    pub fn record(&self, stats: &mut SearchStats) {
        stats.nnue_refreshes = self.refreshes;
        stats.nnue_updates = self.updates;
        stats.nnue_evaluations = self.evaluations;
    }
}

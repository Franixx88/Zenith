//! Per-search lazy NNUE frames. The borrowed model is immutable and kept alive
//! by the search's local Arc; no accumulator can outlive or change its weights.
use super::{MAX_PLY, SearchStats};
use crate::{
    board::Board,
    nnue::{
        LegacyAccumulator, Network,
        v3::{AccumulatorCache, V3Accumulator},
    },
};

enum Frames<'net> {
    Legacy(Vec<Option<LegacyAccumulator<'net>>>),
    V3(Vec<Option<V3Accumulator<'net>>>, AccumulatorCache),
}

trait Frame: Clone {
    fn refreshes(&self) -> u32;
    fn score(&self) -> i32;
    #[cfg(test)]
    fn lanes_for(&self, color: crate::board::Color) -> &[i16];
    #[cfg(test)]
    fn raw(&self) -> i64;
}
macro_rules! frame_impl {
    ($ty:ident) => {
        impl Frame for $ty<'_> {
            fn refreshes(&self) -> u32 {
                self.bucket_refreshes()
            }
            fn score(&self) -> i32 {
                self.evaluate()
            }
            #[cfg(test)]
            fn lanes_for(&self, c: crate::board::Color) -> &[i16] {
                self.lanes(c)
            }
            #[cfg(test)]
            fn raw(&self) -> i64 {
                self.raw_score()
            }
        }
    };
}
frame_impl!(LegacyAccumulator);
frame_impl!(V3Accumulator);

fn evaluate_frames<F: Frame>(
    frames: &mut [Option<F>],
    boards: &[Board],
    ply: usize,
    create: impl Fn(&Board) -> F,
    mut update: impl FnMut(&mut F, &Board),
) -> (i32, u64, u64, u64) {
    let (mut refreshes, mut updates, mut buckets) = (0, 0, 0);
    if frames[ply].is_none() {
        if let Some(ancestor) = (0..ply).rev().find(|&i| frames[i].is_some()) {
            for i in ancestor + 1..=ply {
                let mut frame = frames[i - 1].as_ref().unwrap().clone();
                update(&mut frame, &boards[i]);
                buckets += u64::from(frame.refreshes());
                frames[i] = Some(frame);
                updates += 1;
            }
        } else {
            frames[ply] = Some(create(&boards[ply]));
            refreshes += 1;
        }
    }
    let frame = frames[ply].as_ref().unwrap();
    #[cfg(test)]
    {
        let fresh = create(&boards[ply]);
        for c in [crate::board::Color::White, crate::board::Color::Black] {
            assert_eq!(
                frame.lanes_for(c),
                fresh.lanes_for(c),
                "{}",
                boards[ply].to_fen()
            );
        }
        assert_eq!(frame.raw(), fresh.raw());
    }
    (frame.score(), refreshes, updates, buckets)
}

pub(super) struct NnueStack<'net> {
    network: &'net Network,
    boards: Vec<Board>,
    frames: Frames<'net>,
    refreshes: u64,
    bucket_refreshes: u64,
    updates: u64,
    evaluations: u64,
    #[cfg(test)]
    pub refresh_on_evaluate: bool,
}

impl<'net> NnueStack<'net> {
    pub fn new(network: &'net Network, root: &Board) -> Self {
        let frames = if network.v3_parameters().is_some() {
            let mut frames = vec![None; MAX_PLY];
            frames[0] = Some(network.v3_accumulator(root));
            Frames::V3(frames, AccumulatorCache::new())
        } else {
            let mut frames = vec![None; MAX_PLY];
            frames[0] = Some(network.legacy_accumulator(root));
            Frames::Legacy(frames)
        };
        Self {
            network,
            boards: vec![*root; MAX_PLY],
            frames,
            refreshes: 1,
            bucket_refreshes: 0,
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
            match &mut self.frames {
                Frames::Legacy(f) => f[ply] = None,
                Frames::V3(f, _) => f[ply] = None,
            };
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
        let (score, refreshes, updates, buckets) = match &mut self.frames {
            Frames::Legacy(f) => evaluate_frames(
                f,
                &self.boards,
                ply,
                |b| self.network.legacy_accumulator(b),
                |f, b| {
                    f.update(b);
                },
            ),
            Frames::V3(f, cache) => evaluate_frames(
                f,
                &self.boards,
                ply,
                |b| self.network.v3_accumulator(b),
                |f, b| {
                    f.update_cached(b, cache);
                },
            ),
        };
        self.refreshes += refreshes;
        self.updates += updates;
        self.bucket_refreshes += buckets;
        score
    }

    pub fn record(&self, stats: &mut SearchStats) {
        stats.nnue_refreshes = self.refreshes;
        stats.nnue_bucket_refreshes = self.bucket_refreshes;
        stats.nnue_updates = self.updates;
        stats.nnue_evaluations = self.evaluations;
        if let Frames::V3(_, cache) = &self.frames {
            stats.nnue_cache_hits = cache.hits;
            stats.nnue_cache_misses = cache.misses;
        }
    }
}

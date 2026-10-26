//! Root exclusion inside one iterative-deepening context and shared budget.
use super::*;

impl Context<'_> {
    pub(super) fn search_multiple(
        &mut self,
        board: &Board,
        history: u64,
        repeated: bool,
        mut result: SearchResult,
        on_info: &mut impl FnMut(SearchInfo),
    ) -> SearchResult {
        let roots = self.root_moves.clone();
        let count = self.search.multi_pv.min(roots.len());
        if self.search.worker_id == 0
            && !repeated
            && !self.should_stop(true)
            && let Some(tb) = self.search.syzygy.clone()
            && tb.eligible(board, self.search.probe_limit)
        {
            self.stats.tb_probes += 1;
            match tb.root_multiple(board, &roots, count, || self.should_stop(true)) {
                Ok(Some(lines)) => {
                    self.record_tb_hit();
                    result.variations = lines
                        .into_iter()
                        .map(|(mv, score)| RootVariation {
                            score,
                            pv: vec![mv],
                        })
                        .collect();
                    result.pv = result.variations[0].pv.clone();
                    result.best_move = result.pv.first().copied();
                    result.score = result.variations[0].score;
                    self.publish_multiple(&result, on_info);
                    return result;
                }
                Err(_) => self.stats.tb_errors += 1,
                Ok(None) => {}
            }
        }
        let mut stable = 0;
        let maximum = self.limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        for depth in (1 + (self.search.worker_id % 2) as u8).min(maximum)..=maximum {
            if self.should_stop(true) {
                break;
            }
            let mut remaining = roots.clone();
            let mut lines = Vec::with_capacity(count);
            for rank in 0..count {
                self.root_moves.clone_from(&remaining);
                let guess = result
                    .variations
                    .get(rank)
                    .map_or(result.score, |line| line.score);
                let mut window = if depth >= 4 { 35 } else { INFINITY };
                let mut alpha = if window == INFINITY {
                    -INFINITY
                } else {
                    guess - window
                };
                let mut beta = if window == INFINITY {
                    INFINITY
                } else {
                    guess + window
                };
                let score = loop {
                    let Some(score) =
                        self.negamax(board, i16::from(depth), 0, alpha, beta, history)
                    else {
                        break None;
                    };
                    if score <= alpha || score >= beta {
                        window = (window * 2).min(INFINITY);
                        alpha = if window == INFINITY {
                            -INFINITY
                        } else {
                            score - window
                        };
                        beta = if window == INFINITY {
                            INFINITY
                        } else {
                            score + window
                        };
                        continue;
                    }
                    break Some(score);
                };
                let Some(score) = score else {
                    break;
                };
                let pv = self.pv[0].to_vec();
                let Some(&best) = pv.first() else {
                    break;
                };
                debug_assert!(remaining.contains(&best));
                remaining.retain(|&mv| mv != best);
                lines.push(RootVariation { score, pv });
            }
            // Cancellation must never leave an exclusion for the next search.
            self.root_moves.clone_from(&roots);
            if lines.len() != count {
                break;
            }
            lines.sort_by_key(|line| std::cmp::Reverse(line.score));
            let best = lines[0].pv.first().copied();
            stable = if best == result.best_move {
                stable + 1
            } else {
                0
            };
            let drop = result.score - lines[0].score;
            result.best_move = best;
            result.depth = depth;
            result.score = lines[0].score;
            result.pv = lines[0].pv.clone();
            result.variations = lines;
            self.flush_nodes();
            self.publish_multiple(&result, on_info);
            if result.variations.iter().all(|line| {
                is_mate_score(line.score) && MATE_SCORE - line.score.abs() <= i32::from(depth)
            }) {
                break;
            }
            if depth >= 3
                && let Some(soft) = self.limits.soft_time
                && self.start.elapsed() >= soft_budget(soft, stable, drop, roots.len() == 1)
            {
                break;
            }
        }
        result
    }

    fn publish_multiple(&self, result: &SearchResult, on_info: &mut impl FnMut(SearchInfo)) {
        let nodes = self.total_nodes();
        let elapsed = self.start.elapsed();
        let tbhits = self.total_tbhits();
        for (rank, line) in result.variations.iter().enumerate() {
            on_info(SearchInfo {
                multipv: rank + 1,
                depth: result.depth,
                score: line.score,
                nodes,
                elapsed,
                pv: line.pv.clone(),
                tbhits,
            });
        }
    }
}

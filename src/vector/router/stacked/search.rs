use super::levels::{lists_to_open, StackedIvf, StackedSearchStats};
use crate::schema::Metric;
use crate::vector::router::adaptive::AdaptiveProbe;
use crate::vector::router::arena::VectorArena;
use crate::vector::router::candidate::Candidate;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::VectorStage;

/// Views one segment's stack or several segments' stacks as one logical router.
/// Construction controls the layer sources and where their work is recorded.
pub(crate) enum StackedRouting<'a, S> {
    Segment(&'a StackedIvf<S>),
    Global {
        indexes: Vec<Option<&'a StackedIvf<S>>>,
        totals: &'a mut [StageTotals],
        on_segment: &'a mut dyn FnMut(usize),
    },
}

/// Selects which scored leaf members the caller receives, without changing the walk.
pub(crate) trait RoutingOutput {
    fn record(&mut self, candidate: Candidate<(usize, u32)>);
    fn finish(self, best: AdaptiveProbe<(usize, u32)>) -> Vec<Candidate<(usize, u32)>>;
}

pub(crate) struct TopK;

impl RoutingOutput for TopK {
    fn record(&mut self, _candidate: Candidate<(usize, u32)>) {}

    fn finish(self, best: AdaptiveProbe<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        best.into_sorted_vec()
    }
}

#[derive(Default)]
pub(crate) struct AllScored(Vec<Candidate<(usize, u32)>>);

impl RoutingOutput for AllScored {
    fn record(&mut self, candidate: Candidate<(usize, u32)>) {
        self.0.push(candidate);
    }

    fn finish(self, _best: AdaptiveProbe<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        self.0
    }
}

pub(crate) struct StackedRanking {
    pub(crate) candidates: Vec<Candidate<(usize, u32)>>,
    pub(crate) stats: Vec<StackedSearchStats>,
}

impl<'a, S: VectorArena<Elem = f32>> StackedRouting<'a, S> {
    fn levels(&self) -> Vec<Vec<&'a StackedIvf<S>>> {
        let indexes = match self {
            Self::Segment(index) => vec![Some(*index)],
            Self::Global { indexes, .. } => indexes.clone(),
        };
        indexes
            .into_iter()
            .map(|mut index| {
                let mut levels = Vec::new();
                while let Some(current) = index {
                    levels.push(current);
                    index = current.parent.as_deref();
                }
                levels
            })
            .collect()
    }

    fn measure<R>(&mut self, segment: usize, op: impl FnOnce() -> R) -> R {
        match self {
            Self::Segment(_) => op(),
            Self::Global {
                totals, on_segment, ..
            } => {
                let (result, measured) = measure(|| {
                    let _stage = enter_vector_stage(VectorStage::Routing);
                    op()
                });
                totals[segment].append(&measured);
                on_segment(segment);
                result
            }
        }
    }

    /// Descends the combined layers, opening the nearest lists until their shared
    /// best candidates stop improving. Shorter stacks join at their topmost layer.
    pub(crate) fn search(
        mut self,
        query: &[f32],
        k: usize,
        metric: Metric,
        mut output: impl RoutingOutput,
    ) -> StackedRanking {
        let levels = self.levels();
        let mut stats = vec![StackedSearchStats::default(); levels.len()];
        if k == 0 {
            return StackedRanking {
                candidates: Vec::new(),
                stats,
            };
        }
        let depth = levels.iter().map(Vec::len).max().unwrap_or(0);
        let mut requested = k;
        let open: Vec<_> = (0..depth)
            .map(|level| {
                let indexes = levels.iter().filter_map(|stack| stack.get(level));
                let (lists, members) = indexes.fold((0, 0), |(lists, members), index| {
                    (
                        lists + index.nlist(),
                        members
                            + index
                                .offsets
                                .iter()
                                .map(|&(start, end)| (end - start) as usize)
                                .sum::<usize>(),
                    )
                });
                requested = lists_to_open(lists, members, requested);
                requested
            })
            .collect();
        let mut candidates = Vec::new();
        for level in (0..depth).rev() {
            for (segment, stack) in levels.iter().enumerate() {
                if stack.len() != level + 1 {
                    continue;
                }
                let index = stack[level];
                self.measure(segment, || {
                    candidates.extend((0..index.nlist()).map(|node| {
                        Candidate {
                            sim: index.centroids.similarity(
                                metric,
                                query.len(),
                                node as u32,
                                query,
                            ),
                            node: (segment, node as u32),
                        }
                    }));
                });
                stats[segment].members_scored += index.nlist();
            }
            candidates.sort_unstable_by(|a, b| b.cmp(a));
            candidates.truncate(open[level]);
            let requested = if level == 0 { k } else { open[level - 1] };
            let mut probe = AdaptiveProbe::new(requested);
            for candidate in candidates {
                let (segment, list) = candidate.node;
                let index = levels[segment][level];
                let scanned = self.measure(segment, || {
                    let (start, end) = index.offsets[list as usize];
                    for node in start..end {
                        let candidate = Candidate {
                            sim: index
                                .vectors
                                .similarity(metric, query.len(), node as u32, query),
                            node: (segment, node as u32),
                        };
                        probe.push(candidate);
                        if level == 0 {
                            output.record(candidate);
                        }
                    }
                    (end - start) as usize
                });
                stats[segment].lists_scanned += 1;
                stats[segment].members_scored += scanned;
                if probe.should_stop(scanned) {
                    break;
                }
            }
            if level == 0 {
                return StackedRanking {
                    candidates: output.finish(probe),
                    stats,
                };
            }
            candidates = probe.into_sorted_vec();
        }
        StackedRanking { candidates, stats }
    }
}

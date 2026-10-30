use super::levels::{lists_to_open, StackedIvf, StackedSearchStats};
#[cfg(test)]
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::router::adaptive::AdaptiveProbe;
use crate::vector::router::arena::VectorArena;
use crate::vector::router::candidate::Candidate;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::{VectorElement, VectorStage};

/// Views one segment's stack or several segments' stacks as one logical router.
/// Construction controls the layer sources and where their work is recorded.
pub(crate) enum StackedRouting<'a, S> {
    Segment(&'a StackedIvf<S>),
    Global {
        indexes: Vec<Option<&'a StackedIvf<S>>>,
        partitions: &'a [Option<&'a [f32]>],
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

enum RoutingLayer<'a, S> {
    Stored(&'a StackedIvf<S>),
    Partition {
        centroid: &'a [f32],
        child: &'a StackedIvf<S>,
        child_level: usize,
    },
}

impl<S: VectorArena<Elem = f32>> RoutingLayer<'_, S> {
    fn sizes(&self) -> (usize, usize) {
        match self {
            Self::Stored(index) => (
                index.nlist(),
                index.offsets.iter().map(|&(a, b)| (b - a) as usize).sum(),
            ),
            Self::Partition { child, .. } => (1, child.nlist()),
        }
    }

    fn score_centroids<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        scores: &mut Vec<Similarity>,
        mut visit: impl FnMut(Candidate<u32>),
    ) {
        match self {
            Self::Stored(index) => {
                index.centroids.batch_similarity(
                    query.widened(),
                    query.metric(),
                    query.routing_norm(),
                    0..index.nlist(),
                    scores,
                );
                for (node, &sim) in scores.iter().enumerate() {
                    visit(Candidate {
                        sim,
                        node: node as u32,
                    });
                }
            }
            Self::Partition { centroid, .. } => visit(Candidate {
                sim: query.metric().similarity_with_query_norm(
                    query.widened(),
                    centroid,
                    query.routing_norm(),
                ),
                node: 0,
            }),
        }
    }

    fn score_members<T: VectorElement>(
        &self,
        list: u32,
        query: &VectorQuery<T>,
        scores: &mut Vec<Similarity>,
        mut visit: impl FnMut(Candidate<u32>),
    ) -> usize {
        match self {
            Self::Stored(index) => {
                let (start, end) = index.offsets[list as usize];
                index.vectors.batch_similarity(
                    query.widened(),
                    query.metric(),
                    query.routing_norm(),
                    start as usize..end as usize,
                    scores,
                );
                for (node, &sim) in (start..end).zip(scores.iter()) {
                    visit(Candidate {
                        sim,
                        node: node as u32,
                    });
                }
                (end - start) as usize
            }
            Self::Partition { child, .. } => {
                child.centroids.batch_similarity(
                    query.widened(),
                    query.metric(),
                    query.routing_norm(),
                    0..child.nlist(),
                    scores,
                );
                for (node, &sim) in scores.iter().enumerate() {
                    visit(Candidate {
                        sim,
                        node: node as u32,
                    });
                }
                child.nlist()
            }
        }
    }

    fn child_level(&self, level: usize) -> Option<usize> {
        match self {
            Self::Stored(_) => level.checked_sub(1),
            Self::Partition { child_level, .. } => Some(*child_level),
        }
    }
}

impl<'a, S: VectorArena<Elem = f32>> StackedRouting<'a, S> {
    fn levels(&self) -> Vec<Vec<Option<RoutingLayer<'a, S>>>> {
        let (indexes, partitions) = match self {
            Self::Segment(index) => (vec![Some(*index)], &[][..]),
            Self::Global {
                indexes,
                partitions,
                ..
            } => (indexes.clone(), *partitions),
        };
        let mut stacks: Vec<Vec<_>> = indexes
            .into_iter()
            .map(|mut index| {
                let mut levels = Vec::new();
                while let Some(current) = index {
                    levels.push(Some(RoutingLayer::Stored(current)));
                    index = current.parent.as_deref();
                }
                levels
            })
            .collect();
        let depth = stacks.iter().map(Vec::len).max().unwrap_or(0);
        for (stack, centroid) in stacks.iter_mut().zip(partitions) {
            if let (Some(Some(RoutingLayer::Stored(child))), Some(centroid)) =
                (stack.last(), centroid)
            {
                let child = *child;
                let child_level = stack.len() - 1;
                stack.resize_with(depth, || None);
                stack.push(Some(RoutingLayer::Partition {
                    centroid,
                    child,
                    child_level,
                }));
            }
        }
        stacks
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
    /// best candidates stop improving. Stacks without a partition centroid join at their topmost
    /// layer.
    #[cfg(test)]
    pub(crate) fn search(
        self,
        query: &[f32],
        k: usize,
        metric: Metric,
        output: impl RoutingOutput,
    ) -> StackedRanking {
        self.search_prepared(
            &VectorQuery::new(metric, std::sync::Arc::new(query.to_vec())),
            k,
            output,
        )
    }

    pub(crate) fn search_prepared<T: VectorElement>(
        mut self,
        query: &VectorQuery<T>,
        k: usize,
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
        let mut open = Vec::with_capacity(depth);
        let mut beams = Vec::with_capacity(depth);
        for level in 0..depth {
            let layers: Vec<_> = levels
                .iter()
                .filter_map(|stack| stack.get(level)?.as_ref())
                .collect();
            let mut children: Vec<_> = layers
                .iter()
                .filter_map(|layer| layer.child_level(level))
                .collect();
            children.sort_unstable();
            children.dedup();
            // A layer spanning unequal depths needs room for each child layer's beam.
            let requested = if level == 0 {
                k
            } else {
                children
                    .into_iter()
                    .map(|child| open[child])
                    .fold(0usize, usize::saturating_add)
            };
            let (lists, members) = layers
                .iter()
                .map(|layer| layer.sizes())
                .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
            open.push(lists_to_open(lists, members, requested));
            beams.push(requested);
        }
        let mut pending = vec![Vec::new(); depth];
        let mut scores = Vec::new();
        for level in (0..depth).rev() {
            let mut candidates = std::mem::take(&mut pending[level]);
            for (segment, stack) in levels.iter().enumerate() {
                if stack.len() != level + 1 {
                    continue;
                }
                let layer = stack[level].as_ref().unwrap();
                self.measure(segment, || {
                    layer.score_centroids(query, &mut scores, |candidate| {
                        candidates.push(Candidate {
                            sim: candidate.sim,
                            node: (segment, candidate.node),
                        });
                    })
                });
                stats[segment].members_scored += layer.sizes().0;
            }
            candidates.sort_unstable_by(|a, b| b.cmp(a));
            candidates.truncate(open[level]);
            let mut probe = AdaptiveProbe::new(beams[level]);
            for candidate in candidates {
                let (segment, list) = candidate.node;
                let layer = levels[segment][level].as_ref().unwrap();
                let scanned = self.measure(segment, || {
                    layer.score_members(list, query, &mut scores, |candidate| {
                        let candidate = Candidate {
                            sim: candidate.sim,
                            node: (segment, candidate.node),
                        };
                        probe.push(candidate);
                        if level == 0 {
                            output.record(candidate);
                        }
                    })
                });
                stats[segment].lists_scanned += 1;
                stats[segment].members_scored += scanned;
                if probe.should_stop() {
                    break;
                }
            }
            if level == 0 {
                return StackedRanking {
                    candidates: output.finish(probe),
                    stats,
                };
            }
            for candidate in probe.into_sorted_vec() {
                let child_level = levels[candidate.node.0][level]
                    .as_ref()
                    .unwrap()
                    .child_level(level)
                    .unwrap();
                pending[child_level].push(candidate);
            }
        }
        StackedRanking {
            candidates: output.finish(AdaptiveProbe::new(k)),
            stats,
        }
    }
}

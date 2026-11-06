use super::levels::{ClusterId, StackedIvf, StackedSearchStats};
use crate::vector::router::arena::VectorArena;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::scoring::score_initial;
use crate::vector::router::search::{
    lists_to_open, rerank_exact, OutputPrecision, ScoreBuffer, SearchPolicy, TopCandidates,
};
use crate::vector::router::source::VectorSource;
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

/// Selects exact leaf output; approximate leaves always pass through bounded refinement.
pub(crate) trait RoutingOutput {
    fn record(&mut self, candidate: Candidate<(usize, u32)>);
    fn finish(self, best: TopCandidates<(usize, u32)>) -> Vec<Candidate<(usize, u32)>>;

    /// The walk opened `segment`'s `list` of `level`. Observers only; routing ignores it.
    fn opened(&mut self, _level: usize, _segment: usize, _list: u32) {}

    /// `best` is what `level` retained, before any exact rerank. Observers only.
    fn retained(&mut self, _level: usize, _best: &TopCandidates<(usize, u32)>) {}
}

pub(crate) struct TopK;

impl RoutingOutput for TopK {
    fn record(&mut self, _candidate: Candidate<(usize, u32)>) {}

    fn finish(self, best: TopCandidates<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        best.into_sorted()
    }
}

#[derive(Default)]
pub(crate) struct AllScored(Vec<Candidate<(usize, u32)>>);

impl RoutingOutput for AllScored {
    fn record(&mut self, candidate: Candidate<(usize, u32)>) {
        self.0.push(candidate);
    }

    fn finish(self, _best: TopCandidates<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        self.0
    }
}

pub(crate) struct StackedRanking {
    pub(crate) candidates: Vec<Candidate<(usize, u32)>>,
    pub(crate) stats: Vec<StackedSearchStats>,
}

/// A partition adds one coarse centroid above a segment's stored search tree.
enum RoutingLayer<'a, S> {
    Stored(&'a StackedIvf<S>),
    Partition {
        centroid: &'a [f32],
        child: &'a StackedIvf<S>,
        child_level: usize,
    },
}

impl<S: VectorSource> RoutingLayer<'_, S> {
    fn sizes(&self) -> (usize, usize) {
        match self {
            Self::Stored(index) => (
                index.nlist(),
                index.offsets.iter().map(|&(a, b)| (b - a) as usize).sum(),
            ),
            Self::Partition { child, .. } => (1, child.nlist()),
        }
    }

    fn centroids(&self) -> Option<&S> {
        match self {
            Self::Stored(index) => Some(&index.centroids),
            Self::Partition { .. } => None,
        }
    }

    fn members(&self) -> &S {
        match self {
            Self::Stored(index) => &index.vectors,
            Self::Partition { child, .. } => &child.centroids,
        }
    }

    fn score_centroids<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        out: &mut ScoreBuffer<ClusterId>,
    ) -> bool {
        match self {
            Self::Stored(index) => score_initial(&index.centroids, query, 0..index.nlist(), out),
            Self::Partition { centroid, .. } => {
                out.candidates.clear();
                out.candidates.push(Candidate {
                    sim: query.metric().similarity_with_query_norm(
                        query.widened(),
                        centroid,
                        query.routing_norm(),
                    ),
                    node: ClusterId(0),
                });
                false
            }
        }
    }

    fn score_members<T: VectorElement>(
        &self,
        list: u32,
        query: &VectorQuery<T>,
        out: &mut ScoreBuffer<ClusterId>,
    ) -> bool {
        let rows = match self {
            Self::Stored(index) => {
                let (start, end) = index.offsets[list as usize];
                start as usize..end as usize
            }
            Self::Partition { child, .. } => 0..child.nlist(),
        };
        score_initial(self.members(), query, rows, out)
    }

    fn child_level(&self, level: usize) -> Option<usize> {
        match self {
            Self::Stored(_) => level.checked_sub(1),
            Self::Partition { child_level, .. } => Some(*child_level),
        }
    }
}

impl<'a, S: VectorSource> StackedRouting<'a, S> {
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

    /// Refines only retained identities, grouped by segment and contiguous row runs.
    /// Banks without raw rows keep their estimates until the final routing level.
    fn refine<T: VectorElement>(
        &mut self,
        query: &VectorQuery<T>,
        retained: TopCandidates<(usize, u32)>,
        k: usize,
        sources: &[Option<&S>],
        stats: &mut [StackedSearchStats],
        leaf: bool,
    ) -> TopCandidates<(usize, u32)> {
        let mut selected = retained.into_unordered();
        selected.sort_unstable_by_key(|c| c.node);
        let mut best = TopCandidates::new(k);
        let mut start = 0;
        while start < selected.len() {
            let segment = selected[start].node.0;
            let end = start + selected[start..].partition_point(|c| c.node.0 == segment);
            let source = sources[segment];
            if let Some(raw) = source
                .filter(|source| source.compressed().is_some())
                .and_then(VectorSource::raw)
            {
                let rows: Vec<_> = selected[start..end]
                    .iter()
                    .map(|c| Candidate {
                        sim: c.sim,
                        node: ClusterId(c.node.1),
                    })
                    .collect();
                let exact = self.measure(segment, || {
                    let _stage = enter_vector_stage(VectorStage::RoutingExact);
                    rerank_exact(raw, query, &rows, rows.len())
                });
                stats[segment].members_scored += rows.len();
                if leaf {
                    stats[segment].leaf_exact_scored += rows.len();
                }
                for candidate in exact {
                    best.offer(Candidate {
                        sim: candidate.sim,
                        node: (segment, candidate.node.0),
                    });
                }
            } else {
                for &candidate in &selected[start..end] {
                    best.offer(candidate);
                }
            }
            start = end;
        }
        best
    }

    /// Descends shared layers, retaining one population across segments before refinement.
    pub(crate) fn search<T: VectorElement>(
        mut self,
        query: &VectorQuery<T>,
        k: usize,
        policy: SearchPolicy,
        precision: OutputPrecision,
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
        let has_compressed = levels.iter().flatten().flatten().any(|layer| {
            layer.members().compressed().is_some()
                || layer
                    .centroids()
                    .is_some_and(|source| source.compressed().is_some())
        });
        if has_compressed
            && (!query.routing_norm().is_finite() || query.widened().iter().any(|x| !x.is_finite()))
        {
            let mut best = TopCandidates::new(k);
            for (segment, stack) in levels.iter().enumerate() {
                let Some(Some(layer)) = stack.first() else {
                    continue;
                };
                let raw = layer
                    .members()
                    .raw()
                    .expect("final routing requires raw rows");
                let count = VectorArena::num_vectors(raw, query.widened().len());
                let mut scores = Vec::new();
                self.measure(segment, || {
                    raw.batch_similarity(
                        query.widened(),
                        query.metric(),
                        query.routing_norm(),
                        0..count,
                        &mut scores,
                    )
                });
                stats[segment].members_scored = count;
                stats[segment].leaf_exact_scored = count;
                for (node, sim) in scores.into_iter().enumerate() {
                    best.offer(Candidate {
                        sim,
                        node: (segment, node as u32),
                    });
                }
            }
            return StackedRanking {
                candidates: best.into_sorted(),
                stats,
            };
        }
        let depth = levels.iter().map(Vec::len).max().unwrap_or(0);
        let partition_level = levels
            .iter()
            .any(|stack| matches!(stack.last(), Some(Some(RoutingLayer::Partition { .. }))))
            .then(|| depth - 1);
        let mut open = Vec::with_capacity(depth);
        let mut requested = k;
        let mut partition_beam = 0;
        for level in 0..depth {
            let layers: Vec<_> = levels
                .iter()
                .filter_map(|stack| stack.get(level).and_then(Option::as_ref))
                .collect();
            if partition_level == Some(level) {
                requested = layers
                    .iter()
                    .filter_map(|layer| layer.child_level(level))
                    .map(|child| open[child])
                    .fold(0usize, usize::saturating_add);
                partition_beam = requested;
            }
            let (lists, members) = layers
                .iter()
                .map(|layer| layer.sizes())
                .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
            requested = lists_to_open(
                lists,
                members,
                requested,
                policy.min_fraction,
                policy.list_overscan,
            );
            open.push(requested);
        }
        let mut pending = vec![Vec::new(); partition_level.unwrap_or(0)];
        let mut candidates = Vec::new();
        let mut buffer = ScoreBuffer::default();
        for level in (0..depth).rev() {
            if let Some(direct) = pending.get_mut(level) {
                candidates.append(direct);
            }
            let roots: Vec<_> = levels
                .iter()
                .map(|stack| {
                    (stack.len() == level + 1)
                        .then(|| stack[level].as_ref())
                        .flatten()
                })
                .collect();
            for (segment, layer) in roots
                .iter()
                .enumerate()
                .filter_map(|(i, layer)| layer.map(|l| (i, l)))
            {
                let approximate =
                    self.measure(segment, || layer.score_centroids(query, &mut buffer));
                let count = buffer.candidates.len();
                stats[segment].members_scored += count;
                if approximate {
                    stats[segment].approximate_scored += count;
                }
                candidates.extend(buffer.candidates.drain(..).map(|c| Candidate {
                    sim: c.sim,
                    node: (segment, c.node.0),
                }));
            }
            let centroid_sources: Vec<_> = roots
                .iter()
                .map(|layer| layer.and_then(RoutingLayer::centroids))
                .collect();
            let root_refinement = centroid_sources
                .iter()
                .flatten()
                .any(|source| source.compressed().is_some() && source.raw().is_some());
            let capacity = padded_capacity(
                open[level],
                candidates.len(),
                root_refinement,
                policy.rerank_factor,
            );
            let mut retained = TopCandidates::new(capacity);
            for candidate in candidates.drain(..) {
                retained.offer(candidate);
            }
            candidates = if root_refinement {
                self.refine(
                    query,
                    retained,
                    open[level],
                    &centroid_sources,
                    &mut stats,
                    false,
                )
                .into_sorted()
            } else {
                retained.into_sorted()
            };
            let requested = if partition_level == Some(level) {
                partition_beam
            } else if level == 0 {
                k
            } else {
                open[level - 1]
            }
            .max(1);
            let sources: Vec<_> = levels
                .iter()
                .map(|stack| {
                    stack
                        .get(level)
                        .and_then(Option::as_ref)
                        .map(RoutingLayer::members)
                })
                .collect();
            let refine = (level > 0 || precision == OutputPrecision::Exact)
                && sources
                    .iter()
                    .flatten()
                    .any(|source| source.compressed().is_some() && source.raw().is_some());
            let members = levels
                .iter()
                .filter_map(|stack| stack.get(level).and_then(Option::as_ref))
                .map(|layer| layer.sizes().1)
                .sum();
            let capacity = padded_capacity(requested, members, refine, policy.rerank_factor);
            let mut probe = super::aps::open();
            let mut best = TopCandidates::new(capacity);
            for candidate in candidates {
                let (segment, list) = candidate.node;
                let layer = levels[segment][level].as_ref().unwrap();
                let approximate =
                    self.measure(segment, || layer.score_members(list, query, &mut buffer));
                let scanned = buffer.candidates.len();
                output.opened(level, segment, list);
                stats[segment].lists_scanned += 1;
                stats[segment].members_scored += scanned;
                if approximate {
                    stats[segment].approximate_scored += scanned;
                }
                if level == 0 {
                    stats[segment].leaf_members_scored += scanned;
                    if approximate {
                        stats[segment].leaf_approximate_scored += scanned;
                    } else {
                        stats[segment].leaf_exact_scored += scanned;
                    }
                }
                for candidate in buffer.candidates.drain(..) {
                    let candidate = Candidate {
                        sim: candidate.sim,
                        node: (segment, candidate.node.0),
                    };
                    best.offer(candidate);
                    if level == 0 && !refine {
                        output.record(candidate);
                    }
                }
                if policy.aps && probe.observe(best.worst_when_full()) {
                    break;
                }
            }
            output.retained(level, &best);
            if refine {
                best = self.refine(query, best, requested, &sources, &mut stats, level == 0);
            }
            if level == 0 {
                return StackedRanking {
                    candidates: if refine {
                        best.into_sorted()
                    } else {
                        output.finish(best)
                    },
                    stats,
                };
            }
            candidates = best.into_sorted();
            if partition_level == Some(level) {
                for candidate in candidates.drain(..) {
                    let child_level = levels[candidate.node.0][level]
                        .as_ref()
                        .unwrap()
                        .child_level(level)
                        .unwrap();
                    pending[child_level].push(candidate);
                }
            }
        }
        StackedRanking {
            candidates: output.finish(TopCandidates::new(0)),
            stats,
        }
    }
}

fn padded_capacity(k: usize, available: usize, refine: bool, factor: f32) -> usize {
    let requested = k.min(available);
    let factor = if refine { factor.max(1.0) } else { 1.0 };
    ((requested as f64 * f64::from(factor)).ceil() as usize)
        .max(requested)
        .min(available)
}

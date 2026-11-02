//! Level traversal and precision-stage composition over data-only vector sources.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::arena::VectorArena;
use super::candidate::Candidate;
use super::scoring::{score_initial, ScoringQuery};
use super::source::VectorSource;
use super::stacked::levels::{ClusterId, StackedIvf, StackedSearchStats};
use crate::vector::distance::Similarity;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

/// Reused output and representation-specific scratch for one scoring batch.
pub(crate) struct ScoreBuffer<I> {
    pub(crate) candidates: Vec<Candidate<I>>,
    pub(crate) similarities: Vec<Similarity>,
    pub(crate) estimates: Vec<f32>,
}

impl<I> Default for ScoreBuffer<I> {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            similarities: Vec::new(),
            estimates: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SearchPolicy {
    pub(crate) aps: bool,
    pub(crate) min_fraction: f32,
    pub(crate) rerank_factor: usize,
}

/// A caller explicitly chooses whether selection must produce exact scores.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum OutputPrecision {
    Estimated,
    Exact,
}

fn precision<S: VectorSource>(source: &S) -> OutputPrecision {
    if source.raw().is_some() {
        OutputPrecision::Exact
    } else {
        OutputPrecision::Estimated
    }
}

fn capacity<S: VectorSource>(
    source: &S,
    dim: usize,
    k: usize,
    output: OutputPrecision,
    policy: SearchPolicy,
) -> usize {
    let factor = if source.compressed().is_some() && output == OutputPrecision::Exact {
        policy.rerank_factor.max(1)
    } else {
        1
    };
    k.saturating_mul(factor).min(source.num_vectors(dim))
}

/// Refine a retained population, without opening additional clusters. Sources
/// provide representations; this operation owns final selection and its policy.
fn finish<S: VectorSource, T: VectorElement>(
    source: &S,
    query: &ScoringQuery<'_, T>,
    candidates: TopCandidates<ClusterId>,
    k: usize,
    output: OutputPrecision,
    stats: &mut StackedSearchStats,
) -> Vec<Candidate<ClusterId>> {
    if source.compressed().is_some() && output == OutputPrecision::Exact {
        let raw = source.raw().expect("exact refinement requires raw rows");
        let retained = candidates.into_unordered();
        stats.leaf_exact_scored = retained.len();
        stats.members_scored += retained.len();
        rerank_exact(raw, query.query, &retained, k)
    } else {
        candidates.into_sorted()
    }
}

/// Score/select any vector bank. Root centroids and flat member scans use the
/// same operations as the cluster walk, including optional exact refinement.
pub(crate) fn rank_vectors<S: VectorSource, T: VectorElement>(
    source: &S,
    query: &ScoringQuery<'_, T>,
    k: usize,
    policy: SearchPolicy,
    output: OutputPrecision,
) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
    assert!(
        output != OutputPrecision::Exact || source.raw().is_some(),
        "exact vector output requires raw rows"
    );
    let mut stats = StackedSearchStats::default();
    if k == 0 {
        return (Vec::new(), stats);
    }
    let mut buffer = ScoreBuffer::default();
    let approximate = score_initial(
        source,
        query,
        0..source.num_vectors(query.query.widened().len()),
        &mut buffer,
    );
    let scored = buffer.candidates.len();
    stats.members_scored = scored;
    stats.leaf_members_scored = scored;
    if approximate {
        stats.approximate_scored = scored;
        stats.leaf_approximate_scored = scored;
    } else {
        stats.leaf_exact_scored = scored;
    }
    let mut retained = TopCandidates::new(capacity(
        source,
        query.query.widened().len(),
        k,
        output,
        policy,
    ));
    for candidate in buffer.candidates {
        retained.offer(candidate);
    }
    let ranked = finish(source, query, retained, k, output, &mut stats);
    (ranked, stats)
}

/// Score whole clusters into one retained population; stopping is observed only
/// at completed-cluster boundaries and does not force precision refinement.
pub(crate) fn probe_clusters<S: VectorSource, T: VectorElement>(
    level: &StackedIvf<S>,
    query: &ScoringQuery<'_, T>,
    lists: Vec<Candidate<ClusterId>>,
    capacity: usize,
    aps: bool,
) -> (TopCandidates<ClusterId>, StackedSearchStats) {
    let mut stats = StackedSearchStats::default();
    let mut result = TopCandidates::new(capacity);
    if capacity == 0 {
        return (result, stats);
    }
    let mut buffer = ScoreBuffer::default();
    let mut stop = super::stacked::aps::open();
    for list in lists {
        let (start, end) = level.offsets[usize::from(list.node)];
        let approximate = score_initial(
            &level.vectors,
            query,
            start as usize..end as usize,
            &mut buffer,
        );
        let scored = buffer.candidates.len();
        stats.members_scored += scored;
        stats.leaf_members_scored += scored;
        stats.lists_scanned += 1;
        if approximate {
            stats.approximate_scored += scored;
            stats.leaf_approximate_scored += scored;
        } else {
            stats.leaf_exact_scored += scored;
        }
        for candidate in buffer.candidates.drain(..) {
            result.offer(candidate);
        }
        if aps && stop.observe(result.worst_when_full()) {
            break;
        }
    }
    (result, stats)
}

/// One walk for all level representations. Parent recursion receives the same
/// original query/context; representation-specific preparation is lazy.
pub(crate) fn search<S: VectorSource, T: VectorElement>(
    level: &StackedIvf<S>,
    query: &ScoringQuery<'_, T>,
    k: usize,
    policy: SearchPolicy,
    output: OutputPrecision,
) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
    let members = level.vectors.num_vectors(query.query.widened().len());
    if k == 0 || members == 0 {
        return (Vec::new(), StackedSearchStats::default());
    }
    assert!(
        output != OutputPrecision::Exact || level.vectors.raw().is_some(),
        "exact level output requires raw member rows"
    );
    let open = lists_to_open(level.offsets.len(), members, k, policy.min_fraction);
    let (lists, mut stats) = if let Some(parent) = &level.parent {
        search(parent, query, open, policy, precision(&parent.vectors))
    } else {
        rank_vectors(
            &level.centroids,
            query,
            open,
            policy,
            precision(&level.centroids),
        )
    };
    let (retained, probed) = probe_clusters(
        level,
        query,
        lists,
        capacity(
            &level.vectors,
            query.query.widened().len(),
            k,
            output,
            policy,
        ),
        policy.aps,
    );
    stats.members_scored += probed.members_scored;
    stats.lists_scanned += probed.lists_scanned;
    stats.approximate_scored += probed.approximate_scored;
    stats.leaf_members_scored = probed.leaf_members_scored;
    stats.leaf_approximate_scored = probed.leaf_approximate_scored;
    stats.leaf_exact_scored = probed.leaf_exact_scored;
    let ranked = finish(&level.vectors, query, retained, k, output, &mut stats);
    (ranked, stats)
}

/// Deterministic bounded retention, independent of precision and storage.
pub(crate) struct TopCandidates<I: Ord> {
    capacity: usize,
    heap: BinaryHeap<Reverse<Candidate<I>>>,
}

impl<I: Copy + Ord> TopCandidates<I> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity.min(1024)),
        }
    }

    pub(crate) fn offer(&mut self, candidate: Candidate<I>) {
        if self.capacity == 0 {
            return;
        }
        if self.heap.len() < self.capacity {
            self.heap.push(Reverse(candidate));
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if candidate > worst.0 {
                *worst = Reverse(candidate);
            }
        }
    }

    pub(crate) fn worst_when_full(&self) -> Option<Similarity> {
        self.heap
            .peek()
            .filter(|_| self.heap.len() == self.capacity)
            .map(|worst| worst.0.sim)
    }

    pub(crate) fn into_sorted(self) -> Vec<Candidate<I>> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect()
    }

    /// Retention does not require ordering when another precision stage will
    /// rescore these identities and choose the final order.
    pub(crate) fn into_unordered(self) -> Vec<Candidate<I>> {
        self.heap
            .into_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect()
    }
}

pub(crate) fn lists_to_open(lists: usize, members: usize, k: usize, min_fraction: f32) -> usize {
    use super::stacked::levels::{LIST_OVERSCAN, PARENT_NPROBE_FRACTION};

    if lists == 0 {
        return 0;
    }
    let min_fraction = if min_fraction.is_finite() {
        min_fraction.clamp(0.0, 1.0)
    } else {
        PARENT_NPROBE_FRACTION
    };
    let floor = ((lists as f32 * min_fraction).ceil() as usize).max(1);
    let beam = k.saturating_mul(LIST_OVERSCAN).max(1);
    let requested = ((beam as f64 * lists as f64) / members.max(1) as f64).ceil() as usize;
    requested.max(floor).min(lists).max(1)
}

/// Compose a final routing request. Flat scheduling and exceptional-query
/// fallback use the same bank scoring/refinement operations as the level walk.
pub(crate) fn rank<S: VectorSource, T: VectorElement>(
    level: &StackedIvf<S>,
    query: &VectorQuery<T>,
    params: super::RoutingParams,
) -> (Vec<Candidate<ClusterId>>, StackedSearchStats, bool) {
    let compressed = level.vectors.compressed().is_some();
    let count = level.vectors.num_vectors(query.widened().len());
    let k = if compressed {
        params.k.min(count)
    } else {
        params.k.clamp(1, count.max(1))
    };
    let flat = compressed && params.flat;
    if k == 0 {
        return (Vec::new(), StackedSearchStats::default(), flat);
    }
    let has_compressed = level.vectors.compressed().is_some()
        || level
            .levels()
            .any(|level| level.centroids.compressed().is_some());
    if has_compressed
        && (!query.routing_norm().is_finite() || query.widened().iter().any(|x| !x.is_finite()))
    {
        let candidates: Vec<_> = (0..count)
            .map(|row| Candidate {
                node: ClusterId::from(row),
                sim: Similarity::new(0.0),
            })
            .collect();
        let result = rerank_exact(
            level
                .vectors
                .raw()
                .expect("final routing requires raw rows"),
            query,
            &candidates,
            k,
        );
        let stats = StackedSearchStats {
            leaf_exact_scored: count,
            members_scored: count,
            leaf_members_scored: if compressed { 0 } else { count },
            ..Default::default()
        };
        return (result, stats, flat);
    }
    let context = ScoringQuery::new(query);
    let policy = SearchPolicy {
        aps: params.aps,
        rerank_factor: params.rerank_factor,
        min_fraction: if compressed {
            params.min_beam_fraction
        } else {
            super::stacked::levels::PARENT_NPROBE_FRACTION
        },
    };
    let (ranked, stats) = if flat {
        rank_vectors(&level.vectors, &context, k, policy, OutputPrecision::Exact)
    } else {
        search(level, &context, k, policy, OutputPrecision::Exact)
    };
    (ranked, stats, flat)
}

/// Exact reads are grouped into contiguous selected runs. The heap stores
/// canonical identities, so changing read order does not change tie breaking.
pub(crate) fn rerank_exact<S: VectorArena<Elem = f32>, T: VectorElement>(
    source: &S,
    query: &VectorQuery<T>,
    candidates: &[Candidate<ClusterId>],
    k: usize,
) -> Vec<Candidate<ClusterId>> {
    let mut rows: Vec<_> = candidates.iter().map(|c| c.node).collect();
    rows.sort_unstable();
    let mut result = TopCandidates::new(k);
    let mut scores = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && rows[end].0 == rows[end - 1].0 + 1 {
            end += 1;
        }
        let first = rows[start].0 as usize;
        source.batch_similarity(
            query.widened(),
            query.metric(),
            query.routing_norm(),
            first..first + end - start,
            &mut scores,
        );
        for (&node, &sim) in rows[start..end].iter().zip(&scores) {
            result.offer(Candidate { node, sim });
        }
        start = end;
    }
    result.into_sorted()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::schema::Metric;
    use crate::vector::router::source::{EncodingContext, VectorSet, ROUTER_SEED};
    use crate::vector::router::stacked::levels::InMemoryStore;
    use crate::vector::storage::quantized::sign::SignVectors;

    fn lists(best: Option<usize>) -> StackedIvf<InMemoryStore> {
        let mut members = vec![1.0; 40];
        if let Some(best) = best {
            members[best] = 0.0;
        }
        StackedIvf {
            config: Default::default(),
            parent: None,
            offsets: (0..40).map(|i| (i, i + 1)).collect(),
            centroids: InMemoryStore::new((0..40).map(|i| i as f32).collect(), 1),
            vectors: InMemoryStore::new(members, 1),
        }
    }

    #[test]
    fn retention_does_not_change_list_request_and_aps_waits_for_full_heap() {
        let level = lists(None);
        let base = VectorQuery::new(Metric::L2, Arc::new(vec![0.0]));
        let query = ScoringQuery::new(&base);
        let policy = SearchPolicy {
            aps: false,
            min_fraction: 0.25,
            rerank_factor: 1,
        };
        let open = lists_to_open(40, 40, 1, policy.min_fraction);
        let (lists, _) = rank_vectors(
            &level.centroids,
            &query,
            open,
            policy,
            OutputPrecision::Exact,
        );
        assert_eq!(lists.len(), 10);
        for capacity in [1, 4, 40] {
            let (_, stats) = probe_clusters(&level, &query, lists.clone(), capacity, false);
            assert_eq!(stats.lists_scanned, 10);
        }
        for (capacity, expected) in [(1, 6), (4, 9), (40, 10)] {
            let (_, stats) = probe_clusters(&level, &query, lists.clone(), capacity, true);
            assert_eq!(stats.lists_scanned, expected);
        }
        let (empty, stats) = search(&level, &query, 0, policy, OutputPrecision::Exact);
        assert!(empty.is_empty());
        assert_eq!(stats.members_scored, 0);
    }

    #[test]
    fn widening_selection_recovers_a_misranked_list_without_growing_retention() {
        let level = lists(Some(11));
        let base = VectorQuery::new(Metric::L2, Arc::new(vec![0.0]));
        let query = ScoringQuery::new(&base);
        let policy = SearchPolicy {
            aps: false,
            min_fraction: 0.25,
            rerank_factor: 1,
        };
        let (narrow, _) = search(&level, &query, 1, policy, OutputPrecision::Exact);
        let (wide, _) = search(
            &level,
            &query,
            1,
            SearchPolicy {
                min_fraction: 0.5,
                ..policy
            },
            OutputPrecision::Exact,
        );
        assert_eq!(narrow.len(), 1);
        assert_eq!(wide.len(), 1);
        assert!(wide[0].sim > narrow[0].sim);
        assert_eq!(wide[0].node.0, 11);
    }

    #[test]
    fn mixed_levels_share_query_preparation_and_refine_available_banks() -> crate::Result<()> {
        let dim = 8;
        let context = Arc::new(EncodingContext::new(dim, ROUTER_SEED));
        let encoded = |values: Vec<f32>| -> crate::Result<VectorSet<InMemoryStore>> {
            let raw = InMemoryStore::new(values, dim);
            Ok(VectorSet::Encoded {
                codes: Arc::new(SignVectors::encode(&raw, dim, &context.rotation)?),
                context: context.clone(),
                raw: Some(Arc::new(raw)),
            })
        };
        let raw_members: Vec<f32> = (0..8).flat_map(|i| vec![i as f32 * 0.25; dim]).collect();
        let middle = VectorSet::Raw(Arc::new(InMemoryStore::new(
            [0.125, 0.625, 1.125, 1.625]
                .into_iter()
                .flat_map(|x| vec![x; dim])
                .collect(),
            dim,
        )));
        let level = StackedIvf {
            config: Default::default(),
            parent: Some(Box::new(StackedIvf {
                config: Default::default(),
                parent: None,
                offsets: vec![(0, 2), (2, 4)],
                vectors: middle.clone(),
                centroids: encoded(
                    [0.375, 1.375]
                        .into_iter()
                        .flat_map(|x| vec![x; dim])
                        .collect(),
                )?,
            })),
            offsets: vec![(0, 2), (2, 4), (4, 6), (6, 8)],
            centroids: middle,
            vectors: encoded(raw_members.clone())?,
        };
        let base = VectorQuery::new(Metric::L2, Arc::new(vec![0.8; dim]));
        let query = ScoringQuery::new(&base);
        #[cfg(debug_assertions)]
        quant::rotation::debug_reset_apply_count();
        let policy = SearchPolicy {
            aps: false,
            min_fraction: 1.0,
            rerank_factor: 4,
        };
        let (actual, stats) = search(&level, &query, 2, policy, OutputPrecision::Exact);
        #[cfg(debug_assertions)]
        assert_eq!(quant::rotation::debug_apply_count(), 1);
        let mut expected: Vec<_> = raw_members
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, raw)| Candidate {
                node: ClusterId::from(i),
                sim: Metric::L2.similarity(base.widened(), raw),
            })
            .collect();
        expected.sort_unstable_by(|a, b| b.cmp(a));
        expected.truncate(2);
        assert_eq!(actual, expected);
        assert_eq!(stats.members_scored, 24); // all initial and refining scores
        assert_eq!(stats.approximate_scored, 10); // two root vectors + eight final members
        assert_eq!(stats.leaf_approximate_scored, 8);
        assert_eq!(stats.leaf_exact_scored, 8);
        assert_eq!(stats.lists_scanned, 6); // two parent lists + four child lists
        let router = super::super::stacked::StackedRouter::new(
            level,
            &crate::schema::VectorOptions::new(dim, Metric::L2),
        )?;
        assert!(router.serialize(&mut Vec::new()).is_err());
        Ok(())
    }
}

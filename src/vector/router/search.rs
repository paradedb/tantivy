//! Storage-independent level search. Sources score batches; the caller composes
//! list selection, candidate retention, stopping, and optional exact reranking.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::candidate::Candidate;
use super::stacked::levels::StackedSearchStats;
use crate::vector::distance::Similarity;

/// Reused across batches, including representation-specific scoring scratch.
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

/// A source resolves storage and scores whole batches, without deciding how
/// many candidates to retain or when to stop. Cluster and member identities may
/// differ (for example, a posting list contains addressed document-vector rows).
pub(crate) trait LevelSource<Q>: Sized {
    type ClusterId: Copy + Ord;
    type MemberId: Copy + Ord;

    fn list_count(&self) -> usize;
    fn member_count(&self) -> usize;
    fn score_centroids(&self, query: &Q, out: &mut ScoreBuffer<Self::ClusterId>);
    fn score_members(
        &self,
        list: Self::ClusterId,
        query: &Q,
        out: &mut ScoreBuffer<Self::MemberId>,
    );
}

/// Hierarchy composition is separate from batch access. Parent results must
/// identify child clusters; leaf sources need not implement this trait to probe.
pub(crate) trait Hierarchy<Q>: LevelSource<Q> {
    type Parent: Hierarchy<Q> + LevelSource<Q, MemberId = Self::ClusterId>;
    fn parent(&self) -> Option<&Self::Parent>;
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

pub(crate) fn lists_to_open(lists: usize, members: usize, k: usize) -> usize {
    use super::stacked::levels::{LIST_OVERSCAN, PARENT_NPROBE_FRACTION};

    if lists == 0 {
        return 0;
    }
    let floor = ((lists as f32 * PARENT_NPROBE_FRACTION).ceil() as usize).max(1);
    let beam = k.saturating_mul(LIST_OVERSCAN).max(1);
    let requested = ((beam as f64 * lists as f64) / members.max(1) as f64).ceil() as usize;
    requested.max(floor).min(lists).max(1)
}

/// Score and select centroids without requiring a hierarchy or probing members.
pub(crate) fn rank_centroids<Q, S: LevelSource<Q>>(
    source: &S,
    query: &Q,
    k: usize,
) -> (Vec<Candidate<S::ClusterId>>, StackedSearchStats) {
    let mut buffer = ScoreBuffer::default();
    source.score_centroids(query, &mut buffer);
    let stats = StackedSearchStats {
        lists_scanned: 0,
        members_scored: buffer.candidates.len(),
        leaf_members_scored: 0,
    };
    let mut selected = TopCandidates::new(k);
    for candidate in buffer.candidates {
        selected.offer(candidate);
    }
    (selected.into_sorted(), stats)
}

/// Probe any supplied ranked shortlist. Representation and hierarchy are source
/// concerns; retention and stopping are explicit policies of this invocation.
/// The caller orders the retained set only when its next stage needs ranking.
pub(crate) fn probe_clusters<Q, S: LevelSource<Q>>(
    source: &S,
    query: &Q,
    lists: Vec<Candidate<S::ClusterId>>,
    capacity: usize,
    aps: bool,
) -> (TopCandidates<S::MemberId>, StackedSearchStats) {
    let mut stats = StackedSearchStats::default();
    if capacity == 0 {
        return (TopCandidates::new(0), stats);
    }
    let mut buffer = ScoreBuffer::default();
    let mut result = TopCandidates::new(capacity.min(source.member_count()));
    let mut stop = super::stacked::aps::open();
    for list in lists {
        source.score_members(list.node, query, &mut buffer);
        stats.members_scored += buffer.candidates.len();
        stats.leaf_members_scored += buffer.candidates.len();
        stats.lists_scanned += 1;
        for candidate in buffer.candidates.drain(..) {
            result.offer(candidate);
        }
        if aps && stop.observe(result.worst_when_full()) {
            break;
        }
    }
    (result, stats)
}

/// `k` sizes the traversal; `capacity` independently sizes the output heap.
/// Parents produce ordinary, unpadded shortlists. Precision is a source concern.
/// Parent results are ordered for descent; output ordering belongs to the caller.
pub(crate) fn search<Q, S: Hierarchy<Q>>(
    source: &S,
    query: &Q,
    k: usize,
    capacity: usize,
    aps: bool,
) -> (TopCandidates<S::MemberId>, StackedSearchStats) {
    if k == 0 || capacity == 0 || source.member_count() == 0 {
        return (TopCandidates::new(0), StackedSearchStats::default());
    }
    let open = lists_to_open(source.list_count(), source.member_count(), k);
    let (lists, mut stats) = if let Some(parent) = source.parent() {
        let (selected, stats) = search(parent, query, open, open, aps);
        (selected.into_sorted(), stats)
    } else {
        rank_centroids(source, query, open)
    };
    let (candidates, probed) = probe_clusters(source, query, lists, capacity, aps);
    stats.members_scored += probed.members_scored;
    stats.lists_scanned += probed.lists_scanned;
    stats.leaf_members_scored = probed.leaf_members_scored;
    (candidates, stats)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct Source {
        lists: usize,
        scored: Cell<usize>,
    }

    impl LevelSource<()> for Source {
        type ClusterId = u32;
        type MemberId = u32;
        fn list_count(&self) -> usize {
            self.lists
        }
        fn member_count(&self) -> usize {
            self.lists
        }
        fn score_centroids(&self, _: &(), out: &mut ScoreBuffer<u32>) {
            out.candidates = (0..self.lists)
                .map(|node| Candidate {
                    node: node as u32,
                    sim: Similarity::new(-(node as f32)),
                })
                .collect();
        }
        fn score_members(&self, list: u32, _: &(), out: &mut ScoreBuffer<u32>) {
            self.scored.set(self.scored.get() + 1);
            out.candidates.clear();
            out.candidates.push(Candidate {
                node: list,
                sim: Similarity::new(1.0),
            });
        }
    }

    impl Hierarchy<()> for Source {
        type Parent = Self;
        fn parent(&self) -> Option<&Self> {
            None
        }
    }

    #[test]
    fn retention_does_not_change_list_request_and_aps_waits_for_full_heap() {
        let source = Source {
            lists: 40,
            scored: Cell::new(0),
        };
        // Fixed k=1 selects ten lists. Changing retention alone preserves that.
        for capacity in [1, 4, 40] {
            let (_, stats) = search(&source, &(), 1, capacity, false);
            assert_eq!(stats.lists_scanned, 10);
        }
        let (_, stats) = search(&source, &(), 1, 1, true);
        assert_eq!(stats.lists_scanned, 6); // first full list + five quiet lists
        let (_, stats) = search(&source, &(), 1, 4, true);
        assert_eq!(stats.lists_scanned, 9); // four filling lists + five quiet lists
        let (_, stats) = search(&source, &(), 1, 40, true);
        assert_eq!(stats.lists_scanned, 10); // never fills
        source.scored.set(0);
        assert!(search(&source, &(), 0, 0, true)
            .0
            .into_unordered()
            .is_empty());
        assert_eq!(source.scored.get(), 0);
    }

    #[test]
    fn probing_preserves_addressed_members_without_a_parent_or_shared_id_type() {
        use crate::vector::write::RowAddress;

        struct PostingLists;
        impl LevelSource<()> for PostingLists {
            type ClusterId = u32;
            type MemberId = RowAddress;
            fn list_count(&self) -> usize {
                2
            }
            fn member_count(&self) -> usize {
                4
            }
            fn score_centroids(&self, _: &(), out: &mut ScoreBuffer<u32>) {
                out.candidates = vec![
                    Candidate {
                        node: 4,
                        sim: Similarity::new(2.0),
                    },
                    Candidate {
                        node: 6,
                        sim: Similarity::new(3.0),
                    },
                ];
            }
            fn score_members(&self, list: u32, _: &(), out: &mut ScoreBuffer<RowAddress>) {
                let members = match list {
                    4 => [(10, 8.0), (11, 1.0)],
                    6 => [(40, 2.0), (41, 9.0)],
                    _ => unreachable!(),
                };
                out.candidates = members
                    .into_iter()
                    .map(|(row_id, score)| Candidate {
                        node: RowAddress {
                            segment_ord: 9,
                            row_id,
                        },
                        sim: Similarity::new(score),
                    })
                    .collect();
            }
        }
        let (lists, _) = rank_centroids(&PostingLists, &(), 2);
        assert_eq!(lists.iter().map(|c| c.node).collect::<Vec<_>>(), [6, 4]);
        let (members, stats) = probe_clusters(&PostingLists, &(), lists, 2, false);
        let members = members.into_sorted();
        assert_eq!(
            members.iter().map(|c| c.node).collect::<Vec<_>>(),
            [
                RowAddress {
                    segment_ord: 9,
                    row_id: 41
                },
                RowAddress {
                    segment_ord: 9,
                    row_id: 10
                },
            ]
        );
        assert_eq!(stats.members_scored, 4);
    }
}

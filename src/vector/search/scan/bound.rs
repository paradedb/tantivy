//! The pruning bounds of one query's scans.
//!
//! The collector holds two things every segment scan shares. `collector_topk` is the top-k exact
//! scores from finished segments (seeded into this bound as [`RowAddress`] keys).
//! `shared_topk_threshold` is the threshold participants publish; when the caller provides none,
//! the collector builds a process-local [`AtomicSharedThreshold`] shared by every segment of that
//! search.
//!
//! A [`PruningBound`] is one segment scan. Its [`TopK`] is keyed by [`RowAddress`]: `offer` upserts
//! by address (refine or exact replaces the same row). Once the heap holds `k` distinct addresses,
//! the k-th lower may rise and `offer` CASes the shared slot. Every prune decision loads that one
//! floor via [`PruningBound::shared_threshold`]. An exact score is a zero-width interval. Admission
//! drops a row only when its upper endpoint is strictly below the floor.

use std::collections::HashMap;
use std::sync::Arc;

use crate::collector::SharedThreshold;
use crate::vector::write::RowAddress;
use crate::{Score, SegmentOrdinal};

// --- Scores ---

/// One doc's score bound. An exact score is a zero-width interval: every endpoint is the score.
///
/// The upper endpoint of a quantized row is consumed at admission. Later layers recompute it from
/// the refined columns, so an interval stored here keeps the lower endpoint and the point estimate.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bound {
    Exact(f32),
    Interval { lo: f32, est: f32 },
}

impl Bound {
    pub(crate) fn lower(self) -> f32 {
        match self {
            Bound::Exact(score) | Bound::Interval { lo: score, .. } => score,
        }
    }

    pub(crate) fn estimate(self) -> f32 {
        match self {
            Bound::Exact(score) | Bound::Interval { est: score, .. } => score,
        }
    }
}

/// A document and the bound admission computed for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub(crate) address: RowAddress,
    pub(crate) bound: Bound,
}

impl Candidate {
    pub(crate) fn exact(address: RowAddress, score: f32) -> Self {
        Self {
            address,
            bound: Bound::Exact(score),
        }
    }

    pub(crate) fn interval(address: RowAddress, lo: f32, est: f32) -> Self {
        Self {
            address,
            bound: Bound::Interval { lo, est },
        }
    }
}

// --- Shared threshold ---

/// Raises `shared_topk_threshold` to `score` when `score` is finite and strictly above the value
/// loaded. Returns whether this call stored it.
fn raise(shared_topk_threshold: &dyn SharedThreshold<Score>, score: f32) -> bool {
    if !score.is_finite() {
        return false;
    }
    let mut current = shared_topk_threshold.load();
    loop {
        if current.is_some_and(|(shared, _)| shared >= score) {
            return false;
        }
        match shared_topk_threshold.try_update(&current, (score, SegmentOrdinal::MAX)) {
            Ok(()) => return true,
            Err(actual) => current = actual,
        }
    }
}

/// What merging exact scores into a collector [`TopK`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExactMerge {
    /// Finite scores offered.
    pub(crate) offers: usize,
    /// Offers that set the k-th or raised it.
    pub(crate) tightenings: usize,
}

// --- Top-k ---

/// The `k` best candidates by lower endpoint, keyed by [`RowAddress`].
///
/// Exact scores and quantized intervals share one map. `offer` / upsert replaces the same address
/// in place so an exact cannot double-count with a leftover interval. Once every entry is exact,
/// [`TopK::kth_lower`] is the exact-only k-th.
#[derive(Clone, Debug)]
pub(crate) struct TopK {
    k: usize,
    by_addr: HashMap<RowAddress, Bound>,
}

impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            by_addr: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// The entries currently held, for seeding another bound.
    pub(crate) fn candidates(&self) -> impl Iterator<Item = Candidate> + '_ {
        self.by_addr
            .iter()
            .map(|(&address, &bound)| Candidate { address, bound })
    }

    /// The worst lower endpoint once `k` candidates are in.
    pub(crate) fn kth_lower(&self) -> Option<f32> {
        if self.k == 0 || self.by_addr.len() != self.k {
            return None;
        }
        let lower = self
            .by_addr
            .values()
            .map(|bound| bound.lower())
            .min_by(|a, b| a.total_cmp(b))?;
        lower.is_finite().then_some(lower)
    }

    /// The k-th of the estimates in the map, once it is full. It carries no confidence margin.
    pub(crate) fn kth_estimate(&self) -> Option<f32> {
        if self.k == 0 || self.by_addr.len() != self.k {
            return None;
        }
        self.by_addr
            .values()
            .map(|bound| bound.estimate())
            .min_by(|a, b| a.total_cmp(b))
    }

    /// The k-th exact score once the map holds `k` exact scores and nothing else.
    #[cfg(test)]
    pub(crate) fn exact_kth(&self) -> Option<f32> {
        let exact = self.kth_lower()?;
        self.by_addr
            .values()
            .all(|bound| matches!(bound, Bound::Exact(_)))
            .then_some(exact)
    }

    /// Upserts `candidate` by [`RowAddress`]. A new address enters when the map has room or it
    /// beats the current worst lower; an existing address keeps the better (higher) lower
    /// endpoint. Exact scores and refinements should only raise a row's lower; a lower exact is a
    /// bug. Interval re-offers after a layer refine may dip slightly, so those keep the better
    /// lower without asserting.
    pub(crate) fn upsert(&mut self, candidate: Candidate) {
        if self.k == 0 {
            return;
        }
        let Candidate { address, bound } = candidate;
        if let Some(existing) = self.by_addr.get(&address) {
            let existing_lower = existing.lower();
            let new_lower = bound.lower();
            if new_lower < existing_lower {
                debug_assert!(
                    !matches!(existing, Bound::Exact(_)),
                    "exact score for {address:?} must not be lowered: existing {existing_lower} \
                     new {new_lower}"
                );
                return;
            }
            self.by_addr.insert(address, bound);
            return;
        }
        if self.by_addr.len() < self.k {
            self.by_addr.insert(address, bound);
            return;
        }
        // Full map: linear scan for the worst entry is fine while k (= limit + offset) is small.
        let Some((worst_addr, worst_lower)) = self.worst() else {
            return;
        };
        let lower = bound.lower();
        let beats = lower
            .total_cmp(&worst_lower)
            .then_with(|| worst_addr.cmp(&address))
            .is_gt();
        if beats {
            self.by_addr.remove(&worst_addr);
            self.by_addr.insert(address, bound);
        }
    }

    fn worst(&self) -> Option<(RowAddress, f32)> {
        self.by_addr
            .iter()
            .map(|(&addr, bound)| (addr, bound.lower()))
            .min_by(|(a_addr, a_lo), (b_addr, b_lo)| {
                a_lo.total_cmp(b_lo).then_with(|| a_addr.cmp(b_addr))
            })
    }

    /// Takes exact scores from a finished segment. Non-finite scores are skipped.
    pub(crate) fn merge_exact(
        &mut self,
        scores: impl IntoIterator<Item = (RowAddress, f32)>,
    ) -> ExactMerge {
        let mut merge = ExactMerge::default();
        if self.k == 0 {
            return merge;
        }
        for (address, score) in scores {
            if !score.is_finite() {
                continue;
            }
            merge.offers += 1;
            let before = self.kth_lower();
            self.upsert(Candidate::exact(address, score));
            if self
                .kth_lower()
                .is_some_and(|after| before.is_none_or(|before| after > before))
            {
                merge.tightenings += 1;
            }
        }
        merge
    }

    /// Drops every candidate.
    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.by_addr.clear();
    }
}

// --- Pruning bound ---

/// One segment scan's pruning bound: a local [`TopK`] keyed by [`RowAddress`] and the shared slot.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct PruningBound {
    top: TopK,
    shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
    /// Highest threshold offered to `shared_topk_threshold`.
    last_published: f32,
    /// Successful CASes into `shared_topk_threshold`.
    pub(super) publishes: usize,
}

impl PruningBound {
    /// A bound for `k` docs publishing through `shared_topk_threshold`.
    pub(crate) fn new(k: usize, shared_topk_threshold: Arc<dyn SharedThreshold<Score>>) -> Self {
        Self {
            top: TopK::new(k),
            shared_topk_threshold,
            last_published: f32::NEG_INFINITY,
            publishes: 0,
        }
    }

    /// Seeds from the collector's exact top-k (partial fill when prior segments returned fewer than
    /// `k` hits). CASes the slot when the seed already holds `k` values.
    pub(crate) fn seed(&mut self, collector: &TopK) {
        self.offer(collector.candidates());
    }

    /// Finite score currently in the shared slot, if any.
    pub(crate) fn shared_threshold(&self) -> Option<f32> {
        self.shared_topk_threshold
            .load()
            .map(|(score, _)| score)
            .filter(|score| score.is_finite())
    }

    /// Upserts competitive bounds by [`RowAddress`]. When the local top holds `k` values and the
    /// k-th is above the slot, CASes the slot.
    pub(crate) fn offer(&mut self, scores: impl IntoIterator<Item = Candidate>) {
        let mut any = false;
        for candidate in scores {
            any = true;
            self.top.upsert(candidate);
        }
        if any {
            self.publish();
        }
    }

    /// Local TopK k-th lower. Production prune/router paths use [`Self::shared_threshold`].
    #[cfg(test)]
    pub(crate) fn kth_lower(&self) -> Option<f32> {
        self.top.kth_lower()
    }

    /// Local TopK k-th estimate for adaptive stopping.
    pub(crate) fn kth_estimate(&self) -> Option<f32> {
        self.top.kth_estimate()
    }

    /// Local TopK exact-only k-th.
    #[cfg(test)]
    pub(super) fn exact_kth(&self) -> Option<f32> {
        self.top.exact_kth()
    }

    #[cfg(test)]
    pub(super) fn k(&self) -> usize {
        self.top.k()
    }

    /// Offers the local k-th to the shared slot when it is finite and above the last offer.
    fn publish(&mut self) {
        let Some(kth) = self.top.kth_lower() else {
            return;
        };
        if kth <= self.last_published {
            return;
        }
        self.last_published = kth;
        if raise(&*self.shared_topk_threshold, kth) {
            self.publishes += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::write::RowAddress;

    fn addr(row: u32) -> RowAddress {
        RowAddress {
            segment_ord: 0,
            row_id: row as usize,
        }
    }

    mod top_k {
        use super::addr;
        use crate::vector::search::scan::bound::{Candidate, ExactMerge, TopK};
        use crate::DocId;

        /// Same-address upserts keep the higher lower endpoint and never rewrite with a worse
        /// interval. A raised exact replaces the interval.
        #[test]
        fn test_upsert_keeps_better_lower_for_same_address() {
            let mut top = TopK::new(1);
            top.upsert(Candidate::interval(addr(1), 6.0, 11.0));
            assert_eq!(top.kth_lower(), Some(6.0));
            top.upsert(Candidate::exact(addr(1), 8.0));
            assert_eq!(
                top.exact_kth(),
                Some(8.0),
                "exact raises and replaces the interval"
            );
            top.upsert(Candidate::exact(addr(1), 8.0));
            assert_eq!(
                top.exact_kth(),
                Some(8.0),
                "equal lower may replace in place"
            );

            let mut top = TopK::new(1);
            top.upsert(Candidate::interval(addr(2), 5.0, 10.0));
            top.upsert(Candidate::interval(addr(2), 4.0, 9.0));
            assert_eq!(
                top.kth_lower(),
                Some(5.0),
                "a lower interval must not rewrite"
            );
            top.upsert(Candidate::interval(addr(2), 5.5, 10.0));
            assert_eq!(top.kth_lower(), Some(5.5));
        }

        /// Offering a lower exact for an address that already holds an exact is a bug.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "exact score")]
        fn test_upsert_lower_exact_debug_asserts() {
            let mut top = TopK::new(1);
            top.upsert(Candidate::exact(addr(1), 8.0));
            top.upsert(Candidate::exact(addr(1), 7.0));
        }

        /// The collector top-k is unset until `k` scores are in, then holds the k-th best score
        /// offered whatever the order and grouping of the offers, and never loosens; an
        /// offer counts as a tightening exactly when it moves the k-th.
        #[test]
        fn test_collector_topk_holds_the_kth_best_offer() {
            let mut bound = TopK::new(3);
            assert_eq!(bound.kth_lower(), None);
            let offer = bound.merge_exact([(addr(0), 5.0), (addr(1), 1.0)]);
            assert_eq!(
                offer,
                ExactMerge {
                    offers: 2,
                    tightenings: 0
                }
            );
            assert_eq!(bound.kth_lower(), None, "unset below k");
            assert_eq!(
                bound.merge_exact([(addr(2), 3.0)]).tightenings,
                1,
                "the k-th offer sets it"
            );
            assert_eq!(bound.kth_lower(), Some(1.0));
            assert_eq!(
                bound
                    .merge_exact([(addr(3), 1.0), (addr(4), 0.5)])
                    .tightenings,
                0,
                "ties and worse keep it"
            );
            assert_eq!(bound.kth_lower(), Some(1.0));
            assert_eq!(bound.merge_exact([(addr(5), 4.0)]).tightenings, 1);
            assert_eq!(bound.kth_lower(), Some(3.0));
            let skipped = bound.merge_exact([
                (addr(6), f32::NAN),
                (addr(7), f32::INFINITY),
                (addr(8), f32::NEG_INFINITY),
            ]);
            assert_eq!(
                skipped,
                ExactMerge::default(),
                "non-finite scores are no results"
            );
            assert_eq!(bound.kth_lower(), Some(3.0));
            bound.clear();
            assert_eq!(bound.kth_lower(), None);
            assert_eq!(
                TopK::new(0).merge_exact([(addr(0), 1.0)]),
                ExactMerge::default()
            );
            let mut seeded = TopK::new(4);
            seeded.merge_exact((0..4).map(|i| (addr(i as DocId), 2.5)));
            assert_eq!(seeded.kth_lower(), Some(2.5));

            for seed in 0..256 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let k = 1 + rng.usize(0..12);
                let scores: Vec<f32> = (0..rng.usize(0..40))
                    .map(|_| (rng.f32() * 20.0 - 10.0).round() / 2.0)
                    .collect();
                let mut bound = TopK::new(k);
                let mut offered: Vec<f32> = Vec::new();
                let mut previous = None;
                let mut rest = &scores[..];
                let mut next_doc = 0u32;
                while !rest.is_empty() {
                    let (offer, tail) = rest.split_at(1 + rng.usize(0..rest.len()));
                    rest = tail;
                    let mut tightenings = 0;
                    let batch: Vec<_> = offer
                        .iter()
                        .map(|&score| {
                            let a = addr(next_doc);
                            next_doc += 1;
                            (a, score)
                        })
                        .collect();
                    for &(_, score) in &batch {
                        let before = kth_best(&offered, k);
                        offered.push(score);
                        let after = kth_best(&offered, k);
                        if after.is_some_and(|after| before.is_none_or(|before| after > before)) {
                            tightenings += 1;
                        }
                    }
                    let report = bound.merge_exact(batch);
                    let case = format!("seed {seed} k={k}");
                    assert_eq!(report.offers, offer.len(), "{case}");
                    assert_eq!(report.tightenings, tightenings, "{case}");
                    assert_eq!(bound.kth_lower(), kth_best(&offered, k), "{case}");
                    if let Some(previous) = previous {
                        assert!(
                            bound.kth_lower().is_some_and(|now| now >= previous),
                            "{case}"
                        );
                    }
                    previous = bound.kth_lower();
                }
            }
        }

        /// The k-th largest of `scores`; `None` with fewer than `k`.
        fn kth_best(scores: &[f32], k: usize) -> Option<f32> {
            let mut sorted = scores.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            sorted.get(k - 1).copied()
        }
    }

    mod pruning_bound {
        use std::sync::{Arc, Barrier};

        use super::addr;
        use crate::collector::{AtomicSharedThreshold, SharedThreshold};
        use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
        use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
        use crate::vector::search::scan::bound::{Candidate, PruningBound, TopK};
        use crate::vector::search::scan::columns::BoundaryHarness;
        use crate::vector::write::RowAddress;
        use crate::{DocId, Score};

        fn threshold(value: f32) -> Option<Threshold> {
            Some(Threshold(LowerEndpoint(value)))
        }

        /// One cluster of `(row, estimate, sigma)` rows admitted into a fresh harness at `k`.
        fn admitted(k: usize, kappa: f32, rows: &[(usize, f32, f32)]) -> BoundaryHarness {
            let mut harness = BoundaryHarness::new(rows.len());
            harness.begin_cluster(k);
            for &(row, estimate, sigma) in rows {
                harness.push(row, row as DocId, estimate, sigma);
            }
            harness.finish_cluster(kappa);
            harness
        }

        /// The running and boundary thresholds: the k-th largest lower endpoint with each row's own
        /// sigma, the lowest estimate among the lower-endpoint top k, the same values whether rows
        /// arrive in clusters or all at once, and exact rows joining as zero-width intervals.
        #[test]
        fn test_kth_bound_thresholds() {
            // Ties on estimate break by the row's own sigma: lower endpoints 10, 7, -191.
            let mut tie = admitted(2, 2.0, &[(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 9.0, 100.0)]);
            assert_eq!(tie.own_lower(2), threshold(7.0));
            assert_eq!(tie.boundary_lower(2, 2.0), threshold(7.0));

            // Lower endpoints 10, 7, 4: the top two by lower endpoint are rows 0 and 1, so the APS
            // threshold is 9 though row 2 has the best estimate.
            let mut empty = BoundaryHarness::new(3);
            assert_eq!(empty.own_estimate(2), None);
            let mut aps = admitted(2, 2.0, &[(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 12.0, 4.0)]);
            assert_eq!(aps.own_lower(2), threshold(7.0));
            assert_eq!(aps.own_estimate(2), Some(9.0));
            assert_eq!(aps.own_estimate(4), None);

            // Cluster-local admission keeps the running k-th equal to the full partition's.
            let mut clustered = BoundaryHarness::new(24);
            for cluster in 0..6 {
                clustered.begin_cluster(4);
                for row in cluster * 4..cluster * 4 + 4 {
                    let estimate = 3.0 - row as f32 * 0.071 + (row as f32 * 0.37).sin() * 0.2;
                    clustered.push(
                        row,
                        (row % 12) as DocId,
                        estimate,
                        0.01 + (row % 5) as f32 * 0.003,
                    );
                }
                clustered.finish_cluster(2.0);
                assert_eq!(
                    clustered.own_lower(4),
                    clustered.boundary_lower(4, 2.0),
                    "cluster {cluster}"
                );
            }

            // The running top-k equals an independent sort by lower endpoint, then doc address.
            let clusters: &[&[(usize, f32, f32)]] = &[
                &[(0, 1.0, 0.01), (1, 2.0, 0.02)],
                &[
                    (2, 9.0, 0.03),
                    (3, 8.0, 0.04),
                    (4, 7.0, 0.05),
                    (5, 6.0, 0.06),
                    (6, 5.0, 0.07),
                ],
                &[
                    (7, 0.0, 0.08),
                    (8, 10.0, 0.09),
                    (9, 1.0, 0.10),
                    (10, 9.0, 8.0),
                    (11, 2.0, 0.11),
                ],
            ];
            for k in [1, 3] {
                let mut harness = BoundaryHarness::new(16);
                let mut seen: Vec<(RowAddress, f32)> = Vec::new();
                for (cluster, rows) in clusters.iter().enumerate() {
                    harness.begin_cluster(k);
                    for &(row, estimate, sigma) in *rows {
                        harness.push(row, row as DocId, estimate, sigma);
                        seen.push((addr(row as DocId), estimate - 2.0 * sigma));
                    }
                    harness.finish_cluster(2.0);
                    let mut expected = seen.clone();
                    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    expected.truncate(k);
                    let kth =
                        (expected.len() == k).then(|| Threshold(LowerEndpoint(expected[k - 1].1)));
                    assert_eq!(harness.own_lower(k), kth, "cluster={cluster} k={k}");
                }
            }

            // Exact rows are zero-width intervals: alone they set every threshold to the k-th
            // score.
            let mut exact = BoundaryHarness::new(8);
            for (i, score) in [3.0, 9.0, 5.0, 1.0, 9.0].into_iter().enumerate() {
                exact.push_exact(addr(i as DocId), score, 3);
            }
            assert_eq!(exact.own_lower(3), threshold(5.0));
            assert_eq!(exact.own_estimate(3), Some(5.0));
            assert_eq!(exact.boundary_lower(3, 2.0), threshold(5.0));
            let mut short = BoundaryHarness::new(8);
            for (i, score) in [3.0, 9.0, 5.0].into_iter().enumerate() {
                short.push_exact(addr(i as DocId), score, 4);
            }
            assert_eq!(short.own_lower(4), None);
            assert_eq!(short.own_estimate(4), None);
            assert_eq!(short.boundary_lower(4, 2.0), None);

            // With both kinds, the heap keeps the k best by lower endpoint. Lower endpoints 10, 6,
            // -2 and exact 11, 7 keep 11, 10 and 7, so both thresholds are 7. The
            // estimate 8 belonged to the interval exact 7 evicted.
            // Exact docs use fresh addresses so they compete with (not replace) the quantized rows.
            let quantized = [(0, 12.0, 1.0), (1, 8.0, 1.0), (2, 2.0, 2.0)];
            let mut mixed = admitted(3, 2.0, &quantized);
            assert_eq!(mixed.own_lower(3), threshold(-2.0));
            mixed.push_exact(addr(100), 7.0, 3);
            mixed.push_exact(addr(101), 11.0, 3);
            assert_eq!(mixed.own_lower(3), threshold(7.0));
            assert_eq!(mixed.own_estimate(3), Some(7.0));
            assert_eq!(mixed.boundary_lower(3, 2.0), threshold(7.0));
            mixed.band(3, 2.0);
            assert_eq!(
                mixed.candidates.global_rows(),
                [0, 1],
                "exact rows push row 2 out"
            );
            // Fewer quantized rows than k still reach a threshold through the exact rows.
            let mut wide = admitted(4, 2.0, &quantized);
            wide.push_exact(addr(100), 7.0, 4);
            wide.push_exact(addr(101), 11.0, 4);
            assert_eq!(wide.boundary_lower(4, 2.0), threshold(6.0));

            // A boundary keeps only rows whose upper endpoint reaches the threshold.
            let mut far = BoundaryHarness::new(3);
            for (row, estimate) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
                far.push(row, row as DocId, estimate, 0.1);
            }
            far.band(1, 2.0);
            assert_eq!(far.candidates.global_rows(), [0]);
        }

        /// Exact offer for the same address replaces the interval (no double-count) and only
        /// raises that row's lower endpoint.
        #[test]
        fn test_offer_replaces_same_address() {
            let mut bound = PruningBound::new(2, Arc::new(AtomicSharedThreshold::default()));
            bound.offer([
                Candidate::interval(addr(1), 6.0, 11.0),
                Candidate::interval(addr(2), 5.0, 10.0),
            ]);
            assert_eq!(bound.kth_lower(), Some(5.0));
            // Exact raises addr 1's lower; heap is Exact(8), Interval(5) → kth 5.
            bound.offer([Candidate::exact(addr(1), 8.0)]);
            assert_eq!(bound.kth_lower(), Some(5.0));
            assert_eq!(bound.exact_kth(), None, "still mixed");
            bound.offer([Candidate::exact(addr(2), 8.5)]);
            assert_eq!(bound.exact_kth(), Some(8.0));
        }

        /// When every interval encloses its row's exact score, with or without exactly scored rows,
        /// the running threshold never exceeds the true k-th score, a boundary keeps every true
        /// top-k row, and no exact row stays a candidate.
        #[test]
        fn test_boundary_keeps_enclosed_top_k() {
            // Minimal case: the runner-up's interval is narrow but encloses its exact score.
            let examples: [(f32, f32, f32); 3] = [
                (0.10, 0.015, 0.065),
                (0.09, 0.001, 0.090),
                (0.08, 0.001, 0.080),
            ];
            let rows: Vec<_> = examples
                .iter()
                .enumerate()
                .map(|(row, &(estimate, sigma, truth))| {
                    assert!((truth - estimate).abs() <= QUANTIZED_BOUNDARY_KAPPA * sigma);
                    (row, estimate, sigma)
                })
                .collect();
            let mut harness = admitted(2, QUANTIZED_BOUNDARY_KAPPA, &rows);
            harness.band(2, QUANTIZED_BOUNDARY_KAPPA);
            assert!(
                harness.candidates.doc_ids().contains(&2),
                "the true runner-up survives"
            );

            // Each cluster is quantized, or with exact rows on, exact at random.
            for with_exact in [false, true] {
                for seed in 0..512 {
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let n = 97;
                    let k = 1 + rng.usize(0..20);
                    let mut truth = Vec::new();
                    let mut exact_rows = Vec::new();
                    let mut harness = BoundaryHarness::new(n);
                    for cluster in (0..n).step_by(11) {
                        let exact = with_exact && rng.bool();
                        if !exact {
                            harness.begin_cluster(k);
                        }
                        for row in cluster..(cluster + 11).min(n) {
                            let score = rng.f32() * 20.0 - 10.0;
                            truth.push(score);
                            if exact {
                                exact_rows.push(row);
                                // Distinct addresses from quantized docs (row as DocId).
                                harness.push_exact(addr(1_000 + row as DocId), score, k);
                                continue;
                            }
                            let sigma = 0.001 + rng.f32().powi(3) * 10.0;
                            let estimate =
                                score + (rng.f32() * 2.0 - 1.0) * QUANTIZED_BOUNDARY_KAPPA * sigma;
                            harness.push(row, row as DocId, estimate, sigma);
                        }
                        if !exact {
                            harness.finish_cluster(QUANTIZED_BOUNDARY_KAPPA);
                        }
                        let case = format!("exact={with_exact} seed {seed}, cluster {cluster}");
                        let running = harness.own_lower(k);
                        assert_eq!(
                            running,
                            harness.boundary_lower(k, QUANTIZED_BOUNDARY_KAPPA),
                            "{case}"
                        );
                        if let Some(threshold) = running {
                            let mut sorted = truth.clone();
                            sorted.sort_by(|a, b| b.total_cmp(a));
                            assert!(threshold.0 .0 <= sorted[k - 1] + 1e-5, "{case}");
                        }
                    }
                    let mut order = (0..n).collect::<Vec<_>>();
                    order.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]));
                    harness.band(k, QUANTIZED_BOUNDARY_KAPPA);
                    let survivors = harness.candidates.global_rows();
                    for row in &order[..k] {
                        assert!(
                            exact_rows.contains(row) || survivors.contains(row),
                            "exact={with_exact} seed {seed}, true row {row}"
                        );
                    }
                    assert!(
                        survivors.iter().all(|row| !exact_rows.contains(row)),
                        "exact={with_exact} seed {seed}: an exact row stayed a candidate"
                    );
                }
            }
        }

        /// Collector exacts seed the segment TopK (partial k); the slot is CASed only once full.
        #[test]
        fn test_collector_seeds_segment_topk_for_partial_k() {
            let mut collector = TopK::new(3);
            collector.merge_exact([(addr(10), 4.5), (addr(11), 5.0)]);
            assert_eq!(collector.kth_lower(), None);

            let slot = Arc::new(AtomicSharedThreshold::default());
            let mut bound = PruningBound::new(3, slot.clone());
            bound.seed(&collector);
            assert_eq!(bound.kth_lower(), None);
            assert_eq!(slot.load(), None, "not full yet");

            bound.offer([Candidate::exact(addr(0), 3.0)]);
            assert_eq!(bound.kth_lower(), Some(3.0));
            assert_eq!(slot.load().map(|(s, _)| s), Some(3.0));
        }

        /// The pruning threshold never falls: not as clusters merge into the running top-k, exact
        /// scores arrive, or a boundary folds the layer into the slot.
        #[test]
        fn test_threshold_never_decreases() {
            let value = |rng: &mut fastrand::Rng| (rng.f32() * 20.0 - 10.0).round() / 2.0;
            for seed in 0..512 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let k = 1 + rng.usize(0..12);
                let mut harness = BoundaryHarness::new(64);
                let mut previous: Option<f32> = None;
                let mut row = 0;
                let mut exact_doc = 10_000u32;
                for step in 0..rng.usize(1..64) {
                    match rng.usize(0..3) {
                        0 => {
                            harness.begin_cluster(k);
                            for _ in 0..rng.usize(1..2 * k + 2) {
                                row += 1;
                                let sigma = rng.f32() * 3.0;
                                harness.push(row, row as DocId, value(&mut rng), sigma);
                            }
                            harness.finish_cluster(2.0);
                        }
                        1 => {
                            harness.push_exact(addr(exact_doc), value(&mut rng), k);
                            exact_doc += 1;
                        }
                        _ => harness.band(k, 2.0),
                    }
                    let now = harness.slot_threshold(k);
                    if let Some(previous) = previous {
                        assert!(
                            now.is_none_or(|now| now >= previous),
                            "seed {seed} k={k} step {step}: {now:?} after {previous}"
                        );
                    }
                    if now.is_some() {
                        previous = now;
                    }
                }
            }
        }

        /// `(estimate, sigma)` rows for blocks of a harness, with global rows ascending across
        /// blocks.
        fn blocks(rng: &mut fastrand::Rng) -> Vec<Vec<(usize, f32, f32)>> {
            let mut row = 0;
            (0..1 + rng.usize(0..6))
                .map(|_| {
                    (0..rng.usize(0..12))
                        .map(|_| {
                            row += 1 + rng.usize(0..3);
                            (row, rng.f32() * 20.0 - 10.0, rng.f32() * 3.0)
                        })
                        .collect()
                })
                .collect()
        }

        /// A harness holding `blocks`, each row's doc id derived from its global row.
        fn harness_of(blocks: &[Vec<(usize, f32, f32)>]) -> BoundaryHarness {
            let mut harness = BoundaryHarness::new(64);
            for rows in blocks {
                harness.open_block();
                for &(row, estimate, sigma) in rows {
                    harness.push(row, (row * 7) as DocId, estimate, sigma);
                }
            }
            harness
        }

        /// A recheck drops exactly the block's rows whose upper endpoint is below the threshold,
        /// keeps the rest in row order with their doc ids and column values, and leaves the
        /// other blocks alone.
        #[test]
        fn test_recheck_drops_exactly_the_rows_below_the_threshold() {
            for seed in 0..512 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let blocks = blocks(&mut rng);
                let mut harness = harness_of(&blocks);
                let mut kept = blocks.clone();
                for _ in 0..1 + rng.usize(0..4) {
                    let b = rng.usize(0..blocks.len());
                    let threshold = rng.f32() * 24.0 - 12.0;
                    let before = kept[b].len();
                    kept[b].retain(|&(_, estimate, sigma)| {
                        Threshold(LowerEndpoint(threshold))
                            .admits(Estimate(estimate).upper(sigma, QUANTIZED_BOUNDARY_KAPPA))
                    });
                    let dropped = harness.recheck(b, threshold);
                    let case = format!("seed {seed} block {b} threshold {threshold}");
                    assert_eq!(dropped, before - kept[b].len(), "{case}");
                    let expected = harness_of(&kept);
                    assert_eq!(harness.block_lens(), expected.block_lens(), "{case}");
                    assert_eq!(
                        harness.candidates.column_trace(true),
                        expected.candidates.column_trace(true),
                        "{case}"
                    );
                    assert_eq!(
                        harness.candidates.doc_ids(),
                        expected.candidates.doc_ids(),
                        "{case}"
                    );
                    assert_eq!(
                        harness.candidates.len(),
                        kept.iter().map(Vec::len).sum::<usize>(),
                        "{case}"
                    );
                }
            }
        }

        /// A boundary after rechecks, over blocks with dead entries between their ranges, keeps the
        /// rows and threshold of a boundary over the same blocks with the dropped rows never added.
        #[test]
        fn test_band_after_recheck_matches_rows_removed_up_front() {
            for seed in 0..512 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let k = 1 + rng.usize(0..8);
                let blocks = blocks(&mut rng);
                let exact: Vec<f32> = (0..rng.usize(0..k + 2))
                    .map(|_| rng.f32() * 20.0 - 10.0)
                    .collect();
                let mut rechecked = harness_of(&blocks);
                let mut kept = blocks.clone();
                for _ in 0..rng.usize(0..6) {
                    let b = rng.usize(0..blocks.len());
                    let threshold = rng.f32() * 24.0 - 12.0;
                    rechecked.recheck(b, threshold);
                    kept[b].retain(|&(_, estimate, sigma)| {
                        Threshold(LowerEndpoint(threshold))
                            .admits(Estimate(estimate).upper(sigma, QUANTIZED_BOUNDARY_KAPPA))
                    });
                }
                let mut removed = harness_of(&kept);
                for harness in [&mut rechecked, &mut removed] {
                    for (i, &score) in exact.iter().enumerate() {
                        harness.push_exact(addr(50_000 + i as DocId), score, k);
                    }
                    harness.band(k, QUANTIZED_BOUNDARY_KAPPA);
                }
                let case = format!("seed {seed} k={k}");
                assert_eq!(
                    rechecked.slot_threshold(k),
                    removed.slot_threshold(k),
                    "{case}"
                );
                assert_eq!(rechecked.block_lens(), removed.block_lens(), "{case}");
                assert_eq!(
                    rechecked.candidates.column_trace(true),
                    removed.candidates.column_trace(true),
                    "{case}"
                );
                assert_eq!(
                    rechecked.candidates.doc_ids(),
                    removed.candidates.doc_ids(),
                    "{case}"
                );
            }
        }

        fn slot_score(slot: &AtomicSharedThreshold) -> Option<Score> {
            slot.load().map(|(score, _)| score)
        }

        /// A bound at `k` sharing `shared_topk_threshold`, holding `scores` as this segment's exact
        /// scores.
        fn exact_bound(
            shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
            k: usize,
            scores: &[f32],
        ) -> PruningBound {
            let mut bound = PruningBound::new(k, shared_topk_threshold);
            for (i, &score) in scores.iter().enumerate() {
                bound.offer([Candidate::exact(addr(i as DocId), score)]);
            }
            bound
        }

        /// The slot keeps the highest finite threshold offered to it. A lower offer, a repeat, and
        /// a non-finite score leave it.
        #[test]
        fn test_slot_keeps_the_highest_finite_threshold() {
            let slot = Arc::new(AtomicSharedThreshold::default());
            let shared_topk_threshold: Arc<dyn SharedThreshold<Score>> = slot.clone();
            let _ = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[]);
            assert_eq!(
                slot_score(&slot),
                None,
                "an unset threshold is never offered"
            );

            let mut first = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[5.0, 3.0]);
            assert_eq!(slot_score(&slot), Some(3.0));
            let held = first.shared_threshold();
            first.offer([]);
            assert_eq!(first.shared_threshold(), held);

            let mut second = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[5.0, 2.0]);
            assert_eq!(slot_score(&slot), Some(3.0), "a lower offer stays out");
            second.offer([Candidate::exact(addr(2), 4.0)]);
            assert_eq!(slot_score(&slot), Some(4.0));

            for (i, score) in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
                .into_iter()
                .enumerate()
            {
                let mut bound = exact_bound(Arc::clone(&shared_topk_threshold), 1, &[]);
                bound.offer([Candidate::exact(addr(100 + i as DocId), score)]);
                assert_eq!(slot_score(&slot), Some(4.0), "{score} is never written");
            }

            let mut above = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[9.0, 8.0]);
            assert_eq!(above.shared_threshold(), Some(8.0));
            assert_eq!(slot_score(&slot), Some(8.0));
        }

        /// Threads publishing interleaved, rising thresholds into one slot leave the slot at the
        /// highest threshold any of them reached.
        #[test]
        fn test_concurrent_publishes_leave_the_slot_at_the_max() {
            for seed in 0..32 {
                let slot = Arc::new(AtomicSharedThreshold::default());
                let shared_topk_threshold: Arc<dyn SharedThreshold<Score>> = slot.clone();
                let threads = 2 + seed as usize % 7;
                let barrier = Barrier::new(threads);
                let finals: Vec<Option<f32>> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..threads)
                        .map(|thread| {
                            let (shared_topk_threshold, barrier) =
                                (Arc::clone(&shared_topk_threshold), &barrier);
                            scope.spawn(move || {
                                let mut rng = fastrand::Rng::with_seed(seed * 64 + thread as u64);
                                let k = 1 + rng.usize(0..6);
                                let mut bound = PruningBound::new(k, shared_topk_threshold);
                                barrier.wait();
                                let mut doc = (thread as u32) * 100_000;
                                for _ in 0..rng.usize(1..400) {
                                    bound.offer([Candidate::exact(
                                        addr(doc),
                                        rng.f32() * 100.0 - 50.0,
                                    )]);
                                    doc += 1;
                                    let _ = bound.shared_threshold();
                                }
                                bound.kth_lower()
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let max = finals.iter().flatten().copied().reduce(f32::max);
                assert_eq!(slot.load().map(|(score, _)| score), max, "seed {seed}");
            }
        }

        fn survivors(estimates: &[f32], sigmas: &[f32], k: usize, kappa: f32) -> Vec<usize> {
            let mut harness = BoundaryHarness::new(estimates.len());
            for (row, (&estimate, &sigma)) in estimates.iter().zip(sigmas).enumerate() {
                harness.push(row, row as DocId, estimate, sigma);
            }
            harness.band(k, kappa);
            harness.candidates.global_rows()
        }

        /// Widening kappa, or any one row's sigma, never removes a survivor.
        #[test]
        fn test_widening_never_drops_a_survivor() {
            for seed in 0..256 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let estimates: Vec<_> = (0..41).map(|_| rng.f32() * 20.0 - 10.0).collect();
                let sigmas: Vec<_> = (0..41).map(|_| rng.f32() * 3.0).collect();
                let k = 1 + rng.usize(0..10);
                let original = survivors(&estimates, &sigmas, k, 2.0);
                let wider = survivors(&estimates, &sigmas, k, 2.5);
                assert!(
                    original.iter().all(|row| wider.contains(row)),
                    "kappa seed {seed}"
                );
                for changed in 0..sigmas.len() {
                    let mut enlarged = sigmas.clone();
                    enlarged[changed] += 0.01 + rng.f32() * 10.0;
                    let wider = survivors(&estimates, &enlarged, k, 2.0);
                    assert!(
                        original.iter().all(|row| wider.contains(row)),
                        "sigma seed {seed}, row {changed}"
                    );
                }
            }
        }

        /// Two boundaries select exactly the survivor sets of the kernel crate's reference harness,
        /// and keep the true top k.
        #[test]
        fn test_survivor_sets_match_kernel_harness() {
            const CANDIDATES: usize = 40;
            const TOP_N: usize = 10;
            const KAPPA: f32 = QUANTIZED_BOUNDARY_KAPPA;

            let layer0_scores: Vec<f32> = (0..CANDIDATES)
                .map(|row| 2.0 - row as f32 * 0.09 + (row as f32 * 0.7).sin() * 0.08)
                .collect();
            let layer0_sigmas: Vec<f32> = (0..CANDIDATES)
                .map(|row| 0.03 + (row % 5) as f32 * 0.01)
                .collect();
            let refinements: Vec<f32> = (0..CANDIDATES)
                .map(|row| (row as f32 * 0.41).cos() * 0.06)
                .collect();
            let layer1_sigmas: Vec<f32> = (0..CANDIDATES)
                .map(|row| 0.012 + (row % 3) as f32 * 0.004)
                .collect();

            let lower: Vec<f32> = layer0_scores
                .iter()
                .zip(&layer0_sigmas)
                .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
                .collect();
            let (_, first_kth) = quant::cascade::kth(&lower, TOP_N);
            let harness_first =
                quant::cascade::band_filter(&layer0_scores, &layer0_sigmas, KAPPA, first_kth);

            let mut scan = BoundaryHarness::new(CANDIDATES);
            for row in 0..CANDIDATES {
                scan.push(row, row as DocId, layer0_scores[row], layer0_sigmas[row]);
            }
            scan.band(TOP_N, KAPPA);
            let scan_first: Vec<u32> = scan
                .candidates
                .global_rows()
                .iter()
                .map(|&row| row as u32)
                .collect();
            assert!(
                harness_first.len() < CANDIDATES,
                "the first boundary filters"
            );
            assert_eq!(scan_first, harness_first, "layer-0 survivor set");

            let second_scores: Vec<f32> = harness_first
                .iter()
                .map(|&row| layer0_scores[row as usize] + refinements[row as usize])
                .collect();
            let second_sigmas: Vec<f32> = harness_first
                .iter()
                .map(|&row| layer1_sigmas[row as usize])
                .collect();
            let lower: Vec<f32> = second_scores
                .iter()
                .zip(&second_sigmas)
                .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
                .collect();
            let (_, second_kth) = quant::cascade::kth(&lower, TOP_N);
            let harness_second: Vec<u32> =
                quant::cascade::band_filter(&second_scores, &second_sigmas, KAPPA, second_kth)
                    .iter()
                    .map(|&local| harness_first[local as usize])
                    .collect();

            for (index, row) in scan.candidates.global_rows().into_iter().enumerate() {
                scan.candidates.columns.estimates[index] += refinements[row];
                scan.candidates.columns.sigmas[index] = layer1_sigmas[row];
            }
            scan.band(TOP_N, KAPPA);
            let scan_second: Vec<u32> = scan
                .candidates
                .global_rows()
                .iter()
                .map(|&row| row as u32)
                .collect();
            assert!(
                harness_second.len() < harness_first.len(),
                "the second boundary filters"
            );
            assert_eq!(scan_second, harness_second, "layer-1 survivor set");

            let mut exact_order: Vec<usize> = (0..CANDIDATES).collect();
            exact_order.sort_unstable_by(|&left, &right| {
                (layer0_scores[right] + refinements[right])
                    .total_cmp(&(layer0_scores[left] + refinements[left]))
                    .then_with(|| left.cmp(&right))
            });
            let recalled = exact_order[..TOP_N]
                .iter()
                .filter(|&&row| scan_second.contains(&(row as u32)))
                .count();
            assert_eq!(recalled, TOP_N, "candidate recall matches the harness");
        }
    }
}

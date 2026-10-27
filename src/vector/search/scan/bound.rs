//! The pruning bounds of one query's scans.
//!
//! The collector holds two things every segment scan shares. `collector_topk` is the top-k exact
//! scores from segments this collector has already finished. `shared_topk_threshold` is the
//! threshold the query's participants publish; when the caller provides none, the collector builds
//! a process-local [`AtomicSharedThreshold`] shared by every segment of that search.
//!
//! A [`PruningBound`] is one segment scan. Its [`TopK`] holds only that segment's candidates,
//! ordered by lower endpoint, so the router and APS read [`TopK::kth_lower`] and
//! [`TopK::kth_estimate`]. The collector top-k and `shared_topk_threshold` seed `tau` and then only
//! raise it. An exact score is a zero-width interval. Admission drops a row only when its upper
//! endpoint is strictly below `tau`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

#[cfg(test)]
use crate::collector::AtomicSharedThreshold;
use crate::collector::SharedThreshold;
use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
use crate::vector::search::scan::columns::{Columns, QuantizedCandidates};
use crate::{Score, SegmentOrdinal};

// --- Scores ---

/// One row's score bound. An exact score is a zero-width interval: every endpoint is the score.
///
/// The upper endpoint of a quantized row is consumed at admission. Later layers recompute it from
/// the refined columns, so an interval stored here keeps the lower endpoint and the point estimate.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bound {
    Exact(f32),
    Interval { lo: f32, est: f32 },
}

impl Bound {
    fn lower(self) -> f32 {
        match self {
            Bound::Exact(score) | Bound::Interval { lo: score, .. } => score,
        }
    }

    fn estimate(self) -> f32 {
        match self {
            Bound::Exact(score) | Bound::Interval { est: score, .. } => score,
        }
    }
}

/// A row and the bound admission computed for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub(crate) row: usize,
    pub(crate) bound: Bound,
}

/// Best-first order: higher lower endpoint, then smaller row.
#[derive(Clone, Copy, Debug)]
struct ByLower(Candidate);

impl PartialEq for ByLower {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for ByLower {}
impl PartialOrd for ByLower {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByLower {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .bound
            .lower()
            .total_cmp(&other.0.bound.lower())
            .then_with(|| other.0.row.cmp(&self.0.row))
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

/// The `k` best candidates by lower endpoint.
///
/// Exact scores and quantized intervals share one heap. Once every entry is exact,
/// [`TopK::kth_lower`] is the exact-only k-th. The router reads [`TopK::kth_lower`]; APS reads
/// [`TopK::kth_estimate`].
#[derive(Clone, Debug)]
pub(crate) struct TopK {
    k: usize,
    /// The worst lower endpoint sits on top.
    heap: BinaryHeap<Reverse<ByLower>>,
}

impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
        }
    }

    #[cfg(test)]
    fn k(&self) -> usize {
        self.k
    }

    /// The worst lower endpoint once `k` candidates are in.
    pub(crate) fn kth_lower(&self) -> Option<f32> {
        if self.k == 0 || self.heap.len() != self.k {
            return None;
        }
        let lower = self.heap.peek().unwrap().0 .0.bound.lower();
        lower.is_finite().then_some(lower)
    }

    /// The k-th of the estimates in the heap, once it is full. It carries no confidence margin.
    pub(crate) fn kth_estimate(&self) -> Option<f32> {
        if self.k == 0 || self.heap.len() != self.k {
            return None;
        }
        self.heap
            .iter()
            .map(|Reverse(ByLower(candidate))| candidate.bound.estimate())
            .min_by(|a, b| a.total_cmp(b))
    }

    /// The k-th exact score once the heap holds `k` exact scores and nothing else.
    pub(super) fn exact_kth(&self) -> Option<f32> {
        let exact = self.kth_lower()?;
        self.heap
            .iter()
            .all(|Reverse(ByLower(candidate))| matches!(candidate.bound, Bound::Exact(_)))
            .then_some(exact)
    }

    fn push(&mut self, candidate: Candidate) {
        if self.k == 0 {
            return;
        }
        let item = Reverse(ByLower(candidate));
        if self.heap.len() < self.k {
            self.heap.push(item);
            return;
        }
        if item.0 > self.heap.peek().unwrap().0 {
            self.heap.pop();
            self.heap.push(item);
        }
    }

    /// Records an exact score. A non-finite score is kept, matching a segment scan's own scores.
    pub(crate) fn push_exact(&mut self, score: f32) {
        self.push(Candidate {
            row: 0,
            bound: Bound::Exact(score),
        });
    }

    /// Takes exact scores from a finished segment. Non-finite scores are skipped.
    pub(crate) fn merge_exact(&mut self, scores: impl IntoIterator<Item = f32>) -> ExactMerge {
        let mut merge = ExactMerge::default();
        if self.k == 0 {
            return merge;
        }
        for score in scores {
            if !score.is_finite() {
                continue;
            }
            merge.offers += 1;
            let before = self.kth_lower();
            self.push_exact(score);
            if self
                .kth_lower()
                .is_some_and(|after| before.is_none_or(|before| after > before))
            {
                merge.tightenings += 1;
            }
        }
        merge
    }

    /// Drops the intervals and merges `fresh` ones. Exact scores stay: they do not describe a
    /// layer.
    fn rescore(&mut self, fresh: impl IntoIterator<Item = Candidate>) {
        self.retain_exacts();
        for candidate in fresh {
            self.push(candidate);
        }
    }

    /// Drops quantized intervals. After a boundary, `tau` holds their k-th, and the next exact
    /// scores must not share the heap with them.
    fn retain_exacts(&mut self) {
        let exacts: Vec<Candidate> = self
            .heap
            .drain()
            .filter_map(|Reverse(ByLower(candidate))| {
                matches!(candidate.bound, Bound::Exact(_)).then_some(candidate)
            })
            .collect();
        for candidate in exacts {
            self.heap.push(Reverse(ByLower(candidate)));
        }
    }

    /// Drops every candidate. The collector calls this between searches.
    pub(crate) fn clear(&mut self) {
        self.heap.clear();
    }
}

// --- Pruning bound ---

/// One segment scan's pruning bound.
///
/// `top` is this segment's candidates. `tau` is the pruning threshold: it starts at the higher of
/// the collector's exact k-th and `shared_topk_threshold`, and afterwards only rises, including to
/// `top`'s k-th lower endpoint. The segment top is never seeded from the collector top-k.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct PruningBound {
    /// Only rises. Non-finite until the first finite raise.
    tau: f32,
    top: TopK,
    shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
    /// Highest threshold offered to `shared_topk_threshold`.
    last_published: f32,
    /// Offers that stored a new value in `shared_topk_threshold`.
    pub(super) publishes: usize,
    /// Loads of `shared_topk_threshold` whose score was above `tau`.
    pub(super) raises: usize,
}

impl PruningBound {
    /// A bound for `k` rows. `collector_kth` is the k-th exact score already returned by finished
    /// segments of this collector, when they have filled `k`. `shared_topk_threshold` is the
    /// participants' threshold.
    pub(super) fn new(
        k: usize,
        collector_kth: Option<f32>,
        shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
    ) -> Self {
        let mut bound = Self {
            tau: f32::NEG_INFINITY,
            top: TopK::new(k),
            shared_topk_threshold,
            last_published: f32::NEG_INFINITY,
            publishes: 0,
            raises: 0,
        };
        if let Some(kth) = collector_kth {
            bound.raise_tau(kth);
        }
        bound
    }

    /// No collector score, and a fresh process-local threshold.
    #[cfg(test)]
    pub(super) fn standalone(k: usize) -> Self {
        Self::new(k, None, Arc::new(AtomicSharedThreshold::default()))
    }

    /// The segment's own k-th lower endpoint. The router reads this.
    pub(crate) fn own_lower(&self) -> Option<f32> {
        self.top.kth_lower()
    }

    /// The segment's own APS estimate.
    pub(crate) fn own_estimate(&self) -> Option<f32> {
        self.top.kth_estimate()
    }

    /// The k-th exact score, once this segment's heap is full of exact scores.
    #[inline]
    pub(super) fn exact_kth(&self) -> Option<f32> {
        self.top.exact_kth()
    }

    /// Records an exactly scored row of this segment.
    pub(super) fn push_exact(&mut self, score: f32) {
        self.top.push_exact(score);
        self.note_kth();
    }

    /// Admits one cluster. A row whose upper endpoint is below `threshold` never enters `top`.
    ///
    /// `threshold` already includes `shared_topk_threshold`, so another segment or worker narrows
    /// the rows this segment offers the router and APS. That narrowing is recall-safe: an upper
    /// endpoint below a lower bound on the k-th true score cannot be one of the top k. Every
    /// admitted row joins `top`. Publishes afterwards.
    pub(super) fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        scratch: &Columns,
        kappa: f32,
        threshold: Threshold,
        keep: &mut Vec<u32>,
    ) {
        debug_assert_eq!(scratch.estimates.len(), rows.len());
        let start = rows.rows.start;
        keep.clear();
        for index in 0..rows.len() {
            let estimate = Estimate(scratch.estimates[index]);
            let sigma = scratch.sigmas[index];
            let (lower, upper) = (estimate.lower(sigma, kappa), estimate.upper(sigma, kappa));
            if !threshold.admits(upper) {
                continue;
            }
            let offset = rows
                .offsets
                .map_or(index, |offsets| offsets[index] as usize);
            keep.push(index as u32);
            self.top.push(Candidate {
                row: start + offset,
                bound: Bound::Interval {
                    lo: lower.0,
                    est: estimate.0,
                },
            });
        }
        self.note_kth();
        self.publish();
    }

    /// Merges one cluster's candidates into the segment top-k and publishes.
    pub(super) fn merge_cluster(&mut self, entries: impl IntoIterator<Item = Candidate>) {
        for candidate in entries {
            self.top.push(candidate);
        }
        self.note_kth();
        self.publish();
    }

    /// Raises `tau` to the segment's k-th lower endpoint when that k-th is higher.
    fn note_kth(&mut self) {
        if let Some(kth) = self.top.kth_lower() {
            self.raise_tau(kth);
        }
    }

    /// `tau` when it is finite.
    fn current(&self) -> Option<f32> {
        self.tau.is_finite().then_some(self.tau)
    }

    /// The threshold admission and recheck prune at, after reading `shared_topk_threshold`.
    pub(super) fn pruning_threshold(&mut self) -> Option<Threshold> {
        self.sync();
        self.as_threshold()
    }

    /// The threshold a boundary prunes at, after reading `shared_topk_threshold`.
    pub(super) fn boundary(&mut self, candidates: &QuantizedCandidates) -> Option<Threshold> {
        self.sync();
        self.commit_layer(candidates.live_candidates());
        self.publish();
        self.as_threshold()
    }

    /// Folds this layer's refined intervals into `tau`, then drops them from the heap.
    ///
    /// The k-th lower endpoint of those intervals and the exact scores already stored is a lower
    /// bound on the k-th true score, so `tau` keeps it. The intervals themselves leave: a later
    /// exact score sharing the heap with a leftover lower endpoint would raise `tau` before k
    /// exact scores exist, and the next recheck would drop a true hit.
    fn commit_layer(&mut self, fresh: impl IntoIterator<Item = Candidate>) {
        self.top.rescore(fresh);
        self.note_kth();
        self.top.retain_exacts();
    }

    fn as_threshold(&self) -> Option<Threshold> {
        self.current().map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// Raises `tau` to `value`. A lower or non-finite value leaves it.
    fn raise_tau(&mut self, value: f32) {
        if value.is_finite() && value > self.tau {
            self.tau = value;
        }
    }

    /// Raises `tau` to `shared_topk_threshold`'s score when that score is higher. The ordinal is
    /// ignored.
    fn sync(&mut self) {
        let Some((score, _)) = self.shared_topk_threshold.load() else {
            return;
        };
        if score.is_finite() && self.current().is_none_or(|tau| score > tau) {
            self.raises += 1;
        }
        self.raise_tau(score);
    }

    /// Offers `tau` to `shared_topk_threshold` when it is finite and above the last offer.
    pub(super) fn publish(&mut self) {
        let Some(kth) = self.current() else {
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

    #[cfg(test)]
    pub(super) fn k(&self) -> usize {
        self.top.k()
    }

    #[cfg(test)]
    pub(super) fn threshold(&self) -> Option<f32> {
        self.current()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::{Bound, Candidate, ExactMerge, PruningBound, TopK};
    use crate::collector::{AtomicSharedThreshold, SharedThreshold};
    use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
    use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
    use crate::vector::search::scan::columns::BoundaryHarness;
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
    fn kth_bound_thresholds() {
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

        // The running top-k equals an independent sort by lower endpoint, then row.
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
            let mut seen: Vec<(usize, f32)> = Vec::new();
            for (cluster, rows) in clusters.iter().enumerate() {
                harness.begin_cluster(k);
                for &(row, estimate, sigma) in *rows {
                    harness.push(row, row as DocId, estimate, sigma);
                    seen.push((row, estimate - 2.0 * sigma));
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

        // Exact rows are zero-width intervals: alone they set every threshold to the k-th score.
        let mut exact = BoundaryHarness::new(8);
        for score in [3.0, 9.0, 5.0, 1.0, 9.0] {
            exact.push_exact(score, 3);
        }
        assert_eq!(exact.own_lower(3), threshold(5.0));
        assert_eq!(exact.own_estimate(3), Some(5.0));
        assert_eq!(exact.boundary_lower(3, 2.0), threshold(5.0));
        let mut short = BoundaryHarness::new(8);
        for score in [3.0, 9.0, 5.0] {
            short.push_exact(score, 4);
        }
        assert_eq!(short.own_lower(4), None);
        assert_eq!(short.own_estimate(4), None);
        assert_eq!(short.boundary_lower(4, 2.0), None);

        // With both kinds, the heap keeps the k best by lower endpoint. Lower endpoints 10, 6, -2
        // and exact 11, 7 keep 11, 10 and 7, so both thresholds are 7. The estimate 8 belonged to
        // the interval exact 7 evicted.
        let quantized = [(0, 12.0, 1.0), (1, 8.0, 1.0), (2, 2.0, 2.0)];
        let mut mixed = admitted(3, 2.0, &quantized);
        assert_eq!(mixed.own_lower(3), threshold(-2.0));
        mixed.push_exact(7.0, 3);
        mixed.push_exact(11.0, 3);
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
        wide.push_exact(7.0, 4);
        wide.push_exact(11.0, 4);
        assert_eq!(wide.boundary_lower(4, 2.0), threshold(6.0));

        // A boundary keeps only rows whose upper endpoint reaches the threshold.
        let mut far = BoundaryHarness::new(3);
        for (row, estimate) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
            far.push(row, row as DocId, estimate, 0.1);
        }
        far.band(1, 2.0);
        assert_eq!(far.candidates.global_rows(), [0]);
    }

    /// When every interval encloses its row's exact score, with or without exactly scored rows,
    /// the running threshold never exceeds the true k-th score, a boundary keeps every true top-k
    /// row, and no exact row stays a candidate.
    #[test]
    fn boundary_keeps_enclosed_top_k() {
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
                            harness.push_exact(score, k);
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

    /// The collector top-k is unset until `k` scores are in, then holds the k-th best score offered
    /// whatever the order and grouping of the offers, and never loosens; an offer counts as a
    /// tightening exactly when it moves the k-th.
    #[test]
    fn collector_topk_holds_the_kth_best_offer() {
        let mut bound = TopK::new(3);
        assert_eq!(bound.kth_lower(), None);
        let offer = bound.merge_exact([5.0, 1.0]);
        assert_eq!(
            offer,
            ExactMerge {
                offers: 2,
                tightenings: 0
            }
        );
        assert_eq!(bound.kth_lower(), None, "unset below k");
        assert_eq!(
            bound.merge_exact([3.0]).tightenings,
            1,
            "the k-th offer sets it"
        );
        assert_eq!(bound.kth_lower(), Some(1.0));
        assert_eq!(
            bound.merge_exact([1.0, 0.5]).tightenings,
            0,
            "ties and worse keep it"
        );
        assert_eq!(bound.kth_lower(), Some(1.0));
        assert_eq!(bound.merge_exact([4.0]).tightenings, 1);
        assert_eq!(bound.kth_lower(), Some(3.0));
        let skipped = bound.merge_exact([f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!(
            skipped,
            ExactMerge::default(),
            "non-finite scores are no results"
        );
        assert_eq!(bound.kth_lower(), Some(3.0));
        bound.clear();
        assert_eq!(bound.kth_lower(), None);
        assert_eq!(TopK::new(0).merge_exact([1.0]), ExactMerge::default());
        let mut seeded = TopK::new(4);
        seeded.merge_exact(std::iter::repeat_n(2.5, 4));
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
            while !rest.is_empty() {
                let (offer, tail) = rest.split_at(1 + rng.usize(0..rest.len()));
                rest = tail;
                let mut tightenings = 0;
                for &score in offer {
                    let before = kth_best(&offered, k);
                    offered.push(score);
                    let after = kth_best(&offered, k);
                    if after.is_some_and(|after| before.is_none_or(|before| after > before)) {
                        tightenings += 1;
                    }
                }
                let report = bound.merge_exact(offer.iter().copied());
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

    /// A collector k-th seeds `tau` and stays out of the segment top, so the router still reads an
    /// empty segment. Segment candidates then raise `tau` and never lower it.
    #[test]
    fn collector_kth_seeds_tau_without_entering_the_segment_top() {
        let bound = PruningBound::new(3, Some(4.5), Arc::new(AtomicSharedThreshold::default()));
        assert_eq!(bound.threshold(), Some(4.5));
        assert_eq!(bound.own_lower(), None);
        assert_eq!(bound.own_estimate(), None);

        let mut bound = PruningBound::new(2, Some(1.0), Arc::new(AtomicSharedThreshold::default()));
        bound.merge_cluster([
            Candidate {
                row: 0,
                bound: Bound::Interval { lo: 5.0, est: 6.0 },
            },
            Candidate {
                row: 1,
                bound: Bound::Interval { lo: 3.0, est: 4.0 },
            },
            Candidate {
                row: 2,
                bound: Bound::Interval { lo: 0.0, est: 1.0 },
            },
        ]);
        assert_eq!(bound.own_lower(), Some(3.0));
        assert_eq!(bound.own_estimate(), Some(4.0));
        assert_eq!(bound.threshold(), Some(3.0));
        let raised = bound.threshold();
        bound.push_exact(0.5);
        assert_eq!(bound.threshold(), raised);
    }

    /// The pruning threshold never falls: not as clusters merge into the running top-k, exact
    /// scores arrive, or a boundary folds the layer into `tau`.
    #[test]
    fn threshold_never_decreases() {
        let value = |rng: &mut fastrand::Rng| (rng.f32() * 20.0 - 10.0).round() / 2.0;
        for seed in 0..512 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let k = 1 + rng.usize(0..12);
            let mut harness = BoundaryHarness::new(64);
            let mut previous: Option<f32> = None;
            let mut row = 0;
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
                    1 => harness.push_exact(value(&mut rng), k),
                    _ => harness.band(k, 2.0),
                }
                let now = harness.threshold(k);
                if let Some(previous) = previous {
                    assert!(
                        now.is_some_and(|now| now >= previous),
                        "seed {seed} k={k} step {step}: {now:?} after {previous}"
                    );
                }
                previous = now;
            }
        }
    }

    /// `(estimate, sigma)` rows for blocks of a harness, with global rows ascending across blocks.
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

    /// A recheck drops exactly the block's rows whose upper endpoint is below the threshold, keeps
    /// the rest in row order with their doc ids and column values, and leaves the other blocks
    /// alone.
    #[test]
    fn recheck_drops_exactly_the_rows_below_the_threshold() {
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
    fn band_after_recheck_matches_rows_removed_up_front() {
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
                for &score in &exact {
                    harness.push_exact(score, k);
                }
                harness.band(k, QUANTIZED_BOUNDARY_KAPPA);
            }
            let case = format!("seed {seed} k={k}");
            assert_eq!(rechecked.threshold(k), removed.threshold(k), "{case}");
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
        let mut bound = PruningBound::new(k, None, shared_topk_threshold);
        for &score in scores {
            bound.push_exact(score);
        }
        bound
    }

    /// The slot keeps the highest finite threshold offered to it. A lower offer, a repeat, and a
    /// non-finite score leave it. A sync raises `tau` to the slot and never lowers it.
    #[test]
    fn slot_keeps_the_highest_finite_threshold() {
        let slot = Arc::new(AtomicSharedThreshold::default());
        let shared_topk_threshold: Arc<dyn SharedThreshold<Score>> = slot.clone();
        exact_bound(Arc::clone(&shared_topk_threshold), 2, &[]).publish();
        assert_eq!(
            slot_score(&slot),
            None,
            "an unset threshold is never offered"
        );

        let mut first = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[5.0, 3.0]);
        first.publish();
        assert_eq!(slot_score(&slot), Some(3.0));
        let held = first.threshold();
        first.publish();
        assert_eq!(first.threshold(), held);

        let mut second = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[5.0, 2.0]);
        second.publish();
        assert_eq!(slot_score(&slot), Some(3.0), "a lower offer stays out");
        second.push_exact(4.0);
        second.publish();
        assert_eq!(slot_score(&slot), Some(4.0));

        for score in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut bound = exact_bound(Arc::clone(&shared_topk_threshold), 1, &[score]);
            bound.publish();
            assert_eq!(slot_score(&slot), Some(4.0), "{score} is never written");
        }

        let mut synced = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[5.0, 3.0]);
        synced.sync();
        assert_eq!(synced.threshold(), Some(4.0));
        synced.sync();
        assert_eq!(
            synced.threshold(),
            Some(4.0),
            "an equal score changes nothing"
        );

        let mut above = exact_bound(Arc::clone(&shared_topk_threshold), 2, &[9.0, 8.0]);
        above.sync();
        assert_eq!(above.threshold(), Some(8.0));
        above.publish();
        assert_eq!(slot_score(&slot), Some(8.0));
    }

    /// Threads publishing interleaved, rising thresholds into one slot, some also syncing from
    /// it, leave the slot at the highest threshold any of them reached.
    #[test]
    fn concurrent_publishes_leave_the_slot_at_the_max() {
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
                            let mut bound = PruningBound::new(k, None, shared_topk_threshold);
                            barrier.wait();
                            for _ in 0..rng.usize(1..400) {
                                bound.push_exact(rng.f32() * 100.0 - 50.0);
                                if rng.usize(0..4) == 0 {
                                    bound.sync();
                                }
                                bound.publish();
                            }
                            bound.threshold()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let max = finals.iter().flatten().copied().reduce(f32::max);
            assert_eq!(slot.load().map(|(score, _)| score), max, "seed {seed}");
        }
    }

    /// The k-th largest of `scores`; `None` with fewer than `k`.
    fn kth_best(scores: &[f32], k: usize) -> Option<f32> {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        sorted.get(k - 1).copied()
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
    fn widening_never_drops_a_survivor() {
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
    fn survivor_sets_match_kernel_harness() {
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

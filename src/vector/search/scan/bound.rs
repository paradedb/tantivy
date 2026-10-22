//! The pruning bounds of one query's scans.
//!
//! A [`SharedBound`] is per query, held by the collector: the `k` best exact scores the query's
//! finished segments returned in this process.
//!
//! A [`PruningBound`] is per segment scan, owned by the scan. It keeps two views. The segment's
//! own, its running top-k of lower endpoints and its exact scores, is what the router's feedback
//! and the result heap read. The pruning threshold adds the shared scores and decides admission
//! and every boundary.
//!
//! Every value either bound holds is an exact score of a distinct row or a k-th largest lower
//! endpoint over distinct rows, with k = limit + offset; under the intervals' kappa each is a
//! lower bound on the k-th best score among the rows the query searches. A row drops only when its
//! upper endpoint is strictly below the threshold, so ties are kept.

use std::sync::Mutex;

use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
use crate::vector::search::scan::columns::{Columns, QuantizedCandidates};

// --- Ordering ---

/// The `k`-th largest of two descending sequences, `a` given by length and accessor; `None`
/// while they hold fewer than `k` values together.
#[inline]
pub(crate) fn merged_kth(
    a_len: usize,
    a: impl Fn(usize) -> f32,
    b: &[f32],
    k: usize,
) -> Option<f32> {
    if k == 0 || a_len + b.len() < k {
        return None;
    }
    let (mut i, mut j) = (0, 0);
    let mut kth = f32::NAN;
    for _ in 0..k {
        if j == b.len() || (i < a_len && a(i) >= b[j]) {
            kth = a(i);
            i += 1;
        } else {
            kth = b[j];
            j += 1;
        }
    }
    Some(kth)
}

/// Orders `(lower, row, estimate)` entries by lower endpoint descending, then row ascending.
#[inline]
pub(crate) fn lower_endpoint_order(
    a: &(f32, usize, f32),
    b: &(f32, usize, f32),
) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Inserts `score` into `top`, the `k` largest values held, descending.
#[inline]
fn insert_top_k(top: &mut Vec<f32>, k: usize, score: f32) {
    if k == 0
        || (top.len() == k && score.total_cmp(top.last().unwrap()) != std::cmp::Ordering::Greater)
    {
        return;
    }
    let at = top.partition_point(|kept| kept.total_cmp(&score) != std::cmp::Ordering::Less);
    top.insert(at, score);
    top.truncate(k);
}

// --- Shared bound ---

/// What one segment's offer did to a [`SharedBound`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SharedOffer {
    /// Scores offered.
    pub(crate) offers: usize,
    /// Offers that moved the k-th: set it, or raised it.
    pub(crate) tightenings: usize,
}

/// What one query's scans share beyond their own segment: the `k` best exact scores of the rows
/// its finished segments returned in this process. The returned rows are real results, distinct
/// from every later segment's rows, so each later segment scan carries their scores into its own
/// bound. The scores only ever improve.
///
/// Keeping the `k` scores, not one segment's k-th, lets a segment that returns fewer than `k`
/// rows still raise it.
pub(crate) struct SharedBound {
    k: usize,
    /// The best `k` scores offered, descending.
    scores: Mutex<Vec<f32>>,
}

impl SharedBound {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            scores: Mutex::new(Vec::with_capacity(k)),
        }
    }

    /// Copies the best `k` scores offered, descending, into `out`.
    fn copy_scores(&self, out: &mut Vec<f32>) {
        let kept = self.scores.lock().unwrap();
        out.clear();
        out.extend_from_slice(&kept);
    }

    /// The k-th best score offered; `None` until `k` are in.
    #[cfg(test)]
    pub(crate) fn kth(&self) -> Option<f32> {
        let kept = self.scores.lock().unwrap();
        (self.k > 0 && kept.len() == self.k).then(|| kept[self.k - 1])
    }

    /// Takes the exact scores of the rows a segment returns. A non-finite score is skipped: it is
    /// no result, and a NaN k-th would admit nothing.
    pub(crate) fn offer(&self, scores: impl IntoIterator<Item = f32>) -> SharedOffer {
        let mut offer = SharedOffer::default();
        if self.k == 0 {
            return offer;
        }
        let mut kept = self.scores.lock().unwrap();
        for score in scores {
            if !score.is_finite() {
                continue;
            }
            offer.offers += 1;
            if kept.len() == self.k && score <= kept[self.k - 1] {
                continue;
            }
            let before = (kept.len() == self.k).then(|| kept[self.k - 1]);
            let at = kept.partition_point(|&held| held >= score);
            kept.insert(at, score);
            kept.truncate(self.k);
            if kept.len() == self.k && before.is_none_or(|before| kept[self.k - 1] > before) {
                offer.tightenings += 1;
            }
        }
        offer
    }

    /// Forgets every offer, for the next search of a reused collector.
    pub(crate) fn clear(&self) {
        self.scores.lock().unwrap().clear();
    }

    /// A bound already holding `k` copies of `score`.
    #[cfg(test)]
    pub(crate) fn seeded(k: usize, score: f32) -> Self {
        let bound = Self::new(k);
        bound.offer(std::iter::repeat_n(score, k));
        bound
    }
}

// --- Pruning bound ---

/// One segment scan's pruning bound: the k-th largest lower endpoint over every live candidate,
/// quantized intervals and exactly scored rows (zero-width intervals) alike, including the rows
/// the query's finished segments returned. Only the top `k` of each side can affect it, so each
/// side keeps at most `k` entries.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct PruningBound {
    k: usize,
    /// Top-k exact scores of the segment's own rows, descending. Exact rows never move.
    own_exact: Vec<f32>,
    /// Top-k quantized entries admitted since the last boundary, as `(lower, global row,
    /// estimate)`, ordered by [`lower_endpoint_order`].
    running: Vec<(f32, usize, f32)>,
    /// The estimates of `running`, descending.
    running_estimates: Vec<f32>,
    /// Top-k exact scores of every row the scan knows of, its own and those the query's finished
    /// segments returned, descending. Every score is a distinct row's.
    known_exact: Vec<f32>,
    /// One cluster's entries for the running top-k.
    entries: Vec<(f32, usize, f32)>,
    merge: Vec<(f32, usize, f32)>,
}

impl PruningBound {
    /// A bound for `k` rows that carries in `shared`'s scores.
    pub(super) fn new(k: usize, shared: Option<&SharedBound>) -> Self {
        let mut known_exact = Vec::with_capacity(k);
        if let Some(shared) = shared {
            shared.copy_scores(&mut known_exact);
            known_exact.truncate(k);
        }
        Self {
            k,
            own_exact: Vec::with_capacity(k),
            running: Vec::with_capacity(k),
            running_estimates: Vec::with_capacity(k),
            known_exact,
            entries: Vec::new(),
            merge: Vec::with_capacity(k.saturating_mul(2)),
        }
    }

    // --- The segment's own view ---

    /// The segment's own k-th lower endpoint: the k-th of the running lower endpoints and its own
    /// exact scores; `None` until `k` are in. The router's bounds gate reads it, and admission
    /// merges into the running top-k the rows it admits.
    pub(crate) fn own_lower(&self) -> Option<f32> {
        if self.own_exact.is_empty() {
            return (self.k > 0 && self.running.len() >= self.k)
                .then(|| self.running[self.k - 1].0);
        }
        merged_kth(
            self.running.len(),
            |i| self.running[i].0,
            &self.own_exact,
            self.k,
        )
    }

    /// APS cover's threshold: the k-th of the running point estimates and the segment's own exact
    /// scores; `None` until `k` are in. It never exceeds the k-th best estimate and carries no
    /// confidence margin, so it must not drive pruning.
    pub(crate) fn own_estimate(&self) -> Option<f32> {
        if self.k == 0 {
            return None;
        }
        if self.own_exact.is_empty() {
            if self.running.len() < self.k {
                return None;
            }
            return self
                .running
                .iter()
                .map(|&(_, _, estimate)| estimate)
                .min_by(f32::total_cmp);
        }
        let estimates = &self.running_estimates;
        merged_kth(estimates.len(), |i| estimates[i], &self.own_exact, self.k)
    }

    /// The k-th of the segment's own exact scores: no row scoring below it can enter its top k.
    #[inline]
    pub(super) fn own_exact_kth(&self) -> Option<f32> {
        (self.k > 0 && self.own_exact.len() == self.k).then(|| self.own_exact[self.k - 1])
    }

    // --- Recording ---

    /// Records an exactly scored row.
    pub(super) fn push_exact(&mut self, score: f32) {
        insert_top_k(&mut self.own_exact, self.k, score);
        insert_top_k(&mut self.known_exact, self.k, score);
    }

    /// Admits one cluster's rows, their combined columns in `scratch`: the rows whose upper
    /// endpoint reaches `threshold` go to `keep`, and the rows the segment's own k-th admits join
    /// the running top-k. Those are the rows a scan pruning on its own bound alone admits, so the
    /// running top-k, and the router feedback drawn from it, are that scan's; of them, only rows
    /// whose lower endpoint reaches a full running top-k's k-th are merged, since no other can
    /// enter it.
    pub(super) fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        scratch: &Columns,
        kappa: f32,
        threshold: Threshold,
        keep: &mut Vec<u32>,
    ) {
        debug_assert_eq!(scratch.estimates.len(), rows.len());
        let own = self.own_lower();
        debug_assert!(own.is_none_or(|own| own <= threshold.0 .0));
        let start = rows.rows.start;
        // Unset thresholds admit every row. Ties on the lower endpoint order by row, so an entry
        // equal to the k-th may still enter.
        let own = Threshold(LowerEndpoint(own.unwrap_or(f32::NEG_INFINITY)));
        let entry = self.running_entry().unwrap_or(f32::NEG_INFINITY);
        self.entries.clear();
        keep.clear();
        for index in 0..rows.len() {
            let estimate = Estimate(scratch.estimates[index]);
            let sigma = scratch.sigmas[index];
            let (lower, upper) = (estimate.lower(sigma, kappa), estimate.upper(sigma, kappa));
            // One branch per test, without short-circuiting: entries are rare once the running
            // top-k is full, and candidates are rare once the threshold is set.
            if own.admits(upper) & (lower.0 >= entry) {
                let offset = rows
                    .offsets
                    .map_or(index, |offsets| offsets[index] as usize);
                self.entries.push((lower.0, start + offset, estimate.0));
            }
            if threshold.admits(upper) {
                keep.push(index as u32);
            }
        }
        self.merge_entries();
    }

    /// Merges one cluster's `(lower, global row, estimate)` entries into the running top-k.
    pub(super) fn merge_cluster(&mut self, entries: impl IntoIterator<Item = (f32, usize, f32)>) {
        self.entries.clear();
        self.entries.extend(entries);
        self.merge_entries();
    }

    /// Merges `self.entries` into the running top-k through their own top-k, so the merge stays
    /// O(k).
    fn merge_entries(&mut self) {
        let k = self.k;
        if k == 0 || self.entries.is_empty() {
            return;
        }
        if self.entries.len() > k {
            self.entries
                .select_nth_unstable_by(k - 1, lower_endpoint_order);
            self.entries.truncate(k);
        }
        let entries = std::mem::take(&mut self.entries);
        self.merge_running(&entries);
        self.entries = entries;
    }

    /// Merges one cluster's top-k entries into the running top-k.
    fn merge_running(&mut self, cluster_top: &[(f32, usize, f32)]) {
        self.merge.clear();
        self.merge.extend_from_slice(&self.running);
        self.merge.extend_from_slice(cluster_top);
        self.merge.sort_unstable_by(lower_endpoint_order);
        self.merge.truncate(self.k);
        std::mem::swap(&mut self.running, &mut self.merge);
        self.running_estimates.clear();
        self.running_estimates
            .extend(self.running.iter().map(|&(_, _, estimate)| estimate));
        self.running_estimates
            .sort_unstable_by(|a, b| b.total_cmp(a));
    }

    /// The k-th running lower endpoint once the running top-k is full: an entry below it cannot
    /// enter; `None` while there is room.
    #[inline]
    fn running_entry(&self) -> Option<f32> {
        (self.k > 0 && self.running.len() == self.k).then(|| self.running[self.k - 1].0)
    }

    // --- Pruning decisions ---

    /// The pruning threshold: the k-th of the running lower endpoints and the known exact scores;
    /// `None` until `k` are in.
    #[inline]
    fn threshold(&self) -> Option<f32> {
        merged_kth(
            self.running.len(),
            |i| self.running[i].0,
            &self.known_exact,
            self.k,
        )
    }

    /// The threshold admission prunes a cluster's rows at; `None` while unset, which leaves the
    /// segment's own k-th unset too.
    pub(super) fn admission_threshold(&mut self) -> Option<Threshold> {
        let threshold = self.threshold();
        debug_assert!(threshold.is_some() || self.own_lower().is_none());
        threshold.map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// The threshold a boundary prunes `candidates` at: the k-th largest lower endpoint over the
    /// live candidates and the known exact scores, or the bound's threshold before the boundary
    /// when that is higher, which keeps every row admission dropped below it. Clears the running
    /// top-k, which describes the rows admitted before the boundary.
    pub(super) fn boundary(&mut self, candidates: &mut QuantizedCandidates) -> Option<Threshold> {
        let before = self.threshold();
        let kth = self.boundary_kth(candidates.live_lowers());
        self.running.clear();
        self.running_estimates.clear();
        let kth = match (kth, before) {
            (Some(kth), Some(before)) if before > kth => Some(before),
            (None, before) => before,
            (kth, _) => kth,
        };
        kth.map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// The k-th largest of `candidate_lowers` (one per live quantized candidate, reordered in
    /// place) and the known exact scores.
    fn boundary_kth(&self, candidate_lowers: &mut [f32]) -> Option<f32> {
        let k = self.k;
        if k == 0 {
            return None;
        }
        if self.known_exact.is_empty() {
            if candidate_lowers.len() < k {
                return None;
            }
            let (_, kth, _) = candidate_lowers.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            return Some(*kth);
        }
        if candidate_lowers.len() + self.known_exact.len() < k {
            return None;
        }
        let quantized = k.min(candidate_lowers.len());
        if quantized > 0 && quantized < candidate_lowers.len() {
            candidate_lowers.select_nth_unstable_by(quantized - 1, |a, b| b.total_cmp(a));
        }
        let top = &mut candidate_lowers[..quantized];
        top.sort_unstable_by(|a, b| b.total_cmp(a));
        merged_kth(top.len(), |i| top[i], &self.known_exact, k)
    }

    /// The known exact scores, descending.
    #[cfg(test)]
    pub(super) fn known_exact(&self) -> &[f32] {
        &self.known_exact
    }

    /// `k`, for the boundary harness.
    #[cfg(test)]
    pub(super) fn k_for_test(&self) -> usize {
        self.k
    }

    /// The running top-k, for the boundary harness.
    #[cfg(test)]
    pub(super) fn running_for_test(&self) -> &[(f32, usize, f32)] {
        &self.running
    }
}

#[cfg(test)]
mod tests {
    use super::{lower_endpoint_order, merged_kth, PruningBound, SharedBound, SharedOffer};
    use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
    use crate::vector::search::interval::{LowerEndpoint, Threshold};
    use crate::vector::search::scan::columns::BoundaryHarness;
    use crate::DocId;

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
                assert_eq!(
                    harness.running_rows(),
                    expected.iter().map(|entry| entry.0).collect::<Vec<_>>(),
                    "cluster={cluster} k={k}"
                );
                let kth =
                    (expected.len() == k).then(|| Threshold(LowerEndpoint(expected[k - 1].1)));
                assert_eq!(harness.own_lower(k), kth, "cluster={cluster} k={k}");
            }
        }

        // Two descending sequences merge to their k-th largest.
        let a = [9.0, 4.0, 1.0];
        let at = |position: usize| a[position];
        for (a_len, b, k, expected) in [
            (3, &[8.0, 5.0][..], 1, Some(9.0)),
            (3, &[8.0, 5.0], 3, Some(5.0)),
            (3, &[8.0, 5.0], 5, Some(1.0)),
            (3, &[8.0, 5.0], 6, None),
            (0, &[8.0, 5.0], 2, Some(5.0)),
            (3, &[], 0, None),
        ] {
            assert_eq!(merged_kth(a_len, at, b, k), expected, "k={k}");
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

        // With both kinds, each threshold is the k-th of the union. Lower endpoints 10, 6, -2 and
        // exact 11, 7 give 7; estimates 12, 8, 2 and exact 11, 7 give 8.
        let quantized = [(0, 12.0, 1.0), (1, 8.0, 1.0), (2, 2.0, 2.0)];
        let mut mixed = admitted(3, 2.0, &quantized);
        assert_eq!(mixed.own_lower(3), threshold(-2.0));
        mixed.push_exact(7.0, 3);
        mixed.push_exact(11.0, 3);
        assert_eq!(mixed.own_lower(3), threshold(7.0));
        assert_eq!(mixed.own_estimate(3), Some(8.0));
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

    /// The shared bound is unset until `k` scores are in, then holds the k-th best score offered
    /// whatever the order and grouping of the offers, and never loosens; an offer counts as a
    /// tightening exactly when it moves the k-th.
    #[test]
    fn shared_bound_holds_the_kth_best_offer() {
        let bound = SharedBound::new(3);
        assert_eq!(bound.kth(), None);
        let offer = bound.offer([5.0, 1.0]);
        assert_eq!(
            offer,
            SharedOffer {
                offers: 2,
                tightenings: 0
            }
        );
        assert_eq!(bound.kth(), None, "unset below k");
        assert_eq!(bound.offer([3.0]).tightenings, 1, "the k-th offer sets it");
        assert_eq!(bound.kth(), Some(1.0));
        assert_eq!(
            bound.offer([1.0, 0.5]).tightenings,
            0,
            "ties and worse keep it"
        );
        assert_eq!(bound.kth(), Some(1.0));
        assert_eq!(bound.offer([4.0]).tightenings, 1);
        assert_eq!(bound.kth(), Some(3.0));
        let skipped = bound.offer([f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!(
            skipped,
            SharedOffer::default(),
            "non-finite scores are no results"
        );
        assert_eq!(bound.kth(), Some(3.0));
        bound.clear();
        assert_eq!(bound.kth(), None);
        assert_eq!(SharedBound::new(0).offer([1.0]), SharedOffer::default());
        assert_eq!(SharedBound::seeded(4, 2.5).kth(), Some(2.5));

        for seed in 0..256 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let k = 1 + rng.usize(0..12);
            let scores: Vec<f32> = (0..rng.usize(0..40))
                .map(|_| (rng.f32() * 20.0 - 10.0).round() / 2.0)
                .collect();
            let bound = SharedBound::new(k);
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
                let report = bound.offer(offer.iter().copied());
                let case = format!("seed {seed} k={k}");
                assert_eq!(report.offers, offer.len(), "{case}");
                assert_eq!(report.tightenings, tightenings, "{case}");
                assert_eq!(bound.kth(), kth_best(&offered, k), "{case}");
                if let Some(previous) = previous {
                    assert!(bound.kth().is_some_and(|now| now >= previous), "{case}");
                }
                previous = bound.kth();
            }
        }
    }

    /// The pruning threshold is the k-th of the running lower endpoints and the carried scores,
    /// the seed and every exact score, and a boundary's k-th is that of the candidates' lower
    /// endpoints and the carried scores, whatever the order of merges, exact scores and
    /// boundaries.
    #[test]
    fn threshold_merges_running_and_carried() {
        let value = |rng: &mut fastrand::Rng| (rng.f32() * 20.0 - 10.0).round() / 2.0;
        for seed in 0..512 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let k = 1 + rng.usize(0..12);
            let case = format!("seed {seed} k={k}");
            let seeded: Vec<f32> = (0..rng.usize(0..2 * k)).map(|_| value(&mut rng)).collect();
            let shared = SharedBound::new(k);
            shared.offer(seeded.iter().copied());
            let mut bound = PruningBound::new(k, Some(&shared));
            // Every carried score and every lower endpoint merged since the last boundary.
            let mut carried = seeded.clone();
            let mut running: Vec<f32> = Vec::new();
            let mut row = 0;
            assert_eq!(bound.threshold(), kth_best(&carried, k), "{case}");
            for _ in 0..rng.usize(1..48) {
                match rng.usize(0..4) {
                    0 | 1 => {
                        let mut cluster: Vec<(f32, usize, f32)> = (0..rng.usize(1..2 * k + 2))
                            .map(|_| {
                                row += 1;
                                let lower = value(&mut rng);
                                (lower, row, lower + rng.f32())
                            })
                            .collect();
                        running.extend(cluster.iter().map(|entry| entry.0));
                        cluster.sort_by(lower_endpoint_order);
                        bound.merge_running(&cluster);
                    }
                    2 => {
                        let score = value(&mut rng);
                        carried.push(score);
                        bound.push_exact(score);
                    }
                    _ => {
                        let mut lowers: Vec<f32> =
                            (0..rng.usize(0..3 * k)).map(|_| value(&mut rng)).collect();
                        let expected = kth_best(&[&lowers[..], &carried[..]].concat(), k);
                        assert_eq!(bound.boundary_kth(&mut lowers), expected, "{case}");
                        bound.running.clear();
                        bound.running_estimates.clear();
                        running.clear();
                    }
                }
                let expected = kth_best(&[&running[..], &carried[..]].concat(), k);
                assert_eq!(bound.threshold(), expected, "{case}");
            }
            let mut top = carried.clone();
            top.sort_by(|a, b| b.total_cmp(a));
            top.truncate(k);
            assert_eq!(bound.known_exact(), top, "{case}");
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

//! The pruning bounds of one query's scans.
//!
//! The collector holds two things every segment scan shares. `carried` is the top-k exact scores
//! finished segments in this process have already returned. `global` is the threshold the query's
//! participants publish; it is a [`LocalThreshold`] when the query does not pass one in, so a scan
//! always has a threshold to read and raise.
//!
//! A [`PruningBound`] is one segment scan. Its [`TopK`] is that segment's own candidates: at most
//! `k` intervals, ordered by lower endpoint, and at most `k` exact scores. The router and APS read
//! that top-k and nothing else, which keeps probe counts on the segment-only view. Carried scores
//! and `global` raise the pruning threshold only. The threshold is the higher of a floor that only
//! rises and the k-th of the segment's lower endpoints and every exact score the scan knows.
//! Admission drops a row only when its upper endpoint is strictly below that threshold.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::collector::{AtomicSharedThreshold, SharedThreshold};
use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold, UpperEndpoint};
use crate::vector::search::scan::columns::{Columns, QuantizedCandidates};
use crate::{Score, SegmentOrdinal};

// --- Scores ---

/// A quantized interval's lower endpoint and point estimate. The upper endpoint is consumed at
/// admission; a later layer recomputes it from the refined columns. Exact scores are not intervals:
/// [`TopK`] keeps them in their own list.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bound {
    pub(crate) lo: f32,
    pub(crate) est: f32,
}

/// A row and the bound admission computed for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub(crate) row: usize,
    pub(crate) bound: Bound,
}

/// Best-first order of candidates: higher lower endpoint, then smaller row.
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
            .lo
            .total_cmp(&other.0.bound.lo)
            .then_with(|| other.0.row.cmp(&self.0.row))
    }
}

/// The `k`-th of two descending sequences. `a` is given by length and accessor.
fn kth_of(a_len: usize, a: impl Fn(usize) -> f32, b: &[f32], k: usize) -> Option<f32> {
    if k == 0 || a_len + b.len() < k {
        return None;
    }
    let (mut i, mut j) = (0, 0);
    let mut kth = 0.0;
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

/// Inserts `score` into `top`, the `k` largest values, descending.
fn insert_desc(top: &mut Vec<f32>, k: usize, score: f32) {
    if k == 0
        || (top.len() == k && score.total_cmp(top.last().unwrap()) != std::cmp::Ordering::Greater)
    {
        return;
    }
    let at = top.partition_point(|kept| kept.total_cmp(&score) != std::cmp::Ordering::Less);
    top.insert(at, score);
    top.truncate(k);
}

/// Merges two descending sequences into the `k` largest, descending.
fn merge_desc(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k.min(a.len() + b.len()));
    let (mut i, mut j) = (0, 0);
    while out.len() < k && (i < a.len() || j < b.len()) {
        if j == b.len() || (i < a.len() && a[i] >= b[j]) {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out
}

// --- Shared threshold ---

/// A process-local [`SharedThreshold<Score>`]. Queries that do not share a slot across
/// participants still hold one of these, so `global` is never absent.
#[derive(Default)]
pub(crate) struct LocalThreshold(AtomicSharedThreshold);

impl SharedThreshold<Score> for LocalThreshold {
    fn load(&self) -> Option<(Score, SegmentOrdinal)> {
        self.0.load()
    }

    fn try_update(
        &self,
        expected_threshold: &Option<(Score, SegmentOrdinal)>,
        new_threshold: (Score, SegmentOrdinal),
    ) -> Result<(), Option<(Score, SegmentOrdinal)>> {
        self.0.try_update(expected_threshold, new_threshold)
    }
}

/// Writes the empty sentinel, so the next search of a reused collector starts unset. An external
/// slot is the caller's and is left alone. Loops on [`SharedThreshold::load`] and
/// [`SharedThreshold::try_update`].
pub(crate) fn clear_threshold(global: &dyn SharedThreshold<Score>) {
    let mut current = global.load();
    loop {
        if current.is_none() {
            return;
        }
        match global.try_update(&current, (Score::MIN, SegmentOrdinal::MAX)) {
            Ok(()) => return,
            Err(actual) => current = actual,
        }
    }
}

/// Raises `global` to `score` when `score` is finite and strictly above the value loaded.
/// Returns whether this call stored it. Loops on [`SharedThreshold::load`] and
/// [`SharedThreshold::try_update`].
fn raise(global: &dyn SharedThreshold<Score>, score: f32) -> bool {
    if !score.is_finite() {
        return false;
    }
    let mut current = global.load();
    loop {
        if current.is_some_and(|(shared, _)| shared >= score) {
            return false;
        }
        match global.try_update(&current, (score, SegmentOrdinal::MAX)) {
            Ok(()) => return true,
            Err(actual) => current = actual,
        }
    }
}

/// What merging exact scores into a [`TopK`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExactMerge {
    /// Finite scores offered.
    pub(crate) offers: usize,
    /// Offers that set the k-th or raised it.
    pub(crate) tightenings: usize,
}

// --- Top-k ---

/// The `k` best candidates by lower endpoint, plus the `k` best exact scores.
///
/// Intervals and exact scores are capped separately. The k-th lower endpoint is the k-th of their
/// union, but the APS estimate is the k-th of the interval estimates and the exact scores, and an
/// exact score must stay available to later segments even when an interval's lower endpoint ranks
/// above it. One mixed heap of size `k` drops both of those.
#[derive(Clone, Debug)]
pub(crate) struct TopK {
    k: usize,
    /// At most `k` intervals. The worst lower endpoint sits on top.
    intervals: BinaryHeap<Reverse<ByLower>>,
    /// At most `k` exact scores, descending.
    exact: Vec<f32>,
}

impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            intervals: BinaryHeap::new(),
            exact: Vec::with_capacity(k),
        }
    }

    fn k(&self) -> usize {
        self.k
    }

    /// Exact scores, descending.
    pub(crate) fn exact_scores(&self) -> &[f32] {
        &self.exact
    }

    /// The k-th exact score, once `k` are in.
    pub(crate) fn exact_kth(&self) -> Option<f32> {
        (self.k > 0 && self.exact.len() == self.k).then(|| self.exact[self.k - 1])
    }

    /// The k-th lower endpoint of the intervals and the exact scores. `None` until `k` values are
    /// in. This is the segment's own k-th when the top-k holds only that segment.
    pub(crate) fn kth_lower(&self) -> Option<f32> {
        let lowers = self.lowers_desc();
        kth_of(lowers.len(), |i| lowers[i], &self.exact, self.k)
    }

    /// APS cover: the k-th of the interval point estimates and the exact scores. `None` until `k`
    /// values are in. It carries no confidence margin, so it must not prune.
    pub(crate) fn kth_estimate(&self) -> Option<f32> {
        let estimates = self.estimates_desc();
        kth_of(estimates.len(), |i| estimates[i], &self.exact, self.k)
    }

    /// Records an exact score. A non-finite score is kept, matching a segment scan's own scores;
    /// [`TopK::merge_exact`] is what drops them when the score is offered across segments.
    pub(crate) fn push_exact(&mut self, score: f32) {
        insert_desc(&mut self.exact, self.k, score);
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
            if self.exact.len() == self.k && score <= self.exact[self.k - 1] {
                continue;
            }
            let before = (self.exact.len() == self.k).then(|| self.exact[self.k - 1]);
            insert_desc(&mut self.exact, self.k, score);
            if self.exact.len() == self.k
                && before.is_none_or(|before| self.exact[self.k - 1] > before)
            {
                merge.tightenings += 1;
            }
        }
        merge
    }

    /// Merges interval candidates into the top-k.
    fn merge_intervals(&mut self, mut batch: Vec<Candidate>) {
        let k = self.k;
        if k == 0 || batch.is_empty() {
            return;
        }
        if batch.len() > k {
            batch.select_nth_unstable_by(k - 1, |a, b| ByLower(*b).cmp(&ByLower(*a)));
            batch.truncate(k);
        }
        for candidate in batch {
            self.push_interval(candidate);
        }
    }

    fn push_interval(&mut self, candidate: Candidate) {
        if self.k == 0 {
            return;
        }
        let item = Reverse(ByLower(candidate));
        if self.intervals.len() < self.k {
            self.intervals.push(item);
            return;
        }
        if item.0 > self.intervals.peek().unwrap().0 {
            self.intervals.pop();
            self.intervals.push(item);
        }
    }

    /// The worst interval lower endpoint once `k` intervals are in.
    fn interval_entry(&self) -> Option<f32> {
        (self.k > 0 && self.intervals.len() == self.k)
            .then(|| self.intervals.peek().unwrap().0 .0.bound.lo)
    }

    fn lowers_desc(&self) -> Vec<f32> {
        let mut lowers: Vec<f32> = self
            .intervals
            .iter()
            .map(|Reverse(ByLower(candidate))| candidate.bound.lo)
            .collect();
        lowers.sort_by(|a, b| b.total_cmp(a));
        lowers
    }

    fn estimates_desc(&self) -> Vec<f32> {
        let mut estimates: Vec<f32> = self
            .intervals
            .iter()
            .map(|Reverse(ByLower(candidate))| candidate.bound.est)
            .collect();
        estimates.sort_by(|a, b| b.total_cmp(a));
        estimates
    }

    fn clear_intervals(&mut self) {
        self.intervals.clear();
    }

    /// Drops every candidate. The collector calls this between searches.
    pub(crate) fn clear(&mut self) {
        self.intervals.clear();
        self.exact.clear();
    }

    /// `(lower, row, estimate)` of the intervals, best first.
    #[cfg(test)]
    fn intervals_for_test(&self) -> Vec<(f32, usize, f32)> {
        let mut rows: Vec<(f32, usize, f32)> = self
            .intervals
            .iter()
            .map(|Reverse(ByLower(candidate))| {
                (candidate.bound.lo, candidate.row, candidate.bound.est)
            })
            .collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        rows
    }
}

// --- Pruning bound ---

/// One segment scan's pruning bound.
///
/// `top` is the segment's own candidates. `carried` is the exact scores cloned from finished
/// segments at the start of the scan; they raise the threshold and are not part of `top`, so the
/// router and APS stay on the segment-only view. `tau` is the floor: the highest threshold a
/// boundary, a slot read, or an explicit raise has committed. It only rises. The threshold a
/// decision reads is the higher of `tau` and the k-th of `top`'s lower endpoints and every exact
/// score (`top`'s and `carried`).
#[cfg_attr(test, derive(Clone))]
pub(crate) struct PruningBound {
    /// Committed floor. Only rises. `NEG_INFINITY` until the first raise.
    tau: f32,
    top: TopK,
    carried: TopK,
    global: Arc<dyn SharedThreshold<Score>>,
    /// Highest threshold offered to `global`.
    last_published: f32,
    /// Offers that stored a new value in `global`.
    pub(super) publishes: usize,
    /// Loads of `global` whose score was above the threshold.
    pub(super) raises: usize,
    /// Threshold the last boundary pruned at. Recheck does nothing until the threshold rises past
    /// it, because every survivor already cleared it.
    pruned_at: Option<f32>,
}

impl PruningBound {
    /// A bound for `k` rows. `carried` is the exact top-k cloned from finished segments (its `k`
    /// matches). `global` is the participants' threshold, or a [`LocalThreshold`].
    pub(super) fn new(k: usize, carried: TopK, global: Arc<dyn SharedThreshold<Score>>) -> Self {
        debug_assert_eq!(carried.k(), k);
        Self {
            tau: f32::NEG_INFINITY,
            top: TopK::new(k),
            carried,
            global,
            last_published: f32::NEG_INFINITY,
            publishes: 0,
            raises: 0,
            pruned_at: None,
        }
    }

    /// No carried scores, and a fresh process-local threshold.
    #[cfg(test)]
    pub(super) fn standalone(k: usize) -> Self {
        Self::new(k, TopK::new(k), Arc::new(LocalThreshold::default()))
    }

    /// Gives the bound its own local threshold. A cloned bound publishes here without writing the
    /// threshold it was cloned from, so a peek cannot arm the original.
    #[cfg(test)]
    pub(super) fn detach_global(&mut self) {
        self.global = Arc::new(LocalThreshold::default());
        self.last_published = f32::NEG_INFINITY;
        self.publishes = 0;
        self.raises = 0;
    }

    // --- The segment's own view ---

    /// The segment's own k-th lower endpoint. The router reads this.
    pub(crate) fn own_lower(&self) -> Option<f32> {
        self.top.kth_lower()
    }

    /// The segment's own APS estimate. Carried scores and `global` are not part of it.
    pub(crate) fn own_estimate(&self) -> Option<f32> {
        self.top.kth_estimate()
    }

    /// The k-th of this segment's exact scores. No lower score can enter its result heap.
    #[inline]
    pub(super) fn exact_kth(&self) -> Option<f32> {
        self.top.exact_kth()
    }

    // --- Recording ---

    /// Records an exactly scored row of this segment.
    pub(super) fn push_exact(&mut self, score: f32) {
        self.top.push_exact(score);
    }

    /// Admits one cluster. Rows whose upper endpoint reaches `threshold` go to `keep`. Rows the
    /// segment's own k-th admits join `top`, including rows `threshold` rejects: that top-k is
    /// what a scan pruning on its own bound alone would hold, and the router reads it. Publishes
    /// afterwards.
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
        let own = own.unwrap_or(f32::NEG_INFINITY);
        let entry = self.top.interval_entry().unwrap_or(f32::NEG_INFINITY);
        let mut batch = Vec::new();
        keep.clear();
        for index in 0..rows.len() {
            let estimate = Estimate(scratch.estimates[index]);
            let sigma = scratch.sigmas[index];
            let (lower, upper) = (estimate.lower(sigma, kappa), estimate.upper(sigma, kappa));
            // One branch per test, without short-circuiting: entries are rare once the top-k is
            // full, and candidates are rare once the threshold is set. The segment's own top-k
            // admits on the upper endpoint; the lower endpoint only decides heap membership.
            if upper_reaches(own, upper) & (lower.0 >= entry) {
                let offset = rows
                    .offsets
                    .map_or(index, |offsets| offsets[index] as usize);
                batch.push(Candidate {
                    row: start + offset,
                    bound: Bound {
                        lo: lower.0,
                        est: estimate.0,
                    },
                });
            }
            if threshold.admits(upper) {
                keep.push(index as u32);
            }
        }
        self.top.merge_intervals(batch);
        self.publish();
    }

    /// Merges one cluster's candidates into the segment top-k and publishes.
    pub(super) fn merge_cluster(&mut self, entries: impl IntoIterator<Item = Candidate>) {
        self.top.merge_intervals(entries.into_iter().collect());
        self.publish();
    }

    /// Merges `(lower, row, estimate)` entries. The harness and the direct tests build them this
    /// way; the upper endpoint is unused once the row has been admitted.
    #[cfg(test)]
    fn merge_running(&mut self, cluster: &[(f32, usize, f32)]) {
        self.top.merge_intervals(
            cluster
                .iter()
                .map(|&(lo, row, est)| Candidate {
                    row,
                    bound: Bound { lo, est },
                })
                .collect(),
        );
    }

    // --- Pruning decisions ---

    /// Exact scores the threshold merges: this segment's and the carried ones, descending.
    fn exacts_desc(&self) -> Vec<f32> {
        merge_desc(
            self.top.exact_scores(),
            self.carried.exact_scores(),
            self.top.k(),
        )
    }

    /// The k-th of the live interval lower endpoints and every exact score. `None` while both are
    /// short of `k` and the floor is unset.
    fn live_kth(&self) -> Option<f32> {
        let lowers = self.top.lowers_desc();
        let exact = self.exacts_desc();
        kth_of(lowers.len(), |i| lowers[i], &exact, self.top.k())
    }

    /// The pruning threshold: the higher of the floor and [`PruningBound::live_kth`].
    fn threshold(&self) -> Option<f32> {
        let floor = (self.tau != f32::NEG_INFINITY).then_some(self.tau);
        match (floor, self.live_kth()) {
            (Some(floor), Some(kth)) => Some(if kth > floor { kth } else { floor }),
            (Some(floor), None) => Some(floor),
            (None, kth) => kth,
        }
    }

    /// The threshold admission prunes at, after reading `global`.
    pub(super) fn admission_threshold(&mut self) -> Option<Threshold> {
        self.sync();
        let threshold = self.threshold();
        debug_assert!(threshold.is_some() || self.own_lower().is_none());
        threshold.map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// The threshold a block is rechecked at, after reading `global`. `None` until the threshold
    /// rises past the last boundary.
    pub(super) fn recheck_threshold(&mut self) -> Option<Threshold> {
        self.sync();
        let threshold = self.threshold()?;
        if self
            .pruned_at
            .is_some_and(|pruned_at| threshold <= pruned_at)
        {
            return None;
        }
        Some(Threshold(LowerEndpoint(threshold)))
    }

    /// The threshold a boundary prunes at, after reading `global`.
    ///
    /// Commits the current threshold to the floor, then the k-th of the live candidates' lower
    /// endpoints and the exact scores. Clears the segment's intervals: they describe the layer
    /// just finished, and the floor holds their threshold. The next layer's boundary reads the
    /// refined columns. Publishes afterwards.
    pub(super) fn boundary(&mut self, candidates: &mut QuantizedCandidates) -> Option<Threshold> {
        self.sync();
        if let Some(before) = self.threshold() {
            self.raise_tau(before);
        }
        let kth = self.boundary_kth(candidates.live_lowers());
        self.top.clear_intervals();
        if let Some(kth) = kth {
            self.raise_tau(kth);
        }
        let threshold = self.threshold();
        self.pruned_at = threshold;
        self.publish();
        threshold.map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// The k-th of `candidate_lowers` (reordered in place) and the exact scores.
    fn boundary_kth(&self, candidate_lowers: &mut [f32]) -> Option<f32> {
        let k = self.top.k();
        let exact = self.exacts_desc();
        if k == 0 {
            return None;
        }
        if exact.is_empty() {
            if candidate_lowers.len() < k {
                return None;
            }
            let (_, kth, _) = candidate_lowers.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            return Some(*kth);
        }
        if candidate_lowers.len() + exact.len() < k {
            return None;
        }
        let quantized = k.min(candidate_lowers.len());
        if quantized > 0 && quantized < candidate_lowers.len() {
            candidate_lowers.select_nth_unstable_by(quantized - 1, |a, b| b.total_cmp(a));
        }
        let top = &mut candidate_lowers[..quantized];
        top.sort_unstable_by(|a, b| b.total_cmp(a));
        kth_of(top.len(), |i| top[i], &exact, k)
    }

    /// Raises the floor to `value`. A lower value or NaN leaves it.
    fn raise_tau(&mut self, value: f32) {
        if !value.is_nan() && value > self.tau {
            self.tau = value;
        }
    }

    // --- Participants' threshold ---

    /// Raises the floor to `global`'s score. The ordinal is ignored: pruning keeps ties.
    fn sync(&mut self) {
        let Some((score, _)) = self.global.load() else {
            return;
        };
        if self.threshold().is_none_or(|threshold| score > threshold) {
            self.raises += 1;
        }
        self.raise_tau(score);
    }

    /// Offers the threshold to `global` when it is finite and above the last offer.
    pub(super) fn publish(&mut self) {
        let Some(kth) = self.threshold() else {
            return;
        };
        if !kth.is_finite() || kth <= self.last_published {
            return;
        }
        self.last_published = kth;
        if raise(&*self.global, kth) {
            self.publishes += 1;
        }
    }

    /// The exact scores the scan knows, this segment's and the carried ones, descending.
    #[cfg(test)]
    pub(super) fn known_exact(&self) -> Vec<f32> {
        self.exacts_desc()
    }

    #[cfg(test)]
    pub(super) fn k_for_test(&self) -> usize {
        self.top.k()
    }

    #[cfg(test)]
    pub(super) fn running_for_test(&self) -> Vec<(f32, usize, f32)> {
        self.top.intervals_for_test()
    }

    #[cfg(test)]
    pub(super) fn threshold_for_test(&mut self) -> Option<f32> {
        self.threshold()
    }

    #[cfg(test)]
    pub(super) fn raise_floor_for_test(&mut self, value: f32) {
        self.raise_tau(value);
    }

    #[cfg(test)]
    fn clear_running(&mut self) {
        self.top.clear_intervals();
    }
}

/// `upper` reaches a lower-endpoint threshold. Ties are kept.
#[inline]
fn upper_reaches(threshold: f32, upper: UpperEndpoint) -> bool {
    upper.0 >= threshold
}

/// Best-first order of `(lower, row, estimate)`: higher lower endpoint, then smaller row.
#[cfg(test)]
fn lower_endpoint_order(a: &(f32, usize, f32), b: &(f32, usize, f32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use super::{kth_of, lower_endpoint_order, ExactMerge, LocalThreshold, PruningBound, TopK};
    use crate::collector::{AtomicSharedThreshold, SharedThreshold};
    use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
    use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
    use crate::vector::search::scan::columns::BoundaryHarness;
    use crate::{DocId, Score, SegmentOrdinal};

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
            assert_eq!(kth_of(a_len, at, b, k), expected, "k={k}");
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

    /// The carried top-k is unset until `k` scores are in, then holds the k-th best score offered
    /// whatever the order and grouping of the offers, and never loosens; an offer counts as a
    /// tightening exactly when it moves the k-th.
    #[test]
    fn carried_top_k_holds_the_kth_best_offer() {
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
            let mut carried_top = TopK::new(k);
            carried_top.merge_exact(seeded.iter().copied());
            let mut bound = PruningBound::new(k, carried_top, Arc::new(LocalThreshold::default()));
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
                        bound.clear_running();
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

    /// The pruning threshold never falls: not as clusters merge into the running top-k, exact
    /// scores arrive, a boundary clears the running top-k, or the floor is raised.
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
                match rng.usize(0..4) {
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
                    2 => harness.band(k, 2.0),
                    _ => harness.raise_floor(k, value(&mut rng)),
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

    /// An in-process slot that counts its loads and updates.
    #[derive(Default)]
    struct CountingSlot {
        slot: AtomicSharedThreshold,
        loads: AtomicUsize,
        updates: AtomicUsize,
    }

    impl SharedThreshold<Score> for CountingSlot {
        fn load(&self) -> Option<(Score, SegmentOrdinal)> {
            self.loads.fetch_add(1, Ordering::Relaxed);
            self.slot.load()
        }

        fn try_update(
            &self,
            expected: &Option<(Score, SegmentOrdinal)>,
            new: (Score, SegmentOrdinal),
        ) -> Result<(), Option<(Score, SegmentOrdinal)>> {
            let updated = self.slot.try_update(expected, new);
            if updated.is_ok() {
                self.updates.fetch_add(1, Ordering::Relaxed);
            }
            updated
        }
    }

    impl CountingSlot {
        fn score(&self) -> Option<Score> {
            self.slot.load().map(|(score, _)| score)
        }

        fn counts(&self) -> (usize, usize) {
            (
                self.loads.load(Ordering::Relaxed),
                self.updates.load(Ordering::Relaxed),
            )
        }
    }

    /// A bound at `k` sharing `global`, holding `scores` as this segment's exact scores.
    fn exact_bound(
        global: Arc<dyn SharedThreshold<Score>>,
        k: usize,
        scores: &[f32],
    ) -> PruningBound {
        let mut bound = PruningBound::new(k, TopK::new(k), global);
        for &score in scores {
            bound.push_exact(score);
        }
        bound
    }

    /// A bound publishes its threshold only when it is finite and above its last offer, the slot
    /// takes it only when it is above the slot's score, and a sync raises the threshold to the
    /// slot's score.
    #[test]
    fn publishes_only_higher_finite_thresholds() {
        let slot = Arc::new(CountingSlot::default());
        let global: Arc<dyn SharedThreshold<Score>> = slot.clone();
        exact_bound(Arc::clone(&global), 2, &[]).publish();
        assert_eq!(slot.counts(), (0, 0), "an unset threshold is never offered");
        let mut first = exact_bound(Arc::clone(&global), 2, &[5.0, 3.0]);
        first.publish();
        assert_eq!((slot.score(), slot.counts()), (Some(3.0), (1, 1)));
        first.publish();
        assert_eq!(slot.counts(), (1, 1), "a repeat offer reads nothing");

        let mut second = exact_bound(Arc::clone(&global), 2, &[5.0, 2.0]);
        second.publish();
        assert_eq!(
            (slot.score(), slot.counts()),
            (Some(3.0), (2, 1)),
            "lower is kept out"
        );
        second.publish();
        assert_eq!(slot.counts(), (2, 1), "a repeat offer reads nothing");
        second.push_exact(4.0);
        second.publish();
        assert_eq!((slot.score(), slot.counts()), (Some(4.0), (3, 2)));
        assert_eq!((first.publishes, second.publishes), (1, 1));

        for score in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut bound = exact_bound(Arc::clone(&global), 1, &[score]);
            bound.publish();
            assert_eq!(
                (slot.score(), slot.counts(), bound.publishes),
                (Some(4.0), (3, 2), 0),
                "{score} is never written"
            );
        }

        let mut synced = exact_bound(Arc::clone(&global), 2, &[5.0, 3.0]);
        synced.sync();
        assert_eq!((synced.threshold(), synced.raises), (Some(4.0), 1));
        synced.sync();
        assert_eq!(synced.raises, 1, "an equal score raises nothing");
        let mut above = exact_bound(Arc::clone(&global), 2, &[9.0, 8.0]);
        above.sync();
        assert_eq!((above.threshold(), above.raises), (Some(8.0), 0));
        above.publish();
        assert_eq!((slot.score(), above.publishes), (Some(8.0), 1));

        let mut unshared = PruningBound::standalone(2);
        for score in [9.0, 8.0] {
            unshared.push_exact(score);
        }
        unshared.sync();
        unshared.publish();
        // No participant slot was supplied, but the process-local threshold still takes the one
        // finite offer. Nothing raises the floor from a load.
        assert_eq!((unshared.raises, unshared.publishes), (0, 1));
    }

    /// Threads publishing interleaved, rising thresholds into one slot, some also syncing from
    /// it, leave the slot at the highest threshold any of them reached.
    #[test]
    fn concurrent_publishes_leave_the_slot_at_the_max() {
        for seed in 0..32 {
            let slot = Arc::new(AtomicSharedThreshold::default());
            let global: Arc<dyn SharedThreshold<Score>> = slot.clone();
            let threads = 2 + seed as usize % 7;
            let barrier = Barrier::new(threads);
            let finals: Vec<Option<f32>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|thread| {
                        let (global, barrier) = (Arc::clone(&global), &barrier);
                        scope.spawn(move || {
                            let mut rng = fastrand::Rng::with_seed(seed * 64 + thread as u64);
                            let k = 1 + rng.usize(0..6);
                            let mut bound = PruningBound::new(k, TopK::new(k), global);
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

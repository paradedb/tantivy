//! The scan's pruning bound: the k-th largest lower endpoint over quantized intervals and exact
//! scores.

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

// --- Bound ---

/// The scan's one pruning bound: the k-th largest lower endpoint over every live candidate,
/// quantized intervals and exactly scored rows (zero-width intervals) alike. Only the top `k` of
/// each side can affect it, so each side keeps at most `k` entries.
#[derive(Clone)]
pub(crate) struct KthBound {
    pub(super) k: usize,
    /// Top-k exact scores, descending. Exact rows never move.
    exact: Vec<f32>,
    /// Top-k quantized entries admitted since the last boundary, as `(lower, row, estimate)`,
    /// ordered by [`lower_endpoint_order`].
    pub(super) running: Vec<(f32, usize, f32)>,
    /// The estimates of `running`, descending.
    running_estimates: Vec<f32>,
    merge: Vec<(f32, usize, f32)>,
}

impl KthBound {
    pub(super) fn new(k: usize) -> Self {
        Self {
            k,
            exact: Vec::with_capacity(k),
            running: Vec::with_capacity(k),
            running_estimates: Vec::with_capacity(k),
            merge: Vec::with_capacity(k.saturating_mul(2)),
        }
    }

    /// Records an exactly scored row.
    pub(super) fn push_exact(&mut self, score: f32) {
        if self.k == 0
            || (self.exact.len() == self.k
                && score.total_cmp(self.exact.last().unwrap()) != std::cmp::Ordering::Greater)
        {
            return;
        }
        let at = self
            .exact
            .partition_point(|kept| kept.total_cmp(&score) != std::cmp::Ordering::Less);
        self.exact.insert(at, score);
        self.exact.truncate(self.k);
    }

    /// The k-th exact score: no row scoring below it can enter the top k.
    #[inline]
    pub(super) fn exact_kth(&self) -> Option<f32> {
        (self.k > 0 && self.exact.len() == self.k).then(|| self.exact[self.k - 1])
    }

    /// Merges one cluster's top-k entries into the running top-k.
    pub(super) fn merge_running(&mut self, cluster_top: &[(f32, usize, f32)]) {
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

    /// The bounds gate's threshold: the k-th of the running lower endpoints and the exact scores;
    /// `None` until `k` are in.
    pub(crate) fn running_lower(&self) -> Option<f32> {
        if self.exact.is_empty() {
            return (self.k > 0 && self.running.len() >= self.k)
                .then(|| self.running[self.k - 1].0);
        }
        merged_kth(
            self.running.len(),
            |i| self.running[i].0,
            &self.exact,
            self.k,
        )
    }

    /// APS cover's threshold: the k-th of the running point estimates and the exact scores;
    /// `None` until `k` are in. It never exceeds the k-th best estimate and carries no confidence
    /// margin, so it must not drive pruning.
    pub(crate) fn running_estimate(&self) -> Option<f32> {
        if self.k == 0 {
            return None;
        }
        if self.exact.is_empty() {
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
        merged_kth(estimates.len(), |i| estimates[i], &self.exact, self.k)
    }

    /// The boundary threshold: the k-th largest of `candidate_lowers` (one per live quantized
    /// candidate, reordered in place) and the exact scores. Clears the running top-k, which
    /// describes the candidates before the boundary.
    pub(super) fn boundary_lower(&mut self, candidate_lowers: &mut [f32]) -> Option<f32> {
        self.running.clear();
        self.running_estimates.clear();
        let k = self.k;
        if k == 0 {
            return None;
        }
        if self.exact.is_empty() {
            if candidate_lowers.len() < k {
                return None;
            }
            let (_, kth, _) = candidate_lowers.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            return Some(*kth);
        }
        if candidate_lowers.len() + self.exact.len() < k {
            return None;
        }
        let quantized = k.min(candidate_lowers.len());
        if quantized > 0 && quantized < candidate_lowers.len() {
            candidate_lowers.select_nth_unstable_by(quantized - 1, |a, b| b.total_cmp(a));
        }
        let top = &mut candidate_lowers[..quantized];
        top.sort_unstable_by(|a, b| b.total_cmp(a));
        merged_kth(top.len(), |i| top[i], &self.exact, k)
    }
}

#[cfg(test)]
mod tests {
    use super::merged_kth;
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
        assert_eq!(tie.running_lower(2), threshold(7.0));
        assert_eq!(tie.boundary_lower(2, 2.0), threshold(7.0));

        // Lower endpoints 10, 7, 4: the top two by lower endpoint are rows 0 and 1, so the APS
        // threshold is 9 though row 2 has the best estimate.
        let mut empty = BoundaryHarness::new(3);
        assert_eq!(empty.running_estimate(2), None);
        let mut aps = admitted(2, 2.0, &[(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 12.0, 4.0)]);
        assert_eq!(aps.running_lower(2), threshold(7.0));
        assert_eq!(aps.running_estimate(2), Some(9.0));
        assert_eq!(aps.running_estimate(4), None);

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
                clustered.running_lower(4),
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
                assert_eq!(harness.running_lower(k), kth, "cluster={cluster} k={k}");
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
        assert_eq!(exact.running_lower(3), threshold(5.0));
        assert_eq!(exact.running_estimate(3), Some(5.0));
        assert_eq!(exact.boundary_lower(3, 2.0), threshold(5.0));
        let mut short = BoundaryHarness::new(8);
        for score in [3.0, 9.0, 5.0] {
            short.push_exact(score, 4);
        }
        assert_eq!(short.running_lower(4), None);
        assert_eq!(short.running_estimate(4), None);
        assert_eq!(short.boundary_lower(4, 2.0), None);

        // With both kinds, each threshold is the k-th of the union. Lower endpoints 10, 6, -2 and
        // exact 11, 7 give 7; estimates 12, 8, 2 and exact 11, 7 give 8.
        let quantized = [(0, 12.0, 1.0), (1, 8.0, 1.0), (2, 2.0, 2.0)];
        let mut mixed = admitted(3, 2.0, &quantized);
        assert_eq!(mixed.running_lower(3), threshold(-2.0));
        mixed.push_exact(7.0, 3);
        mixed.push_exact(11.0, 3);
        assert_eq!(mixed.running_lower(3), threshold(7.0));
        assert_eq!(mixed.running_estimate(3), Some(8.0));
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
                    let running = harness.running_lower(k);
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

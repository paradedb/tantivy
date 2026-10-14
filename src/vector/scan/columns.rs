//! Row-parallel quantized candidate columns, with admission into the bound and boundary selection.

use std::ops::Range;

use super::bound::{lower_endpoint_order, KthBound};
use crate::vector::interval::{Estimate, LowerEndpoint, Threshold};
use crate::vector::prepared::ArithmeticError;
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::DocId;

/// One row surviving a quantized boundary.
#[derive(Clone, Copy)]
struct QuantizedCandidate {
    row: usize,
    doc: DocId,
    base: f32,
    raw_prefix: f32,
    estimate: f32,
    sigma: f32,
    residual_norm_squared: f32,
    gamma: f32,
    sign_query_error_term: f32,
    arithmetic_variance: ArithmeticError,
}

/// Row-parallel quantized scan columns, with the admission and boundary selection over them.
pub(super) struct QuantizedCandidates {
    pub(super) rows: Vec<usize>,
    pub(super) docs: Vec<DocId>,
    pub(super) bases: Vec<f32>,
    pub(super) raw_prefixes: Vec<f32>,
    pub(super) estimates: Vec<f32>,
    pub(super) sigmas: Vec<f32>,
    pub(super) residual_norm_squared: Vec<f32>,
    pub(super) gammas: Vec<f32>,
    pub(super) sign_query_error_terms: Vec<f32>,
    pub(super) arithmetic_variances: Vec<ArithmeticError>,
    /// Confidence multiplier of every interval endpoint.
    kappa: f32,
    boundary_scratch: Vec<QuantizedCandidate>,
    cluster_top: Vec<(f32, usize, f32)>,
    lowers: Vec<f32>,
}

impl QuantizedCandidates {
    #[cfg(test)]
    pub(super) fn estimate_trace(&self) -> Vec<(usize, u32)> {
        (0..self.len())
            .map(|i| (self.rows[i], self.estimates[i].to_bits()))
            .collect()
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            docs: Vec::with_capacity(capacity),
            bases: Vec::with_capacity(capacity),
            raw_prefixes: Vec::with_capacity(capacity),
            estimates: Vec::with_capacity(capacity),
            sigmas: Vec::with_capacity(capacity),
            residual_norm_squared: Vec::with_capacity(capacity),
            gammas: Vec::with_capacity(capacity),
            sign_query_error_terms: Vec::with_capacity(capacity),
            arithmetic_variances: Vec::with_capacity(capacity),
            kappa: QUANTIZED_BOUNDARY_KAPPA,
            boundary_scratch: Vec::with_capacity(capacity),
            cluster_top: Vec::new(),
            lowers: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(super) fn push(
        &mut self,
        row: usize,
        doc: DocId,
        base: f32,
        raw_prefix: f32,
        estimate: f32,
        sigma: f32,
        residual_norm_squared: f32,
        gamma: f32,
        sign_query_error_term: f32,
        arithmetic_variance: ArithmeticError,
    ) {
        self.rows.push(row);
        self.docs.push(doc);
        self.bases.push(base);
        self.raw_prefixes.push(raw_prefix);
        self.estimates.push(estimate);
        self.sigmas.push(sigma);
        self.residual_norm_squared.push(residual_norm_squared);
        self.gammas.push(gamma);
        self.sign_query_error_terms.push(sign_query_error_term);
        self.arithmetic_variances.push(arithmetic_variance);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_selected(
        &mut self,
        rows: Range<usize>,
        offsets: Option<&[usize]>,
        doc_ids: Option<&[DocId]>,
        bases: &[f32],
        raw_prefixes: &[f32],
        estimates: &[f32],
        sigmas: &[f32],
        residual_norms_squared: &[f32],
        gammas: &[f32],
        sign_query_error_terms: &[f32],
        arithmetic_variances: &[ArithmeticError],
    ) {
        let len = offsets.map_or(rows.len(), <[usize]>::len);
        debug_assert_eq!(bases.len(), len);
        debug_assert_eq!(raw_prefixes.len(), len);
        debug_assert_eq!(estimates.len(), len);
        debug_assert_eq!(sigmas.len(), len);
        debug_assert_eq!(residual_norms_squared.len(), len);
        debug_assert_eq!(gammas.len(), len);
        debug_assert_eq!(sign_query_error_terms.len(), len);
        debug_assert_eq!(arithmetic_variances.len(), len);
        match offsets {
            None => self.rows.extend(rows),
            Some(offsets) => self.rows.extend(offsets.iter().map(|&offset| {
                debug_assert!(offset < rows.len());
                rows.start + offset
            })),
        }
        match doc_ids {
            Some(docs) => {
                debug_assert_eq!(docs.len(), len);
                self.docs.extend_from_slice(docs);
            }
            // Filler: the cluster's admitted record says its doc ids come from the column.
            None => self.docs.resize(self.docs.len() + len, 0),
        }
        self.bases.extend_from_slice(bases);
        self.raw_prefixes.extend_from_slice(raw_prefixes);
        self.estimates.extend_from_slice(estimates);
        self.sigmas.extend_from_slice(sigmas);
        self.residual_norm_squared
            .extend_from_slice(residual_norms_squared);
        self.gammas.extend_from_slice(gammas);
        self.sign_query_error_terms
            .extend_from_slice(sign_query_error_terms);
        self.arithmetic_variances
            .extend_from_slice(arithmetic_variances);
    }

    #[inline(always)]
    fn estimate(&self, index: usize) -> Estimate {
        Estimate(self.estimates[index])
    }

    fn materialize(&self, index: usize) -> QuantizedCandidate {
        QuantizedCandidate {
            row: self.rows[index],
            doc: self.docs[index],
            base: self.bases[index],
            raw_prefix: self.raw_prefixes[index],
            estimate: self.estimates[index],
            sigma: self.sigmas[index],
            residual_norm_squared: self.residual_norm_squared[index],
            gamma: self.gammas[index],
            sign_query_error_term: self.sign_query_error_terms[index],
            arithmetic_variance: self.arithmetic_variances[index],
        }
    }

    #[inline]
    fn lower(&self, index: usize) -> f32 {
        self.estimate(index).lower(self.sigmas[index], self.kappa).0
    }

    /// Merges the candidates appended since `cluster_start` into the bound's running top-k,
    /// through the cluster's own top-k so the merge stays O(k).
    pub(super) fn finish_cluster(&mut self, cluster_start: usize, bound: &mut KthBound) {
        let k = bound.k;
        if k == 0 || cluster_start >= self.len() {
            return;
        }
        let mut cluster_top = std::mem::take(&mut self.cluster_top);
        cluster_top.clear();
        cluster_top.extend(
            (cluster_start..self.len())
                .map(|index| (self.lower(index), self.rows[index], self.estimates[index])),
        );
        if cluster_top.len() > k {
            cluster_top.select_nth_unstable_by(k - 1, lower_endpoint_order);
            cluster_top.truncate(k);
        }
        bound.merge_running(&cluster_top);
        self.cluster_top = cluster_top;
    }

    /// Keeps the candidates whose upper endpoint reaches the boundary threshold, row-ordered.
    /// Selecting by lower endpoint itself matters: ordering by estimate can prune a true top-k
    /// row even when every confidence interval encloses its exact score.
    pub(super) fn band(&mut self, bound: &mut KthBound) {
        debug_assert!(
            self.estimates
                .iter()
                .zip(&self.sigmas)
                .all(|(&estimate, &sigma)| estimate.is_finite() && sigma.is_finite()),
            "quantized boundary inputs must be finite"
        );
        let mut lowers = std::mem::take(&mut self.lowers);
        lowers.clear();
        lowers.extend((0..self.len()).map(|index| self.lower(index)));
        let threshold = bound
            .boundary_lower(&mut lowers)
            .map(|kth| Threshold(LowerEndpoint(kth)));
        self.lowers = lowers;
        let mut survivors = std::mem::take(&mut self.boundary_scratch);
        survivors.clear();
        for index in 0..self.len() {
            if threshold.is_none_or(|kth| {
                kth.admits(self.estimate(index).upper(self.sigmas[index], self.kappa))
            }) {
                survivors.push(self.materialize(index));
            }
        }
        survivors.sort_unstable_by_key(|candidate| candidate.row);
        self.replace_with_boundary_survivors(&survivors);
        self.boundary_scratch = survivors;
    }

    fn replace_with_boundary_survivors(&mut self, survivors: &[QuantizedCandidate]) {
        self.rows.clear();
        self.docs.clear();
        self.bases.clear();
        self.raw_prefixes.clear();
        self.estimates.clear();
        self.sigmas.clear();
        self.residual_norm_squared.clear();
        self.gammas.clear();
        self.sign_query_error_terms.clear();
        self.arithmetic_variances.clear();
        self.rows.reserve(survivors.len());
        self.docs.reserve(survivors.len());
        self.bases.reserve(survivors.len());
        self.raw_prefixes.reserve(survivors.len());
        self.estimates.reserve(survivors.len());
        self.sigmas.reserve(survivors.len());
        self.residual_norm_squared.reserve(survivors.len());
        self.gammas.reserve(survivors.len());
        self.sign_query_error_terms.reserve(survivors.len());
        self.arithmetic_variances.reserve(survivors.len());
        for survivor in survivors {
            self.push(
                survivor.row,
                survivor.doc,
                survivor.base,
                survivor.raw_prefix,
                survivor.estimate,
                survivor.sigma,
                survivor.residual_norm_squared,
                survivor.gamma,
                survivor.sign_query_error_term,
                survivor.arithmetic_variance,
            );
        }
    }
}

/// Drives candidate admission, the bound and boundaries without a segment, for unit tests.
#[cfg(test)]
pub(super) struct BoundaryHarness {
    pub(super) candidates: QuantizedCandidates,
    bound: Option<KthBound>,
    cluster_start: Option<usize>,
}

#[cfg(test)]
impl BoundaryHarness {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            candidates: QuantizedCandidates::with_capacity(capacity),
            bound: None,
            cluster_start: None,
        }
    }

    fn bound(&mut self, k: usize) -> &mut KthBound {
        let bound = self.bound.get_or_insert_with(|| KthBound::new(k));
        assert_eq!(bound.k, k, "one k per harness");
        bound
    }

    pub(super) fn begin_cluster(&mut self, k: usize) {
        assert!(self.cluster_start.is_none());
        self.bound(k);
        self.cluster_start = Some(self.candidates.len());
    }

    /// Appends a row to the current cluster.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push(
        &mut self,
        row: usize,
        doc: DocId,
        base: f32,
        raw_prefix: f32,
        estimate: f32,
        sigma: f32,
        residual_norm_squared: f32,
        gamma: f32,
        sign_query_error_term: f32,
    ) {
        self.candidates.push(
            row,
            doc,
            base,
            raw_prefix,
            estimate,
            sigma,
            residual_norm_squared,
            gamma,
            sign_query_error_term,
            ArithmeticError::default(),
        );
    }

    pub(super) fn finish_cluster_bound(&mut self) {
        self.finish_cluster_bound_with_kappa(QUANTIZED_BOUNDARY_KAPPA);
    }

    pub(super) fn finish_cluster_bound_with_kappa(&mut self, kappa: f32) {
        let start = self.cluster_start.take().expect("begin_cluster first");
        self.candidates.kappa = kappa;
        let bound = self
            .bound
            .as_mut()
            .expect("begin_cluster creates the bound");
        self.candidates.finish_cluster(start, bound);
    }

    pub(super) fn running_pessimistic_kth(&mut self, k: usize, kappa: f32) -> Option<Threshold> {
        assert_eq!(self.candidates.kappa, kappa);
        self.bound(k)
            .running_lower()
            .map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// `None` for a `k` above the harness's: the running top-k holds fewer than `k` entries.
    pub(super) fn running_estimate_kth(&mut self, k: usize) -> Option<f32> {
        let bound = self.bound.as_ref()?;
        if k > bound.k {
            return None;
        }
        self.bound(k).running_estimate()
    }

    /// The boundary threshold the next band would apply, leaving the running top-k intact.
    pub(super) fn pessimistic_kth(&mut self, k: usize, kappa: f32) -> Option<Threshold> {
        let mut bound = self.bound(k).clone();
        self.candidates.kappa = kappa;
        let mut lowers: Vec<f32> = (0..self.candidates.len())
            .map(|index| self.candidates.lower(index))
            .collect();
        bound
            .boundary_lower(&mut lowers)
            .map(|kth| Threshold(LowerEndpoint(kth)))
    }

    pub(super) fn band(&mut self, k: usize, kappa: f32) {
        self.candidates.kappa = kappa;
        let bound = self.bound.get_or_insert_with(|| KthBound::new(k));
        self.candidates.band(bound);
    }

    /// Rows of the running top-k, in bound order.
    pub(super) fn running_rows(&self) -> Vec<usize> {
        self.bound.as_ref().map_or_else(Vec::new, |bound| {
            bound.running.iter().map(|e| e.1).collect()
        })
    }
}

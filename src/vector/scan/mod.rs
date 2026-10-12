//! The IVF segment scan: quantized candidates, boundaries and exact scoring.

pub(in crate::vector) mod bound;
pub(in crate::vector) mod columns;
pub(in crate::vector) mod estimate;
pub(in crate::vector) mod heap;
pub(in crate::vector) mod layer_read;
#[cfg(test)]
mod quantized_boundary_tests;
#[cfg(test)]
pub(in crate::vector) mod tests;

use crate::vector::interval::Threshold;
#[cfg(test)]
use crate::vector::prepared::ArithmeticError;
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::vector::scan::bound::lower_endpoint_order;
use crate::vector::scan::columns::{QuantizedCandidate, QuantizedCandidates};
use crate::{DocId, Score};

pub(super) struct QuantizedScanCtx {
    pub(super) candidates: QuantizedCandidates,
    boundary_scratch: Vec<QuantizedCandidate>,
    /// Query-residual norms by cluster.
    cluster_query_norms: Vec<f32>,
    /// Running top lower endpoints, all evaluated with `bound_kappa`.
    bound_top: Vec<usize>,
    bound_kappa: f32,
    boundary_passed: bool,
    /// Cluster-local selection scratch.
    cluster_top: Vec<usize>,
    cluster_top_n: usize,
    cluster_start: Option<usize>,
    /// Top-k merge scratch.
    bound_merge: Vec<usize>,
    kth_scratch: Vec<usize>,
}

impl QuantizedScanCtx {
    pub(super) fn new(max_doc: DocId, candidate_capacity: usize) -> Self {
        let distinct_capacity = candidate_capacity.min(max_doc as usize);
        Self {
            candidates: QuantizedCandidates::with_capacity(candidate_capacity),
            boundary_scratch: Vec::with_capacity(candidate_capacity),
            cluster_query_norms: Vec::new(),
            bound_top: Vec::new(),
            bound_kappa: QUANTIZED_BOUNDARY_KAPPA,
            boundary_passed: false,
            cluster_top: Vec::new(),
            cluster_top_n: 0,
            cluster_start: None,
            bound_merge: Vec::new(),
            kth_scratch: Vec::with_capacity(distinct_capacity),
        }
    }

    pub(super) fn begin_cluster(&mut self, top_n: usize) {
        debug_assert!(self.cluster_start.is_none());
        debug_assert!(self.cluster_top.is_empty());
        self.cluster_top.reserve(top_n);
        self.bound_top.reserve(top_n);
        self.bound_merge.reserve(top_n.saturating_mul(2));
        self.cluster_top_n = top_n;
        self.cluster_start = Some(self.candidates.len());
    }

    /// Appends an eligible row to scan columns.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn push(
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

    pub(super) fn set_cluster_query_norm(&mut self, cluster: usize, query_norm: f32) {
        if self.cluster_query_norms.len() <= cluster {
            self.cluster_query_norms.resize(cluster + 1, f32::NAN);
        }
        self.cluster_query_norms[cluster] = query_norm;
    }

    pub(super) fn cluster_query_norm(&self, cluster: usize) -> f32 {
        let query_norm = self.cluster_query_norms[cluster];
        debug_assert!(query_norm.is_finite());
        query_norm
    }

    /// Merges one cluster into the running admission top-k.
    pub(super) fn finish_cluster_bound(&mut self) {
        self.finish_cluster_bound_with_kappa(QUANTIZED_BOUNDARY_KAPPA);
    }

    fn finish_cluster_bound_with_kappa(&mut self, kappa: f32) {
        debug_assert!(self.bound_top.is_empty() || self.bound_kappa == kappa);
        self.bound_kappa = kappa;
        let cluster_start = self
            .cluster_start
            .take()
            .expect("finish_cluster_bound requires begin_cluster");
        let top_n = std::mem::take(&mut self.cluster_top_n);
        if top_n == 0 || cluster_start >= self.candidates.len() {
            self.cluster_top.clear();
            return;
        }

        self.cluster_top.clear();
        self.cluster_top
            .extend(cluster_start..self.candidates.len());
        if self.cluster_top.len() > top_n {
            self.cluster_top
                .select_nth_unstable_by(top_n - 1, |&a, &b| {
                    lower_endpoint_order(&self.candidates, a, b, kappa)
                });
            self.cluster_top.truncate(top_n);
        }

        self.bound_merge.clear();
        self.bound_merge.extend_from_slice(&self.bound_top);
        for &index in &self.cluster_top {
            self.bound_merge.push(index);
        }
        self.cluster_top.clear();
        self.bound_merge
            .sort_unstable_by(|&a, &b| lower_endpoint_order(&self.candidates, a, b, kappa));
        self.bound_merge.truncate(top_n);
        std::mem::swap(&mut self.bound_top, &mut self.bound_merge);
    }

    pub(super) fn running_pessimistic_kth(&self, top_n: usize, kappa: f32) -> Option<Threshold> {
        debug_assert!(!self.boundary_passed);
        if top_n == 0 || self.bound_top.len() < top_n {
            return None;
        }
        debug_assert_eq!(self.bound_kappa, kappa);
        let index = *self.bound_top.last().unwrap();
        Some(Threshold(
            self.candidates
                .estimate(index)
                .lower(self.candidates.sigmas[index], kappa),
        ))
    }

    /// The lowest point estimate among the running top-k rows, `None`
    /// until `top_n` rows are in. `top_n` rows score at least this, so it
    /// never exceeds the k-th best estimate; unlike
    /// [`Self::running_pessimistic_kth`] it carries no confidence margin,
    /// so it must not drive pruning.
    pub(super) fn running_estimate_kth(&self, top_n: usize) -> Option<Score> {
        debug_assert!(!self.boundary_passed);
        if top_n == 0 || self.bound_top.len() < top_n {
            return None;
        }
        self.bound_top
            .iter()
            .map(|&index| self.candidates.estimates[index])
            .min_by(f32::total_cmp)
    }

    /// Select by lower endpoint itself; ordering by estimate can prune a true top-k row
    /// even when every confidence interval encloses its exact score.
    fn pessimistic_kth(&mut self, top_n: usize, kappa: f32) -> Option<Threshold> {
        debug_assert!(
            self.candidates
                .estimates
                .iter()
                .zip(&self.candidates.sigmas)
                .all(|(&estimate, &sigma)| estimate.is_finite() && sigma.is_finite()),
            "quantized boundary inputs must be finite"
        );
        if top_n == 0 || self.candidates.len() < top_n {
            return None;
        }
        self.kth_scratch.clear();
        self.kth_scratch.extend(0..self.candidates.len());
        let (_, selected, _) = self
            .kth_scratch
            .select_nth_unstable_by(top_n - 1, |&a, &b| {
                lower_endpoint_order(&self.candidates, a, b, kappa)
            });
        let index = *selected;
        Some(Threshold(
            self.candidates
                .estimate(index)
                .lower(self.candidates.sigmas[index], kappa),
        ))
    }

    pub(super) fn band(&mut self, top_n: usize, kappa: f32) {
        let pessimistic_kth = self.pessimistic_kth(top_n, kappa);
        self.boundary_scratch.clear();
        for index in 0..self.candidates.len() {
            if pessimistic_kth.is_none_or(|kth| {
                kth.admits(
                    self.candidates
                        .estimate(index)
                        .upper(self.candidates.sigmas[index], kappa),
                )
            }) {
                self.boundary_scratch
                    .push(self.candidates.materialize(index));
            }
        }
        self.boundary_scratch
            .sort_unstable_by_key(|candidate| candidate.row);
        self.candidates
            .replace_with_boundary_survivors(&self.boundary_scratch);
        // `bound_top` indexes the pre-boundary column layout; nothing may read the
        // running admission threshold after a boundary.
        self.bound_top.clear();
        self.boundary_passed = true;
        debug_assert!(self.cluster_start.is_none());
    }
}

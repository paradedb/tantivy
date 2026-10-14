//! Row-parallel quantized candidate columns.

use std::ops::Range;

use crate::vector::interval::Estimate;
use crate::vector::prepared::ArithmeticError;
use crate::vector::scan::layer_read::Selection;
use crate::DocId;

/// One row surviving a quantized boundary.
#[derive(Clone, Copy)]
pub(super) struct QuantizedCandidate {
    pub(super) row: usize,
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

/// Row-parallel quantized scan columns.
pub(in crate::vector) struct QuantizedCandidates {
    pub(in crate::vector) rows: Vec<usize>,
    pub(in crate::vector) docs: Vec<DocId>,
    pub(super) bases: Vec<f32>,
    pub(super) raw_prefixes: Vec<f32>,
    pub(super) estimates: Vec<f32>,
    pub(super) sigmas: Vec<f32>,
    pub(super) residual_norm_squared: Vec<f32>,
    pub(super) gammas: Vec<f32>,
    pub(super) sign_query_error_terms: Vec<f32>,
    pub(super) arithmetic_variances: Vec<ArithmeticError>,
}

impl QuantizedCandidates {
    #[cfg(test)]
    pub(in crate::vector) fn estimate_trace(&self) -> Vec<(usize, u32)> {
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
        }
    }

    #[inline]
    pub(in crate::vector) fn len(&self) -> usize {
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
    pub(in crate::vector) fn append_selected(
        &mut self,
        rows: Range<usize>,
        selection: &Selection<'_>,
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
        let len = selection.len(&rows);
        debug_assert!(doc_ids.is_none_or(|docs| docs.len() == rows.len()));
        debug_assert_eq!(bases.len(), len);
        debug_assert_eq!(raw_prefixes.len(), len);
        debug_assert_eq!(estimates.len(), len);
        debug_assert_eq!(sigmas.len(), len);
        debug_assert_eq!(residual_norms_squared.len(), len);
        debug_assert_eq!(gammas.len(), len);
        debug_assert_eq!(sign_query_error_terms.len(), len);
        debug_assert_eq!(arithmetic_variances.len(), len);
        match selection {
            Selection::All => self.rows.extend(rows),
            Selection::Rows(offsets) => self.rows.extend(offsets.iter().map(|&offset| {
                debug_assert!(offset < rows.len());
                rows.start + offset
            })),
            Selection::None => unreachable!("empty selections are skipped before scoring"),
        }
        match (selection, doc_ids) {
            (Selection::All, Some(docs)) => self.docs.extend_from_slice(docs),
            (Selection::Rows(offsets), Some(docs)) => {
                self.docs.extend(offsets.iter().map(|&row| docs[row]))
            }
            (_, None) => self.docs.resize(self.docs.len() + len, DocId::MAX),
            (Selection::None, _) => unreachable!("empty selections are skipped before scoring"),
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
    pub(super) fn estimate(&self, index: usize) -> Estimate {
        Estimate(self.estimates[index])
    }

    pub(super) fn materialize(&self, index: usize) -> QuantizedCandidate {
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

    pub(super) fn replace_with_boundary_survivors(&mut self, survivors: &[QuantizedCandidate]) {
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

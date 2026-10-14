//! Quantized candidates: their rows, doc ids and value columns, their admission into the pruning
//! bound, and boundary selection.

use std::ops::Range;

use crate::vector::quantization::{ArithmeticError, QUANTIZED_BOUNDARY_KAPPA};
use crate::vector::search::interval::{Estimate, LowerEndpoint, Threshold};
use crate::vector::search::scan::bound::{lower_endpoint_order, KthBound};
use crate::DocId;

// --- Columns ---

/// The per-candidate values every layer reads and refines.
#[derive(Default)]
pub struct Columns {
    pub(crate) bases: Vec<f32>,
    pub(crate) raw_prefixes: Vec<f32>,
    pub(crate) estimates: Vec<f32>,
    pub(crate) sigmas: Vec<f32>,
    pub(crate) residual_norm_squared: Vec<f32>,
    pub(crate) gammas: Vec<f32>,
    pub(crate) sign_query_error_terms: Vec<f32>,
    pub(crate) arithmetic_variances: Vec<ArithmeticError>,
}

impl Columns {
    pub(crate) fn len(&self) -> usize {
        self.estimates.len()
    }

    /// Appends every entry of `other`.
    pub(crate) fn extend_from(&mut self, other: &Columns) {
        self.bases.extend_from_slice(&other.bases);
        self.raw_prefixes.extend_from_slice(&other.raw_prefixes);
        self.estimates.extend_from_slice(&other.estimates);
        self.sigmas.extend_from_slice(&other.sigmas);
        self.residual_norm_squared
            .extend_from_slice(&other.residual_norm_squared);
        self.gammas.extend_from_slice(&other.gammas);
        self.sign_query_error_terms
            .extend_from_slice(&other.sign_query_error_terms);
        self.arithmetic_variances
            .extend_from_slice(&other.arithmetic_variances);
    }

    /// Overwrites `into` with the entries at `order`, in that order.
    pub(crate) fn gather(&self, order: &[usize], into: &mut Columns) {
        fn pick<T: Copy>(from: &[T], order: &[usize], into: &mut Vec<T>) {
            into.clear();
            into.extend(order.iter().map(|&index| from[index]));
        }
        pick(&self.bases, order, &mut into.bases);
        pick(&self.raw_prefixes, order, &mut into.raw_prefixes);
        pick(&self.estimates, order, &mut into.estimates);
        pick(&self.sigmas, order, &mut into.sigmas);
        pick(
            &self.residual_norm_squared,
            order,
            &mut into.residual_norm_squared,
        );
        pick(&self.gammas, order, &mut into.gammas);
        pick(
            &self.sign_query_error_terms,
            order,
            &mut into.sign_query_error_terms,
        );
        pick(
            &self.arithmetic_variances,
            order,
            &mut into.arithmetic_variances,
        );
    }
}

// --- Candidates ---

/// The live quantized candidates: global rows, doc ids and value columns, row-aligned.
pub(crate) struct QuantizedCandidates {
    pub(crate) rows: Vec<usize>,
    /// Valid for clusters admitted with given doc ids; filler otherwise.
    pub(crate) docs: Vec<DocId>,
    pub(crate) columns: Columns,
    /// Confidence multiplier of every interval endpoint.
    kappa: f32,
    /// Gather targets, swapped in after a boundary.
    spare_rows: Vec<usize>,
    spare_docs: Vec<DocId>,
    spare: Columns,
    order: Vec<usize>,
    cluster_top: Vec<(f32, usize, f32)>,
    lowers: Vec<f32>,
}

impl QuantizedCandidates {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            docs: Vec::with_capacity(capacity),
            columns: Columns::default(),
            kappa: QUANTIZED_BOUNDARY_KAPPA,
            spare_rows: Vec::new(),
            spare_docs: Vec::new(),
            spare: Columns::default(),
            order: Vec::new(),
            cluster_top: Vec::new(),
            lowers: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// `(row, estimate bits)` of every candidate, for traces.
    #[cfg(test)]
    pub(crate) fn estimate_trace(&self) -> Vec<(usize, u32)> {
        (0..self.len())
            .map(|i| (self.rows[i], self.columns.estimates[i].to_bits()))
            .collect()
    }

    /// `(row, [raw prefix, base, estimate, sigma] bits)` of every candidate, for traces.
    #[cfg(test)]
    pub(crate) fn column_trace(&self) -> Vec<(usize, [u32; 4])> {
        let columns = &self.columns;
        (0..self.len())
            .map(|i| {
                let bits = [
                    columns.raw_prefixes[i],
                    columns.bases[i],
                    columns.estimates[i],
                    columns.sigmas[i],
                ];
                (self.rows[i], bits.map(f32::to_bits))
            })
            .collect()
    }

    #[inline(always)]
    fn estimate(&self, index: usize) -> Estimate {
        Estimate(self.columns.estimates[index])
    }

    #[inline]
    fn lower(&self, index: usize) -> f32 {
        self.estimate(index)
            .lower(self.columns.sigmas[index], self.kappa)
            .0
    }

    /// Appends one cluster's selected rows, spanning `rows`, with their doc ids and `columns`.
    pub(crate) fn append(
        &mut self,
        rows: Range<usize>,
        offsets: Option<&[usize]>,
        docs: Option<&[DocId]>,
        columns: &Columns,
    ) {
        let len = offsets.map_or(rows.len(), <[usize]>::len);
        debug_assert_eq!(columns.len(), len);
        match offsets {
            None => self.rows.extend(rows),
            Some(offsets) => self
                .rows
                .extend(offsets.iter().map(|&offset| rows.start + offset)),
        }
        match docs {
            Some(docs) => {
                debug_assert_eq!(docs.len(), len);
                self.docs.extend_from_slice(docs);
            }
            // Filler: the cluster's admitted record says its doc ids come from the column.
            None => self.docs.resize(self.docs.len() + len, 0),
        }
        self.columns.extend_from(columns);
    }

    /// Merges the candidates appended since `cluster_start` into the bound's running top-k,
    /// through the cluster's own top-k so the merge stays O(k).
    pub(crate) fn finish_cluster(&mut self, cluster_start: usize, bound: &mut KthBound) {
        let k = bound.k;
        if k == 0 || cluster_start >= self.len() {
            return;
        }
        let mut cluster_top = std::mem::take(&mut self.cluster_top);
        cluster_top.clear();
        cluster_top.extend((cluster_start..self.len()).map(|index| {
            (
                self.lower(index),
                self.rows[index],
                self.columns.estimates[index],
            )
        }));
        if cluster_top.len() > k {
            cluster_top.select_nth_unstable_by(k - 1, lower_endpoint_order);
            cluster_top.truncate(k);
        }
        bound.merge_running(&cluster_top);
        self.cluster_top = cluster_top;
    }

    /// Keeps the candidates whose upper endpoint reaches the k-th lower endpoint, in row order.
    ///
    /// Selecting by lower endpoint matters: ordering by estimate can prune a true top-k row even
    /// when every interval encloses its exact score.
    pub(crate) fn band(&mut self, bound: &mut KthBound) {
        debug_assert!(
            self.columns
                .estimates
                .iter()
                .zip(&self.columns.sigmas)
                .all(|(estimate, sigma)| estimate.is_finite() && sigma.is_finite()),
            "quantized boundary inputs must be finite"
        );
        let mut lowers = std::mem::take(&mut self.lowers);
        lowers.clear();
        lowers.extend((0..self.len()).map(|index| self.lower(index)));
        let threshold = bound
            .boundary_lower(&mut lowers)
            .map(|kth| Threshold(LowerEndpoint(kth)));
        self.lowers = lowers;
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        order.extend((0..self.len()).filter(|&index| {
            threshold.is_none_or(|kth| {
                kth.admits(
                    self.estimate(index)
                        .upper(self.columns.sigmas[index], self.kappa),
                )
            })
        }));
        order.sort_unstable_by_key(|&index| self.rows[index]);
        self.keep(&order);
        self.order = order;
    }

    /// Removes the candidates at `ranges`, which ascend and do not overlap, keeping row order.
    pub(crate) fn retire(&mut self, ranges: &[Range<usize>]) {
        debug_assert!(ranges.windows(2).all(|pair| pair[0].end <= pair[1].start));
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        let mut next = 0;
        for range in ranges {
            order.extend(next..range.start);
            next = range.end;
        }
        order.extend(next..self.len());
        self.keep(&order);
        self.order = order;
    }

    /// Keeps the candidates at `order`, in that order.
    fn keep(&mut self, order: &[usize]) {
        self.spare_rows.clear();
        self.spare_rows
            .extend(order.iter().map(|&index| self.rows[index]));
        self.spare_docs.clear();
        self.spare_docs
            .extend(order.iter().map(|&index| self.docs[index]));
        self.columns.gather(order, &mut self.spare);
        std::mem::swap(&mut self.rows, &mut self.spare_rows);
        std::mem::swap(&mut self.docs, &mut self.spare_docs);
        std::mem::swap(&mut self.columns, &mut self.spare);
    }
}

// --- Test harness ---

/// Drives candidate admission, the bound and boundaries without a segment.
#[cfg(test)]
pub(crate) struct BoundaryHarness {
    pub(crate) candidates: QuantizedCandidates,
    bound: Option<KthBound>,
    cluster_start: Option<usize>,
}

#[cfg(test)]
impl BoundaryHarness {
    pub(crate) fn new(capacity: usize) -> Self {
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

    pub(crate) fn begin_cluster(&mut self, k: usize) {
        assert!(self.cluster_start.is_none());
        self.bound(k);
        self.cluster_start = Some(self.candidates.len());
    }

    /// Appends a row with `estimate` and `sigma` to the current cluster.
    pub(crate) fn push(&mut self, row: usize, doc: DocId, estimate: f32, sigma: f32) {
        let one = Columns {
            bases: vec![0.0],
            raw_prefixes: vec![estimate],
            estimates: vec![estimate],
            sigmas: vec![sigma],
            residual_norm_squared: vec![1.0],
            gammas: vec![1.0],
            sign_query_error_terms: vec![0.0],
            arithmetic_variances: vec![ArithmeticError::default()],
        };
        self.candidates.rows.push(row);
        self.candidates.docs.push(doc);
        self.candidates.columns.extend_from(&one);
    }

    /// Records an exactly scored row in the bound.
    pub(crate) fn push_exact(&mut self, score: f32, k: usize) {
        self.bound(k).push_exact(score);
    }

    pub(crate) fn finish_cluster(&mut self, kappa: f32) {
        let start = self.cluster_start.take().expect("begin_cluster first");
        self.candidates.kappa = kappa;
        let bound = self
            .bound
            .as_mut()
            .expect("begin_cluster creates the bound");
        self.candidates.finish_cluster(start, bound);
    }

    pub(crate) fn running_lower(&mut self, k: usize) -> Option<Threshold> {
        self.bound(k)
            .running_lower()
            .map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// `None` for a `k` above the harness's: the running top-k holds fewer than `k` entries.
    pub(crate) fn running_estimate(&mut self, k: usize) -> Option<f32> {
        let bound = self.bound.as_ref()?;
        if k > bound.k {
            return None;
        }
        self.bound(k).running_estimate()
    }

    /// The threshold the next boundary would apply, leaving the running top-k intact.
    pub(crate) fn boundary_lower(&mut self, k: usize, kappa: f32) -> Option<Threshold> {
        let mut bound = self.bound(k).clone();
        self.candidates.kappa = kappa;
        let mut lowers: Vec<f32> = (0..self.candidates.len())
            .map(|index| self.candidates.lower(index))
            .collect();
        bound
            .boundary_lower(&mut lowers)
            .map(|kth| Threshold(LowerEndpoint(kth)))
    }

    pub(crate) fn band(&mut self, k: usize, kappa: f32) {
        self.candidates.kappa = kappa;
        let bound = self.bound.get_or_insert_with(|| KthBound::new(k));
        self.candidates.band(bound);
    }

    /// Rows of the running top-k, in bound order.
    pub(crate) fn running_rows(&self) -> Vec<usize> {
        self.bound.as_ref().map_or_else(Vec::new, |bound| {
            bound.running.iter().map(|entry| entry.1).collect()
        })
    }
}

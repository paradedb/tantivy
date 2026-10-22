//! Quantized candidates: one block per admitted cluster, their value columns and interval
//! endpoints, and their selection in place at a threshold.

use std::ops::Range;

use crate::vector::distance::Similarity;
use crate::vector::quantization::{ArithmeticError, QUANTIZED_BOUNDARY_KAPPA};
use crate::vector::search::cluster_source::ClusterRows;
#[cfg(test)]
use crate::vector::search::interval::LowerEndpoint;
use crate::vector::search::interval::{Estimate, Threshold, UpperEndpoint};
#[cfg(test)]
use crate::vector::search::scan::bound::PruningBound;
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
    /// Columns with room for `capacity` entries each.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
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

    /// Copies every value of entry `from` over entry `to`.
    fn move_entry(&mut self, from: usize, to: usize) {
        self.bases[to] = self.bases[from];
        self.raw_prefixes[to] = self.raw_prefixes[from];
        self.estimates[to] = self.estimates[from];
        self.sigmas[to] = self.sigmas[from];
        self.residual_norm_squared[to] = self.residual_norm_squared[from];
        self.gammas[to] = self.gammas[from];
        self.sign_query_error_terms[to] = self.sign_query_error_terms[from];
        self.arithmetic_variances[to] = self.arithmetic_variances[from];
    }

    /// Keeps the first `len` entries.
    fn truncate(&mut self, len: usize) {
        self.bases.truncate(len);
        self.raw_prefixes.truncate(len);
        self.estimates.truncate(len);
        self.sigmas.truncate(len);
        self.residual_norm_squared.truncate(len);
        self.gammas.truncate(len);
        self.sign_query_error_terms.truncate(len);
        self.arithmetic_variances.truncate(len);
    }

    /// Appends the entries of `from` that `keep` selects, column by column.
    fn extend_selected(&mut self, from: &Columns, keep: &[u32]) {
        self.bases
            .extend(keep.iter().map(|&i| from.bases[i as usize]));
        self.raw_prefixes
            .extend(keep.iter().map(|&i| from.raw_prefixes[i as usize]));
        self.estimates
            .extend(keep.iter().map(|&i| from.estimates[i as usize]));
        self.sigmas
            .extend(keep.iter().map(|&i| from.sigmas[i as usize]));
        self.residual_norm_squared
            .extend(keep.iter().map(|&i| from.residual_norm_squared[i as usize]));
        self.gammas
            .extend(keep.iter().map(|&i| from.gammas[i as usize]));
        self.sign_query_error_terms.extend(
            keep.iter()
                .map(|&i| from.sign_query_error_terms[i as usize]),
        );
        self.arithmetic_variances
            .extend(keep.iter().map(|&i| from.arithmetic_variances[i as usize]));
    }
}

// --- Candidates ---

/// One admitted cluster's candidates: a contiguous block, in row order.
#[derive(Clone, Debug)]
pub(crate) struct AdmittedCluster {
    pub(crate) cluster: usize,
    /// The cluster's global row range.
    pub(crate) rows: Range<usize>,
    pub(crate) centroid: Similarity,
    /// The block's candidates; empty once retired or pruned away.
    pub(crate) at: Range<usize>,
    /// Where the block's doc ids start in `docs`, when the source supplied them.
    pub(crate) docs: Option<usize>,
}

/// The quantized candidates: one block per admitted cluster, compacted in place at each
/// boundary.
pub(crate) struct QuantizedCandidates {
    /// Cluster-local offsets, ascending within each block.
    offsets: Vec<u32>,
    /// Doc ids of the blocks admitted with them.
    docs: Vec<DocId>,
    /// Estimator state, aligned with `offsets`.
    pub(crate) columns: Columns,
    /// In admission order, which is storage order; compaction walks this.
    blocks: Vec<AdmittedCluster>,
    /// Confidence multiplier of every interval endpoint.
    kappa: f32,
    /// The live candidates' lower endpoints, written for a boundary.
    lowers: Vec<f32>,
}

impl QuantizedCandidates {
    /// Room for `capacity` candidates in the offsets, every column and the boundary's lower
    /// endpoints; doc ids are reserved when a block first brings them.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            offsets: Vec::with_capacity(capacity),
            docs: Vec::new(),
            columns: Columns::with_capacity(capacity),
            blocks: Vec::new(),
            kappa: QUANTIZED_BOUNDARY_KAPPA,
            lowers: Vec::with_capacity(capacity),
        }
    }

    /// The confidence multiplier of every interval endpoint.
    pub(crate) fn kappa(&self) -> f32 {
        self.kappa
    }

    /// The live candidates: every block's range, summed.
    pub(crate) fn len(&self) -> usize {
        self.blocks.iter().map(|block| block.at.len()).sum()
    }

    /// Appends `rows` as a block around `centroid`, with its doc ids when the source supplied
    /// them; returns the block's range, whose columns the caller writes.
    pub(crate) fn push_block(
        &mut self,
        rows: &ClusterRows<'_>,
        centroid: Similarity,
    ) -> Range<usize> {
        let start = self.offsets.len();
        match rows.offsets {
            Some(offsets) => self.offsets.extend_from_slice(offsets),
            None => self.offsets.extend((0..).take(rows.rows.len())),
        }
        let at = start..self.offsets.len();
        let docs = rows.docs.map(|docs| {
            debug_assert_eq!(docs.len(), at.len());
            if self.docs.capacity() == 0 {
                self.docs.reserve(self.offsets.capacity());
            }
            let start = self.docs.len();
            self.docs.extend_from_slice(docs);
            start
        });
        self.blocks.push(AdmittedCluster {
            cluster: rows.cluster,
            rows: rows.rows.clone(),
            centroid,
            at: at.clone(),
            docs,
        });
        at
    }

    /// Appends the rows of `rows` that `keep` selects as a block around `centroid`, taking their
    /// combined columns from `scratch`. Returns the block's range.
    pub(crate) fn push_selected(
        &mut self,
        rows: &ClusterRows<'_>,
        centroid: Similarity,
        scratch: &Columns,
        keep: &[u32],
    ) -> Range<usize> {
        debug_assert_eq!(scratch.estimates.len(), rows.len());
        let start = self.offsets.len();
        match rows.offsets {
            Some(offsets) => self
                .offsets
                .extend(keep.iter().map(|&i| offsets[i as usize])),
            // A whole-cluster selection's offsets are the row indices themselves.
            None => self.offsets.extend_from_slice(keep),
        }
        let at = start..self.offsets.len();
        let docs = rows.docs.map(|docs| {
            debug_assert_eq!(docs.len(), rows.len());
            if self.docs.capacity() == 0 {
                self.docs.reserve(self.offsets.capacity());
            }
            let start = self.docs.len();
            self.docs.extend(keep.iter().map(|&i| docs[i as usize]));
            start
        });
        self.columns.extend_selected(scratch, keep);
        self.blocks.push(AdmittedCluster {
            cluster: rows.cluster,
            rows: rows.rows.clone(),
            centroid,
            at: at.clone(),
            docs,
        });
        at
    }

    /// Block `b`.
    pub(crate) fn block(&self, b: usize) -> &AdmittedCluster {
        &self.blocks[b]
    }

    /// A view of block `b`: its offsets and doc ids borrowed, with its centroid score.
    pub(crate) fn rows(&self, b: usize) -> ClusterRows<'_> {
        let block = &self.blocks[b];
        ClusterRows {
            cluster: block.cluster,
            rows: block.rows.clone(),
            offsets: Some(&self.offsets[block.at.clone()]),
            docs: block
                .docs
                .map(|start| &self.docs[start..start + block.at.len()]),
            centroid: Some(block.centroid),
        }
    }

    /// Drops, in place, block `b`'s candidates whose upper endpoint is below `threshold`, keeping
    /// row order and the doc ids aligned; the block's range shrinks and the entries past it stay
    /// dead until the next boundary compacts them. Returns the rows dropped.
    pub(crate) fn recheck(&mut self, b: usize, threshold: Threshold) -> usize {
        let (at, docs) = (self.blocks[b].at.clone(), self.blocks[b].docs);
        let mut write = at.start;
        for read in at.clone() {
            if !threshold.admits(self.upper(read)) {
                continue;
            }
            if write != read {
                self.offsets[write] = self.offsets[read];
                self.columns.move_entry(read, write);
                if let Some(docs) = docs {
                    self.docs[docs + (write - at.start)] = self.docs[docs + (read - at.start)];
                }
            }
            write += 1;
        }
        self.blocks[b].at.end = write;
        at.end - write
    }

    /// Empties block `b`'s range; its candidates go at the next boundary.
    pub(crate) fn retire(&mut self, b: usize) {
        let at = &mut self.blocks[b].at;
        at.end = at.start;
    }

    /// Writes the block indices, sorted by cluster id, to `out`. Blocks are never removed, only
    /// emptied, so the indices stay valid.
    pub(crate) fn order_by_cluster(&self, out: &mut Vec<usize>) {
        out.clear();
        out.extend(0..self.blocks.len());
        out.sort_unstable_by_key(|&b| self.blocks[b].cluster);
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

    #[inline]
    fn upper(&self, index: usize) -> UpperEndpoint {
        self.estimate(index)
            .upper(self.columns.sigmas[index], self.kappa)
    }

    /// The `(lower, global row, estimate)` entries of the last block's candidates, at `at`.
    pub(crate) fn entries(&self, at: Range<usize>) -> impl Iterator<Item = (f32, usize, f32)> + '_ {
        let block = self.blocks.last().expect("a block was just pushed");
        debug_assert_eq!(block.at, at, "the block just pushed");
        let start = block.rows.start;
        at.map(move |index| {
            (
                self.lower(index),
                start + self.offsets[index] as usize,
                self.columns.estimates[index],
            )
        })
    }

    /// Keeps, in place, each block's candidates whose upper endpoint reaches `threshold`, a
    /// boundary's; selecting by lower endpoint, not estimate, keeps every true top-k row whose
    /// interval encloses its score. Blocks keep their row order and a retired block keeps nothing.
    /// Blocks are walked in admission order, which is storage order, so writes never pass reads.
    pub(crate) fn band(&mut self, threshold: Option<Threshold>) {
        let (mut write, mut docs_write) = (0, 0);
        for b in 0..self.blocks.len() {
            let (at, docs) = (self.blocks[b].at.clone(), self.blocks[b].docs);
            let (start, docs_start) = (write, docs_write);
            for read in at.clone() {
                if threshold.is_none_or(|kth| kth.admits(self.upper(read))) {
                    self.offsets[write] = self.offsets[read];
                    self.columns.move_entry(read, write);
                    if let Some(docs) = docs {
                        self.docs[docs_write] = self.docs[docs + (read - at.start)];
                        docs_write += 1;
                    }
                    write += 1;
                }
            }
            self.blocks[b].at = start..write;
            self.blocks[b].docs = docs.map(|_| docs_start);
        }
        self.offsets.truncate(write);
        self.columns.truncate(write);
        self.docs.truncate(docs_write);
    }

    /// The lower endpoint of every live candidate, in a buffer the caller may reorder.
    pub(crate) fn live_lowers(&mut self) -> &mut [f32] {
        let mut lowers = std::mem::take(&mut self.lowers);
        lowers.clear();
        for block in &self.blocks {
            debug_assert!(
                block
                    .at
                    .clone()
                    .all(|index| self.columns.estimates[index].is_finite()
                        && self.columns.sigmas[index].is_finite()),
                "quantized boundary inputs must be finite"
            );
            lowers.extend(block.at.clone().map(|index| self.lower(index)));
        }
        self.lowers = lowers;
        &mut self.lowers
    }
}

// --- Traces ---

#[cfg(test)]
impl QuantizedCandidates {
    /// `(global row, position)` of every live candidate, block by block: in admission order, or
    /// by cluster id, which is row order.
    fn live(&self, row_order: bool) -> Vec<(usize, usize)> {
        let mut order: Vec<usize> = (0..self.blocks.len()).collect();
        if row_order {
            self.order_by_cluster(&mut order);
        }
        order
            .iter()
            .flat_map(|&b| {
                let block = &self.blocks[b];
                block
                    .at
                    .clone()
                    .map(move |index| (block.rows.start + self.offsets[index] as usize, index))
            })
            .collect()
    }

    /// The live candidates' global rows, in row order.
    pub(crate) fn global_rows(&self) -> Vec<usize> {
        self.live(true).into_iter().map(|(row, _)| row).collect()
    }

    /// The live candidates' doc ids, in row order, for the blocks admitted with them.
    pub(crate) fn doc_ids(&self) -> Vec<DocId> {
        let mut order = Vec::new();
        self.order_by_cluster(&mut order);
        order
            .iter()
            .filter_map(|&b| {
                let block = &self.blocks[b];
                block
                    .docs
                    .map(|start| &self.docs[start..start + block.at.len()])
            })
            .flatten()
            .copied()
            .collect()
    }

    /// `(row, estimate bits)` of every live candidate, in admission or row order.
    pub(crate) fn estimate_trace(&self, row_order: bool) -> Vec<(usize, u32)> {
        self.live(row_order)
            .into_iter()
            .map(|(row, index)| (row, self.columns.estimates[index].to_bits()))
            .collect()
    }

    /// `(row, [raw prefix, base, estimate, sigma] bits)` of every live candidate, in admission or
    /// row order.
    pub(crate) fn column_trace(&self, row_order: bool) -> Vec<(usize, [u32; 4])> {
        let columns = &self.columns;
        self.live(row_order)
            .into_iter()
            .map(|(row, index)| {
                let bits = [
                    columns.raw_prefixes[index],
                    columns.bases[index],
                    columns.estimates[index],
                    columns.sigmas[index],
                ];
                (row, bits.map(f32::to_bits))
            })
            .collect()
    }

    /// Opens an empty test block whose offsets are global rows. Its cluster id is its index, so
    /// cluster order is admission order.
    fn open_test_block(&mut self) {
        let start = self.offsets.len();
        self.blocks.push(AdmittedCluster {
            cluster: self.blocks.len(),
            rows: 0..0,
            centroid: Similarity::new(0.0),
            at: start..start,
            docs: Some(self.docs.len()),
        });
    }

    /// Appends `row` with `doc`, `estimate` and `sigma` to the last test block.
    fn push_test_row(&mut self, row: usize, doc: DocId, estimate: f32, sigma: f32) {
        self.offsets.push(row as u32);
        self.docs.push(doc);
        let columns = &mut self.columns;
        columns.bases.push(0.0);
        columns.raw_prefixes.push(estimate);
        columns.estimates.push(estimate);
        columns.sigmas.push(sigma);
        columns.residual_norm_squared.push(1.0);
        columns.gammas.push(1.0);
        columns.sign_query_error_terms.push(0.0);
        columns
            .arithmetic_variances
            .push(ArithmeticError::default());
        if let Some(block) = self.blocks.last_mut() {
            block.at.end += 1;
        }
    }
}

// --- Test harness ---

/// Drives candidate admission, the bound and boundaries without a segment.
#[cfg(test)]
pub(crate) struct BoundaryHarness {
    pub(crate) candidates: QuantizedCandidates,
    bound: Option<PruningBound>,
    /// Whether the last block is a cluster `begin_cluster` opened and `finish_cluster` has not
    /// closed yet.
    cluster_open: bool,
    /// Whether the last block takes more rows: an open cluster, or rows pushed outside one.
    block_open: bool,
}

#[cfg(test)]
impl BoundaryHarness {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            candidates: QuantizedCandidates::with_capacity(capacity),
            bound: None,
            cluster_open: false,
            block_open: false,
        }
    }

    fn bound(&mut self, k: usize) -> &mut PruningBound {
        let bound = self.bound.get_or_insert_with(|| PruningBound::new(k, None));
        assert_eq!(bound.k_for_test(), k, "one k per harness");
        bound
    }

    pub(crate) fn begin_cluster(&mut self, k: usize) {
        assert!(!self.cluster_open);
        self.bound(k);
        self.candidates.open_test_block();
        self.cluster_open = true;
        self.block_open = true;
    }

    /// Appends a row with `estimate` and `sigma` to the current cluster.
    pub(crate) fn push(&mut self, row: usize, doc: DocId, estimate: f32, sigma: f32) {
        if !self.block_open {
            self.candidates.open_test_block();
            self.block_open = true;
        }
        self.candidates.push_test_row(row, doc, estimate, sigma);
    }

    /// Records an exactly scored row in the bound.
    pub(crate) fn push_exact(&mut self, score: f32, k: usize) {
        self.bound(k).push_exact(score);
    }

    pub(crate) fn finish_cluster(&mut self, kappa: f32) {
        assert!(self.cluster_open, "begin_cluster first");
        self.cluster_open = false;
        self.block_open = false;
        self.candidates.kappa = kappa;
        let at = self
            .candidates
            .blocks
            .last()
            .map_or(0..0, |block| block.at.clone());
        let bound = self
            .bound
            .as_mut()
            .expect("begin_cluster creates the bound");
        bound.merge_cluster(self.candidates.entries(at));
    }

    pub(crate) fn own_lower(&mut self, k: usize) -> Option<Threshold> {
        self.bound(k)
            .own_lower()
            .map(|kth| Threshold(LowerEndpoint(kth)))
    }

    /// `None` for a `k` above the harness's: the running top-k holds fewer than `k` entries.
    pub(crate) fn own_estimate(&mut self, k: usize) -> Option<f32> {
        let bound = self.bound.as_ref()?;
        if k > bound.k_for_test() {
            return None;
        }
        self.bound(k).own_estimate()
    }

    /// The threshold the next boundary would apply, leaving the running top-k intact.
    pub(crate) fn boundary_lower(&mut self, k: usize, kappa: f32) -> Option<Threshold> {
        let mut bound = self.bound(k).clone();
        self.candidates.kappa = kappa;
        bound.boundary(&mut self.candidates)
    }

    pub(crate) fn band(&mut self, k: usize, kappa: f32) {
        self.candidates.kappa = kappa;
        self.block_open = false;
        let bound = self.bound.get_or_insert_with(|| PruningBound::new(k, None));
        let threshold = bound.boundary(&mut self.candidates);
        self.candidates.band(threshold);
    }

    /// The bound's pruning threshold.
    pub(crate) fn threshold(&mut self, k: usize) -> Option<f32> {
        self.bound(k).threshold_for_test()
    }

    /// Raises the bound's floor.
    pub(crate) fn raise_floor(&mut self, k: usize, value: f32) {
        self.bound(k).raise_floor_for_test(value);
    }

    /// Opens an empty block that takes the next rows pushed, outside any cluster.
    pub(crate) fn open_block(&mut self) {
        assert!(!self.cluster_open);
        self.candidates.open_test_block();
        self.block_open = true;
    }

    /// Each block's live candidate count, in admission order.
    pub(crate) fn block_lens(&self) -> Vec<usize> {
        self.candidates
            .blocks
            .iter()
            .map(|block| block.at.len())
            .collect()
    }

    /// Rechecks block `b` against `threshold`; returns the rows dropped.
    pub(crate) fn recheck(&mut self, b: usize, threshold: f32) -> usize {
        self.block_open = false;
        self.candidates
            .recheck(b, Threshold(LowerEndpoint(threshold)))
    }

    /// Rows of the running top-k, in bound order.
    pub(crate) fn running_rows(&self) -> Vec<usize> {
        self.bound.as_ref().map_or_else(Vec::new, |bound| {
            bound
                .running_for_test()
                .iter()
                .map(|entry| entry.1)
                .collect()
        })
    }
}

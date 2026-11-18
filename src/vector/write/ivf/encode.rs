//! The encode step: the stored centroids, and a quantized field's rows encoded while the assign
//! pass still holds them decoded ([`RowEncoder`]), into the records its Data entry's blocks are
//! written from.

use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use quant::cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, EncodedBatch, LayerSpec,
    PreparedCentroidWorkspace, QueryRotationPlan,
};
use quant::model::Grid;
use rayon::prelude::*;

use super::records::{MergedRecords, RecordHeader, RecordLayout, RecordRuns, RECORD_SORT_BYTES};
use super::FieldMerge;
use crate::indexer::segment_updater::CancelSentinel;
use crate::schema::VectorOptions;
use crate::vector::clustering::{decode_row_append, encode_vector, AssignedBatch, IvfCentroids};
use crate::vector::distance::{maybe_normalize_bytes, normalize_f32_inplace, NormalizeOutcome};
use crate::vector::storage::metadata::{Slot, SlotType, VectorColMetadata};
use crate::vector::write::RowAddress;
use crate::vector::{VectorElement, VectorQuantizationConfig};
use crate::{DocId, TantivyError};

/// The stored centroid rows, normalized as the field's vectors are.
pub(super) fn encode_centroids(
    merge: &FieldMerge,
    centroids: &IvfCentroids,
    num_centroids: usize,
) -> crate::Result<Vec<u8>> {
    let opts = merge.opts;
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let centroid_rows: Vec<Vec<f32>> = centroid_matrix
        .values
        .chunks_exact(opts.dim())
        .map(|centroid| centroid.to_vec())
        .collect();
    let mut centroid_bytes = Vec::with_capacity(num_centroids * opts.bytes_per_vector());
    for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
        let mut bytes = encode_vector(centroid, opts.dim())?;
        let outcome = maybe_normalize_bytes(opts, &mut bytes);
        if outcome == NormalizeOutcome::NonFinite {
            log::warn!(
                "non-finite centroid {centroid_ord} in field '{}' written un-normalized during \
                 merge",
                merge.name,
            );
        }
        centroid_bytes.extend_from_slice(&bytes);
    }
    Ok(centroid_bytes)
}

/// Bytes the quantized encode holds at once: decoded row copies being encoded, and encoded records
/// not yet spilled, half each.
const GROUP_DECODED_BYTES: usize = 256 << 20;

/// Test override of [`GROUP_DECODED_BYTES`]; 0 keeps the constant. The split into chunks and runs
/// never changes a written byte, so a test may set it while other tests merge.
#[cfg(test)]
pub(super) static GROUP_DECODED_BYTES_FOR_TESTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn group_decoded_bytes() -> usize {
    #[cfg(test)]
    {
        let bytes = GROUP_DECODED_BYTES_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed);
        if bytes > 0 {
            return bytes;
        }
    }
    GROUP_DECODED_BYTES
}

/// Rows one encode tile holds: about 1 MiB of decoded rows and encode scratch.
pub(super) fn encode_tile_rows(
    opts: &VectorOptions,
    quantization: Option<&VectorQuantizationConfig>,
) -> usize {
    let row_bytes = opts.bytes_per_vector();
    let fixed_scratch = row_bytes + opts.dim() + opts.dim().div_ceil(64) * 8;
    let per_row_scratch = 2 * row_bytes + quantization.map_or(0, |c| c.bytes_per_row()) + 16;
    (1usize << 20)
        .saturating_sub(fixed_scratch)
        .checked_div(per_row_scratch)
        .unwrap_or(0)
        .max(1)
}

// --- Row encoder ---

/// Encodes a quantized field's rows as the assign pass hands them over: each row against each of
/// its clusters, into records the Data entry's blocks are written from, so no source row is read
/// again.
///
/// Each batch is walked in chunks on the calling thread; a chunk's row copies are grouped by
/// cluster and encoded on the pool, one prepared centroid per (chunk, cluster) group. A chunk's
/// copies and the records not yet spilled stay within [`GROUP_DECODED_BYTES`].
pub(super) struct RowEncoder<'a> {
    spec: EncodeSpec<'a>,
    /// Most row copies one chunk holds decoded.
    chunk_copies: usize,
    runs: RecordRuns<'a>,
    cancel: &'a dyn CancelSentinel,
    copies: Vec<RowCopy>,
    values: Vec<f32>,
    /// Documents with a primary row.
    pub(super) primaries: usize,
    /// Rows, replicas included.
    pub(super) total: usize,
    /// Copying, normalizing and encoding rows.
    pub(super) encode: Duration,
    /// Spilling runs of records.
    pub(super) spill: Duration,
}

/// What every row encode of a field shares.
struct EncodeSpec<'a> {
    centroid_bytes: &'a [u8],
    dim: usize,
    plan: Arc<QueryRotationPlan>,
    specs: Vec<LayerSpec>,
    grids: Vec<Grid>,
    slots: Vec<Slot>,
    layout: RecordLayout,
    tile_rows: usize,
    scores_by_distance: bool,
    normalize: bool,
}

/// One row copy of an assigned batch: the batch row, encoded against one of its clusters.
#[derive(Clone, Copy)]
struct RowCopy {
    cluster: u32,
    /// The row's position within its batch.
    row: u32,
    primary: bool,
}

impl<'a> RowEncoder<'a> {
    pub(super) fn new(merge: &FieldMerge<'a>, centroid_bytes: &'a [u8]) -> crate::Result<Self> {
        let (opts, quantization) = (merge.opts, merge.quantization);
        debug_assert!(quantization.is_some(), "plain rows are read, not encoded");
        let meta = VectorColMetadata::build_ivf(opts, quantization)?;
        let slots = meta.slots();
        let (specs, grids) = meta.runtime();
        let layout = RecordLayout::new(&slots);
        let budget = group_decoded_bytes();
        let copy_budget = budget / 2;
        let record_capacity = (budget - copy_budget) / (layout.len + RECORD_SORT_BYTES);
        let chunk_copies = (copy_budget / opts.bytes_per_vector())
            .min(record_capacity)
            .max(1);
        let runs = RecordRuns::new(
            merge.ctx.target_segment.index().directory(),
            layout.len,
            record_capacity,
        );
        Ok(RowEncoder {
            spec: EncodeSpec {
                centroid_bytes,
                dim: opts.dim(),
                plan: Arc::new(QueryRotationPlan::new(opts.dim(), &specs)),
                specs,
                grids,
                slots,
                layout,
                tile_rows: encode_tile_rows(opts, quantization),
                scores_by_distance: opts.metric().scores_by_distance(),
                normalize: opts.needs_normalization(),
            },
            chunk_copies,
            runs,
            cancel: merge.ctx.cancel,
            copies: Vec::new(),
            values: Vec::new(),
            primaries: 0,
            total: 0,
            encode: Duration::ZERO,
            spill: Duration::ZERO,
        })
    }

    /// Encodes one assigned batch whose rows have the given source addresses and target
    /// documents, in chunks of at most `chunk_copies` row copies; a row is never split.
    pub(super) fn accept(
        &mut self,
        batch: &AssignedBatch,
        docs: &[(RowAddress, DocId)],
    ) -> crate::Result<()> {
        let rows = batch.rows.len();
        let mut start = 0;
        while start < rows {
            let mut end = start;
            let mut copies = 0;
            while end < rows {
                let row_copies = 1 + usize::from(batch.pairs[end].1.is_some());
                if copies > 0 && copies + row_copies > self.chunk_copies {
                    break;
                }
                copies += row_copies;
                end += 1;
            }
            self.encode_chunk(batch, docs, start..end)?;
            start = end;
        }
        Ok(())
    }

    /// Every encoded record in (cluster, target doc) order.
    pub(super) fn finish(self) -> std::io::Result<MergedRecords> {
        self.runs.into_merged()
    }

    /// Encodes the batch rows `range`: one copy per (row, cluster), grouped by cluster.
    fn encode_chunk(
        &mut self,
        batch: &AssignedBatch,
        docs: &[(RowAddress, DocId)],
        range: Range<usize>,
    ) -> crate::Result<()> {
        if self.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        self.copies.clear();
        for row in range.clone() {
            let (primary, secondary) = batch.pairs[row];
            let row = row as u32;
            self.copies.push(RowCopy {
                cluster: primary,
                row,
                primary: true,
            });
            if let Some(secondary) = secondary {
                self.copies.push(RowCopy {
                    cluster: secondary,
                    row,
                    primary: false,
                });
            }
        }
        self.copies
            .sort_unstable_by_key(|copy| (copy.cluster, copy.row));
        self.primaries += range.len();
        self.total += self.copies.len();

        let spill_start = Instant::now();
        let records = self.runs.append(self.copies.len())?;
        self.spill += spill_start.elapsed();

        let encode_start = Instant::now();
        let spec = &self.spec;
        self.values.clear();
        self.values.resize(self.copies.len() * spec.dim, 0.0);
        let groups = cluster_groups(&self.copies, &mut self.values, records, spec);
        let vectors = &batch.vectors.values[..];
        groups.into_par_iter().try_for_each_init(
            || EncodeWorkspace {
                centroid: Vec::with_capacity(spec.dim),
                prepared: PreparedCentroidWorkspace::new(Arc::clone(&spec.plan)),
                batch: BatchEncodeWorkspace::with_capacity(spec.dim, spec.tile_rows, &spec.specs),
            },
            |workspace, group| group.encode(spec, vectors, docs, workspace),
        )?;
        self.encode += encode_start.elapsed();
        Ok(())
    }
}

/// Splits a chunk's copies, sorted by cluster, with their decoded rows and their records, into one
/// group per cluster.
fn cluster_groups<'c>(
    mut copies: &'c [RowCopy],
    mut values: &'c mut [f32],
    mut records: &'c mut [u8],
    spec: &EncodeSpec,
) -> Vec<ClusterCopies<'c>> {
    let mut groups = Vec::new();
    while let Some(first) = copies.first() {
        let len = copies.partition_point(|copy| copy.cluster == first.cluster);
        let (group_copies, rest_copies) = copies.split_at(len);
        let (group_values, rest_values) = std::mem::take(&mut values).split_at_mut(len * spec.dim);
        let (group_records, rest_records) =
            std::mem::take(&mut records).split_at_mut(len * spec.layout.len);
        groups.push(ClusterCopies {
            cluster: first.cluster,
            copies: group_copies,
            values: group_values,
            records: group_records,
        });
        (copies, values, records) = (rest_copies, rest_values, rest_records);
    }
    groups
}

/// One pool thread's encode state.
struct EncodeWorkspace {
    centroid: Vec<f32>,
    prepared: PreparedCentroidWorkspace,
    batch: BatchEncodeWorkspace,
}

/// One cluster's copies within a chunk, with their decoded rows and their records.
struct ClusterCopies<'c> {
    cluster: u32,
    copies: &'c [RowCopy],
    values: &'c mut [f32],
    records: &'c mut [u8],
}

impl ClusterCopies<'_> {
    /// Copies each row, normalized as the field's vectors are, writes each record's header, then
    /// encodes the rows in tiles of `tile_rows` against the cluster's prepared centroid and fills
    /// each record's slot bytes. A row's codes do not depend on the rows that share its tile.
    fn encode(
        self,
        spec: &EncodeSpec,
        vectors: &[f32],
        docs: &[(RowAddress, DocId)],
        workspace: &mut EncodeWorkspace,
    ) -> crate::Result<()> {
        let dim = spec.dim;
        let stride = dim * f32::SIZE_BYTES;
        let record_len = spec.layout.len;
        workspace.centroid.clear();
        decode_row_append::<f32>(
            &spec.centroid_bytes[self.cluster as usize * stride..][..stride],
            dim,
            &mut workspace.centroid,
        )?;
        let prepared = workspace.prepared.prepare(&workspace.centroid);
        for ((copy, value), record) in self
            .copies
            .iter()
            .zip(self.values.chunks_exact_mut(dim))
            .zip(self.records.chunks_exact_mut(record_len))
        {
            let row = copy.row as usize;
            value.copy_from_slice(&vectors[row * dim..(row + 1) * dim]);
            if spec.normalize {
                normalize_f32_inplace(value);
            }
            RecordHeader {
                cluster: copy.cluster,
                doc: docs[row].1,
                primary: copy.primary,
            }
            .write(record);
        }
        for (tile, tile_records) in self
            .values
            .chunks_mut(spec.tile_rows * dim)
            .zip(self.records.chunks_mut(spec.tile_rows * record_len))
        {
            let rows = tile.len() / dim;
            let batch = encode_batch_in_place_with_workspace(
                tile,
                rows,
                prepared,
                &spec.specs,
                &spec.grids,
                &mut workspace.batch,
                spec.scores_by_distance,
            );
            debug_assert!(
                batch.residual_norms_squared.iter().all(|n| n.is_finite())
                    && batch
                        .layers
                        .iter()
                        .all(|layer| layer.scales.iter().all(|s| s.is_finite())),
                "a row outside the admitted range reached encode: non-finite residual norm or \
                 scale"
            );
            write_encoded_fields(tile_records, spec, batch);
        }
        Ok(())
    }
}

/// Fills each record's slot bytes from its row of an encoded tile: the bytes the row contributes
/// to each column of its block.
fn write_encoded_fields(records: &mut [u8], spec: &EncodeSpec, batch: &EncodedBatch) {
    for (row, record) in records.chunks_exact_mut(spec.layout.len).enumerate() {
        for &(slot, offset, stride) in &spec.layout.fields {
            let field = &mut record[offset..offset + stride];
            match &spec.slots[slot].slot_type {
                SlotType::ResidualNorms => {
                    field.copy_from_slice(&batch.residual_norms_squared[row].to_le_bytes())
                }
                SlotType::QuantLayerCodes { layer, .. } => field.copy_from_slice(
                    &batch.layers[*layer as usize].codes[row * stride..(row + 1) * stride],
                ),
                SlotType::QuantLayerScales { layer } => {
                    field.copy_from_slice(&batch.layers[*layer as usize].scales[row].to_le_bytes())
                }
                SlotType::QuantLayerGammas { layer } => {
                    field.copy_from_slice(&batch.layers[*layer as usize].gammas[row].to_le_bytes())
                }
                SlotType::QuantLayerErrors { layer } => field.copy_from_slice(
                    &batch.layers[*layer as usize].corrected_error_ratios[row].to_le_bytes(),
                ),
                SlotType::QuantLayerConstants { layer } => field
                    .copy_from_slice(&batch.layers[*layer as usize].constants[row].to_le_bytes()),
                SlotType::DocIds | SlotType::Rows { .. } => {
                    unreachable!("records carry neither document ids nor rows as slots")
                }
            }
        }
    }
}

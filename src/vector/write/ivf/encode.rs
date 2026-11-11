//! The encode step: the stored centroids, and the Data entry's cluster blocks of
//! full-precision rows, doc ids and, for a quantized field, the encoded layer columns.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use common::CountingWriter;
use quant::cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, EncodedBatch, LayerSpec,
    PreparedCentroidWorkspace, QueryRotationPlan,
};
use quant::model::Grid;
use rayon::prelude::*;

use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::directory::{CompositeWrite, WritePtr};
use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::clustering::{decode_row_append, encode_vector, IvfCentroids};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::storage::blocks::{
    block_len, column_range, pad, write_metadata, BlockOffsetsWriter,
};
use crate::vector::storage::header::{VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::{Assignment, Slot, SlotType, VectorColMetadata};
use crate::vector::{BlockAddress, VectorElement, ENTRY_ALIGN};
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

/// Decoded rows one group of clusters holds between its read and its write.
const GROUP_DECODED_BYTES: usize = 256 << 20;

/// Test override of [`GROUP_DECODED_BYTES`]; 0 keeps the constant. Grouping never changes a written
/// byte, so a test may set it while other tests merge.
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

/// Writes the field's Data entry: metadata, then one aligned block per cluster of doc ids, encoded
/// columns and, for plain storage, the rows themselves. A replica row is encoded against its own
/// block's centroid like any row. Returns each target document's primary location.
///
/// A quantized field reads, normalizes and decodes a group of clusters on the calling thread,
/// encodes the group's clusters in parallel on the rayon pool, then writes their blocks in cluster
/// order on the calling thread. Each cluster encodes in tiles of `tile_rows` against its own
/// prepared centroid, as a serial loop would, so every byte written is the same.
pub(super) fn write_data_entry(
    merge: &FieldMerge,
    vec_write: &mut CompositeWrite,
    assigned_vectors: &[AssignedVector],
    cluster_offsets: &[u64],
    centroid_bytes: &[u8],
    assignment: Assignment,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Vec<BlockAddress>> {
    let (ctx, opts, quantization) = (merge.ctx, merge.opts, merge.quantization);
    let num_centroids = cluster_offsets.len() - 1;

    // IdMaps are emitted after Data so inter-entry padding never enters an id map.
    let id_map_start = Instant::now();
    if num_centroids >= u32::MAX as usize {
        return Err(TantivyError::InvalidArgument(
            "too many clusters for document locations".into(),
        ));
    }
    let mut locations = IdMap::empty_locations(merge.num_target_docs);
    timings.id_map_write = id_map_start.elapsed();

    // Data entry: metadata followed by aligned cluster blocks.
    let encode_start = Instant::now();
    let meta = VectorColMetadata::build_ivf(opts, quantization)?.with_assignment(assignment);
    let slots = meta.slots();
    // Plain blocks stream their rows first; quantized blocks keep the rows in `ExactRows` and
    // buffer every column.
    let rows_streamed = matches!(
        slots.first().map(|slot| &slot.slot_type),
        Some(SlotType::Rows { .. })
    );
    let doc_ids_slot = slots
        .iter()
        .position(|slot| matches!(slot.slot_type, SlotType::DocIds))
        .expect("clustered blocks carry document ids");
    timings.pad_bytes += vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
    let data_start = vec_write.written_bytes();
    let data = vec_write.for_field_with_idx(merge.field, VectorEntry::Data.index());
    let entry_start = data.written_bytes();
    let pos = write_metadata(data, &meta)?;
    let mut block_offsets = BlockOffsetsWriter::new(data.written_bytes() - entry_start);
    timings.pad_bytes += pos - 4 - meta.to_bytes().len();
    let row_bytes = opts.bytes_per_vector();
    let fixed_scratch = row_bytes + opts.dim() + opts.dim().div_ceil(64) * 8;
    let per_row_scratch = 2 * row_bytes + quantization.map_or(0, |c| c.bytes_per_row()) + 16;
    let tile_rows = (1usize << 20)
        .saturating_sub(fixed_scratch)
        .checked_div(per_row_scratch)
        .unwrap_or(0)
        .max(1);
    let mut blocks = BlockWriter {
        data,
        entry_start,
        pos,
        block_offsets: &mut block_offsets,
        slots: &slots,
        cancel: ctx.cancel,
        timings,
    };
    let reads = ClusterReads {
        merge,
        assigned_vectors,
        cluster_offsets,
        doc_ids_slot,
        tile_rows,
    };
    if quantization.is_some() {
        debug_assert!(
            !rows_streamed,
            "quantized blocks keep their rows in ExactRows"
        );
        write_quantized_blocks(&reads, &mut blocks, &mut locations, centroid_bytes, &meta)?;
    } else {
        write_plain_blocks(&reads, &mut blocks, &mut locations, rows_streamed)?;
    }
    let pos = blocks.pos;
    let entry_len = block_offsets.finish(data)?;
    timings.pad_bytes += entry_len - pos - (num_centroids + 1) * 8 - 8;
    assert_eq!(
        (data.written_bytes() - entry_start) as usize % ENTRY_ALIGN,
        0
    );
    data.flush()?;
    timings.vec_bytes += vec_write.written_bytes() - data_start;
    timings.encode = encode_start.elapsed();
    Ok(locations)
}

/// Where the Data entry's cluster rows come from.
struct ClusterReads<'a> {
    merge: &'a FieldMerge<'a>,
    assigned_vectors: &'a [AssignedVector],
    cluster_offsets: &'a [u64],
    doc_ids_slot: usize,
    tile_rows: usize,
}

impl ClusterReads<'_> {
    fn rows(&self, cluster: usize) -> std::ops::Range<usize> {
        self.cluster_offsets[cluster] as usize..self.cluster_offsets[cluster + 1] as usize
    }

    /// Visits `cluster`'s rows in tiles of `tile_rows`, polling cancellation before each tile:
    /// appends each row's doc id to `doc_ids`, points each primary document's location at its row,
    /// and hands `row` the stored row, normalized as the field's vectors are.
    fn read(
        &self,
        cluster: usize,
        doc_ids: &mut Vec<u8>,
        locations: &mut [BlockAddress],
        timings: &mut IvfBuildTimings,
        mut row: impl FnMut(&[u8]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        let merge = self.merge;
        let opts = merge.opts;
        let mut normalized = Vec::with_capacity(opts.bytes_per_vector());
        let mut local = 0u32;
        for tile in self.assigned_vectors[self.rows(cluster)].chunks(self.tile_rows) {
            if merge.ctx.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            for assigned in tile {
                timings.encode_reads += 1;
                doc_ids.extend_from_slice(&assigned.target_doc_id.to_le_bytes());
                if assigned.primary {
                    locations[assigned.target_doc_id as usize] =
                        BlockAddress::new(cluster as u32, local);
                }
                local += 1;
                let bytes = merge.readers[assigned.source.segment_ord as usize]
                    .vector_bytes_for_row(assigned.source.block_address)?;
                if opts.needs_normalization() {
                    normalized.clear();
                    normalized.extend_from_slice(&bytes);
                    maybe_normalize_bytes(opts, &mut normalized);
                    row(&normalized)?;
                } else {
                    row(&bytes)?;
                }
            }
        }
        Ok(())
    }
}

/// Writes cluster blocks to the Data entry in cluster order.
struct BlockWriter<'a, 'w> {
    data: &'w mut CountingWriter<WritePtr>,
    entry_start: u64,
    /// Bytes of the entry written so far.
    pos: usize,
    block_offsets: &'a mut BlockOffsetsWriter,
    slots: &'a [Slot],
    cancel: &'a dyn CancelSentinel,
    timings: &'a mut IvfBuildTimings,
}

impl BlockWriter<'_, '_> {
    fn written(&self) -> u64 {
        self.data.written_bytes() - self.entry_start
    }

    /// Records an empty cluster, which has no block.
    fn empty(&mut self) {
        let written = self.written();
        self.block_offsets.push(written, 0);
    }

    /// Pads and writes the buffered columns `first_buffered..` of an `n`-row block that starts at
    /// `block_start`, then pads the block to its length and records its offset. Column flushes poll
    /// cancellation; the buffers are cleared with their capacity kept.
    fn finish_block(
        &mut self,
        block_start: usize,
        n: usize,
        first_buffered: usize,
        bufs: &mut [Vec<u8>],
    ) -> crate::Result<()> {
        for idx in first_buffered..self.slots.len() {
            let col = column_range(self.slots, n, idx);
            let padding = block_start + col.start - self.pos;
            pad(self.data, padding)?;
            self.timings.pad_bytes += padding;
            assert_eq!(bufs[idx].len(), col.len());
            for chunk in bufs[idx].chunks(1 << 20) {
                if self.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                self.data.write_all(chunk)?;
            }
            bufs[idx].clear();
            self.pos = block_start + col.end;
        }
        let padding = block_start + block_len(self.slots, n) - self.pos;
        pad(self.data, padding)?;
        self.timings.pad_bytes += padding;
        self.pos += padding;
        let written = self.written();
        assert_eq!(written, self.pos as u64);
        self.block_offsets.push(written, n);
        Ok(())
    }
}

/// Plain blocks, one cluster at a time: rows stream straight to disk, doc ids gather in a buffer.
fn write_plain_blocks(
    reads: &ClusterReads,
    blocks: &mut BlockWriter,
    locations: &mut [BlockAddress],
    rows_streamed: bool,
) -> crate::Result<()> {
    let first_buffered = usize::from(rows_streamed);
    let mut bufs: Vec<Vec<u8>> = blocks.slots.iter().map(|_| Vec::new()).collect();
    for cluster in 0..reads.cluster_offsets.len() - 1 {
        let n = reads.rows(cluster).len();
        if n == 0 {
            blocks.empty();
            continue;
        }
        let block_start = blocks.pos;
        assert_eq!(blocks.written(), block_start as u64);
        let mut doc_ids = std::mem::take(&mut bufs[reads.doc_ids_slot]);
        let BlockWriter {
            data, pos, timings, ..
        } = blocks;
        reads.read(cluster, &mut doc_ids, locations, timings, |row| {
            if rows_streamed {
                data.write_all(row)?;
                *pos += row.len();
            }
            Ok(())
        })?;
        bufs[reads.doc_ids_slot] = doc_ids;
        blocks.finish_block(block_start, n, first_buffered, &mut bufs)?;
    }
    Ok(())
}

/// One quantized cluster between its read and its write.
struct PendingCluster {
    cluster: usize,
    rows: usize,
    /// The decoded rows, row-major, encoded in place.
    values: Vec<f32>,
    /// One buffer per slot: the doc ids from the read, then the encoded columns.
    bufs: Vec<Vec<u8>>,
}

/// One pool thread's encode state.
struct EncodeWorkspace {
    centroid: Vec<f32>,
    prepared: PreparedCentroidWorkspace,
    batch: BatchEncodeWorkspace,
}

/// What every cluster's encode shares.
struct EncodeSpec<'a> {
    centroid_bytes: &'a [u8],
    dim: usize,
    specs: &'a [LayerSpec],
    grids: &'a [Grid],
    slots: &'a [Slot],
    tile_rows: usize,
    scores_by_distance: bool,
}

impl PendingCluster {
    /// Encodes the rows in tiles of `tile_rows` against the cluster's prepared centroid and appends
    /// each tile's columns, then frees the decoded rows.
    fn encode(&mut self, spec: &EncodeSpec, workspace: &mut EncodeWorkspace) -> crate::Result<()> {
        let stride = spec.dim * f32::SIZE_BYTES;
        workspace.centroid.clear();
        decode_row_append::<f32>(
            &spec.centroid_bytes[self.cluster * stride..][..stride],
            spec.dim,
            &mut workspace.centroid,
        )?;
        let prepared = workspace.prepared.prepare(&workspace.centroid);
        for tile in self.values.chunks_mut(spec.tile_rows * spec.dim) {
            let rows = tile.len() / spec.dim;
            let batch = encode_batch_in_place_with_workspace(
                tile,
                rows,
                prepared,
                spec.specs,
                spec.grids,
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
            append_encoded_columns(&mut self.bufs, spec.slots, batch)?;
        }
        self.values = Vec::new();
        Ok(())
    }
}

/// Quantized blocks, a group of clusters at a time. A group fills up to [`GROUP_DECODED_BYTES`] of
/// decoded rows, so the pool has at least twice its thread count of clusters to balance whenever
/// that many fit; a cluster larger than the cap forms a group of its own.
fn write_quantized_blocks(
    reads: &ClusterReads,
    blocks: &mut BlockWriter,
    locations: &mut [BlockAddress],
    centroid_bytes: &[u8],
    meta: &VectorColMetadata,
) -> crate::Result<()> {
    let merge = reads.merge;
    let dim = merge.opts.dim();
    let (specs, grids) = meta.runtime();
    let plan = Arc::new(QueryRotationPlan::new(dim, &specs));
    let spec = EncodeSpec {
        centroid_bytes,
        dim,
        specs: &specs,
        grids: &grids,
        slots: blocks.slots,
        tile_rows: reads.tile_rows,
        scores_by_distance: merge.opts.metric().scores_by_distance(),
    };
    let row_bytes = dim * f32::SIZE_BYTES;
    let cap = group_decoded_bytes();
    let num_clusters = reads.cluster_offsets.len() - 1;
    let mut group_start = 0;
    while group_start < num_clusters {
        // Calling thread: doc ids, locations, and each cluster's rows read, normalized and decoded.
        let mut group_end = group_start;
        let mut group_bytes = 0;
        let mut pending = Vec::new();
        while group_end < num_clusters {
            let rows = reads.rows(group_end).len();
            if group_end > group_start && group_bytes + rows * row_bytes > cap {
                break;
            }
            group_bytes += rows * row_bytes;
            let mut cluster = PendingCluster {
                cluster: group_end,
                rows,
                values: Vec::with_capacity(rows * dim),
                bufs: blocks.slots.iter().map(|_| Vec::new()).collect(),
            };
            let mut doc_ids = Vec::with_capacity(rows * std::mem::size_of::<DocId>());
            reads.read(group_end, &mut doc_ids, locations, blocks.timings, |row| {
                decode_row_append::<f32>(row, dim, &mut cluster.values)
            })?;
            cluster.bufs[reads.doc_ids_slot] = doc_ids;
            pending.push(cluster);
            group_end += 1;
        }

        // Pool: one prepared-centroid and batch-encode workspace per thread.
        pending.par_iter_mut().try_for_each_init(
            || EncodeWorkspace {
                centroid: Vec::with_capacity(dim),
                prepared: PreparedCentroidWorkspace::new(Arc::clone(&plan)),
                batch: BatchEncodeWorkspace::with_capacity(dim, reads.tile_rows, &specs),
            },
            |workspace, cluster| cluster.encode(&spec, workspace),
        )?;

        // Calling thread: pad, flush the columns and record each block, in cluster order.
        for mut cluster in pending {
            if cluster.rows == 0 {
                blocks.empty();
                continue;
            }
            let block_start = blocks.pos;
            assert_eq!(blocks.written(), block_start as u64);
            blocks.finish_block(block_start, cluster.rows, 0, &mut cluster.bufs)?;
        }
        group_start = group_end;
    }
    Ok(())
}

/// Appends one encoded tile's columns to the cluster's column buffers, in slot order; rows and
/// document ids are gathered elsewhere.
fn append_encoded_columns(
    bufs: &mut [Vec<u8>],
    slots: &[Slot],
    batch: &EncodedBatch,
) -> std::io::Result<()> {
    for (idx, slot) in slots.iter().enumerate() {
        match &slot.slot_type {
            SlotType::ResidualNorms => {
                write_f32_run(&mut bufs[idx], &batch.residual_norms_squared)?
            }
            SlotType::QuantLayerCodes { layer, .. } => {
                bufs[idx].extend_from_slice(&batch.layers[*layer as usize].codes)
            }
            SlotType::QuantLayerScales { layer } => {
                write_f32_run(&mut bufs[idx], &batch.layers[*layer as usize].scales)?
            }
            SlotType::QuantLayerGammas { layer } => {
                write_u16_run(&mut bufs[idx], &batch.layers[*layer as usize].gammas)?
            }
            SlotType::QuantLayerErrors { layer } => write_u16_run(
                &mut bufs[idx],
                &batch.layers[*layer as usize].corrected_error_ratios,
            )?,
            SlotType::QuantLayerConstants { layer } => {
                write_f32_run(&mut bufs[idx], &batch.layers[*layer as usize].constants)?
            }
            SlotType::DocIds | SlotType::Rows { .. } => (),
        }
    }
    Ok(())
}

fn write_u16_run(writer: &mut impl Write, values: &[u16]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_run(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

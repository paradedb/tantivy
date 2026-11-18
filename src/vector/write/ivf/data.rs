//! The Data entry: metadata, then one aligned block per cluster of document ids, encoded columns
//! and, for plain storage, the rows themselves.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use common::CountingWriter;
use quant::cascade::{BatchEncodeWorkspace, PreparedCentroidWorkspace, QueryRotationPlan};
use rayon::prelude::*;

use super::encode::{group_decoded_bytes, EncodeSpec, EncodeWorkspace, PendingCluster};
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::directory::{CompositeWrite, WritePtr};
use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::clustering::decode_row_append;
use crate::vector::distance::maybe_normalize_bytes;
use crate::vector::storage::blocks::{
    block_len, column_range, pad, write_metadata, BlockOffsetsWriter,
};
use crate::vector::storage::header::{VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::{Assignment, Slot, SlotType, VectorColMetadata};
use crate::vector::{BlockAddress, VectorElement, ENTRY_ALIGN};
use crate::{DocId, TantivyError};

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
    if num_centroids >= u32::MAX as usize {
        return Err(TantivyError::InvalidArgument(
            "too many clusters for document locations".into(),
        ));
    }
    let mut locations = IdMap::empty_locations(merge.num_target_docs);

    // Data entry: metadata followed by aligned cluster blocks.
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
    let write_start = Instant::now();
    timings.pad_bytes += vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
    let data_start = vec_write.written_bytes();
    let data = vec_write.for_field_with_idx(merge.field, VectorEntry::Data.index());
    let entry_start = data.written_bytes();
    let pos = write_metadata(data, &meta)?;
    let mut block_offsets = BlockOffsetsWriter::new(data.written_bytes() - entry_start);
    timings.pad_bytes += pos - 4 - meta.to_bytes().len();
    timings.data_write += write_start.elapsed();
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
    let write_start = Instant::now();
    let entry_len = block_offsets.finish(data)?;
    timings.pad_bytes += entry_len - pos - (num_centroids + 1) * 8 - 8;
    assert_eq!(
        (data.written_bytes() - entry_start) as usize % ENTRY_ALIGN,
        0
    );
    data.flush()?;
    timings.data_write += write_start.elapsed();
    timings.vec_bytes += vec_write.written_bytes() - data_start;
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
        let read_start = Instant::now();
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
        timings.encode_read += read_start.elapsed();
        bufs[reads.doc_ids_slot] = doc_ids;
        let write_start = Instant::now();
        blocks.finish_block(block_start, n, first_buffered, &mut bufs)?;
        blocks.timings.data_write += write_start.elapsed();
    }
    Ok(())
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
        let read_start = Instant::now();
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
        blocks.timings.encode_read += read_start.elapsed();

        // Pool: one prepared-centroid and batch-encode workspace per thread.
        let encode_start = Instant::now();
        pending.par_iter_mut().try_for_each_init(
            || EncodeWorkspace {
                centroid: Vec::with_capacity(dim),
                prepared: PreparedCentroidWorkspace::new(Arc::clone(&plan)),
                batch: BatchEncodeWorkspace::with_capacity(dim, reads.tile_rows, &specs),
            },
            |workspace, cluster| cluster.encode(&spec, workspace),
        )?;
        blocks.timings.encode += encode_start.elapsed();

        // Calling thread: pad, flush the columns and record each block, in cluster order.
        let write_start = Instant::now();
        for mut cluster in pending {
            if cluster.rows == 0 {
                blocks.empty();
                continue;
            }
            let block_start = blocks.pos;
            assert_eq!(blocks.written(), block_start as u64);
            blocks.finish_block(block_start, cluster.rows, 0, &mut cluster.bufs)?;
        }
        blocks.timings.data_write += write_start.elapsed();
        group_start = group_end;
    }
    Ok(())
}

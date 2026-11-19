//! The Data entry: metadata, then one aligned block per cluster of document ids, encoded columns
//! and, for plain storage, the rows themselves. Plain blocks read their rows from the sources as
//! they are written; quantized blocks are written from the records the assign pass encoded.

use std::io::Write;
use std::time::Instant;

use common::CountingWriter;

use super::assign::ClusterRows;
use super::encode::encode_tile_rows;
use super::records::{MergedRecords, RecordHeader, RecordLayout};
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::directory::{CompositeWrite, WritePtr};
use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::distance::maybe_normalize_bytes;
use crate::vector::storage::blocks::{
    block_len, column_range, pad, write_metadata, BlockOffsetsWriter,
};
use crate::vector::storage::header::{VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::{Assignment, Slot, SlotType, VectorColMetadata};
use crate::vector::{BlockAddress, ENTRY_ALIGN};
use crate::TantivyError;

/// Writes the field's Data entry: metadata, then one aligned block per cluster of doc ids, encoded
/// columns and, for plain storage, the rows themselves. A replica row is encoded against its own
/// block's centroid like any row. Returns each target document's primary location.
pub(super) fn write_data_entry(
    merge: &FieldMerge,
    vec_write: &mut CompositeWrite,
    rows: ClusterRows,
    num_centroids: usize,
    assignment: Assignment,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Vec<BlockAddress>> {
    let (ctx, opts, quantization) = (merge.ctx, merge.opts, merge.quantization);

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
    let mut blocks = BlockWriter {
        data,
        entry_start,
        pos,
        block_offsets: &mut block_offsets,
        slots: &slots,
        cancel: ctx.cancel,
        timings,
    };
    match rows {
        ClusterRows::Addresses {
            assigned_vectors,
            cluster_offsets,
        } => {
            debug_assert_eq!(cluster_offsets.len(), num_centroids + 1);
            let reads = ClusterReads {
                merge,
                assigned_vectors: &assigned_vectors,
                cluster_offsets: &cluster_offsets,
                doc_ids_slot,
                tile_rows: encode_tile_rows(opts, quantization),
            };
            write_plain_blocks(&reads, &mut blocks, &mut locations, rows_streamed)?;
        }
        ClusterRows::Encoded(mut records) => {
            debug_assert!(
                !rows_streamed,
                "quantized blocks keep their rows in ExactRows"
            );
            let write_start = Instant::now();
            write_encoded_blocks(&mut records, &mut blocks, &mut locations, num_centroids)?;
            blocks.timings.data_write += write_start.elapsed();
        }
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

/// Where a plain Data entry's cluster rows come from.
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

/// Quantized blocks from the encoded records, in cluster order: each record appends its document
/// to the block's DocIds column and its slot bytes to the other columns, and points a primary
/// document's location at its row. A cluster without records has no block.
fn write_encoded_blocks(
    records: &mut MergedRecords,
    blocks: &mut BlockWriter,
    locations: &mut [BlockAddress],
    num_centroids: usize,
) -> crate::Result<()> {
    /// Records merged between cancellation polls.
    const POLL_RECORDS: u32 = 1 << 16;
    let layout = RecordLayout::new(blocks.slots);
    let mut bufs: Vec<Vec<u8>> = blocks.slots.iter().map(|_| Vec::new()).collect();
    // Clusters below `next_cluster` have their block or empty entry; `open` is the cluster being
    // filled and its block's start.
    let mut next_cluster = 0usize;
    let mut open: Option<(usize, usize)> = None;
    let mut rows = 0u32;
    while let Some(record) = records.next_record()? {
        let RecordHeader {
            cluster,
            doc,
            primary,
        } = RecordHeader::read(record);
        let cluster = cluster as usize;
        if open.map(|(open, _)| open) != Some(cluster) {
            if let Some((open, block_start)) = open.take() {
                blocks.finish_block(block_start, rows as usize, 0, &mut bufs)?;
                next_cluster = open + 1;
            }
            assert!(
                cluster >= next_cluster && cluster < num_centroids,
                "encoded records out of cluster order"
            );
            while next_cluster < cluster {
                blocks.empty();
                next_cluster += 1;
            }
            let block_start = blocks.pos;
            assert_eq!(blocks.written(), block_start as u64);
            open = Some((cluster, block_start));
            rows = 0;
        }
        bufs[layout.doc_ids_slot].extend_from_slice(&doc.to_le_bytes());
        for &(slot, offset, stride) in &layout.fields {
            bufs[slot].extend_from_slice(&record[offset..offset + stride]);
        }
        if primary {
            locations[doc as usize] = BlockAddress::new(cluster as u32, rows);
        }
        rows += 1;
        if rows % POLL_RECORDS == 0 && blocks.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
    }
    if let Some((open, block_start)) = open {
        blocks.finish_block(block_start, rows as usize, 0, &mut bufs)?;
        next_cluster = open + 1;
    }
    while next_cluster < num_centroids {
        blocks.empty();
        next_cluster += 1;
    }
    Ok(())
}

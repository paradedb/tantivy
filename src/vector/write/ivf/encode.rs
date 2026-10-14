//! The encode step: the stored centroids and their bounds, and the Data entry's cluster blocks of
//! full-precision rows, doc ids and, for a quantized field, the encoded layer columns.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, EncodedBatch,
    PreparedCentroidWorkspace, QueryRotationPlan,
};

use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::directory::CompositeWrite;
use crate::schema::VectorDType;
use crate::vector::clustering::{decode_row_append, encode_vector, IvfCentroids};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::storage::blocks::{
    block_len, column_range, pad, write_metadata, BlockDirectory,
};
use crate::vector::storage::header::{VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::DocLocation;
use crate::vector::storage::metadata::{Slot, SlotType, VectorColMetadata};
use crate::vector::{residual_norm, BoundsBuilder, ENTRY_ALIGN};
use crate::TantivyError;

/// The stored centroid rows, normalized as the field's vectors are, and a bounds builder that
/// saturates every cluster whose stored centroid is not finite.
pub(super) fn encode_centroids(
    merge: &FieldMerge,
    centroids: &IvfCentroids,
    num_centroids: usize,
) -> crate::Result<(Vec<u8>, BoundsBuilder)> {
    let opts = merge.opts;
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let centroid_rows: Vec<Vec<f32>> = centroid_matrix
        .values
        .chunks_exact(opts.dim())
        .map(|centroid| centroid.to_vec())
        .collect();
    let mut centroid_bytes = Vec::with_capacity(num_centroids * opts.bytes_per_vector());
    let mut bounds_builder = BoundsBuilder::new(num_centroids);
    let mut stored_centroid = Vec::with_capacity(opts.dim());
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
        stored_centroid.clear();
        decode_row_append::<f32>(&bytes, opts.dim(), &mut stored_centroid)?;
        if outcome != NormalizeOutcome::Normalized
            || stored_centroid.iter().any(|value| !value.is_finite())
        {
            bounds_builder.saturate(centroid_ord);
        }
        centroid_bytes.extend_from_slice(&bytes);
    }
    Ok((centroid_bytes, bounds_builder))
}

/// Writes the field's Data entry: metadata, then one aligned block per cluster of rows, doc ids
/// and encoded columns. Returns each target document's location.
pub(super) fn write_data_entry(
    merge: &FieldMerge,
    vec_write: &mut CompositeWrite,
    assigned_vectors: &[AssignedVector],
    cluster_offsets: &[u64],
    centroid_bytes: &[u8],
    bounds_builder: &mut BoundsBuilder,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Vec<DocLocation>> {
    let (ctx, opts, quantization) = (merge.ctx, merge.opts, merge.quantization);
    let num_centroids = cluster_offsets.len() - 1;
    let residual: fn(&[u8], &[f32]) -> f32 = match opts.dtype() {
        VectorDType::F32 => residual_norm::<f32>,
    };
    let centroid_stride = opts.bytes_per_vector();

    // IdMaps are emitted after Data so inter-entry padding never enters an id map.
    let id_map_start = Instant::now();
    if num_centroids >= u32::MAX as usize {
        return Err(TantivyError::InvalidArgument(
            "too many clusters for document locations".into(),
        ));
    }
    let mut locations = vec![DocLocation::ABSENT; merge.num_target_docs as usize];
    timings.id_map_write = id_map_start.elapsed();

    // Data entry: metadata followed by aligned cluster blocks.
    let encode_start = Instant::now();
    let meta = VectorColMetadata::build_ivf(opts, quantization)?;
    let slots = meta.slots();
    timings.pad_bytes += vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
    let data_start = vec_write.written_bytes();
    let data = vec_write.for_field_with_idx(merge.field, VectorEntry::Data.index());
    let entry_start = data.written_bytes();
    let mut pos = write_metadata(data, &meta)?;
    let mut directory = BlockDirectory::new(data.written_bytes() - entry_start);
    timings.pad_bytes += pos - 4 - meta.to_bytes().len();
    let (specs, grids) = meta.runtime();
    let row_bytes = opts.bytes_per_vector();
    let fixed_scratch = row_bytes + opts.dim() + opts.dim().div_ceil(64) * 8;
    let per_row_scratch = 2 * row_bytes + quantization.map_or(0, |c| c.bytes_per_row()) + 16;
    let tile_rows = (1usize << 20)
        .saturating_sub(fixed_scratch)
        .checked_div(per_row_scratch)
        .unwrap_or(0)
        .max(1);
    let mut centroid_workspace = quantization.map(|_| {
        PreparedCentroidWorkspace::new(Arc::new(QueryRotationPlan::new(opts.dim(), &specs)))
    });
    let mut encode_workspace =
        quantization.map(|_| BatchEncodeWorkspace::with_capacity(opts.dim(), tile_rows, &specs));
    let mut bufs: Vec<Vec<u8>> = slots.iter().map(|_| Vec::new()).collect();
    let mut normalized = Vec::with_capacity(row_bytes);
    let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
    let mut centroid = Vec::with_capacity(opts.dim());
    for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        let n = end - start;
        if n == 0 {
            directory.push(data.written_bytes() - entry_start, offsets[1] as u32);
            continue;
        }
        let block_start = pos;
        assert_eq!(data.written_bytes() - entry_start, block_start as u64);
        centroid.clear();
        decode_row_append::<f32>(
            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
            opts.dim(),
            &mut centroid,
        )?;
        let prepared = centroid_workspace
            .as_mut()
            .map(|workspace| workspace.prepare(&centroid));
        // Rows stream directly to disk; only encoded columns need cluster buffers.
        let mut local = 0u32;
        for tile in assigned_vectors[start..end].chunks(tile_rows) {
            if ctx.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            batch_values.clear();
            for assigned in tile {
                timings.source_reads += 1;
                bufs[1].extend_from_slice(&assigned.target_doc_id.to_le_bytes());
                locations[assigned.target_doc_id as usize] = DocLocation {
                    cluster: cluster as u32,
                    local,
                };
                local += 1;
                let bytes = merge.readers[assigned.source.segment_ord as usize]
                    .vector_bytes_for_row(assigned.source.row_id)?;
                let row: &[u8] = if opts.needs_normalization() {
                    normalized.clear();
                    normalized.extend_from_slice(&bytes);
                    maybe_normalize_bytes(opts, &mut normalized);
                    &normalized
                } else {
                    &bytes
                };
                data.write_all(row)?;
                pos += row.len();
                bounds_builder.add_native(cluster, residual(row, &centroid));
                if quantization.is_some() {
                    decode_row_append::<f32>(row, opts.dim(), &mut batch_values)?;
                }
            }
            if let Some(prepared) = prepared.as_ref() {
                let batch = encode_batch_in_place_with_workspace(
                    &mut batch_values,
                    tile.len(),
                    prepared,
                    &specs,
                    &grids,
                    encode_workspace.as_mut().unwrap(),
                    opts.metric() == crate::schema::Metric::L2,
                );
                append_encoded_columns(&mut bufs, &slots, batch)?;
            }
        }
        // Column flushes poll cancellation and retain capacity for the next cluster.
        for idx in 1..slots.len() {
            let col = column_range(&slots, n, idx);
            let padding = block_start + col.start - pos;
            pad(data, padding)?;
            timings.pad_bytes += padding;
            assert_eq!(bufs[idx].len(), col.len());
            for chunk in bufs[idx].chunks(1 << 20) {
                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                data.write_all(chunk)?;
            }
            bufs[idx].clear();
            pos = block_start + col.end;
        }
        let padding = block_start + block_len(&slots, n) - pos;
        pad(data, padding)?;
        timings.pad_bytes += padding;
        pos += padding;
        assert_eq!(data.written_bytes() - entry_start, pos as u64);
        directory.push(data.written_bytes() - entry_start, offsets[1] as u32);
    }
    let entry_len = directory.finish(data)?;
    timings.pad_bytes += entry_len - pos - (num_centroids + 1) * 12 - 8;
    assert_eq!(
        (data.written_bytes() - entry_start) as usize % ENTRY_ALIGN,
        0
    );
    data.flush()?;
    timings.vec_bytes += vec_write.written_bytes() - data_start;
    timings.encode = encode_start.elapsed();
    Ok(locations)
}

/// Appends one encoded tile's columns to the cluster's column buffers, in slot order.
fn append_encoded_columns(
    bufs: &mut [Vec<u8>],
    slots: &[Slot],
    batch: &EncodedBatch,
) -> std::io::Result<()> {
    for (idx, slot) in slots.iter().enumerate().skip(1) {
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
            SlotType::DocIds => (),
            SlotType::Rows { .. } => unreachable!(),
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

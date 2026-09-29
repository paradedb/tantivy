//! IVF merge-time clustering and vector encoding.

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
use cascade::prepare_centroid;
#[cfg(test)]
use cascade::LayerSpec;
use cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, PreparedCentroidWorkspace,
    QueryRotationPlan,
};
#[cfg(test)]
use quant_model::Grid;

#[cfg(test)]
use super::decode_row;
use super::{
    decode_row_append, encode_vector, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix,
    IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
#[cfg(test)]
use crate::schema::Metric;
use crate::schema::{Field, FieldType, VectorDType, VectorOptions};
use crate::vector::blocks::{block_len, column_range, finish_data, pad, write_metadata};
#[cfg(test)]
use crate::vector::distance::l2_squared;
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{
    write_centroid_header, write_vector_header, CentroidSlot, VectorEntry, HEADER_LEN,
};
use crate::vector::metadata::{SlotType, VectorColMetadata};
use crate::vector::router::{BuiltRouter, RouterKind};
use crate::vector::{
    residual_norm, BoundKind, BoundsBuilder, VectorQuantizationConfig, MAX_ELEM_BYTES, VEC_EXT,
};
use crate::{DocId, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// Per-field IVF build counters and timings, reported on `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    /// Source vector lookups, including lookups for documents without a vector.
    source_reads: usize,
    spill_bytes: usize,
    pad_bytes: usize,
    /// Bytes written to this field's vector entries, including entry padding.
    /// File headers and the composite footer are excluded.
    vec_bytes: u64,
    train: Duration,
    assign: Duration,
    id_map_write: Duration,
    encode: Duration,
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

#[cfg(test)]
fn quantization_runtime(
    config: &VectorQuantizationConfig,
    opts: &VectorOptions,
) -> crate::Result<(Vec<LayerSpec>, Vec<Grid>)> {
    Ok(VectorColMetadata::build_ivf(opts, Some(config))?.runtime())
}

/// Writes an empty IVF field to both vector composites.
fn write_empty_field_slots(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    field: Field,
    opts: &VectorOptions,
    router: &BuiltRouter,
    quantization: Option<&VectorQuantizationConfig>,
) -> crate::Result<()> {
    let meta = VectorColMetadata::build_ivf(opts, quantization)?;
    vec_write.align_next_field(MAX_ELEM_BYTES, HEADER_LEN)?;
    let data = vec_write.for_field_with_idx(field, VectorEntry::Data.index());
    let start = data.written_bytes();
    let len = write_metadata(data, &meta)?;
    finish_data(data, len)?;
    assert_eq!((data.written_bytes() - start) as usize % MAX_ELEM_BYTES, 0);
    {
        let centroids_w =
            centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
        IvfIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let offsets_w = centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
        IvfIndex::serialize_offsets(&[0u64], offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
        IvfIndex::serialize_bounds(BoundKind::Ball, &[], bounds_w)?;
        bounds_w.flush()?;
    }
    {
        let router_w = centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
        router.serialize(router_w)?;
        router_w.flush()?;
    }
    Ok(())
}

fn build_router(
    router: RouterKind,
    opts: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<BuiltRouter> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let shape = (matrix.rows, matrix.dims, matrix.values.len());
    let router = router.build(opts, centroids)?;
    let IvfCentroids::F32(matrix) = &*centroids;
    if (matrix.rows, matrix.dims, matrix.values.len()) != shape {
        return Err(TantivyError::InvalidArgument(
            "Router changed the centroid matrix shape while building".to_string(),
        ));
    }
    Ok(router)
}

pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: Option<&dyn IvfClusterer>,
    router: Option<RouterKind>,
) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }

    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }

    let clusterer = clusterer.ok_or_else(|| {
        TantivyError::InvalidArgument(
            "vector_clustering_threshold selected IVF merge, but no IvfClusterer is configured"
                .to_string(),
        )
    })?;
    let router = router.ok_or_else(|| {
        TantivyError::InvalidArgument(
            "vector_clustering_threshold selected IVF merge, but no Router is configured"
                .to_string(),
        )
    })?;

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
    if num_target_docs == 0 {
        return Ok(());
    }

    let settings = clusterer.merge_settings(num_target_docs as usize)?;
    let directory = ctx.target_segment.index().directory();
    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let centroids_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(CENTROIDS_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_vector_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_file = directory.open_write(&centroids_path)?;
    write_centroid_header(&mut centroids_file)?;
    let mut centroids_write = CompositeWrite::wrap(centroids_file);

    let mut id_maps = Vec::new();
    let mut build_reports = Vec::new();
    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let quantization = ctx
            .settings
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name());
        if let Some(config) = quantization {
            config.validate(opts)?;
        }
        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let vector_count = field_readers
            .iter()
            .map(|reader| reader.num_vectors())
            .sum::<usize>();
        if vector_count == 0 {
            let mut centroids = IvfCentroids::F32(IvfMatrix {
                values: Vec::new(),
                rows: 0,
                dims: opts.dim(),
            });
            let router = build_router(router, opts, &mut centroids)?;
            id_maps.push((field, Vec::new()));
            write_empty_field_slots(
                &mut vec_write,
                &mut centroids_write,
                field,
                opts,
                &router,
                quantization,
            )?;
            continue;
        }
        let training_sample_size = {
            let ratio = f64::from(settings.training_sample_ratio).clamp(f64::MIN_POSITIVE, 1.0);
            let target = ((vector_count as f64) * ratio).ceil() as usize;
            target.clamp(1, vector_count)
        };
        let training_sample_interval = (vector_count / training_sample_size).max(1);

        let residual: fn(&[u8], &[f32]) -> f32 = match opts.dtype() {
            VectorDType::F32 => residual_norm::<f32>,
        };
        let centroid_stride = opts.bytes_per_vector();

        match opts.dtype() {
            VectorDType::F32 => {
                let field_build_start = Instant::now();
                let mut timings = IvfBuildTimings::default();
                let mut training_values = Vec::with_capacity(training_sample_size * opts.dim());
                let mut training_doc_ids = Vec::with_capacity(training_sample_size);
                let mut target_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                    let reader = &field_readers[source_doc_addr.segment_ord as usize];
                    timings.source_reads += 1;
                    if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            decode_row_append::<f32>(&bytes, opts.dim(), &mut training_values)?;
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert!(
                    if ctx.readers.iter().any(|reader| reader.has_deletes()) {
                        present_vector_ord <= vector_count
                    } else {
                        present_vector_ord == vector_count
                    },
                    "{present_vector_ord} alive docs with vectors vs {vector_count} reported by \
                     source count()"
                );
                if training_doc_ids.is_empty() {
                    // `vector_count > 0`, yet the alive-doc walk found
                    // nothing to sample: every vector-bearing doc was
                    // deleted. Write the same empty slots as the
                    // no-vectors fast path — skipping the field would
                    // leave its slots missing from composites the other
                    // fields still write, and the reader errors on
                    // missing slots.
                    let mut centroids = IvfCentroids::F32(IvfMatrix {
                        values: Vec::new(),
                        rows: 0,
                        dims: opts.dim(),
                    });
                    let router = build_router(router, opts, &mut centroids)?;
                    id_maps.push((field, Vec::new()));
                    write_empty_field_slots(
                        &mut vec_write,
                        &mut centroids_write,
                        field,
                        opts,
                        &router,
                        quantization,
                    )?;
                    continue;
                }

                let training_rows = training_doc_ids.len();
                let training_vectors = IvfTrainingVectors::F32(IvfTrainingBatch {
                    doc_ids: training_doc_ids,
                    matrix: IvfMatrix {
                        values: training_values,
                        rows: training_rows,
                        dims: opts.dim(),
                    },
                });
                let train_start = Instant::now();
                let mut centroids = clusterer.train(opts, training_vectors)?;

                timings.train = train_start.elapsed();

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }

                let IvfCentroids::F32(centroid_matrix) = &centroids;
                if centroid_matrix.dims != opts.dim() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced centroids with {} dimensions, expected {}",
                        centroid_matrix.dims,
                        opts.dim()
                    )));
                }
                if centroid_matrix.values.len() != centroid_matrix.rows * centroid_matrix.dims {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced {} centroid values for {} rows x {} dimensions",
                        centroid_matrix.values.len(),
                        centroid_matrix.rows,
                        centroid_matrix.dims
                    )));
                }
                if centroid_matrix.rows == 0 {
                    return Err(TantivyError::InvalidArgument(
                        "IvfClusterer produced zero centroids".to_string(),
                    ));
                }
                let num_centroids = centroid_matrix.rows;

                let router = build_router(router, opts, &mut centroids)?;
                let IvfCentroids::F32(centroid_matrix) = &centroids;

                // Float working copy of the trained centroids — the
                // `.centroids` encode below reads per-row slices. Encoding +
                // Cosine normalization happen at the `.centroids` write
                // below.
                let centroid_rows: Vec<Vec<f32>> = centroid_matrix
                    .values
                    .chunks_exact(opts.dim())
                    .map(|centroid| centroid.to_vec())
                    .collect();

                let mut assigned_vectors = Vec::with_capacity(vector_count);
                let mut target_doc_id: DocId = 0;
                {
                    let mut batch_values = Vec::with_capacity(
                        settings.assign_batch_size.min(vector_count) * opts.dim(),
                    );
                    let mut batch_doc_ids =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut batch_sources =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut flush_assign_batch =
                        |batch_values: &mut Vec<f32>,
                         batch_doc_ids: &mut Vec<DocId>,
                         batch_sources: &mut Vec<(DocId, usize, DocId)>|
                         -> crate::Result<()> {
                            if batch_doc_ids.is_empty() {
                                return Ok(());
                            }
                            if ctx.cancel.wants_cancel() {
                                return Err(TantivyError::Cancelled);
                            }
                            let batch_len = batch_doc_ids.len();
                            let assign_start = Instant::now();
                            let clusters = clusterer.assign(
                                opts,
                                IvfVectors::F32(IvfVectorBatch {
                                    doc_ids: batch_doc_ids.as_slice(),
                                    matrix: IvfMatrixView {
                                        values: batch_values.as_slice(),
                                        rows: batch_len,
                                        dims: opts.dim(),
                                    },
                                }),
                                &centroids,
                            )?;
                            timings.assign += assign_start.elapsed();
                            if clusters.len() != batch_len {
                                return Err(TantivyError::InvalidArgument(format!(
                                    "IvfClusterer assigned {} clusters for {} vectors",
                                    clusters.len(),
                                    batch_len
                                )));
                            }
                            for (cluster, (target_doc_id, source_segment_ord, source_doc_id)) in
                                clusters.into_iter().zip(batch_sources.drain(..))
                            {
                                let cluster = cluster as usize;
                                if cluster >= num_centroids {
                                    return Err(TantivyError::InvalidArgument(format!(
                                        "IvfClusterer assigned vector to cluster {cluster}, but \
                                         only {num_centroids} centroids were trained"
                                    )));
                                }
                                assigned_vectors.push(AssignedVector {
                                    cluster,
                                    target_doc_id,
                                    source_segment_ord,
                                    source_doc_id,
                                });
                            }
                            batch_values.clear();
                            batch_doc_ids.clear();
                            Ok(())
                        };
                    for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                        let reader = &field_readers[source_doc_addr.segment_ord as usize];
                        timings.source_reads += 1;
                        if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                            batch_doc_ids.push(target_doc_id);
                            decode_row_append::<f32>(&bytes, opts.dim(), &mut batch_values)?;
                            batch_sources.push((
                                target_doc_id,
                                source_doc_addr.segment_ord as usize,
                                source_doc_addr.doc_id,
                            ));
                            if batch_doc_ids.len() == settings.assign_batch_size {
                                flush_assign_batch(
                                    &mut batch_values,
                                    &mut batch_doc_ids,
                                    &mut batch_sources,
                                )?;
                            }
                        }
                        target_doc_id += 1;
                    }
                    flush_assign_batch(&mut batch_values, &mut batch_doc_ids, &mut batch_sources)?;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert_eq!(assigned_vectors.len(), present_vector_ord);
                // The `.centroids` doc count: one posting row per document.
                let num_present_docs = assigned_vectors.len();

                let mut cluster_counts = vec![0usize; num_centroids];
                for assigned_vector in &assigned_vectors {
                    cluster_counts[assigned_vector.cluster] += 1;
                }

                assigned_vectors
                    .sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

                let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
                let mut next_offset = 0u64;
                cluster_offsets.push(next_offset);
                for cluster_count in cluster_counts {
                    next_offset += cluster_count as u64;
                    cluster_offsets.push(next_offset);
                }

                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                let mut bounds_builder = BoundsBuilder::new(num_centroids);
                let mut stored_centroid = Vec::with_capacity(opts.dim());
                for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    let outcome = maybe_normalize_bytes(opts, &mut bytes);
                    if outcome == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite centroid {centroid_ord} in field '{}' written \
                             un-normalized during merge",
                            entry.name(),
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

                // IdMaps are emitted after Data so inter-entry padding never enters an id map.
                let id_map_start = Instant::now();
                let row_doc_ids: Vec<DocId> =
                    assigned_vectors.iter().map(|v| v.target_doc_id).collect();
                id_maps.push((field, row_doc_ids));
                timings.id_map_write = id_map_start.elapsed();

                // Data entry: metadata followed by aligned cluster blocks.
                let encode_start = Instant::now();
                let meta = VectorColMetadata::build_ivf(opts, quantization)?;
                let slots = meta.slots();
                timings.pad_bytes += vec_write.align_next_field(MAX_ELEM_BYTES, HEADER_LEN)?;
                let data_start = vec_write.written_bytes();
                let data = vec_write.for_field_with_idx(field, VectorEntry::Data.index());
                let entry_start = data.written_bytes();
                let mut pos = write_metadata(data, &meta)?;
                timings.pad_bytes += pos - 4 - meta.to_bytes().len();
                let (specs, grids) = meta.runtime();
                let row_bytes = opts.bytes_per_vector();
                let fixed_scratch = row_bytes + opts.dim() + opts.dim().div_ceil(64) * 8;
                let per_row_scratch =
                    2 * row_bytes + quantization.map_or(0, |c| c.bytes_per_row()) + 16;
                let tile_rows = (1usize << 20)
                    .saturating_sub(fixed_scratch)
                    .checked_div(per_row_scratch)
                    .unwrap_or(0)
                    .max(1);
                let mut centroid_workspace = quantization.map(|_| {
                    PreparedCentroidWorkspace::new(Arc::new(QueryRotationPlan::new(
                        opts.dim(),
                        &specs,
                    )))
                });
                let mut encode_workspace = quantization
                    .map(|_| BatchEncodeWorkspace::with_capacity(opts.dim(), tile_rows, &specs));
                let mut bufs: Vec<Vec<u8>> = slots.iter().map(|_| Vec::new()).collect();
                let mut normalized = Vec::with_capacity(row_bytes);
                let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
                let mut centroid = Vec::with_capacity(opts.dim());
                for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
                    let start = offsets[0] as usize;
                    let end = offsets[1] as usize;
                    let n = end - start;
                    if n == 0 {
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
                    for tile in assigned_vectors[start..end].chunks(tile_rows) {
                        if ctx.cancel.wants_cancel() {
                            return Err(TantivyError::Cancelled);
                        }
                        batch_values.clear();
                        for assigned in tile {
                            timings.source_reads += 1;
                            let bytes = field_readers[assigned.source_segment_ord]
                                .vector_bytes(assigned.source_doc_id)?
                                .ok_or_else(|| {
                                    TantivyError::InternalError("missing source vector".into())
                                })?;
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
                            for (idx, slot) in slots.iter().enumerate().skip(1) {
                                match &slot.slot_type {
                                    SlotType::ResidualNorms => write_f32_run(
                                        &mut bufs[idx],
                                        &batch.residual_norms_squared,
                                    )?,
                                    SlotType::QuantLayerCodes { layer, .. } => bufs[idx]
                                        .extend_from_slice(&batch.layers[*layer as usize].codes),
                                    SlotType::QuantLayerScales { layer } => write_f32_run(
                                        &mut bufs[idx],
                                        &batch.layers[*layer as usize].scales,
                                    )?,
                                    SlotType::QuantLayerGammas { layer } => write_u16_run(
                                        &mut bufs[idx],
                                        &batch.layers[*layer as usize].gammas,
                                    )?,
                                    SlotType::QuantLayerErrors { layer } => write_u16_run(
                                        &mut bufs[idx],
                                        &batch.layers[*layer as usize].corrected_error_ratios,
                                    )?,
                                    SlotType::QuantLayerConstants { layer } => write_f32_run(
                                        &mut bufs[idx],
                                        &batch.layers[*layer as usize].constants,
                                    )?,
                                    SlotType::Rows { .. } => unreachable!(),
                                }
                            }
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
                }
                let entry_len = finish_data(data, pos)?;
                timings.pad_bytes += entry_len - pos;
                assert_eq!(
                    (data.written_bytes() - entry_start) as usize % MAX_ELEM_BYTES,
                    0
                );
                data.flush()?;
                timings.vec_bytes += vec_write.written_bytes() - data_start;
                timings.encode = encode_start.elapsed();

                {
                    let centroids_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
                    IvfIndex::serialize_centroids(
                        num_centroids,
                        num_present_docs,
                        &centroid_bytes,
                        opts,
                        centroids_w,
                    )?;
                    centroids_w.flush()?;
                }
                {
                    let offsets_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
                    IvfIndex::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }
                {
                    let bounds_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
                    IvfIndex::serialize_bounds(
                        BoundKind::Ball,
                        &bounds_builder.finish(),
                        bounds_w,
                    )?;
                    bounds_w.flush()?;
                }

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                let router_w =
                    centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
                router.serialize(router_w)?;
                router_w.flush()?;

                build_reports.push((
                    field,
                    timings,
                    field_build_start.elapsed(),
                    num_centroids,
                    vector_count,
                ));
            }
        }
    }

    for (field, docs) in id_maps {
        let started = Instant::now();
        let id_map_start = vec_write.written_bytes();
        let id_map = vec_write.for_field_with_idx(field, VectorEntry::IdMap.index());
        IdMap::serialize_explicit(&docs, id_map)?;
        id_map.flush()?;
        if let Some((_, timings, total, num_centroids, vector_count)) =
            build_reports.iter_mut().find(|(f, ..)| *f == field)
        {
            timings.vec_bytes += vec_write.written_bytes() - id_map_start;
            let elapsed = started.elapsed();
            timings.id_map_write += elapsed;
            *total += elapsed;
            log::info!(
                target: "paradedb::ivf_build",
                "ivf_build timings_ms train={} assign={} id_map_write={} encode={} total={} \
                 centroids={} vectors={} source_reads={} spill_bytes={} vec_bytes={} pad_bytes={} encode_ms={}",
                timings.train.as_millis(),
                timings.assign.as_millis(),
                timings.id_map_write.as_millis(),
                timings.encode.as_millis(),
                total.as_millis(),
                num_centroids,
                vector_count,
                timings.source_reads, timings.spill_bytes, timings.vec_bytes, timings.pad_bytes,
                timings.encode.as_millis(),
            );
        }
    }
    vec_write.close()?;
    centroids_write.close()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::prepared::QuantizedQueryCtx;
    use crate::vector::tests::ground_truth;
    use crate::vector::{
        TopDocsByVectorSimilarity, VectorEstimatorMeasurements, VectorEstimatorQuery,
        VectorEstimatorSource, VectorQuantizationLayer,
    };
    use crate::{Index, TantivyDocument};

    #[test]
    fn quantization_merge_source_has_no_estimator_analysis_entrypoint() {
        let source = include_str!("plugin.rs");
        let test_module_start = source
            .rfind("\n#[cfg(test)]\nmod tests {")
            .expect("plugin source must retain one terminal test module");
        let production_source = &source[..test_module_start];
        for forbidden in [
            "build_grid(",
            "audit_prefix_error_model(",
            "prepare_fp_query(",
            "audit_error",
            "diagnostic_error",
            "VectorEstimatorMeasurements",
            "VectorEstimatorQuery",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "quantization merge production source contains forbidden analysis hook {forbidden}"
            );
        }
    }

    #[test]
    fn merge_runtime_uses_persisted_grid_and_rho() {
        let config = quant_fixture_config_for(100, Metric::Dot, &[1, 4]);
        let (_, resolved) =
            quantization_runtime(&config, &VectorOptions::new(100, Metric::Dot)).unwrap();
        assert_eq!(resolved.len(), 2);
        for (layer, grid) in resolved.iter().enumerate() {
            let persisted = config
                .grids
                .iter()
                .find(|grid| grid.bits == config.layers[layer].bits)
                .unwrap();
            assert_eq!(grid.bits, persisted.bits);
            if grid.bits != 1 {
                assert_eq!(grid.points, persisted.points);
            } else {
                assert!(grid.points.is_empty());
            }
            assert_eq!(grid.rho_model, persisted.rho_model);
        }
    }

    // Mixed field contracts retain exact entry lengths and aligned Data boundaries.
    fn check_two_field_entries(empty_second: bool) -> crate::Result<()> {
        use common::{HasLen, OwnedBytes};

        use crate::directory::{CompositeFile, FileHandle, FileSlice};
        use crate::vector::blocks::{block_align, data_entry_len, Blocks};
        use crate::vector::header::read_vector_header;
        #[derive(Debug)]
        struct ByteAddressed(Vec<u8>);
        impl HasLen for ByteAddressed {
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        impl FileHandle for ByteAddressed {
            fn read_bytes(&self, range: std::ops::Range<usize>) -> std::io::Result<OwnedBytes> {
                Ok(OwnedBytes::new(self.0[range].to_vec()))
            }
            fn storage_block_len(&self) -> Option<usize> {
                Some(1)
            }
        }
        let opts = VectorOptions::new(65, Metric::L2);
        let mut schema = Schema::builder();
        let plain = schema.add_vector_field("plain", opts.clone());
        let quant = schema.add_vector_field("quant", opts.clone());
        let config = VectorQuantizationConfig::materialize(
            "quant".into(),
            &opts,
            vec![VectorQuantizationLayer { bits: 1, seed: 17 }],
        )?;
        let index = Index::builder()
            .schema(schema.build())
            .settings(IndexSettings {
                vector_clustering_threshold: 1,
                vector_quantization: vec![config],
                ..Default::default()
            })
            .ivf_clusterer(Arc::new(QuantFixtureClusterer {
                dim: 65,
                metric: Metric::L2,
            }))
            .ivf_router(RouterKind::Rng)?
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for doc in 0..7 {
            let values = vec![doc as f32 / 7.0; 65];
            let mut document = TantivyDocument::new();
            document.add_vector(plain, &values);
            if !empty_second {
                document.add_vector(quant, &values);
            }
            writer.add_document(document)?;
            if doc == 2 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        writer.wait_merging_threads()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let file = segment.open_read(SegmentComponent::Custom(VEC_EXT.into()))?;
        let file = FileSlice::new(Arc::new(ByteAddressed(file.read_bytes()?.to_vec())));
        let (_, body) = read_vector_header(&file)?;
        let composite = CompositeFile::open(&body)?;
        let mut data_ends = Vec::new();
        let mut id_starts = Vec::new();
        let mut aligns = Vec::new();
        for (field, count) in [(plain, 7), (quant, if empty_second { 0 } else { 7 })] {
            let vector = segment.vector_index(field)?;
            assert!(vector.metadata().is_some());
            let ivf = vector.index().unwrap();
            let rows = (0..ivf.num_clusters())
                .map(|b| ivf.cluster_range(b).start)
                .chain(std::iter::once(count))
                .collect();
            let data = composite
                .open_read_with_idx(field, VectorEntry::Data.index())
                .unwrap();
            let id_map = composite
                .open_read_with_idx(field, VectorEntry::IdMap.index())
                .unwrap();
            assert_eq!(id_map.len(), 1 + count * std::mem::size_of::<DocId>());
            let start = data.storage_block_ord(0).unwrap();
            assert_eq!(start % MAX_ELEM_BYTES, 0);
            assert_eq!(data.len() % MAX_ELEM_BYTES, 0);
            let blocks = Blocks::open(data.clone(), &opts, count, Some(rows))?;
            assert_eq!(
                data.len(),
                data_entry_len(*blocks.block_starts.last().unwrap() as usize)
            );
            aligns.push(block_align(&blocks.slots));
            data_ends.push(start + data.len());
            id_starts.push(id_map.storage_block_ord(0).unwrap());
        }
        assert_ne!(aligns[0], aligns[1]);
        assert_eq!(data_ends.iter().max(), id_starts.iter().min());
        assert!(data_ends
            .iter()
            .all(|end| id_starts.iter().all(|start| end <= start)));
        Ok(())
    }

    // Empty quantized IVF fields still carry exact metadata-only Data and Explicit IdMap entries.
    #[test]
    fn two_field_ivf_includes_empty_field() -> crate::Result<()> {
        check_two_field_entries(true)
    }

    // Plain and SignPlane blocks use different element sizes in one composite file.
    #[test]
    fn mixed_plain_quantized_entries_have_exact_lengths() -> crate::Result<()> {
        check_two_field_entries(false)
    }

    const QUANT_FIXTURE_DIM: usize = 64;

    struct QuantFixtureClusterer {
        dim: usize,
        metric: Metric,
    }

    impl IvfClusterer for QuantFixtureClusterer {
        fn training_sample_ratio(&self) -> f32 {
            0.5
        }

        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfTrainingVectors,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), self.dim);
            let values = match self.metric {
                Metric::L2 => [0.0_f32, 1.0]
                    .into_iter()
                    .flat_map(|center| std::iter::repeat_n(center, self.dim))
                    .collect(),
                Metric::Cosine | Metric::Dot => {
                    let mut values = vec![0.0; 2 * self.dim];
                    values[0] = 1.0;
                    values[self.dim + 1] = 1.0;
                    values
                }
            };
            Ok(IvfCentroids::F32(IvfMatrix {
                values,
                rows: 2,
                dims: self.dim,
            }))
        }

        fn assign(
            &self,
            _options: &VectorOptions,
            vectors: IvfVectors<'_>,
            _centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            let IvfVectors::F32(vectors) = vectors;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(self.dim)
                .map(|row| match self.metric {
                    Metric::L2 => u32::from(row[0] >= 0.5),
                    Metric::Cosine | Metric::Dot => u32::from(row[1] > row[0]),
                })
                .collect())
        }
    }

    fn quant_fixture_config(dim: usize) -> VectorQuantizationConfig {
        quant_fixture_config_for(dim, Metric::L2, &[1, 4])
    }

    fn quant_fixture_config_for(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
    ) -> VectorQuantizationConfig {
        let seeds = [0x1111, 0x2222, 0x3333, 0x4444];
        VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &VectorOptions::new(dim, metric),
            schedule
                .iter()
                .enumerate()
                .map(|(layer, &bits)| VectorQuantizationLayer {
                    bits,
                    seed: seeds[layer],
                })
                .collect(),
        )
        .unwrap()
    }

    fn fixture_vector(metric: Metric, dim: usize, doc: usize) -> Vec<f32> {
        match metric {
            Metric::L2 => {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                (0..dim)
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect()
            }
            Metric::Cosine | Metric::Dot => {
                let cluster = usize::from(doc >= 4);
                let mut vector: Vec<f32> = (0..dim)
                    .map(|coordinate| ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.025)
                    .collect();
                vector[cluster] += 1.0;
                vector
            }
        }
    }

    fn fixture_estimator_queries_for(metric: Metric, dim: usize) -> Vec<Vec<f32>> {
        (0..4)
            .map(|query| match metric {
                Metric::L2 => {
                    let center = if query < 2 { 0.0 } else { 1.0 };
                    (0..dim)
                        .map(|coordinate| {
                            center + ((query * dim + coordinate) as f32 * 0.023).cos() * 0.1
                        })
                        .collect()
                }
                Metric::Cosine => {
                    let cluster = usize::from(query >= 2);
                    let mut vector: Vec<f32> = (0..dim)
                        .map(|coordinate| ((query * dim + coordinate) as f32 * 0.023).cos() * 0.025)
                        .collect();
                    vector[cluster] += 1.0;
                    vector
                }
                Metric::Dot => {
                    unreachable!("quantized matrix fixture covers L2 and cosine")
                }
            })
            .collect()
    }

    fn fixture_estimator_queries(dim: usize) -> Vec<Vec<f32>> {
        fixture_estimator_queries_for(Metric::L2, dim)
    }

    fn estimator_measurements(
        vector_reader: &crate::vector::VectorIndexReader,
        queries: &[Vec<f32>],
        sample_rows: usize,
        alive: Option<&crate::fastfield::AliveBitSet>,
    ) -> crate::Result<Option<VectorEstimatorMeasurements>> {
        let queries = queries
            .iter()
            .cloned()
            .map(|values| VectorEstimatorQuery {
                values,
                excluded_doc_id: None,
            })
            .collect::<Vec<_>>();
        Ok(vector_reader.measure_estimator_queries(
            VectorEstimatorSource::Provided,
            &queries,
            sample_rows,
            alive,
        )?)
    }

    fn build_quantized_fixture_with_schedule(dim: usize, quantized: bool) -> crate::Result<Index> {
        build_quantized_fixture_case(dim, Metric::L2, &[1, 4], quantized)
    }

    fn build_quantized_fixture_case(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
    ) -> crate::Result<Index> {
        build_quantized_fixture_case_with_router(dim, metric, schedule, quantized, RouterKind::Rng)
    }

    fn build_quantized_fixture_case_with_router(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
        router: RouterKind,
    ) -> crate::Result<Index> {
        build_quantized_fixture_in_directory_with_router(
            dim,
            metric,
            schedule,
            quantized,
            crate::directory::RamDirectory::create(),
            router,
        )
    }

    fn build_quantized_fixture_in_directory(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
        directory: impl crate::directory::Directory,
    ) -> crate::Result<Index> {
        build_quantized_fixture_in_directory_with_router(
            dim,
            metric,
            schedule,
            quantized,
            directory,
            RouterKind::Rng,
        )
    }

    fn build_quantized_fixture_in_directory_with_router(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
        directory: impl crate::directory::Directory,
        router: RouterKind,
    ) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_vector_field("embedding", VectorOptions::new(dim, metric));
        let label_field = schema_builder.add_text_field("label", STRING | STORED);
        let schema = schema_builder.build();
        let mut settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..Default::default()
        };
        if quantized {
            settings.vector_quantization = vec![quant_fixture_config_for(dim, metric, schedule)];
        }
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(QuantFixtureClusterer { dim, metric }))
            .ivf_router(router)?
            .create(directory)?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mut segments = Vec::new();
        for doc in 0..8 {
            let vector = fixture_vector(metric, dim, doc);
            let mut document = TantivyDocument::new();
            document.add_vector(field, &vector);
            document.add_text(label_field, format!("d{doc}"));
            if doc % 2 == 0 {
                document.add_text(label_field, "keep");
            }
            writer.add_document(document)?;
            if doc == 3 || doc == 7 {
                writer.commit()?;
                for id in index.searchable_segment_ids()? {
                    if !segments.contains(&id) {
                        segments.push(id);
                    }
                }
            }
        }
        writer.merge(&segments).wait()?;
        writer.wait_merging_threads()?;
        Ok(index)
    }

    fn build_quantized_fixture(dim: usize, quantized: bool) -> crate::Result<Index> {
        build_quantized_fixture_with_schedule(dim, quantized)
    }

    fn build_flat_quantized_fixture(dim: usize) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field =
            schema_builder.add_vector_field("embedding", VectorOptions::new(dim, Metric::L2));
        let schema = schema_builder.build();
        let settings = IndexSettings {
            vector_clustering_threshold: usize::MAX,
            vector_quantization: vec![quant_fixture_config(dim)],
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for doc in 0..8 {
            let center = if doc < 4 { 0.0 } else { 1.0 };
            let vector: Vec<f32> = (0..dim)
                .map(|coordinate| center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1)
                .collect();
            let mut document = TantivyDocument::new();
            document.add_vector(field, &vector);
            writer.add_document(document)?;
        }
        writer.commit()?;
        Ok(index)
    }

    fn fixture_expected(query: &[f32], dim: usize, top_n: usize) -> Vec<(u32, u32)> {
        let mut expected: Vec<(f32, u32)> = (0..8)
            .map(|doc| {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                let vector: Vec<f32> = (0..dim)
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect();
                (-l2_squared(query, &vector), doc as u32)
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        expected[..top_n]
            .iter()
            .map(|&(score, doc)| (score.to_bits(), doc))
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    enum QuantizedMatrixScenario {
        None,
        Filter,
        Deletes,
    }

    fn fixture_search_query(metric: Metric, dim: usize) -> Vec<f32> {
        match metric {
            Metric::L2 => vec![0.05; dim],
            Metric::Cosine | Metric::Dot => {
                let mut query: Vec<f32> = (0..dim)
                    .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos() * 0.01)
                    .collect();
                query[0] += 0.8;
                query[1] += 0.6;
                query
            }
        }
    }

    fn fixture_filter_docs(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<crate::DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut admitted = std::collections::HashSet::new();
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment, &mut |docs| {
                admitted.extend(
                    docs.iter()
                        .copied()
                        .map(|doc| crate::DocAddress::new(segment_ord as u32, doc)),
                );
            })?;
        }
        Ok(admitted)
    }

    fn fixture_exact_hits(
        index: &Index,
        metric: Metric,
        query: &[f32],
        filter: Option<&dyn Query>,
        top_n: usize,
    ) -> crate::Result<Vec<(crate::Score, crate::DocAddress)>> {
        let field = index.schema().get_field("embedding")?;
        let mut hits = ground_truth::top_k(index, field, metric, query, 8)?;
        if let Some(filter) = filter {
            let admitted = fixture_filter_docs(index, filter)?;
            hits.retain(|(_, address)| admitted.contains(address));
        }
        hits.truncate(top_n);
        Ok(hits)
    }

    fn assert_matrix_results(
        context: &str,
        actual: &[(crate::Score, crate::DocAddress)],
        expected: &[(crate::Score, crate::DocAddress)],
        stats: &crate::vector::backend::ProbeStats,
    ) {
        if actual == expected {
            return;
        }

        let actual_docs: std::collections::HashSet<_> =
            actual.iter().map(|(_, address)| address.doc_id).collect();
        let missing = expected
            .iter()
            .map(|(_, address)| address.doc_id)
            .find(|doc| !actual_docs.contains(doc));
        let attribution = if let Some(doc) = missing {
            if !stats.quantized_trace.scored_docs.contains(&doc) {
                "routing/admission miss"
            } else if stats
                .quantized_trace
                .boundary_docs
                .iter()
                .any(|survivors| !survivors.contains(&doc))
            {
                "band drop"
            } else if !stats.quantized_trace.rerank_docs.contains(&doc) {
                "rerank fetch bug"
            } else {
                "rerank scoring/order bug"
            }
        } else {
            "rerank scoring/order bug"
        };
        panic!(
            "{context}: {attribution}; actual={actual:?} expected={expected:?} trace={:?}",
            stats.quantized_trace
        );
    }

    fn assert_quantized_matrix_storage(
        index: &Index,
        metric: Metric,
        schedule: &[u8],
    ) -> crate::Result<()> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let field = index.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vector_reader.index().expect("matrix fixture must be IVF");
        assert_eq!(ivf.num_rows(), 8);
        let quantized = vector_reader
            .quantization()
            .expect("matrix fixture must carry quantized slots");
        assert_eq!(
            quantized
                .index_ctx()
                .meta
                .layers()
                .iter()
                .map(|layer| layer.bits())
                .collect::<Vec<_>>(),
            schedule
        );
        for row in 0..ivf.num_rows() {
            assert!(quantized.residual_norm(row)?.is_finite());
            for (layer, stored) in quantized.layers().iter().enumerate() {
                assert!((1.0..=4.0).contains(&stored.gamma(row)?));
                let corrected_error = stored.error_ratio(row)?;
                assert!(corrected_error.is_finite() && corrected_error >= 0.0);
                assert_eq!(
                    stored.constant(row)?.is_some(),
                    metric == Metric::L2,
                    "layer {layer} split-constant presence must follow the metric at row {row}"
                );
            }
        }
        Ok(())
    }

    fn run_quantized_matrix_query(
        index: &Index,
        metric: Metric,
        schedule: &[u8],
        scenario: QuantizedMatrixScenario,
        depth: usize,
    ) -> crate::Result<()> {
        const TOP_N: usize = 3;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let label = index.schema().get_field("label")?;
        let query = fixture_search_query(metric, index.settings().vector_quantization[0].dim);
        let keep = TermQuery::new(
            Term::from_field_text(label, "keep"),
            IndexRecordOption::Basic,
        );
        let filter: &dyn Query = if matches!(scenario, QuantizedMatrixScenario::Filter) {
            &keep
        } else {
            &AllQuery
        };
        let expected = fixture_exact_hits(index, metric, &query, Some(filter), TOP_N)?;
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), TOP_N)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            })
            .with_max_scan_levels(depth);
        let fruit = searcher.search(filter, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        let context =
            format!("metric={metric:?} schedule={schedule:?} scenario={scenario:?} depth={depth}");
        assert_matrix_results(&context, &fruit.results, &expected, stats);

        let layer0 = stats.layers.get(0).expect("layer 0 must execute");
        assert_eq!(
            stats.layer0_eligible,
            layer0.scored(),
            "{context}: layer 0 must score exactly the admitted eligible rows"
        );
        assert_eq!(
            stats.eligible_charged,
            layer0.scored(),
            "{context}: the probe budget must charge exactly the selected rows"
        );
        assert!(layer0.scored() > 0, "{context}: {stats:?}");
        assert!(
            layer0.survivors() <= layer0.scored(),
            "{context}: {stats:?}"
        );
        if depth == 1 {
            assert!(stats.layers.get(1).is_none(), "{context}: {stats:?}");
        } else {
            let layer1 = stats.layers.get(1).expect("layer 1 must execute");
            assert_eq!(
                layer1.scored(),
                layer0.survivors(),
                "{context}: every boundary-0 survivor must be refined"
            );
            assert!(
                layer1.survivors() <= layer1.scored(),
                "{context}: {stats:?}"
            );
        }
        if metric == Metric::L2 && matches!(scenario, QuantizedMatrixScenario::None) {
            assert!(
                layer0.survivors() < layer0.scored(),
                "{context}: the unfiltered L2 matrix cell must prove that boundary 0 measurably \
                 drops at least one scored candidate: {stats:?}"
            );
        }
        assert_eq!(
            stats.quantized_trace.boundary_docs.len(),
            depth,
            "{context}: one identity snapshot per executed boundary"
        );
        for boundary in &stats.quantized_trace.boundary_docs {
            assert!(
                boundary
                    .iter()
                    .all(|doc| stats.quantized_trace.scored_docs.contains(doc)),
                "{context}: a later layer retained a row absent from layer-0 selection"
            );
        }
        assert!(
            stats.quantized_trace.rerank_docs.iter().all(|doc| stats
                .quantized_trace
                .boundary_docs
                .last()
                .is_some_and(|boundary| boundary.contains(doc))),
            "{context}: rerank read a row absent from the final selection"
        );
        for max_probe_fraction in [0.25, 1.0] {
            const ELIGIBILITY_TOP_N: usize = 8;
            let params = AdaptiveProbeParams {
                max_probe_fraction,
                min_probe_clusters: 2,
                ..Default::default()
            };
            let quantized = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params.clone())
                    .with_max_scan_levels(depth),
            )?;
            let level0 = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params)
                    .with_max_scan_levels(0),
            )?;
            assert_eq!(
                quantized.stats[0].quantized_trace.scored_docs,
                level0.stats[0].quantized_trace.scored_docs,
                "{context}: selected candidate set at probe fraction {max_probe_fraction}"
            );
        }
        match scenario {
            QuantizedMatrixScenario::Filter => {
                assert!(stats.pruned_filter > 0, "{context}: {stats:?}")
            }
            QuantizedMatrixScenario::Deletes => {
                assert!(stats.pruned_dead > 0, "{context}: {stats:?}")
            }
            QuantizedMatrixScenario::None => {}
        }
        Ok(())
    }

    fn quantized_vec_file(index: &Index) -> crate::Result<Vec<u8>> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        Ok(searcher.segment_readers()[0]
            .open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?
            .read_bytes()?
            .to_vec())
    }

    fn assert_relative_1e5(actual: f32, expected: f32, context: &str) {
        let tolerance = 1e-5 * expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    fn assert_quantized_bridge_exactness(dim: usize) -> crate::Result<()> {
        let index = build_quantized_fixture(dim, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = (
            quantized.index_ctx().specs.clone(),
            quantized.index_ctx().grids.clone(),
        );
        let query: Vec<f32> = (0..dim)
            .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos())
            .collect();
        let harness = cascade::prepare_split_query(&query, &specs, &grids, 4);
        let scan = QuantizedQueryCtx::new(Arc::clone(quantized.index_ctx()), query);

        for row in 0..ivf.num_rows() {
            let mut scan_sum = 0.0;
            let mut harness_sum = 0.0;
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let codes = stored.code_bytes(row)?;
                let scale = stored.scale(row)?;
                let constant = stored.constant(row)?;
                let scan_estimate = scan.score_layer(layer, &codes, scale, constant)?;
                let harness_estimate = match constant {
                    Some(constant) => {
                        harness.score_layer(layer, &codes, scale, constant, specs[layer])
                    }
                    None => {
                        harness.score_layer_without_constant(layer, &codes, scale, specs[layer])
                    }
                };
                assert_relative_1e5(
                    scan_estimate,
                    harness_estimate,
                    &format!("d={dim} row={row} layer={layer}"),
                );
                scan_sum += scan_estimate;
                harness_sum += harness_estimate;
            }
            assert_relative_1e5(scan_sum, harness_sum, &format!("d={dim} row={row} summed"));
        }
        Ok(())
    }

    #[test]
    fn bridge_exactness_d768_and_d100() -> crate::Result<()> {
        assert_quantized_bridge_exactness(768)?;
        assert_quantized_bridge_exactness(100)
    }

    // Settings version three remains a valid write target for vector format four.
    #[test]
    fn settings_v3_builds_v4_and_readers_ignore_global_changes() -> crate::Result<()> {
        let mut index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let field = index.schema().get_field("embedding")?;
        let config = &index.settings().vector_quantization[0];
        assert_eq!(config.format_version, 3);
        config.validate(&VectorOptions::new(QUANT_FIXTURE_DIM, Metric::L2))?;
        assert_eq!(&quantized_vec_file(&index)?[..4], &[4, 0, 0, 0]);
        index.settings_mut().vector_quantization.clear();
        let searcher = index.reader()?.searcher();
        let vector = searcher.segment_readers()[0].vector_index(field)?;
        assert_eq!(vector.quantization().unwrap().layers().len(), 2);
        assert!(vector.quantization().unwrap().residual_norm(0)?.is_finite());
        Ok(())
    }

    #[test]
    fn level_zero_matches_unquantized_ivf() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let unquantized = build_quantized_fixture(DIM, false)?;
        let quantized = build_quantized_fixture(DIM, true)?;
        let field = unquantized.schema().get_field("embedding")?;
        let unquantized_fruit = unquantized.reader()?.searcher().search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query.clone(), 3)
                .with_adaptive_params(params.clone()),
        )?;
        let level_zero_collector = TopDocsByVectorSimilarity::new(field, query, 3)
            .with_adaptive_params(params)
            .with_max_scan_levels(0);
        let quantized_reader = quantized.reader()?;
        let quantized_searcher = quantized_reader.searcher();
        assert!(quantized_searcher.segment_readers()[0]
            .vector_index(field)?
            .quantization()
            .is_some());
        let level_zero_fruit = quantized_searcher.search(&AllQuery, &level_zero_collector)?;
        assert!(level_zero_collector.quantized_query_count() == 0);

        assert_eq!(level_zero_fruit.results, unquantized_fruit.results);
        assert_eq!(level_zero_fruit.stats.len(), 1);
        assert_eq!(unquantized_fruit.stats.len(), 1);
        let level_zero = &level_zero_fruit.stats[0];
        let baseline = &unquantized_fruit.stats[0];
        assert!(level_zero.layers.get(0).is_none(), "{level_zero:?}");
        assert!(level_zero.routing_visited_count > 0, "{level_zero:?}");
        assert!(level_zero.clusters_probed() > 0, "{level_zero:?}");
        assert!(level_zero.candidates_scored > 0, "{level_zero:?}");
        assert!(level_zero.exact_scan_ns.is_some(), "{level_zero:?}");
        assert_eq!(level_zero.candidates_scored, baseline.candidates_scored);
        assert_eq!(level_zero.exact_rows_read, baseline.exact_rows_read);
        assert_eq!(level_zero.postings_row, baseline.postings_row);
        assert_eq!(level_zero.postings_skipped, baseline.postings_skipped);
        assert_eq!(
            level_zero.routing_visited_count,
            baseline.routing_visited_count
        );
        assert_eq!(
            level_zero.work_charged.to_bits(),
            baseline.work_charged.to_bits()
        );
        Ok(())
    }

    // Metadata reports the stored contract for each backend, independently of the build target.
    #[test]
    fn public_metadata_reports_stored_contract_after_settings_change() -> crate::Result<()> {
        use crate::vector::{Partition, Quantizer, VectorColMetadata};
        for mode in 0..3 {
            let mut index = match mode {
                0 => build_quantized_fixture(64, false)?,
                1 => build_quantized_fixture(64, true)?,
                _ => build_flat_quantized_fixture(64)?,
            };
            let field = index.schema().get_field("embedding")?;
            let before = {
                let reader = index.reader()?;
                let searcher = reader.searcher();
                let vector = searcher.segment_readers()[0].vector_index(field)?;
                let meta = vector.metadata().expect("stored field");
                assert_eq!(meta.field().dim(), 64);
                assert_eq!(meta.field().dtype(), crate::schema::VectorDType::F32);
                assert_eq!(meta.field().metric(), Metric::L2);
                assert_eq!(
                    meta.field().norm_policy(),
                    crate::vector::VectorNormPolicy::None
                );
                if mode == 2 {
                    assert!(matches!(
                        meta.field().partition(),
                        Partition::Uniform { .. }
                    ));
                } else {
                    assert!(matches!(meta.field().partition(), Partition::Clusters));
                }
                if mode == 1 {
                    assert!(matches!(meta, VectorColMetadata::Quantized { .. }));
                    assert_eq!(
                        meta.layers()
                            .iter()
                            .map(Quantizer::bits)
                            .collect::<Vec<_>>(),
                        [1, 4]
                    );
                    assert_eq!(
                        meta.quantized_bytes_per_row(),
                        Some(quant_fixture_config(64).bytes_per_row())
                    );
                    assert!(matches!(meta.layers()[0], Quantizer::SignPlane { .. }));
                } else {
                    assert!(matches!(meta, VectorColMetadata::Plain(_)));
                    assert!(meta.layers().is_empty());
                    assert_eq!(meta.quantized_bytes_per_row(), None);
                }
                meta.to_bytes()
            };
            index.settings_mut().vector_quantization =
                vec![quant_fixture_config_for(64, Metric::L2, &[4])];
            let reader = index.reader()?;
            let searcher = reader.searcher();
            let vector = searcher.segment_readers()[0].vector_index(field)?;
            assert_eq!(vector.metadata().unwrap().to_bytes(), before);
        }
        let absent = crate::vector::VectorIndexReader::empty(VectorOptions::new(64, Metric::L2));
        assert!(absent.metadata().is_none());
        Ok(())
    }

    // Each stored schedule has one cache cell and merged search equals the segment union.
    #[test]
    fn mixed_segments_share_preparation_by_stored_metadata() -> crate::Result<()> {
        use crate::collector::Collector;
        for mixed in [false, true] {
            let mut index = build_quantized_fixture_case(64, Metric::L2, &[1, 4], true)?;
            let field = index.schema().get_field("embedding")?;
            let mut existing = index.searchable_segment_ids()?;
            for _ in 0..2 {
                index.settings_mut().vector_quantization = vec![quant_fixture_config_for(
                    64,
                    Metric::L2,
                    if mixed { &[1] } else { &[1, 4] },
                )];
                let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for doc in 0..8 {
                    let mut document = TantivyDocument::new();
                    document.add_vector(field, &fixture_vector(Metric::L2, 64, doc));
                    writer.add_document(document)?;
                    if doc == 3 || doc == 7 {
                        writer.commit()?;
                    }
                }
                let fresh: Vec<_> = index
                    .searchable_segment_ids()?
                    .into_iter()
                    .filter(|id| !existing.contains(id))
                    .collect();
                writer.merge(&fresh).wait()?;
                writer.wait_merging_threads()?;
                existing = index.searchable_segment_ids()?;
            }
            index.settings_mut().vector_quantization.clear();
            let searcher = index.reader()?.searcher();
            assert_eq!(searcher.segment_readers().len(), 3);
            let make_collector = || {
                TopDocsByVectorSimilarity::new(field, fixture_search_query(Metric::L2, 64), 5)
                    .with_adaptive_params(AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    })
            };
            let collector = make_collector();
            let combined = searcher.search(&AllQuery, &collector)?;
            assert_eq!(collector.quantized_query_count(), if mixed { 2 } else { 1 });
            let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let mut union = Vec::new();
            for (ord, segment) in searcher.segment_readers().iter().enumerate() {
                let single = make_collector();
                let fruit = single.collect_segment(weight.as_ref(), ord as u32, segment)?;
                union.extend(single.merge_fruits(vec![fruit])?.results);
            }
            union.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            union.truncate(5);
            assert_eq!(combined.results, union);
        }
        Ok(())
    }

    #[test]
    fn reused_collector_prepares_query_per_quantization_config() -> crate::Result<()> {
        const DIM: usize = 64;
        let field_of = |index: &Index| index.schema().get_field("embedding");
        let first = build_quantized_fixture_case(DIM, Metric::L2, &[1, 4], true)?;
        let second = build_quantized_fixture_case(DIM, Metric::L2, &[4], true)?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 2,
            ..Default::default()
        };
        let query = vec![0.05_f32; DIM];
        let collector = |field| {
            TopDocsByVectorSimilarity::new(field, query.clone(), 3)
                .with_adaptive_params(params.clone())
        };

        let reused = collector(field_of(&first)?);
        first.reader()?.searcher().search(&AllQuery, &reused)?;
        assert_eq!(reused.quantized_query_count(), 1);
        let mut reused_fruit = second.reader()?.searcher().search(&AllQuery, &reused)?;
        let mut fresh_fruit = second
            .reader()?
            .searcher()
            .search(&AllQuery, &collector(field_of(&second)?))?;
        assert_eq!(reused.quantized_query_count(), 2);
        for stats in reused_fruit.stats.iter_mut().chain(&mut fresh_fruit.stats) {
            stats.clear_stage_timings();
        }

        assert_eq!(reused_fruit.results, fresh_fruit.results);
        assert_eq!(
            format!("{:?}", reused_fruit.stats),
            format!("{:?}", fresh_fruit.stats)
        );
        Ok(())
    }

    #[test]
    fn level_zero_flat_segment_remains_exact() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let expected = fixture_expected(&query, DIM, 3);
        let flat = build_flat_quantized_fixture(DIM)?;
        let reader = flat.reader()?;
        let searcher = reader.searcher();
        let field = flat.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.index().is_none());
        assert!(vector_reader.quantization().is_none());

        let fruit = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query, 3).with_max_scan_levels(0),
        )?;
        assert_eq!(
            fruit
                .results
                .iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            expected
        );
        let stats = &fruit.stats[0];
        assert_eq!(stats.exact_rows_read, 8, "{stats:?}");
        assert_eq!(stats.routing_visited_count, 0, "{stats:?}");
        assert_eq!(stats.clusters_probed(), 0, "{stats:?}");
        assert!(stats.layers.get(0).is_none(), "{stats:?}");
        Ok(())
    }

    #[test]
    fn merge_quantization_matches_kernel_harness_and_is_reproducible() -> crate::Result<()> {
        let index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        assert_eq!(ivf.num_rows(), 8);
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = (
            quantized.index_ctx().specs.clone(),
            quantized.index_ctx().grids.clone(),
        );
        let centroid_bytes = ivf.centroid_bytes()?;
        let centroid_stride = QUANT_FIXTURE_DIM * std::mem::size_of::<f32>();

        for cluster in 0..ivf.num_clusters() {
            let centroid = decode_row::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                QUANT_FIXTURE_DIM,
            )?;
            let prepared = prepare_centroid(&centroid, &specs);
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(
                    &vector_reader.vector_bytes_for_row(row)?,
                    QUANT_FIXTURE_DIM,
                )?;
                let mut expected_input = vector.clone();
                let mut workspace = cascade::BatchEncodeWorkspace::new();
                let expected = cascade::encode_batch_in_place_with_workspace(
                    &mut expected_input,
                    1,
                    &prepared,
                    &specs,
                    &grids,
                    &mut workspace,
                    true,
                );
                assert_eq!(
                    quantized.residual_norm(row)?.to_bits(),
                    l2_squared(&vector, &centroid).to_bits()
                );
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    let stored_codes = stored.code_bytes(row)?;
                    assert_eq!(
                        stored_codes.len(),
                        QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                        "divisible dimensions use exact byte strides"
                    );
                    assert_eq!(
                        stored_codes.as_slice(),
                        expected.layers[layer].codes.as_slice()
                    );
                    assert_eq!(stored.scale(row)?, expected.layers[layer].scales[0]);
                    assert_eq!(
                        stored.gamma(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(expected.layers[layer].gammas[0]).to_bits()
                    );
                    assert_eq!(
                        stored.error_ratio(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(
                            expected.layers[layer].corrected_error_ratios[0]
                        )
                        .to_bits()
                    );
                    assert_eq!(
                        stored
                            .constant(row)?
                            .expect("L2 fixture requires split constants")
                            .to_bits(),
                        expected.layers[layer].constants[0].to_bits()
                    );
                }
            }
        }

        let query = vec![0.05_f32; QUANT_FIXTURE_DIM];
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            });
        let quantized_fruit = searcher.search(&AllQuery, &collector)?;
        assert!(collector.quantized_query_count() > 0);
        assert_eq!(quantized_fruit.stats.len(), 1);
        let stats = &quantized_fruit.stats[0];
        let layer0 = stats.layers.get(0).expect("layer 0 must execute");
        let layer1 = stats.layers.get(1).expect("layer 1 must execute");
        assert!(layer0.scored() > 0, "{stats:?}");
        assert!(layer0.survivors() <= layer0.scored(), "{stats:?}");
        assert_eq!(
            layer1.scored(),
            layer0.survivors(),
            "the two-layer fixture refines every first-boundary survivor: {stats:?}"
        );
        assert!(layer1.survivors() <= layer0.survivors(), "{stats:?}");
        assert!(stats.rerank_rows <= layer1.survivors(), "{stats:?}");
        assert_eq!(stats.exact_rows_read, stats.rerank_rows, "{stats:?}");
        let hits = quantized_fruit.results;
        let mut expected: Vec<(f32, u32)> = (0..8)
            .map(|doc| {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                let vector: Vec<f32> = (0..QUANT_FIXTURE_DIM)
                    .map(|coordinate| {
                        center + ((doc * QUANT_FIXTURE_DIM + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect();
                (-l2_squared(&query, &vector), doc as u32)
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        assert_eq!(
            hits.iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            expected[..3]
                .iter()
                .map(|&(score, doc)| (score.to_bits(), doc))
                .collect::<Vec<_>>()
        );

        let first = quantized_vec_file(&index)?;
        let second = quantized_vec_file(&build_quantized_fixture(QUANT_FIXTURE_DIM, true)?)?;
        assert_eq!(
            first, second,
            "fixed assignment and seeds must be byte-identical"
        );
        Ok(())
    }

    #[test]
    fn quantized_bounds_gate_skips_provably_useless_cluster() -> crate::Result<()> {
        let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 1,
                ..Default::default()
            })
            .with_max_scan_levels(1);
        let fruit = searcher.search(&AllQuery, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        assert_eq!(
            fruit.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
            "{stats:?}"
        );
        assert_eq!(stats.bounds_skips, 1, "{stats:?}");
        assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
        assert_eq!(stats.bound_armed_probe_sum, 0, "{stats:?}");
        assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
        Ok(())
    }

    #[test]
    fn quantized_probe_terminates_at_ceiling() -> crate::Result<()> {
        let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 0.1,
                min_probe_clusters: 1,
                ..Default::default()
            })
            .with_max_scan_levels(1);
        let fruit = searcher.search(&AllQuery, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        assert_eq!(
            fruit.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
            "{stats:?}"
        );
        assert_eq!(
            stats.termination,
            crate::vector::ProbeTermination::Ceiling,
            "{stats:?}"
        );
        assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
            "{stats:?}"
        );
        Ok(())
    }

    /// Under the stacked router the quantized loop runs APS: a loose
    /// recall target stops before the full budget is spent.
    #[test]
    fn quantized_probe_terminates_at_recall_target() -> crate::Result<()> {
        let index = build_quantized_fixture_case_with_router(
            100,
            Metric::L2,
            &[1],
            true,
            RouterKind::Stacked,
        )?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 1,
                recall_target: 0.5,
                ..Default::default()
            })
            .with_max_scan_levels(1);
        let fruit = searcher.search(&AllQuery, &collector)?;
        let stats = &fruit.stats[0];
        assert_eq!(
            stats.termination,
            crate::vector::ProbeTermination::RecallTarget,
            "{stats:?}"
        );
        assert!(
            stats
                .recall_estimate
                .is_some_and(|estimate| estimate >= 0.5),
            "{stats:?}"
        );
        assert_eq!(
            fruit.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
            "{stats:?}"
        );
        Ok(())
    }

    #[test]
    fn quantized_filtered_rows_are_not_charged() -> crate::Result<()> {
        let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
        let field = index.schema().get_field("embedding")?;
        let label = index.schema().get_field("label")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let keep = TermQuery::new(
            Term::from_field_text(label, "keep"),
            IndexRecordOption::Basic,
        );
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 2,
            ..Default::default()
        };
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let (_, n_avg, open_share) =
            params.resolved_work_budget(ivf.num_clusters(), ivf.num_docs())?;
        let row = (1.0 - open_share) / n_avg;
        // Request all fixture rows so both scans open every cluster.
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 8)
            .with_adaptive_params(params)
            .with_max_scan_levels(1);
        let unfiltered = searcher.search(&AllQuery, &collector)?;
        let filtered = searcher.search(&keep, &collector)?;
        assert_eq!(
            unfiltered.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 8)?
        );
        assert_eq!(
            filtered.results,
            fixture_exact_hits(&index, Metric::L2, &query, Some(&keep), 8)?
        );
        assert_eq!(unfiltered.stats.len(), 1);
        assert_eq!(filtered.stats.len(), 1);
        let unfiltered = &unfiltered.stats[0];
        let filtered = &filtered.stats[0];
        assert_eq!(
            filtered.candidates_scored,
            fixture_filter_docs(&index, &keep)?.len(),
            "{filtered:?}"
        );
        let expected = (unfiltered.candidates_scored - filtered.candidates_scored) as f64 * row;
        assert!(
            ((unfiltered.work_charged - filtered.work_charged) as f64 - expected).abs() < 1e-5,
            "unfiltered={unfiltered:?}; filtered={filtered:?}; expected difference={expected}"
        );
        Ok(())
    }

    #[test]
    fn quantized_top_n_fixture_matrix_matches_direct_exact_oracle() -> crate::Result<()> {
        const SCHEDULES: &[&[u8]] = &[&[1], &[1, 4], &[1, 1, 4], &[2, 4]];

        for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
            for &schedule in SCHEDULES {
                let dim = 100;

                let primary = build_quantized_fixture_case(dim, metric, schedule, true)?;
                assert_quantized_matrix_storage(&primary, metric, schedule)?;
                for scenario in [
                    QuantizedMatrixScenario::None,
                    QuantizedMatrixScenario::Filter,
                ] {
                    for depth in 1..=schedule.len() {
                        run_quantized_matrix_query(&primary, metric, schedule, scenario, depth)?;
                    }
                }

                let label = primary.schema().get_field("label")?;
                let mut writer: crate::IndexWriter<TantivyDocument> =
                    primary.writer_with_num_threads(1, 30_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for doc in [0, 4] {
                    writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
                }
                writer.commit()?;
                drop(writer);
                for depth in 1..=schedule.len() {
                    run_quantized_matrix_query(
                        &primary,
                        metric,
                        schedule,
                        QuantizedMatrixScenario::Deletes,
                        depth,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn general_dimension_quantized_bridge_at_d100() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture(DIM, true)?;
        let reader = index.reader().map_err(|error| {
            TantivyError::InternalError(format!("general-d reader open failed: {error}"))
        })?;
        reader.reload().map_err(|error| {
            TantivyError::InternalError(format!("general-d reader reload failed: {error}"))
        })?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field).map_err(|error| {
            TantivyError::InternalError(format!("general-d vector reader failed: {error}"))
        })?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = (
            quantized.index_ctx().specs.clone(),
            quantized.index_ctx().grids.clone(),
        );
        let centroid_bytes = ivf.centroid_bytes()?;
        let centroid_stride = DIM * std::mem::size_of::<f32>();

        for cluster in 0..ivf.num_clusters() {
            let centroid = decode_row::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                DIM,
            )?;
            let prepared = prepare_centroid(&centroid, &specs);
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(&vector_reader.vector_bytes_for_row(row)?, DIM)?;
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let expected = cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    assert_eq!(stored.code_bytes(row)?.as_slice(), expected.codes[layer]);
                    assert_eq!(stored.scale(row)?, expected.scales[layer]);
                    assert_eq!(
                        stored
                            .constant(row)?
                            .expect("L2 fixture requires split constants")
                            .to_bits(),
                        expected.constants[layer].to_bits()
                    );
                }
            }
        }

        let query = vec![0.05_f32; DIM];
        let quantized_hits = searcher
            .search(
                &AllQuery,
                &TopDocsByVectorSimilarity::new(field, query.clone(), 3).with_adaptive_params(
                    AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    },
                ),
            )?
            .results;
        assert_eq!(
            quantized_hits
                .iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            fixture_expected(&query, DIM, 3)
        );
        Ok(())
    }

    #[test]
    fn l2_quantized_fixture_growth_matches_768_1_plus_4_ledger() -> crate::Result<()> {
        const DIM: usize = 768;
        const ROWS: usize = 8;
        let config = quant_fixture_config(DIM);
        assert_eq!(config.bytes_per_row(), 508);

        let plain = quantized_vec_file(&build_quantized_fixture(DIM, false)?)?;
        let quantized = quantized_vec_file(&build_quantized_fixture(DIM, true)?)?;
        let physical_growth = quantized.len() - plain.len();
        let logical_growth = ROWS * config.bytes_per_row();
        println!(
            "VECTOR_QUANTIZATION_SIZE dim={DIM} rows={ROWS} plain_bytes={} quantized_bytes={} \
             physical_growth={physical_growth} logical_growth={logical_growth}",
            plain.len(),
            quantized.len(),
        );
        assert!(physical_growth >= logical_growth);
        let opts = VectorOptions::new(DIM, Metric::L2);
        let plain_meta = VectorColMetadata::build_ivf(&opts, None)?;
        let quant_meta = VectorColMetadata::build_ivf(&opts, Some(&config))?;
        let entry_len = |meta: &VectorColMetadata| {
            use crate::vector::blocks::{align_up, block_align, data_entry_len};
            data_entry_len(
                align_up(4 + meta.to_bytes().len(), block_align(&meta.slots()))
                    + 2 * block_len(&meta.slots(), ROWS / 2),
            )
        };
        let expected_growth = entry_len(&quant_meta) - entry_len(&plain_meta);
        assert!(
            physical_growth.abs_diff(expected_growth) <= 8,
            "only composite footer varints may differ"
        );
        assert_eq!(logical_growth, ROWS * 508);
        Ok(())
    }

    #[test]
    fn estimator_measurement_uses_production_path_and_is_centered() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.quantization().is_some());
        let queries = fixture_estimator_queries(DIM);
        let estimator_queries = queries
            .iter()
            .cloned()
            .map(|values| VectorEstimatorQuery {
                values,
                excluded_doc_id: None,
            })
            .collect::<Vec<_>>();
        let measurements = vector_reader
            .measure_estimator_queries(
                VectorEstimatorSource::Provided,
                &estimator_queries,
                1_000,
                None,
            )?
            .expect("quantized slots remain available to explicit diagnostics");
        assert_eq!(measurements.source(), VectorEstimatorSource::Provided);
        assert_eq!(measurements.sample_rows(), 8);
        assert_eq!(measurements.query_count(), queries.len() as u32);
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == 8 * queries.len() as u64));
        for (depth, moments) in measurements.aggregate().iter().enumerate() {
            let bias = moments
                .bias()
                .expect("fixture must produce estimator errors");
            assert!(
                bias.abs() <= 0.3,
                "depth {} normalized estimator bias {bias} exceeds 0.3",
                depth + 1
            );
        }

        let fruit = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, queries[0].clone(), 3).with_adaptive_params(
                AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 2,
                    ..Default::default()
                },
            ),
        )?;
        assert!(fruit.stats[0].layers.get(0).is_some());
        assert!(fruit.stats[0].postings_row > 0);
        Ok(())
    }

    #[test]
    fn estimator_samples_only_live_posting_rows() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let alive = crate::fastfield::AliveBitSet::for_test_from_deleted_docs(&[1, 3], 8);
        let live_posting_rows = (0..vector_reader.index().unwrap().num_rows())
            .filter(|&row| alive.is_alive(vector_reader.doc_id_at(row)))
            .count();
        assert_eq!(live_posting_rows, 6);

        let queries = fixture_estimator_queries(DIM);
        let measurements =
            estimator_measurements(vector_reader.as_ref(), &queries, usize::MAX, Some(&alive))?
                .unwrap();
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| { depth.sample_count == (live_posting_rows * queries.len()) as u64 }));

        let bounded =
            estimator_measurements(vector_reader.as_ref(), &queries, 5, Some(&alive))?.unwrap();
        assert!(bounded
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == (5 * queries.len()) as u64));
        Ok(())
    }

    #[test]
    fn held_out_estimator_excludes_the_source_row() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let queries = vector_reader
            .sample_estimator_pseudo_queries(1, segment.alive_bitset())?
            .expect("quantized fixture must support pseudo-query sampling");
        assert_eq!(queries.len(), 1);
        assert!(queries.iter().all(|query| query.excluded_doc_id.is_some()));

        let measurements = vector_reader
            .measure_estimator_queries(
                VectorEstimatorSource::HeldOut,
                &queries,
                usize::MAX,
                segment.alive_bitset(),
            )?
            .expect("quantized fixture must support estimator measurement");
        assert_eq!(measurements.source(), VectorEstimatorSource::HeldOut);
        assert_eq!(measurements.sample_rows(), 8);
        assert_eq!(measurements.query_count(), 1);
        assert!(measurements
            .aggregate()
            .iter()
            .all(|moments| moments.sample_count == 7));
        Ok(())
    }
    #[test]
    fn layer_zero_io_matches_requested_page_spans() -> crate::Result<()> {
        use crate::vector::storage_io::test_support::{PagedDirectory, PAGE_BYTES};
        use crate::vector::Stage;
        let directory = PagedDirectory::default();
        let index =
            build_quantized_fixture_in_directory(64, Metric::L2, &[1], true, directory.clone())?;
        let field = index.schema().get_field("embedding")?;
        let searcher = index.reader()?.searcher();
        directory.reads.lock().unwrap().clear();
        let result = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, fixture_search_query(Metric::L2, 64), 16)
                .with_adaptive_params(AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 2,
                    ..Default::default()
                }),
        )?;
        let reads = directory.reads.lock().unwrap();
        let ranges: Vec<_> = reads
            .iter()
            .filter_map(|(stage, range)| matches!(stage, Stage::LayerScan(0)).then_some(range))
            .collect();
        let io = result.stats[0].layers.get(0).unwrap().io;
        assert!(io.reads > 0);
        assert_eq!(io.reads, ranges.len() as u64);
        assert_eq!(
            io.bytes_read,
            ranges.iter().map(|r| r.len() as u64).sum::<u64>()
        );
        assert_eq!(
            io.storage_blocks,
            ranges
                .iter()
                .filter(|r| !r.is_empty())
                .map(|r| ((r.end - 1) / PAGE_BYTES - r.start / PAGE_BYTES + 1) as u64)
                .sum::<u64>()
        );
        Ok(())
    }
}

//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;
use std::time::{Duration, Instant};

use cascade::{encode_batch_in_place, prepare_centroid, LayerSpec};
use quant_model::{build_grid, Grid};

use super::{
    decode_row, encode_vector, BuiltRouter, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix,
    IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, RoutingIndex,
    CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, Metric, VectorDType, VectorOptions};
use crate::vector::distance::{l2_squared, maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{centroid_slot, vec_slot, write_header, CURRENT, HEADER_LEN};
use crate::vector::{
    quantized_code_stride, residual_norm, BoundKind, BoundsBuilder, NeighborhoodGraphConfig,
    RelativeNeighborhoodGraph, VectorQuantizationConfig, VectorQuantizer,
    QUANTIZED_CODE_ALIGNMENT, VEC_EXT,
};
use crate::{DocId, Executor, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// A multi-threaded [`Executor`] when the host has the parallelism, the
/// single-threaded one otherwise.
fn build_executor(name: &'static str) -> crate::Result<Executor> {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if num_threads > 1 {
        Executor::multi_thread(num_threads, name)
    } else {
        Ok(Executor::single_thread())
    }
}

/// Per-field IVF build timings (one phase per field), emitted at end of build
/// as a parseable `log::info!` line on target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    train: Duration,
    assign: Duration,
    posting_write: Duration,
    quantize: Duration,
}

struct QuantizedLayerSlots {
    codes: Vec<u8>,
    scales: Vec<u16>,
    constants: Vec<f32>,
}

fn quantization_runtime(
    config: &VectorQuantizationConfig,
    opts: &VectorOptions,
) -> crate::Result<(Vec<LayerSpec>, Vec<Grid>)> {
    config.validate(opts)?;
    let specs = config
        .layers
        .iter()
        .map(|layer| LayerSpec {
            bits: layer.bits,
            seed: layer.seed,
            rotate: true,
        })
        .collect();
    let grids = config
        .layers
        .iter()
        .map(|layer| {
            let modeled = build_grid(config.dim, layer.bits);
            match layer.quantizer {
                // The sign encoder does not consume grid points. This format-shaped
                // placeholder keeps the width-generic cascade API allocation-free.
                VectorQuantizer::RaBitQ => Grid {
                    bits: 1,
                    points: vec![-1.0, 1.0],
                    rho_model: modeled.rho_model,
                },
                VectorQuantizer::TurboQuant => {
                    let grid = config
                        .grids
                        .iter()
                        .find(|grid| grid.bits == layer.bits)
                        .expect("validated TurboQuant grid must be present");
                    Grid {
                        bits: grid.bits,
                        points: grid.points.clone(),
                        rho_model: modeled.rho_model,
                    }
                }
            }
        })
        .collect();
    Ok((specs, grids))
}

fn write_quantized_slots(
    vec_write: &mut CompositeWrite,
    field: Field,
    layers: &[QuantizedLayerSlots],
    residual_norms: Option<&[f32]>,
) -> crate::Result<()> {
    for (layer, encoded) in layers.iter().enumerate() {
        vec_write.align_next_field(QUANTIZED_CODE_ALIGNMENT, HEADER_LEN)?;
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_codes(layer));
            writer.write_all(&encoded.codes)?;
            writer.flush()?;
        }
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_scales(layer));
            for &scale in &encoded.scales {
                writer.write_all(&scale.to_le_bytes())?;
            }
            writer.flush()?;
        }
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_constants(layer));
            for &constant in &encoded.constants {
                writer.write_all(&constant.to_le_bytes())?;
            }
            writer.flush()?;
        }
    }
    if let Some(residual_norms) = residual_norms {
        let writer = vec_write.for_field_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
        for &residual_norm in residual_norms {
            writer.write_all(&residual_norm.to_le_bytes())?;
        }
        writer.flush()?;
    }
    Ok(())
}

/// Write `field`'s slots in both composites as an empty IVF field: empty
/// Explicit id-map, empty rows, zero centroids, zero docs, and a single
/// zero cluster offset. Every vector field must own its slots in every
/// IVF segment — the reader treats a missing slot as corruption — so both
/// the "sources report no vectors" fast path and the "every vector-bearing
/// doc is deleted" path write this same shape.
fn write_empty_field_slots(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    field: Field,
    opts: &VectorOptions,
    quantization: Option<&VectorQuantizationConfig>,
) -> crate::Result<()> {
    // `.vec`: empty Explicit id-map + empty rows.
    {
        let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
        IdMap::serialize_explicit(&[], id_map_w)?;
        id_map_w.flush()?;
    }
    {
        let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
        rows_w.flush()?;
    }
    if let Some(config) = quantization {
        let layers: Vec<QuantizedLayerSlots> = config
            .layers
            .iter()
            .map(|_| QuantizedLayerSlots {
                codes: Vec::new(),
                scales: Vec::new(),
                constants: Vec::new(),
            })
            .collect();
        let residual_norms = config.needs_residual_norm().then_some([].as_slice());
        write_quantized_slots(vec_write, field, &layers, residual_norms)?;
    }
    // `.centroids`: zero centroids, zero docs, single zero offset, and an
    // empty (but present — the slot is mandatory in V2) bounds slot.
    {
        let centroids_w = centroids_write.for_field_with_idx(field, centroid_slot::CENTROIDS);
        IvfIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let offsets_w = centroids_write.for_field_with_idx(field, centroid_slot::OFFSETS);
        IvfIndex::serialize_offsets(&[0u64], offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, centroid_slot::BOUNDS);
        IvfIndex::serialize_bounds(BoundKind::Ball, &[], bounds_w)?;
        bounds_w.flush()?;
    }
    Ok(())
}

pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: Option<&dyn IvfClusterer>,
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
    write_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_file = directory.open_write(&centroids_path)?;
    write_header(&mut centroids_file)?;
    let mut centroids_write = CompositeWrite::wrap(centroids_file);

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
        // Per-segment readers for this field (cached on the SegmentReaders).
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
            write_empty_field_slots(
                &mut vec_write,
                &mut centroids_write,
                field,
                opts,
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
        let mut current_cluster = usize::MAX;
        let mut current_centroid: Vec<f32> = Vec::new();

        match opts.dtype() {
            VectorDType::F32 => {
                let field_build_start = Instant::now();
                let mut timings = IvfBuildTimings::default();
                let mut training_values = Vec::with_capacity(training_sample_size * opts.dim());
                let mut training_doc_ids = Vec::with_capacity(training_sample_size);
                let mut target_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let reader = &field_readers[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            training_values
                                .extend_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                // Rows written for docs deleted afterwards still count toward
                // the sources' `count()` (neither layout rewrites `.vec` on
                // delete), so the alive-doc walk can come up short of
                // `vector_count` — never over. Equality holds exactly when no
                // source carries deletes.
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
                    write_empty_field_slots(
                        &mut vec_write,
                        &mut centroids_write,
                        field,
                        opts,
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
                let centroids = clusterer.train(opts, training_vectors)?;

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

                // Optional router (slot [2]). Default clusterers return `None`
                // and the merge builds a routing RNG. When present, a stacked
                // build's canonical cluster-sort permutation is applied to the
                // trained centroid matrix before assign.
                let built_router = clusterer.build_router(opts, &centroids)?;
                let centroids = match &built_router {
                    Some(BuiltRouter::Stacked { perm, .. }) => {
                        let IvfCentroids::F32(matrix) = &centroids;
                        if perm.len() != num_centroids {
                            return Err(TantivyError::InvalidArgument(format!(
                                "build_router returned a permutation over {} centroids, expected \
                                 {num_centroids}",
                                perm.len()
                            )));
                        }
                        let dims = matrix.dims;
                        let mut values = vec![0.0f32; matrix.values.len()];
                        let mut seen = vec![false; num_centroids];
                        for (old, &new) in perm.iter().enumerate() {
                            let new = new as usize;
                            if new >= num_centroids || seen[new] {
                                return Err(TantivyError::InvalidArgument(
                                    "build_router permutation is not a bijection".to_string(),
                                ));
                            }
                            seen[new] = true;
                            values[new * dims..(new + 1) * dims]
                                .copy_from_slice(&matrix.values[old * dims..(old + 1) * dims]);
                        }
                        IvfCentroids::F32(IvfMatrix {
                            values,
                            rows: matrix.rows,
                            dims,
                        })
                    }
                    _ => centroids,
                };
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
                            // Poll for cancellation once per batch so a large
                            // assign phase (minutes on a big segment) stays
                            // interruptible instead of only checking at phase
                            // boundaries.
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
                    for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                        let reader = &field_readers[old_doc_addr.segment_ord as usize];
                        if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                            batch_doc_ids.push(target_doc_id);
                            batch_values.extend_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                            batch_sources.push((
                                target_doc_id,
                                old_doc_addr.segment_ord as usize,
                                old_doc_addr.doc_id,
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
                // Same alive-doc walk as the training pass above — it must
                // have found the same vectors (deletes make both fall short
                // of `vector_count` together, so compare them to each other).
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

                // `.centroids` slot [0] payload, built BEFORE the posting
                // rows so the bounds fold below measures residuals against
                // the STORED centroid. K-means cluster means are not
                // unit-norm; for Cosine+F32 normalize each centroid here so
                // the search path can score both docs and centroids with
                // the same `dot * inv_norm_q` fast kernel.
                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                // P1: `BoundsBuilder` is the ONLY producer of bounds. The
                // fold runs over THIS merge's re-assignment output against
                // the NEW centroids — combining the sources' stored bounds
                // would be unsound (their centroids no longer exist), which
                // is why no bound-combining API exists.
                let mut bounds_builder = BoundsBuilder::new(num_centroids);
                for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    // Centroids are means of ingest-validated rows, so
                    // NonFinite is should-never-happen; same warn-and-write
                    // policy as the posting rows below.
                    let outcome = maybe_normalize_bytes(opts, &mut bytes);
                    if outcome == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite centroid {centroid_ord} in field '{}' written \
                             un-normalized during merge",
                            entry.name(),
                        );
                    }
                    let stored = decode_row::<f32>(&bytes, opts.dim())?;
                    // A degenerate centroid — non-finite, or zero-norm under
                    // cosine renormalization — anchors no residual geometry:
                    // SATURATE, so the cluster always probes. (A non-finite
                    // centroid would also self-saturate through non-finite
                    // residuals, but only if the cluster has members.)
                    if outcome != NormalizeOutcome::Normalized
                        || stored.iter().any(|value| !value.is_finite())
                    {
                        bounds_builder.saturate(centroid_ord);
                    }
                    centroid_bytes.extend_from_slice(&bytes);
                }

                let posting_start = Instant::now();
                // `.vec` slot [0]: the row→doc_id permutation (Explicit), in
                // cluster-sorted row order — parallel to the rows in slot [1].
                {
                    let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
                    let row_doc_ids: Vec<DocId> = assigned_vectors
                        .iter()
                        .map(|assigned_vector| assigned_vector.target_doc_id)
                        .collect();
                    IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
                    id_map_w.flush()?;
                }

                // `.vec` slot [1]: the cluster-sorted vector rows.
                {
                    // Poll for cancellation every this-many rows during the
                    // posting-write phase — often enough to stay responsive,
                    // rare enough to keep the FFI cancel check off the per-row
                    // path.
                    const CANCEL_POLL_ROWS: usize = 4096;
                    let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
                    let needs_norm = opts.needs_normalization();
                    let mut row_buf: Vec<u8> = Vec::with_capacity(opts.bytes_per_vector());
                    for (row_idx, assigned_vector) in assigned_vectors.iter().enumerate() {
                        if row_idx % CANCEL_POLL_ROWS == 0 && ctx.cancel.wants_cancel() {
                            return Err(TantivyError::Cancelled);
                        }
                        let reader = &field_readers[assigned_vector.source_segment_ord];
                        let bytes = reader
                            .vector_bytes(assigned_vector.source_doc_id)?
                            .ok_or_else(|| {
                                TantivyError::InternalError(format!(
                                    "missing source vector for doc {:?}",
                                    assigned_vector.source_doc_id
                                ))
                            })?;
                        // Sources are already unit-normalized at ingest for
                        // Cosine+F32 (see `FlatVecWriter`), but re-normalize on
                        // the way into the cluster rows so the IVF invariant —
                        // the query path scores pre-normalized rows — holds
                        // locally, even for a source segment written before
                        // ingest-time normalization existed. Idempotent. L2/Dot
                        // don't normalize and write the source bytes directly;
                        // Cosine+F32 copies into one buffer reused across rows.
                        //
                        // Ingest rejects non-finite vectors, so NonFinite here
                        // is a should-never-happen path: erroring would wedge
                        // merge retries forever on one poison doc, and dropping
                        // the row would desync the already-computed assignments
                        // and IdMap. Warn-and-write-as-is is visible,
                        // self-limiting, and non-desyncing.
                        let written_bytes: &[u8] = if needs_norm {
                            row_buf.clear();
                            row_buf.extend_from_slice(&bytes);
                            if maybe_normalize_bytes(opts, &mut row_buf)
                                == NormalizeOutcome::NonFinite
                            {
                                log::warn!(
                                    "non-finite vector in field '{}' (doc {}) written \
                                     un-normalized during merge",
                                    entry.name(),
                                    assigned_vector.target_doc_id,
                                );
                            }
                            &row_buf
                        } else {
                            &bytes
                        };
                        rows_w.write_all(written_bytes)?;
                        // P1: the bounds fold — every written row, using the exact
                        // bytes written above against the stored centroid. A
                        // non-finite row residual saturates its cluster
                        // inside `add_native`.
                        if assigned_vector.cluster != current_cluster {
                            current_cluster = assigned_vector.cluster;
                            current_centroid = decode_row::<f32>(
                                &centroid_bytes[current_cluster * centroid_stride..]
                                    [..centroid_stride],
                                opts.dim(),
                            )?;
                        }
                        bounds_builder.add_native(
                            assigned_vector.cluster,
                            residual(written_bytes, &current_centroid),
                        );
                    }
                    rows_w.flush()?;
                }
                timings.posting_write = posting_start.elapsed();

                if let Some(config) = quantization {
                    let quantize_start = Instant::now();
                    let (specs, grids) = quantization_runtime(config, opts)?;
                    let num_rows = assigned_vectors.len();
                    let mut encoded_layers: Vec<QuantizedLayerSlots> = config
                        .layers
                        .iter()
                        .map(|layer| QuantizedLayerSlots {
                            codes: Vec::with_capacity(
                                num_rows * quantized_code_stride(opts.dim(), layer.bits),
                            ),
                            scales: Vec::with_capacity(num_rows),
                            constants: Vec::with_capacity(num_rows),
                        })
                        .collect();
                    let mut residual_norms = config
                        .needs_residual_norm()
                        .then(|| Vec::with_capacity(num_rows));

                    // Two row-major f32 tile buffers live in the cascade
                    // encoder. Rotation adds one d-sized transient buffer.
                    const MAX_QUANTIZATION_SCRATCH_BYTES: usize = 1 << 20;
                    let row_bytes = opts.dim() * std::mem::size_of::<f32>();
                    let tile_rows = MAX_QUANTIZATION_SCRATCH_BYTES
                        .saturating_sub(row_bytes)
                        .checked_div(2 * row_bytes)
                        .unwrap_or(0)
                        .max(1);
                    let needs_norm = opts.needs_normalization();
                    let mut normalized = Vec::with_capacity(opts.bytes_per_vector());
                    let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
                    for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        if start == end {
                            continue;
                        }
                        let centroid = decode_row::<f32>(
                            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                            opts.dim(),
                        )?;
                        let prepared = prepare_centroid(&centroid, &specs);
                        for tile in assigned_vectors[start..end].chunks(tile_rows) {
                            if ctx.cancel.wants_cancel() {
                                return Err(TantivyError::Cancelled);
                            }
                            batch_values.clear();
                            for assigned_vector in tile {
                                let reader = &field_readers[assigned_vector.source_segment_ord];
                                let bytes = reader
                                    .vector_bytes(assigned_vector.source_doc_id)?
                                    .ok_or_else(|| {
                                        TantivyError::InternalError(format!(
                                            "missing source vector for doc {:?}",
                                            assigned_vector.source_doc_id
                                        ))
                                    })?;
                                let encoded_bytes: &[u8] = if needs_norm {
                                    normalized.clear();
                                    normalized.extend_from_slice(&bytes);
                                    if maybe_normalize_bytes(opts, &mut normalized)
                                        == NormalizeOutcome::NonFinite
                                    {
                                        log::warn!(
                                            "non-finite vector in field '{}' (doc {}) encoded \
                                             un-normalized during merge",
                                            entry.name(),
                                            assigned_vector.target_doc_id,
                                        );
                                    }
                                    &normalized
                                } else {
                                    &bytes
                                };
                                batch_values.extend_from_slice(&decode_row::<f32>(
                                    encoded_bytes,
                                    opts.dim(),
                                )?);
                            }
                            if let Some(residual_norms) = residual_norms.as_mut() {
                                residual_norms.extend(
                                    batch_values
                                        .chunks_exact(opts.dim())
                                        .map(|row| l2_squared(row, &centroid)),
                                );
                            }
                            let batch = encode_batch_in_place(
                                &mut batch_values,
                                tile.len(),
                                &prepared,
                                &specs,
                                &grids,
                            );
                            for (target, layer) in encoded_layers.iter_mut().zip(batch.layers) {
                                target.codes.extend_from_slice(&layer.codes);
                                target.scales.extend_from_slice(&layer.scales);
                                target.constants.extend_from_slice(&layer.constants);
                            }
                        }
                    }
                    write_quantized_slots(
                        &mut vec_write,
                        field,
                        &encoded_layers,
                        residual_norms.as_deref(),
                    )?;
                    timings.quantize = quantize_start.elapsed();
                }

                {
                    let centroids_w =
                        centroids_write.for_field_with_idx(field, centroid_slot::CENTROIDS);
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
                        centroids_write.for_field_with_idx(field, centroid_slot::OFFSETS);
                    IvfIndex::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }
                // `.centroids` slot [3]: the per-cluster centroid bounds
                // this merge's fold produced.
                {
                    let bounds_w = centroids_write.for_field_with_idx(field, centroid_slot::BOUNDS);
                    IvfIndex::serialize_bounds(
                        BoundKind::Ball,
                        &bounds_builder.finish(),
                        bounds_w,
                    )?;
                    bounds_w.flush()?;
                }

                // `.centroids` slot [2]: the tagged router. Skipped when no
                // clusterer router was built and `num_centroids <= 1`.
                if num_centroids > 1 {
                    if ctx.cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    let router_w = centroids_write.for_field_with_idx(field, centroid_slot::ROUTER);
                    match built_router.as_ref() {
                        Some(BuiltRouter::Stacked { index, .. }) => {
                            RoutingIndex::serialize_stacked(CURRENT, index, router_w)?;
                        }
                        Some(BuiltRouter::Graph(graph)) => {
                            RoutingIndex::serialize_graph(CURRENT, graph, router_w)?;
                        }
                        None => {
                            let mut rng = RelativeNeighborhoodGraph::new(
                                centroid_matrix.values.as_slice(),
                                opts.dim(),
                                opts.metric(),
                                NeighborhoodGraphConfig::default(),
                            );
                            rng.build(&build_executor("rng-build-")?);
                            RoutingIndex::serialize_graph(CURRENT, &rng, router_w)?;
                        }
                    }
                    router_w.flush()?;
                }

                log::info!(
                    target: "paradedb::ivf_build",
                    "ivf_build timings_ms train={} assign={} posting_write={} quantize={} total={} \
                     centroids={} vectors={}",
                    timings.train.as_millis(),
                    timings.assign.as_millis(),
                    timings.posting_write.as_millis(),
                    timings.quantize.as_millis(),
                    field_build_start.elapsed().as_millis(),
                    num_centroids,
                    vector_count,
                );
            }
        }
    }

    vec_write.close()?;
    centroids_write.close()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;

    use common::{BinarySerializable, VInt};

    use super::*;
    use crate::directory::{CompositeFile, DirectoryClone, FileSlice, TerminatingWrite};
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::header::read_header;
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::prepared::{QuantizedIndexCtx, QuantizedQueryCtx};
    use crate::vector::tests::ground_truth;
    use crate::vector::{
        TopDocsByVectorSimilarity, VectorCalibrationMeasurements,
        VectorQuantizationCalibrationSource, VectorQuantizationLayer,
    };
    use crate::{Index, TantivyDocument};

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
                Metric::Cosine => {
                    let mut values = vec![0.0; 2 * self.dim];
                    values[0] = 1.0;
                    values[self.dim + 1] = 1.0;
                    values
                }
                Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
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
                    Metric::Cosine => u32::from(row[1] > row[0]),
                    Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
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
                    quantizer: if bits == 1 {
                        VectorQuantizer::RaBitQ
                    } else {
                        VectorQuantizer::TurboQuant
                    },
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
            Metric::Cosine => {
                let cluster = usize::from(doc >= 4);
                let mut vector: Vec<f32> = (0..dim)
                    .map(|coordinate| ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.025)
                    .collect();
                vector[cluster] += 1.0;
                vector
            }
            Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
        }
    }

    fn fixture_calibration_queries_for(metric: Metric, dim: usize) -> Vec<Vec<f32>> {
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

    fn fixture_calibration_queries(dim: usize) -> Vec<Vec<f32>> {
        fixture_calibration_queries_for(Metric::L2, dim)
    }

    fn persist_fixture_calibration_for(
        mut index: Index,
        dim: usize,
        metric: Metric,
    ) -> crate::Result<Index> {
        let reader = index.reader()?;
        reader.reload().map_err(|error| {
            TantivyError::InternalError(format!("calibration fixture reload failed: {error}"))
        })?;
        let field = index.schema().get_field("embedding")?;
        let queries = fixture_calibration_queries_for(metric, dim);
        let mut measurements = VectorCalibrationMeasurements::default();
        for segment in reader.searcher().segment_readers() {
            if let Some(segment_measurements) = segment
                .vector_index(field)?
                .calibrate_external_queries(&queries, 1_000, segment.alive_bitset())
                .map_err(|error| {
                    TantivyError::InternalError(format!(
                        "calibration fixture measurement failed: {error}"
                    ))
                })?
            {
                measurements.merge(&segment_measurements)?;
            }
        }
        let calibration = measurements.finish(VectorQuantizationCalibrationSource::RealQuery)?;
        let previous = index.load_metas()?;
        let mut updated = index.load_metas()?;
        updated.index_settings.vector_quantization[0]
            .install_real_query_calibration(calibration)?;
        crate::indexer::segment_updater::save_metas(&updated, &previous, index.directory())?;
        let reopened = Index::open(index.directory().box_clone()).map_err(|error| {
            TantivyError::InternalError(format!("calibration fixture reopen failed: {error}"))
        })?;
        assert_eq!(
            reopened.settings().vector_quantization[0].calibration(),
            updated.index_settings.vector_quantization[0].calibration()
        );
        *index.settings_mut() = reopened.settings().clone();
        Ok(index)
    }

    fn build_quantized_fixture_with_calibration(
        dim: usize,
        quantized: bool,
        calibrated: bool,
    ) -> crate::Result<Index> {
        build_quantized_fixture_case_with_calibration(
            dim,
            Metric::L2,
            &[1, 4],
            quantized,
            calibrated,
        )
    }

    fn build_quantized_fixture_case_with_calibration(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
        calibrated: bool,
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
            .ivf_clusterer(Arc::new(QuantFixtureClusterer {
                dim,
                metric,
            }))
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
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
            }
        }
        let mut segments = index.searchable_segment_ids()?;
        segments.sort();
        writer.merge(&segments).wait()?;
        writer.wait_merging_threads()?;
        if quantized && calibrated {
            persist_fixture_calibration_for(index, dim, metric)
        } else {
            Ok(index)
        }
    }

    fn build_quantized_fixture(dim: usize, quantized: bool) -> crate::Result<Index> {
        build_quantized_fixture_with_calibration(dim, quantized, true)
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

    fn fixture_hits(index: &Index, query: Vec<f32>) -> crate::Result<Vec<(u32, u32)>> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let collector = TopDocsByVectorSimilarity::new(field, query, 3).with_adaptive_params(
            AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            },
        );
        Ok(searcher
            .search(&AllQuery, &collector)?
            .results
            .iter()
            .map(|&(score, address)| (score.to_bits(), address.doc_id))
            .collect())
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
            Metric::Cosine => {
                let mut query: Vec<f32> = (0..dim)
                    .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos() * 0.01)
                    .collect();
                query[0] += 0.8;
                query[1] += 0.6;
                query
            }
            Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
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
                .config()
                .layers
                .iter()
                .map(|layer| layer.bits)
                .collect::<Vec<_>>(),
            schedule
        );
        assert_eq!(
            quantized.config().needs_residual_norm(),
            metric == Metric::L2
        );
        for row in 0..ivf.num_rows() {
            assert_eq!(
                quantized.residual_norm(row)?.is_some(),
                metric == Metric::L2,
                "slot 14 presence must follow the metric at row {row}"
            );
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
        let collector = TopDocsByVectorSimilarity::new(field, query, TOP_N)
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
        assert_eq!(
            stats.quantized_trace.boundary_docs.len(),
            depth,
            "{context}: one identity snapshot per executed boundary"
        );
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

    fn vec_composite_slot(
        bytes: &[u8],
        field: Field,
        slot: usize,
    ) -> crate::Result<Option<Vec<u8>>> {
        let (_, body) = read_header(&FileSlice::from(bytes.to_vec()))?;
        Ok(CompositeFile::open(&body)?
            .open_read_with_idx(field, slot)
            .map(|slice| slice.read_bytes().map(|bytes| bytes.to_vec()))
            .transpose()?)
    }

    fn inject_legacy_slot15(bytes: &[u8], field: Field, payload: &[u8]) -> crate::Result<Vec<u8>> {
        let (header, body) = bytes.split_at(HEADER_LEN);
        let footer_len = u32::from_le_bytes(body[body.len() - 4..].try_into().unwrap()) as usize;
        let footer_start = body.len() - 4 - footer_len;
        let mut footer_reader = &body[footer_start..body.len() - 4];
        let entry_count = VInt::deserialize(&mut footer_reader)?.0 as usize;
        let mut entries = Vec::with_capacity(entry_count);
        let mut absolute_offset = 0_u64;
        for _ in 0..entry_count {
            absolute_offset += VInt::deserialize(&mut footer_reader)?.0;
            let entry_field = Field::deserialize(&mut footer_reader)?;
            let entry_slot = VInt::deserialize(&mut footer_reader)?.0;
            entries.push((absolute_offset, entry_field, entry_slot));
        }
        assert!(footer_reader.is_empty());

        let mut new_footer = Vec::new();
        VInt((entry_count + 1) as u64).serialize(&mut new_footer)?;
        let mut previous_offset = 0_u64;
        for &(offset, entry_field, entry_slot) in &entries {
            VInt(offset - previous_offset).serialize(&mut new_footer)?;
            entry_field.serialize(&mut new_footer)?;
            VInt(entry_slot).serialize(&mut new_footer)?;
            previous_offset = offset;
        }
        let legacy_offset = footer_start as u64;
        VInt(legacy_offset - previous_offset).serialize(&mut new_footer)?;
        field.serialize(&mut new_footer)?;
        VInt(vec_slot::QUANTIZED_CALIBRATION as u64).serialize(&mut new_footer)?;

        let mut injected = Vec::with_capacity(bytes.len() + payload.len() + new_footer.len());
        injected.extend_from_slice(header);
        injected.extend_from_slice(&body[..footer_start]);
        injected.extend_from_slice(payload);
        injected.extend_from_slice(&new_footer);
        injected.extend_from_slice(&(new_footer.len() as u32).to_le_bytes());
        Ok(injected)
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
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
        let query: Vec<f32> = (0..dim)
            .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos())
            .collect();
        let harness = cascade::prepare_split_query(&query, &specs, &grids, 4);
        let scan = QuantizedQueryCtx::new(
            QuantizedIndexCtx::resolve_from_config(quantized.config().clone()).unwrap(),
            query,
        );

        for row in 0..ivf.num_rows() {
            let mut scan_sum = 0.0;
            let mut harness_sum = 0.0;
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let codes = stored.code_bytes(row)?;
                let scale = stored.scale(row)?;
                let constant = stored.constant(row)?;
                let scan_estimate = scan.score_layer(layer, &codes, scale, constant);
                let harness_estimate =
                    harness.score_layer(layer, &codes, scale, constant, specs[layer]);
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
    fn gate_a_bridge_exactness_d768_and_d100() -> crate::Result<()> {
        assert_quantized_bridge_exactness(768)?;
        assert_quantized_bridge_exactness(100)
    }

    #[test]
    fn gate_c_level_zero_matches_unquantized_ivf() -> crate::Result<()> {
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
        let quantized_storage = quantized_searcher.segment_readers()[0].vector_index(field)?;
        let quantized_field = quantized_storage
            .quantization()
            .expect("fixture must carry quantized slots");
        assert!(!quantized_field.index_ctx_is_initialized());
        let level_zero_fruit = quantized_searcher.search(&AllQuery, &level_zero_collector)?;
        assert_eq!(level_zero_collector.cached_quantized_query_count(), 0);
        assert!(!quantized_field.index_ctx_is_initialized());

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
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
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
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let expected = cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
                assert_eq!(
                    quantized
                        .residual_norm(row)?
                        .expect("L2 fixture requires residual norm")
                        .to_bits(),
                    l2_squared(&vector, &centroid).to_bits()
                );
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    let stored_codes = stored.code_bytes(row)?;
                    assert_eq!(
                        stored_codes.len(),
                        QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                        "divisible-d V3 fixture must retain its original row stride"
                    );
                    assert_eq!(stored_codes.as_slice(), expected.codes[layer]);
                    assert_eq!(stored.scale(row)?, expected.scales[layer]);
                    assert_eq!(
                        stored.constant(row)?.to_bits(),
                        expected.constants[layer].to_bits()
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
        assert_eq!(collector.cached_quantized_query_count(), 1);
        assert!(quantized.index_ctx_is_initialized());
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
    fn quantized_top_n_fixture_matrix_matches_direct_exact_oracle() -> crate::Result<()> {
        const SCHEDULES: [[u8; 2]; 2] = [[1, 4], [1, 1]];

        for metric in [Metric::Cosine, Metric::L2] {
            for schedule in &SCHEDULES {
                // One complete metric/schedule cell carries the non-64-aligned
                // dimension through none, filter, delete, and both active prefix depths.
                let dim = if metric == Metric::L2 && schedule == &[1, 4] {
                    100
                } else {
                    QUANT_FIXTURE_DIM
                };

                let primary = build_quantized_fixture_case_with_calibration(
                    dim, metric, schedule, true, true,
                )?;
                assert_quantized_matrix_storage(&primary, metric, schedule)?;
                for scenario in [
                    QuantizedMatrixScenario::None,
                    QuantizedMatrixScenario::Filter,
                ] {
                    for depth in 1..=2 {
                        run_quantized_matrix_query(&primary, metric, schedule, scenario, depth)?;
                    }
                }

                // Tombstone one deterministic row in each primary cluster.
                let label = primary.schema().get_field("label")?;
                let mut writer: crate::IndexWriter<TantivyDocument> =
                    primary.writer_with_num_threads(1, 30_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for doc in [0, 4] {
                    writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
                }
                writer.commit()?;
                drop(writer);
                for depth in 1..=2 {
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
    fn slot15_is_not_written_and_legacy_bytes_are_ignored() -> crate::Result<()> {
        let index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_calibration_queries(QUANT_FIXTURE_DIM).remove(0);
        let baseline = fixture_hits(&index, query.clone())?;

        let original = quantized_vec_file(&index)?;
        assert_eq!(
            vec_composite_slot(&original, field, vec_slot::QUANTIZED_CALIBRATION)?,
            None,
            "the V3 writer must not emit retired slot 15"
        );

        // Shape the payload like retired metadata, but make its values
        // deliberately nonsensical. Opening and scoring must depend only on
        // settings-backed calibration and therefore ignore these bytes.
        let mut legacy_payload = 3_u32.to_le_bytes().to_vec();
        legacy_payload.extend_from_slice(&[0xA5; 28]);
        let injected = inject_legacy_slot15(&original, field, &legacy_payload)?;
        assert_eq!(
            vec_composite_slot(&injected, field, vec_slot::QUANTIZED_CALIBRATION)?,
            Some(legacy_payload)
        );

        let segment = index
            .searchable_segments()?
            .into_iter()
            .next()
            .expect("one merged fixture segment");
        let path = segment.relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
        index
            .directory()
            .delete(&path)
            .expect("delete fixture .vec before slot-15 rewrite");
        let mut writer = index.directory().open_write(&path)?;
        writer.write_all(&injected)?;
        writer.terminate()?;

        assert_eq!(fixture_hits(&index, query)?, baseline);
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
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
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
                        stored.constant(row)?.to_bits(),
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
    fn quantized_fixture_growth_matches_768_1_plus_4_ledger() -> crate::Result<()> {
        const DIM: usize = 768;
        const ROWS: usize = 8;
        let config = quant_fixture_config(DIM);
        assert_eq!(config.bytes_per_row(), 496);

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
        assert!(physical_growth <= logical_growth + 512);
        assert!(
            (logical_growth as f64 / (ROWS * DIM * 4) as f64 - 0.161_458_333_333).abs() < 1e-12
        );
        Ok(())
    }

    #[test]
    fn uncalibrated_quantized_slots_use_ivf_fp32_fallback() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_calibration(DIM, true, false)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.quantization().is_none());

        let queries = fixture_calibration_queries(DIM);
        let measurements = vector_reader
            .calibrate_external_queries(&queries, 1_000, None)?
            .expect("uncalibrated slots remain available to explicit calibration");
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == 8 * queries.len() as u64));

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
        assert!(fruit.stats[0].layers.get(0).is_none());
        assert!(fruit.stats[0].postings_row > 0);
        Ok(())
    }

    #[test]
    fn external_calibration_samples_only_live_rows() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_calibration(DIM, true, false)?;
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

        let queries = fixture_calibration_queries(DIM);
        let measurements = vector_reader
            .calibrate_external_queries(&queries, usize::MAX, Some(&alive))?
            .unwrap();
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| { depth.sample_count == (live_posting_rows * queries.len()) as u64 }));

        let bounded = vector_reader
            .calibrate_external_queries(&queries, 5, Some(&alive))?
            .unwrap();
        assert!(bounded
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == (5 * queries.len()) as u64));
        Ok(())
    }
}

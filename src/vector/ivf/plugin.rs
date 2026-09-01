//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;
use std::time::{Duration, Instant};

use super::{
    decode_row, encode_vector, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix, IvfMatrixView,
    IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, VectorDType, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{centroid_slot, vec_slot, write_header, CURRENT};
use crate::vector::router::RouterFactory;
use crate::vector::{residual_norm, BoundKind, BoundsBuilder, VEC_EXT};
use crate::{DocId, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// Per-field IVF build timings (one phase per field), emitted at end of build
/// as a parseable `log::info!` line on target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    train: Duration,
    assign: Duration,
    posting_write: Duration,
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
    router: &dyn crate::vector::Router,
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
    {
        let router_w = centroids_write.for_field_with_idx(field, centroid_slot::ROUTER);
        router.serialize(router_w)?;
        router_w.flush()?;
    }
    Ok(())
}

fn build_router(
    router_factory: &dyn RouterFactory,
    opts: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<Box<dyn crate::vector::Router>> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let shape = (matrix.rows, matrix.dims, matrix.values.len());
    let router = router_factory.build(opts, centroids)?;
    let IvfCentroids::F32(matrix) = &*centroids;
    if (matrix.rows, matrix.dims, matrix.values.len()) != shape {
        return Err(TantivyError::InvalidArgument(
            "Router changed the centroid matrix shape while building".to_string(),
        ));
    }
    let router_version = router.vector_file_version();
    if router_version != CURRENT {
        return Err(TantivyError::InvalidArgument(format!(
            "router {} requires vector file version {:?}, but this merge writes {:?}",
            router.id(),
            router_version,
            CURRENT
        )));
    }
    Ok(router)
}

pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: Option<&dyn IvfClusterer>,
    router_factory: Option<&dyn RouterFactory>,
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
    let router_factory = router_factory.ok_or_else(|| {
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
            let mut centroids = IvfCentroids::F32(IvfMatrix {
                values: Vec::new(),
                rows: 0,
                dims: opts.dim(),
            });
            let router = build_router(router_factory, opts, &mut centroids)?;
            write_empty_field_slots(
                &mut vec_write,
                &mut centroids_write,
                field,
                opts,
                router.as_ref(),
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
                    let mut centroids = IvfCentroids::F32(IvfMatrix {
                        values: Vec::new(),
                        rows: 0,
                        dims: opts.dim(),
                    });
                    let router = build_router(router_factory, opts, &mut centroids)?;
                    write_empty_field_slots(
                        &mut vec_write,
                        &mut centroids_write,
                        field,
                        opts,
                        router.as_ref(),
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

                let router = build_router(router_factory, opts, &mut centroids)?;
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
                // The `.centroids` doc count: one posting row per distinct doc.
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
                        // P1: the bounds fold — every written row (all rows
                        // are native without replication), the exact bytes
                        // written above against the stored centroid. A
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

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                let router_w = centroids_write.for_field_with_idx(field, centroid_slot::ROUTER);
                router.serialize(router_w)?;
                router_w.flush()?;

                log::info!(
                    target: "paradedb::ivf_build",
                    "ivf_build timings_ms train={} assign={} posting_write={} total={} \
                     centroids={} vectors={}",
                    timings.train.as_millis(),
                    timings.assign.as_millis(),
                    timings.posting_write.as_millis(),
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

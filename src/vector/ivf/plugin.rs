//! The IVF field build: assign rows against the index-level centroid index
//! and serialize one field's `.vec` slots.
//!
//! Both clustered write paths funnel here: the per-commit serialize
//! ([`VecWriter`](crate::vector::VecWriter)) over its in-memory buffers,
//! and the merge ([`merge_ivf`]) streaming rows out of its source
//! segments. (Indexes without a centroid index write the flat layout
//! instead — see [`flat`](crate::vector::flat).) Neither trains anything —
//! training happened wherever the consumer ran it before index creation;
//! here vectors are only assigned, with tantivy's own selector over the
//! set's stored rows.

use std::io::Write;
use std::time::{Duration, Instant};

use common::BitSet;

use super::assignments::{assign_cells, CentroidSelector};
use super::graph::RelativeNeighborhoodGraph;
use super::{decode_row, SegmentClusters};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, norm_squared_bytes_wide, NormalizeOutcome};
use crate::vector::header::{vec_slot, write_header};
use crate::vector::id_map::IdMap;
use crate::vector::ivf::centroid_index::{CentroidIndexReader, FieldCentroids};
use crate::vector::{residual_norm, BoundKind, BoundsBuilder, BoundsScope, VEC_EXT};
use crate::{DocId, Executor, TantivyError};

/// Vectors decoded and assigned per selector call. Bounds the assign
/// pass's working set; parallelism happens inside
/// [`assign_cells`](super::assign::assign_cells).
const ASSIGN_BATCH_SIZE: usize = 2048;

/// Cosine-normalized rows have unit norm up to f32 rounding; a stored
/// centroid farther from 1 than this was NOT normalized (a zero-norm
/// degenerate row) and anchors no residual geometry.
const UNIT_NORM_TOLERANCE: f64 = 1e-3;

struct AssignedVector {
    cluster: usize,
    doc_id: DocId,
    /// Caller-defined handle resolving back to the row's bytes.
    handle: u64,
    /// `true` for the primary assignment, `false` for a replica entry.
    /// The bounds fold covers native rows only (`bounds_scope = native`).
    native: bool,
}

/// Per-field IVF build timings, emitted as a parseable `log::info!` line on
/// target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    selector_build: Duration,
    assign: Duration,
    posting_write: Duration,
}

/// Everything one field's IVF build needs besides the rows themselves.
pub(crate) struct IvfFieldWriteParams<'a> {
    pub(crate) field: Field,
    pub(crate) opts: &'a VectorOptions,
    pub(crate) set: &'a FieldCentroids,
    /// The set's persisted routing graph (from the cached
    /// [`CachedCentroidIndex`](crate::vector::ivf::centroid_index::CachedCentroidIndex)),
    /// reused as the assignment selector for large sets.
    pub(crate) router: Option<
        &'a RelativeNeighborhoodGraph<crate::vector::ivf::centroid_index::UnitNormRowsArena>,
    >,
    /// Cells per vector (primary + replicas); clamped to the centroid count.
    pub(crate) replicas: usize,
    pub(crate) bounds_scope: BoundsScope,
    pub(crate) executor: &'a Executor,
    pub(crate) cancel: &'a dyn CancelSentinel,
    /// Field name, for log/error messages.
    pub(crate) field_name: &'a str,
}

/// Assign one field's rows against the set and write its `.vec` slots.
///
/// * `iterate` — walks the field's present rows in ascending target-doc order, invoking the sink
///   with `(target_doc_id, handle, row_bytes)`; called exactly once.
/// * `with_row` — resolves a `handle` back to the row's bytes for the posting-write pass, invoking
///   the sink with them.
///
/// Writes nothing (and opens no slots) when `iterate` yields no rows —
/// a field with no vectors owns no slots. Returns the number of distinct
/// docs written.
pub(crate) fn write_ivf_field(
    vec_write: &mut CompositeWrite,
    params: &IvfFieldWriteParams<'_>,
    iterate: &mut dyn FnMut(
        &mut dyn FnMut(DocId, u64, &[u8]) -> crate::Result<()>,
    ) -> crate::Result<()>,
    with_row: &mut dyn FnMut(u64, &mut dyn FnMut(&[u8]) -> crate::Result<()>) -> crate::Result<()>,
) -> crate::Result<usize> {
    let opts = params.opts;
    let dim = opts.dim();
    let num_centroids = params.set.num_centroids();
    let field_build_start = Instant::now();
    let mut timings = IvfBuildTimings::default();

    // Primary + replicas cannot exceed the distinct cells that exist.
    let cells_per_vector = params.replicas.max(1).min(num_centroids.max(1));

    let selector_start = Instant::now();
    let selector = CentroidSelector::for_set(params.set, params.router, opts, cells_per_vector)?;
    timings.selector_build = selector_start.elapsed();

    // Pass A: assign every present row, batched. Replica entries are
    // appended inline — the write below tolerates any per-doc multiplicity.
    let assign_start = Instant::now();
    let mut assigned: Vec<AssignedVector> = Vec::new();
    let mut num_present_docs = 0usize;
    {
        let mut batch_values: Vec<f32> = Vec::with_capacity(ASSIGN_BATCH_SIZE * dim);
        let mut batch_rows: Vec<(DocId, u64)> = Vec::with_capacity(ASSIGN_BATCH_SIZE);
        let mut flush = |batch_values: &mut Vec<f32>,
                         batch_rows: &mut Vec<(DocId, u64)>|
         -> crate::Result<()> {
            if batch_rows.is_empty() {
                return Ok(());
            }
            // Poll for cancellation once per batch so a large assign phase
            // stays interruptible instead of only checking at boundaries.
            if params.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            let cells = assign_cells(
                &selector,
                opts.metric(),
                dim,
                batch_values,
                cells_per_vector,
                params.executor,
            )?;
            debug_assert_eq!(cells.len(), batch_rows.len());
            for (cells, (doc_id, handle)) in cells.into_iter().zip(batch_rows.drain(..)) {
                let Some((&primary, replica_cells)) = cells.split_first() else {
                    return Err(TantivyError::InternalError(format!(
                        "assignment returned no cell for doc {doc_id} in field '{}' \
                         ({num_centroids} centroids)",
                        params.field_name,
                    )));
                };
                num_present_docs += 1;
                assigned.push(AssignedVector {
                    cluster: primary,
                    doc_id,
                    handle,
                    native: true,
                });
                for &cell in replica_cells {
                    debug_assert_ne!(cell, primary, "selector returned duplicate cells");
                    assigned.push(AssignedVector {
                        cluster: cell,
                        doc_id,
                        handle,
                        native: false,
                    });
                }
            }
            batch_values.clear();
            Ok(())
        };
        iterate(&mut |doc_id, handle, bytes| {
            batch_values.extend_from_slice(&decode_row::<f32>(bytes, dim)?);
            batch_rows.push((doc_id, handle));
            if batch_rows.len() == ASSIGN_BATCH_SIZE {
                flush(&mut batch_values, &mut batch_rows)?;
            }
            Ok(())
        })?;
        flush(&mut batch_values, &mut batch_rows)?;
    }
    timings.assign = assign_start.elapsed();
    if num_present_docs == 0 {
        // No rows ⟹ no slots: the reader's empty path.
        return Ok(0);
    }

    let mut cluster_counts = vec![0usize; num_centroids];
    for assigned_vector in &assigned {
        cluster_counts[assigned_vector.cluster] += 1;
    }
    assigned.sort_unstable_by_key(|vector| (vector.cluster, vector.doc_id));

    let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
    let mut next_offset = 0u64;
    cluster_offsets.push(next_offset);
    for cluster_count in cluster_counts {
        next_offset += cluster_count as u64;
        cluster_offsets.push(next_offset);
    }

    // The bounds fold measures residuals against the centroid index's stored centroid
    // bytes. A degenerate stored centroid — non-finite, or non-unit under
    // Cosine (a zero-norm row normalization left as-is) — anchors no
    // residual geometry: SATURATE, so the cluster always probes.
    // `BoundsBuilder` is the ONLY producer of bounds, folded over THIS
    // segment's native rows; source segments' bounds are never combined.
    let mut bounds_builder = BoundsBuilder::new(num_centroids);
    // The scope captured in the stored settings at index build. `native` —
    // fold primary assignments only — is the only variant; a future scope
    // must decide its fold here.
    let BoundsScope::Native = params.bounds_scope;
    let mut centroid_row: Vec<f32> = Vec::new();
    for cluster in 0..num_centroids {
        let bytes = params.set.centroid_bytes(cluster);
        let norm_sq = norm_squared_bytes_wide::<f32>(bytes);
        let non_finite = !norm_sq.is_finite();
        let non_unit =
            opts.needs_normalization() && (norm_sq.sqrt() - 1.0).abs() > UNIT_NORM_TOLERANCE;
        if non_finite || non_unit {
            bounds_builder.saturate(cluster);
        }
    }

    let residual: fn(&[u8], &[f32]) -> f32 = residual_norm::<f32>;
    let posting_start = Instant::now();

    // `.vec` slot [0]: the row→doc_id permutation, in cluster-sorted row
    // order — parallel to the rows in slot [1].
    {
        let id_map_w = vec_write.for_field_with_idx(params.field, vec_slot::ID_MAP);
        let row_doc_ids: Vec<DocId> = assigned.iter().map(|vector| vector.doc_id).collect();
        IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
        id_map_w.flush()?;
    }

    // `.vec` slot [1]: the cluster-sorted vector rows.
    {
        // Poll for cancellation every this-many rows during the
        // posting-write phase — often enough to stay responsive, rare
        // enough to keep the cancel check off the per-row path.
        const CANCEL_POLL_ROWS: usize = 4096;
        let rows_w = vec_write.for_field_with_idx(params.field, vec_slot::ROWS);
        let needs_norm = opts.needs_normalization();
        let mut current_cluster = usize::MAX;
        let mut row_buf: Vec<u8> = Vec::with_capacity(opts.bytes_per_vector());
        for (row_idx, assigned_vector) in assigned.iter().enumerate() {
            if row_idx % CANCEL_POLL_ROWS == 0 && params.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            with_row(assigned_vector.handle, &mut |bytes| {
                // Rows are already unit-normalized at ingest for
                // Cosine+F32, but re-normalize on the way into the
                // cluster rows so the IVF invariant — the query path
                // scores pre-normalized rows — holds locally.
                // Idempotent. L2/Dot write the source bytes directly.
                //
                // Ingest rejects non-finite vectors, so NonFinite here is
                // a should-never-happen path: erroring would wedge merge
                // retries forever on one poison doc, and dropping the row
                // would desync the already-computed assignments and
                // IdMap. Warn-and-write-as-is is visible, self-limiting,
                // and non-desyncing.
                let written_bytes: &[u8] = if needs_norm {
                    row_buf.clear();
                    row_buf.extend_from_slice(bytes);
                    if maybe_normalize_bytes(opts, &mut row_buf) == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite vector in field '{}' (doc {}) written un-normalized",
                            params.field_name,
                            assigned_vector.doc_id,
                        );
                    }
                    &row_buf
                } else {
                    bytes
                };
                rows_w.write_all(written_bytes)?;
                // The bounds fold — NATIVE rows only, the exact bytes
                // written above against the set's stored centroid. A
                // non-finite row residual saturates its cluster inside
                // `add_native`.
                if assigned_vector.native {
                    if assigned_vector.cluster != current_cluster {
                        current_cluster = assigned_vector.cluster;
                        centroid_row =
                            decode_row::<f32>(params.set.centroid_bytes(current_cluster), dim)?;
                    }
                    bounds_builder.add_native(
                        assigned_vector.cluster,
                        residual(written_bytes, &centroid_row),
                    );
                }
                Ok(())
            })?;
        }
        rows_w.flush()?;
    }
    timings.posting_write = posting_start.elapsed();

    {
        let offsets_w = vec_write.for_field_with_idx(params.field, vec_slot::OFFSETS);
        SegmentClusters::serialize_offsets(&cluster_offsets, offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = vec_write.for_field_with_idx(params.field, vec_slot::BOUNDS);
        SegmentClusters::serialize_bounds(BoundKind::Ball, &bounds_builder.finish(), bounds_w)?;
        bounds_w.flush()?;
    }
    {
        let meta_w = vec_write.for_field_with_idx(params.field, vec_slot::IVF_META);
        SegmentClusters::serialize_ivf_meta(num_present_docs, num_centroids, meta_w)?;
        meta_w.flush()?;
    }

    log::info!(
        target: "paradedb::ivf_build",
        "ivf_build timings_ms selector_build={} assign={} posting_write={} total={} replicas={} \
         centroids={} vectors={}",
        timings.selector_build.as_millis(),
        timings.assign.as_millis(),
        timings.posting_write.as_millis(),
        field_build_start.elapsed().as_millis(),
        cells_per_vector,
        num_centroids,
        num_present_docs,
    );
    Ok(num_present_docs)
}

/// Sentinel for "this source doc is not in the target" (deleted).
const DOC_DROPPED: DocId = DocId::MAX;

/// Merge one field into the clustered layout. Clustered sources merge by
/// postings WITHOUT re-assigning: each cluster's rows are concatenated
/// across the sources, remapping doc ids. Flat sources (mutable/staging
/// segments moved into this index) have no cells yet, so ONLY their rows
/// are assigned here.
///
/// The postings carry-over is sound because the index has exactly one
/// immutable centroid index: the sources assigned against the very
/// centroids this target uses, and assignment is deterministic, so
/// re-running it would reproduce these exact cells. Replica entries ride
/// along in the postings, so the per-vector k-NN never runs for them.
///
/// `old_to_new[segment_ord][doc_id]` is the target doc id, or
/// [`DOC_DROPPED`] for a doc the merge is dropping.
///
/// Returns the distinct doc count written.
fn merge_ivf_field(
    vec_write: &mut CompositeWrite,
    params: &IvfFieldWriteParams<'_>,
    field_readers: &[std::sync::Arc<crate::vector::VectorIndexReader>],
    old_to_new: &[Vec<DocId>],
    num_target_docs: u32,
) -> crate::Result<usize> {
    let field = params.field;
    let num_centroids = params.set.num_centroids();
    let cancel = params.cancel;
    let field_build_start = Instant::now();

    // Phase 0: assign the flat sources' surviving rows — the only rows
    // without cells. `(cluster, target_doc, segment_ord, row, native)`,
    // sorted by (cluster, target_doc) so the gather below can merge them
    // in with one cursor.
    let assign_start = Instant::now();
    let mut flat_entries: Vec<(u32, DocId, u32, u32, bool)> = Vec::new();
    {
        let flat_sources: Vec<usize> = field_readers
            .iter()
            .enumerate()
            .filter(|(_, reader)| reader.clusters().is_none() && reader.num_vectors() > 0)
            .map(|(ord, _)| ord)
            .collect();
        if !flat_sources.is_empty() {
            let dim = params.opts.dim();
            let cells_per_vector = params.replicas.max(1).min(num_centroids.max(1));
            let selector = CentroidSelector::for_set(
                params.set,
                params.router,
                params.opts,
                cells_per_vector,
            )?;
            let mut batch_values: Vec<f32> = Vec::with_capacity(ASSIGN_BATCH_SIZE * dim);
            let mut batch_rows: Vec<(DocId, u32, u32)> = Vec::with_capacity(ASSIGN_BATCH_SIZE);
            let mut flush = |batch_values: &mut Vec<f32>,
                             batch_rows: &mut Vec<(DocId, u32, u32)>|
             -> crate::Result<()> {
                if batch_rows.is_empty() {
                    return Ok(());
                }
                if cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                let cells = assign_cells(
                    &selector,
                    params.opts.metric(),
                    dim,
                    batch_values,
                    cells_per_vector,
                    params.executor,
                )?;
                for (cells, (target_doc, seg, row)) in cells.into_iter().zip(batch_rows.drain(..)) {
                    let Some((&primary, replica_cells)) = cells.split_first() else {
                        return Err(TantivyError::InternalError(format!(
                            "assignment returned no cell for doc {target_doc} in field '{}' \
                             ({num_centroids} centroids)",
                            params.field_name,
                        )));
                    };
                    flat_entries.push((primary as u32, target_doc, seg, row, true));
                    for &cell in replica_cells {
                        flat_entries.push((cell as u32, target_doc, seg, row, false));
                    }
                }
                batch_values.clear();
                Ok(())
            };
            for ord in flat_sources {
                let reader = &field_readers[ord];
                for row in 0..reader.num_vectors() {
                    let doc = reader.doc_id_at(row);
                    let target_doc = old_to_new[ord][doc as usize];
                    if target_doc == DOC_DROPPED {
                        continue;
                    }
                    let bytes = reader.vector_bytes_for_row(row)?;
                    batch_values.extend_from_slice(&decode_row::<f32>(&bytes, dim)?);
                    batch_rows.push((target_doc, ord as u32, row as u32));
                    if batch_rows.len() == ASSIGN_BATCH_SIZE {
                        flush(&mut batch_values, &mut batch_rows)?;
                    }
                }
            }
            flush(&mut batch_values, &mut batch_rows)?;
            flat_entries.sort_unstable_by_key(|entry| (entry.0, entry.1));
        }
    }
    let assign = assign_start.elapsed();

    // Bounds for the flat-assigned rows: the same native fold the write
    // path runs, including the degenerate-centroid saturation. Clustered
    // sources' bounds combine by max below.
    let flat_bounds: Option<Vec<f32>> = if flat_entries.is_empty() {
        None
    } else {
        let dim = params.opts.dim();
        let mut builder = BoundsBuilder::new(num_centroids);
        for cluster in 0..num_centroids {
            let bytes = params.set.centroid_bytes(cluster);
            let norm_sq = norm_squared_bytes_wide::<f32>(bytes);
            let non_finite = !norm_sq.is_finite();
            let non_unit = params.opts.needs_normalization()
                && (norm_sq.sqrt() - 1.0).abs() > UNIT_NORM_TOLERANCE;
            if non_finite || non_unit {
                builder.saturate(cluster);
            }
        }
        let mut current_cluster = usize::MAX;
        let mut centroid_row: Vec<f32> = Vec::new();
        for &(cluster, _, seg, row, native) in &flat_entries {
            if !native {
                continue;
            }
            let cluster = cluster as usize;
            if cluster != current_cluster {
                current_cluster = cluster;
                centroid_row = decode_row::<f32>(params.set.centroid_bytes(cluster), dim)?;
            }
            let bytes = field_readers[seg as usize].vector_bytes_for_row(row as usize)?;
            builder.add_native(cluster, residual_norm::<f32>(&bytes, &centroid_row));
        }
        Some(builder.finish())
    };
    // One entry per surviving posting row: (target doc, source segment,
    // source row), grouped by cluster and ascending by target doc within
    // each cluster — the layout the reader's binary search relies on.
    let mut memberships: Vec<(DocId, u32, u32)> = Vec::new();
    let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
    cluster_offsets.push(0);
    // Distinct target docs, counted across every cell a doc lives in.
    let mut seen_docs = BitSet::with_max_value(num_target_docs);
    let mut per_cluster: Vec<(DocId, u32, u32)> = Vec::new();

    let gather_start = Instant::now();
    let mut flat_cursor = 0usize;
    for cluster in 0..num_centroids {
        if cluster % 4096 == 0 && cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        per_cluster.clear();
        for (segment_ord, reader) in field_readers.iter().enumerate() {
            let Some(ivf) = reader.clusters() else {
                continue;
            };
            let Some(rows) = ivf.non_empty_cluster_range(cluster) else {
                continue;
            };
            for row in rows {
                let doc = reader.doc_id_at(row);
                let target_doc = old_to_new[segment_ord][doc as usize];
                if target_doc == DOC_DROPPED {
                    continue;
                }
                per_cluster.push((target_doc, segment_ord as u32, row as u32));
            }
        }
        while let Some(&(c, target_doc, seg, row, _)) = flat_entries.get(flat_cursor) {
            if c as usize != cluster {
                break;
            }
            per_cluster.push((target_doc, seg, row));
            flat_cursor += 1;
        }
        // Each source's run is already ascending in ITS doc ids, but the
        // target ordering interleaves segments (and a doc-id mapping need
        // not be monotonic), so sort rather than assume.
        per_cluster.sort_unstable_by_key(|entry| entry.0);
        for entry in &per_cluster {
            seen_docs.insert(entry.0);
        }
        memberships.extend_from_slice(&per_cluster);
        cluster_offsets.push(memberships.len() as u64);
    }
    let gather = gather_start.elapsed();
    let num_present_docs = seen_docs.len();
    if num_present_docs == 0 {
        // No surviving vectors: the field owns no slots, matching the
        // assignment path's empty-field behavior.
        return Ok(0);
    }

    let posting_start = Instant::now();
    {
        let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
        let row_doc_ids: Vec<DocId> = memberships.iter().map(|entry| entry.0).collect();
        IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
        id_map_w.flush()?;
    }
    {
        const CANCEL_POLL_ROWS: usize = 4096;
        let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
        for (row_idx, (_, segment_ord, row)) in memberships.iter().enumerate() {
            if row_idx % CANCEL_POLL_ROWS == 0 && cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            // Copied verbatim (flat and clustered sources alike): rows
            // were normalized at ingest, and the clustered sources'
            // centroids are the same ones this segment points at.
            let bytes = field_readers[*segment_ord as usize].vector_bytes_for_row(*row as usize)?;
            rows_w.write_all(&bytes)?;
        }
        rows_w.flush()?;
    }
    let posting_write = posting_start.elapsed();

    {
        let offsets_w = vec_write.for_field_with_idx(field, vec_slot::OFFSETS);
        SegmentClusters::serialize_offsets(&cluster_offsets, offsets_w)?;
        offsets_w.flush()?;
    }
    {
        // Bounds combine as an element-wise MAX. This is sound only
        // because the centroids are shared and immutable: every source
        // radius was measured against the same anchor this segment will
        // be probed against. (It was unsound under per-segment training,
        // where each radius pointed at a centroid that no longer
        // existed.) Deletes can leave a radius wider than the surviving
        // rows warrant — conservative, so it costs pruning, never
        // correctness.
        let mut kind = BoundKind::Ball;
        let mut combined = vec![0.0f32; num_centroids];
        for reader in field_readers {
            let Some(ivf) = reader.clusters() else {
                continue;
            };
            let bounds = ivf.bounds();
            if bounds.kind() != kind {
                return Err(TantivyError::InternalError(format!(
                    "source segments disagree on bound kind ({:?} vs {kind:?})",
                    bounds.kind()
                )));
            }
            kind = bounds.kind();
            for (cluster, slot) in combined.iter_mut().enumerate() {
                let r = bounds.ball_r(cluster);
                if r > *slot || !r.is_finite() {
                    *slot = r;
                }
            }
        }
        if let Some(flat_bounds) = &flat_bounds {
            for (slot, &r) in combined.iter_mut().zip(flat_bounds) {
                if r > *slot || !r.is_finite() {
                    *slot = r;
                }
            }
        }
        let bounds_w = vec_write.for_field_with_idx(field, vec_slot::BOUNDS);
        SegmentClusters::serialize_bounds(kind, &combined, bounds_w)?;
        bounds_w.flush()?;
    }
    {
        let meta_w = vec_write.for_field_with_idx(field, vec_slot::IVF_META);
        SegmentClusters::serialize_ivf_meta(num_present_docs, num_centroids, meta_w)?;
        meta_w.flush()?;
    }

    log::info!(
        target: "paradedb::ivf_build",
        "ivf_merge timings_ms assign={} gather={} posting_write={} total={} centroids={} rows={} \
         vectors={} flat_rows={} reassigned=false",
        assign.as_millis(),
        gather.as_millis(),
        posting_write.as_millis(),
        field_build_start.elapsed().as_millis(),
        num_centroids,
        memberships.len(),
        num_present_docs,
        flat_entries.len(),
    );
    Ok(num_present_docs)
}

/// Merge source vectors into the target segment's `.vec`, reassigning
/// every row against the index's newest centroid index.
pub(crate) fn merge_ivf(ctx: &PluginMergeContext) -> crate::Result<()> {
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

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
    if num_target_docs == 0 {
        return Ok(());
    }

    let index = ctx.target_segment.index();
    let meta = index.load_metas()?;
    // Flat segments exist to be clustered at their first merge; a merge
    // whose TARGET has nothing to cluster against is a misuse, not a
    // tier. (The flat/staging tier is bounded and never merges.)
    let Some(centroid_index) = meta.centroid_index.as_ref() else {
        return Err(TantivyError::InvalidArgument(
            "cannot merge vector segments in an index without a centroid index; flat segments \
             only merge into a clustered index"
                .to_string(),
        ));
    };
    let set_search = index.cached_centroid_index()?;
    let directory = index.directory();
    let set_reader = CentroidIndexReader::open(directory, std::path::Path::new(centroid_index))?;

    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    // Flat sources are bounded (the mutable tier); no thread pool for
    // their assignment, same as the per-commit path.
    let executor = Executor::single_thread();

    // Source doc -> target doc, built once for every field. `DOC_DROPPED`
    // marks a doc the merge is dropping (deleted, or otherwise absent
    // from the mapping).
    let mut old_to_new: Vec<Vec<DocId>> = ctx
        .readers
        .iter()
        .map(|reader| vec![DOC_DROPPED; reader.max_doc() as usize])
        .collect();
    {
        let mut target_doc_id: DocId = 0;
        for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
            old_to_new[old_doc_addr.segment_ord as usize][old_doc_addr.doc_id as usize] =
                target_doc_id;
            target_doc_id += 1;
        }
        debug_assert_eq!(target_doc_id, num_target_docs);
    }

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
            continue;
        }
        let field_centroids = set_reader.field_centroids(field, opts)?;
        let num_centroids = field_centroids.num_centroids();
        for reader in &field_readers {
            if let Some(ivf) = reader.clusters() {
                if ivf.num_clusters() != num_centroids {
                    return Err(TantivyError::InternalError(format!(
                        "source segment holds {} clusters but the centroid index holds \
                         {num_centroids}",
                        ivf.num_clusters(),
                    )));
                }
            }
        }
        // Every clustered source assigned against THIS set (there is
        // exactly one, immutable for the index's life), and assignment is
        // deterministic, so the merged postings are exactly what
        // re-assignment would produce — carry them over instead of
        // re-running a k-NN per vector. Flat sources are the exception:
        // their rows are assigned inside.
        let params = IvfFieldWriteParams {
            router: set_search
                .field_router(field)
                .and_then(|router| router.graph()),
            field,
            opts,
            set: &field_centroids,
            replicas: index.settings().vector_replicas,
            bounds_scope: index.settings().vector_bounds_scope,
            executor: &executor,
            cancel: ctx.cancel,
            field_name: entry.name(),
        };
        merge_ivf_field(
            &mut vec_write,
            &params,
            &field_readers,
            &old_to_new,
            num_target_docs,
        )?;
    }

    vec_write.close()?;
    Ok(())
}

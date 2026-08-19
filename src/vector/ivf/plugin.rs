//! The IVF field build: assign rows against the index-level centroid set
//! and serialize one field's `.vec` slots.
//!
//! From format V3 on this is the only per-segment layout, so both write
//! paths funnel here: the per-commit serialize
//! ([`VecWriter`](crate::vector::VecWriter)) over its in-memory buffers,
//! and the merge ([`merge_ivf`]) streaming rows out of its source
//! segments. Neither trains anything — training happened wherever the
//! consumer ran it before index creation; here vectors are only assigned,
//! with tantivy's own selector over the set's stored rows.

use std::io::Write;
use std::time::{Duration, Instant};

use super::assign::{assign_cells, build_executor, CentroidSelector};
use super::{decode_row, IvfIndex};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::vector::centroid_set::{CentroidSetReader, FieldCentroids};
use crate::vector::distance::{maybe_normalize_bytes, norm_squared_bytes_wide, NormalizeOutcome};
use crate::vector::header::{vec_slot, write_header};
use crate::vector::id_map::IdMap;
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
    pub(crate) set_version: u64,
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
    let centroid_values = params.set.values_f32(opts)?;
    let selector = CentroidSelector::build(opts.metric(), &centroid_values, dim, cells_per_vector)?;
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
                        "assignment returned no cell for doc {doc_id} in field \
                         '{}' ({num_centroids} centroids)",
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

    // The bounds fold measures residuals against the SET's stored centroid
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
        IvfIndex::serialize_offsets(&cluster_offsets, offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = vec_write.for_field_with_idx(params.field, vec_slot::BOUNDS);
        IvfIndex::serialize_bounds(BoundKind::Ball, &bounds_builder.finish(), bounds_w)?;
        bounds_w.flush()?;
    }
    {
        let meta_w = vec_write.for_field_with_idx(params.field, vec_slot::IVF_META);
        IvfIndex::serialize_ivf_meta(num_present_docs, num_centroids, params.set_version, meta_w)?;
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

/// Merge source vectors into the target segment's `.vec`, reassigning
/// every row against the index's newest centroid set.
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
    let newest_set = meta
        .centroid_sets
        .iter()
        .max_by_key(|set| set.version)
        .ok_or_else(|| {
            TantivyError::InvalidArgument(
                "index has vector fields but no centroid set; the centroid set must be provided \
                 at index creation via IndexBuilder::centroid_index"
                    .to_string(),
            )
        })?;
    let directory = index.directory();
    let set_reader =
        CentroidSetReader::open(directory, std::path::Path::new(&newest_set.filename))?;

    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);

    let executor = build_executor("ivf-assign-")?;

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
        // Multi-version merges are unsupported: every source segment must
        // have assigned against the set this merge assigns against.
        for reader in &field_readers {
            if let Some(source_ivf) = reader.index() {
                if source_ivf.centroid_set_version() != set_reader.version() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "segments assigned against different centroid set versions ({} vs {}); \
                         multi-version merge is not supported",
                        source_ivf.centroid_set_version(),
                        set_reader.version(),
                    )));
                }
            }
        }
        let vector_count = field_readers
            .iter()
            .map(|reader| reader.num_vectors())
            .sum::<usize>();
        if vector_count == 0 {
            continue;
        }
        let field_centroids = set_reader.field_centroids(field, opts)?;
        let params = IvfFieldWriteParams {
            field,
            opts,
            set: &field_centroids,
            set_version: set_reader.version(),
            replicas: ctx.settings.vector_replicas,
            bounds_scope: ctx.settings.vector_bounds_scope,
            executor: &executor,
            cancel: ctx.cancel,
            field_name: entry.name(),
        };
        let pack = |segment_ord: usize, doc_id: DocId| -> u64 {
            ((segment_ord as u64) << 32) | doc_id as u64
        };
        write_ivf_field(
            &mut vec_write,
            &params,
            &mut |sink| {
                let mut target_doc_id: DocId = 0;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let reader = &field_readers[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                        sink(
                            target_doc_id,
                            pack(old_doc_addr.segment_ord as usize, old_doc_addr.doc_id),
                            &bytes,
                        )?;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                Ok(())
            },
            &mut |handle, sink| {
                let segment_ord = (handle >> 32) as usize;
                let doc_id = handle as u32;
                let bytes = field_readers[segment_ord]
                    .vector_bytes(doc_id)?
                    .ok_or_else(|| {
                        TantivyError::InternalError(format!(
                            "missing source vector for doc {doc_id} in segment {segment_ord}"
                        ))
                    })?;
                sink(&bytes)
            },
        )?;
    }

    vec_write.close()?;
    Ok(())
}

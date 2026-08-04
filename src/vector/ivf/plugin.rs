//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;
use std::time::{Duration, Instant};

use super::{
    decode_row, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix, IvfMatrixView, IvfTrainingBatch,
    IvfTrainingVectors, IvfVectorBatch, IvfVectors, Sq4Centroids, CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, Metric, VectorDType, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, normalize_f32_inplace, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{centroid_slot, vec_slot, write_header};
use crate::vector::{
    residual_norm, sq4_row_norms, sq4_stride, ArenaScorer, BoundKind, BoundsBuilder, BoundsScope,
    NeighborhoodGraphConfig, NodeId, RelativeNeighborhoodGraph, Similarity, Sq4Arena, Sq4Params,
    VectorArena, Workspace, VEC_EXT,
};
use crate::{DocId, Executor, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
    /// `true` for the primary assignment, `false` for a replica entry.
    /// The bounds fold covers native rows only (`bounds_scope = native`).
    native: bool,
}

/// How a vector's `replicas - 1` non-primary cells are picked from the
/// trained centroids. Constructed once per field when `replicas > 1` and
/// queried during the assign loop. A `Graph` selector is not transient:
/// the same instance is serialized as the `.centroids` slot [2] routing
/// graph afterwards.
///
/// The graph index is a recall structure for *large* centroid sets. When
/// the whole set fits within the search's own `ef` visit budget the brute
/// scan is at most as expensive — and exact: an approximate graph over a
/// handful of points can return fewer than `knn` neighbours, silently
/// under-replicating small indexes.
enum ReplicaSelector<'a> {
    /// Exact k-NN scan over the quantized centroids (small centroid sets).
    Exact,
    /// Approximate k-NN via a [`RelativeNeighborhoodGraph`] borrowing the
    /// packed SQ4 centroid arena (large centroid sets).
    Graph(RelativeNeighborhoodGraph<Sq4Arena<'a>>),
}

/// Centroid ids of the `knn` centroids most similar to `query`, best
/// first — the exact counterpart of the graph selector's search, ranked by
/// the same [`VectorArena::similarity`] key the query-time router uses
/// (raw dot for Dot — ||q||-invariant, see [`build_centroid_graph`]). Ties
/// break on centroid id so selection is deterministic.
fn exact_nearest_centroids<A: VectorArena<Elem = f32>>(
    metric: Metric,
    arena: &A,
    num_centroids: usize,
    dim: usize,
    query: &[f32],
    knn: usize,
) -> Vec<usize> {
    let scorer = arena.scorer(metric, dim, query);
    let mut scored: Vec<(Similarity, usize)> = (0..num_centroids)
        .map(|id| (scorer.score(id as NodeId), id))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.truncate(knn);
    scored.into_iter().map(|(_, id)| id).collect()
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

/// Builds the replica-selection [`RelativeNeighborhoodGraph`] over the
/// quantized centroid `arena` with search beam `ef`. Borrows the packed
/// codes; after the assign loop the same graph is persisted as the
/// `.centroids` slot [2] routing graph.
fn build_centroid_graph<'a>(
    metric: Metric,
    arena: Sq4Arena<'a>,
    dim: usize,
    ef: usize,
) -> crate::Result<RelativeNeighborhoodGraph<Sq4Arena<'a>>> {
    // Replica cells must predict the query-time router
    // (`rank_clusters`), which ranks centroids by the field metric —
    // so the graph selects with that metric directly. For Dot,
    // centroid ranking by dot(q, c) is invariant to ||q||: a
    // v-directed query ranks cells by dot(v, c), so raw dot IS the
    // query-consistent criterion, and its magnitude bias mirrors the
    // router's. The pathological case — one dominant long centroid
    // attracting many replicas — is a cluster-balance concern, not a
    // selection concern. Must stay consistent with
    // `exact_nearest_centroids`.
    let config = NeighborhoodGraphConfig {
        ef,
        ..Default::default()
    };
    let mut rng = RelativeNeighborhoodGraph::new(arena, dim, metric, config);
    rng.build(&build_executor("replica-rng-")?);
    Ok(rng)
}

/// Per-field IVF build timings (one phase per field), emitted at end of build
/// as a parseable `log::info!` line on target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    train: Duration,
    selector_build: Duration,
    assign: Duration,
    replica_knn: Duration,
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
        IvfIndex::serialize_centroids(
            0,
            0,
            &Sq4Params::fit(&[], opts.dim()),
            &[],
            &[],
            opts,
            centroids_w,
        )?;
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
            write_empty_field_slots(&mut vec_write, &mut centroids_write, field, opts)?;
            continue;
        }
        let num_centroids = settings.num_centroids.min(vector_count);
        let training_sample_size =
            vector_count.min(num_centroids.saturating_mul(settings.training_samples_per_centroid));
        let training_sample_interval = (vector_count / training_sample_size).max(1);

        let residual: fn(&[u8], &[f32]) -> f32 = match opts.dtype() {
            VectorDType::F32 => residual_norm::<f32>,
        };
        let mut current_cluster = usize::MAX;
        let mut current_centroid: Vec<f32> = Vec::new();

        match opts.dtype() {
            VectorDType::F32 => {
                let field_build_start = Instant::now();
                let mut timings = IvfBuildTimings::default();
                let replicas = settings.replicas.max(1);
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
                    write_empty_field_slots(&mut vec_write, &mut centroids_write, field, opts)?;
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
                let trained = clusterer.train(opts, training_vectors, num_centroids)?;
                timings.train = train_start.elapsed();

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }

                let IvfCentroids::F32(mut centroid_matrix) = trained else {
                    return Err(TantivyError::InvalidArgument(
                        "IvfClusterer::train must return F32 centroids; the merge quantizes them"
                            .to_string(),
                    ));
                };
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
                if centroid_matrix.rows != num_centroids {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced {} centroids, but {num_centroids} were requested",
                        centroid_matrix.rows
                    )));
                }

                // Cosine: normalize the trained centroids BEFORE fitting
                // the quantizer — k-means cluster means are not unit-norm,
                // and the codes should reconstruct (near-)unit directions.
                // A degenerate row (zero-norm, or non-finite anywhere)
                // saturates its cluster's bound below; SQ4 cannot encode
                // non-finite values, so such rows quantize as code 0.
                let mut degenerate = vec![false; num_centroids];
                if opts.needs_normalization() {
                    for (ord, row) in centroid_matrix
                        .values
                        .chunks_exact_mut(opts.dim())
                        .enumerate()
                    {
                        if normalize_f32_inplace(row) != NormalizeOutcome::Normalized {
                            degenerate[ord] = true;
                        }
                    }
                }
                for (ord, row) in centroid_matrix.values.chunks_exact(opts.dim()).enumerate() {
                    if row.iter().any(|value| !value.is_finite()) {
                        degenerate[ord] = true;
                        log::warn!(
                            "non-finite centroid {ord} in field '{}' quantized lossily during \
                             merge; its cluster bound is saturated",
                            entry.name(),
                        );
                    }
                }

                // SQ4-compress the trained set and DROP the f32 matrix.
                // Everything downstream — replica selection, assignment,
                // the routing RNG build, the bounds fold — sees only the
                // packed codes (via `Sq4Arena` or per-row decode into
                // O(dim) scratch), so the merge's persistent centroid
                // footprint is the codes plus the 2·dim params.
                let params = Sq4Params::fit(&centroid_matrix.values, opts.dim());
                let mut codes = Vec::with_capacity(num_centroids * sq4_stride(opts.dim()));
                for row in centroid_matrix.values.chunks_exact(opts.dim()) {
                    params.quantize_row_into(row, &mut codes);
                }
                drop(centroid_matrix);
                let norms_sq = sq4_row_norms(&codes, &params);
                let centroids = IvfCentroids::Sq4(Sq4Centroids {
                    codes,
                    params,
                    rows: num_centroids,
                });
                let IvfCentroids::Sq4(sq4) = &centroids else {
                    unreachable!("constructed as Sq4 above")
                };
                let centroid_arena = Sq4Arena::new(&sq4.codes, &sq4.params, &norms_sq);

                // Fixed-k replication: pick a selector ONCE before the assign
                // loop, and only when `replicas > 1`. At `replicas == 1`
                // nothing is built or allocated — the layout stays
                // primary-only. Small centroid sets — anything the search's
                // own `ef` budget would visit wholesale anyway — use the
                // exact scan; only larger sets pay for a transient
                // neighborhood graph.
                let dim = opts.dim();
                let ef_search = (replicas * 4).max(64);
                let replica_selector = if replicas <= 1 {
                    None
                } else if num_centroids <= ef_search {
                    Some(ReplicaSelector::Exact)
                } else {
                    let build_start = Instant::now();
                    let graph =
                        build_centroid_graph(opts.metric(), centroid_arena, dim, ef_search)?;
                    timings.selector_build = build_start.elapsed();
                    Some(ReplicaSelector::Graph(graph))
                };
                // One per-query scratch reused across every graph replica
                // lookup in this field's assign loop — never per vector.
                let mut replica_ws = Workspace::new();
                // Replica cells accumulated during assign; appended as extra
                // entries AFTER the primary pass so primary membership is
                // untouched. Empty at `replicas == 1`.
                let mut replica_entries: Vec<AssignedVector> = Vec::new();

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
                            for (
                                i,
                                (cluster, (target_doc_id, source_segment_ord, source_doc_id)),
                            ) in clusters
                                .into_iter()
                                .zip(batch_sources.drain(..))
                                .enumerate()
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
                                    native: true,
                                });
                                // Fixed-k replication: take the `replicas - 1`
                                // nearest NON-primary centroids from the
                                // selector. The top-k includes the primary;
                                // drop it (build-time dedup) so a vector is
                                // never written into its primary list twice.
                                if let Some(selector) = replica_selector.as_ref() {
                                    let v = &batch_values[i * dim..(i + 1) * dim];
                                    let knn_start = Instant::now();
                                    let nearest = match selector {
                                        ReplicaSelector::Exact => exact_nearest_centroids(
                                            opts.metric(),
                                            &centroid_arena,
                                            num_centroids,
                                            dim,
                                            v,
                                            replicas,
                                        ),
                                        ReplicaSelector::Graph(graph) => {
                                            let seeds: Vec<NodeId> = (0..graph.len())
                                                .step_by((graph.len() / 8).max(1))
                                                .take(8)
                                                .map(|node| node as NodeId)
                                                .collect();
                                            graph
                                                .search(&mut replica_ws, v, &seeds, replicas)
                                                .0
                                                .into_iter()
                                                .map(|candidate| candidate.node as usize)
                                                .collect()
                                        }
                                    };
                                    timings.replica_knn += knn_start.elapsed();
                                    let mut added = 0usize;
                                    for cell in nearest {
                                        if added >= replicas - 1 {
                                            break;
                                        }
                                        if cell == cluster {
                                            continue;
                                        }
                                        replica_entries.push(AssignedVector {
                                            cluster: cell,
                                            target_doc_id,
                                            source_segment_ord,
                                            source_doc_id,
                                            native: false,
                                        });
                                        added += 1;
                                    }
                                }
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
                // The `.centroids` doc count: distinct docs assigned, captured
                // before the replica append turns `assigned_vectors` into
                // posting rows (one entry per cell a doc lives in).
                let num_present_docs = assigned_vectors.len();

                // Fixed-k replication: append the accumulated replica cells as
                // extra entries — the write path below already tolerates more
                // than one entry per vector. Cells index the trained centroids,
                // whose count is fixed here; guard anyway so a bad cell id can
                // never index out of bounds.
                for entry in replica_entries.drain(..) {
                    if entry.cluster < num_centroids {
                        assigned_vectors.push(entry);
                    }
                }

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

                // P1: `BoundsBuilder` is the ONLY producer of bounds. The
                // fold runs over THIS merge's re-assignment output against
                // the NEW centroids — combining the sources' stored bounds
                // would be unsound (their centroids no longer exist), which
                // is why no bound-combining API exists.
                let mut bounds_builder = BoundsBuilder::new(num_centroids);
                // The scope captured in the stored settings at build.
                // `native` — fold primary assignments only — is the only
                // variant; a future scope must decide its fold here.
                let BoundsScope::Native = ctx.settings.vector_bounds_scope;
                // A degenerate source centroid — non-finite, or zero-norm
                // under cosine renormalization — anchors no residual
                // geometry: SATURATE, so the cluster always probes.
                for (centroid_ord, &is_degenerate) in degenerate.iter().enumerate() {
                    if is_degenerate {
                        bounds_builder.saturate(centroid_ord);
                    }
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
                        // P1: the bounds fold — NATIVE rows only, the exact
                        // bytes written above against the RECONSTRUCTED
                        // centroid: the anchor must be the point the router
                        // scores, and the router scores the codes. For
                        // cosine the routing key cos(q, recon) is
                        // scale-invariant — anchored at the NORMALIZED
                        // reconstruction — so the decoded centroid is
                        // renormalized to match; a reconstruction that
                        // cannot be (zero-norm) anchors no chord geometry
                        // and saturates. A non-finite row residual
                        // saturates its cluster inside `add_native`.
                        if assigned_vector.native {
                            if assigned_vector.cluster != current_cluster {
                                current_cluster = assigned_vector.cluster;
                                current_centroid.resize(opts.dim(), 0.0);
                                sq4.params.decode_row_into(
                                    sq4.row(current_cluster),
                                    &mut current_centroid,
                                );
                                if opts.needs_normalization()
                                    && normalize_f32_inplace(&mut current_centroid)
                                        != NormalizeOutcome::Normalized
                                {
                                    bounds_builder.saturate(current_cluster);
                                }
                            }
                            bounds_builder.add_native(
                                assigned_vector.cluster,
                                residual(written_bytes, &current_centroid),
                            );
                        }
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
                        &sq4.params,
                        &norms_sq,
                        &sq4.codes,
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

                // `.centroids` slot [2]: the RNG over the centroids, so a query
                // can route to its nearest clusters without scanning all of
                // them. Skipped for degenerate centroid counts — the reader
                // treats the absent slot as "route by linear scan", which a
                // 0-or-1-centroid segment doesn't need a graph for.
                //
                // A graph replica selector is the same graph over the same
                // arena with the same metric, so it is serialized directly
                // instead of rebuilding. Its config differs only in `ef`,
                // which affects the query beam width, not the built edges
                // (build/refine search with a beam of
                // `max(ef, num_candidates)`, and `ef <= num_candidates` for
                // any realistic `replicas`) — the persisted graph is
                // unchanged either way.
                if num_centroids > 1 {
                    if ctx.cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    let graph_w = centroids_write.for_field_with_idx(field, centroid_slot::GRAPH);
                    match replica_selector.as_ref() {
                        Some(ReplicaSelector::Graph(graph)) => graph.serialize(graph_w)?,
                        // `replicas == 1` or the exact-selector regime: no
                        // graph exists yet, build one just for routing.
                        _ => {
                            let mut rng = RelativeNeighborhoodGraph::new(
                                centroid_arena,
                                opts.dim(),
                                opts.metric(),
                                NeighborhoodGraphConfig::default(),
                            );
                            rng.build(&build_executor("rng-build-")?);
                            rng.serialize(graph_w)?;
                        }
                    }
                    graph_w.flush()?;
                }

                log::info!(
                    target: "paradedb::ivf_build",
                    "ivf_build timings_ms train={} selector_build={} assign={} replica_knn={} \
                     posting_write={} total={} replicas={} centroids={} vectors={}",
                    timings.train.as_millis(),
                    timings.selector_build.as_millis(),
                    timings.assign.as_millis(),
                    timings.replica_knn.as_millis(),
                    timings.posting_write.as_millis(),
                    field_build_start.elapsed().as_millis(),
                    replicas,
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
    use super::*;

    /// Pins the Dot selection semantics: replica cells follow RAW dot —
    /// the query-time router's ranking — not angular order. Centroid
    /// norms are deliberately unequal so the two orderings disagree.
    #[test]
    fn dot_selector_uses_raw_dot_not_angular() {
        let centroids: Vec<f32> = vec![
            10.0, 0.0, // long, off-direction: dot 10, cosine 0.45
            0.0, 1.0, // short, near-direction: dot 2, cosine 0.89
            7.0, 7.0, // long, near-direction: dot 21, cosine 0.95
        ];
        let query = [1.0_f32, 2.0];
        let picked = exact_nearest_centroids(Metric::Dot, &centroids, 3, 2, &query, 3);
        // Raw-dot order: [7,7] (21), then [10,0] (10), then [0,1] (2).
        // Angular order would put [0,1] ahead of [10,0].
        assert_eq!(picked, vec![2, 0, 1], "must rank by raw dot");
    }
}

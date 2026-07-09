//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::cmp::Ordering;
use std::io::Write;
use std::time::{Duration, Instant};

use hnsw_rs::prelude::{DistCosine, DistL2, Hnsw};

use super::{
    decode_row, encode_vector, CentroidsMeta, IvfCentroids, IvfClusterer, IvfMatrixView,
    IvfVectorBatch, IvfVectors, CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{FieldType, Metric, VectorDType};
use crate::vector::distance::{cosine, l2_squared, maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::write_header;
use crate::vector::reader::{VectorColumnReader, VectorReader};
use crate::vector::{NeighborhoodGraphConfig, RelativeNeighborhoodGraph, VEC_EXT};
use crate::{DocId, Executor, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// How a vector's `replicas - 1` non-primary cells are picked from the
/// trained centroids. Constructed once per field when `replicas > 1`,
/// queried during the assign loop, then dropped — never serialized.
///
/// HNSW is a recall structure for *large* centroid sets. When the whole set
/// fits within the search's own `ef` visit budget the brute scan is at most
/// as expensive — and exact: a parallel-built HNSW over a handful of points
/// can come out sparse enough that `search` returns fewer than `knn`
/// neighbours, silently under-replicating small indexes.
enum ReplicaSelector {
    /// Exact k-NN scan over the trained centroids (small centroid sets).
    Exact,
    /// Approximate k-NN via a transient HNSW (large centroid sets).
    Hnsw(CentroidHnsw),
}

/// Centroid ids of the `knn` nearest centroids to `query`, nearest first —
/// the exact counterpart of [`CentroidHnsw::nearest`], same distance family
/// per metric. Ties break on centroid id so selection is deterministic.
fn exact_nearest_centroids(
    metric: Metric,
    centroid_rows: &[Vec<f32>],
    query: &[f32],
    knn: usize,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = centroid_rows
        .iter()
        .enumerate()
        .map(|(id, centroid)| {
            let d = match metric {
                // Angular distance, like `DistCosine`; handles un-normalized
                // centroids the same way.
                Metric::Cosine | Metric::Dot => 1.0 - cosine(query, centroid.as_slice()),
                // Squared L2 orders identically to L2.
                Metric::L2 => l2_squared(query, centroid.as_slice()),
            };
            (d, id)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(knn);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// Transient build-time HNSW over the trained centroids, used to pick a
/// vector's `replicas - 1` nearest non-primary cells when the centroid set
/// is large enough that an approximate index beats the brute scan (see
/// [`ReplicaSelector`]). The distance must match how the primary is
/// assigned: angular (cosine) for Cosine/Dot — both clustered in angular
/// space — and L2 for L2. `DistCosine` handles un-normalized centroids, so
/// no normalized copy is kept.
enum CentroidHnsw {
    Angular(Hnsw<'static, f32, DistCosine>),
    L2(Hnsw<'static, f32, DistL2>),
}

impl CentroidHnsw {
    fn build(metric: Metric, centroids: &[Vec<f32>]) -> Self {
        let n = centroids.len();
        // hnsw_rs caps `max_nb_connection` at 256; small centroid sets need it
        // below `n`. ef_construction/max_layer are standard build-quality knobs.
        let max_nb_connection = 24.min(n.saturating_sub(1)).max(1);
        let ef_construction = 200;
        let max_layer = 16;
        let data: Vec<(&[f32], usize)> = centroids
            .iter()
            .enumerate()
            .map(|(id, c)| (c.as_slice(), id))
            .collect();
        if matches!(metric, Metric::Cosine | Metric::Dot) {
            let hnsw = Hnsw::<f32, DistCosine>::new(
                max_nb_connection,
                n,
                max_layer,
                ef_construction,
                DistCosine {},
            );
            hnsw.parallel_insert_slice(&data);
            CentroidHnsw::Angular(hnsw)
        } else {
            let hnsw = Hnsw::<f32, DistL2>::new(
                max_nb_connection,
                n,
                max_layer,
                ef_construction,
                DistL2 {},
            );
            hnsw.parallel_insert_slice(&data);
            CentroidHnsw::L2(hnsw)
        }
    }

    /// Centroid ids of the `knn` nearest centroids to `query`, nearest first.
    fn nearest(&self, query: &[f32], knn: usize, ef: usize) -> Vec<usize> {
        let neighbours = match self {
            CentroidHnsw::Angular(h) => h.search(query, knn, ef),
            CentroidHnsw::L2(h) => h.search(query, knn, ef),
        };
        neighbours.into_iter().map(|nb| nb.d_id).collect()
    }
}

/// Per-field IVF build timings (one phase per field), emitted at end of build
/// as a parseable `log::info!` line on target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    train: Duration,
    hnsw_build: Duration,
    assign: Duration,
    replica_knn: Duration,
    posting_write: Duration,
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
    let source_readers: Vec<VectorReader> = ctx
        .readers
        .iter()
        .map(VectorReader::open)
        .collect::<crate::Result<Vec<_>>>()?;

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
    let mut centroids_write = CompositeWrite::wrap(directory.open_write(&centroids_path)?);

    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let vector_count = source_readers
            .iter()
            .map(|reader| reader.count(field))
            .sum::<crate::Result<usize>>()?;
        if vector_count == 0 {
            // `.vec`: empty Explicit id-map + empty rows.
            {
                let id_map_w = vec_write.for_field_with_idx(field, 0);
                IdMap::serialize_explicit(&[], id_map_w)?;
                id_map_w.flush()?;
            }
            {
                let rows_w = vec_write.for_field_with_idx(field, 1);
                rows_w.flush()?;
            }
            // `.centroids`: zero centroids, single zero offset.
            {
                let centroids_w = centroids_write.for_field_with_idx(field, 0);
                CentroidsMeta::serialize_centroids(0, &[], opts, centroids_w)?;
                centroids_w.flush()?;
            }
            {
                let offsets_w = centroids_write.for_field_with_idx(field, 1);
                CentroidsMeta::serialize_offsets(&[0u64], offsets_w)?;
                offsets_w.flush()?;
            }
            continue;
        }
        let columns: Vec<_> = source_readers
            .iter()
            .map(|reader| reader.open_column(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let num_centroids = settings.num_centroids.min(vector_count);
        let training_sample_size =
            vector_count.min(num_centroids.saturating_mul(settings.training_samples_per_centroid));
        let training_sample_interval = (vector_count / training_sample_size).max(1);
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
                    let column = &columns[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            training_values
                                .extend_from_slice(&decode_row::<f32>(bytes, opts.dim())?);
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert_eq!(present_vector_ord, vector_count);
                if training_doc_ids.is_empty() {
                    continue;
                }

                let training_vectors = IvfVectors::F32(IvfVectorBatch {
                    doc_ids: &training_doc_ids,
                    matrix: IvfMatrixView {
                        values: &training_values,
                        rows: training_doc_ids.len(),
                        dims: opts.dim(),
                    },
                });
                let train_start = Instant::now();
                let centroids = clusterer.train(opts, training_vectors, num_centroids)?;
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
                if centroid_matrix.rows != num_centroids {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced {} centroids, but {num_centroids} were requested",
                        centroid_matrix.rows
                    )));
                }
                // Float working copy of the trained centroids — the replica
                // HNSW indexes float rows. Encoding + Cosine normalization
                // happen at the `.centroids` write below.
                let centroid_rows: Vec<Vec<f32>> = centroid_matrix
                    .values
                    .chunks_exact(opts.dim())
                    .map(|centroid| centroid.to_vec())
                    .collect();

                // Fixed-k replication: pick a selector ONCE before the assign
                // loop, and only when `replicas > 1`. At `replicas == 1`
                // nothing is built or allocated — the layout stays
                // primary-only. Small centroid sets — anything the search's
                // own `ef` budget would visit wholesale anyway — use the
                // exact scan; only larger sets pay for a transient HNSW.
                let dim = opts.dim();
                let ef_search = (replicas * 4).max(64);
                let replica_selector = if replicas <= 1 {
                    None
                } else if num_centroids <= ef_search {
                    Some(ReplicaSelector::Exact)
                } else {
                    let hnsw_start = Instant::now();
                    let hnsw = CentroidHnsw::build(opts.metric(), &centroid_rows);
                    timings.hnsw_build = hnsw_start.elapsed();
                    Some(ReplicaSelector::Hnsw(hnsw))
                };
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
                                            &centroid_rows,
                                            v,
                                            replicas,
                                        ),
                                        ReplicaSelector::Hnsw(hnsw) => {
                                            hnsw.nearest(v, replicas, ef_search)
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
                        let column = &columns[old_doc_addr.segment_ord as usize];
                        if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                            batch_doc_ids.push(target_doc_id);
                            batch_values.extend_from_slice(&decode_row::<f32>(bytes, opts.dim())?);
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
                debug_assert_eq!(assigned_vectors.len(), vector_count);

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

                let posting_start = Instant::now();
                // `.vec` slot [0]: the row→doc_id permutation (Explicit), in
                // cluster-sorted row order — parallel to the rows in slot [1].
                {
                    let id_map_w = vec_write.for_field_with_idx(field, 0);
                    let row_doc_ids: Vec<DocId> = assigned_vectors
                        .iter()
                        .map(|assigned_vector| assigned_vector.target_doc_id)
                        .collect();
                    IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
                    id_map_w.flush()?;
                }

                // `.vec` slot [1]: the cluster-sorted vector rows.
                {
                    let rows_w = vec_write.for_field_with_idx(field, 1);
                    let needs_norm = opts.needs_normalization();
                    let mut row_buf: Vec<u8> = Vec::with_capacity(opts.bytes_per_vector());
                    for assigned_vector in &assigned_vectors {
                        let column = &columns[assigned_vector.source_segment_ord];
                        let bytes = column
                            .vector_bytes_at(assigned_vector.source_doc_id)
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
                        if needs_norm {
                            row_buf.clear();
                            row_buf.extend_from_slice(bytes);
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
                            rows_w.write_all(&row_buf)?;
                        } else {
                            rows_w.write_all(bytes)?;
                        }
                    }
                    rows_w.flush()?;
                }
                timings.posting_write = posting_start.elapsed();

                // `.centroids`: routing — centroids in slot [0], cluster
                // offsets in slot [1]. K-means cluster means are not
                // unit-norm; for Cosine+F32 normalize each centroid here so
                // the search path can score both docs and centroids with the
                // same `dot * inv_norm_q` fast kernel.
                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    // Centroids are means of ingest-validated rows, so
                    // NonFinite is should-never-happen; same warn-and-write
                    // policy as the posting rows above.
                    if maybe_normalize_bytes(opts, &mut bytes) == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite centroid {centroid_ord} in field '{}' written \
                             un-normalized during merge",
                            entry.name(),
                        );
                    }
                    centroid_bytes.extend_from_slice(&bytes);
                }
                {
                    let centroids_w = centroids_write.for_field_with_idx(field, 0);
                    CentroidsMeta::serialize_centroids(
                        num_centroids,
                        &centroid_bytes,
                        opts,
                        centroids_w,
                    )?;
                    centroids_w.flush()?;
                }
                {
                    let offsets_w = centroids_write.for_field_with_idx(field, 1);
                    CentroidsMeta::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }

                // `.centroids` slot [2]: the RNG over the centroids, so a query
                // can route to its nearest clusters without scanning all of
                // them. Skipped for degenerate centroid counts — the reader
                // treats the absent slot as "route by linear scan".
                if num_centroids > 1 {
                    if ctx.cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    let num_threads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    let executor = if num_threads > 1 {
                        Executor::multi_thread(num_threads, "rng-build-")?
                    } else {
                        Executor::single_thread()
                    };
                    let mut rng = RelativeNeighborhoodGraph::new(
                        centroid_matrix.values.as_slice(),
                        opts.dim(),
                        opts.metric(),
                        NeighborhoodGraphConfig::default(),
                    );
                    rng.build(&executor);
                    let graph_w = centroids_write.for_field_with_idx(field, 2);
                    rng.serialize(graph_w)?;
                    graph_w.flush()?;
                }

                log::info!(
                    target: "paradedb::ivf_build",
                    "ivf_build timings_ms train={} hnsw_build={} assign={} replica_knn={} \
                     posting_write={} total={} replicas={} centroids={} vectors={}",
                    timings.train.as_millis(),
                    timings.hnsw_build.as_millis(),
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

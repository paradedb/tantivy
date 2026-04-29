use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use common::file_slice::FileSlice;
use common::OwnedBytes;
use smallvec::SmallVec;

use crate::directory::{CompositeFile, CompositeWrite};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::vector::cluster::centroid_index::CentroidIndex;
use crate::vector::cluster::kmeans::{run_kmeans_with_config, KMeansConfig};
use crate::vector::cluster::sampler::VectorSamplerFactory;
use crate::vector::math::l2_distance_sqr;
use crate::vector::rotation::{DynamicRotator, RotatorType};
use crate::vector::turboquant::TurboQuantizer;
use crate::vector::Metric;
use crate::{DocId, Segment};

pub const WINDOW_SIZE: usize = 122_880;

fn component() -> SegmentComponent {
    SegmentComponent::Custom("cluster".to_string())
}

#[derive(Clone)]
pub struct ClusterFieldConfig {
    pub field: Field,
    pub dims: usize,
    pub padded_dims: usize,
    pub metric: Metric,
    pub rotator: Arc<DynamicRotator>,
    pub rotator_seed: u64,
    /// `TurboQuantizer` used to encode per-doc records into the
    /// per-cluster batched layout in `.cluster`. Bit width, rotator
    /// seeds and codebook all live here so a query-side
    /// `TurboQuantQuery` constructed against the same parameters
    /// can decode them.
    pub quantizer: TurboQuantizer,
}

impl ClusterFieldConfig {
    pub fn new(
        field: Field,
        dims: usize,
        padded_dims: usize,
        metric: Metric,
        rotator: Arc<DynamicRotator>,
        rotator_seed: u64,
        quantizer: TurboQuantizer,
    ) -> Self {
        Self {
            field,
            dims,
            padded_dims,
            metric,
            rotator,
            rotator_seed,
            quantizer,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorFieldMeta {
    pub dims: usize,
    pub padded_dims: usize,
    pub metric: Metric,
    pub rotator_type: RotatorType,
    pub rotator_seed: u64,
    /// Bytes per TurboQuant-encoded record stored in the cluster
    /// batch_data, after the per-cluster doc-id prefix.
    pub tqvec_bytes_per_record: u32,
}

/// SPANN-style replicated assignment: assign each vector to multiple
/// nearby centroids instead of just the nearest. Recovers the recall
/// IVF normally loses on boundary docs (those whose true neighbours
/// straddle a Voronoi edge).
///
/// Algorithm per vector `p`:
///   1. Find the nearest centroid `c_min` and its distance `d_min`.
///   2. ε-cone: keep every centroid `c_i` with `dist(p, c_i) ≤ (1 + epsilon) · d_min`.
///   3. RNG prune: among the ε-cone candidates, in distance order, keep `c_i` only if no
///      already-selected `c_j` is closer to `c_i` than `p` is. Ensures replicas span different
///      directions around `p` instead of clustering on top of each other.
///   4. Cap at `max_replicas`.
///
/// Storage cost is roughly proportional to the average replication
/// factor. With `epsilon = 0.1` and well-separated clusters, most
/// interior docs end up at replica = 1 and the per-index average is
/// typically 2–3×, well below the worst-case 8×.
#[derive(Clone, Debug)]
pub struct ReplicationConfig {
    /// `ε` in `dist(p, c_i) ≤ (1 + ε) · dist(p, c_nearest)`.
    /// Typical: `0.05`–`0.15`. Larger ε → more replicas → more storage.
    pub epsilon: f32,
    /// Hard cap on replicas per vector. Default 8 (matches SPANN paper).
    pub max_replicas: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_replicas: 8,
        }
    }
}

/// Per-doc list of cluster ids the doc is assigned to. Single-assign
/// callers wrap in a 1-element `SmallVec`. Replicated callers fill up
/// to `max_replicas` entries (default 8 = inline capacity, no heap
/// allocation in the common case).
type Assignment = SmallVec<[u16; 8]>;

pub struct ClusterConfig {
    pub clustering_threshold: u32,
    pub sample_ratio: f32,
    pub sample_cap: usize,
    pub kmeans: KMeansConfig,
    pub num_clusters_fn: Arc<dyn Fn(usize) -> usize + Send + Sync>,
    pub fields: Vec<ClusterFieldConfig>,
    pub sampler_factory: Arc<dyn VectorSamplerFactory>,
    /// `None` (default) = single-assignment IVF: each vector lives in
    /// exactly one cluster. `Some(_)` = SPANN-style replicated
    /// assignment with the provided ε / cap. See [`ReplicationConfig`].
    pub replication: Option<ReplicationConfig>,
    /// When `true`, `ClusterPluginWriter::serialize` emits an empty
    /// cluster section for every field — no k-means, no encode, no
    /// transpose. The `.tqvec` file is unaffected (written by the
    /// sibling `TqVecPlugin`) so per-doc records are still persisted
    /// and queryable via the collector's unclustered fallback.
    ///
    /// Intended for bulk loaders (`CREATE INDEX` in pg_search) that
    /// produce many short-lived intermediate segments. With the
    /// records-in-cluster layout in place, "deferred" simply means
    /// "single trivial cluster on serialize"; the records are still
    /// written so a later merge can read them back via
    /// `transposed::extract_record`.
    ///
    /// After the bulk load finishes, the caller should force a
    /// single-source merge (`writer.merge(&[id])`) per surviving
    /// segment — that runs `ClusterPlugin::merge` which always
    /// k-means-clusters, populating the `.cluster` file with the
    /// proper multi-cluster layout.
    ///
    /// Default `false`.
    pub defer_clustering: bool,
}

pub struct ClusterPlugin {
    config: Arc<ClusterConfig>,
}

impl ClusterPlugin {
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

struct ClusterBatchData {
    doc_ids: Vec<DocId>,
    /// TurboQuant-encoded records, one per doc, in the same order as
    /// `doc_ids`. On-disk layout per cluster:
    /// `[doc_ids: 4 × N][tqvec_records: bpr × N]`.
    tqvec_records: Vec<u8>,
    num_batches: u32,
}

struct ClusterData {
    centroid_index: CentroidIndex,
    cluster_batch_data: Vec<ClusterBatchData>,
    num_clusters: usize,
    num_docs: usize,
    dims: usize,
}

fn sample_indices(total: usize, ratio: f32, cap: usize) -> Vec<usize> {
    let target = ((total as f32 * ratio) as usize).min(cap).max(1).min(total);
    let step = total / target;
    (0..total).step_by(step.max(1)).take(target).collect()
}

fn build_cluster_data(
    centroids: Vec<Vec<f32>>,
    _assignments: &[Assignment],
    num_docs: usize,
    metric: Metric,
) -> ClusterData {
    let num_clusters = centroids.len();
    let dims = centroids.first().map_or(0, |v| v.len());
    let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
    let centroid_index = CentroidIndex::build(centroids, centroid_ids, metric);

    ClusterData {
        centroid_index,
        cluster_batch_data: Vec::new(),
        num_clusters,
        num_docs,
        dims,
    }
}

struct ClusterResult {
    data: ClusterData,
    centroids: Vec<Vec<f32>>,
    assignments: Vec<Assignment>,
}

/// Two-pass build of `Vec<ClusterBatchData>` from raw vectors + an
/// already-decided cluster assignment:
///
/// * Pass 1: encode every doc once into a single contiguous doc-major scratch buffer (`win_num_docs
///   * bpr` bytes). One `quantizer.encode_into` call per doc, no per-doc allocation.
/// * Pass 2: bucket doc indices by cluster, then for each cluster transpose its docs into 16-doc
///   SIMD-friendly batches (`transposed::encode_batch`).
///
/// Used by both `serialize` (where assignments come from
/// `cluster_from_vectors` or the trivial-single-cluster path) and
/// `merge` (where assignments come from a different vector source —
/// the merge calls this with already-encoded records, see
/// `encode_window_into_clusters_from_records` below).
fn encode_window_into_clusters(
    win_vectors: &[Vec<f32>],
    quantizer: &TurboQuantizer,
    assignments: &[Assignment],
    num_clusters: usize,
) -> Vec<ClusterBatchData> {
    let bpr = quantizer.bytes_per_record();
    let padded_dim = quantizer.padded_dim;
    let win_num_docs = win_vectors.len();

    let mut all_records = vec![0u8; win_num_docs * bpr];
    for (local_doc_id, vec) in win_vectors.iter().enumerate() {
        quantizer.encode_into(
            vec,
            &mut all_records[local_doc_id * bpr..(local_doc_id + 1) * bpr],
        );
    }

    transpose_records_into_clusters(&all_records, bpr, padded_dim, assignments, num_clusters)
}

/// Like `encode_window_into_clusters` but takes already-encoded
/// records rather than raw f32 vectors. The merge path uses this:
/// it pulls each source doc's TurboQuant record bytes out of the
/// source segment's cluster file (via `transposed::extract_record`)
/// and feeds them here directly, avoiding a re-encode.
fn encode_window_into_clusters_from_records(
    all_records: &[u8],
    bpr: usize,
    padded_dim: usize,
    assignments: &[Assignment],
    num_clusters: usize,
) -> Vec<ClusterBatchData> {
    transpose_records_into_clusters(all_records, bpr, padded_dim, assignments, num_clusters)
}

fn transpose_records_into_clusters(
    all_records: &[u8],
    bpr: usize,
    padded_dim: usize,
    assignments: &[Assignment],
    num_clusters: usize,
) -> Vec<ClusterBatchData> {
    use crate::vector::turboquant::transposed::{self, BATCH_DOCS};

    // Replicated assignment: a single local_doc_id may appear in
    // multiple cluster_doc_ids[k] arrays. Records get duplicated into
    // each cluster's batch automatically — the inner loop reads
    // `all_records[did * bpr..]` per cluster's doc list, so the same
    // record bytes get transposed N times when N replicas exist.
    let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![Vec::new(); num_clusters];
    for (local_doc_id, cids) in assignments.iter().enumerate() {
        for &cid in cids {
            cluster_doc_ids[cid as usize].push(local_doc_id as DocId);
        }
    }

    let bb = transposed::batch_bytes(padded_dim);
    cluster_doc_ids
        .into_iter()
        .map(|doc_ids| {
            let num_docs = doc_ids.len();
            let num_batches = num_docs.div_ceil(BATCH_DOCS);
            let mut tqvec_records = vec![0u8; num_batches * bb];
            for batch_idx in 0..num_batches {
                let start = batch_idx * BATCH_DOCS;
                let end = (start + BATCH_DOCS).min(num_docs);
                let mut slot_recs: Vec<&[u8]> = Vec::with_capacity(BATCH_DOCS);
                for slot in start..end {
                    let did = doc_ids[slot] as usize;
                    slot_recs.push(&all_records[did * bpr..(did + 1) * bpr]);
                }
                let off = batch_idx * bb;
                transposed::encode_batch(&slot_recs, padded_dim, &mut tqvec_records[off..off + bb]);
            }
            ClusterBatchData {
                doc_ids,
                tqvec_records,
                num_batches: num_batches as u32,
            }
        })
        .collect()
}

/// Single-cluster window: one centroid (the elementwise mean of all
/// vectors), every doc assigned to cluster 0. Used for segments
/// below `clustering_threshold` and (eventually) for the deferred-
/// build fast path. Avoids running k-means but still produces a
/// cluster file with the same on-disk shape, which means the query
/// path has only one code path to handle and the merge path can
/// extract records uniformly.
fn cluster_as_single(vectors: &[Vec<f32>], metric: Metric) -> ClusterResult {
    let num_docs = vectors.len();
    let dims = vectors.first().map_or(0, |v| v.len());

    let mut mean = vec![0.0f32; dims];
    if !vectors.is_empty() {
        for v in vectors {
            for (m, &x) in mean.iter_mut().zip(v.iter()) {
                *m += x;
            }
        }
        let inv = 1.0 / vectors.len() as f32;
        for m in &mut mean {
            *m *= inv;
        }
    }

    let centroids = vec![mean];
    let assignments: Vec<Assignment> = (0..num_docs).map(|_| smallvec::smallvec![0u16]).collect();
    let data = build_cluster_data(centroids.clone(), &assignments, num_docs, metric);

    ClusterResult {
        data,
        centroids,
        assignments,
    }
}

fn cluster_from_vectors(
    vectors: &[Vec<f32>],
    config: &ClusterConfig,
    field_config: &ClusterFieldConfig,
) -> crate::Result<ClusterResult> {
    let num_docs = vectors.len();
    let sample_ids = sample_indices(num_docs, config.sample_ratio, config.sample_cap);
    let sampled: Vec<Vec<f32>> = sample_ids.iter().map(|&i| vectors[i].clone()).collect();

    let k = (config.num_clusters_fn)(num_docs);
    let k = k.min(sampled.len()).max(1);
    let result = run_kmeans_with_config(&sampled, k, config.kmeans.clone());

    // `assign` takes the points flat to avoid a second API arity;
    // the build path is the only caller with `Vec<Vec<f32>>`, so we
    // do the (one-time) flatten here.
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut points_flat: Vec<f32> = Vec::with_capacity(num_docs * dim);
    for v in vectors {
        points_flat.extend_from_slice(v);
    }
    let assignments = assign(
        &points_flat,
        dim,
        &result.centroids,
        config.replication.as_ref(),
    );

    let centroids = result.centroids.clone();
    let data = build_cluster_data(
        result.centroids,
        &assignments,
        num_docs,
        field_config.metric,
    );

    Ok(ClusterResult {
        data,
        centroids,
        assignments,
    })
}

/// Assign each `point` to its nearest `centroid` using triangle-inequality
/// pruning. Single-threaded by design — pg_search runs this inside a
/// PostgreSQL backend where rayon isn't safe.
///
/// Assign each point to one or more cluster centroids.
///
/// `points_flat` is `n × dim` floats laid out back-to-back. Both the
/// build path (which has `&[Vec<f32>]` from k-means input) and the
/// merge path (which already has rotated vectors in a flat buffer)
/// pass flat — the build path flattens at its call site to avoid the
/// API juggling two input shapes.
///
/// **Without replication** (`replication: None`): Elkan-style
/// triangle-inequality pruned nearest-centroid search. For each point
/// x, computes `d² = ‖x - c_j‖²` against every centroid, but skips
/// any `c_j` where the precomputed `‖c_best - c_j‖²` exceeds
/// `4·d_best²` — that bound proves `c_j` can't beat the current best.
/// On well-separated data this prunes 70–95% of the inner distance
/// comps. Returns a 1-element `Assignment` per point.
///
/// **With replication** (`replication: Some(rc)`): SPANN-style ε-cone
/// + RNG selection.
///   1. Distance from p to every centroid.
///   2. ε-cone: candidates with `d² ≤ (1+ε)² · d_min²`, sorted ascending.
///   3. Greedy RNG: keep `c_i` only if every already-selected `c_j` is farther from `c_i` than `p`
///      is. This ensures replicas span different *directions* around `p` rather than redundantly
///      clustering on top of each other.
///   4. Cap at `rc.max_replicas`.
///
/// The K×K centroid-centroid distance matrix is built once per call
/// (regardless of replication mode) and reused for triangle-inequality
/// pruning OR RNG checks. Per-point cost is O(K) distance comps plus
/// either O(1) prune-or-update (single) or O(R²) RNG checks (replicated,
/// R ≤ max_replicas ≤ 8). Both modes are single-threaded by design —
/// pg_search runs this inside a PostgreSQL backend where rayon isn't safe.
fn assign(
    points_flat: &[f32],
    dim: usize,
    centroids: &[Vec<f32>],
    replication: Option<&ReplicationConfig>,
) -> Vec<Assignment> {
    let n = if dim == 0 { 0 } else { points_flat.len() / dim };
    let k = centroids.len();
    if k == 0 {
        return vec![SmallVec::new(); n];
    }
    if k == 1 {
        return vec![smallvec::smallvec![0u16]; n];
    }

    // K×K pairwise squared centroid distances. Used both for triangle-
    // inequality pruning (single-assign fast path) and RNG checks
    // (replicated path). For K = 400 that's 160 k floats = 640 KB,
    // comfortably in L2.
    let mut cc_dist_sq = vec![0.0f32; k * k];
    for i in 0..k {
        for j in (i + 1)..k {
            let d = l2_distance_sqr(&centroids[i], &centroids[j]);
            cc_dist_sq[i * k + j] = d;
            cc_dist_sq[j * k + i] = d;
        }
    }

    // Replication is meaningful only when ε > 0 AND max_replicas > 1;
    // otherwise fall through to the single-assign fast path.
    let replication = replication.filter(|rc| rc.max_replicas > 1 && rc.epsilon > 0.0);
    let max_replicas = replication.map(|rc| rc.max_replicas.min(k)).unwrap_or(1);
    let cone_factor_sq = replication.map(|rc| (1.0 + rc.epsilon).powi(2));

    let mut assignments: Vec<Assignment> = Vec::with_capacity(n);
    let mut candidates: Vec<(f32, u16)> = Vec::new();
    if cone_factor_sq.is_some() {
        candidates.reserve(k);
    }
    let mut diag_total_cone_candidates: u64 = 0;
    let mut diag_total_replicas: u64 = 0;

    for doc_id in 0..n {
        let point = &points_flat[doc_id * dim..(doc_id + 1) * dim];

        if let Some(cone_factor_sq) = cone_factor_sq {
            // Replicated: distance to every centroid (no triangle
            // pruning since we need the full distribution to build
            // the ε-cone), then ε-cone + RNG selection.
            let mut nearest = 0u16;
            let mut d_min_sq = f32::INFINITY;
            candidates.clear();
            for (j, c) in centroids.iter().enumerate() {
                let d = l2_distance_sqr(point, c);
                if d < d_min_sq {
                    d_min_sq = d;
                    nearest = j as u16;
                }
                candidates.push((d, j as u16));
            }

            let threshold_sq = d_min_sq * cone_factor_sq;
            candidates.retain(|&(d, j)| j != nearest && d <= threshold_sq);
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            diag_total_cone_candidates += candidates.len() as u64 + 1;

            let mut selected: Assignment = smallvec::smallvec![nearest];
            for &(d_pi, ci) in &candidates {
                if selected.len() >= max_replicas {
                    break;
                }
                // RNG: keep ci only if every already-selected cj is
                // farther from ci than p is. Equivalently: skip ci if
                // any cj is closer to ci than p.
                let pass = selected
                    .iter()
                    .all(|&cj| cc_dist_sq[cj as usize * k + ci as usize] > d_pi);
                if pass {
                    selected.push(ci);
                }
            }
            diag_total_replicas += selected.len() as u64;
            assignments.push(selected);
        } else {
            // Single-assign with Elkan-style triangle pruning.
            let mut best = 0u16;
            let mut d_best_sq = l2_distance_sqr(point, &centroids[0]);
            for j in 1..k {
                let cc = cc_dist_sq[best as usize * k + j];
                if cc > 4.0 * d_best_sq {
                    continue;
                }
                let d_j_sq = l2_distance_sqr(point, &centroids[j]);
                if d_j_sq < d_best_sq {
                    d_best_sq = d_j_sq;
                    best = j as u16;
                }
            }
            assignments.push(smallvec::smallvec![best]);
        }
    }
    if cone_factor_sq.is_some() && n > 0 {
        eprintln!(
            "assign() n={} k={} eps={:.3} cap={} avg_cone_candidates={:.3} avg_replicas={:.3}",
            n,
            k,
            replication.map(|rc| rc.epsilon).unwrap_or(0.0),
            max_replicas,
            diag_total_cone_candidates as f64 / n as f64,
            diag_total_replicas as f64 / n as f64,
        );
    }
    assignments
}

fn train_centroids(
    sampler: &dyn crate::vector::cluster::sampler::VectorSampler,
    field_config: &ClusterFieldConfig,
    config: &ClusterConfig,
    num_docs: usize,
) -> crate::Result<Vec<Vec<f32>>> {
    let sample_ids = sample_indices(num_docs, config.sample_ratio, config.sample_cap);
    let sample_doc_ids: Vec<DocId> = sample_ids.iter().map(|&i| i as DocId).collect();
    let sampled_vecs = sampler.sample_vectors(field_config.field, &sample_doc_ids)?;

    let valid_vecs: Vec<Vec<f32>> = sampled_vecs.into_iter().flatten().collect();
    if valid_vecs.is_empty() {
        return Err(crate::TantivyError::InternalError(
            "no valid vectors returned by sampler".into(),
        ));
    }

    let k = (config.num_clusters_fn)(num_docs);
    let k = k.min(valid_vecs.len()).max(1);
    let result = run_kmeans_with_config(&valid_vecs, k, config.kmeans.clone());
    Ok(result.centroids)
}

fn assign_nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::INFINITY;
    for (ci, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance_sqr(vec, centroid);
        if dist < best_dist {
            best_dist = dist;
            best = ci;
        }
    }
    best
}

/// Hot section: header + centroids + batch_meta. Read on every query for
/// centroid search and to know each cluster's doc_ids / records offsets.
///
/// Per-cluster meta (20 B): num_batches, num_docs, doc_ids_offset,
/// records_offset, records_len. The doc_ids_len for cluster i is
/// implicitly `num_docs[i] * 4`.
fn serialize_cluster_hot(data: &ClusterData, w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&(data.num_clusters as u32).to_le_bytes())?;
    w.write_all(&(data.num_docs as u32).to_le_bytes())?;
    w.write_all(&(data.dims as u32).to_le_bytes())?;

    let ci_bytes = data.centroid_index.save_to_bytes()?;
    w.write_all(&(ci_bytes.len() as u32).to_le_bytes())?;
    w.write_all(&ci_bytes)?;

    // Cold layout has doc_ids region first (all clusters back-to-back),
    // then records region (all clusters back-to-back). Compute per-
    // cluster offsets so the reader can fetch each region independently.
    let total_doc_ids_bytes: u32 = data
        .cluster_batch_data
        .iter()
        .map(|b| (b.doc_ids.len() * 4) as u32)
        .sum();

    let mut doc_ids_cursor: u32 = 0;
    let mut records_cursor: u32 = total_doc_ids_bytes;
    for batch in &data.cluster_batch_data {
        let doc_ids_len = (batch.doc_ids.len() * 4) as u32;
        let records_len = batch.tqvec_records.len() as u32;
        w.write_all(&batch.num_batches.to_le_bytes())?;
        w.write_all(&(batch.doc_ids.len() as u32).to_le_bytes())?;
        w.write_all(&doc_ids_cursor.to_le_bytes())?;
        w.write_all(&records_cursor.to_le_bytes())?;
        w.write_all(&records_len.to_le_bytes())?;
        doc_ids_cursor += doc_ids_len;
        records_cursor += records_len;
    }

    Ok(())
}

/// Cold section: split into a doc_ids region followed by a records
/// region. Each region holds all clusters' bytes back-to-back in
/// cluster-id order. The two-region layout lets the query path issue
/// one coalesced read for the small (~1 KB/cluster) doc_ids prefixes,
/// intersect them against any filter bitset, and skip the much larger
/// records region (~120 KB/cluster) for clusters whose docs are all
/// filtered out.
///
/// ```text
/// [c0_doc_ids][c1_doc_ids]...[cK-1_doc_ids][c0_records][c1_records]...[cK-1_records]
/// ```
fn serialize_cluster_cold(data: &ClusterData, w: &mut dyn Write) -> crate::Result<()> {
    for batch in &data.cluster_batch_data {
        for &did in &batch.doc_ids {
            w.write_all(&did.to_le_bytes())?;
        }
    }
    for batch in &data.cluster_batch_data {
        w.write_all(&batch.tqvec_records)?;
    }
    Ok(())
}

fn serialize_empty_hot(w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&0u32.to_le_bytes())?; // num_clusters = 0
    w.write_all(&0u32.to_le_bytes())?; // num_docs = 0
    w.write_all(&0u32.to_le_bytes())?; // dims = 0
    Ok(())
}

fn serialize_field_meta(meta: &VectorFieldMeta, w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&(meta.dims as u32).to_le_bytes())?;
    w.write_all(&(meta.padded_dims as u32).to_le_bytes())?;
    w.write_all(&[match meta.metric {
        Metric::L2 => 0u8,
        Metric::InnerProduct => 1u8,
    }])?;
    w.write_all(&[meta.rotator_type as u8])?;
    w.write_all(&meta.rotator_seed.to_le_bytes())?;
    w.write_all(&meta.tqvec_bytes_per_record.to_le_bytes())?;
    Ok(())
}

fn deserialize_field_meta(data: &[u8], pos: &mut usize) -> VectorFieldMeta {
    let read_u32 = |p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };
    let dims = read_u32(pos) as usize;
    let padded_dims = read_u32(pos) as usize;
    let metric = if data[*pos] == 0 {
        Metric::L2
    } else {
        Metric::InnerProduct
    };
    *pos += 1;
    let rotator_type = RotatorType::from_u8(data[*pos]).unwrap_or(RotatorType::FhtKacRotator);
    *pos += 1;
    let rotator_seed = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    let tqvec_bytes_per_record = read_u32(pos);
    VectorFieldMeta {
        dims,
        padded_dims,
        metric,
        rotator_type,
        rotator_seed,
        tqvec_bytes_per_record,
    }
}

const FIELD_META_SIZE: usize = 4 + 4 + 1 + 1 + 8 + 4; // 22 bytes

impl VectorFieldMeta {
    fn from_config(cfg: &ClusterFieldConfig) -> Self {
        Self {
            dims: cfg.dims,
            padded_dims: cfg.padded_dims,
            metric: cfg.metric,
            rotator_type: cfg.rotator.rotator_type(),
            rotator_seed: cfg.rotator_seed,
            tqvec_bytes_per_record: cfg.quantizer.bytes_per_record() as u32,
        }
    }
}

/// Per-window directory entry size: doc_offset + num_docs + hot_size + cold_size.
const WINDOW_DIR_ENTRY_SIZE: usize = 16;

/// File layout:
///   [u32 num_windows][u32 window_size][22 B field_meta]
///   directory: [u32 doc_offset][u32 num_docs][u32 hot_size][u32 cold_size] × num_windows
///   per window in order: hot bytes, then cold bytes
///
/// Reading this lets the reader pull only each window's small hot section
/// (centroids + batch_meta) up front, leaving the much larger cold section
/// (batch_data) to be sliced lazily per probed cluster.
fn serialize_windowed_field(
    windows: &[(u32, u32, ClusterData)],
    meta: &VectorFieldMeta,
    w: &mut dyn Write,
) -> crate::Result<()> {
    let mut serialized: Vec<(u32, u32, Vec<u8>, Vec<u8>)> = Vec::with_capacity(windows.len());
    for (doc_offset, num_docs, data) in windows {
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        serialize_cluster_hot(data, &mut hot)?;
        serialize_cluster_cold(data, &mut cold)?;
        serialized.push((*doc_offset, *num_docs, hot, cold));
    }

    let num_windows = serialized.len() as u32;
    w.write_all(&num_windows.to_le_bytes())?;
    w.write_all(&(WINDOW_SIZE as u32).to_le_bytes())?;
    serialize_field_meta(meta, w)?;

    for (doc_offset, num_docs, hot, cold) in &serialized {
        w.write_all(&doc_offset.to_le_bytes())?;
        w.write_all(&num_docs.to_le_bytes())?;
        w.write_all(&(hot.len() as u32).to_le_bytes())?;
        w.write_all(&(cold.len() as u32).to_le_bytes())?;
    }

    for (_, _, hot, cold) in &serialized {
        w.write_all(hot)?;
        w.write_all(cold)?;
    }

    Ok(())
}

fn serialize_windowed_empty(
    w: &mut dyn Write,
    num_docs: usize,
    meta: &VectorFieldMeta,
) -> crate::Result<()> {
    let num_windows = num_docs.div_ceil(WINDOW_SIZE).max(1) as u32;
    w.write_all(&num_windows.to_le_bytes())?;
    w.write_all(&(WINDOW_SIZE as u32).to_le_bytes())?;
    serialize_field_meta(meta, w)?;

    // Empty windows have hot_size = 12 (just the cluster header) and cold_size = 0.
    let mut offset = 0u32;
    let mut window_doc_counts = Vec::with_capacity(num_windows as usize);
    for _ in 0..num_windows {
        let window_docs = (num_docs as u32)
            .saturating_sub(offset)
            .min(WINDOW_SIZE as u32);
        window_doc_counts.push(window_docs);
        w.write_all(&offset.to_le_bytes())?;
        w.write_all(&window_docs.to_le_bytes())?;
        w.write_all(&12u32.to_le_bytes())?; // hot_size
        w.write_all(&0u32.to_le_bytes())?; // cold_size
        offset += window_docs;
    }

    for _ in 0..num_windows {
        serialize_empty_hot(w)?;
    }

    Ok(())
}

impl SegmentPlugin for ClusterPlugin {
    fn name(&self) -> &str {
        "cluster"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["cluster"]
    }

    fn write_phase(&self) -> u32 {
        3
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(ClusterPluginWriter {
            config: self.config.clone(),
            per_field_vectors: HashMap::new(),
        }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file_slice = ctx.segment_reader.open_read(component())?;
        let composite = CompositeFile::open(&file_slice)?;
        let segment_id = ctx.segment.id();

        let mut field_readers = HashMap::new();
        for field_cfg in &self.config.fields {
            if let Some(field_slice) = composite.open_read(field_cfg.field) {
                let reader = ClusterFieldReader::open(field_slice, field_cfg.metric)?;
                field_readers.insert(field_cfg.field, reader);
            }
        }
        let _ = segment_id; // formerly used to key the hot-bytes cache

        Ok(Arc::new(ClusterPluginReader { field_readers }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let num_docs = ctx.doc_id_mapping.iter_old_doc_addrs().count();

        let cluster_write = ctx.target_segment.open_write(component())?;
        let mut cluster_composite = CompositeWrite::wrap(cluster_write);

        if num_docs == 0 {
            for field_cfg in &self.config.fields {
                let cw = cluster_composite.for_field(field_cfg.field);
                let meta = VectorFieldMeta::from_config(field_cfg);
                serialize_windowed_empty(cw, num_docs, &meta)?;
                cw.flush()?;
            }
            cluster_composite.close()?;
            return Ok(());
        }

        let sampler = self
            .config
            .sampler_factory
            .create_sampler(ctx.readers, ctx.doc_id_mapping)?;

        // Materialise the merge mapping once; the assignment loop is
        // O(num_docs) and we want O(1) lookup per doc.
        let target_to_source: Vec<crate::DocAddress> =
            ctx.doc_id_mapping.iter_old_doc_addrs().collect();

        // Open source `.cluster` readers up front. We extract per-doc
        // records out of these (via `transposed::extract_record`)
        // instead of going through the heap sampler — same idea as
        // the old tqvec fast path, just sourced from the cluster
        // file since tqvec no longer exists.
        let source_cluster_readers: Vec<Option<Arc<ClusterPluginReader>>> = ctx
            .readers
            .iter()
            .map(|r| {
                r.plugin_reader::<ClusterPluginReader>("cluster")
                    .ok()
                    .flatten()
            })
            .collect();

        for field_cfg in &self.config.fields {
            let quantizer = &field_cfg.quantizer;
            assert_eq!(
                quantizer.bit_width,
                crate::vector::turboquant::transposed::TRANSPOSED_BIT_WIDTH,
                "cluster plugin transposed layout requires bit_width = 4",
            );
            let bpr = quantizer.bytes_per_record();
            let padded_dim = quantizer.padded_dim;

            // Per-source-segment locator: source_doc_id →
            // (window_idx, cluster_id, slot_in_cluster). Built once
            // per source segment by walking that segment's clusters.
            let source_locators: Vec<Option<SourceLocator>> = source_cluster_readers
                .iter()
                .map(|reader| {
                    reader
                        .as_ref()
                        .and_then(|r| r.field_reader(field_cfg.field).map(SourceLocator::build))
                })
                .collect();

            let num_windows = num_docs.div_ceil(WINDOW_SIZE);
            let mut windows: Vec<(u32, u32, ClusterData)> = Vec::with_capacity(num_windows);

            for win_idx in 0..num_windows {
                let win_start = win_idx * WINDOW_SIZE;
                let win_end = (win_start + WINDOW_SIZE).min(num_docs);
                let win_num_docs = win_end - win_start;

                // Phase 1: extract source records for every doc in
                // the target window. One contiguous scratch buffer
                // (no per-doc Vec allocs); also captures rotated-
                // dequantized vectors used for cluster assignment.
                //
                // We resolve each target doc to its source
                // (segment, window, cluster, slot), then sort by
                // (segment, window, cluster) so consecutive docs from
                // the same source cluster process against a single
                // `cluster_batch_raw` fetch. Without grouping, a
                // 250-doc cluster gets re-fetched (and its bytes
                // re-pinned + re-copied) 250 times — pathological
                // for large windows.
                let mut all_records = vec![0u8; win_num_docs * bpr];
                let mut to_extract: Vec<DocLookup> = Vec::with_capacity(win_num_docs);
                for local_doc_id in 0..win_num_docs {
                    let global_doc_id = win_start + local_doc_id;
                    let addr = target_to_source[global_doc_id];
                    let seg_ord = addr.segment_ord as usize;
                    let Some(loc) = source_locators.get(seg_ord).and_then(|l| l.as_ref()) else {
                        continue;
                    };
                    let &(win_idx_src, cluster_id, slot) =
                        match loc.locator.get(addr.doc_id as usize) {
                            Some(triple) if triple.0 != u32::MAX => triple,
                            _ => continue,
                        };
                    to_extract.push(DocLookup {
                        local_doc_id: local_doc_id as u32,
                        seg_ord: seg_ord as u32,
                        win_idx: win_idx_src,
                        cluster_id,
                        slot,
                    });
                }
                to_extract.sort_unstable_by_key(|d| (d.seg_ord, d.win_idx, d.cluster_id));

                use crate::vector::turboquant::transposed::{self, BATCH_DOCS};
                let bb = transposed::batch_bytes(padded_dim);
                let mut i = 0;
                while i < to_extract.len() {
                    let key = (
                        to_extract[i].seg_ord,
                        to_extract[i].win_idx,
                        to_extract[i].cluster_id,
                    );
                    let start = i;
                    while i < to_extract.len()
                        && (
                            to_extract[i].seg_ord,
                            to_extract[i].win_idx,
                            to_extract[i].cluster_id,
                        ) == key
                    {
                        i += 1;
                    }

                    let loc = source_locators[key.0 as usize].as_ref().unwrap();
                    let win = loc.cluster_field.window_reader(key.1 as usize);
                    let Ok(Some((_doc_ids, _meta, raw))) = win.cluster_batch_raw(key.2 as usize)
                    else {
                        continue;
                    };
                    // raw is records-only (the new cluster_batch_raw
                    // splits doc_ids out into a separate read), so
                    // batch offsets start at 0.
                    for d in &to_extract[start..i] {
                        let batch_idx = (d.slot as usize) / BATCH_DOCS;
                        let in_batch_slot = (d.slot as usize) % BATCH_DOCS;
                        let off = batch_idx * bb;
                        if off + bb > raw.len() {
                            continue;
                        }
                        let rec_slice = &mut all_records
                            [d.local_doc_id as usize * bpr..(d.local_doc_id as usize + 1) * bpr];
                        transposed::extract_record(
                            &raw[off..off + bb],
                            in_batch_slot,
                            padded_dim,
                            rec_slice,
                        );
                    }
                }
                // Docs that we couldn't locate stay zero-filled —
                // score will be garbage but the doc count stays
                // consistent (same fallback as before).

                // One contiguous buffer holding all rotated vectors
                // back-to-back (`win_num_docs * padded_dim` f32s).
                // Was a `Vec<Vec<f32>>` — that produced ~122k small
                // allocations per window, which showed up as 25% of
                // CPU in xzm_free during a stuck CREATE INDEX.
                let mut rotated_flat = vec![0.0f32; win_num_docs * padded_dim];
                for local_doc_id in 0..win_num_docs {
                    let rec_slice = &all_records[local_doc_id * bpr..(local_doc_id + 1) * bpr];
                    let rv_slice = &mut rotated_flat
                        [local_doc_id * padded_dim..(local_doc_id + 1) * padded_dim];
                    quantizer.dequantize_into(rec_slice, rv_slice);
                }

                // Phase 2: choose target centroids. Below threshold:
                // single trivial centroid (no k-means). Above:
                // train via heap sampler (same as before — k-means
                // wants un-quantized samples for accuracy).
                let (centroids, num_clusters): (Vec<Vec<f32>>, usize) = if (win_num_docs as u32)
                    < self.config.clustering_threshold
                {
                    // Centroid value doesn't matter for a
                    // single-cluster window — query path will
                    // probe it unconditionally — so use the
                    // un-rotated zero vector.
                    (vec![vec![0.0; field_cfg.dims]], 1)
                } else {
                    let win_sampler = WindowSampler {
                        inner: sampler.as_ref(),
                        doc_offset: win_start,
                    };
                    let cs = train_centroids(&win_sampler, field_cfg, &self.config, win_num_docs)?;
                    let n = cs.len();
                    (cs, n)
                };

                // Phase 3: assign each doc to nearest target
                // centroid (in rotated space, since `rotated_flat`
                // is already in rotated coordinates). Triangle-
                // inequality pruning skips most centroid distance
                // computations once a tight upper bound is in hand;
                // brute-force here was 56% of CPU on a 1024-dim
                // bioasq merge profile.
                let assignments: Vec<Assignment> = if num_clusters == 1 {
                    (0..win_num_docs)
                        .map(|_| smallvec::smallvec![0u16])
                        .collect()
                } else {
                    let rotated_centroids: Vec<Vec<f32>> = centroids
                        .iter()
                        .map(|c| quantizer.rotator().rotate(c))
                        .collect();
                    assign(
                        &rotated_flat,
                        padded_dim,
                        &rotated_centroids,
                        self.config.replication.as_ref(),
                    )
                };

                // Phase 4: bucket records into target clusters and
                // transpose.
                let cluster_batch_data = encode_window_into_clusters_from_records(
                    &all_records,
                    bpr,
                    padded_dim,
                    &assignments,
                    num_clusters,
                );

                let mut data =
                    build_cluster_data(centroids, &assignments, win_num_docs, field_cfg.metric);
                data.cluster_batch_data = cluster_batch_data;

                windows.push((win_start as u32, win_num_docs as u32, data));
            }

            let cw = cluster_composite.for_field(field_cfg.field);
            let meta = VectorFieldMeta::from_config(field_cfg);
            serialize_windowed_field(&windows, &meta, cw)?;
            cw.flush()?;
        }

        cluster_composite.close()?;
        Ok(())
    }
}

/// Per-source-segment lookup table: `source_doc_id → (window_idx,
/// cluster_id, slot)`. Built once when a merge starts so per-doc
/// extraction is `O(1)`. The `cluster_field` reference is kept alive
/// through the lifetime of the lookup so cluster batch reads (which
/// return `OwnedBytes` slices into the cluster file) stay valid.
struct SourceLocator<'a> {
    cluster_field: &'a ClusterFieldReader,
    /// Indexed by source_doc_id. `(u32::MAX, _, _)` marks docs not
    /// found — the source segment may have docs in unclustered or
    /// empty windows that we still want to skip cleanly.
    locator: Vec<(u32, u32, u32)>,
}

impl<'a> SourceLocator<'a> {
    fn build(cluster_field: &'a ClusterFieldReader) -> Self {
        // Materialise (window_idx, cluster_id, slot) for every
        // source_doc_id we can find. Slots within a cluster
        // correspond to position in the doc-id prefix; the same
        // ordering is used by `encode_window_into_clusters_from_records`
        // (target side) and `transposed::encode_batch` (per batch
        // of 16 docs).
        let mut locator: Vec<(u32, u32, u32)> = Vec::new();
        for win_idx in 0..cluster_field.num_windows() {
            let win = cluster_field.window_reader(win_idx);
            let win_offset = win.doc_offset as usize;
            for cluster_id in 0..win.num_clusters() {
                let Ok(Some(doc_ids)) = win.cluster_doc_ids(cluster_id) else {
                    continue;
                };
                for (slot, &local_doc_id) in doc_ids.iter().enumerate() {
                    let global = win_offset + local_doc_id as usize;
                    if global >= locator.len() {
                        locator.resize(global + 1, (u32::MAX, 0, 0));
                    }
                    locator[global] = (win_idx as u32, cluster_id as u32, slot as u32);
                }
            }
        }
        Self {
            cluster_field,
            locator,
        }
    }
}

/// One source-doc lookup row used to drive Phase-1 extraction in
/// `merge`. We resolve every target doc into one of these, sort by
/// `(seg_ord, win_idx, cluster_id)`, and stream-process by group so
/// each source cluster's bytes get fetched (and PG-buffer-pinned) once
/// instead of once per contributing doc.
struct DocLookup {
    local_doc_id: u32,
    seg_ord: u32,
    win_idx: u32,
    cluster_id: u32,
    slot: u32,
}

/// Build a `ClusterBatchData` for external-quantizer (TurboQuant) mode
/// using the 16-doc transposed layout from
/// `vector::turboquant::transposed`. Cluster bytes are
/// `[doc_ids: 4 × N][batches: ⌈N/16⌉ × batch_bytes(padded_dim)]`,
/// where each batch is one coord-major SIMD chunk.
///
/// All input records must be `bit_width = 4` records of length
/// `bytes_per_record`. Tail batches with fewer than 16 docs pad the
/// remaining lanes with zeros (γ = 0, so the lanes' contribution is
/// neutral and the collector simply ignores them via doc-id bounds).
fn build_cluster_batch_data_external(
    docs: &[(DocId, Vec<u8>)],
    bytes_per_record: usize,
    padded_dim: usize,
) -> ClusterBatchData {
    use crate::vector::turboquant::transposed::{self, BATCH_DOCS};

    let num_docs = docs.len();
    let num_batches = num_docs.div_ceil(BATCH_DOCS);
    let bb = transposed::batch_bytes(padded_dim);
    let mut tqvec_records = vec![0u8; num_batches * bb];

    let zero_rec = vec![0u8; bytes_per_record];
    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * BATCH_DOCS;
        let batch_end = (batch_start + BATCH_DOCS).min(num_docs);
        let mut slot_recs: Vec<&[u8]> = Vec::with_capacity(BATCH_DOCS);
        for slot in batch_start..batch_end {
            let r = &docs[slot].1;
            if r.len() == bytes_per_record {
                slot_recs.push(r.as_slice());
            } else {
                // Defensive: missing or wrong-length records get
                // zero-filled. Score for the slot is garbage, but
                // doc-id bounds in the collector skip it anyway.
                slot_recs.push(zero_rec.as_slice());
            }
        }
        let out_off = batch_idx * bb;
        transposed::encode_batch(
            &slot_recs,
            padded_dim,
            &mut tqvec_records[out_off..out_off + bb],
        );
    }

    ClusterBatchData {
        doc_ids: docs.iter().map(|(d, _)| *d).collect(),
        tqvec_records,
        num_batches: num_batches as u32,
    }
}

struct WindowSampler<'a> {
    inner: &'a (dyn crate::vector::cluster::sampler::VectorSampler + 'a),
    doc_offset: usize,
}

unsafe impl<'a> Send for WindowSampler<'a> {}
unsafe impl<'a> Sync for WindowSampler<'a> {}

impl<'a> crate::vector::cluster::sampler::VectorSampler for WindowSampler<'a> {
    fn sample_vectors(
        &self,
        field: Field,
        doc_ids: &[DocId],
    ) -> crate::Result<Vec<Option<Vec<f32>>>> {
        let mapped: Vec<DocId> = doc_ids
            .iter()
            .map(|&id| (id as usize + self.doc_offset) as DocId)
            .collect();
        self.inner.sample_vectors(field, &mapped)
    }

    fn dims(&self, field: Field) -> usize {
        self.inner.dims(field)
    }
}

pub struct ClusterPluginWriter {
    config: Arc<ClusterConfig>,
    per_field_vectors: HashMap<Field, Vec<Vec<f32>>>,
}

impl ClusterPluginWriter {
    pub fn ingest_vectors<D: Document>(&mut self, doc: &D, schema: &Schema) {
        for field_cfg in &self.config.fields {
            let vectors = self
                .per_field_vectors
                .entry(field_cfg.field)
                .or_insert_with(Vec::new);

            let mut found = false;
            for (field, value) in doc.iter_fields_and_values() {
                if field != field_cfg.field {
                    continue;
                }
                let field_entry = schema.get_field_entry(field);
                if !matches!(field_entry.field_type(), FieldType::Vector(_)) {
                    continue;
                }
                let value = value.as_value();
                if let Some(vec_data) = value.as_leaf().and_then(|leaf| leaf.as_vector()) {
                    vectors.push(vec_data.to_vec());
                    found = true;
                    break;
                }
            }
            if !found {
                vectors.push(vec![0.0f32; field_cfg.dims]);
            }
        }
    }
}

impl PluginWriter for ClusterPluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        _doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let cluster_write = segment.open_write(component())?;
        let mut cluster_composite = CompositeWrite::wrap(cluster_write);

        // Defer mode: skip k-means but still encode + transpose
        // every doc into a single trivial cluster per window. The
        // records have to be on disk for the post-load force merge
        // to read them back (there's no separate doc-major store).
        // The savings vs the normal path are k-means itself
        // (~1-5 s per 100K-doc segment).
        if self.config.defer_clustering {
            for field_cfg in &self.config.fields {
                let quantizer = &field_cfg.quantizer;
                assert_eq!(
                    quantizer.bit_width,
                    crate::vector::turboquant::transposed::TRANSPOSED_BIT_WIDTH,
                    "cluster plugin transposed layout requires bit_width = 4",
                );

                let vectors = self.per_field_vectors.get(&field_cfg.field);
                let num_docs = vectors.map_or(0, |v| v.len());
                let meta = VectorFieldMeta::from_config(field_cfg);
                let cw = cluster_composite.for_field(field_cfg.field);

                if num_docs == 0 {
                    serialize_windowed_empty(cw, num_docs, &meta)?;
                    cw.flush()?;
                    continue;
                }

                let vectors = vectors.unwrap();
                let num_windows = num_docs.div_ceil(WINDOW_SIZE);
                let mut windows: Vec<(u32, u32, ClusterData)> = Vec::with_capacity(num_windows);
                for win_idx in 0..num_windows {
                    let win_start = win_idx * WINDOW_SIZE;
                    let win_end = (win_start + WINDOW_SIZE).min(num_docs);
                    let win_vectors = &vectors[win_start..win_end];
                    let result = cluster_as_single(win_vectors, field_cfg.metric);
                    let mut data = result.data;
                    let win_data = encode_window_into_clusters(
                        win_vectors,
                        quantizer,
                        &result.assignments,
                        data.num_clusters,
                    );
                    data.cluster_batch_data = win_data;
                    windows.push((win_start as u32, win_vectors.len() as u32, data));
                }
                serialize_windowed_field(&windows, &meta, cw)?;
                cw.flush()?;
            }
            cluster_composite.close()?;
            return Ok(());
        }

        for field_cfg in &self.config.fields {
            let quantizer = &field_cfg.quantizer;
            assert_eq!(
                quantizer.bit_width,
                crate::vector::turboquant::transposed::TRANSPOSED_BIT_WIDTH,
                "cluster plugin transposed layout requires bit_width = 4",
            );

            let vectors = self.per_field_vectors.get(&field_cfg.field);
            let num_docs = vectors.map_or(0, |v| v.len());
            let meta = VectorFieldMeta::from_config(field_cfg);

            if num_docs == 0 {
                let cw = cluster_composite.for_field(field_cfg.field);
                serialize_windowed_empty(cw, num_docs, &meta)?;
                cw.flush()?;
                continue;
            }

            let vectors = vectors.unwrap();
            let num_windows = num_docs.div_ceil(WINDOW_SIZE);
            let mut windows: Vec<(u32, u32, ClusterData)> = Vec::with_capacity(num_windows);

            for win_idx in 0..num_windows {
                let win_start = win_idx * WINDOW_SIZE;
                let win_end = (win_start + WINDOW_SIZE).min(num_docs);
                let win_vectors = &vectors[win_start..win_end];
                let win_num_docs = win_vectors.len();

                // Below threshold: skip k-means, build a single
                // trivial cluster (centroid = mean of all docs)
                // containing every doc. Above threshold: run
                // k-means as usual. Both paths produce the same
                // on-disk shape; the only difference is the number
                // of clusters and how they were chosen.
                let result = if (win_num_docs as u32) < self.config.clustering_threshold {
                    cluster_as_single(win_vectors, field_cfg.metric)
                } else {
                    cluster_from_vectors(win_vectors, &self.config, field_cfg)?
                };
                let mut data = result.data;
                let assignments = result.assignments;

                let win_data = encode_window_into_clusters(
                    win_vectors,
                    quantizer,
                    &assignments,
                    data.num_clusters,
                );
                data.cluster_batch_data = win_data;

                windows.push((win_start as u32, win_num_docs as u32, data));
            }

            let cw = cluster_composite.for_field(field_cfg.field);
            serialize_windowed_field(&windows, &meta, cw)?;
            cw.flush()?;
        }

        cluster_composite.close()?;
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.per_field_vectors
            .values()
            .map(|vecs| vecs.iter().map(|v| v.len() * 4).sum::<usize>())
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct ClusterPluginReader {
    field_readers: HashMap<Field, ClusterFieldReader>,
}

impl ClusterPluginReader {
    pub fn field_reader(&self, field: Field) -> Option<&ClusterFieldReader> {
        self.field_readers.get(&field)
    }
}

impl PluginReader for ClusterPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Parse `num_docs` little-endian u32 doc_ids out of a contiguous byte
/// slice. The slice must be at least `num_docs * 4` bytes long.
fn parse_doc_ids(bytes: &[u8], num_docs: usize) -> Vec<DocId> {
    let mut out = Vec::with_capacity(num_docs);
    for i in 0..num_docs {
        let off = i * 4;
        out.push(u32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
    }
    out
}

/// Result of a single coalesced multi-range read: the list of merged
/// byte ranges (`runs`), the bytes returned for each run, and a parallel
/// `run_of` index that maps each input request (already sorted by
/// offset) to its run.
struct CoalescedReadResult {
    runs: Vec<(usize, usize)>,
    run_bytes: Vec<common::OwnedBytes>,
    run_of: Vec<usize>,
}

impl CoalescedReadResult {
    /// For input request index `si` (post-sort), return
    /// `(run_start_offset, &run_bytes)`. Caller computes the relative
    /// slice within `run_bytes` using its own `(off, len)`.
    fn bytes_for(&self, si: usize) -> (usize, common::OwnedBytes) {
        let r = self.run_of[si];
        (self.runs[r].0, self.run_bytes[r].clone())
    }
}

/// Sort `requests` by offset (in place) and issue one
/// `read_bytes_slice` per coalesced run. `requests` is `(out_idx,
/// offset, len)` triples; offsets are merged into the same run when
/// the gap is at most `gap_tolerance` bytes.
fn coalesced_read(
    file: &FileSlice,
    requests: &mut [(usize, usize, usize)],
    gap_tolerance: usize,
) -> crate::Result<CoalescedReadResult> {
    requests.sort_unstable_by_key(|x| x.1);

    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_of: Vec<usize> = Vec::with_capacity(requests.len());
    for &(_, off, len) in requests.iter() {
        if let Some(last) = runs.last_mut() {
            if off <= last.1 + gap_tolerance {
                last.1 = last.1.max(off + len);
                run_of.push(runs.len() - 1);
                continue;
            }
        }
        runs.push((off, off + len));
        run_of.push(runs.len() - 1);
    }

    let mut run_bytes: Vec<common::OwnedBytes> = Vec::with_capacity(runs.len());
    for &(start, end) in &runs {
        run_bytes.push(file.read_bytes_slice(start..end)?);
    }

    Ok(CoalescedReadResult {
        runs,
        run_bytes,
        run_of,
    })
}

pub struct ClusterBatchMeta {
    pub num_batches: u32,
    pub num_docs: u32,
    /// Offset of this cluster's doc_id prefix within the window's
    /// cold section. doc_ids span is `num_docs * 4` bytes.
    pub doc_ids_offset: u32,
    /// Offset of this cluster's records within the window's cold
    /// section. records span is `records_len` bytes.
    pub records_offset: u32,
    pub records_len: u32,
}

pub struct WindowReader {
    centroid_index: Option<CentroidIndex>,
    batch_meta: Vec<ClusterBatchMeta>,
    batch_data: FileSlice,
    pub doc_offset: u32,
    pub num_docs: u32,
    dims: usize,
}

impl WindowReader {
    /// Parse a window from its hot bytes (centroids + batch_meta) plus a
    /// `FileSlice` for its cold section (batch_data, read lazily).
    fn open(
        hot: &[u8],
        cold_slice: FileSlice,
        doc_offset: u32,
        num_docs_in_window: u32,
        metric: Metric,
    ) -> crate::Result<Self> {
        if hot.len() < 12 {
            return Err(crate::TantivyError::InternalError(
                "window hot section too short".into(),
            ));
        }

        let num_clusters = u32::from_le_bytes([hot[0], hot[1], hot[2], hot[3]]) as usize;
        let _num_docs_field = u32::from_le_bytes([hot[4], hot[5], hot[6], hot[7]]) as usize;
        let dims = u32::from_le_bytes([hot[8], hot[9], hot[10], hot[11]]) as usize;

        if num_clusters == 0 {
            return Ok(Self {
                centroid_index: None,
                batch_meta: Vec::new(),
                batch_data: FileSlice::empty(),
                doc_offset,
                num_docs: num_docs_in_window,
                dims,
            });
        }

        let mut offset = 12;
        let ci_len = u32::from_le_bytes([
            hot[offset],
            hot[offset + 1],
            hot[offset + 2],
            hot[offset + 3],
        ]) as usize;
        offset += 4;

        let ci_bytes = &hot[offset..offset + ci_len];
        let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
        let centroid_index = CentroidIndex::load_from_bytes(ci_bytes, centroid_ids, dims, metric)?;
        offset += ci_len;

        let mut batch_meta = Vec::with_capacity(num_clusters);
        let read_u32 = |hot: &[u8], pos: usize| -> u32 {
            u32::from_le_bytes([hot[pos], hot[pos + 1], hot[pos + 2], hot[pos + 3]])
        };
        for _ in 0..num_clusters {
            let nb = read_u32(hot, offset);
            offset += 4;
            let nd = read_u32(hot, offset);
            offset += 4;
            let dio = read_u32(hot, offset);
            offset += 4;
            let ro = read_u32(hot, offset);
            offset += 4;
            let rl = read_u32(hot, offset);
            offset += 4;
            batch_meta.push(ClusterBatchMeta {
                num_batches: nb,
                num_docs: nd,
                doc_ids_offset: dio,
                records_offset: ro,
                records_len: rl,
            });
        }

        Ok(Self {
            centroid_index: Some(centroid_index),
            batch_meta,
            batch_data: cold_slice,
            doc_offset,
            num_docs: num_docs_in_window,
            dims,
        })
    }

    pub fn is_clustered(&self) -> bool {
        self.centroid_index.is_some()
    }

    pub fn num_clusters(&self) -> usize {
        self.centroid_index.as_ref().map_or(0, |ci| ci.len())
    }

    pub fn search_centroids(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        match &self.centroid_index {
            Some(ci) => ci.search(query, ef_search),
            None => vec![],
        }
    }

    pub fn has_batch_data(&self, cluster_id: usize) -> bool {
        cluster_id < self.batch_meta.len() && self.batch_meta[cluster_id].num_batches > 0
    }

    /// Read a single cluster's doc_id prefix (small, `num_docs * 4` bytes).
    /// Used as a cheap filter-bitset overlap pre-check before deciding
    /// whether to fetch the much larger records region.
    pub fn cluster_doc_ids(&self, cluster_id: usize) -> crate::Result<Option<Vec<DocId>>> {
        if cluster_id >= self.batch_meta.len() {
            return Ok(None);
        }
        let meta = &self.batch_meta[cluster_id];
        if meta.num_batches == 0 {
            return Ok(None);
        }
        let num_docs = meta.num_docs as usize;
        let off = meta.doc_ids_offset as usize;
        let prefix = self.batch_data.read_bytes_slice(off..off + num_docs * 4)?;
        Ok(Some(parse_doc_ids(&prefix, num_docs)))
    }

    /// Coalesced fetch of doc_ids prefixes for many clusters. Doc_ids
    /// for all clusters are stored contiguous at the start of the cold
    /// section in cluster-id order, so unless `cluster_ids` is sparse
    /// across many clusters this typically resolves to one (or a
    /// handful of) backing reads. Output is aligned to the input.
    pub fn cluster_doc_ids_many(
        &self,
        cluster_ids: &[u32],
        gap_tolerance: usize,
    ) -> crate::Result<Vec<Option<Vec<DocId>>>> {
        let n = cluster_ids.len();
        let mut out: Vec<Option<Vec<DocId>>> = (0..n).map(|_| None).collect();

        let mut sorted: Vec<(usize, usize, usize)> = Vec::with_capacity(n);
        for (idx, &cid) in cluster_ids.iter().enumerate() {
            let cid_us = cid as usize;
            if cid_us >= self.batch_meta.len() {
                continue;
            }
            let m = &self.batch_meta[cid_us];
            if m.num_batches == 0 {
                continue;
            }
            sorted.push((idx, m.doc_ids_offset as usize, m.num_docs as usize * 4));
        }
        if sorted.is_empty() {
            return Ok(out);
        }
        let runs_and_bytes = coalesced_read(&self.batch_data, &mut sorted, gap_tolerance)?;

        for (si, &(out_idx, off, len)) in sorted.iter().enumerate() {
            let (run_start, ref bytes) = runs_and_bytes.bytes_for(si);
            let rel = off - run_start;
            let prefix = &bytes[rel..rel + len];
            let num_docs = len / 4;
            out[out_idx] = Some(parse_doc_ids(prefix, num_docs));
        }
        Ok(out)
    }

    pub fn cluster_batch_raw(
        &self,
        cluster_id: usize,
    ) -> crate::Result<Option<(Vec<DocId>, &ClusterBatchMeta, common::OwnedBytes)>> {
        if cluster_id >= self.batch_meta.len() {
            return Ok(None);
        }
        let meta = &self.batch_meta[cluster_id];
        if meta.num_batches == 0 {
            return Ok(None);
        }
        let doc_ids = self.cluster_doc_ids(cluster_id)?.unwrap_or_default();
        let records = self.batch_data.read_bytes_slice(
            meta.records_offset as usize..meta.records_offset as usize + meta.records_len as usize,
        )?;
        Ok(Some((doc_ids, meta, records)))
    }

    /// Fetch records-region bytes for a set of clusters, coalescing
    /// adjacent ranges into one backing read. Records for all clusters
    /// are stored contiguous in the cold section in cluster-id order
    /// (right after the doc_ids region), so the typical query path
    /// where `cluster_ids` is a small probe subset still resolves to
    /// a handful of reads.
    ///
    /// `gap_tolerance`: merge two byte ranges in a single read when
    /// their gap (bytes of unprobed clusters between them) is at most
    /// this value.
    ///
    /// Doc_ids are *not* included in the result — fetch them via
    /// `cluster_doc_ids_many` first to drive a filter-bitset overlap
    /// check before paying for these records reads.
    ///
    /// Returns a Vec aligned to the input.
    pub fn cluster_records_raw_many(
        &self,
        cluster_ids: &[u32],
        gap_tolerance: usize,
    ) -> crate::Result<Vec<Option<(&ClusterBatchMeta, common::OwnedBytes)>>> {
        let n = cluster_ids.len();
        let mut out: Vec<Option<(&ClusterBatchMeta, common::OwnedBytes)>> =
            (0..n).map(|_| None).collect();

        let mut sorted: Vec<(usize, usize, usize)> = Vec::with_capacity(n);
        for (idx, &cid) in cluster_ids.iter().enumerate() {
            let cid_us = cid as usize;
            if cid_us >= self.batch_meta.len() {
                continue;
            }
            let m = &self.batch_meta[cid_us];
            if m.num_batches == 0 {
                continue;
            }
            sorted.push((idx, m.records_offset as usize, m.records_len as usize));
        }
        if sorted.is_empty() {
            return Ok(out);
        }
        let runs_and_bytes = coalesced_read(&self.batch_data, &mut sorted, gap_tolerance)?;

        for (si, &(out_idx, off, len)) in sorted.iter().enumerate() {
            let (run_start, ref bytes) = runs_and_bytes.bytes_for(si);
            let rel = off - run_start;
            let cid_us = cluster_ids[out_idx] as usize;
            let meta = &self.batch_meta[cid_us];
            out[out_idx] = Some((meta, bytes.slice(rel..rel + len)));
        }
        Ok(out)
    }
}

pub struct ClusterFieldReader {
    windows: Vec<WindowReader>,
    dim_bytes: usize,
    field_meta: Option<VectorFieldMeta>,
}

/// Header: u32 num_windows + u32 window_size + FIELD_META_SIZE B field_meta.
const CLUSTER_FIELD_HEADER_SIZE: usize = 8 + FIELD_META_SIZE;

/// Read header + directory + every window's hot section from `file_slice`
/// into a contiguous packed buffer. Cold sections are skipped: the packed
/// layout is `[header][directory][hot_0][hot_1]...[hot_{n-1}]`, whereas
/// the on-disk file interleaves hot and cold per window. This packing is
/// the unit cached by `HotBytesCache` — ~5 MB per index for Cohere 1M, vs
/// ~130 MB for the full file.
fn pack_hot_bytes(file_slice: &FileSlice) -> crate::Result<Vec<u8>> {
    let header = file_slice.read_bytes_slice(0..CLUSTER_FIELD_HEADER_SIZE)?;
    let num_windows = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;

    let dir_end = CLUSTER_FIELD_HEADER_SIZE + num_windows * WINDOW_DIR_ENTRY_SIZE;
    let dir = file_slice.read_bytes_slice(CLUSTER_FIELD_HEADER_SIZE..dir_end)?;

    let mut file_cumulative = dir_end;
    let mut hot_total = 0usize;
    let mut window_info: Vec<(usize, usize, usize)> = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let off = i * WINDOW_DIR_ENTRY_SIZE;
        let hot_size =
            u32::from_le_bytes([dir[off + 8], dir[off + 9], dir[off + 10], dir[off + 11]]) as usize;
        let cold_size =
            u32::from_le_bytes([dir[off + 12], dir[off + 13], dir[off + 14], dir[off + 15]])
                as usize;
        window_info.push((file_cumulative, hot_size, cold_size));
        hot_total += hot_size;
        file_cumulative += hot_size + cold_size;
    }

    let packed_size = dir_end + hot_total;
    let mut packed = Vec::with_capacity(packed_size);
    packed.extend_from_slice(&header);
    packed.extend_from_slice(&dir);
    for (file_off, hot_size, _) in &window_info {
        let hot = file_slice.read_bytes_slice(*file_off..*file_off + *hot_size)?;
        packed.extend_from_slice(&hot);
    }
    Ok(packed)
}

impl ClusterFieldReader {
    fn open(file_slice: FileSlice, metric: Metric) -> crate::Result<Self> {
        // Pack header + directory + per-window hot sections into one
        // contiguous buffer. Cold (batch_data) reads stay against the
        // original `file_slice`. Each backend pays this once per
        // segment-open and keeps it in process memory.
        let packed = OwnedBytes::new(pack_hot_bytes(&file_slice)?);
        Self::parse_packed(&packed, file_slice, metric)
    }

    /// Parse the packed buffer into windowed readers. The packed layout is
    /// produced by [`pack_hot_bytes`]. Cold `batch_data` slices are
    /// resolved against the original `file_slice` using offsets derived
    /// from the directory.
    fn parse_packed(packed: &[u8], file_slice: FileSlice, metric: Metric) -> crate::Result<Self> {
        if packed.len() < CLUSTER_FIELD_HEADER_SIZE {
            return Err(crate::TantivyError::InternalError(
                "cluster field header underflow".into(),
            ));
        }
        let num_windows = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]) as usize;
        let _window_size =
            u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]) as usize;
        let mut field_meta_pos = 8;
        let field_meta = Some(deserialize_field_meta(packed, &mut field_meta_pos));

        let dir_end = CLUSTER_FIELD_HEADER_SIZE + num_windows * WINDOW_DIR_ENTRY_SIZE;
        let dir = &packed[CLUSTER_FIELD_HEADER_SIZE..dir_end];

        // Two cumulative offsets: one into the packed buffer (hot only),
        // one into the original file (hot + cold interleaved).
        let mut packed_cumulative = dir_end;
        let mut file_cumulative = dir_end;

        let mut windows = Vec::with_capacity(num_windows);
        let mut dim_bytes = 0usize;
        for i in 0..num_windows {
            let off = i * WINDOW_DIR_ENTRY_SIZE;
            let win_doc_offset =
                u32::from_le_bytes([dir[off], dir[off + 1], dir[off + 2], dir[off + 3]]);
            let win_num_docs =
                u32::from_le_bytes([dir[off + 4], dir[off + 5], dir[off + 6], dir[off + 7]]);
            let hot_size =
                u32::from_le_bytes([dir[off + 8], dir[off + 9], dir[off + 10], dir[off + 11]])
                    as usize;
            let cold_size =
                u32::from_le_bytes([dir[off + 12], dir[off + 13], dir[off + 14], dir[off + 15]])
                    as usize;

            let hot = &packed[packed_cumulative..packed_cumulative + hot_size];
            packed_cumulative += hot_size;

            let cold_slice = if cold_size > 0 {
                let cold_off = file_cumulative + hot_size;
                file_slice.slice(cold_off..cold_off + cold_size)
            } else {
                FileSlice::empty()
            };
            file_cumulative += hot_size + cold_size;

            let win_reader =
                WindowReader::open(hot, cold_slice, win_doc_offset, win_num_docs, metric)?;
            if win_reader.dims > 0 {
                dim_bytes = win_reader.dims / 8;
            }
            windows.push(win_reader);
        }

        Ok(Self {
            windows,
            dim_bytes,
            field_meta,
        })
    }

    pub fn is_clustered(&self) -> bool {
        self.windows.iter().any(|w| w.is_clustered())
    }

    pub fn num_clusters(&self) -> usize {
        self.windows.iter().map(|w| w.num_clusters()).sum()
    }

    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }

    pub fn window_reader(&self, idx: usize) -> &WindowReader {
        &self.windows[idx]
    }

    pub fn search_centroids(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        if self.windows.len() == 1 {
            return self.windows[0].search_centroids(query, ef_search);
        }
        let mut all = Vec::new();
        for win in &self.windows {
            all.extend(win.search_centroids(query, ef_search));
        }
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(ef_search);
        all
    }

    pub fn has_batch_data(&self, cluster_id: usize) -> bool {
        if self.windows.len() == 1 {
            return self.windows[0].has_batch_data(cluster_id);
        }
        let mut remaining = cluster_id;
        for win in &self.windows {
            let nc = win.num_clusters();
            if remaining < nc {
                return win.has_batch_data(remaining);
            }
            remaining -= nc;
        }
        false
    }

    pub fn cluster_batch_raw(
        &self,
        cluster_id: usize,
    ) -> crate::Result<Option<(Vec<DocId>, &ClusterBatchMeta, common::OwnedBytes)>> {
        if self.windows.len() == 1 {
            return self.windows[0].cluster_batch_raw(cluster_id);
        }
        let mut remaining = cluster_id;
        for win in &self.windows {
            let nc = win.num_clusters();
            if remaining < nc {
                return win.cluster_batch_raw(remaining);
            }
            remaining -= nc;
        }
        Ok(None)
    }

    pub fn dim_bytes(&self) -> usize {
        self.dim_bytes
    }

    pub fn field_meta(&self) -> Option<&VectorFieldMeta> {
        self.field_meta.as_ref()
    }
}

#[derive(Clone)]
pub struct ProbeConfig {
    /// Hard cap on clusters probed per window per query.
    pub max_probe: usize,
    /// Centroid-distance ratio used by both the per-cluster and the
    /// per-window early-stop. A cluster (or window) is considered
    /// unable to beat the current top-K heap when its nearest
    /// centroid distance exceeds the best already-probed centroid
    /// distance times this factor. `f32::INFINITY` disables the
    /// early-stops.
    pub distance_ratio: f32,
    /// Probe at least this many clusters before the early-stop is
    /// allowed to fire (per window).
    pub min_probe: usize,
    /// Adaptive probe iteration size — the first probe pass within a
    /// window touches `initial_probe` clusters; each follow-up pass
    /// touches `probe_step` more, capped at `max_probe`. Set
    /// `initial_probe = max_probe` (the default) to reproduce the
    /// historical "probe everything in one pass" behaviour. Smaller
    /// values cut work for selective filters by stopping the moment
    /// the heap holds K valid candidates.
    pub initial_probe: usize,
    pub probe_step: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        // Adaptive on by default: probe 8 clusters per window in the
        // first pass, +8 per follow-up pass, terminate as soon as the
        // top-K heap is full and the next un-probed centroid is more
        // than 1.5× the best already-probed centroid distance. The
        // same 1.5× ratio gates the cross-window skip. `max_probe`
        // (= 50) caps total clusters scored per window so a query
        // that never satisfies the early-stop still bounds its work.
        //
        // Measured on bioasq_10m with a 1% uniform label filter:
        // ~30% buffer + latency reduction vs single-pass at full
        // recall@10. Unfiltered queries pay a small (~15%) overhead
        // for the extra iteration round trips but don't lose recall.
        Self {
            max_probe: 50,
            distance_ratio: 1.5,
            min_probe: 1,
            initial_probe: 8,
            probe_step: 8,
        }
    }
}

impl ProbeConfig {
    pub fn new(max_probe: usize, distance_ratio: f32) -> Self {
        Self {
            max_probe,
            distance_ratio,
            min_probe: 1,
            initial_probe: max_probe,
            probe_step: max_probe,
        }
    }

    pub fn with_min_probe(mut self, min_probe: usize) -> Self {
        self.min_probe = min_probe;
        self
    }

    /// Configure adaptive probe iteration. `initial` is the size of
    /// the first probe pass per window; `step` is the size of each
    /// follow-up pass. The collector iterates up to `max_probe` in
    /// total, terminating early when the top-K heap holds K
    /// candidates and the next un-probed centroid is too far to
    /// plausibly beat the heap (per `distance_ratio`).
    pub fn with_adaptive(mut self, initial: usize, step: usize) -> Self {
        self.initial_probe = initial;
        self.probe_step = step.max(1);
        self
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for assignment internals. The full round-trip
    //! plugin lifecycle lives in `crate::vector::cluster::tests`.

    use super::*;

    /// Test helper: flatten a `Vec<Vec<f32>>` and call `assign` with
    /// the given replication config.
    fn assign_rep(
        points: &[Vec<f32>],
        centroids: &[Vec<f32>],
        epsilon: f32,
        max_replicas: usize,
    ) -> Vec<Assignment> {
        let dim = points.first().map(|p| p.len()).unwrap_or(0);
        let mut flat: Vec<f32> = Vec::with_capacity(points.len() * dim);
        for p in points {
            flat.extend_from_slice(p);
        }
        let rc = ReplicationConfig {
            epsilon,
            max_replicas,
        };
        assign(&flat, dim, centroids, Some(&rc))
    }

    /// Geometry test for replicated assignment: place centroids in a
    /// hex around the origin, then probe interior / boundary / vertex
    /// positions and assert the replication factor matches what the
    /// ε-cone + RNG rule predicts.
    ///
    /// 6 centroids on a unit hexagon (60° apart). Test points:
    ///   - Deep interior: at one centroid → replicates only to itself.
    ///   - Boundary: midway between two adjacent centroids → replicates to those 2.
    ///   - Center: equidistant from all 6 → ε-cone catches all 6, RNG keeps a subset.
    #[test]
    fn assign_replicated_hex_geometry() {
        let centroids: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let theta = std::f32::consts::PI * (i as f32) / 3.0;
                vec![theta.cos(), theta.sin()]
            })
            .collect();

        // Interior: sit on top of centroid 0 → only c0 should be assigned.
        let interior = centroids[0].clone();
        let a = assign_rep(&[interior], &centroids, 0.1, 8);
        assert_eq!(a[0].len(), 1, "interior point should not replicate");
        assert_eq!(a[0][0], 0);

        // Boundary: midpoint of c0 and c1. With ε = 0.1, both c0 and
        // c1 land in the cone (they're equidistant) and they pass RNG
        // because dist(c0, c1) > dist(p, c0) when p is the midpoint.
        let boundary = vec![
            0.5 * (centroids[0][0] + centroids[1][0]),
            0.5 * (centroids[0][1] + centroids[1][1]),
        ];
        let a = assign_rep(&[boundary], &centroids, 0.1, 8);
        assert_eq!(
            a[0].len(),
            2,
            "boundary midpoint should replicate to two adjacent centroids, got {:?}",
            a[0]
        );
        let mut got: Vec<u16> = a[0].iter().copied().collect();
        got.sort();
        assert_eq!(got, vec![0, 1]);

        // Centre of the hex: equidistant from all 6 centroids. ε-cone
        // accepts all of them; RNG prunes redundant pairs and should
        // leave a subset that spans the polygon — typically 2 or 3
        // (the directions that aren't blocked by an earlier-selected
        // centroid). Assert ≥ 2 and ≤ all-6.
        let centre = vec![0.0_f32, 0.0_f32];
        let a = assign_rep(&[centre], &centroids, 0.1, 8);
        assert!(
            a[0].len() >= 2 && a[0].len() <= 6,
            "centre should replicate but RNG should prune some, got {:?}",
            a[0]
        );
    }

    /// `epsilon = 0` should reduce to single assignment for any point
    /// not exactly equidistant from multiple centroids.
    #[test]
    fn assign_replicated_zero_epsilon_is_single() {
        let centroids: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![5.0, 10.0]];
        let points: Vec<Vec<f32>> = vec![vec![1.0, 0.5], vec![9.0, 0.5], vec![5.0, 9.0]];
        let a = assign_rep(&points, &centroids, 0.0, 8);
        for assignment in &a {
            assert_eq!(
                assignment.len(),
                1,
                "epsilon=0 must produce single assignment off-boundary"
            );
        }
    }

    /// `max_replicas = 1` always degenerates to single assignment, even
    /// at boundary positions.
    #[test]
    fn assign_replicated_max_one_is_single() {
        let centroids: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let boundary = vec![0.5, 0.0];
        let a = assign_rep(&[boundary], &centroids, 1.0, 1);
        assert_eq!(a[0].len(), 1);
    }

    /// RNG pruning must reject candidates that are closer to an
    /// already-selected centroid than to the point. Construct three
    /// nearly-collinear centroids: `c0`, `c1`, `c2` along the x-axis
    /// at 0, 1, 2. Place the point at x=0.5. All three are within ε,
    /// but RNG should reject `c2` because `dist(c1, c2) = 1` is less
    /// than `dist(p, c2) = 1.5`.
    #[test]
    fn assign_replicated_rng_prunes_collinear() {
        let centroids: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        // Place near c0 so c1 is in the ε-cone but c2 might not be.
        // Use a wide ε so c2 IS in the cone, then RNG must prune it.
        // d(p, c0) = 0.1, d(p, c1) = 0.9, d(p, c2) = 1.9.
        // ε-cone with ε = 100: threshold = 0.1 × 101 = 10.1, all in.
        // RNG: c1 selected after c0. For c2: d(c1, c2) = 1 vs d(p, c2) = 1.9 → 1 < 1.9 → prune.
        let p = vec![0.1, 0.0];
        let a = assign_rep(&[p], &centroids, 100.0, 8);
        let mut got: Vec<u16> = a[0].iter().copied().collect();
        got.sort();
        assert_eq!(
            got,
            vec![0, 1],
            "RNG should keep nearest two and prune c2 (closer to c1 than to p)"
        );
    }
}

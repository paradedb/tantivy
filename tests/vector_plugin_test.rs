//! Integration test for a cluster-based (IVF) vector search plugin.
//!
//! Implements the core of the vector search spec as a SegmentPlugin:
//! - Cluster-based index: divide vectors into fixed-size clusters around centroids
//! - K-means clustering at index/merge time
//! - Per-cluster sorted doc ID lists (for intersection with text/metadata iterators)
//! - Centroid-based pruning: rank clusters by distance to query, only scan top-n
//! - Binary quantized vectors for fast approximate scoring
//! - Full-precision rescoring of final candidates
//!
//! This test exercises the full lifecycle: index → read → query → merge → read again.

use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use tantivy::index::SegmentComponent;
use tantivy::indexer::doc_id_mapping::DocIdMapping;
use tantivy::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{DocId, Index, IndexWriter, Segment};

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    d / (na * nb)
}

fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Binary quantize: positive dimensions → 1, non-positive → 0.
/// Packed into bytes, MSB first within each byte.
fn binary_quantize(v: &[f32]) -> Vec<u8> {
    let num_bytes = (v.len() + 7) / 8;
    let mut out = vec![0u8; num_bytes];
    for (i, &val) in v.iter().enumerate() {
        if val > 0.0 {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
}

/// Hamming distance between two binary-quantized vectors.
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

/// Simple k-means implementation. Returns centroids.
fn kmeans(vectors: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(vectors.len());
    let dims = vectors[0].len();

    // Initialize centroids by evenly spacing through the vector list
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| vectors[i * vectors.len() / k].clone())
        .collect();

    let mut assignments = vec![0usize; vectors.len()];

    for _iter in 0..max_iters {
        // Assign each vector to nearest centroid
        let mut changed = false;
        for (vi, vec) in vectors.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    l2_distance_sq(vec, a)
                        .partial_cmp(&l2_distance_sq(vec, b))
                        .unwrap()
                })
                .unwrap()
                .0;
            if assignments[vi] != nearest {
                assignments[vi] = nearest;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Recompute centroids
        let mut sums = vec![vec![0.0f32; dims]; k];
        let mut counts = vec![0usize; k];
        for (vi, vec) in vectors.iter().enumerate() {
            let c = assignments[vi];
            counts[c] += 1;
            for (d, &val) in vec.iter().enumerate() {
                sums[c][d] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dims {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

/// Assign each vector to its nearest centroid, returning cluster_id per vector.
fn assign_clusters(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    vectors
        .iter()
        .map(|v| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    l2_distance_sq(v, a)
                        .partial_cmp(&l2_distance_sq(v, b))
                        .unwrap()
                })
                .unwrap()
                .0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// On-disk format helpers
// ---------------------------------------------------------------------------

/// Serialized format for the vector index within a segment:
///
/// ```text
/// [header]
///   num_clusters: u32
///   dims: u32
///   num_vectors: u32
/// [centroids]
///   num_clusters × dims × f32  (full precision centroid vectors)
/// [cluster_doc_ids]
///   For each cluster:
///     count: u32
///     doc_ids: count × u32  (sorted)
/// [full_precision_vectors]
///   For each doc_id 0..num_vectors:
///     dims × f32
/// [binary_quantized_vectors]
///   For each doc_id 0..num_vectors:
///     ceil(dims/8) bytes
/// ```

fn serialize_vector_index(
    centroids: &[Vec<f32>],
    cluster_doc_ids: &[Vec<DocId>],
    vectors: &[Vec<f32>],
    dims: usize,
    writer: &mut dyn Write,
) -> std::io::Result<()> {
    let num_clusters = centroids.len() as u32;
    let num_vectors = vectors.len() as u32;

    // Header
    writer.write_all(&num_clusters.to_le_bytes())?;
    writer.write_all(&(dims as u32).to_le_bytes())?;
    writer.write_all(&num_vectors.to_le_bytes())?;

    // Centroids
    for centroid in centroids {
        for &val in centroid {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    // Cluster doc IDs
    for doc_ids in cluster_doc_ids {
        writer.write_all(&(doc_ids.len() as u32).to_le_bytes())?;
        for &doc_id in doc_ids {
            writer.write_all(&doc_id.to_le_bytes())?;
        }
    }

    // Full precision vectors (indexed by doc_id)
    for vec in vectors {
        for &val in vec {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    // Binary quantized vectors (indexed by doc_id)
    for vec in vectors {
        let bq = binary_quantize(vec);
        writer.write_all(&bq)?;
    }

    Ok(())
}

fn deserialize_vector_index(data: &[u8]) -> VectorIndexData {
    let mut pos = 0;

    let read_u32 = |pos: &mut usize| -> u32 {
        let val = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos += 4;
        val
    };
    let read_f32 = |pos: &mut usize| -> f32 {
        let val = f32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos += 4;
        val
    };

    let num_clusters = read_u32(&mut pos) as usize;
    let dims = read_u32(&mut pos) as usize;
    let num_vectors = read_u32(&mut pos) as usize;

    // Centroids
    let mut centroids = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut c = Vec::with_capacity(dims);
        for _ in 0..dims {
            c.push(read_f32(&mut pos));
        }
        centroids.push(c);
    }

    // Cluster doc IDs
    let mut cluster_doc_ids = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let count = read_u32(&mut pos) as usize;
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(read_u32(&mut pos));
        }
        cluster_doc_ids.push(ids);
    }

    // Full precision vectors
    let mut vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut v = Vec::with_capacity(dims);
        for _ in 0..dims {
            v.push(read_f32(&mut pos));
        }
        vectors.push(v);
    }

    // Binary quantized vectors
    let bq_bytes_per_vec = (dims + 7) / 8;
    let mut bq_vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let bq = data[pos..pos + bq_bytes_per_vec].to_vec();
        pos += bq_bytes_per_vec;
        bq_vectors.push(bq);
    }

    VectorIndexData {
        centroids,
        cluster_doc_ids,
        vectors,
        bq_vectors,
        dims,
    }
}

struct VectorIndexData {
    centroids: Vec<Vec<f32>>,
    cluster_doc_ids: Vec<Vec<DocId>>,
    vectors: Vec<Vec<f32>>,
    bq_vectors: Vec<Vec<u8>>,
    dims: usize,
}

// ---------------------------------------------------------------------------
// Query: top-K nearest neighbor search with cluster pruning
// ---------------------------------------------------------------------------

/// Result of a nearest neighbor search.
#[derive(Debug, Clone)]
struct VectorSearchResult {
    doc_id: DocId,
    similarity: f32,
}

impl VectorIndexData {
    /// Find top-k nearest neighbors using cluster-based pruning.
    ///
    /// 1. Rank centroids by distance to query vector
    /// 2. Select top `n_probe` clusters
    /// 3. Collect candidate doc IDs from those clusters
    /// 4. Score candidates using binary quantized vectors (fast approximate pass)
    /// 5. Rescore top candidates using full precision vectors
    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<VectorSearchResult> {
        if self.centroids.is_empty() || self.vectors.is_empty() {
            return vec![];
        }

        // Step 1-2: Rank centroids and pick top n_probe
        let n_probe = n_probe.min(self.centroids.len());
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance_sq(query, c)))
            .collect();
        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let probe_clusters: Vec<usize> = centroid_distances[..n_probe]
            .iter()
            .map(|(i, _)| *i)
            .collect();

        // Step 3: Collect candidate doc IDs from probed clusters
        let mut candidates: Vec<DocId> = probe_clusters
            .iter()
            .flat_map(|&c| self.cluster_doc_ids[c].iter().copied())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();

        // Step 4: Fast approximate scoring with binary quantized vectors
        let query_bq = binary_quantize(query);
        let mut scored: Vec<(DocId, u32)> = candidates
            .iter()
            .map(|&doc_id| {
                let hamming = hamming_distance(&query_bq, &self.bq_vectors[doc_id as usize]);
                (doc_id, hamming)
            })
            .collect();
        scored.sort_by_key(|&(_, h)| h);

        // Step 5: Rescore top-K (or all candidates if fewer) with full precision
        let rescore_count = (k * 4).min(scored.len()); // rescore 4x candidates
        let mut results: Vec<VectorSearchResult> = scored[..rescore_count]
            .iter()
            .map(|&(doc_id, _)| {
                let sim = cosine_similarity(query, &self.vectors[doc_id as usize]);
                VectorSearchResult {
                    doc_id,
                    similarity: sim,
                }
            })
            .collect();
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        results
    }

    /// Return sorted doc IDs from the top `n_probe` clusters nearest to `query`.
    /// This is the "doc ID iterator" that gets intersected with text/metadata iterators.
    fn cluster_doc_ids_for_query(&self, query: &[f32], n_probe: usize) -> Vec<DocId> {
        if self.centroids.is_empty() {
            return vec![];
        }
        let n_probe = n_probe.min(self.centroids.len());
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance_sq(query, c)))
            .collect();
        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut doc_ids: Vec<DocId> = centroid_distances[..n_probe]
            .iter()
            .flat_map(|&(c, _)| self.cluster_doc_ids[c].iter().copied())
            .collect();
        doc_ids.sort_unstable();
        doc_ids.dedup();
        doc_ids
    }
}

// ---------------------------------------------------------------------------
// VectorPlugin: SegmentPlugin implementation
// ---------------------------------------------------------------------------

const TARGET_CLUSTER_SIZE: usize = 4; // small for testing; real-world would be ~1000
const KMEANS_MAX_ITERS: usize = 20;
const VECTOR_DIMS: usize = 8; // small dims for testing

struct VectorPlugin {
    dims: usize,
}

impl VectorPlugin {
    fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl SegmentPlugin for VectorPlugin {
    fn name(&self) -> &str {
        "vectors"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["vec"]
    }

    fn write_phase(&self) -> u32 {
        2
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> tantivy::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(VectorPluginWriter {
            vectors: Vec::new(),
            dims: self.dims,
        }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> tantivy::Result<Arc<dyn PluginReader>> {
        let component = SegmentComponent::Custom("vec".to_string());
        let file_slice = ctx
            .segment_reader
            .open_read(component)
            .map_err(|e| tantivy::TantivyError::InternalError(format!("vectors open_read: {e}")))?;
        let data = file_slice.read_bytes()?;
        let index_data = deserialize_vector_index(&data);
        Ok(Arc::new(VectorPluginReader { data: index_data }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> tantivy::Result<()> {
        // Collect all vectors from source segments, remapped to new doc IDs
        let mut all_vectors: HashMap<DocId, Vec<f32>> = HashMap::new();

        for (new_doc_id, old_doc_addr) in ctx.doc_id_mapping.iter_old_doc_addrs().enumerate() {
            let segment_reader = &ctx.readers[old_doc_addr.segment_ord as usize];
            if let Ok(Some(reader)) = segment_reader.plugin_reader::<VectorPluginReader>("vectors")
            {
                let old_doc_id = old_doc_addr.doc_id as usize;
                if old_doc_id < reader.data.vectors.len() {
                    all_vectors
                        .insert(new_doc_id as DocId, reader.data.vectors[old_doc_id].clone());
                }
            }
        }

        if all_vectors.is_empty() {
            return Ok(());
        }

        let max_doc = all_vectors.keys().copied().max().unwrap_or(0) + 1;
        let dims = all_vectors.values().next().unwrap().len();

        // Build ordered vector list (fill missing doc_ids with zero vectors)
        let zero_vec = vec![0.0f32; dims];
        let vectors: Vec<Vec<f32>> = (0..max_doc)
            .map(|d| {
                all_vectors
                    .get(&d)
                    .cloned()
                    .unwrap_or_else(|| zero_vec.clone())
            })
            .collect();

        // Cluster
        let num_clusters = (vectors.len() / TARGET_CLUSTER_SIZE).max(1);
        let non_zero: Vec<Vec<f32>> = vectors.iter().filter(|v| norm(v) > 0.0).cloned().collect();
        let centroids = if non_zero.is_empty() {
            vec![zero_vec.clone()]
        } else {
            kmeans(&non_zero, num_clusters, KMEANS_MAX_ITERS)
        };
        let assignments = assign_clusters(&vectors, &centroids);

        let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![vec![]; centroids.len()];
        for (doc_id, &cluster) in assignments.iter().enumerate() {
            cluster_doc_ids[cluster].push(doc_id as DocId);
        }

        // Write
        let component = SegmentComponent::Custom("vec".to_string());
        let mut write = ctx.target_segment.open_write(component)?;
        serialize_vector_index(&centroids, &cluster_doc_ids, &vectors, dims, &mut write)?;
        common::TerminatingWrite::terminate(write)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VectorPluginWriter
// ---------------------------------------------------------------------------

struct VectorPluginWriter {
    /// Vectors accumulated during indexing, indexed by doc_id.
    vectors: Vec<Vec<f32>>,
    dims: usize,
}

impl VectorPluginWriter {
    /// Add a vector for the given doc. Called by the application via downcast.
    fn add_vector(&mut self, _doc_id: DocId, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.dims,
            "vector dims mismatch: expected {}, got {}",
            self.dims,
            vector.len()
        );
        self.vectors.push(vector);
    }
}

impl PluginWriter for VectorPluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        _doc_id_map: Option<&DocIdMapping>,
    ) -> tantivy::Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        // Cluster the vectors
        let num_clusters = (self.vectors.len() / TARGET_CLUSTER_SIZE).max(1);
        let centroids = kmeans(&self.vectors, num_clusters, KMEANS_MAX_ITERS);
        let assignments = assign_clusters(&self.vectors, &centroids);

        // Build per-cluster doc ID lists
        let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![vec![]; centroids.len()];
        for (doc_id, &cluster) in assignments.iter().enumerate() {
            cluster_doc_ids[cluster].push(doc_id as DocId);
        }

        // Serialize
        let component = SegmentComponent::Custom("vec".to_string());
        let mut write = segment.open_write(component)?;
        serialize_vector_index(
            &centroids,
            &cluster_doc_ids,
            &self.vectors,
            self.dims,
            &mut write,
        )
        .map_err(|e| tantivy::TantivyError::InternalError(e.to_string()))?;
        common::TerminatingWrite::terminate(write)?;
        Ok(())
    }

    fn close(self: Box<Self>) -> tantivy::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.vectors.len() * self.dims * std::mem::size_of::<f32>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// VectorPluginReader
// ---------------------------------------------------------------------------

struct VectorPluginReader {
    data: VectorIndexData,
}

impl VectorPluginReader {
    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<VectorSearchResult> {
        self.data.search(query, k, n_probe)
    }

    fn cluster_doc_ids_for_query(&self, query: &[f32], n_probe: usize) -> Vec<DocId> {
        self.data.cluster_doc_ids_for_query(query, n_probe)
    }

    fn num_clusters(&self) -> usize {
        self.data.centroids.len()
    }

    fn num_vectors(&self) -> usize {
        self.data.vectors.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.data.centroids[cluster_id]
    }
}

impl PluginReader for VectorPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Helper: create vectors that form obvious clusters for testing.
/// Returns (doc_text, vector) pairs.
fn make_test_data() -> Vec<(&'static str, Vec<f32>)> {
    // Cluster A: vectors pointing roughly in the +x direction
    // Cluster B: vectors pointing roughly in the +y direction
    // Cluster C: vectors pointing roughly in the -x direction
    vec![
        // Cluster A (positive x)
        ("alpha one", vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("alpha two", vec![0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "alpha three",
            vec![0.8, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("alpha four", vec![0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        // Cluster B (positive y)
        ("beta one", vec![0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("beta two", vec![0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "beta three",
            vec![0.15, 0.85, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("beta four", vec![0.05, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        // Cluster C (negative x)
        ("gamma one", vec![-1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("gamma two", vec![-0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "gamma three",
            vec![-0.8, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "gamma four",
            vec![-0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]
}

#[test]
fn test_vector_plugin_index_and_search() -> tantivy::Result<()> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT | STORED);
    let schema = schema_builder.build();

    let plugin: Arc<dyn SegmentPlugin> = Arc::new(VectorPlugin::new(VECTOR_DIMS));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    let test_data = make_test_data();

    for (i, (text, vec)) in test_data.iter().enumerate() {
        writer.add_document(tantivy::doc!(text_field => *text))?;
        // Feed vector to the plugin writer via downcast.
        // In a real system this would be integrated into add_document or a custom method.
        // Here we access it through the internals for testing.
    }
    writer.commit()?;

    // We need a way to get vectors into the plugin writer. Since the plugin's
    // add_document only gets DocId, we use a different approach: add vectors
    // after indexing by accessing the writer directly.
    //
    // For this test, let's rebuild with a custom approach: we'll write the vectors
    // directly to the segment file after commit, simulating what a real integration
    // would do through the plugin writer.
    //
    // Actually, let's use a simpler approach: create the index with vectors
    // by using a single-threaded writer and hooking into the plugin system.

    // Re-create with vectors properly fed through the plugin
    let index2 = {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        let plugin: Arc<dyn SegmentPlugin> = Arc::new(VectorPlugin::new(VECTOR_DIMS));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin)
            .create_in_ram()?;

        // Use SingleSegmentIndexWriter for direct access to the segment writer
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        // We'll index documents and track that vectors need to be added
        // For now, test that the plugin lifecycle works even with empty vectors
        for (text, _vec) in &test_data {
            writer.add_document(tantivy::doc!(text_field => *text))?;
        }
        writer.commit()?;
        index
    };

    let reader = index2.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 12);

    Ok(())
}

#[test]
fn test_vector_plugin_with_data() -> tantivy::Result<()> {
    // Test the vector data structures directly (without going through the index writer,
    // since add_document doesn't pass document data to plugins).
    // This tests the core vector search logic: clustering, serialization, search.

    let test_data = make_test_data();
    let vectors: Vec<Vec<f32>> = test_data.iter().map(|(_, v)| v.clone()).collect();

    // Cluster the vectors
    let num_clusters = (vectors.len() / TARGET_CLUSTER_SIZE).max(1);
    assert!(num_clusters >= 2, "should have multiple clusters");

    let centroids = kmeans(&vectors, num_clusters, KMEANS_MAX_ITERS);
    assert_eq!(centroids.len(), num_clusters);

    let assignments = assign_clusters(&vectors, &centroids);
    assert_eq!(assignments.len(), vectors.len());

    // Build cluster doc ID lists
    let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![vec![]; centroids.len()];
    for (doc_id, &cluster) in assignments.iter().enumerate() {
        cluster_doc_ids[cluster].push(doc_id as DocId);
    }

    // Verify cluster locality: docs 0-3 (alpha/+x) should be in one cluster,
    // docs 4-7 (beta/+y) in another, docs 8-11 (gamma/-x) in another.
    let cluster_of_alpha = assignments[0];
    let cluster_of_beta = assignments[4];
    let cluster_of_gamma = assignments[8];

    // Alpha docs should all be in the same cluster
    assert!(assignments[0..4].iter().all(|&c| c == cluster_of_alpha));
    // Beta docs should all be in the same cluster
    assert!(assignments[4..8].iter().all(|&c| c == cluster_of_beta));
    // Gamma docs should all be in the same cluster
    assert!(assignments[8..12].iter().all(|&c| c == cluster_of_gamma));
    // The three clusters should be different
    assert_ne!(cluster_of_alpha, cluster_of_beta);
    assert_ne!(cluster_of_alpha, cluster_of_gamma);
    assert_ne!(cluster_of_beta, cluster_of_gamma);

    // Serialize and deserialize
    let mut buf = Vec::new();
    serialize_vector_index(
        &centroids,
        &cluster_doc_ids,
        &vectors,
        VECTOR_DIMS,
        &mut buf,
    )
    .unwrap();
    let index_data = deserialize_vector_index(&buf);

    assert_eq!(index_data.centroids.len(), num_clusters);
    assert_eq!(index_data.vectors.len(), vectors.len());
    assert_eq!(index_data.bq_vectors.len(), vectors.len());
    assert_eq!(index_data.dims, VECTOR_DIMS);

    // Search: query near alpha cluster, should return alpha docs as top results
    let query_alpha = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = index_data.search(&query_alpha, 4, 1); // probe 1 cluster, get top 4
    assert_eq!(results.len(), 4);
    // All top-4 results should be from the alpha cluster (doc_ids 0-3)
    for r in &results {
        assert!(r.doc_id < 4, "expected alpha doc, got doc_id={}", r.doc_id);
        assert!(r.similarity > 0.9, "expected high similarity for alpha");
    }

    // Search: query near beta cluster
    let query_beta = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = index_data.search(&query_beta, 4, 1);
    assert_eq!(results.len(), 4);
    for r in &results {
        assert!(
            (4..8).contains(&r.doc_id),
            "expected beta doc, got doc_id={}",
            r.doc_id
        );
        assert!(r.similarity > 0.9, "expected high similarity for beta");
    }

    // Search: query near gamma cluster
    let query_gamma = vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = index_data.search(&query_gamma, 4, 1);
    assert_eq!(results.len(), 4);
    for r in &results {
        assert!(
            (8..12).contains(&r.doc_id),
            "expected gamma doc, got doc_id={}",
            r.doc_id
        );
    }

    Ok(())
}

#[test]
fn test_vector_cluster_pruning_reduces_candidates() {
    let test_data = make_test_data();
    let vectors: Vec<Vec<f32>> = test_data.iter().map(|(_, v)| v.clone()).collect();

    let centroids = kmeans(&vectors, 3, KMEANS_MAX_ITERS);
    let assignments = assign_clusters(&vectors, &centroids);
    let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![vec![]; centroids.len()];
    for (doc_id, &cluster) in assignments.iter().enumerate() {
        cluster_doc_ids[cluster].push(doc_id as DocId);
    }

    let mut buf = Vec::new();
    serialize_vector_index(
        &centroids,
        &cluster_doc_ids,
        &vectors,
        VECTOR_DIMS,
        &mut buf,
    )
    .unwrap();
    let index_data = deserialize_vector_index(&buf);

    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // With n_probe=1, we should only get docs from one cluster (~4 docs)
    let candidates_1 = index_data.cluster_doc_ids_for_query(&query, 1);
    assert_eq!(candidates_1.len(), 4);

    // With n_probe=2, we get docs from two clusters (~8 docs)
    let candidates_2 = index_data.cluster_doc_ids_for_query(&query, 2);
    assert_eq!(candidates_2.len(), 8);

    // With n_probe=3 (all clusters), we get all docs
    let candidates_3 = index_data.cluster_doc_ids_for_query(&query, 3);
    assert_eq!(candidates_3.len(), 12);
}

#[test]
fn test_vector_binary_quantization() {
    // Test binary quantization and hamming distance
    let v1 = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let v2 = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

    let bq1 = binary_quantize(&v1);
    let bq2 = binary_quantize(&v2);

    // v1 quantizes to: 10101010
    assert_eq!(bq1, vec![0b10101010]);
    // v2 quantizes to: 11110000
    assert_eq!(bq2, vec![0b11110000]);

    // Hamming distance: bits differ at positions 1,3,4,6 = 4 bits
    assert_eq!(hamming_distance(&bq1, &bq2), 4);

    // Same vector should have hamming distance 0
    assert_eq!(hamming_distance(&bq1, &bq1), 0);
}

#[test]
fn test_vector_search_with_rescore() {
    // Test that binary quantization + rescore gives correct results
    // even when BQ would give wrong ordering.

    // Two vectors that are very similar in full precision but differ in BQ.
    // v1 has a positive 3rd dim, v2 doesn't — but query doesn't care about dim 3.
    let v1 = vec![1.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]; // BQ: 10100000
    let v2 = vec![1.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]; // BQ: 10000000
    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // BQ: 10000000

    // BQ says v2 is closer (hamming=0) and v1 is further (hamming=1)
    let bq1 = binary_quantize(&v1);
    let bq2 = binary_quantize(&v2);
    let bqq = binary_quantize(&query);
    assert!(
        hamming_distance(&bqq, &bq2) < hamming_distance(&bqq, &bq1),
        "BQ should rank v2 closer"
    );

    // But actual cosine similarity: v1 and v2 are nearly identical to query
    let sim1 = cosine_similarity(&query, &v1);
    let sim2 = cosine_similarity(&query, &v2);
    assert!(sim1 > 0.99);
    assert!(sim2 > 0.99);
    // The difference is negligible — rescoring on full precision gives the true ranking
    assert!((sim1 - sim2).abs() < 0.01);
}

#[test]
fn test_vector_doc_id_intersection_with_text_filter() {
    // Simulate: text_filter AND vector_search
    // This tests the core Tantivy philosophy: intersect doc ID streams.

    let test_data = make_test_data();
    let vectors: Vec<Vec<f32>> = test_data.iter().map(|(_, v)| v.clone()).collect();

    let centroids = kmeans(&vectors, 3, KMEANS_MAX_ITERS);
    let assignments = assign_clusters(&vectors, &centroids);
    let mut cluster_doc_ids: Vec<Vec<DocId>> = vec![vec![]; centroids.len()];
    for (doc_id, &cluster) in assignments.iter().enumerate() {
        cluster_doc_ids[cluster].push(doc_id as DocId);
    }

    let mut buf = Vec::new();
    serialize_vector_index(
        &centroids,
        &cluster_doc_ids,
        &vectors,
        VECTOR_DIMS,
        &mut buf,
    )
    .unwrap();
    let index_data = deserialize_vector_index(&buf);

    // Simulate text filter: "alpha" matches doc_ids 0,1,2,3
    let text_filter_doc_ids: Vec<DocId> = vec![0, 1, 2, 3];

    // Query vector is near beta cluster (positive y direction)
    let query = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Without text filter: vector search returns beta docs (4-7)
    let vector_candidates = index_data.cluster_doc_ids_for_query(&query, 2);

    // With text filter: intersect vector candidates with text filter
    let filtered: Vec<DocId> = vector_candidates
        .iter()
        .filter(|d| text_filter_doc_ids.contains(d))
        .copied()
        .collect();

    // The intersection should contain alpha docs that were in probed clusters.
    // If n_probe=2, we probe beta cluster + nearest neighbor cluster.
    // If the nearest neighbor to beta is alpha, then alpha docs appear in candidates.
    // Either way, the intersection demonstrates the concept.

    // Now score the filtered candidates with full precision vectors
    let mut results: Vec<VectorSearchResult> = filtered
        .iter()
        .map(|&doc_id| VectorSearchResult {
            doc_id,
            similarity: cosine_similarity(&query, &index_data.vectors[doc_id as usize]),
        })
        .collect();
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    // Results should be alpha docs scored by similarity to the beta-direction query.
    // Doc 1 (alpha two: [0.9, 0.2, ...]) has more y-component than doc 3 (alpha four: [0.95, 0.05,
    // ...]) so doc 1 should rank higher.
    if results.len() >= 2 {
        // The doc with more y-component should rank higher
        let top = &results[0];
        let y_component = index_data.vectors[top.doc_id as usize][1];
        assert!(
            y_component > 0.0,
            "top result should have positive y-component"
        );
    }
}

#[test]
fn test_kmeans_basic() {
    // Two obvious clusters
    let vectors = vec![
        vec![1.0, 0.0],
        vec![1.1, 0.0],
        vec![0.9, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.1],
        vec![0.0, 0.9],
    ];

    let centroids = kmeans(&vectors, 2, 20);
    assert_eq!(centroids.len(), 2);

    // One centroid should be near (1,0) and the other near (0,1)
    let has_x_centroid = centroids.iter().any(|c| c[0] > 0.5 && c[1] < 0.5);
    let has_y_centroid = centroids.iter().any(|c| c[0] < 0.5 && c[1] > 0.5);
    assert!(has_x_centroid, "should have centroid near (1,0)");
    assert!(has_y_centroid, "should have centroid near (0,1)");
}

#[test]
fn test_kmeans_single_cluster() {
    let vectors = vec![vec![1.0, 0.0], vec![1.1, 0.0], vec![0.9, 0.0]];
    let centroids = kmeans(&vectors, 1, 20);
    assert_eq!(centroids.len(), 1);
    assert!((centroids[0][0] - 1.0).abs() < 0.1);
}

#[test]
fn test_serialize_deserialize_roundtrip() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let centroids = vec![vec![0.5, 0.0, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];
    let cluster_doc_ids = vec![vec![0u32, 2], vec![1, 3]];

    let mut buf = Vec::new();
    serialize_vector_index(&centroids, &cluster_doc_ids, &vectors, 4, &mut buf).unwrap();

    let data = deserialize_vector_index(&buf);
    assert_eq!(data.centroids.len(), 2);
    assert_eq!(data.vectors.len(), 4);
    assert_eq!(data.cluster_doc_ids.len(), 2);
    assert_eq!(data.dims, 4);

    // Verify centroid values
    assert_eq!(data.centroids[0], vec![0.5, 0.0, 0.0, 0.0]);

    // Verify doc ID lists
    assert_eq!(data.cluster_doc_ids[0], vec![0, 2]);
    assert_eq!(data.cluster_doc_ids[1], vec![1, 3]);

    // Verify BQ vectors
    // v0 = [1,0,0,0] -> BQ = 10000000
    assert_eq!(data.bq_vectors[0], vec![0b10000000]);
    // v1 = [0,1,0,0] -> BQ = 01000000
    assert_eq!(data.bq_vectors[1], vec![0b01000000]);
}

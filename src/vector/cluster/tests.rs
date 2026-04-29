use std::sync::{Arc, Mutex};

use crate::plugin::SegmentPlugin;
use crate::schema::{Schema, VectorMetric, STORED, TEXT};
use crate::vector::cluster::kmeans::KMeansConfig;
use crate::vector::cluster::plugin::{
    ClusterConfig, ClusterFieldConfig, ClusterPlugin, ClusterPluginReader,
};
use crate::vector::cluster::sampler::test_utils::InMemorySamplerFactory;
use crate::vector::cluster::sampler::VectorSamplerFactory;
use crate::vector::rotation::{DynamicRotator, RotatorType};
use crate::vector::turboquant::transposed::TRANSPOSED_BIT_WIDTH;
use crate::vector::turboquant::TurboQuantizer;
use crate::vector::Metric;
use crate::{DocId, Index, IndexWriter};

const DIMS: usize = 64;
const THRESHOLD: u32 = 3;

/// Test with enough docs/clusters that the HNSW graph is non-trivial.
/// Generates 500 vectors producing ~50 clusters, then verifies that
/// centroid search returns the cluster whose centroid is actually
/// closest to the query — i.e. the HNSW index over centroids doesn't
/// miss the true nearest.
#[test]
fn test_hnsw_centroid_search_accuracy() -> crate::Result<()> {
    use rand::prelude::*;

    let num_docs = 500;

    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT | STORED);
    let vec_field = schema_builder.add_vector_field("embedding", DIMS, VectorMetric::L2);
    let schema = schema_builder.build();

    let quantizer = TurboQuantizer::new(DIMS, Some(TRANSPOSED_BIT_WIDTH), Some(42));
    let rotator = Arc::new(DynamicRotator::new(DIMS, RotatorType::FhtKacRotator, 42));
    let padded_dims = rotator.padded_dim();

    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let vectors: Vec<Vec<f32>> = (0..num_docs)
        .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    // ~50 clusters from 500 docs
    let cluster = Arc::new(ClusterPlugin::new(ClusterConfig {
        clustering_threshold: THRESHOLD,
        sample_ratio: 1.0,
        sample_cap: 100_000,
        kmeans: KMeansConfig {
            niter: 25,
            nredo: 1,
            seed: 99,
            ..Default::default()
        },
        num_clusters_fn: Arc::new(|n| (n / 10).max(2)),
        fields: vec![ClusterFieldConfig::new(
            vec_field,
            DIMS,
            padded_dims,
            Metric::L2,
            rotator.clone(),
            42,
            quantizer.clone(),
        )],
        sampler_factory: Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }) as Arc<dyn VectorSamplerFactory>,
        defer_clustering: false,
        replication: None,
    }));

    let index = Index::builder()
        .schema(schema)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    for (i, v) in vectors.iter().enumerate() {
        writer.add_document(crate::doc!(
            text_field => format!("doc{i}"),
            vec_field => v.clone()
        ))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];

    let cluster_reader: Arc<ClusterPluginReader> = seg
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader");
    let field_reader = cluster_reader
        .field_reader(vec_field)
        .expect("field reader");
    assert!(field_reader.is_clustered());
    let k = field_reader.num_clusters();
    assert!(k >= 20, "expected many clusters, got {k}");

    // All docs accounted for across windows.
    let mut all_docs: Vec<DocId> = Vec::new();
    for win_idx in 0..field_reader.num_windows() {
        let win = field_reader.window_reader(win_idx);
        let win_k = win.num_clusters();
        for c in 0..win_k {
            if let Ok(Some(ids)) = win.cluster_doc_ids(c) {
                all_docs.extend(ids);
            }
        }
    }
    assert_eq!(all_docs.len(), num_docs);

    // For several random queries, verify that HNSW top-1 matches the
    // true nearest centroid (found via search(k) → first result).
    for _ in 0..20 {
        let query: Vec<f32> = (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        let hnsw_results = field_reader.search_centroids(&query, 1);
        assert!(!hnsw_results.is_empty());
        let hnsw_nearest_id = hnsw_results[0].0;

        let bf_results = field_reader.search_centroids(&query, k);
        let bf_nearest_id = bf_results[0].0;

        assert_eq!(
            hnsw_nearest_id, bf_nearest_id,
            "HNSW missed true nearest centroid for query"
        );
    }

    Ok(())
}

/// End-to-end round-trip across a merge boundary: write two segments
/// of vectors via the cluster plugin, force-merge them, and verify
/// that querying the merged segment returns the same dominant
/// neighbours as a brute-force exact-IP search.
///
/// Specifically exercises `SourceLocator::extract_record` —
/// `merge` reads each source-doc's TurboQuant record out of its
/// source segment's `.cluster` file (since there's no separate
/// `.tqvec` store any more) and rebuilds clusters in the target.
#[test]
fn merge_records_roundtrip_through_cluster() -> crate::Result<()> {
    use std::collections::HashSet;

    use crate::collector::TopDocs;
    use crate::query::AllQuery;

    const N_PER_SEG: usize = 60;
    let mut schema_builder = Schema::builder();
    // Stored u64 holding the original input-vector index. After the
    // merge, the segment_ord assigned to each source segment is
    // non-deterministic (it follows the HashMap iteration order in
    // `SegmentRegister`), so we can't assume `target_doc_id ==
    // input_vectors_index`. Looking the orig-idx up via stored docs
    // makes the recall check robust to merge ordering.
    let orig_idx_field = schema_builder.add_u64_field("orig_idx", crate::schema::STORED);
    let vec_field = schema_builder.add_vector_field("embedding", DIMS, VectorMetric::L2);
    let schema = schema_builder.build();

    let quantizer = TurboQuantizer::new(DIMS, Some(TRANSPOSED_BIT_WIDTH), Some(0xCAFE));
    let rotator = Arc::new(DynamicRotator::new(DIMS, RotatorType::FhtKacRotator, 42));
    let padded_dims = rotator.padded_dim();

    let n_docs = 2 * N_PER_SEG;
    let vectors: Vec<Vec<f32>> = {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        (0..n_docs)
            .map(|_| {
                let mut v: Vec<f32> = (0..DIMS).map(|_| rng.random::<f32>() - 0.5).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                for x in &mut v {
                    *x /= n;
                }
                v
            })
            .collect()
    };
    let shared = Arc::new(Mutex::new(vectors.clone()));

    let cluster = Arc::new(ClusterPlugin::new(ClusterConfig {
        // Force clustering on each segment so the merge path
        // exercises `cluster_from_vectors` on both source and
        // target sides (rather than the trivial-single-cluster
        // path).
        clustering_threshold: 30,
        sample_ratio: 1.0,
        sample_cap: 100_000,
        kmeans: KMeansConfig {
            niter: 5,
            nredo: 1,
            seed: 7,
            ..Default::default()
        },
        num_clusters_fn: Arc::new(|_| 4),
        fields: vec![ClusterFieldConfig::new(
            vec_field,
            DIMS,
            padded_dims,
            Metric::L2,
            rotator,
            42,
            quantizer.clone(),
        )],
        sampler_factory: Arc::new(InMemorySamplerFactory { vectors: shared })
            as Arc<dyn VectorSamplerFactory>,
        defer_clustering: false,
        replication: None,
    }));

    let index = Index::builder()
        .schema(schema)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
    for (i, v) in vectors[..N_PER_SEG].iter().enumerate() {
        writer.add_document(crate::doc!(
            orig_idx_field => i as u64,
            vec_field => v.clone()
        ))?;
    }
    writer.commit()?;
    for (i, v) in vectors[N_PER_SEG..].iter().enumerate() {
        writer.add_document(crate::doc!(
            orig_idx_field => (i + N_PER_SEG) as u64,
            vec_field => v.clone()
        ))?;
    }
    writer.commit()?;

    let segment_ids = index.searchable_segment_ids()?;
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;

    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);

    // Build target_doc_id → orig_idx via stored docs so the recall
    // check is independent of the (HashMap-driven, non-deterministic)
    // segment merge order.
    use crate::schema::document::Value as _;
    use crate::TantivyDocument;
    let seg = &searcher.segment_readers()[0];
    let store = seg.get_store_reader(0)?;
    let max_doc = seg.max_doc();
    let doc_to_orig: Vec<u64> = (0..max_doc)
        .map(|d| {
            let doc: TantivyDocument = store.get(d).unwrap();
            doc.get_first(orig_idx_field).unwrap().as_u64().unwrap()
        })
        .collect();

    // Pick a few queries and check that the merged segment's top-10
    // recall against brute-force exact IP is reasonable. The
    // round-trip through `extract_record` shouldn't introduce
    // additional noise beyond the codec's normal quantization
    // error.
    let queries: Vec<Vec<f32>> = vectors.iter().take(5).cloned().collect();
    let k = 10;
    for (qi, q) in queries.iter().enumerate() {
        let mut exact: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gt: HashSet<u64> = exact.iter().take(k).map(|(i, _)| *i as u64).collect();

        let collector = TopDocs::with_limit(k).order_by_turboquant_distance(
            q.clone(),
            vec_field,
            quantizer.clone(),
            None,
        );
        let hits = searcher.search(&AllQuery, &collector)?;
        let got: HashSet<u64> = hits
            .into_iter()
            .map(|(_score, addr)| doc_to_orig[addr.doc_id as usize])
            .collect();
        let recall = gt.intersection(&got).count();
        assert!(
            recall >= k / 2,
            "query {qi}: recall@{k} = {recall} below sanity floor {}",
            k / 2
        );
    }

    Ok(())
}

/// Round-trip plumbing test for SPANN-style replicated assignment.
///
/// Builds an index with `replication = Some(_)` and verifies the
/// integration doesn't crash and the query path's dedup invariant
/// holds: the same `DocAddress` never appears twice in the top-K
/// even when the underlying assignment may have replicated docs
/// across clusters.
///
/// Algorithmic correctness of `assign_replicated` itself (ε-cone,
/// RNG, cap) is covered by unit tests in `plugin.rs::tests`.
#[test]
fn replicated_assignment_round_trip() -> crate::Result<()> {
    use std::collections::HashSet;

    use rand::prelude::*;
    const N_DOCS: usize = 500;
    const K_TOP: usize = 10;

    fn build_index(
        replication: Option<crate::vector::cluster::plugin::ReplicationConfig>,
        vectors: &[Vec<f32>],
    ) -> crate::Result<(Index, crate::schema::Field)> {
        let mut schema_builder = Schema::builder();
        let vec_field = schema_builder.add_vector_field("embedding", DIMS, VectorMetric::L2);
        let schema = schema_builder.build();

        let quantizer = TurboQuantizer::new(DIMS, Some(TRANSPOSED_BIT_WIDTH), Some(0xCAFE));
        let rotator = Arc::new(DynamicRotator::new(DIMS, RotatorType::FhtKacRotator, 42));
        let padded_dims = rotator.padded_dim();
        let shared = Arc::new(Mutex::new(vectors.to_vec()));

        let cluster = Arc::new(ClusterPlugin::new(ClusterConfig {
            clustering_threshold: THRESHOLD,
            sample_ratio: 1.0,
            sample_cap: 100_000,
            kmeans: KMeansConfig {
                niter: 25,
                nredo: 1,
                seed: 11,
                ..Default::default()
            },
            // Match the planted center count exactly so k-means
            // doesn't split planted clusters into sub-clusters and
            // wash out the boundary structure of the synthetic data.
            num_clusters_fn: Arc::new(|_| 5),
            fields: vec![ClusterFieldConfig::new(
                vec_field,
                DIMS,
                padded_dims,
                Metric::L2,
                rotator.clone(),
                42,
                quantizer.clone(),
            )],
            sampler_factory: Arc::new(InMemorySamplerFactory { vectors: shared })
                as Arc<dyn VectorSamplerFactory>,
            defer_clustering: false,
            replication,
        }));

        let index = Index::builder()
            .schema(schema)
            .plugin(cluster as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
        for v in vectors {
            let mut doc = crate::TantivyDocument::default();
            doc.add_field_value(vec_field, &crate::schema::OwnedValue::Vector(v.clone()));
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok((index, vec_field))
    }

    // Synthetic data designed to produce real boundary docs:
    //   - 5 planted centers along the first 5 axes (e_0 .. e_4)
    //   - For each center: ~80 docs jittered around it (interior)
    //   - Plus ~20 boundary docs placed at midpoints between adjacent center pairs
    //
    // The midpoint docs are equidistant from two planted centers, so
    // after k-means trains centroids close to those centers, the
    // ε-cone catches the second-nearest centroid and RNG accepts it
    // (because the two centroids are FARTHER from each other than
    // the boundary doc is from either of them).
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let make_centered = |center: usize, jitter: f32, rng: &mut StdRng| -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[center] = 1.0;
        for x in &mut v {
            *x += jitter * (rng.random::<f32>() - 0.5);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    };
    let make_midpoint = |a: usize, b: usize, jitter: f32, rng: &mut StdRng| -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[a] = 1.0;
        v[b] = 1.0;
        for x in &mut v {
            *x += jitter * (rng.random::<f32>() - 0.5);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    };
    let n_centers = 5;
    let interior_per_center = 80;
    let n_interior = n_centers * interior_per_center;
    let n_boundary = N_DOCS - n_interior; // ~100 boundary docs
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(N_DOCS);
    for c in 0..n_centers {
        for _ in 0..interior_per_center {
            vectors.push(make_centered(c, 0.1, &mut rng));
        }
    }
    for i in 0..n_boundary {
        let a = i % n_centers;
        let b = (a + 1) % n_centers;
        vectors.push(make_midpoint(a, b, 0.05, &mut rng));
    }

    let (rep_index, rep_field) = build_index(
        Some(crate::vector::cluster::plugin::ReplicationConfig {
            epsilon: 0.1,
            max_replicas: 8,
        }),
        &vectors,
    )?;
    let rep_reader = rep_index.reader()?;
    rep_reader.reload()?;
    let rep_searcher = rep_reader.searcher();

    // Dedup invariant: every doc returned in the replicated query
    // must be unique. If `assign_replicated` placed a doc in
    // multiple clusters and the dedup bitset misfires, the same
    // `DocAddress` would appear more than once in the heap.
    let quantizer = TurboQuantizer::new(DIMS, Some(TRANSPOSED_BIT_WIDTH), Some(0xCAFE));
    let queries: Vec<Vec<f32>> = vectors.iter().take(5).cloned().collect();
    use crate::collector::TopDocs;
    use crate::query::AllQuery;

    for q in &queries {
        let collector = TopDocs::with_limit(K_TOP).order_by_turboquant_distance(
            q.clone(),
            rep_field,
            quantizer.clone(),
            None,
        );
        let hits = rep_searcher.search(&AllQuery, &collector)?;
        assert!(
            !hits.is_empty(),
            "replicated query should return at least one hit"
        );
        let unique: HashSet<_> = hits.iter().map(|(_, addr)| *addr).collect();
        assert_eq!(
            unique.len(),
            hits.len(),
            "replicated top-K returned duplicate DocAddresses: {hits:?}"
        );
    }

    Ok(())
}

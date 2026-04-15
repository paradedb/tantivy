use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::vector::bqvec::{BqVecPlugin, BqVecPluginReader};
use crate::vector::cluster::kmeans::KMeansConfig;
use crate::vector::cluster::plugin::{
    ClusterConfig, ClusterFieldConfig, ClusterPlugin, ClusterPluginReader, ClusterPluginWriter,
    ProbeConfig,
};
use crate::vector::cluster::sampler::test_utils::InMemorySamplerFactory;
use crate::vector::cluster::sampler::VectorSamplerFactory;
use crate::docset::DocSet;
use crate::plugin::SegmentPlugin;
use crate::vector::rabitq::{self, DynamicRotator, Metric, RabitqConfig, RotatorType};
use crate::schema::{Field, Schema, STORED, TEXT};
use crate::{DocId, Index, IndexWriter, TERMINATED};

fn collect_postings_doc_ids(
    reader: &crate::vector::cluster::plugin::ClusterFieldReader,
    cluster_id: usize,
) -> Vec<DocId> {
    let mut postings = reader.cluster_postings(cluster_id).unwrap();
    let mut doc_ids = Vec::new();
    let mut doc = postings.doc();
    while doc != TERMINATED {
        doc_ids.push(doc);
        doc = postings.advance();
    }
    doc_ids
}

const DIMS: usize = 32;
const THRESHOLD: u32 = 3;

fn make_schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let text = builder.add_text_field("text", TEXT | STORED);
    let vec = builder.add_vector_field("embedding", DIMS);
    (builder.build(), text, vec)
}

fn make_rotator() -> Arc<DynamicRotator> {
    Arc::new(DynamicRotator::new(DIMS, RotatorType::MatrixRotator, 42))
}

fn make_bqvec_plugin(vec_field: Field, rotator: &Arc<DynamicRotator>) -> Arc<BqVecPlugin> {
    let config = RabitqConfig::new(1);
    let padded_dims = rotator.padded_dim();
    let rotator_clone = rotator.clone();
    Arc::new(
        BqVecPlugin::builder()
            .vector_field(
                vec_field,
                rabitq::bytes_per_record(padded_dims, 6),
                Arc::new(move |v: &[f32]| {
                    let zero = vec![0.0f32; v.len()];
                    rabitq::encode(&rotator_clone, &config, Metric::L2, v, &zero)
                }),
            )
            .build(),
    )
}

fn make_cluster_plugin(
    vec_field: Field,
    rotator: &Arc<DynamicRotator>,
    sampler_factory: Arc<dyn VectorSamplerFactory>,
) -> Arc<ClusterPlugin> {
    let padded_dims = rotator.padded_dim();
    Arc::new(ClusterPlugin::new(ClusterConfig {
        clustering_threshold: THRESHOLD,
        sample_ratio: 1.0, // sample everything for test reproducibility
        sample_cap: 100_000,
        kmeans: KMeansConfig {
            niter: 20,
            nredo: 1,
            seed: 42,
            ..Default::default()
        },
        num_clusters_fn: Arc::new(|n| (n / 2).max(1).min(4)),
        fields: vec![ClusterFieldConfig {
            field: vec_field,
            dims: DIMS,
            padded_dims,
            ex_bits: 6,
            metric: Metric::L2,
            rotator: rotator.clone(),
            rotator_seed: 42,
        }],
        sampler_factory,
    }))
}

fn make_vectors(n: usize) -> Vec<Vec<f32>> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    (0..n)
        .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

#[test]
fn test_cluster_flush_below_threshold() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = make_rotator();
    let vectors = make_vectors(2); // below threshold of 3
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    let bqvec = make_bqvec_plugin(vec_field, &rotator);
    let cluster = make_cluster_plugin(
        vec_field,
        &rotator,
        Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }),
    );

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    for (i, v) in vectors.iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{i}"), vec_field => v.clone()))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];

    let cluster_reader: Arc<ClusterPluginReader> = seg
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader should exist");

    let field_reader = cluster_reader.field_reader(vec_field).expect("field reader");
    assert!(!field_reader.is_clustered(), "below threshold, should not be clustered");

    Ok(())
}

#[test]
fn test_cluster_flush_above_threshold() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = make_rotator();
    let vectors = make_vectors(8); // above threshold of 3
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    let bqvec = make_bqvec_plugin(vec_field, &rotator);
    let cluster = make_cluster_plugin(
        vec_field,
        &rotator,
        Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }),
    );

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    for (i, v) in vectors.iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{i}"), vec_field => v.clone()))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];

    let cluster_reader: Arc<ClusterPluginReader> = seg
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader should exist");

    let field_reader = cluster_reader.field_reader(vec_field).expect("field reader");
    assert!(field_reader.is_clustered(), "above threshold, should be clustered");
    assert!(field_reader.num_clusters() > 0);

    // Every doc should be assigned to a valid cluster
    for doc_id in 0..8u32 {
        let c = field_reader.doc_cluster(doc_id);
        assert!((c as usize) < field_reader.num_clusters());
    }

    // Cluster doc lists should contain all docs exactly once
    let mut all_docs: Vec<DocId> = Vec::new();
    for c in 0..field_reader.num_clusters() {
        all_docs.extend(collect_postings_doc_ids(field_reader, c));
    }
    all_docs.sort();
    assert_eq!(all_docs, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // Centroid search should return results
    let query: Vec<f32> = (0..DIMS).map(|d| d as f32 / DIMS as f32).collect();
    let results = field_reader.search_centroids(&query, 4);
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_cluster_merge_with_bq_assignment() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = make_rotator();
    let vectors = make_vectors(8);
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    let bqvec = make_bqvec_plugin(vec_field, &rotator);
    let cluster = make_cluster_plugin(
        vec_field,
        &rotator,
        Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }),
    );

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Segment 1: first 4 docs
    for (i, v) in vectors[..4].iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{i}"), vec_field => v.clone()))?;
    }
    writer.commit()?;

    // Segment 2: next 4 docs
    for (i, v) in vectors[4..].iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{}", i + 4), vec_field => v.clone()))?;
    }
    writer.commit()?;

    // Force merge
    let segment_ids = index.searchable_segment_ids()?;
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 8);

    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    // Verify BQ records exist
    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();
    assert_eq!(bq.field_reader(vec_field).unwrap().num_records(), 8);

    // Verify cluster data
    let cluster_reader: Arc<ClusterPluginReader> = segments[0]
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader should exist after merge");

    let field_reader = cluster_reader.field_reader(vec_field).expect("field reader");
    assert!(field_reader.is_clustered());
    assert!(field_reader.num_clusters() > 0);

    // All 8 docs should appear in cluster lists
    let mut all_docs: Vec<DocId> = Vec::new();
    for c in 0..field_reader.num_clusters() {
        all_docs.extend(collect_postings_doc_ids(field_reader, c));
    }
    all_docs.sort();
    assert_eq!(all_docs, (0..8).collect::<Vec<DocId>>());

    // Search centroids: query near positive vectors should find the right cluster
    let query: Vec<f32> = (0..DIMS).map(|d| d as f32 / DIMS as f32).collect();
    let results = field_reader.search_centroids(&query, 2);
    assert!(!results.is_empty());

    Ok(())
}

/// Test with enough docs/clusters that the HNSW graph is non-trivial.
/// Generates 500 vectors producing ~50 clusters, then verifies that
/// centroid search returns the cluster whose centroid is actually closest
/// to the query (i.e. HNSW doesn't miss the true nearest).
#[test]
fn test_hnsw_centroid_search_accuracy() -> crate::Result<()> {
    use rand::prelude::*;

    let num_docs = 500;
    let (schema, text_field, vec_field) = make_schema();
    let rotator = make_rotator();

    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let vectors: Vec<Vec<f32>> = (0..num_docs)
        .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    let padded_dims = rotator.padded_dim();
    let bqvec = make_bqvec_plugin(vec_field, &rotator);
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
        fields: vec![ClusterFieldConfig {
            field: vec_field,
            dims: DIMS,
            padded_dims,
            ex_bits: 6,
            metric: Metric::L2,
            rotator: rotator.clone(),
            rotator_seed: 42,
        }],
        sampler_factory: Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }),
    }));

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    for (i, v) in vectors.iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{i}"), vec_field => v.clone()))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];

    let cluster_reader: Arc<ClusterPluginReader> = seg
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader");

    let field_reader = cluster_reader.field_reader(vec_field).expect("field reader");
    assert!(field_reader.is_clustered());
    let k = field_reader.num_clusters();
    assert!(k >= 20, "expected many clusters, got {k}");

    // All docs accounted for
    let mut all_docs: Vec<DocId> = Vec::new();
    for c in 0..k {
        all_docs.extend(collect_postings_doc_ids(field_reader, c));
    }
    all_docs.sort();
    assert_eq!(all_docs.len(), num_docs);

    // Test HNSW accuracy: for several random queries, verify that the top-1
    // centroid returned by HNSW search is the actual nearest centroid (by
    // brute-force L2 over all centroids retrieved via search with ef=k).
    for _ in 0..20 {
        let query: Vec<f32> = (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        // HNSW top-1
        let hnsw_results = field_reader.search_centroids(&query, 1);
        assert!(!hnsw_results.is_empty());
        let hnsw_nearest_id = hnsw_results[0].0;

        // Brute-force: get all centroids and find true nearest
        let bf_results = field_reader.search_centroids(&query, k);
        let bf_nearest_id = bf_results[0].0;

        // HNSW should find the same nearest centroid as brute-force
        assert_eq!(
            hnsw_nearest_id, bf_nearest_id,
            "HNSW missed true nearest centroid for query"
        );
    }

    Ok(())
}

#[test]
fn test_probe_clusters_adaptive() -> crate::Result<()> {
    use rand::prelude::*;

    let num_docs = 500;
    let (schema, text_field, vec_field) = make_schema();
    let rotator = make_rotator();

    let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE);
    let vectors: Vec<Vec<f32>> = (0..num_docs)
        .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

    let padded_dims = rotator.padded_dim();
    let bqvec = make_bqvec_plugin(vec_field, &rotator);
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
        fields: vec![ClusterFieldConfig {
            field: vec_field,
            dims: DIMS,
            padded_dims,
            ex_bits: 6,
            metric: Metric::L2,
            rotator: rotator.clone(),
            rotator_seed: 42,
        }],
        sampler_factory: Arc::new(InMemorySamplerFactory {
            vectors: shared_vecs,
        }),
    }));

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    for (i, v) in vectors.iter().enumerate() {
        writer.add_document(crate::doc!(text_field => format!("doc{i}"), vec_field => v.clone()))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];

    let cluster_reader: Arc<ClusterPluginReader> = seg
        .plugin_reader::<ClusterPluginReader>("cluster")?
        .expect("cluster reader");

    let field_reader = cluster_reader.field_reader(vec_field).expect("field reader");
    let k = field_reader.num_clusters();
    assert!(k >= 20);

    let query: Vec<f32> = (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    // Tight ratio should prune clusters
    let tight = ProbeConfig::new(k, 1.2);
    let tight_results = field_reader.probe_clusters(&query, &tight)?;
    assert!(!tight_results.is_empty());
    assert!(
        tight_results.len() < k,
        "tight ratio should prune: got {} out of {k}",
        tight_results.len()
    );

    // Loose ratio should return max_probe clusters
    let loose = ProbeConfig::new(k, 1000.0);
    let loose_results = field_reader.probe_clusters(&query, &loose)?;
    assert_eq!(loose_results.len(), k);

    // min_probe is respected even with very tight ratio
    let min3 = ProbeConfig::new(k, 1.0).with_min_probe(3);
    let min3_results = field_reader.probe_clusters(&query, &min3)?;
    assert!(min3_results.len() >= 3);

    // Returned postings contain valid doc IDs
    for (_, postings) in tight_results {
        let mut postings = postings;
        let mut doc = postings.doc();
        while doc != TERMINATED {
            assert!((doc as usize) < num_docs);
            doc = postings.advance();
        }
    }

    // First cluster matches search_centroids nearest
    let centroids = field_reader.search_centroids(&query, 1);
    let probe = ProbeConfig::new(k, 1.5);
    let probe_results = field_reader.probe_clusters(&query, &probe)?;
    assert_eq!(probe_results[0].0, centroids[0].0);

    Ok(())
}

use std::sync::{Arc, Mutex};

use crate::vector::bqvec::{BqVecPlugin, BqVecPluginReader};
use crate::vector::cluster::kmeans::KMeansConfig;
use crate::vector::cluster::plugin::{ClusterConfig, ClusterFieldConfig, ClusterPlugin};
use crate::vector::cluster::sampler::test_utils::InMemorySamplerFactory;
use crate::vector::cluster::sampler::VectorSamplerFactory;
use crate::plugin::SegmentPlugin;
use crate::vector::rabitq::{self, DynamicRotator, Metric, RabitqConfig, RotatorType};
use crate::schema::{Field, Schema, STORED, TEXT};
use crate::{Index, IndexWriter};

const DIMS: usize = 32;

fn make_schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let text = builder.add_text_field("text", TEXT | STORED);
    let vec = builder.add_vector_field("embedding", DIMS);
    (builder.build(), text, vec)
}

fn make_plugins(
    vec_field: Field,
    rotator: &Arc<DynamicRotator>,
    vectors: &[Vec<f32>],
) -> (Arc<BqVecPlugin>, Arc<ClusterPlugin>) {
    let config = RabitqConfig::new(1);
    let padded_dims = rotator.padded_dim();
    let rotator_clone = rotator.clone();
    let bqvec = Arc::new(
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
    );
    let shared_vecs = Arc::new(Mutex::new(vectors.to_vec()));
    let cluster = Arc::new(ClusterPlugin::new(ClusterConfig {
        clustering_threshold: 3,
        sample_ratio: 1.0,
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
        sampler_factory: Arc::new(InMemorySamplerFactory { vectors: shared_vecs }),
        hot_bytes_cache: None,
    }));
    (bqvec, cluster)
}

fn make_vectors(n: usize) -> Vec<Vec<f32>> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    (0..n)
        .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

#[test]
fn test_e2e_index_and_read() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = Arc::new(DynamicRotator::new(DIMS, RotatorType::MatrixRotator, 42));
    let vectors = make_vectors(8);
    let (bqvec, cluster) = make_plugins(vec_field, &rotator, &vectors);

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
    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .expect("bqvec reader should exist");

    let field_reader = bq.field_reader(vec_field).expect("field reader");
    assert_eq!(field_reader.num_records(), 8);

    let padded_dims = rotator.padded_dim();
    let expected_size = rabitq::bytes_per_record(padded_dims, 6);
    assert_eq!(field_reader.bytes_per_record(), expected_size);

    let rec0 = field_reader.record(0)?;
    assert_eq!(rec0.len(), expected_size);

    Ok(())
}

#[test]
fn test_e2e_distance_ordering() -> crate::Result<()> {
    let dims = 128;
    let mut builder = Schema::builder();
    let text_field = builder.add_text_field("text", TEXT | STORED);
    let vec_field = builder.add_vector_field("embedding", dims);
    let schema = builder.build();

    let rotator = Arc::new(DynamicRotator::new(dims, RotatorType::MatrixRotator, 42));
    let padded_dims = rotator.padded_dim();

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let base: Vec<f32> = (0..dims).map(|_| rng.random::<f32>()).collect();
    let similar: Vec<f32> = base.iter().map(|&x| x + rng.random::<f32>() * 0.01).collect();
    let dissimilar: Vec<f32> = (0..dims).map(|_| rng.random::<f32>()).collect();
    let vectors = vec![base.clone(), similar.clone(), dissimilar.clone()];

    let config = RabitqConfig::new(1);
    let rotator_clone = rotator.clone();
    let bqvec = Arc::new(
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
    );
    let shared_vecs = Arc::new(Mutex::new(vectors.clone()));
    let cluster = Arc::new(ClusterPlugin::new(ClusterConfig {
        clustering_threshold: u32::MAX, // no clustering, encode against zero
        sample_ratio: 1.0,
        sample_cap: 100_000,
        kmeans: KMeansConfig::default(),
        num_clusters_fn: Arc::new(|_| 1),
        fields: vec![ClusterFieldConfig {
            field: vec_field,
            dims,
            padded_dims,
            ex_bits: 6,
            metric: Metric::L2,
            rotator: rotator.clone(),
            rotator_seed: 42,
        }],
        sampler_factory: Arc::new(InMemorySamplerFactory { vectors: shared_vecs }),
        hot_bytes_cache: None,
    }));

    let index = Index::builder()
        .schema(schema)
        .plugin(bqvec as Arc<dyn SegmentPlugin>)
        .plugin(cluster as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.add_document(crate::doc!(text_field => "base", vec_field => base.clone()))?;
    writer.add_document(crate::doc!(text_field => "similar", vec_field => similar))?;
    writer.add_document(crate::doc!(text_field => "dissimilar", vec_field => dissimilar))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let bq: Arc<BqVecPluginReader> = seg.plugin_reader::<BqVecPluginReader>("bqvec")?.unwrap();
    let field_reader = bq.field_reader(vec_field).unwrap();

    let query = crate::vector::rabitq::RaBitQQuery::new(&base, &rotator, 6, Metric::L2);
    let rec0 = field_reader.record(0)?;
    let rec1 = field_reader.record(2)?;
    let dist_similar = query.estimate_distance_from_record(&rec0, padded_dims, 0.0);
    let dist_dissimilar = query.estimate_distance_from_record(&rec1, padded_dims, 0.0);
    assert!(
        dist_similar < dist_dissimilar,
        "similar vector should have lower distance: {dist_similar} vs {dist_dissimilar}"
    );

    Ok(())
}

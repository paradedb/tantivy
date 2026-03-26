use std::sync::Arc;

use crate::bqvec::{BqVecPlugin, BqVecPluginReader};
use crate::plugin::SegmentPlugin;
use crate::rabitq::{self, DynamicRotator, Metric, RabitqConfig, RotatorType};
use crate::schema::{Schema, STORED, TEXT};
use crate::{Index, IndexWriter};

fn make_schema() -> (Schema, crate::schema::Field, crate::schema::Field) {
    let mut builder = Schema::builder();
    let text = builder.add_text_field("text", TEXT | STORED);
    let vec = builder.add_vector_field("embedding", 32);
    (builder.build(), text, vec)
}

/// Build a plugin with rabitq encoding for the given vector field.
fn make_plugin(vec_field: crate::schema::Field, rotator: &Arc<DynamicRotator>) -> Arc<BqVecPlugin> {
    let config = RabitqConfig::new(1); // 1-bit binary
    let padded_dims = rotator.padded_dim();
    let rotator_clone = rotator.clone();
    Arc::new(
        BqVecPlugin::builder()
            .vector_field(
                vec_field,
                rabitq::bytes_per_record(padded_dims, 0),
                Arc::new(move |v: &[f32]| rabitq::encode(&rotator_clone, &config, Metric::L2, v)),
            )
            .build(),
    )
}

#[test]
fn test_e2e_index_and_read() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = Arc::new(DynamicRotator::new(32, RotatorType::MatrixRotator, 42));
    let plugin = make_plugin(vec_field, &rotator);

    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Vectors as document fields — no staging queue!
    let v0: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
    let v1: Vec<f32> = (0..32).map(|i| -(i as f32) / 32.0).collect();
    let v2: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    writer.add_document(crate::doc!(text_field => "hello", vec_field => v0.clone()))?;
    writer.add_document(crate::doc!(text_field => "world", vec_field => v1.clone()))?;
    writer.add_document(crate::doc!(text_field => "foo", vec_field => v2.clone()))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .expect("bqvec reader should exist");

    let field_reader = bq.field_reader(vec_field).expect("field reader");
    assert_eq!(field_reader.num_records(), 3);

    // Records should be non-empty and the right size
    let padded_dims = rotator.padded_dim();
    let expected_size = rabitq::bytes_per_record(padded_dims, 0);
    assert_eq!(field_reader.bytes_per_record(), expected_size);

    let rec0 = field_reader.record(0)?;
    assert_eq!(rec0.len(), expected_size);

    Ok(())
}

#[test]
fn test_e2e_distance_ordering() -> crate::Result<()> {
    // Use 128 dims for better quantization quality
    let mut builder = Schema::builder();
    let text_field = builder.add_text_field("text", TEXT | STORED);
    let vec_field = builder.add_vector_field("embedding", 128);
    let schema = builder.build();

    let rotator = Arc::new(DynamicRotator::new(128, RotatorType::MatrixRotator, 42));
    let plugin = make_plugin(vec_field, &rotator);

    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Three vectors: query-like, similar-to-query, dissimilar-to-query
    let query_vec: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
    let similar_vec: Vec<f32> = (0..128).map(|i| i as f32 / 128.0 + 0.01).collect();
    let dissimilar_vec: Vec<f32> = (0..128).map(|i| -(i as f32) / 128.0).collect();

    writer.add_document(crate::doc!(text_field => "similar", vec_field => similar_vec))?;
    writer.add_document(crate::doc!(text_field => "dissimilar", vec_field => dissimilar_vec))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let bq: Arc<BqVecPluginReader> = seg.plugin_reader::<BqVecPluginReader>("bqvec")?.unwrap();
    let field_reader = bq.field_reader(vec_field).unwrap();
    let padded_dims = rotator.padded_dim();

    let query = rabitq::prepare_query(&rotator, &query_vec, 0, Metric::L2);

    let rec0 = field_reader.record(0)?;
    let rec1 = field_reader.record(1)?;
    let dist_similar = query.estimate_distance_from_record(&rec0, padded_dims);
    let dist_dissimilar = query.estimate_distance_from_record(&rec1, padded_dims);

    assert!(
        dist_similar < dist_dissimilar,
        "similar vector should be closer: {dist_similar} vs {dist_dissimilar}"
    );

    Ok(())
}

#[test]
fn test_e2e_merge() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = Arc::new(DynamicRotator::new(32, RotatorType::MatrixRotator, 42));
    let plugin = make_plugin(vec_field, &rotator);

    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Segment 1
    let v0: Vec<f32> = vec![1.0; 32];
    writer.add_document(crate::doc!(text_field => "a", vec_field => v0))?;
    writer.commit()?;

    // Segment 2
    let v1: Vec<f32> = vec![2.0; 32];
    writer.add_document(crate::doc!(text_field => "b", vec_field => v1))?;
    writer.commit()?;

    // Force merge
    let segment_ids = index.searchable_segment_ids()?;
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 2);

    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();
    let field_reader = bq.field_reader(vec_field).unwrap();
    assert_eq!(field_reader.num_records(), 2);

    // Both records should be readable
    let rec0 = field_reader.record(0)?;
    let rec1 = field_reader.record(1)?;
    assert_eq!(rec0.len(), rec1.len());

    Ok(())
}

#[test]
fn test_e2e_merge_with_deletes() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = Arc::new(DynamicRotator::new(32, RotatorType::MatrixRotator, 42));
    let plugin = make_plugin(vec_field, &rotator);

    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Segment 1: two docs
    writer.add_document(crate::doc!(text_field => "keep", vec_field => vec![1.0f32; 32]))?;
    writer.add_document(crate::doc!(text_field => "deleteme", vec_field => vec![2.0f32; 32]))?;
    writer.commit()?;

    // Segment 2: one doc
    writer.add_document(crate::doc!(text_field => "also", vec_field => vec![3.0f32; 32]))?;
    writer.commit()?;

    // Delete
    writer.delete_term(crate::Term::from_field_text(text_field, "deleteme"));
    writer.commit()?;

    // Merge
    let segment_ids = index.searchable_segment_ids()?;
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 2);

    let bq: Arc<BqVecPluginReader> = searcher.segment_readers()[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();
    let field_reader = bq.field_reader(vec_field).unwrap();
    assert_eq!(field_reader.num_records(), 2);

    Ok(())
}

#[test]
fn test_e2e_zero_fill_missing_vector() -> crate::Result<()> {
    let (schema, text_field, vec_field) = make_schema();
    let rotator = Arc::new(DynamicRotator::new(32, RotatorType::MatrixRotator, 42));
    let plugin = make_plugin(vec_field, &rotator);

    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Doc with vector
    writer.add_document(crate::doc!(text_field => "has_vec", vec_field => vec![1.0f32; 32]))?;
    // Doc WITHOUT vector — should get zero-filled record
    writer.add_document(crate::doc!(text_field => "no_vec"))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let bq: Arc<BqVecPluginReader> = searcher.segment_readers()[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();
    let field_reader = bq.field_reader(vec_field).unwrap();
    assert_eq!(field_reader.num_records(), 2);

    // Both records should be readable
    let rec0 = field_reader.record(0)?;
    let rec1 = field_reader.record(1)?;
    assert_eq!(rec0.len(), rec1.len());

    // The zero-filled record should be all zeros
    assert!(
        rec1.iter().all(|&b| b == 0),
        "missing vector should be zero-filled"
    );

    Ok(())
}

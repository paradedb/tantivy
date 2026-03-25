//! Integration tests for the binary quantized vector storage plugin.

use std::sync::Arc;

use tantivy::bqvec::{BqVecPlugin, BqVecPluginReader};
use tantivy::plugin::SegmentPlugin;
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{Index, IndexWriter};

fn make_schema() -> (Schema, tantivy::schema::Field) {
    let mut builder = Schema::builder();
    let text = builder.add_text_field("text", TEXT | STORED);
    (builder.build(), text)
}

#[test]
fn test_bqvec_index_and_read() -> tantivy::Result<()> {
    let (schema, text_field) = make_schema();

    let dims = 64; // 64 bits = 8 bytes per vector
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Stage vectors before each add_document.
    plugin.stage_vector(vec![0xFF; 8]); // all 1s
    writer.add_document(tantivy::doc!(text_field => "hello"))?;

    plugin.stage_vector(vec![0x00; 8]); // all 0s
    writer.add_document(tantivy::doc!(text_field => "world"))?;

    plugin.stage_vector(vec![0xAA; 8]); // alternating 10101010
    writer.add_document(tantivy::doc!(text_field => "foo"))?;

    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .expect("bqvec reader should exist");

    assert_eq!(bq.dimensions(), 64);
    assert_eq!(bq.num_vectors(), 3);
    assert_eq!(bq.bytes_per_vector(), 8);

    // O(1) access
    assert_eq!(bq.vector(0), &[0xFF; 8]);
    assert_eq!(bq.vector(1), &[0x00; 8]);
    assert_eq!(bq.vector(2), &[0xAA; 8]);

    Ok(())
}

#[test]
fn test_bqvec_zero_fill_when_no_vector_staged() -> tantivy::Result<()> {
    let (schema, text_field) = make_schema();

    let dims = 16; // 2 bytes per vector
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Stage vectors for docs 0 and 2, explicit zeros for doc 1.
    // Because IndexWriter::add_document is async (sends to indexing thread),
    // all staging must happen in document order before the corresponding
    // add_document calls — the FIFO queue maps 1:1 with doc_ids.
    plugin.stage_vector(vec![0xAB, 0xCD]);
    plugin.stage_vector(vec![0x00, 0x00]); // explicit zero for doc 1
    plugin.stage_vector(vec![0x12, 0x34]);

    writer.add_document(tantivy::doc!(text_field => "a"))?;
    writer.add_document(tantivy::doc!(text_field => "b"))?;
    writer.add_document(tantivy::doc!(text_field => "c"))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let bq: Arc<BqVecPluginReader> = seg
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();

    assert_eq!(bq.vector(0), &[0xAB, 0xCD]);
    assert_eq!(bq.vector(1), &[0x00, 0x00]);
    assert_eq!(bq.vector(2), &[0x12, 0x34]);

    Ok(())

}

#[test]
fn test_bqvec_auto_zero_fill() -> tantivy::Result<()> {
    // When no vectors are staged at all, all docs get zero vectors.
    let (schema, text_field) = make_schema();

    let dims = 16;
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    writer.add_document(tantivy::doc!(text_field => "a"))?;
    writer.add_document(tantivy::doc!(text_field => "b"))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let bq: Arc<BqVecPluginReader> = seg
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();

    assert_eq!(bq.num_vectors(), 2);
    assert_eq!(bq.vector(0), &[0x00, 0x00]);
    assert_eq!(bq.vector(1), &[0x00, 0x00]);

    Ok(())
}

#[test]
fn test_bqvec_merge() -> tantivy::Result<()> {
    let (schema, text_field) = make_schema();

    let dims = 32; // 4 bytes per vector
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Segment 1: two docs.
    plugin.stage_vector(vec![0x11; 4]);
    writer.add_document(tantivy::doc!(text_field => "a"))?;
    plugin.stage_vector(vec![0x22; 4]);
    writer.add_document(tantivy::doc!(text_field => "b"))?;
    writer.commit()?;

    // Segment 2: one doc.
    plugin.stage_vector(vec![0x33; 4]);
    writer.add_document(tantivy::doc!(text_field => "c"))?;
    writer.commit()?;

    // Force merge.
    let segment_ids = index.searchable_segment_ids()?;
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 3);

    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);

    let bq: Arc<BqVecPluginReader> = segments[0]
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();

    assert_eq!(bq.num_vectors(), 3);
    // After a stacked merge the order is seg0 docs then seg1 docs.
    assert_eq!(bq.vector(0), &[0x11; 4]);
    assert_eq!(bq.vector(1), &[0x22; 4]);
    assert_eq!(bq.vector(2), &[0x33; 4]);

    Ok(())
}

#[test]
fn test_bqvec_merge_with_deletes() -> tantivy::Result<()> {
    let (schema, text_field) = make_schema();

    let dims = 24; // 3 bytes per vector
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Segment 1: two docs.
    plugin.stage_vector(vec![0xAA; 3]);
    writer.add_document(tantivy::doc!(text_field => "keep_a"))?;
    plugin.stage_vector(vec![0xBB; 3]);
    writer.add_document(tantivy::doc!(text_field => "deleteme"))?;
    writer.commit()?;

    // Segment 2: one doc.
    plugin.stage_vector(vec![0xCC; 3]);
    writer.add_document(tantivy::doc!(text_field => "keep_b"))?;
    writer.commit()?;

    // Delete the middle document.
    writer.delete_term(tantivy::Term::from_field_text(text_field, "deleteme"));
    writer.commit()?;

    // Force merge of both segments — the deleted doc should be excluded.
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

    assert_eq!(bq.num_vectors(), 2);
    // keep_a (0xAA) and keep_b (0xCC) survive; deleteme is gone.
    assert_eq!(bq.vector(0), &[0xAA; 3]);
    assert_eq!(bq.vector(1), &[0xCC; 3]);

    Ok(())
}

#[test]
fn test_bqvec_large_dimensions() -> tantivy::Result<()> {
    let (schema, text_field) = make_schema();

    let dims = 768; // 96 bytes per vector (typical for binary quantized embeddings)
    let plugin = Arc::new(BqVecPlugin::new(dims));
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    // Create a recognizable pattern.
    let vec0: Vec<u8> = (0..96).map(|i| i as u8).collect();
    let vec1: Vec<u8> = (0..96).map(|i| (255 - i) as u8).collect();

    plugin.stage_vector(vec0.clone());
    writer.add_document(tantivy::doc!(text_field => "doc0"))?;
    plugin.stage_vector(vec1.clone());
    writer.add_document(tantivy::doc!(text_field => "doc1"))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let bq: Arc<BqVecPluginReader> = seg
        .plugin_reader::<BqVecPluginReader>("bqvec")?
        .unwrap();

    assert_eq!(bq.dimensions(), 768);
    assert_eq!(bq.bytes_per_vector(), 96);
    assert_eq!(bq.vector(0), vec0.as_slice());
    assert_eq!(bq.vector(1), vec1.as_slice());

    Ok(())
}

#[test]
fn test_bqvec_plugin_metadata() {
    let plugin = BqVecPlugin::new(128);
    assert_eq!(plugin.name(), "bqvec");
    assert_eq!(plugin.extensions(), vec!["bqvec"]);
    assert_eq!(plugin.write_phase(), 2);
    assert_eq!(plugin.dimensions(), 128);
    assert_eq!(plugin.bytes_per_vector(), 16);
}

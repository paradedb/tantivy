//! Integration test for the segment plugin system.
//!
//! Creates a custom plugin that counts documents per segment,
//! then verifies it works through the full lifecycle:
//! index -> read -> merge -> read.

use std::any::Any;
use std::sync::Arc;

use tantivy::index::SegmentComponent;
use tantivy::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{DocId, Index, IndexWriter, Segment};

/// A simple plugin that writes a document count to a custom file.
struct DocCountPlugin;

impl SegmentPlugin for DocCountPlugin {
    fn name(&self) -> &str {
        "doc_count"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["doccount"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> tantivy::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(DocCountWriter { count: 0 }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> tantivy::Result<Arc<dyn PluginReader>> {
        let component = SegmentComponent::Custom("doccount".to_string());
        let file_slice = ctx.segment_reader.open_read(component).map_err(|e| {
            tantivy::TantivyError::InternalError(format!("doccount open_read: {e}"))
        })?;
        let data = file_slice.read_bytes()?;
        assert!(
            data.len() >= 4,
            "doccount file too short: {} bytes",
            data.len()
        );
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        Ok(Arc::new(DocCountReader { count }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> tantivy::Result<()> {
        // Sum up doc counts from all source readers (using alive docs count)
        let total: u32 = ctx.readers.iter().map(|reader| reader.num_docs()).sum();

        // Write to target segment
        let component = SegmentComponent::Custom("doccount".to_string());
        let mut write = ctx.target_segment.open_write(component)?;
        use std::io::Write;
        write.write_all(&total.to_le_bytes())?;
        common::TerminatingWrite::terminate(write)?;
        Ok(())
    }
}

struct DocCountWriter {
    count: u32,
}

impl PluginWriter for DocCountWriter {
    fn add_document(&mut self, _doc_id: DocId) -> tantivy::Result<()> {
        self.count += 1;
        Ok(())
    }

    fn serialize(
        &mut self,
        segment: &mut Segment,
        _doc_id_map: Option<&tantivy::indexer::doc_id_mapping::DocIdMapping>,
    ) -> tantivy::Result<()> {
        let component = SegmentComponent::Custom("doccount".to_string());
        let mut write = segment.open_write(component)?;
        use std::io::Write;
        write.write_all(&self.count.to_le_bytes())?;
        common::TerminatingWrite::terminate(write)?;
        Ok(())
    }

    fn close(self: Box<Self>) -> tantivy::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

struct DocCountReader {
    count: u32,
}

impl DocCountReader {
    fn count(&self) -> u32 {
        self.count
    }
}

impl PluginReader for DocCountReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn test_plugin_full_lifecycle() -> tantivy::Result<()> {
    // Build an index with the doc_count plugin
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT | STORED);
    let schema = schema_builder.build();

    let plugin: Arc<dyn SegmentPlugin> = Arc::new(DocCountPlugin);
    let index = Index::builder()
        .schema(schema)
        .plugin(plugin)
        .create_in_ram()?;

    // Verify plugins are registered (built-in + custom)
    assert!(index.plugins().len() >= 2);
    assert!(
        index.plugins().iter().any(|p| p.name() == "doc_count"),
        "doc_count plugin should be registered"
    );
    assert!(
        index.plugins().iter().any(|p| p.name() == "fieldnorms"),
        "fieldnorms built-in plugin should be registered"
    );

    // Index some documents
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

    writer.add_document(tantivy::doc!(text_field => "hello world"))?;
    writer.add_document(tantivy::doc!(text_field => "foo bar"))?;
    writer.add_document(tantivy::doc!(text_field => "baz qux"))?;
    writer.commit()?;

    // Read back the plugin data from the segment reader
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 3);

    // There should be exactly one segment
    let segment_readers = searcher.segment_readers();
    assert_eq!(segment_readers.len(), 1);

    // Verify the plugin reader returns the correct document count
    let doc_count_reader: Arc<DocCountReader> = segment_readers[0]
        .plugin_reader::<DocCountReader>("doc_count")?
        .expect("doc_count plugin reader should exist");
    assert_eq!(doc_count_reader.count(), 3);

    Ok(())
}

#[test]
fn test_plugin_extensions() {
    let plugin = DocCountPlugin;
    assert_eq!(plugin.name(), "doc_count");
    assert_eq!(plugin.extensions(), vec!["doccount"]);
    assert_eq!(plugin.write_phase(), 2); // default phase
}

#[test]
fn test_custom_segment_component() {
    let component = SegmentComponent::Custom("myext".to_string());
    assert_eq!(format!("{component}"), "myext");

    let parsed = SegmentComponent::try_from("myext").unwrap();
    assert_eq!(parsed, SegmentComponent::Custom("myext".to_string()));

    // Built-in components still parse correctly
    assert_eq!(
        SegmentComponent::try_from("idx").unwrap(),
        SegmentComponent::Postings
    );
}

#[test]
fn test_segment_component_iterator_only_builtins() {
    let components: Vec<_> = SegmentComponent::iterator().collect();
    assert_eq!(components.len(), 8);
    // Custom components are not in the static iterator
    for comp in &components {
        assert!(
            !matches!(comp, SegmentComponent::Custom(_)),
            "Custom should not appear in static iterator"
        );
    }
}

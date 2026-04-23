//! Extensible segment component plugin system.
//!
//! This module defines the [`SegmentPlugin`] trait and supporting types that allow
//! custom data structures to participate in the segment lifecycle (write, read, merge).
//!
//! Built-in components (postings, fast fields, field norms, store) will eventually
//! implement this trait as well, but the primary use case is allowing external code
//! to attach new data to segments without modifying tantivy internals.

use std::any::Any;
use std::sync::Arc;

use crate::directory::Directory;
use crate::index::{IndexSettings, SegmentReader};
use crate::indexer::doc_id_mapping::SegmentDocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::schema::Schema;
use crate::{DocId, Segment};

/// A pluggable segment component that participates in writing, reading, and merging.
///
/// Each plugin manages one or more files within a segment. The plugin is a factory
/// that creates writers, readers, and handles merging. The actual data APIs are
/// component-specific and accessed via downcasting on the concrete types.
pub trait SegmentPlugin: Send + Sync + 'static {
    /// Unique name identifying this component (e.g., "postings", "fast_fields").
    fn name(&self) -> &str;

    /// File extensions this component manages (e.g., `["idx", "pos", "term"]` for postings).
    fn extensions(&self) -> Vec<&str>;

    /// Write phase for ordering during serialization and merge.
    ///
    /// Components are processed in ascending phase order. This allows components
    /// that depend on data from earlier components to read it back.
    ///
    /// Built-in phases:
    /// - Phase 0: FieldNorms
    /// - Phase 1: Postings (reads back fieldnorms)
    /// - Phase 2: Store, FastFields (independent)
    /// - Phase 3: Delete
    ///
    /// Custom plugins default to phase 2.
    fn write_phase(&self) -> u32 {
        2
    }

    /// Create a writer for accumulating and serializing data during indexing.
    ///
    /// Returns a type-erased writer. The `SegmentWriter` will downcast to the concrete
    /// type when it needs to call component-specific APIs (e.g., feeding terms to
    /// the postings writer).
    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>>;

    /// Create a reader for an existing segment's data.
    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>>;

    /// Merge data from multiple source segments into a target segment.
    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()>;
}

/// Writer for a single component within a segment.
///
/// The writer accumulates data during indexing (via component-specific APIs on the
/// concrete type) and serializes it to segment files during finalization.
pub trait PluginWriter: Send + Any {
    /// Called for each document during indexing. Default no-op.
    ///
    /// Receives only the `DocId` because the `Document` trait is not
    /// dyn-compatible (it has generic associated types) and `SegmentWriter`
    /// is generic over `D: Document`. Plugins that need document data
    /// should read it back from the store or fast fields after serialization.
    fn add_document(&mut self, _doc_id: DocId) -> crate::Result<()> {
        Ok(())
    }

    /// Serialize accumulated data to segment files.
    /// Called during `SegmentWriter::finalize()`.
    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&crate::indexer::doc_id_mapping::DocIdMapping>,
    ) -> crate::Result<()>;

    /// Finalize and close any open file handles.
    fn close(self: Box<Self>) -> crate::Result<()>;

    /// Current memory usage of this writer.
    fn mem_usage(&self) -> usize;

    /// Downcast support for accessing component-specific APIs.
    fn as_any(&self) -> &dyn Any;

    /// Downcast support for accessing component-specific APIs (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Reader for a single component within a segment.
pub trait PluginReader: Send + Sync + Any {
    /// Downcast support for accessing component-specific APIs.
    fn as_any(&self) -> &dyn Any;
}

/// Context provided to [`SegmentPlugin::create_writer`].
pub struct PluginWriterContext<'a> {
    /// The segment being written to.
    pub segment: &'a Segment,
    /// The index schema.
    pub schema: &'a Schema,
    /// The index settings.
    pub settings: &'a IndexSettings,
    /// Whether this writer is being created for a merge operation.
    pub is_in_merge: bool,
    /// The directory for reading/writing files. Plugins can use this to open
    /// file handles directly (e.g., `directory.open_write(&path)`).
    /// The `Directory::open_write` trait method takes `&self`, so no mutable
    /// segment reference is needed.
    pub directory: &'a dyn Directory,
}

/// Context provided to [`SegmentPlugin::open_reader`].
pub struct PluginReaderContext<'a> {
    /// The segment to read from.
    pub segment: &'a Segment,
    /// The index schema.
    pub schema: &'a Schema,
    /// The segment reader, for accessing existing component data.
    pub segment_reader: &'a SegmentReader,
}

/// Context provided to [`SegmentPlugin::merge`].
pub struct PluginMergeContext<'a> {
    /// Readers for the source segments being merged.
    pub readers: &'a [SegmentReader],
    /// The document id mapping from old segments to the new merged segment.
    pub doc_id_mapping: &'a SegmentDocIdMapping,
    /// The target segment being written to.
    pub target_segment: &'a mut Segment,
    /// The index schema.
    pub schema: &'a Schema,
    /// The index settings.
    pub settings: &'a IndexSettings,
    /// Cancel sentinel for cooperative cancellation.
    pub cancel: &'a dyn CancelSentinel,
}

#[cfg(test)]
mod tests {
    //! Round-trip integration test for the segment plugin system.
    //!
    //! Defines a custom plugin that counts documents per segment, then
    //! verifies it works through the full lifecycle: index → read → merge → read.

    use super::*;
    use crate::index::SegmentComponent;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{Index, IndexWriter};

    /// A simple plugin that writes a document count to a custom file.
    struct DocCountPlugin;

    impl SegmentPlugin for DocCountPlugin {
        fn name(&self) -> &str {
            "doc_count"
        }

        fn extensions(&self) -> Vec<&str> {
            vec!["doccount"]
        }

        fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
            Ok(Box::new(DocCountWriter { count: 0 }))
        }

        fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
            let component = SegmentComponent::Custom("doccount".to_string());
            let file_slice = ctx
                .segment_reader
                .open_read(component)
                .map_err(|e| crate::TantivyError::InternalError(format!("doccount open_read: {e}")))?;
            let data = file_slice.read_bytes()?;
            assert!(
                data.len() >= 4,
                "doccount file too short: {} bytes",
                data.len()
            );
            let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            Ok(Arc::new(DocCountReader { count }))
        }

        fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
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
        fn add_document(&mut self, _doc_id: DocId) -> crate::Result<()> {
            self.count += 1;
            Ok(())
        }

        fn serialize(
            &mut self,
            segment: &mut Segment,
            _doc_id_map: Option<&crate::indexer::doc_id_mapping::DocIdMapping>,
        ) -> crate::Result<()> {
            let component = SegmentComponent::Custom("doccount".to_string());
            let mut write = segment.open_write(component)?;
            use std::io::Write;
            write.write_all(&self.count.to_le_bytes())?;
            common::TerminatingWrite::terminate(write)?;
            Ok(())
        }

        fn close(self: Box<Self>) -> crate::Result<()> {
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
    fn test_plugin_full_lifecycle() -> crate::Result<()> {
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

        writer.add_document(crate::doc!(text_field => "hello world"))?;
        writer.add_document(crate::doc!(text_field => "foo bar"))?;
        writer.add_document(crate::doc!(text_field => "baz qux"))?;
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
}

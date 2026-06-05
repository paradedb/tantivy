//! Extensible segment component plugin system.
//!
//! This module defines the [`SegmentPlugin`] trait and supporting types that allow
//! custom data structures to participate in the segment lifecycle (write, read, merge).
//!
//! Built-in components (postings, fast fields, field norms, store) will eventually
//! implement this trait as well, but the primary use case is allowing external code
//! to attach new data to segments without modifying tantivy internals.

use std::any::Any;
use std::collections::BTreeMap;

use common::HasLen;

use crate::directory::Directory;
use crate::index::{IndexSettings, SegmentComponent, SegmentReader};
use crate::indexer::doc_id_mapping::SegmentDocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::schema::Schema;
use crate::space_usage::ComponentSpaceUsage;
use crate::Segment;

/// A pluggable segment component that participates in writing and merging.
///
/// Each plugin manages one or more files within a segment. The plugin is a factory
/// that creates writers and handles merging. The actual data APIs are
/// component-specific and accessed via downcasting on the concrete types.
pub trait SegmentPlugin: Send + Sync + 'static {
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

    /// Merge data from multiple source segments into a target segment.
    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()>;

    /// Report on-disk space usage of this component, keyed by component name.
    ///
    /// The returned entries are merged into [`SegmentSpaceUsage`]. The default
    /// implementation emits one [`ComponentSpaceUsage::Basic`] entry per file in
    /// [`extensions()`](Self::extensions); built-in plugins override this to report
    /// richer per-field breakdowns under the keys the named accessors expect.
    ///
    /// [`SegmentSpaceUsage`]: crate::space_usage::SegmentSpaceUsage
    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let mut usage = BTreeMap::new();
        for ext in self.extensions() {
            let file = segment_reader.open_read(SegmentComponent::Custom(ext.to_string()))?;
            usage.insert(
                ext.to_string(),
                ComponentSpaceUsage::Basic(file.len().into()),
            );
        }
        Ok(usage)
    }
}

/// Writer for a single component within a segment.
///
/// The writer accumulates data during indexing (via component-specific APIs on the
/// concrete type) and serializes it to segment files during finalization.
pub trait PluginWriter: Send + Any {
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
    /// Whether the document store should be ignored for this segment.
    pub ignore_store: bool,
    /// The directory for reading/writing files. Plugins can use this to open
    /// file handles directly (e.g., `directory.open_write(&path)`).
    /// The `Directory::open_write` trait method takes `&self`, so no mutable
    /// segment reference is needed.
    pub directory: &'a dyn Directory,
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
    /// Whether the document store should be ignored for this segment.
    pub ignore_store: bool,
    /// Cancel sentinel for cooperative cancellation.
    pub cancel: &'a dyn CancelSentinel,
}

#[cfg(test)]
mod tests {
    //! Round-trip integration test for the segment plugin system.
    //!
    //! Defines a custom marker plugin, then verifies it works through the
    //! full lifecycle: write → read → merge → read.

    use std::sync::Arc;

    use super::*;
    use crate::index::SegmentComponent;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{Index, IndexWriter};

    const MARKER: u32 = 0xDEADBEEF;

    /// A simple plugin that writes a fixed marker to a custom file.
    struct MarkerPlugin;

    impl SegmentPlugin for MarkerPlugin {
        fn extensions(&self) -> Vec<&str> {
            vec!["marker"]
        }

        fn create_writer(
            &self,
            _ctx: &PluginWriterContext,
        ) -> crate::Result<Box<dyn PluginWriter>> {
            Ok(Box::new(MarkerWriter))
        }

        fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
            let component = SegmentComponent::Custom("marker".to_string());
            let mut write = ctx.target_segment.open_write(component)?;
            use std::io::Write;
            write.write_all(&MARKER.to_le_bytes())?;
            common::TerminatingWrite::terminate(write)?;
            Ok(())
        }
    }

    struct MarkerWriter;

    impl PluginWriter for MarkerWriter {
        fn serialize(
            &mut self,
            segment: &mut Segment,
            _doc_id_map: Option<&crate::indexer::doc_id_mapping::DocIdMapping>,
        ) -> crate::Result<()> {
            let component = SegmentComponent::Custom("marker".to_string());
            let mut write = segment.open_write(component)?;
            use std::io::Write;
            write.write_all(&MARKER.to_le_bytes())?;
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

    #[test]
    fn test_plugin_full_lifecycle() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        let plugin: Arc<dyn SegmentPlugin> = Arc::new(MarkerPlugin);
        let index = Index::builder()
            .schema(schema)
            .register_plugin(plugin)
            .create_in_ram()?;

        assert!(index.plugins().len() >= 2);
        assert!(
            index
                .plugins()
                .iter()
                .any(|p| p.extensions().contains(&"marker")),
            "marker plugin should be registered"
        );
        assert!(
            index
                .plugins()
                .iter()
                .any(|p| p.extensions().contains(&"fieldnorm")),
            "fieldnorms built-in plugin should be registered"
        );

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.add_document(crate::doc!(text_field => "hello world"))?;
        writer.add_document(crate::doc!(text_field => "foo bar"))?;
        writer.add_document(crate::doc!(text_field => "baz qux"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 3);

        let segment_readers = searcher.segment_readers();
        assert_eq!(segment_readers.len(), 1);

        let data = segment_readers[0]
            .open_read(SegmentComponent::Custom("marker".to_string()))?
            .read_bytes()?;
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(value, MARKER);

        Ok(())
    }

    #[test]
    fn test_plugin_extensions() {
        let plugin = MarkerPlugin;
        assert_eq!(plugin.extensions(), vec!["marker"]);
        assert_eq!(plugin.write_phase(), 2);
    }

    #[test]
    fn test_reopen_without_plugin_fails_closed() -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::TantivyError;

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        // Build an index with the custom plugin and persist a segment.
        let dir = RamDirectory::create();
        let plugin: Arc<dyn SegmentPlugin> = Arc::new(MarkerPlugin);
        let index = Index::builder()
            .schema(schema)
            .register_plugin(plugin)
            .create(dir.clone())?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.add_document(crate::doc!(text_field => "hello world"))?;
            writer.commit()?;
        }

        // The committed segment records that it requires the "marker" extension.
        let segment_metas = index.searchable_segment_metas()?;
        assert_eq!(segment_metas.len(), 1);
        assert_eq!(
            segment_metas[0].plugin_extensions(),
            &["marker".to_string()]
        );

        // Reopen without re-registering the plugin: writing must fail closed
        // rather than silently dropping the plugin's data.
        let reopened = Index::open(dir.clone())?;
        let err = reopened
            .writer_with_num_threads::<crate::TantivyDocument>(1, 15_000_000)
            .err()
            .expect("writer creation should fail when the plugin is not registered");
        assert!(
            matches!(err, TantivyError::MissingPlugin(ref exts) if exts.contains("marker")),
            "expected MissingPlugin error, got {err:?}"
        );

        // Re-registering the plugin clears the guard.
        let mut reopened = reopened;
        reopened.register_plugin(Arc::new(MarkerPlugin));
        let _writer: IndexWriter = reopened.writer_with_num_threads(1, 15_000_000)?;

        Ok(())
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

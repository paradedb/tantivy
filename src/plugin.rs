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

use crate::index::{IndexSettings, SegmentReader};
use crate::indexer::doc_id_mapping::SegmentDocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::schema::Schema;
use crate::{DocId, Segment, TantivyDocument};

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
    /// Custom plugins use this to extract data from documents.
    /// Built-in components (postings, fast fields) may ignore this and use
    /// their own specialized paths.
    fn add_document(
        &mut self,
        _doc_id: DocId,
        _doc: &TantivyDocument,
        _schema: &Schema,
    ) -> crate::Result<()> {
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
}

/// Context provided to [`SegmentPlugin::open_reader`].
pub struct PluginReaderContext<'a> {
    /// The segment to read from.
    pub segment: &'a Segment,
    /// The index schema.
    pub schema: &'a Schema,
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

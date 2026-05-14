//! Stub writer for the flat vector plugin.
//!
//! Until the on-disk format lands, this writer is unregistered and
//! every method is a `todo!()`. The type surface is stable so that
//! `SegmentWriter` integration can be added in a single commit alongside
//! the real implementation.

use std::any::Any;

use crate::index::Segment;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::PluginWriter;
use crate::schema::document::Document;
use crate::schema::Schema;
use crate::DocId;

pub struct FlatVecPluginWriter {
    // TODO: per-field byte buffers + present-doc-id lists, captured
    // num_docs for sizing the presence bitmap.
}

impl FlatVecPluginWriter {
    pub(crate) fn stub() -> Self {
        Self {}
    }

    /// Append the vector-typed fields of a document.
    ///
    /// Called from `SegmentWriter::add_document` (parallel to how
    /// `FastFieldsWriter::add_document` is fed). The real implementation
    /// extracts raw bytes from each vector-typed field and appends them
    /// to a per-field dense byte blob, marking the doc as present.
    pub fn add_document<D: Document>(
        &mut self,
        _doc_id: DocId,
        _doc: &D,
        _schema: &Schema,
    ) -> crate::Result<()> {
        todo!("flat vector writer add_document")
    }

    /// Sets the total doc count used to size the presence bitmap. Called
    /// from `SegmentWriter::finalize` before [`PluginWriter::serialize`].
    pub fn set_num_docs(&mut self, _num_docs: DocId) {
        todo!("flat vector writer set_num_docs")
    }
}

impl PluginWriter for FlatVecPluginWriter {
    fn serialize(
        &mut self,
        _segment: &mut Segment,
        _doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        todo!("flat vector writer serialize")
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

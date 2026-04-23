//! FieldNorms as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `FieldNormsWriter`, `FieldNormsSerializer`, and
//! `FieldNormReaders` types behind the plugin interface so that fieldnorms
//! participate in the unified plugin lifecycle.

use std::any::Any;
use std::sync::Arc;

use crate::directory::Directory;
use crate::fieldnorm::{FieldNormReader, FieldNormReaders, FieldNormsSerializer, FieldNormsWriter};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::{DocId, Segment};

/// Built-in plugin for field norms.
///
/// Field norms track the number of tokens per field per document, used for
/// BM25 scoring. This is the simplest built-in component and serves as the
/// reference implementation for the plugin pattern.
pub struct FieldNormsPlugin;

impl SegmentPlugin for FieldNormsPlugin {
    fn name(&self) -> &str {
        "fieldnorms"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["fieldnorm"]
    }

    fn write_phase(&self) -> u32 {
        0 // Must be written first; postings reads fieldnorms back.
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let writer = FieldNormsWriter::for_schema(ctx.schema);
        // During merge, the merge() method handles file creation directly.
        // Only open the file during normal indexing.
        let serializer = if !ctx.is_in_merge {
            let path = ctx.segment.relative_path(SegmentComponent::FieldNorms);
            let write = ctx.directory.open_write(&path)?;
            Some(FieldNormsSerializer::from_write(write)?)
        } else {
            None
        };
        Ok(Box::new(FieldNormsPluginWriter { writer, serializer }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file = ctx.segment_reader.open_read(SegmentComponent::FieldNorms)?;
        let readers = FieldNormReaders::open(file)?;
        Ok(Arc::new(FieldNormsPluginReader(readers)))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let path = ctx
            .target_segment
            .relative_path(SegmentComponent::FieldNorms);
        let write = ctx.target_segment.index().directory().open_write(&path)?;
        let mut serializer = FieldNormsSerializer::from_write(write)?;

        let schema = ctx.schema;
        let fields = FieldNormsWriter::fields_with_fieldnorm(schema);
        let max_doc: usize = ctx.readers.iter().map(|r| r.num_docs() as usize).sum();
        let mut fieldnorms_data = Vec::with_capacity(max_doc);

        for field in fields {
            if ctx.cancel.wants_cancel() {
                return Err(crate::TantivyError::Cancelled);
            }
            fieldnorms_data.clear();
            let fieldnorms_readers: Vec<FieldNormReader> = ctx
                .readers
                .iter()
                .map(|reader| reader.get_fieldnorms_reader(field))
                .collect::<Result<_, _>>()?;
            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let reader = &fieldnorms_readers[old_doc_addr.segment_ord as usize];
                let fieldnorm_id = reader.fieldnorm_id(old_doc_addr.doc_id);
                fieldnorms_data.push(fieldnorm_id);
            }
            serializer.serialize_field(field, &fieldnorms_data)?;
        }
        serializer.close()?;
        Ok(())
    }
}

/// Plugin writer wrapping [`FieldNormsWriter`] and [`FieldNormsSerializer`].
///
/// Exposes `record()` and `fill_up_to_max_doc()` for the `SegmentWriter`
/// to call via downcast.
pub struct FieldNormsPluginWriter {
    pub writer: FieldNormsWriter,
    serializer: Option<FieldNormsSerializer>,
}

impl FieldNormsPluginWriter {
    /// Record a fieldnorm value. Called by `SegmentWriter` via downcast.
    pub fn record(&mut self, doc: DocId, field: crate::schema::Field, fieldnorm: u32) {
        self.writer.record(doc, field, fieldnorm);
    }

    /// Pad fieldnorms to max_doc. Called before serialize.
    pub fn fill_up_to_max_doc(&mut self, max_doc: DocId) {
        self.writer.fill_up_to_max_doc(max_doc);
    }
}

impl PluginWriter for FieldNormsPluginWriter {
    fn serialize(
        &mut self,
        _segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        if let Some(serializer) = self.serializer.take() {
            self.writer
                .serialize(serializer, doc_id_map)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        // If serializer wasn't consumed by serialize(), close it now.
        if let Some(serializer) = self.serializer {
            serializer
                .close()
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.writer.mem_usage()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Plugin reader wrapping [`FieldNormReaders`].
pub struct FieldNormsPluginReader(pub FieldNormReaders);

impl FieldNormsPluginReader {
    /// Access the underlying [`FieldNormReaders`].
    pub fn readers(&self) -> &FieldNormReaders {
        &self.0
    }
}

impl PluginReader for FieldNormsPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

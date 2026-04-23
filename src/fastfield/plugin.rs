//! FastFields as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `FastFieldsWriter` and `FastFieldReaders` types behind
//! the plugin interface so that fast fields participate in the unified plugin lifecycle.

use std::any::Any;
use std::sync::Arc;

use columnar::{
    ColumnType, ColumnarReader, MergeRowOrder, RowAddr, ShuffleMergeOrder, StackMergeOrder,
};
use common::TerminatingWrite;
use measure_time::debug_time;

use crate::directory::{Directory, WritePtr};
use crate::fastfield::{FastFieldReaders, FastFieldsWriter};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::{DocIdMapping, MappingType, SegmentDocIdMapping};
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::schema::{value_type_to_column_type, Schema};
use crate::Segment;

/// Built-in plugin for fast fields (columnar storage).
///
/// Fast fields provide column-oriented random access to document field values,
/// used for sorting, aggregation, filtering, and scoring.
pub struct FastFieldsPlugin;

impl SegmentPlugin for FastFieldsPlugin {
    fn name(&self) -> &str {
        "fast_fields"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["fast"]
    }

    fn write_phase(&self) -> u32 {
        2
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let tokenizer_manager = ctx.segment.index().fast_field_tokenizer().clone();
        let writer =
            FastFieldsWriter::from_schema_and_tokenizer_manager(ctx.schema, tokenizer_manager)?;

        // During merge, the merge() method handles file creation directly.
        // Only open the file during normal indexing.
        let fast_field_write = if !ctx.is_in_merge {
            let path = ctx
                .segment
                .relative_path(SegmentComponent::FastFields);
            let write = ctx.directory.open_write(&path)?;
            Some(write)
        } else {
            None
        };

        Ok(Box::new(FastFieldsPluginWriter {
            writer: Some(writer),
            fast_field_write,
        }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file = ctx
            .segment_reader
            .open_read(SegmentComponent::FastFields)?;
        let schema = ctx.schema.clone();
        let readers = FastFieldReaders::open(file, schema)
            .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        Ok(Arc::new(FastFieldsPluginReader(readers)))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        debug_time!("write-fast-fields");
        let path = ctx
            .target_segment
            .relative_path(SegmentComponent::FastFields);
        let mut fast_field_wrt: WritePtr = ctx
            .target_segment
            .index()
            .directory()
            .open_write(&path)?;

        let required_columns = extract_fast_field_required_columns(ctx.schema);
        let columnars: Vec<&ColumnarReader> = ctx
            .readers
            .iter()
            .map(|reader| reader.fast_fields().columnar())
            .collect();

        // Clone the doc_id_mapping since convert_to_merge_order consumes it by value
        let doc_id_mapping = ctx.doc_id_mapping.clone();
        let merge_row_order = convert_to_merge_order(&columnars[..], doc_id_mapping);

        let cancel = ctx.cancel;
        columnar::merge_columnar(
            &columnars[..],
            &required_columns,
            merge_row_order,
            &mut fast_field_wrt,
            || cancel.wants_cancel(),
        )?;

        fast_field_wrt.terminate()?;
        Ok(())
    }
}

/// Plugin writer wrapping [`FastFieldsWriter`].
///
/// Exposes `add_document()` and `sort_order()` for the `SegmentWriter`
/// to call via downcast.
pub struct FastFieldsPluginWriter {
    /// The inner writer is wrapped in an `Option` because `FastFieldsWriter::serialize`
    /// takes `self` by value. We `.take()` it during serialize.
    pub writer: Option<FastFieldsWriter>,
    fast_field_write: Option<WritePtr>,
}

impl FastFieldsPluginWriter {
    /// Access the inner writer for `add_document` calls.
    /// Panics if called after serialization.
    pub fn writer_mut(&mut self) -> &mut FastFieldsWriter {
        self.writer
            .as_mut()
            .expect("FastFieldsWriter already consumed by serialize")
    }

    /// Access the inner writer immutably (e.g. for `sort_order`).
    /// Panics if called after serialization.
    pub fn writer(&self) -> &FastFieldsWriter {
        self.writer
            .as_ref()
            .expect("FastFieldsWriter already consumed by serialize")
    }
}

impl PluginWriter for FastFieldsPluginWriter {
    fn serialize(
        &mut self,
        _segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        if let (Some(writer), Some(wrt)) = (self.writer.take(), self.fast_field_write.as_mut()) {
            writer
                .serialize(wrt, doc_id_map)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        if let Some(wrt) = self.fast_field_write {
            wrt.terminate()?;
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.writer.as_ref().map_or(0, |w| w.mem_usage())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Plugin reader wrapping [`FastFieldReaders`].
pub struct FastFieldsPluginReader(pub FastFieldReaders);

impl FastFieldsPluginReader {
    /// Access the underlying [`FastFieldReaders`].
    pub fn readers(&self) -> &FastFieldReaders {
        &self.0
    }
}

impl PluginReader for FastFieldsPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// --- Helper functions moved from merger.rs ---

fn convert_to_merge_order(
    columnars: &[&ColumnarReader],
    doc_id_mapping: SegmentDocIdMapping,
) -> MergeRowOrder {
    match doc_id_mapping.mapping_type() {
        MappingType::Stacked => MergeRowOrder::Stack(StackMergeOrder::stack(columnars)),
        MappingType::StackedWithDeletes | MappingType::Shuffled => {
            let new_row_id_to_old_row_id: Vec<RowAddr> = doc_id_mapping
                .new_doc_id_to_old_doc_addr
                .into_iter()
                .map(|doc_addr| RowAddr {
                    segment_ord: doc_addr.segment_ord,
                    row_id: doc_addr.doc_id,
                })
                .collect();
            MergeRowOrder::Shuffled(ShuffleMergeOrder {
                new_row_id_to_old_row_id,
                alive_bitsets: doc_id_mapping.alive_bitsets,
            })
        }
    }
}

fn extract_fast_field_required_columns(schema: &Schema) -> Vec<(String, ColumnType)> {
    schema
        .fields()
        .map(|(_, field_entry)| field_entry)
        .filter(|field_entry| field_entry.is_fast())
        .filter_map(|field_entry| {
            let column_name = field_entry.name().to_string();
            let column_type = value_type_to_column_type(field_entry.field_type().value_type())?;
            Some((column_name, column_type))
        })
        .collect()
}

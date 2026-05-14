use std::io::Write;
use std::sync::Arc;

use super::presence::Presence;
use super::reader::FlatVecReader;
use super::writer::FlatVecPluginWriter;
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::schema::FieldType;
use crate::DocId;

pub struct FlatVecPlugin;

impl SegmentPlugin for FlatVecPlugin {
    fn name(&self) -> &str {
        "flat_vec"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["flatvec"]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(FlatVecPluginWriter::for_schema(ctx.schema)))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        Ok(Arc::new(FlatVecReader::open(ctx)?))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let has_vector_field = ctx
            .schema
            .fields()
            .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
        if !has_vector_field {
            return Ok(());
        }
        // Symmetric short-circuit with `IvfVecPlugin::merge`: exactly
        // one of the two writes a vector file per merge. At/above the
        // clustering threshold IVF takes over and flat writes nothing.
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        if (target_docs as usize) >= ctx.settings.vector_clustering_threshold() {
            return Ok(());
        }
        if ctx.cancel.wants_cancel() {
            return Err(crate::TantivyError::Cancelled);
        }
        let path = ctx
            .target_segment
            .relative_path(SegmentComponent::Custom("flatvec".to_string()));
        let write = ctx.target_segment.index().directory().open_write(&path)?;
        let source_readers: Vec<Option<Arc<FlatVecReader>>> = ctx
            .readers
            .iter()
            .map(|reader| reader.plugin_reader::<FlatVecReader>("flat_vec"))
            .collect::<crate::Result<Vec<_>>>()?;

        // Per-source-segment cached vector columns for each vector field.
        // We open them lazily inside the field loop.

        let mut composite = CompositeWrite::wrap(write);

        // num_docs in the target segment = number of alive docs aggregated
        // by the doc_id_mapping. Each call to iter_old_doc_addrs yields
        // exactly num_new_doc_ids items.
        let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum::<u32>();

        for (field, entry) in ctx.schema.fields() {
            let _opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };

            // Per-segment column views for this field (lazy open).
            let columns: Vec<_> = source_readers
                .iter()
                .map(|reader_opt| {
                    reader_opt
                        .as_ref()
                        .and_then(|reader| reader.open_column(field))
                })
                .collect();

            // Walk the target doc-id space, copying bytes for docs whose
            // source had a value and recording their new doc-id in the
            // target's present list.
            let mut target_present: Vec<DocId> = Vec::new();
            let mut target_rows: Vec<u8> = Vec::new();
            let mut new_doc_id: DocId = 0;
            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                if let Some(column) = &columns[old_doc_addr.segment_ord as usize] {
                    if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                        target_present.push(new_doc_id);
                        target_rows.extend_from_slice(bytes);
                    }
                }
                new_doc_id += 1;
            }

            // Sanity: the mapping iterator should yield exactly num_target_docs items.
            debug_assert_eq!(new_doc_id, num_target_docs);

            // Slice (field, 0): presence section (Full or Optional).
            let bitmap_w = composite.for_field_with_idx(field, 0);
            Presence::serialize(&target_present, num_target_docs, bitmap_w)?;
            bitmap_w.flush()?;

            // Slice (field, 1): dense f32 LE rows for present docs only.
            let rows_w = composite.for_field_with_idx(field, 1);
            rows_w.write_all(&target_rows)?;
            rows_w.flush()?;
        }
        composite.close()?;
        Ok(())
    }
}

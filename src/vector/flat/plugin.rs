//! Flat vector segment merging.

use std::io::Write;

use super::id_map::IdMap;
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::FieldType;
use crate::vector::header::write_vector_header;
use crate::vector::VEC_EXT;
use crate::DocId;

/// Merges source vectors into a flat target segment.
pub(crate) fn merge_flat(ctx: &PluginMergeContext) -> crate::Result<()> {
    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }
    if ctx.cancel.wants_cancel() {
        return Err(crate::TantivyError::Cancelled);
    }
    let path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let mut write = ctx.target_segment.index().directory().open_write(&path)?;
    write_vector_header(&mut write)?;
    let mut composite = CompositeWrite::wrap(write);

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum::<u32>();

    for (field, entry) in ctx.schema.fields() {
        let _opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };

        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;

        let mut target_present: Vec<DocId> = Vec::new();
        let mut target_doc_id: DocId = 0;
        {
            let rows_w = composite.for_field_with_idx(field, 1);
            for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                let reader = &field_readers[source_doc_addr.segment_ord as usize];
                if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                    target_present.push(target_doc_id);
                    rows_w.write_all(&bytes)?;
                }
                target_doc_id += 1;
            }
            rows_w.flush()?;
        }

        debug_assert_eq!(target_doc_id, num_target_docs);

        let id_map_w = composite.for_field_with_idx(field, 0);
        IdMap::serialize(&target_present, num_target_docs, id_map_w)?;
        id_map_w.flush()?;
    }
    composite.close()?;
    Ok(())
}

//! Flat vector segment merging.

use std::io::Write;

use super::id_map::IdMap;
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::FieldType;
use crate::vector::blocks::{align_up, block_align, finish_data, pad, write_metadata};
use crate::vector::header::{write_vector_header, VectorEntry, HEADER_LEN};
use crate::vector::metadata::{VectorColMetadata, FLAT_ROWS_PER_BLOCK};
use crate::vector::{MAX_ELEM_BYTES, VEC_EXT};
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

    let mut id_maps = Vec::new();
    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
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
            let meta = VectorColMetadata::build_flat(opts);
            let align = block_align(&meta.slots());
            composite.align_next_field(MAX_ELEM_BYTES, HEADER_LEN)?;
            let rows_w = composite.for_field_with_idx(field, VectorEntry::Data.index());
            let start = rows_w.written_bytes();
            write_metadata(rows_w, &meta)?;
            let mut block_bytes = 0;
            // Row groups can be streamed without knowing the final vector count.

            for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                if ctx.cancel.wants_cancel() {
                    return Err(crate::TantivyError::Cancelled);
                }
                let reader = &field_readers[source_doc_addr.segment_ord as usize];
                if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                    target_present.push(target_doc_id);
                    rows_w.write_all(&bytes)?;
                    block_bytes += bytes.len();
                    if target_present.len() % FLAT_ROWS_PER_BLOCK as usize == 0 {
                        pad(rows_w, align_up(block_bytes, align) - block_bytes)?;
                        block_bytes = 0;
                    }
                }
                target_doc_id += 1;
            }
            pad(rows_w, align_up(block_bytes, align) - block_bytes)?;
            let len = (rows_w.written_bytes() - start) as usize;
            finish_data(rows_w, len)?;
            assert_eq!(
                (rows_w.written_bytes() - start) as usize % MAX_ELEM_BYTES,
                0
            );
            rows_w.flush()?;
        }

        debug_assert_eq!(target_doc_id, num_target_docs);

        id_maps.push((field, target_present));
    }
    for (field, present) in id_maps {
        IdMap::serialize(
            &present,
            num_target_docs,
            composite.for_field_with_idx(field, VectorEntry::IdMap.index()),
        )?;
    }
    composite.close()?;
    Ok(())
}

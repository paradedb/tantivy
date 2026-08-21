//! The flat (doc-ordered) `.vec` layout: slots `[0]`/`[1]` only, no
//! clustering.
//!
//! Written by indexes WITHOUT a centroid index — the mutable/staging tier,
//! where segments are bounded and searched exhaustively (exact, so fresh
//! data is always found regardless of how well it fits any vocabulary).
//! Indexes with a set never write it; a flat segment entering such an
//! index (moved in from a staging index) is clustered at its first merge.

use std::io::Write;

use super::id_map::IdMap;
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType};
use crate::vector::header::{vec_slot, write_header};
use crate::vector::VEC_EXT;
use crate::{DocId, TantivyError};

/// Write one field's flat slots: the `Identity`/`Bitmap` id-map and the
/// doc-ordered rows. `present` ascends; `row_bytes` is parallel to it.
pub(crate) fn write_flat_field(
    vec_write: &mut CompositeWrite,
    field: Field,
    present: &[DocId],
    row_bytes: &[u8],
    num_docs: u32,
) -> crate::Result<()> {
    {
        let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
        IdMap::serialize(present, num_docs, id_map_w)?;
        id_map_w.flush()?;
    }
    {
        let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
        rows_w.write_all(row_bytes)?;
        rows_w.flush()?;
    }
    Ok(())
}

/// Merge for an index WITHOUT a centroid index: every source is flat, and
/// the target stays flat — raw rows copied in target doc order.
pub(crate) fn merge_flat(ctx: &PluginMergeContext) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }
    let path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let mut write = ctx.target_segment.index().directory().open_write(&path)?;
    write_header(&mut write)?;
    let mut composite = CompositeWrite::wrap(write);

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();

    for (field, entry) in ctx.schema.fields() {
        if !matches!(entry.field_type(), FieldType::Vector(_)) {
            continue;
        }
        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;
        for reader in &field_readers {
            if reader.index().is_some() {
                return Err(TantivyError::InternalError(
                    "clustered vector segment in an index without a centroid index".to_string(),
                ));
            }
        }

        let mut target_present: Vec<DocId> = Vec::new();
        let mut new_doc_id: DocId = 0;
        {
            let rows_w = composite.for_field_with_idx(field, vec_slot::ROWS);
            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let reader = &field_readers[old_doc_addr.segment_ord as usize];
                if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                    target_present.push(new_doc_id);
                    rows_w.write_all(&bytes)?;
                }
                new_doc_id += 1;
            }
            rows_w.flush()?;
        }
        debug_assert_eq!(new_doc_id, num_target_docs);

        let id_map_w = composite.for_field_with_idx(field, vec_slot::ID_MAP);
        IdMap::serialize(&target_present, num_target_docs, id_map_w)?;
        id_map_w.flush()?;
    }
    composite.close()?;
    Ok(())
}

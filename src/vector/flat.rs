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
use crate::directory::CompositeWrite;
use crate::schema::Field;
use crate::vector::header::vec_slot;
use crate::DocId;

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

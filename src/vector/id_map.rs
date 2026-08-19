//! Per-segment row→doc_id map for vector fields.
//!
//! Stored as slot `[0]` of the `.vec` composite file, parallel to the dense
//! row blob in slot `[1]`. Rows are cluster-sorted (there is no flat layout
//! from format V3 on), so the map is always the explicit permutation: one
//! little-endian `u32` doc id per row. The leading variant tag survives
//! from the retired flat generation — `Explicit` was tag 2 — so a stale
//! pre-V3 body can never be misparsed as a permutation.
//!
//! ## On-disk layout
//!
//! ```text
//! [u8 variant_tag = 2] [body: row→doc_id permutation, one u32 LE per row]
//! ```

use std::io::{self, Write};
use std::mem::size_of;

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::DocId;

pub(crate) const VARIANT_EXPLICIT: u8 = 2;

/// The row→doc_id permutation, held as the raw little-endian body (one
/// `u32` per row) so it can be decoded a row at a time.
#[derive(Debug)]
pub struct IdMap(OwnedBytes);

impl IdMap {
    pub fn serialize_explicit<W: Write>(row_doc_ids: &[DocId], out: &mut W) -> io::Result<()> {
        out.write_all(&[VARIANT_EXPLICIT])?;
        for doc_id in row_doc_ids {
            doc_id.serialize(out)?;
        }
        Ok(())
    }

    /// Parse a serialized id-map section.
    pub fn open(file_slice: FileSlice) -> io::Result<IdMap> {
        if file_slice.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "id map section is empty",
            ));
        }
        let tag = file_slice.slice(0..1).read_bytes()?[0];
        if tag != VARIANT_EXPLICIT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown id map variant tag: {tag}"),
            ));
        }
        let bytes = file_slice.slice_from(1).read_bytes()?;
        if bytes.len() % size_of::<DocId>() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "explicit id map body is not a whole number of u32 doc ids",
            ));
        }
        Ok(IdMap(bytes))
    }

    /// Number of stored rows.
    pub fn num_rows(&self) -> u32 {
        (self.0.len() / size_of::<DocId>()) as u32
    }

    /// The doc id stored at `row` of the cluster-sorted permutation,
    /// decoded on demand from the pinned bytes. Caller guarantees
    /// `row < num_rows`.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let start = row * size_of::<DocId>();
        DocId::from_le_bytes(self.0[start..start + size_of::<DocId>()].try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_round_trip() {
        // A cluster-sorted permutation: rows 0..2 are cluster 0 (docs 1,4),
        // rows 2..4 are cluster 1 (docs 0,3) — not globally sorted.
        let row_doc_ids: Vec<DocId> = vec![1, 4, 0, 3];
        let mut buf = Vec::new();
        IdMap::serialize_explicit(&row_doc_ids, &mut buf).unwrap();
        assert_eq!(buf[0], VARIANT_EXPLICIT);

        let p = IdMap::open(FileSlice::from(buf)).unwrap();
        assert_eq!(p.num_rows(), 4);
        for (row, &doc) in row_doc_ids.iter().enumerate() {
            assert_eq!(p.doc_id_at(row), doc);
        }
    }

    #[test]
    fn test_stale_flat_tags_rejected() {
        // Tags 0 (Identity) and 1 (Bitmap) belonged to the retired flat
        // layout; a pre-V3 body must be rejected, not misparsed.
        for tag in [0u8, 1u8, 3u8] {
            let err = IdMap::open(FileSlice::from(vec![tag])).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn test_ragged_body_rejected() {
        let mut buf = Vec::new();
        IdMap::serialize_explicit(&[1, 2], &mut buf).unwrap();
        buf.push(0xFF);
        let err = IdMap::open(FileSlice::from(buf)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}

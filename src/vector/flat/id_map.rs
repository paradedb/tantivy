//! Per-segment mappings between stored vector rows and document identifiers.

use std::io::{self, Write};
use std::mem::size_of;

use columnar::column_index::{open_optional_index, serialize_optional_index, OptionalIndex, Set};
use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::DocId;

const VARIANT_IDENTITY: u8 = 0;
const VARIANT_BITMAP: u8 = 1;
/// Explicit row-to-document mapping tag.
pub(crate) const VARIANT_EXPLICIT: u8 = 2;

/// Decodes one document identifier from an explicit map body.
#[inline]
fn explicit_doc_id_at(bytes: &[u8], row: usize) -> DocId {
    let start = row * size_of::<DocId>();
    DocId::from_le_bytes(bytes[start..start + size_of::<DocId>()].try_into().unwrap())
}

/// Maps stored vector rows to document identifiers.
pub enum IdMap {
    /// Identity mapping for fields present on every document.
    Identity { num_docs: u32 },
    /// Bitmap mapping for optional vector fields.
    Bitmap(OptionalIndex),
    /// Explicit row-to-document mapping for cluster-ordered rows.
    Explicit(OwnedBytes),
}

impl IdMap {
    /// Serializes an identity or bitmap mapping for sorted document identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error when the output cannot be written.
    pub fn serialize<W: Write>(
        present_doc_ids: &[DocId],
        num_docs: u32,
        out: &mut W,
    ) -> io::Result<()> {
        if present_doc_ids.len() == num_docs as usize {
            out.write_all(&[VARIANT_IDENTITY])?;
        } else {
            out.write_all(&[VARIANT_BITMAP])?;
            serialize_optional_index(&present_doc_ids, num_docs, out)?;
        }
        Ok(())
    }

    /// Serializes an explicit row-to-document mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when the output cannot be written.
    pub fn serialize_explicit<W: Write>(row_doc_ids: &[DocId], out: &mut W) -> io::Result<()> {
        out.write_all(&[VARIANT_EXPLICIT])?;
        for doc_id in row_doc_ids {
            doc_id.serialize(out)?;
        }
        Ok(())
    }

    /// Opens a serialized row-to-document mapping.
    ///
    /// # Errors
    ///
    /// Returns an error for unreadable or invalid input.
    pub fn open(file_slice: FileSlice, num_docs: u32) -> io::Result<IdMap> {
        if file_slice.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "id map section is empty",
            ));
        }
        let tag = file_slice.slice(0..1).read_bytes()?[0];
        let body = file_slice.slice_from(1);
        match tag {
            VARIANT_IDENTITY => Ok(IdMap::Identity { num_docs }),
            VARIANT_BITMAP => Ok(IdMap::Bitmap(open_optional_index(body)?)),
            VARIANT_EXPLICIT => {
                let bytes = body.read_bytes()?;
                if bytes.len() % size_of::<DocId>() != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "explicit id map body is not a whole number of u32 doc ids",
                    ));
                }
                Ok(IdMap::Explicit(bytes))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown id map variant tag: {other}"),
            )),
        }
    }

    /// Returns the number of stored rows.
    pub fn num_rows(&self) -> u32 {
        match self {
            IdMap::Identity { num_docs } => *num_docs,
            IdMap::Bitmap(idx) => idx.num_non_nulls(),
            IdMap::Explicit(bytes) => (bytes.len() / size_of::<DocId>()) as u32,
        }
    }

    /// Returns whether the document has a stored row.
    #[cfg(test)]
    #[inline]
    pub fn contains(&self, doc_id: DocId) -> bool {
        match self {
            IdMap::Identity { num_docs } => doc_id < *num_docs,
            IdMap::Bitmap(idx) => Set::contains(idx, doc_id),
            IdMap::Explicit(bytes) => {
                let num_rows = bytes.len() / size_of::<DocId>();
                (0..num_rows).any(|row| explicit_doc_id_at(bytes, row) == doc_id)
            }
        }
    }

    /// Returns the stored row for a document when present.
    #[inline]
    pub fn rank_if_exists(&self, doc_id: DocId) -> Option<u32> {
        match self {
            IdMap::Identity { num_docs } => {
                debug_assert!(doc_id < *num_docs, "doc_id {doc_id} >= num_docs {num_docs}");
                Some(doc_id)
            }
            IdMap::Bitmap(idx) => Set::rank_if_exists(idx, doc_id),
            IdMap::Explicit(bytes) => {
                let num_rows = bytes.len() / size_of::<DocId>();
                (0..num_rows)
                    .find(|&row| explicit_doc_id_at(bytes, row) == doc_id)
                    .map(|row| row as u32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(present: &[DocId], num_docs: u32) -> IdMap {
        let mut buf = Vec::new();
        IdMap::serialize(present, num_docs, &mut buf).unwrap();
        IdMap::open(FileSlice::from(buf), num_docs).unwrap()
    }

    #[test]
    fn test_all_present_uses_identity_variant() {
        let n = 100u32;
        let present: Vec<DocId> = (0..n).collect();

        let mut buf = Vec::new();
        IdMap::serialize(&present, n, &mut buf).unwrap();
        assert_eq!(buf.len(), 1, "Identity variant should write only the tag");
        assert_eq!(buf[0], VARIANT_IDENTITY);

        let p = IdMap::open(FileSlice::from(buf), n).unwrap();
        assert!(matches!(p, IdMap::Identity { num_docs } if num_docs == n));
        assert_eq!(p.num_rows(), n);
        for d in 0..n {
            assert!(p.contains(d));
            assert_eq!(p.rank_if_exists(d), Some(d));
        }
        assert!(!p.contains(n));
    }

    #[test]
    fn test_none_present_uses_bitmap_variant() {
        let p = round_trip(&[], 100);
        assert!(matches!(p, IdMap::Bitmap(_)));
        assert_eq!(p.num_rows(), 0);
        for d in 0..100 {
            assert!(!p.contains(d));
            assert_eq!(p.rank_if_exists(d), None);
        }
    }

    #[test]
    fn test_sparse_uses_bitmap_variant() {
        let present: Vec<DocId> = vec![3, 7, 11, 12, 50, 99];
        let p = round_trip(&present, 100);
        assert!(matches!(p, IdMap::Bitmap(_)));
        assert_eq!(p.num_rows(), 6);
        for (row, &doc) in present.iter().enumerate() {
            assert!(p.contains(doc));
            assert_eq!(p.rank_if_exists(doc), Some(row as u32));
        }
        for d in [0u32, 1, 2, 4, 5, 6, 8, 9, 10, 13, 49, 51, 98] {
            assert!(!p.contains(d));
            assert_eq!(p.rank_if_exists(d), None);
        }
    }

    #[test]
    fn test_bitmap_across_blocks() {
        let n = 1500u32;
        let present: Vec<DocId> = (0..n).filter(|d| d % 3 == 0).collect();
        let p = round_trip(&present, n);
        assert!(matches!(p, IdMap::Bitmap(_)));
        assert_eq!(p.num_rows() as usize, present.len());
        for (row, &doc) in present.iter().enumerate() {
            assert_eq!(p.rank_if_exists(doc), Some(row as u32));
        }
        for d in 0..n {
            if d % 3 != 0 {
                assert!(!p.contains(d));
            }
        }
    }

    #[test]
    fn test_doc_id_beyond_num_docs() {
        let p = round_trip(&[1, 5], 10);
        assert!(!p.contains(10));
        assert!(!p.contains(100));
        assert_eq!(p.rank_if_exists(10), None);
    }

    #[test]
    fn test_explicit_round_trip() {
        let row_doc_ids: Vec<DocId> = vec![1, 4, 0, 3];
        let mut buf = Vec::new();
        IdMap::serialize_explicit(&row_doc_ids, &mut buf).unwrap();
        assert_eq!(buf[0], VARIANT_EXPLICIT);

        let p = IdMap::open(FileSlice::from(buf), 5).unwrap();
        assert!(matches!(p, IdMap::Explicit(_)));
        assert_eq!(p.num_rows(), 4);
        for (row, &doc) in row_doc_ids.iter().enumerate() {
            assert!(p.contains(doc));
            assert_eq!(p.rank_if_exists(doc), Some(row as u32));
        }
        assert!(!p.contains(2));
        assert_eq!(p.rank_if_exists(2), None);
    }
}

//! Row-to-document addressing for vector columns.
use std::io::{self, Write};

use columnar::column_index::{open_optional_index, serialize_optional_index, OptionalIndex, Set};
use common::{HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::vector::storage_io::VectorRead;
use crate::DocId;

const VARIANT_IDENTITY: u8 = 0;
const VARIANT_BITMAP: u8 = 1;
const VARIANT_EXPLICIT: u8 = 2;

pub enum IdMap {
    Identity { num_docs: u32 },
    Bitmap(OptionalIndex),
    Explicit(OwnedBytes),
}

impl IdMap {
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

    pub(crate) fn serialize_explicit<W: Write>(docs: &[DocId], out: &mut W) -> io::Result<()> {
        out.write_all(&[VARIANT_EXPLICIT])?;
        for doc in docs {
            out.write_all(&doc.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn open(file_slice: FileSlice, num_docs: u32) -> io::Result<Self> {
        if file_slice.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "id map section is empty",
            ));
        }
        let tag = file_slice.slice_to(1).read_bytes()?[0];
        let body = file_slice.slice_from(1);
        match tag {
            VARIANT_IDENTITY if body.len() == 0 => Ok(Self::Identity { num_docs }),
            VARIANT_BITMAP => Ok(Self::Bitmap(open_optional_index(body)?)),
            VARIANT_EXPLICIT if body.len() % 4 == 0 => {
                let mut bytes = Vec::with_capacity(body.len());
                body.read_vector_chunks(&mut |chunk| bytes.extend_from_slice(chunk))?;
                Ok(Self::Explicit(OwnedBytes::new(bytes)))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid id map variant or length",
            )),
        }
    }

    pub(crate) fn validate_rows(&self, rows: &[usize]) -> io::Result<()> {
        let Self::Explicit(docs) = self else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "clustered lookup requires an explicit row map",
            ));
        };
        if rows.first() != Some(&0)
            || rows.last().copied() != Some(docs.len() / 4)
            || rows.windows(2).any(|range| range[0] > range[1])
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "row document map must cover all cluster rows",
            ));
        }
        Ok(())
    }
    /// Number of present vector rows.
    pub fn num_rows(&self) -> u32 {
        match self {
            Self::Identity { num_docs } => *num_docs,
            Self::Bitmap(idx) => idx.num_non_nulls(),
            Self::Explicit(docs) => (docs.len() / 4) as u32,
        }
    }
    #[cfg(test)]
    pub fn contains(&self, doc: DocId) -> bool {
        self.rank_if_exists(doc).is_some()
    }
    /// Resolves a flat document by rank without searching rows.
    pub fn rank_if_exists(&self, doc: DocId) -> Option<u32> {
        match self {
            Self::Identity { num_docs } => (doc < *num_docs).then_some(doc),
            Self::Bitmap(idx) => Set::rank_if_exists(idx, doc),
            Self::Explicit(_) => unreachable!("clustered documents require a row search"),
        }
    }
    /// Resolves a vector row to its document ID.
    pub(crate) fn doc_at(&self, row: u32) -> DocId {
        match self {
            Self::Identity { .. } => row,
            Self::Bitmap(idx) => idx.select(row),
            Self::Explicit(docs) => {
                let start = row as usize * 4;
                DocId::from_le_bytes(docs[start..start + 4].try_into().unwrap())
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

        // Wire-level: the serialized output is exactly 1 byte (just the
        // variant tag); no body — num_docs comes from the caller.
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
        // Out-of-range queries are the caller's responsibility:
        // `contains` returns false, but `rank_if_exists` requires
        // `doc_id < num_docs` (asserted in debug builds).
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
        // Exercise multiple roaring-style blocks (each spans 64K docs).
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
    fn explicit_rows_round_trip_and_validate_clusters() {
        let mut bytes = Vec::new();
        IdMap::serialize_explicit(&[1, 4, 0, 3], &mut bytes).unwrap();
        assert_eq!(bytes.len(), 1 + 4 * 4);
        let map = IdMap::open(FileSlice::from(bytes.clone()), 5).unwrap();
        map.validate_rows(&[0, 2, 2, 4]).unwrap();
        assert_eq!(
            (0..4).map(|row| map.doc_at(row)).collect::<Vec<_>>(),
            [1, 4, 0, 3]
        );
        for offsets in [
            &[0, 3][..],
            &[0, 2, 3],
            &[1, 2, 4],
            &[0, 4, 2, 4],
            &[0, 5, 4],
        ] {
            assert!(map.validate_rows(offsets).is_err());
        }
        bytes.pop();
        assert!(IdMap::open(FileSlice::from(bytes), 5).is_err());
        let mut empty = Vec::new();
        IdMap::serialize_explicit(&[], &mut empty).unwrap();
        IdMap::open(FileSlice::from(empty), 0)
            .unwrap()
            .validate_rows(&[0])
            .unwrap();
    }
}

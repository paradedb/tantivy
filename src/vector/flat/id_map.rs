//! Document addressing for vector columns. Flat columns use identity or bitmap rank;
//! clustered columns use fixed-width document locations read one entry at a time.
use std::io::{self, Write};

use columnar::column_index::{open_optional_index, serialize_optional_index, OptionalIndex, Set};
use common::HasLen;

use crate::directory::FileSlice;
use crate::vector::storage_io::VectorRead;
use crate::DocId;

const VARIANT_IDENTITY: u8 = 0;
const VARIANT_BITMAP: u8 = 1;
const VARIANT_DOC_LOCATIONS: u8 = 3;

/// A document's cluster and local row; the sentinel represents a missing vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DocLocation {
    pub(crate) cluster: u32,
    pub(crate) local: u32,
}
impl DocLocation {
    /// Sentinel for a document without a vector; local must be zero.
    pub(crate) const ABSENT: Self = Self {
        cluster: u32::MAX,
        local: 0,
    };
}

/// Document addressing with deferred, fixed-width clustered lookups.
pub enum IdMap {
    /// Every document has a vector at the same row ordinal.
    Identity { num_docs: u32 },
    /// Present documents address dense rows by bitmap rank.
    Bitmap(OptionalIndex),
    /// One cluster/local pair per segment document; the body remains on storage.
    DocLocations(FileSlice),
}
impl IdMap {
    /// Serializes sorted flat document ids, eliding a fully populated bitmap.
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
    /// Serializes exactly one location for every segment document.
    pub(crate) fn serialize_locations<W: Write>(
        locations: &[DocLocation],
        out: &mut W,
    ) -> io::Result<()> {
        out.write_all(&[VARIANT_DOC_LOCATIONS])?;
        for location in locations {
            out.write_all(&location.cluster.to_le_bytes())?;
            out.write_all(&location.local.to_le_bytes())?;
        }
        Ok(())
    }
    /// Validates the tag and table length without reading clustered table contents.
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
            VARIANT_DOC_LOCATIONS if body.len() as u64 == u64::from(num_docs) * 8 => {
                Ok(Self::DocLocations(body))
            }
            VARIANT_DOC_LOCATIONS => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "document location table length does not equal 8 * max_doc",
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid id map variant or length",
            )),
        }
    }
    /// Reads one document location and validates both coordinates before exposing it.
    pub(crate) fn locate(&self, doc: DocId, rows: &[usize]) -> io::Result<Option<DocLocation>> {
        let Self::DocLocations(body) = self else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "clustered lookup requires DocLocations",
            ));
        };
        let Some(start) = (doc as usize).checked_mul(8) else {
            return Ok(None);
        };
        if start >= body.len() {
            return Ok(None);
        }
        let bytes = body.slice(start..start + 8).read_vector_bytes()?;
        Self::decode_location(&bytes, rows)
    }

    fn decode_location(bytes: &[u8], rows: &[usize]) -> io::Result<Option<DocLocation>> {
        let cluster = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let local = u32::from_le_bytes(bytes[4..].try_into().unwrap());
        if cluster == u32::MAX {
            return if local == 0 {
                Ok(None)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "absent document location has nonzero local row",
                ))
            };
        }
        let cluster_idx = cluster as usize;
        if cluster_idx >= rows.len().saturating_sub(1)
            || local as usize >= rows[cluster_idx + 1] - rows[cluster_idx]
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "document location is outside its cluster",
            ));
        }
        Ok(Some(DocLocation { cluster, local }))
    }

    pub(crate) fn row_doc_ids(&self, rows: &[usize]) -> io::Result<Box<[DocId]>> {
        let Self::DocLocations(body) = self else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "clustered lookup requires DocLocations",
            ));
        };
        self.row_doc_ids_for(rows, 0..(body.len() / 8) as DocId, true)
    }

    pub(crate) fn filtered_row_doc_ids(
        &self,
        rows: &[usize],
        filter: &common::BitSet,
    ) -> io::Result<Box<[DocId]>> {
        let selected = (0..filter.max_value().div_ceil(64)).flat_map(|bucket| {
            filter
                .tinyset(bucket)
                .into_iter()
                .map(move |bit| bucket * 64 + bit)
        });
        self.row_doc_ids_for(rows, selected, false)
    }

    fn row_doc_ids_for(
        &self,
        rows: &[usize],
        mut selected: impl Iterator<Item = DocId>,
        complete: bool,
    ) -> io::Result<Box<[DocId]>> {
        let Self::DocLocations(body) = self else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "clustered lookup requires DocLocations",
            ));
        };
        let mut docs = vec![DocId::MAX; rows.last().copied().unwrap_or(0)];
        let mut last_local = vec![u32::MAX; rows.len().saturating_sub(1)];
        let mut next = selected.next();
        let mut offset = 0;
        let mut entry = [0; 8];
        let mut error = None;
        body.read_vector_chunks(&mut |bytes| {
            let end = offset + bytes.len();
            while error.is_none() {
                let Some(doc) = next else {
                    break;
                };
                let start = doc as usize * 8;
                if start >= end {
                    break;
                }
                let first = start.max(offset);
                let last = (start + 8).min(end);
                entry[first - start..last - start]
                    .copy_from_slice(&bytes[first - offset..last - offset]);
                if last < start + 8 {
                    break;
                }
                match Self::decode_location(&entry, rows) {
                    Ok(Some(location)) => {
                        let cluster = location.cluster as usize;
                        let previous = &mut last_local[cluster];
                        if *previous != u32::MAX && *previous >= location.local {
                            error = Some(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "document locations must ascend within each cluster",
                            ));
                            break;
                        }
                        *previous = location.local;
                        docs[rows[cluster] + location.local as usize] = doc;
                    }
                    Ok(None) => (),
                    Err(err) => {
                        error = Some(err);
                        break;
                    }
                }
                next = selected.next();
            }
            offset = end;
        })?;
        if let Some(err) = error {
            return Err(err);
        }
        if next.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "document id is outside the location table",
            ));
        }
        if complete && docs.contains(&DocId::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "document locations must cover all rows in ascending cluster order",
            ));
        }
        Ok(docs.into_boxed_slice())
    }
    /// Number of present flat rows; clustered row counts reside in cluster offsets.
    pub fn num_rows(&self) -> u32 {
        match self {
            Self::Identity { num_docs } => *num_docs,
            Self::Bitmap(idx) => idx.num_non_nulls(),
            Self::DocLocations(_) => unreachable!("cluster offsets determine row count"),
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
            Self::DocLocations(_) => unreachable!("clustered documents require locate"),
        }
    }
    /// Resolves a flat row by bitmap select or identity.
    pub(crate) fn doc_at(&self, row: u32) -> DocId {
        match self {
            Self::Identity { .. } => row,
            Self::Bitmap(idx) => idx.select(row),
            Self::DocLocations(_) => unreachable!("clustered rows contain DocIds"),
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
    fn cached_locations_handle_split_records() {
        use std::ops::Range;
        use std::sync::Arc;

        use common::OwnedBytes;

        use crate::directory::FileHandle;

        #[derive(Debug)]
        struct Fragments(Vec<u8>, usize);
        impl HasLen for Fragments {
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        impl FileHandle for Fragments {
            fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
                Ok(OwnedBytes::new(self.0[range].to_vec()))
            }
            fn read_bytes_chunks(
                &self,
                range: Range<usize>,
                visitor: &mut dyn FnMut(&[u8]),
            ) -> io::Result<()> {
                for chunk in self.0[range].chunks(self.1) {
                    visitor(chunk);
                }
                Ok(())
            }
        }
        let locations = [
            DocLocation {
                cluster: 2,
                local: 0,
            },
            DocLocation::ABSENT,
            DocLocation {
                cluster: 0,
                local: 0,
            },
            DocLocation {
                cluster: 2,
                local: 1,
            },
        ];
        let mut bytes = Vec::new();
        IdMap::serialize_locations(&locations, &mut bytes).unwrap();
        for chunk in [1, 3, 7, 8, 9, 17] {
            let map =
                IdMap::open(FileSlice::new(Arc::new(Fragments(bytes.clone(), chunk))), 4).unwrap();
            assert_eq!(&*map.row_doc_ids(&[0, 1, 1, 3]).unwrap(), &[2, 0, 3]);
            for (selected, expected) in [
                (vec![1, 3], vec![DocId::MAX, DocId::MAX, 3]),
                (vec![2], vec![2, DocId::MAX, DocId::MAX]),
                (vec![], vec![DocId::MAX; 3]),
            ] {
                let mut filter = common::BitSet::with_max_value(4);
                for doc in selected {
                    filter.insert(doc);
                }
                assert_eq!(
                    &*map.filtered_row_doc_ids(&[0, 1, 1, 3], &filter).unwrap(),
                    &expected
                );
            }
        }
    }

    // Every present document resolves to a checked pair; sentinels and invalid pairs are distinct.
    #[test]
    fn locations_round_trip_absence_and_corruption() {
        let expected = [
            DocLocation {
                cluster: 1,
                local: 0,
            },
            DocLocation {
                cluster: 0,
                local: 0,
            },
            DocLocation::ABSENT,
            DocLocation {
                cluster: 1,
                local: 1,
            },
            DocLocation {
                cluster: 0,
                local: 1,
            },
        ];
        let mut bytes = Vec::new();
        IdMap::serialize_locations(&expected, &mut bytes).unwrap();
        assert_eq!(bytes.len(), 1 + 8 * expected.len());
        let map = IdMap::open(FileSlice::from(bytes.clone()), 5).unwrap();
        assert_eq!(&*map.row_doc_ids(&[0, 2, 4]).unwrap(), &[1, 4, 0, 3]);
        for (doc, &location) in expected.iter().enumerate() {
            assert_eq!(
                map.locate(doc as u32, &[0, 2, 4]).unwrap(),
                (location != DocLocation::ABSENT).then_some(location)
            );
        }
        assert_eq!(map.locate(5, &[0, 2, 4]).unwrap(), None);
        for location in [
            DocLocation {
                cluster: 2,
                local: 0,
            },
            DocLocation {
                cluster: 0,
                local: 2,
            },
            DocLocation {
                cluster: u32::MAX,
                local: 1,
            },
        ] {
            let mut corrupt = bytes.clone();
            corrupt[1..5].copy_from_slice(&location.cluster.to_le_bytes());
            corrupt[5..9].copy_from_slice(&location.local.to_le_bytes());
            let map = IdMap::open(FileSlice::from(corrupt), 5).unwrap();
            assert_eq!(
                map.locate(0, &[0, 2, 4]).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            assert_eq!(
                map.row_doc_ids(&[0, 2, 4]).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
        let mut duplicate = expected;
        duplicate[0] = expected[1];
        let mut missing = expected;
        missing[0] = DocLocation::ABSENT;
        let mut descending = expected;
        descending.swap(0, 3);
        for locations in [duplicate, missing, descending] {
            let mut corrupt = Vec::new();
            IdMap::serialize_locations(&locations, &mut corrupt).unwrap();
            let map = IdMap::open(FileSlice::from(corrupt), 5).unwrap();
            assert_eq!(
                map.row_doc_ids(&[0, 2, 4]).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
        assert!(IdMap::open(FileSlice::from(bytes.clone()), 4).is_err());
        bytes[0] = 2;
        assert!(IdMap::open(FileSlice::from(bytes), 5).is_err());
    }
}

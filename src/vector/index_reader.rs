//! Per-(segment, field) vector reader, modeled on
//! [`InvertedIndexReader`](crate::index::InvertedIndexReader).
//!
//! One [`VectorIndexReader`] serves one vector field of one segment, opened
//! (and cached) via
//! [`SegmentReader::vector_index`](crate::SegmentReader::vector_index). Small
//! routing state is parsed once and pinned in memory, while the bulk payload
//! stays behind [`FileSlice`]s and is fetched with ranged reads at query time.
//!
//! The segment's `.vec` composite holds one of two layouts, discriminated
//! by the [`IdMap`] variant tag in slot `[0]`:
//!
//! * clustered — an `Explicit` (cluster-sorted) id-map, the dense rows in slot `[1]`, and slots
//!   `[2..=4]` the per-segment IVF remainder parsed into [`SegmentClusters`] — offsets, bounds, and
//!   doc counts. The centroid rows themselves live in the index-level `centroids` file (see
//!   [`centroid_index`](super::centroid_index)); this reader never touches them.
//! * flat — an `Identity`/`Bitmap` (doc-ordered) id-map and the rows, nothing else. Written by
//!   indexes without a centroid index (the mutable/staging tier) and searched exhaustively;
//!   [`Self::index`] is `None`.
//!
//! A field with no vectors owns no slots at all; any other partial slot
//! set — or a tag that disagrees with the slots present — is corrupt, not
//! old.

use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use super::header::{read_header, vec_slot, VectorFileVersion};
use super::id_map::{IdMap, VARIANT_EXPLICIT};
use super::ivf::SegmentClusters;
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    /// Distinct documents with a vector in this field. The per-cluster
    /// numbers (`cluster_stats`, [`VectorIndexReader::cluster_sizes`]) count
    /// posting rows, so with replication their sum exceeds `num_vectors`.
    pub num_vectors: usize,
    /// `0` for a flat (unclustered) segment, which assigned against
    /// nothing and is searched exhaustively.
    pub num_centroids: usize,
    pub cluster_stats: VectorClusterStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_cluster_size: f64,
    pub empty_clusters: usize,
}

/// Per-(segment, field) vector reader: the row store plus the per-segment
/// IVF remainder. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct docs with a vector (the persisted IVF doc count; the row
    /// total replication inflates is [`SegmentClusters::num_rows`]).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: Option<Arc<IdMap>>,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    index: Option<Arc<SegmentClusters>>,
}

pub(crate) struct VectorIndexMetadata {
    options: VectorOptions,
    num_vectors: usize,
    present: bool,
    id_map_slice: Option<FileSlice>,
    flat_id_map: Option<Arc<IdMap>>,
    rows_slice: FileSlice,
    index: Option<Arc<SegmentClusters>>,
}

pub(crate) fn visit_rows(
    slice: &FileSlice,
    stride: usize,
    rows: std::ops::Range<usize>,
    scratch: &mut Vec<u8>,
    mut visitor: impl FnMut(usize, &[u8]),
) -> crate::Result<()> {
    scratch.clear();
    if rows.is_empty() {
        return Ok(());
    }
    if stride == 0 {
        return Err(TantivyError::InvalidArgument(
            "vector stride is zero".into(),
        ));
    }
    let expected = rows.len() * stride;
    let mut received = 0;
    let mut row = rows.start;
    let mut invalid = false;
    slice.read_bytes_chunks(rows.start * stride..rows.end * stride, &mut |mut bytes| {
        if invalid || bytes.len() > expected - received {
            invalid = true;
            return;
        }
        received += bytes.len();
        if !scratch.is_empty() {
            let take = (stride - scratch.len()).min(bytes.len());
            scratch.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if scratch.len() < stride {
                return;
            }
            visitor(row, scratch);
            row += 1;
            scratch.clear();
        }
        let mut whole_rows = bytes.chunks_exact(stride);
        for bytes in &mut whole_rows {
            visitor(row, bytes);
            row += 1;
        }
        let remainder = whole_rows.remainder();
        if !remainder.is_empty() && scratch.capacity() < stride {
            scratch.reserve_exact(stride);
        }
        scratch.extend_from_slice(remainder);
    })?;
    if invalid || received != expected || row != rows.end || !scratch.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "vector row chunks do not cover the requested range",
        )
        .into());
    }
    Ok(())
}

impl VectorIndexMetadata {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// [`empty`](Self::empty) placeholder when the segment carries no vector
    /// data for the field (no `.vec` file, or the field has no slots in it),
    /// mirroring `SegmentReader::inverted_index`.
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment_reader.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(opts) => opts.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )));
            }
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (version, body) = read_header(&vec_file)?;
        // V3 moved the centroids to the index level, removed the flat
        // layout, and folded the per-segment remainder into `.vec`. There
        // is no pre-V3 execution path to fall back to — an old segment is
        // refused with the one remedy there is.
        if version < VectorFileVersion::V3 {
            return Err(TantivyError::InvalidArgument(format!(
                "Vector file predates the V3 index-level centroid format; the segment must be \
                 rebuilt with the current index version: <{:?}>",
                entry.name()
            )));
        }
        let vec_composite = CompositeFile::open(&body)?;
        let slots = (
            vec_composite.open_read_with_idx(field, vec_slot::ID_MAP),
            vec_composite.open_read_with_idx(field, vec_slot::ROWS),
            vec_composite.open_read_with_idx(field, vec_slot::OFFSETS),
            vec_composite.open_read_with_idx(field, vec_slot::BOUNDS),
            vec_composite.open_read_with_idx(field, vec_slot::IVF_META),
        );
        let (id_map_slice, rows_slice, ivf_slices) = match slots {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, Some((c, d, e))),
            (Some(a), Some(b), None, None, None) => (a, b, None),
            (None, None, None, None, None) => return Ok(Self::empty(options)),
            _ => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has a partial `.vec` slot set — the file is corrupt",
                    entry.name()
                )));
            }
        };

        let (num_rows, flat_id_map) = id_map_metadata(&id_map_slice, segment_reader.max_doc())?;
        if flat_id_map.is_some() == ivf_slices.is_some() {
            return Err(TantivyError::InternalError(format!(
                "vector field {:?}: the id-map variant disagrees with the slots present — the \
                 file is corrupt",
                entry.name()
            )));
        }
        let index = match ivf_slices {
            Some((offsets_slice, bounds_slice, meta_slice)) => Some(Arc::new(
                SegmentClusters::open(&options, offsets_slice, bounds_slice, meta_slice)?,
            )),
            None => None,
        };

        if let Some(index) = &index {
            if index.num_rows() != num_rows {
                return Err(TantivyError::InternalError(
                    "IVF id-map length does not match the cluster offsets".to_string(),
                ));
            }
        }
        if rows_slice.len() != num_rows * options.bytes_per_vector() {
            return Err(TantivyError::InternalError(format!(
                "vector rows length {} does not match {} rows of {} bytes",
                rows_slice.len(),
                num_rows,
                options.bytes_per_vector()
            )));
        }

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map_slice: Some(id_map_slice),
            flat_id_map: flat_id_map.map(Arc::new),
            index,
        })
    }

    fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            id_map_slice: None,
            flat_id_map: None,
            rows_slice: FileSlice::empty(),
            index: None,
        }
    }

    pub(crate) fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub(crate) fn clusters(&self) -> Option<&SegmentClusters> {
        self.index.as_deref()
    }
}

fn id_map_metadata(slice: &FileSlice, num_docs: u32) -> std::io::Result<(usize, Option<IdMap>)> {
    if !slice.is_empty() && slice.slice(0..1).read_bytes()?[0] == VARIANT_EXPLICIT {
        let bytes = slice.len() - 1;
        if bytes % std::mem::size_of::<DocId>() != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "explicit id map body is not a whole number of u32 doc ids",
            ));
        }
        Ok((bytes / std::mem::size_of::<DocId>(), None))
    } else {
        let map = IdMap::open(slice.clone(), num_docs)?;
        Ok((map.num_rows() as usize, Some(map)))
    }
}

impl VectorIndexReader {
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let metadata = segment_reader.vector_index_metadata(field)?;
        if !metadata.present {
            return Ok(Self::empty(metadata.options.clone()));
        }
        let id_map = match &metadata.flat_id_map {
            Some(map) => Some(Arc::clone(map)),
            None => metadata
                .id_map_slice
                .as_ref()
                .map(|slice| IdMap::open(slice.clone(), segment_reader.max_doc()).map(Arc::new))
                .transpose()?,
        };
        Ok(Self {
            options: metadata.options.clone(),
            num_vectors: metadata.num_vectors,
            present: metadata.present,
            id_map,
            rows_slice: metadata.rows_slice.clone(),
            index: metadata.index.clone(),
        })
    }

    /// The no-data placeholder: zero vectors, no index. Every accessor
    /// behaves as an empty column, so callers never branch on presence.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            rows_slice: FileSlice::empty(),
            id_map: None,
            index: None,
        }
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The per-segment IVF remainder; `None` for a flat (unclustered)
    /// segment and for the [`empty`](Self::empty) placeholder.
    pub fn clusters(&self) -> Option<&SegmentClusters> {
        self.index.as_deref()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
    pub fn info(&self) -> Option<VectorInfo> {
        if !self.present {
            return None;
        }
        let Some(index) = self.index.as_ref() else {
            // Flat (unclustered) segment: zero centroids, version 0.
            return Some(VectorInfo {
                num_vectors: self.num_vectors,
                num_centroids: 0,
                cluster_stats: VectorClusterStats {
                    min_cluster_size: 0,
                    max_cluster_size: 0,
                    avg_cluster_size: 0.0,
                    empty_clusters: 0,
                },
            });
        };
        let mut empty_clusters = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;
        let mut total_cluster_size = 0;
        for cluster_size in index.cluster_sizes() {
            empty_clusters += usize::from(cluster_size == 0);
            min_cluster_size = min_cluster_size.min(cluster_size);
            max_cluster_size = max_cluster_size.max(cluster_size);
            total_cluster_size += cluster_size;
        }
        let num_centroids = index.num_clusters();
        let avg_cluster_size = if num_centroids == 0 {
            0.0
        } else {
            total_cluster_size as f64 / num_centroids as f64
        };
        let min_cluster_size = if num_centroids == 0 {
            0
        } else {
            min_cluster_size
        };
        Some(VectorInfo {
            num_vectors: self.num_vectors,
            num_centroids,
            cluster_stats: VectorClusterStats {
                min_cluster_size,
                max_cluster_size,
                avg_cluster_size,
                empty_clusters,
            },
        })
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// segment has no vector data for the field.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// The raw little-endian bytes of `doc_id`'s vector, fetched with one
    /// stride-sized ranged read; `None` if the doc has no vector.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// The raw bytes of the single vector row at `row` of the dense rows
    /// slot, fetched with one stride-sized ranged read
    /// (`row * stride..(row + 1) * stride`). The caller resolves `row`
    /// beforehand (e.g. from a cluster's row range), so no doc→row lookup
    /// happens here.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        let num_rows = self.id_map.as_ref().map(|map| map.num_rows()).unwrap_or(0);
        if row >= num_rows as usize {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let bytes = self
            .rows_slice
            .slice(row * stride..(row + 1) * stride)
            .read_bytes()?;
        Ok(bytes)
    }

    /// Read a contiguous range of dense vector rows.
    pub fn vector_bytes_for_rows(&self, rows: std::ops::Range<usize>) -> crate::Result<OwnedBytes> {
        let num_rows = self.id_map.as_ref().map(|map| map.num_rows()).unwrap_or(0) as usize;
        if rows.start > rows.end || rows.end > num_rows {
            return Err(TantivyError::InvalidArgument(format!(
                "vector rows {rows:?} are out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        Ok(self
            .rows_slice
            .slice(rows.start * stride..rows.end * stride)
            .read_bytes()?)
    }

    /// Visits rows in order, reusing `scratch` only for rows split across storage chunks.
    pub fn visit_vector_rows(
        &self,
        rows: std::ops::Range<usize>,
        scratch: &mut Vec<u8>,
        visitor: impl FnMut(usize, &[u8]),
    ) -> crate::Result<()> {
        let num_rows = self.id_map.as_ref().map(|map| map.num_rows()).unwrap_or(0) as usize;
        if rows.start > rows.end || rows.end > num_rows {
            return Err(TantivyError::InvalidArgument(format!(
                "vector rows {rows:?} are out of bounds"
            )));
        }
        visit_rows(
            &self.rows_slice,
            self.options.bytes_per_vector(),
            rows,
            scratch,
            visitor,
        )
    }

    pub(crate) fn visit_vector_row_fragments(
        &self,
        rows: std::ops::Range<usize>,
        mut visitor: impl FnMut(usize, usize, &[u8]),
    ) -> crate::Result<()> {
        let num_rows = self.id_map.as_ref().map(|map| map.num_rows()).unwrap_or(0) as usize;
        if rows.start > rows.end || rows.end > num_rows {
            return Err(TantivyError::InvalidArgument(format!(
                "vector rows {rows:?} are out of bounds"
            )));
        }
        if rows.is_empty() {
            return Ok(());
        }
        let stride = self.options.bytes_per_vector();
        if stride == 0 {
            return Err(TantivyError::InvalidArgument(
                "vector stride is zero".into(),
            ));
        }
        let expected = rows.len() * stride;
        let (mut received, mut offset, mut row) = (0, 0, rows.start);
        let mut invalid = false;
        self.rows_slice.read_bytes_chunks(
            rows.start * stride..rows.end * stride,
            &mut |mut bytes| {
                if invalid || bytes.len() > expected - received {
                    invalid = true;
                    return;
                }
                received += bytes.len();
                while !bytes.is_empty() {
                    let len = (stride - offset).min(bytes.len());
                    visitor(row, offset, &bytes[..len]);
                    bytes = &bytes[len..];
                    offset += len;
                    if offset == stride {
                        row += 1;
                        offset = 0;
                    }
                }
            },
        )?;
        if invalid || received != expected || row != rows.end || offset != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "vector row fragments do not cover the requested range",
            )
            .into());
        }
        Ok(())
    }

    /// The doc id stored at `row` — decoded from the pinned permutation
    /// for clustered segments, positional for flat ones. Panics on the
    /// empty placeholder.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        self.id_map
            .as_ref()
            .expect("doc_id_at called on a segment with no vector data")
            .doc_id_at(row)
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if the segment
    /// has no vector data or `cluster` is out of bounds.
    pub fn cluster_doc_ids(&self, cluster: usize) -> Option<Vec<DocId>> {
        let index = self.index.as_ref()?;
        if cluster >= index.num_clusters() {
            return None;
        }
        Some(
            index
                .cluster_range(cluster)
                .map(|row| self.doc_id_at(row))
                .collect(),
        )
    }

    /// Doc → dense row. Flat rows ascend by doc id, so the id-map ranks
    /// directly. Clustered rows are cluster-sorted and ascending by doc id
    /// within each cluster, so this scans clusters and binary-searches each
    /// one over the pinned id-map bytes.
    pub(crate) fn row_id(&self, doc_id: DocId) -> Option<usize> {
        use std::cmp::Ordering;
        let id_map = self.id_map.as_ref()?;
        let Some(index) = self.index.as_ref() else {
            return id_map.rank_if_exists(doc_id).map(|row| row as usize);
        };
        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            let mut lo = rows.start;
            let mut hi = rows.end;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                match self.doc_id_at(mid).cmp(&doc_id) {
                    Ordering::Less => lo = mid + 1,
                    Ordering::Greater => hi = mid,
                    Ordering::Equal => return Some(mid),
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::ops::Range;
    use std::sync::Arc;

    use super::*;
    use crate::directory::FileHandle;
    use crate::schema::Metric;

    #[derive(Debug)]
    struct HeaderOnlyIdMap;

    impl HasLen for HeaderOnlyIdMap {
        fn len(&self) -> usize {
            1 + 10_000_000 * std::mem::size_of::<DocId>()
        }
    }

    impl FileHandle for HeaderOnlyIdMap {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            assert_eq!(range, 0..1, "routing must not materialize document ids");
            Ok(OwnedBytes::new(vec![VARIANT_EXPLICIT]))
        }
    }

    #[test]
    fn explicit_id_map_metadata_reads_only_the_tag() -> crate::Result<()> {
        let slice = FileSlice::new(Arc::new(HeaderOnlyIdMap));
        let (rows, map) = id_map_metadata(&slice, 10_000_000)?;
        assert_eq!(rows, 10_000_000);
        assert!(map.is_none());
        Ok(())
    }

    #[test]
    fn id_map_metadata_matches_full_reader_validation() -> crate::Result<()> {
        let mut encodings = Vec::new();
        for ids in [&[0, 1, 2][..], &[0, 2][..], &[][..]] {
            let mut bytes = Vec::new();
            IdMap::serialize(ids, 3, &mut bytes)?;
            encodings.push(bytes);
        }
        let mut explicit = Vec::new();
        IdMap::serialize_explicit(&[2, 0, 1], &mut explicit)?;
        encodings.push(explicit);
        for bytes in encodings {
            let slice = FileSlice::from(bytes);
            let reader = IdMap::open(slice.clone(), 3)?;
            let (rows, map) = id_map_metadata(&slice, 3)?;
            assert_eq!(rows, reader.num_rows() as usize);
            assert_eq!(map.is_some(), reader.is_flat());
        }
        for bytes in [vec![], vec![255], vec![VARIANT_EXPLICIT, 0]] {
            let slice = FileSlice::from(bytes);
            let expected = IdMap::open(slice.clone(), 3).err().unwrap();
            let actual = id_map_metadata(&slice, 3).err().unwrap();
            assert_eq!(actual.kind(), expected.kind());
            assert_eq!(actual.to_string(), expected.to_string());
        }
        Ok(())
    }

    #[test]
    fn full_reader_reuses_routing_metadata() -> crate::Result<()> {
        let fixture =
            crate::vector::tests::TestVectorIndex::builder(crate::schema::VectorDType::F32)
                .build()?;
        let field = fixture.embedding_field();
        let searcher = fixture.index.reader()?.searcher();
        for segment in searcher.segment_readers() {
            let metadata = segment.vector_index_metadata(field)?;
            let cached = segment.vector_index_metadata(field)?;
            assert!(Arc::ptr_eq(&metadata, &cached));
            let full = segment.vector_index(field)?;
            assert!(Arc::ptr_eq(
                metadata.index.as_ref().unwrap(),
                full.index.as_ref().unwrap()
            ));
            assert_eq!(metadata.num_vectors, full.num_vectors());
            let expected = IdMap::open(
                metadata.id_map_slice.as_ref().unwrap().clone(),
                segment.max_doc(),
            )?;
            for row in 0..expected.num_rows() as usize {
                assert_eq!(full.doc_id_at(row), expected.doc_id_at(row));
            }
        }
        Ok(())
    }

    #[derive(Debug)]
    struct FragmentedFile {
        bytes: Vec<u8>,
        chunk_size: usize,
        length_delta: isize,
    }

    impl HasLen for FragmentedFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for FragmentedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }

        fn read_bytes_chunks(
            &self,
            range: Range<usize>,
            visitor: &mut dyn FnMut(&[u8]),
        ) -> io::Result<()> {
            let mut start = range.start;
            let end = range.end.checked_add_signed(self.length_delta).unwrap();
            while start < end {
                let next = ((start / self.chunk_size + 1) * self.chunk_size).min(end);
                visitor(&self.bytes[start..next]);
                start = next;
            }
            Ok(())
        }
    }

    fn fragmented_reader(
        dim: usize,
        chunk_size: usize,
        offset: usize,
        length_delta: isize,
    ) -> (VectorIndexReader, Arc<FragmentedFile>) {
        let mut bytes = vec![0; offset];
        for value in 0..dim * 5 {
            bytes.extend_from_slice(&(value as f32 * 0.25).to_le_bytes());
        }
        let end = bytes.len();
        bytes.extend_from_slice(&[0; 16]);
        let file = Arc::new(FragmentedFile {
            bytes,
            chunk_size,
            length_delta,
        });
        let reader = VectorIndexReader {
            options: VectorOptions::new(dim, Metric::Dot),
            num_vectors: 5,
            present: true,
            id_map: Some(Arc::new(IdMap::Identity { num_docs: 5 })),
            rows_slice: FileSlice::new(file.clone()).slice(offset..end),
            index: None,
        };
        (reader, file)
    }

    #[test]
    fn row_visitor_preserves_fragmented_rows() -> crate::Result<()> {
        for dim in [1, 3, 1024, 4099] {
            for chunk_size in [1, 3, 7, 8160, 16321] {
                let (reader, _) = fragmented_reader(dim, chunk_size, 5, 0);
                let stride = reader.options.bytes_per_vector();
                let mut scratch = Vec::new();
                for rows in [0..5, 1..4, 4..5, 2..2] {
                    let expected = reader.vector_bytes_for_rows(rows.clone())?;
                    let mut visited = Vec::new();
                    reader.visit_vector_rows(rows.clone(), &mut scratch, |row, bytes| {
                        assert_eq!(row, rows.start + visited.len());
                        assert_eq!(bytes.len(), stride);
                        visited.push(bytes.to_vec());
                    })?;
                    assert_eq!(visited.concat(), expected.as_slice());
                    assert!(scratch.is_empty());
                    assert!(scratch.capacity() <= stride);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn row_visitor_borrows_whole_rows() -> crate::Result<()> {
        let (reader, file) = fragmented_reader(3, 48, 0, 0);
        let mut scratch = Vec::new();
        reader.visit_vector_rows(0..5, &mut scratch, |row, bytes| {
            assert_eq!(
                bytes.as_ptr() as usize,
                file.bytes.as_ptr() as usize + row * 12
            );
        })?;
        assert_eq!(scratch.capacity(), 0);
        Ok(())
    }

    #[test]
    fn row_visitor_rejects_invalid_ranges_and_incomplete_data() {
        let (reader, _) = fragmented_reader(3, 7, 5, 0);
        for rows in [4..6, 3..2, 6..6] {
            assert!(reader
                .visit_vector_rows(rows, &mut Vec::new(), |_, _| panic!("invalid row visited"))
                .is_err());
        }
        for delta in [-1, 1] {
            let (reader, _) = fragmented_reader(3, 7, 5, delta);
            let err = reader
                .visit_vector_rows(1..4, &mut Vec::new(), |_, _| {})
                .unwrap_err();
            assert!(err.to_string().contains("do not cover the requested range"));
        }
    }

    #[test]
    fn row_fragments_borrow_and_cover_arbitrary_byte_boundaries() -> crate::Result<()> {
        for dim in [1, 3, 1024, 4099] {
            for chunk_size in [1, 3, 7, 8160, 16321] {
                let (reader, file) = fragmented_reader(dim, chunk_size, 5, 0);
                let stride = reader.options.bytes_per_vector();
                for rows in [0..5, 1..4, 4..5, 2..2] {
                    let expected = reader.vector_bytes_for_rows(rows.clone())?;
                    let mut actual = Vec::new();
                    reader.visit_vector_row_fragments(rows.clone(), |row, offset, bytes| {
                        assert_eq!(row, rows.start + actual.len() / stride);
                        assert_eq!(offset, actual.len() % stride);
                        assert_eq!(
                            bytes.as_ptr() as usize,
                            file.bytes.as_ptr() as usize + 5 + row * stride + offset,
                        );
                        assert!(!bytes.is_empty() && offset + bytes.len() <= stride);
                        actual.extend_from_slice(bytes);
                    })?;
                    assert_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn row_fragments_reject_invalid_ranges_and_incomplete_data() {
        let (reader, _) = fragmented_reader(3, 7, 5, 0);
        for rows in [4..6, 3..2, 6..6] {
            assert!(reader
                .visit_vector_row_fragments(rows, |_, _, _| panic!("invalid row visited"))
                .is_err());
        }
        for delta in [-1, 1] {
            let (reader, _) = fragmented_reader(3, 7, 5, delta);
            let err = reader
                .visit_vector_row_fragments(1..4, |_, _, _| {})
                .unwrap_err();
            assert!(err.to_string().contains("do not cover the requested range"));
        }
    }
}

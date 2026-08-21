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
//!   `[2..=4]` the per-segment IVF remainder parsed into [`IvfIndex`] — offsets, bounds, and the
//!   centroid-set version the segment assigned against. The centroid rows themselves live in the
//!   index-level `centroids.<version>` file (see [`centroid_set`](super::centroid_set)); this
//!   reader never touches them.
//! * flat — an `Identity`/`Bitmap` (doc-ordered) id-map and the rows, nothing else. Written by
//!   indexes without a centroid set (the mutable/staging tier) and searched exhaustively;
//!   [`Self::index`] is `None`.
//!
//! A field with no vectors owns no slots at all; any other partial slot
//! set — or a tag that disagrees with the slots present — is corrupt, not
//! old.

use common::{HasLen, OwnedBytes};

use super::header::{read_header, vec_slot, VectorFileVersion};
use super::id_map::IdMap;
use super::ivf::IvfIndex;
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
    /// total replication inflates is [`IvfIndex::num_rows`]).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: Option<IdMap>,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    index: Option<IvfIndex>,
}

impl VectorIndexReader {
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

        let id_map = IdMap::open(id_map_slice, segment_reader.max_doc())?;
        if id_map.is_flat() == ivf_slices.is_some() {
            return Err(TantivyError::InternalError(format!(
                "vector field {:?}: the id-map variant disagrees with the slots present — the \
                 file is corrupt",
                entry.name()
            )));
        }
        let index = match ivf_slices {
            Some((offsets_slice, bounds_slice, meta_slice)) => Some(IvfIndex::open(
                &options,
                offsets_slice,
                bounds_slice,
                meta_slice,
            )?),
            None => None,
        };

        let num_rows = id_map.num_rows() as usize;
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
            id_map: Some(id_map),
            index,
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
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
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
        let num_rows = self.id_map.as_ref().map(IdMap::num_rows).unwrap_or(0);
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

//! The `.centroids` file and its reader, [`ClusterIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`ClusterIndex::open`] that parses them
//! back sit side by side.
//!
//! The on-disk file is a 4-byte format-version stamp (see `vector::storage::header`)
//! followed by a [`CompositeFile`](crate::directory::CompositeFile). Written
//! per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `DocLocations`). The composite has four slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride),
//!     rows in the router's canonical cluster-sorted order when a stacked
//!     router was built
//! [1] cumulative cluster row offsets (N + 1 little-endian u64s)
//! [2] a router-kind byte followed by the selected router's payload, REQUIRED
//! ```
//!
//! The V4 layout omits the former bounds slot. Earlier layouts require rebuilding.
use std::io::{self, Write};
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::blocks::Blocks;
use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::schema::{Metric, VectorOptions};
use crate::vector::router::{OpenedRouter, RouterIter, RouterKind, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::VectorElement;

/// Lazy cumulative row offsets from slot 1 of `.centroids`.
pub(crate) struct ClusterOffsets {
    slice: FileSlice,
    num_rows: usize,
}

impl ClusterOffsets {
    pub(crate) fn open(slice: FileSlice) -> crate::Result<Self> {
        if slice.len() < 8 || slice.len() % 8 != 0 {
            return Err(DataCorruption::comment_only("invalid cluster offset length").into());
        }
        let first = slice.slice_to(8).read_bytes()?;
        if u64::from_le_bytes(first.as_slice().try_into().unwrap()) != 0 {
            return Err(DataCorruption::comment_only("cluster offsets must start at zero").into());
        }
        let last = if slice.len() == 8 {
            first
        } else {
            slice.slice_from(slice.len() - 8).read_bytes()?
        };
        let num_rows = u64::from_le_bytes(last.as_slice().try_into().unwrap());
        if num_rows > u64::from(u32::MAX) {
            return Err(DataCorruption::comment_only("cluster row count overflow").into());
        }
        Ok(Self {
            slice,
            num_rows: num_rows as usize,
        })
    }

    pub(crate) fn num_clusters(&self) -> usize {
        self.slice.len() / 8 - 1
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub(crate) fn range(&self, cluster: usize) -> crate::Result<Range<usize>> {
        if cluster >= self.num_clusters() {
            return Err(DataCorruption::comment_only("cluster offset out of range").into());
        }
        let bytes = self
            .slice
            .slice(cluster * 8..(cluster + 2) * 8)
            .read_vector_bytes(VectorReadKind::Other)?;
        let first = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let last = u64::from_le_bytes(bytes[8..].try_into().unwrap());
        if first > last || last > self.num_rows as u64 {
            return Err(DataCorruption::comment_only("invalid cluster row boundaries").into());
        }
        Ok(first as usize..last as usize)
    }
}

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and the
/// router's lists. The centroid vectors stay behind a [`FileSlice`] and are read in contiguous
/// batches as routing visits them. Everything row-scale (the rows and id-map) lives on
/// [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub(crate) struct ClusterIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field.
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words).
    centroids_slice: FileSlice,
    /// Bytes per centroid row: the field's vector stride.
    centroid_stride: usize,
    /// Lazy byte and row boundaries shared with vector-column readers.
    blocks: Arc<Blocks>,
    metric: Metric,
    router: OpenedRouter,
}

impl ClusterIndex {
    /// Write slot `[0]` of the `.centroids` composite for a field. `num_docs`
    /// is the number of distinct docs assigned, not the posting-row total.
    pub(crate) fn serialize_centroids<W: Write + ?Sized>(
        num_centroids: usize,
        num_docs: usize,
        centroid_bytes: &[u8],
        options: &VectorOptions,
        out: &mut W,
    ) -> io::Result<()> {
        let expected = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if centroid_bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IVF centroid byte length",
            ));
        }
        u32::try_from(num_centroids)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32"))?
            .serialize(out)?;
        u32::try_from(num_docs)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "doc count exceeds u32"))?
            .serialize(out)?;
        out.write_all(centroid_bytes)
    }

    /// Write slot `[1]` of the `.centroids` composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Parse a field's `.centroids` slots. Only the count words, the offsets,
    /// and the router topology are materialized; the centroid
    /// rows stay behind a [`FileSlice`] for lazy range reads.
    /// The router is opened as the kind persisted in `router_slice`.
    pub(crate) fn open(
        options: &VectorOptions,
        centroids_slice: FileSlice,
        blocks: Arc<Blocks>,
        router_slice: FileSlice,
    ) -> crate::Result<Self> {
        let count_words = 2 * mem::size_of::<u32>();
        if centroids_slice.len() < count_words {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroids slot is smaller than its count words",
            )
            .into());
        }
        let header = centroids_slice.slice_to(count_words).read_bytes()?;
        let mut reader = header.as_slice();
        let num_centroids = u32::deserialize(&mut reader)? as usize;
        let num_docs = u32::deserialize(&mut reader)? as usize;
        let centroid_len = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        let centroids_slice = centroids_slice.slice_from(count_words);
        if centroids_slice.len() != centroid_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroid byte length mismatch",
            )
            .into());
        }

        if num_centroids != blocks.num_blocks() || !blocks.clustered() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "centroid count differs from vector directory",
            )
            .into());
        }

        let router = RouterKind::open(router_slice, centroids_slice.clone(), options)?;

        let index = ClusterIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            centroid_stride: options.bytes_per_vector(),
            blocks,
            metric: options.metric(),
            router,
        };
        // Every distinct doc owns at least its primary row, so a doc count
        // above the row total means a corrupt file.
        if index.num_docs > index.num_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF doc count exceeds the posting-row total",
            )
            .into());
        }
        Ok(index)
    }

    pub(crate) fn num_clusters(&self) -> usize {
        self.num_centroids
    }

    pub(crate) fn router(&self) -> RouterKind {
        self.router.kind()
    }

    pub(crate) fn router_levels(&self) -> Vec<usize> {
        self.router.level_sizes()
    }

    /// Distinct docs with a vector.
    pub(crate) fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Total posting rows across all clusters.
    pub(crate) fn num_rows(&self) -> usize {
        self.blocks.num_rows()
    }

    pub(crate) fn cluster_range(&self, cluster: usize) -> crate::Result<Range<usize>> {
        self.blocks.row_range(cluster)
    }

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = crate::Result<usize>> + '_ {
        (0..self.num_centroids).map(|cluster| self.blocks.rows_in(cluster))
    }

    /// One centroid row as a file slice.
    pub(crate) fn centroid_row_slice(&self, cluster: usize) -> crate::Result<FileSlice> {
        if cluster >= self.num_centroids {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("centroid {cluster} is out of range"),
            )
            .into());
        }
        let start = cluster * self.centroid_stride;
        Ok(self
            .centroids_slice
            .slice(start..start + self.centroid_stride))
    }

    /// One centroid row, read alone and charged to the running vector stage.
    pub(crate) fn centroid_row(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        Ok(self
            .centroid_row_slice(cluster)?
            .read_vector_bytes(VectorReadKind::Other)?)
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests only. Routing fetches selected list ranges through the lazy arena.
    pub(crate) fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    /// Rank this segment's clusters for `query`, nearest first. `params`
    /// steers the stacked router only (how many clusters the caller will
    /// probe); other routers ignore it.
    pub(crate) fn rank_clusters<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        params: RoutingParams,
    ) -> RouterIter {
        #[cfg(test)]
        if let Some(clusters) = crate::vector::router::test_clusters() {
            return clusters;
        }
        debug_assert_eq!(query.metric(), self.metric);
        self.router.rank(query, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_offsets_validate_framing_and_requested_boundaries() -> crate::Result<()> {
        let encode = |values: &[u64]| -> FileSlice {
            FileSlice::from(
                values
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        for bytes in [vec![], vec![0; 7], vec![0; 9]] {
            assert!(ClusterOffsets::open(FileSlice::from(bytes)).is_err());
        }
        for values in [&[1][..], &[1, 2], &[0, u64::MAX]] {
            assert!(ClusterOffsets::open(encode(values)).is_err());
        }
        let empty = ClusterOffsets::open(encode(&[0]))?;
        assert_eq!(empty.num_clusters(), 0);
        assert_eq!(empty.num_rows(), 0);
        assert!(empty.range(0).is_err());

        let offsets = ClusterOffsets::open(encode(&[0, 0, 3, 3, 8, 8]))?;
        assert_eq!(offsets.num_clusters(), 5);
        assert_eq!(offsets.num_rows(), 8);
        assert_eq!(offsets.range(0)?, 0..0);
        assert_eq!(offsets.range(1)?, 0..3);
        assert_eq!(offsets.range(3)?, 3..8);
        assert_eq!(offsets.range(4)?, 8..8);
        assert!(offsets.range(5).is_err());

        let descending = ClusterOffsets::open(encode(&[0, 3, 2, 8]))?;
        assert!(descending.range(1).is_err());
        let past_end = ClusterOffsets::open(encode(&[0, 9, 8]))?;
        assert!(past_end.range(0).is_err());
        Ok(())
    }

}

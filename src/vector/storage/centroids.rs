//! The `.centroids` file and its reader, [`ClusterIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`ClusterIndex::open`] that parses them
//! back sit side by side.
//!
//! The on-disk file is a 4-byte format-version stamp (see `vector::storage::header`)
//! followed by a [`CompositeFile`](crate::directory::CompositeFile). Written
//! per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `DocLocations`). The composite has two slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride),
//!     rows in the router's canonical cluster-sorted order when a stacked
//!     router was built
//! [2] a router-kind byte followed by the selected router's payload, REQUIRED
//! ```
//!
//! V5 stores block byte offsets and row counts only in `.vec`. Earlier layouts require rebuilding.
use std::io::{self, Write};
use std::mem;

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::router::{OpenedRouter, RouterIter, RouterKind, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::storage::blocks::BlockOffsets;
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::VectorElement;

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Router topology is pinned. Centroid vectors remain lazy in `.centroids`; block
/// offsets and row counts remain lazy in `.vec`. Vector columns and the id-map live on
/// [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub(crate) struct ClusterIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field.
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words).
    centroids_slice: FileSlice,
    /// Bytes per centroid row: the field's vector stride.
    centroid_stride: usize,
    /// The same lazy tuple table used by the vector-column reader.
    block_offsets: BlockOffsets,
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

    /// Parse a field's `.centroids` slots. Count words and router topology are
    /// materialized; centroid rows and block offsets remain lazy.
    /// The router is opened as the kind persisted in `router_slice`.
    pub(crate) fn open(
        options: &VectorOptions,
        centroids_slice: FileSlice,
        block_offsets: BlockOffsets,
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

        if num_centroids != block_offsets.num_blocks() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "centroid count differs from vector block count",
            )
            .into());
        }

        let router = RouterKind::open(router_slice, centroids_slice.clone(), options)?;

        let index = ClusterIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            centroid_stride: options.bytes_per_vector(),
            block_offsets,
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
        self.block_offsets.num_rows()
    }

    pub(crate) fn cluster_num_rows(&self, cluster: usize) -> crate::Result<usize> {
        self.block_offsets.num_rows_in(cluster)
    }

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = crate::Result<usize>> + '_ {
        let mut total = 0usize;
        (0..self.num_centroids).map(move |cluster| {
            let count = self.cluster_num_rows(cluster)?;
            total = total.checked_add(count).ok_or_else(|| {
                crate::error::DataCorruption::comment_only("cluster row total overflow")
            })?;
            if total > self.num_rows()
                || (cluster + 1 == self.num_centroids && total != self.num_rows())
            {
                return Err(crate::error::DataCorruption::comment_only(
                    "block counts differ from total row count",
                )
                .into());
            }
            Ok(count)
        })
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
    use crate::vector::storage::blocks::{
        block_len, pad, write_metadata, BlockOffsetsWriter, Blocks,
    };
    use crate::vector::storage::metadata::VectorColMetadata;

    fn blocks(
        counts: &[usize],
        total: Option<u64>,
    ) -> crate::Result<crate::vector::storage::blocks::BlockOffsets> {
        let options = VectorOptions::new(1, Metric::L2);
        let metadata = VectorColMetadata::build_ivf(&options, None)?;
        let mut data = Vec::new();
        write_metadata(&mut data, &metadata)?;
        let mut offsets = BlockOffsetsWriter::new(data.len() as u64);
        for &count in counts {
            pad(&mut data, block_len(&metadata.slots(), count))?;
            offsets.push(data.len() as u64, count);
        }
        offsets.finish(&mut data)?;
        if let Some(total) = total {
            let at = data.len() - 16;
            data[at..at + 8].copy_from_slice(&total.to_le_bytes());
        }
        Ok(Blocks::open(FileSlice::from(data), &options, true)?.block_offsets())
    }
    #[test]
    fn centroid_count_must_match_vector_blocks() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let mut centers = Vec::new();
        ClusterIndex::serialize_centroids(2, 0, &[0; 8], &options, &mut centers)?;
        for count in [0, 1, 3] {
            let error = ClusterIndex::open(
                &options,
                FileSlice::from(centers.clone()),
                blocks(&vec![0; count], None)?,
                FileSlice::empty(),
            )
            .err()
            .expect("mismatched block count");
            assert!(error
                .to_string()
                .contains("centroid count differs from vector block count"));
        }
        Ok(())
    }
    #[test]
    fn cluster_counts_must_sum_to_the_footer_total() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let mut centers = Vec::new();
        ClusterIndex::serialize_centroids(2, 2, &[0; 8], &options, &mut centers)?;
        let index = ClusterIndex::open(
            &options,
            FileSlice::from(centers),
            blocks(&[1, 1], Some(3))?,
            FileSlice::from(vec![2]),
        )?;
        assert!(index
            .cluster_sizes()
            .collect::<crate::Result<Vec<_>>>()
            .is_err());
        Ok(())
    }
}

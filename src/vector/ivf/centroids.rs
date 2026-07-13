//! The `.centroids` file: IVF cluster routing, written per field.
//!
//! Present only for IVF segments (⟺   the field's `.vec` `IdMap` is `Explicit`).
//! A [`CompositeFile`](crate::directory::CompositeFile) with three slots per
//! field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride)
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] RNG over the centroids (see `Graph::serialize` for the layout)
//! ```
//!
//! One dense `centroid_id = 0..N` indexes all three: `cluster_offsets[c]` is
//! the first row of cluster `c` in the parallel `.vec` rows/`IdMap`, and graph
//! node `c` is centroid `c` (its vector is row `c` of slot `[0]`, which is why
//! the graph slot stores no vectors of its own). Slot `[2]` is optional —
//! absent for degenerate centroid counts, where routing falls back to a linear
//! scan of the centroids.

use std::io::{self, Write};
use std::mem;

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::schema::VectorOptions;

pub(crate) struct CentroidsMeta {
    pub(crate) num_centroids: usize,
    /// Distinct documents with a vector in this field — the segment's logical
    /// vector count, written at merge time. Rows including replicas are
    /// [`Self::num_rows`].
    pub(crate) num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words), deferred:
    /// routing fetches per-node ranges through a
    /// [`FileSliceArena`](crate::vector::FileSliceArena) rather than
    /// materializing `num_centroids × stride` bytes.
    pub(crate) centroids_slice: FileSlice,
    pub(crate) cluster_offsets: OwnedBytes,
}

impl CentroidsMeta {
    /// Write slot `[0]` (num_centroids + num_docs + centroid bytes) of the
    /// `.centroids` composite for a field. `num_docs` is the number of
    /// distinct docs assigned — NOT the posting-row total, which replication
    /// can multiply and which slot `[1]`'s offsets already encode.
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

    /// Write slot `[1]` (cluster offsets prefix sum) of the `.centroids`
    /// composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Parse a field's two `.centroids` slots. Only the count words and the
    /// offsets are materialized; the centroid rows stay behind the returned
    /// [`FileSlice`] for lazy per-node reads.
    pub(crate) fn open(
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        options: &VectorOptions,
    ) -> io::Result<Self> {
        let count_words = 2 * mem::size_of::<u32>();
        if centroids_slice.len() < count_words {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroids slot is smaller than its count words",
            ));
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
            ));
        }

        let cluster_offsets = offsets_slice.read_bytes()?;
        let expected_offsets = (num_centroids + 1)
            .checked_mul(mem::size_of::<u64>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "cluster offset length overflow")
            })?;
        if cluster_offsets.len() != expected_offsets {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF cluster offset byte length mismatch",
            ));
        }
        let meta = Self {
            num_centroids,
            num_docs,
            centroids_slice,
            cluster_offsets,
        };
        // Every distinct doc owns at least its primary row, so a doc count
        // above the row total means a corrupt (or differently-framed) file.
        if meta.num_docs > meta.num_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF doc count exceeds the posting-row total",
            ));
        }
        Ok(meta)
    }

    pub(crate) fn cluster_offset(&self, cluster: usize) -> u64 {
        let start = cluster * mem::size_of::<u64>();
        let end = start + mem::size_of::<u64>();
        u64::from_le_bytes(self.cluster_offsets[start..end].try_into().unwrap())
    }

    /// Total posting rows across all clusters — memberships, counting a
    /// replicated doc once per cell it lives in. Distinct docs are
    /// `self.num_docs`.
    pub(crate) fn num_rows(&self) -> usize {
        self.cluster_offset(self.num_centroids) as usize
    }

    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_centroids).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }
}

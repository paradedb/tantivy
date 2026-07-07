//! The `.centroids` file: IVF cluster routing, written per field.
//!
//! Present only for IVF segments (⟺ the field's `.vec` `IdMap` is `Explicit`).
//! A [`CompositeFile`](crate::directory::CompositeFile) with two slots per
//! field:
//!
//! ```text
//! [0] num_centroids (u32) + centroid_bytes (N · stride)
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! ```
//!
//! One dense `centroid_id = 0..N` indexes both: `cluster_offsets[c]` is the
//! first row of cluster `c` in the parallel `.vec` rows/`IdMap`.

use std::io::{self, Write};
use std::mem;

use common::{BinarySerializable, OwnedBytes};

use crate::directory::FileSlice;
use crate::schema::VectorOptions;

pub(crate) struct CentroidsMeta {
    pub(crate) num_centroids: usize,
    pub(crate) centroid_bytes: OwnedBytes,
    pub(crate) cluster_offsets: OwnedBytes,
}

impl CentroidsMeta {
    /// Write slot `[0]` (num_centroids + centroid bytes) of the `.centroids`
    /// composite for a field.
    pub(crate) fn serialize_centroids<W: Write + ?Sized>(
        num_centroids: usize,
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

    /// Parse a field's two `.centroids` slots.
    pub(crate) fn open(
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        options: &VectorOptions,
    ) -> io::Result<Self> {
        let centroids_bytes = centroids_slice.read_bytes()?;
        let mut reader = centroids_bytes.as_slice();
        let num_centroids = u32::deserialize(&mut reader)? as usize;
        let centroid_len = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if reader.len() != centroid_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroid byte length mismatch",
            ));
        }
        let centroid_start = centroids_bytes.len() - reader.len();
        let centroid_bytes = centroids_bytes.slice(centroid_start..centroids_bytes.len());

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
        Ok(Self {
            num_centroids,
            centroid_bytes,
            cluster_offsets,
        })
    }

    pub(crate) fn cluster_offset(&self, cluster: usize) -> u64 {
        let start = cluster * mem::size_of::<u64>();
        let end = start + mem::size_of::<u64>();
        u64::from_le_bytes(self.cluster_offsets[start..end].try_into().unwrap())
    }

    pub(crate) fn num_vectors(&self) -> usize {
        self.cluster_offset(self.num_centroids) as usize
    }
}

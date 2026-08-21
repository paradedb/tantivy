//! Format version for the vector files: the per-segment `.vec` and the
//! index-level `centroids.<version>` set file.
//!
//! A fixed 4-byte header (a `u32` version) is prepended to every file, ahead of
//! the [`CompositeFile`](crate::directory::CompositeFile) body. The version is
//! the wire-layout *generation* — bump it when the framing changes incompatibly.

use std::io::{self, Read, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::FileSlice;

/// Length of the version header in bytes (a single `u32`).
pub(crate) const HEADER_LEN: usize = 4;

/// On-disk format version of a vector file (`.vec` or `centroids.<version>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VectorFileVersion {
    V1 = 1,
    /// The retired `.centroids` sidecar carried per-cluster centroid bounds
    /// (slot `[3]`) as a REQUIRED slot.
    V2 = 2,
    /// Centroids are an index-level artifact: the `centroids.<version>` set
    /// file holds the centroid rows and the routing structure once per
    /// index, and every segment assigns against it. `.vec` becomes the
    /// single per-segment file — cluster-sorted rows plus the per-segment
    /// remainder (offsets, bounds, IVF meta) — and the per-segment
    /// `.centroids` sidecar and the flat (doc-ordered) layout are gone.
    /// A pre-V3 `.vec` is rejected at open with a REINDEX message.
    V3 = 3,
}

/// `.vec` composite slot indices. Slots `[2..=4]` exist exactly when the
/// field has vector rows in the segment (there is no flat layout from
/// [`VectorFileVersion::V3`] on); a field with no vectors owns no slots at
/// all, and a partial slot set is corrupt.
pub(crate) mod vec_slot {
    /// Row→doc-id permutation (`IdMap::Explicit`), cluster-sorted, parallel
    /// to [`ROWS`].
    pub(crate) const ID_MAP: usize = 0;
    /// The stored vector rows, cluster-sorted.
    pub(crate) const ROWS: usize = 1;
    /// Per-cluster posting offsets: `u64[C+1]` prefix sum over the rows.
    pub(crate) const OFFSETS: usize = 2;
    /// Per-cluster centroid bounds: a segment-level kind byte, then the
    /// per-cluster payload folded over this segment's NATIVE rows.
    pub(crate) const BOUNDS: usize = 3;
    /// Per-segment IVF metadata: distinct doc count, centroid count, and
    /// the centroid-set version this segment assigned against.
    pub(crate) const IVF_META: usize = 4;
}

/// `centroids.<version>` composite slot indices (the index-level set file).
/// The body is prefixed by the set's `u64` version, after the format header
/// and before the composite.
pub(crate) mod centroid_index_slot {
    /// `num_centroids: u32` + the centroid rows (normalized at creation).
    pub(crate) const CENTROIDS: usize = 0;
    /// The routing structure over the centroids. OPTIONAL: absent for
    /// degenerate centroid counts (C <= 1). First byte is a router-kind
    /// tag — 0 is tantivy's RNG, values >= 128 are consumer-defined.
    pub(crate) const ROUTER: usize = 1;
}

/// Version stamped into newly written vector files.
pub(crate) const CURRENT: VectorFileVersion = VectorFileVersion::V3;

impl BinarySerializable for VectorFileVersion {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u32).serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        match u32::deserialize(reader)? {
            1 => Ok(VectorFileVersion::V1),
            2 => Ok(VectorFileVersion::V2),
            3 => Ok(VectorFileVersion::V3),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported vector file format version: {other}"),
            )),
        }
    }
}

/// Write the current version header. Call before wrapping the writer in a
/// [`CompositeWrite`](crate::directory::CompositeWrite); the composite's
/// offsets are self-relative, so the header does not perturb them.
pub(crate) fn write_header<W: Write + ?Sized>(writer: &mut W) -> io::Result<()> {
    CURRENT.serialize(writer)
}

/// Parse the version header and return it alongside the composite body (the
/// file slice past the header). Errors if the version is unknown or newer than
/// [`CURRENT`].
pub(crate) fn read_header(file: &FileSlice) -> io::Result<(VectorFileVersion, FileSlice)> {
    if file.len() < HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "vector file is smaller than its header",
        ));
    }
    let header_bytes = file.slice_to(HEADER_LEN).read_bytes()?;
    let version = VectorFileVersion::deserialize(&mut header_bytes.as_slice())?;
    Ok((version, file.slice_from(HEADER_LEN)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_round_trip() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_LEN);
        assert_eq!(buf, vec![3, 0, 0, 0]);

        let (version, body) = read_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V3);
        assert_eq!(body.len(), 0);
    }

    #[test]
    fn test_header_preserves_body() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        buf.extend_from_slice(b"composite-bytes");

        let (version, body) = read_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V3);
        assert_eq!(body.read_bytes().unwrap().as_slice(), b"composite-bytes");
    }

    /// A prior generation still PARSES here — the header module knows
    /// versions, not policy. Rejecting a pre-V3 `.vec` is the vector
    /// reader's job, where the REINDEX hint can be phrased.
    #[test]
    fn test_prior_version_parses() {
        for (raw, expected) in [(1u32, VectorFileVersion::V1), (2, VectorFileVersion::V2)] {
            let buf = raw.to_le_bytes().to_vec();
            let (version, _) = read_header(&FileSlice::from(buf)).unwrap();
            assert_eq!(version, expected);
        }
    }

    #[test]
    fn test_future_version_rejected() {
        let buf = 4u32.to_le_bytes().to_vec();
        let err = read_header(&FileSlice::from(buf)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_truncated_header_rejected() {
        let buf = vec![3u8, 0];
        let err = read_header(&FileSlice::from(buf)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}

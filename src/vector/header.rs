//! Format version for the per-segment vector files (`.vec` and `.centroids`).
//!
//! A fixed 4-byte header (a `u32` version) is prepended to every file, ahead of
//! the [`CompositeFile`](crate::directory::CompositeFile) body. The version is
//! the wire-layout *generation* — bump it when the framing changes incompatibly.
//!
//! For `.vec`, the version is orthogonal to the
//! [`IdMap`](super::flat::id_map) variant, which selects the storage *mode*
//! (flat vs IVF) within a generation. For `.centroids`, it versions the IVF
//! routing composite (centroids, cluster offsets, optional graph, required
//! bounds).

use std::io::{self, Read, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::FileSlice;

/// Length of the version header in bytes (a single `u32`).
pub(crate) const HEADER_LEN: usize = 4;

/// On-disk format version of a vector segment file (`.vec` or `.centroids`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VectorFileVersion {
    V1 = 1,
    /// `.centroids` carries per-cluster centroid bounds (slot `[3]`) as a
    /// REQUIRED slot. A V1 `.centroids` is rejected at open with a REINDEX
    /// message: the bounds gate certifies skips against the slot, and a
    /// silently absent bound is indistinguishable from a zero one. `.vec`
    /// is unaffected by the change and V1 `.vec` files stay readable —
    /// flat segments have no clusters and no bounds.
    V2 = 2,
}

/// `.centroids` composite slot indices. Slot `[2]` is OPTIONAL and slot
/// `[3]` is MANDATORY from [`VectorFileVersion::V2`] on; crossing them
/// turns a missing-because-degenerate graph into a corrupt-file error, or
/// a missing bound into a silently absent one. They live beside the
/// version because that is what decides which of them must be present.
pub(crate) mod centroid_slot {
    /// The centroid rows themselves.
    pub(crate) const CENTROIDS: usize = 0;
    /// Per-cluster posting offsets.
    pub(crate) const OFFSETS: usize = 1;
    /// The routing graph. OPTIONAL: the write side skips it for
    /// degenerate centroid counts, and its absence is normal.
    pub(crate) const GRAPH: usize = 2;
    /// Per-cluster centroid bounds. MANDATORY from V2 on: a V2 file
    /// without this slot is corrupt, not old.
    pub(crate) const BOUNDS: usize = 3;
    /// Per-posting-row residual norms (`||x - c||` against the row's own
    /// cluster centroid), in row order. OPTIONAL: absence disables the
    /// row-level triangle-inequality gate, which is fail-open by
    /// construction — a missing radius only costs pruning, never
    /// correctness.
    pub(crate) const RADII: usize = 4;
}

/// `.vec` composite slot indices. A different file with a different
/// layout from `.centroids` — naming both is what stops one file's slot
/// number being read against the other's meaning.
pub(crate) mod vec_slot {
    /// Dense row-id to doc-id map.
    pub(crate) const ID_MAP: usize = 0;
    /// The stored vector rows.
    pub(crate) const ROWS: usize = 1;
}

/// Version stamped into newly written vector files.
pub(crate) const CURRENT: VectorFileVersion = VectorFileVersion::V2;

impl BinarySerializable for VectorFileVersion {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u32).serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        match u32::deserialize(reader)? {
            1 => Ok(VectorFileVersion::V1),
            2 => Ok(VectorFileVersion::V2),
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
        assert_eq!(buf, vec![2, 0, 0, 0]);

        let (version, body) = read_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V2);
        assert_eq!(body.len(), 0);
    }

    #[test]
    fn test_header_preserves_body() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        buf.extend_from_slice(b"composite-bytes");

        let (version, body) = read_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V2);
        assert_eq!(body.read_bytes().unwrap().as_slice(), b"composite-bytes");
    }

    /// A prior generation still PARSES here — the header module knows
    /// versions, not policy. Rejecting a V1 `.centroids` is the vector
    /// reader's job, where the REINDEX hint can be phrased.
    #[test]
    fn test_prior_version_parses() {
        let buf = 1u32.to_le_bytes().to_vec();
        let (version, _) = read_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V1);
    }

    #[test]
    fn test_future_version_rejected() {
        let buf = 3u32.to_le_bytes().to_vec();
        let err = read_header(&FileSlice::from(buf)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_truncated_header_rejected() {
        let buf = vec![2u8, 0];
        let err = read_header(&FileSlice::from(buf)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}

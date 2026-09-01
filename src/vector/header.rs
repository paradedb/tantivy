//! Format version for the per-segment vector files (`.vec` and `.centroids`).
//!
//! A fixed 4-byte header (a `u32` version) is prepended to every file, ahead of
//! the [`CompositeFile`](crate::directory::CompositeFile) body. The version is
//! the wire-layout *generation* — bump it when the framing changes incompatibly.
//!
//! For `.vec`, the version is orthogonal to the
//! [`IdMap`](super::flat::id_map) variant, which selects the storage *mode*
//! (flat vs IVF) within a generation. For `.centroids`, it versions the IVF
//! routing composite (centroids, cluster offsets, optional router, required
//! bounds).

use std::io::{self, Read, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::FileSlice;

/// Length of the version header in bytes (a single `u32`).
pub(crate) const HEADER_LEN: usize = 4;

/// On-disk format version of a vector segment file (`.vec` or `.centroids`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VectorFileVersion {
    V1 = 1,
    /// `.centroids` carries per-cluster centroid bounds (slot `[3]`) as a
    /// REQUIRED slot: the bounds gate certifies skips against it, and a
    /// silently absent bound is indistinguishable from a zero one — so a V2
    /// file missing the slot is corrupt. A V1 `.centroids` (which shipped,
    /// and legitimately predates the slot) still opens: its clusters get
    /// SATURATED bounds (`f32::INFINITY`, always probe), correct but
    /// unpruned until the next merge rewrites the segment. `.vec` is
    /// unaffected by the change and V1 `.vec` files stay readable — flat
    /// segments have no clusters and no bounds.
    V2 = 2,
    /// `.centroids` slot `[2]` carries a self-describing router envelope.
    /// V2 files keep the bare graph layout in that slot.
    V3 = 3,
}

/// `.centroids` composite slot indices for the V3 layout.
pub(crate) mod centroid_slot {
    /// The centroid rows themselves.
    pub(crate) const CENTROIDS: usize = 0;
    /// Per-cluster posting offsets.
    pub(crate) const OFFSETS: usize = 1;
    /// The self-describing router payload.
    pub(crate) const ROUTER: usize = 2;
    /// Per-cluster centroid bounds.
    pub(crate) const BOUNDS: usize = 3;
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
        let buf = 4u32.to_le_bytes().to_vec();
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

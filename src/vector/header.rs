//! Header and entry assignments for per-segment vector files.

use std::io::{self, Read, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::FileSlice;

/// Length of the version header in bytes.
pub(crate) const HEADER_LEN: usize = 4;

/// On-disk vector file version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VectorFileVersion {
    V1 = 1,
    /// `.centroids` includes required per-cluster bounds.
    V2 = 2,
    /// `.centroids` includes a tagged router and `.vec` includes quantized slots.
    V3 = 3,
    /// Block-major vector columns with per-field metadata.
    V4 = 4,
}

impl BinarySerializable for VectorFileVersion {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u32).serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        match u32::deserialize(reader)? {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            4 => Ok(Self::V4),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported vector file format version: {other}"),
            )),
        }
    }
}

/// Format identifier written to `.vec` files.
pub(crate) const VECTOR_FILE_FORMAT_VERSION: u32 = VectorFileVersion::V4 as u32;
/// Version written to `.vec` files.
pub(crate) const CURRENT_VECTOR: VectorFileVersion = VectorFileVersion::V4;
/// Version written to `.centroids` files.
pub(crate) const CURRENT_CENTROID: VectorFileVersion = VectorFileVersion::V3;

/// `.centroids` composite slot indices.
pub(crate) mod centroid_slot {
    /// Centroid vectors.
    pub(crate) const CENTROIDS: usize = 0;
    /// Per-cluster posting offsets.
    pub(crate) const OFFSETS: usize = 1;
    /// Router kind and payload (V3).
    pub(crate) const ROUTER: usize = 2;
    /// Per-cluster centroid bounds.
    pub(crate) const BOUNDS: usize = 3;
}

/// Slots in a centroid composite file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum CentroidSlot {
    /// Centroid vectors.
    Centroids = centroid_slot::CENTROIDS,
    /// Posting offsets.
    Offsets = centroid_slot::OFFSETS,
    /// Routing payload.
    Router = centroid_slot::ROUTER,
    /// Cluster bounds.
    Bounds = centroid_slot::BOUNDS,
}

impl CentroidSlot {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

/// Composite entries of every vector field. Column slots live inside Data blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum VectorEntry {
    /// Lazy doc-to-location map for clustered fields; identity or bitmap for flat fields.
    IdMap = 0,
    /// Stored metadata and block columns.
    Data = 1,
}
impl VectorEntry {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}
/// Accepted vector grammars; all other versions require rebuilding.
pub(crate) const SUPPORTED_VECTOR: &[VectorFileVersion] = &[VectorFileVersion::V4];

fn write_header<W: Write + ?Sized>(writer: &mut W, version: VectorFileVersion) -> io::Result<()> {
    version.serialize(writer)
}

fn parse_header(file: &FileSlice, file_kind: &str) -> io::Result<(VectorFileVersion, FileSlice)> {
    if file.len() < HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{file_kind} file is smaller than its header"),
        ));
    }
    let header_bytes = file.slice_to(HEADER_LEN).read_bytes()?;
    let version = VectorFileVersion::deserialize(&mut header_bytes.as_slice())?;
    Ok((version, file.slice_from(HEADER_LEN)))
}

/// Writes a `.vec` header.
pub(crate) fn write_vector_header<W: Write + ?Sized>(writer: &mut W) -> io::Result<()> {
    debug_assert_eq!(CURRENT_VECTOR as u32, VECTOR_FILE_FORMAT_VERSION);
    write_header(writer, CURRENT_VECTOR)
}

/// Validates a `.vec` header and returns its version and composite body.
pub(crate) fn read_vector_header(
    file: &FileSlice,
) -> crate::Result<(VectorFileVersion, FileSlice)> {
    if file.len() < HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "vector file is smaller than its header",
        )
        .into());
    }
    let bytes = file.slice_to(HEADER_LEN).read_bytes()?;
    let index_version = u32::deserialize(&mut bytes.as_slice())?;
    let version = SUPPORTED_VECTOR
        .iter()
        .copied()
        .find(|version| *version as u32 == index_version)
        .ok_or(crate::TantivyError::IncompatibleIndex(
            crate::directory::error::Incompatibility::VectorFormatMismatch {
                index_version,
                supported_version: VECTOR_FILE_FORMAT_VERSION,
            },
        ))?;
    Ok((version, file.slice_from(HEADER_LEN)))
}

/// Writes a `.centroids` header.
pub(crate) fn write_centroid_header<W: Write + ?Sized>(writer: &mut W) -> io::Result<()> {
    write_header(writer, CURRENT_CENTROID)
}

/// Parses a `.centroids` header and returns its version and composite body.
pub(crate) fn read_centroid_header(file: &FileSlice) -> io::Result<(VectorFileVersion, FileSlice)> {
    parse_header(file, "centroid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_header_round_trip() {
        let mut buf = Vec::new();
        write_vector_header(&mut buf).unwrap();
        assert_eq!(buf, [4, 0, 0, 0]);

        let (version, body) = read_vector_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V4);
        assert_eq!(body.len(), 0);
    }

    #[test]
    fn vector_header_preserves_body() {
        let mut buf = Vec::new();
        write_vector_header(&mut buf).unwrap();
        buf.extend_from_slice(b"composite-bytes");

        let (_, body) = read_vector_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(body.read_bytes().unwrap().as_slice(), b"composite-bytes");
    }

    #[test]
    fn vector_headers_before_v4_require_rebuild() {
        for version in [
            VectorFileVersion::V1,
            VectorFileVersion::V2,
            VectorFileVersion::V3,
        ] {
            let mut buf = Vec::new();
            version.serialize(&mut buf).unwrap();
            let error = read_vector_header(&FileSlice::from(buf)).unwrap_err();
            assert!(error.to_string().contains("rebuild required"));
        }
    }

    #[test]
    fn truncated_vector_header_is_rejected() {
        let error = read_vector_header(&FileSlice::from(vec![2u8, 0])).unwrap_err();
        assert!(
            matches!(error, crate::TantivyError::IoError(ref error) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
    }
}

//! Header and slot assignments for per-segment vector files.

use std::io::{self, Read, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::FileSlice;

/// Length of the version header in bytes.
pub(crate) const HEADER_LEN: usize = 4;

/// On-disk vector file version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VectorFileVersion {
    V1 = 1,
    /// `.centroids` includes required per-cluster bounds.
    V2 = 2,
    /// `.centroids` includes a tagged router and `.vec` includes quantized slots.
    V3 = 3,
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
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported vector file format version: {other}"),
            )),
        }
    }
}

/// Format identifier written to `.vec` files.
pub(crate) const VECTOR_FILE_FORMAT_VERSION: u32 = VectorFileVersion::V3 as u32;
/// Version written to `.vec` files.
pub(crate) const CURRENT_VECTOR: VectorFileVersion = VectorFileVersion::V3;
/// Version written to `.centroids` files.
pub(crate) const CURRENT_CENTROID: VectorFileVersion = VectorFileVersion::V3;
/// Version used for centroid router serialization.
pub(crate) const CURRENT: VectorFileVersion = CURRENT_CENTROID;

/// `.centroids` composite slot indices.
pub(crate) mod centroid_slot {
    /// Centroid vectors.
    pub(crate) const CENTROIDS: usize = 0;
    /// Per-cluster posting offsets.
    pub(crate) const OFFSETS: usize = 1;
    /// Tagged router at V3 and bare graph at V2.
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

/// Slots in a vector composite file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum VectorSlot {
    /// Row-to-document map.
    IdMap = 0,
    /// Full-precision vectors.
    Rows = 1,
    /// Residual squared norms.
    ResidualNorms = 2,
    /// Layer-0 packed codes.
    Layer0Codes = 3,
    /// Layer-0 scale, gamma, and error sidecar.
    Layer0Sidecar = 4,
    /// Layer-0 L2 constants.
    Layer0Constants = 5,
    /// Layer-1 packed codes.
    Layer1Codes = 6,
    /// Layer-1 scale, gamma, and error sidecar.
    Layer1Sidecar = 7,
    /// Layer-1 L2 constants.
    Layer1Constants = 8,
    /// Layer-2 packed codes.
    Layer2Codes = 9,
    /// Layer-2 scale, gamma, and error sidecar.
    Layer2Sidecar = 10,
    /// Layer-2 L2 constants.
    Layer2Constants = 11,
}

impl VectorSlot {
    pub(crate) const COUNT: usize = 12;

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) const fn codes(layer: usize) -> Self {
        match layer {
            0 => Self::Layer0Codes,
            1 => Self::Layer1Codes,
            2 => Self::Layer2Codes,
            _ => panic!("vector quantization supports at most three layers"),
        }
    }

    pub(crate) const fn sidecar(layer: usize) -> Self {
        match layer {
            0 => Self::Layer0Sidecar,
            1 => Self::Layer1Sidecar,
            2 => Self::Layer2Sidecar,
            _ => panic!("vector quantization supports at most three layers"),
        }
    }

    pub(crate) const fn constants(layer: usize) -> Self {
        match layer {
            0 => Self::Layer0Constants,
            1 => Self::Layer1Constants,
            2 => Self::Layer2Constants,
            _ => panic!("vector quantization supports at most three layers"),
        }
    }
}

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
    write_header(writer, CURRENT_VECTOR)
}

/// Validates a `.vec` header and returns its version and composite body.
pub(crate) fn read_vector_header(file: &FileSlice) -> io::Result<(VectorFileVersion, FileSlice)> {
    let (version, body) = parse_header(file, "vector")?;
    if version != CURRENT_VECTOR {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "vector file format version {} is unsupported; rebuild required",
                version as u32
            ),
        ));
    }
    Ok((version, body))
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
        assert_eq!(buf, [3, 0, 0, 0]);

        let (version, body) = read_vector_header(&FileSlice::from(buf)).unwrap();
        assert_eq!(version, VectorFileVersion::V3);
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
    fn vector_headers_before_v3_require_rebuild() {
        for version in [VectorFileVersion::V1, VectorFileVersion::V2] {
            let mut buf = Vec::new();
            version.serialize(&mut buf).unwrap();
            let error = read_vector_header(&FileSlice::from(buf)).unwrap_err();
            assert!(error.to_string().contains("rebuild required"));
        }
    }

    #[test]
    fn truncated_vector_header_is_rejected() {
        let error = read_vector_header(&FileSlice::from(vec![2u8, 0])).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn quantized_slots_are_layer_separated() {
        assert_eq!(VectorSlot::IdMap.index(), 0);
        assert_eq!(VectorSlot::Rows.index(), 1);
        assert_eq!(VectorSlot::ResidualNorms.index(), 2);
        assert_eq!(VectorSlot::codes(0).index(), 3);
        assert_eq!(VectorSlot::sidecar(0).index(), 4);
        assert_eq!(VectorSlot::constants(0).index(), 5);
        assert_eq!(VectorSlot::codes(1).index(), 6);
        assert_eq!(VectorSlot::sidecar(1).index(), 7);
        assert_eq!(VectorSlot::constants(1).index(), 8);
        assert_eq!(VectorSlot::codes(2).index(), 9);
        assert_eq!(VectorSlot::sidecar(2).index(), 10);
        assert_eq!(VectorSlot::constants(2).index(), 11);
        assert_eq!(VectorSlot::COUNT, 12);
    }
}

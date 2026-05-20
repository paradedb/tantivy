use std::io::{self, Write};
use std::mem;

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::vector::VectorOptions;

pub(crate) const VECMETA_EXT: &str = "vecmeta";

const FORMAT_FLAT: u8 = 0;
const FORMAT_IVF: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VectorStorageFormat {
    Flat,
    Ivf,
}

impl VectorStorageFormat {
    pub(crate) fn serialize<W: Write + ?Sized>(self, writer: &mut W) -> io::Result<()> {
        let code = match self {
            Self::Flat => FORMAT_FLAT,
            Self::Ivf => FORMAT_IVF,
        };
        writer.write_all(&[code])
    }

    fn from_code(code: u8) -> io::Result<Self> {
        match code {
            FORMAT_FLAT => Ok(Self::Flat),
            FORMAT_IVF => Ok(Self::Ivf),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported vector storage format",
            )),
        }
    }
}

pub(crate) struct VectorSegmentMeta {
    pub(crate) format: VectorStorageFormat,
    pub(crate) payload: FileSlice,
}

impl VectorSegmentMeta {
    pub(crate) fn open(file_slice: FileSlice) -> io::Result<Self> {
        if file_slice.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vector metadata is empty",
            ));
        }
        let format = VectorStorageFormat::from_code(file_slice.slice(0..1).read_bytes()?[0])?;
        let payload = file_slice.slice_from(1);
        if format == VectorStorageFormat::Flat && payload.len() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector metadata has trailing bytes",
            ));
        }
        Ok(Self { format, payload })
    }
}

pub(crate) struct IvfFieldMeta {
    pub(crate) num_centroids: usize,
    pub(crate) centroid_bytes: OwnedBytes,
    pub(crate) cluster_offsets: OwnedBytes,
}

impl IvfFieldMeta {
    pub(crate) fn serialize<W: Write + ?Sized>(
        &self,
        writer: &mut W,
        options: &VectorOptions,
    ) -> io::Result<()> {
        let expected_centroid_bytes = self
            .num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if self.centroid_bytes.len() != expected_centroid_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IVF centroid byte length",
            ));
        }
        let expected_cluster_offsets = (self.num_centroids + 1)
            .checked_mul(mem::size_of::<u64>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "cluster offset length overflow")
            })?;
        if self.cluster_offsets.len() != expected_cluster_offsets {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IVF cluster offset byte length",
            ));
        }
        u32::try_from(self.num_centroids)
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32")
            })?
            .serialize(writer)?;
        writer.write_all(self.centroid_bytes.as_slice())?;
        writer.write_all(self.cluster_offsets.as_slice())?;
        Ok(())
    }

    pub(crate) fn open(file_slice: FileSlice, options: &VectorOptions) -> io::Result<Self> {
        let bytes = file_slice.read_bytes()?;
        let mut reader = bytes.as_slice();
        let num_centroids = u32::deserialize(&mut reader)? as usize;
        let centroid_len = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if reader.len() < centroid_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing IVF centroid bytes",
            ));
        }
        let centroid_start = bytes.len() - reader.len();
        let centroid_end = centroid_start + centroid_len;
        let centroid_bytes = bytes.slice(centroid_start..centroid_end);
        reader = &reader[centroid_len..];
        let cluster_offsets_len = (num_centroids + 1)
            .checked_mul(mem::size_of::<u64>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "cluster offset length overflow")
            })?;
        if reader.len() < cluster_offsets_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing IVF cluster offsets",
            ));
        }
        let cluster_offsets_start = bytes.len() - reader.len();
        let cluster_offsets_end = cluster_offsets_start + cluster_offsets_len;
        let cluster_offsets = bytes.slice(cluster_offsets_start..cluster_offsets_end);
        reader = &reader[cluster_offsets_len..];
        if !reader.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing IVF metadata bytes",
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

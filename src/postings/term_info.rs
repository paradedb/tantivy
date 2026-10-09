use std::io;
use std::ops::Range;

use common::{BinarySerializable, FixedSize};

/// On-disk term metadata format, independent of the dictionary backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TermInfoVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl TermInfoVersion {
    pub(crate) const fn serialized_size(self) -> usize {
        let base = 3 * u32::SIZE_IN_BYTES + 2 * u64::SIZE_IN_BYTES;
        match self {
            Self::V1 => base,
            Self::V2 => base + u64::SIZE_IN_BYTES,
            Self::V3 => base + 2 * u64::SIZE_IN_BYTES,
        }
    }
}

impl BinarySerializable for TermInfoVersion {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u32).serialize(writer)
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        match u32::deserialize(reader)? {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            version => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported term metadata version {version}"),
            )),
        }
    }
}

/// `TermInfo` wraps the metadata associated with a Term.
/// It is segment-local.
#[derive(Debug, Default, Eq, PartialEq, Clone)]
pub struct TermInfo {
    /// Number of documents in the segment containing the term
    pub doc_freq: u32,
    /// Byte range of the posting list within the postings (`.idx`) file.
    pub postings_range: Range<usize>,
    /// Byte range of the positions of this terms in the positions (`.pos`) file.
    pub positions_range: Range<usize>,
    /// Byte offset of this term's norms in the field's `.pnorm` data, when enabled.
    pub pnorms_offset: Option<u64>,
    /// Byte offset of the optional membership bitmap in the field's `.bmap` data.
    pub bitmap_offset: Option<u64>,
}

impl TermInfo {
    pub(crate) fn posting_num_bytes(&self) -> u32 {
        let num_bytes = self.postings_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }

    pub(crate) fn positions_num_bytes(&self) -> u32 {
        let num_bytes = self.positions_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }
}

impl FixedSize for TermInfo {
    /// Size required for the binary serialization of a `TermInfo` object.
    /// This is large, but in practise, `TermInfo` are encoded in blocks and
    /// only the first `TermInfo` of a block is serialized uncompressed.
    /// The subsequent `TermInfo` are delta encoded and bitpacked.
    const SIZE_IN_BYTES: usize = TermInfoVersion::V3.serialized_size();
}

impl TermInfo {
    pub(crate) fn serialize_versioned<W: io::Write + ?Sized>(
        &self,
        writer: &mut W,
        version: TermInfoVersion,
    ) -> io::Result<()> {
        self.doc_freq.serialize(writer)?;
        (self.postings_range.start as u64).serialize(writer)?;
        self.posting_num_bytes().serialize(writer)?;
        (self.positions_range.start as u64).serialize(writer)?;
        self.positions_num_bytes().serialize(writer)?;
        match version {
            TermInfoVersion::V1 => Ok(()),
            TermInfoVersion::V2 => self.pnorms_offset.unwrap_or(u64::MAX).serialize(writer),
            TermInfoVersion::V3 => {
                self.pnorms_offset.unwrap_or(u64::MAX).serialize(writer)?;
                self.bitmap_offset.unwrap_or(u64::MAX).serialize(writer)
            }
        }
    }

    pub(crate) fn deserialize_versioned<R: io::Read>(
        reader: &mut R,
        version: TermInfoVersion,
    ) -> io::Result<Self> {
        let doc_freq = u32::deserialize(reader)?;
        let postings_start_offset = u64::deserialize(reader)? as usize;
        let postings_num_bytes = u32::deserialize(reader)? as usize;
        let postings_end_offset = postings_start_offset + postings_num_bytes;
        let positions_start_offset = u64::deserialize(reader)? as usize;
        let positions_num_bytes = u32::deserialize(reader)? as usize;
        let positions_end_offset = positions_start_offset + positions_num_bytes;
        let pnorms_offset = match version {
            TermInfoVersion::V1 => None,
            TermInfoVersion::V2 | TermInfoVersion::V3 => {
                let offset = u64::deserialize(reader)?;
                (offset != u64::MAX).then_some(offset)
            }
        };
        let bitmap_offset = if version == TermInfoVersion::V3 {
            let offset = u64::deserialize(reader)?;
            (offset != u64::MAX).then_some(offset)
        } else {
            None
        };
        Ok(TermInfo {
            doc_freq,
            postings_range: postings_start_offset..postings_end_offset,
            positions_range: positions_start_offset..positions_end_offset,
            pnorms_offset,
            bitmap_offset,
        })
    }
}

impl BinarySerializable for TermInfo {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.serialize_versioned(writer, TermInfoVersion::V3)
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Self::deserialize_versioned(reader, TermInfoVersion::V3)
    }
}

#[cfg(test)]
mod tests {

    use super::{TermInfo, TermInfoVersion};
    use crate::tests::fixed_size_test;

    #[test]
    fn versioned_roundtrip() -> std::io::Result<()> {
        for version in [
            TermInfoVersion::V1,
            TermInfoVersion::V2,
            TermInfoVersion::V3,
        ] {
            for pnorms_offset in [None, Some(0), Some(1 << 40)] {
                let mut expected = TermInfo {
                    doc_freq: 3,
                    postings_range: 10..20,
                    positions_range: 30..40,
                    pnorms_offset,
                    bitmap_offset: Some(1 << 40),
                };
                let mut bytes = Vec::new();
                expected.serialize_versioned(&mut bytes, version)?;
                assert_eq!(bytes.len(), version.serialized_size());
                if version == TermInfoVersion::V1 {
                    expected.pnorms_offset = None;
                }
                if version != TermInfoVersion::V3 {
                    expected.bitmap_offset = None;
                }
                let mut bytes = bytes.as_slice();
                assert_eq!(
                    TermInfo::deserialize_versioned(&mut bytes, version)?,
                    expected
                );
                assert!(bytes.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn test_fixed_size() {
        fixed_size_test::<TermInfo>();
    }
}

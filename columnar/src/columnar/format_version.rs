use core::fmt;
use std::fmt::{Display, Formatter};
use std::io;
use std::io::Write;

use common::HasLen;
use common::file_slice::FileSlice;

use crate::InvalidData;

pub const VERSION_FOOTER_NUM_BYTES: usize = MAGIC_BYTES.len() + std::mem::size_of::<u32>();

/// We end the file by these 4 bytes just to somewhat identify that
/// this is indeed a columnar file.
const MAGIC_BYTES: [u8; 4] = [2, 113, 119, 66];

pub fn footer() -> [u8; VERSION_FOOTER_NUM_BYTES] {
    let mut footer_bytes = [0u8; VERSION_FOOTER_NUM_BYTES];
    footer_bytes[0..4].copy_from_slice(&CURRENT_VERSION.to_bytes());
    footer_bytes[4..8].copy_from_slice(&MAGIC_BYTES[..]);
    footer_bytes
}

pub fn parse_footer(footer_bytes: [u8; VERSION_FOOTER_NUM_BYTES]) -> Result<Version, InvalidData> {
    if footer_bytes[4..8] != MAGIC_BYTES {
        return Err(InvalidData);
    }
    Version::try_from_bytes(footer_bytes[0..4].try_into().unwrap())
}

pub const CURRENT_VERSION: Version = Version::V3;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum Version {
    V1 = 1u32,
    V2 = 2u32,
    V3 = 3u32,
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Version::V1 => write!(f, "v1"),
            Version::V2 => write!(f, "v2"),
            Version::V3 => write!(f, "v3"),
        }
    }
}

impl Version {
    fn to_bytes(self) -> [u8; 4] {
        (self as u32).to_le_bytes()
    }

    fn try_from_bytes(bytes: [u8; 4]) -> Result<Version, InvalidData> {
        let code = u32::from_le_bytes(bytes);
        match code {
            1u32 => Ok(Version::V1),
            2u32 => Ok(Version::V2),
            3u32 => Ok(Version::V3),
            _ => Err(InvalidData),
        }
    }

    /// Width of the byte length written after a column index, a multivalued index's optional
    /// index and a dictionary. A merge can grow them past 4 GiB, so V3 uses `u64`, V1 and V2 `u32`.
    pub(crate) fn len_trailer_num_bytes(self) -> usize {
        match self {
            Version::V1 | Version::V2 => std::mem::size_of::<u32>(),
            Version::V3 => std::mem::size_of::<u64>(),
        }
    }

    /// Appends `len` in the trailer width of this version.
    pub(crate) fn write_len_trailer(self, len: u64, output: &mut impl Write) -> io::Result<()> {
        match self {
            Version::V1 | Version::V2 => {
                let len = u32::try_from(len).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{len} bytes do not fit the u32 length of columnar format {self}"),
                    )
                })?;
                output.write_all(&len.to_le_bytes())
            }
            Version::V3 => output.write_all(&len.to_le_bytes()),
        }
    }

    /// Splits the trailer off `file_slice` and returns the body with the length it holds. The
    /// length covers a prefix of the body, so a larger value is corrupt.
    pub(crate) fn split_len_trailer(self, file_slice: FileSlice) -> io::Result<(FileSlice, usize)> {
        let (body, trailer) = file_slice.split_from_end(self.len_trailer_num_bytes());
        let trailer = trailer.read_bytes()?;
        let len = match self {
            Version::V1 | Version::V2 => {
                u64::from(u32::from_le_bytes(trailer.as_slice().try_into().unwrap()))
            }
            Version::V3 => u64::from_le_bytes(trailer.as_slice().try_into().unwrap()),
        };
        let len = usize::try_from(len)
            .ok()
            .filter(|len| *len <= body.len())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "columnar section of {len} bytes exceeds its {} byte body",
                        body.len()
                    ),
                )
            })?;
        Ok((body, len))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_footer_deserialization() {
        let parsed_version: Version = parse_footer(footer()).unwrap();
        assert_eq!(Version::V3, parsed_version);
    }

    #[test]
    fn test_version_serialization() {
        let version_to_tests: Vec<u32> = [0, 1 << 8, 1 << 16, 1 << 24]
            .iter()
            .copied()
            .flat_map(|offset| (0..255).map(move |el| el + offset))
            .collect();
        let mut valid_versions: HashSet<u32> = HashSet::default();
        for &i in &version_to_tests {
            let version_res = Version::try_from_bytes(i.to_le_bytes());
            if let Ok(version) = version_res {
                assert_eq!(version.to_bytes(), i.to_le_bytes());
                valid_versions.insert(i);
            }
        }
        assert_eq!(valid_versions.len(), 3);
    }

    #[test]
    fn test_len_trailer_round_trip() {
        for version in [Version::V1, Version::V2, Version::V3] {
            for len in [0usize, 1, 7] {
                let mut buffer = vec![0xABu8; 7];
                version.write_len_trailer(len as u64, &mut buffer).unwrap();
                assert_eq!(buffer.len(), 7 + version.len_trailer_num_bytes());
                let (body, parsed) = version.split_len_trailer(FileSlice::from(buffer)).unwrap();
                assert_eq!(body.len(), 7);
                assert_eq!(parsed, len);
            }
        }
    }

    #[test]
    fn test_len_trailer_past_u32() {
        let len = u64::from(u32::MAX) + 1;
        let mut buffer = Vec::new();
        Version::V3.write_len_trailer(len, &mut buffer).unwrap();
        assert_eq!(buffer, len.to_le_bytes());
        for version in [Version::V1, Version::V2] {
            let error = version.write_len_trailer(len, &mut Vec::new()).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }

    #[test]
    fn test_len_trailer_rejects_length_past_body() {
        let mut buffer = vec![0u8; 3];
        Version::V3.write_len_trailer(4, &mut buffer).unwrap();
        let error = Version::V3
            .split_len_trailer(FileSlice::from(buffer))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}

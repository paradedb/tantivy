use std::io;

use common::HasLen;

use crate::directory::CompositeFile;
use crate::schema::Field;

/// Norm storage declared by a zero-length entry in the postings composite directory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum NormStorage {
    /// Document-addressed norms in the legacy fieldnorm component.
    Legacy = 1,
    /// Posting-addressed norms in the pnorm component.
    Posting = 2,
    /// Field norms are disabled in the schema.
    Disabled = 3,
}

impl NormStorage {
    pub(crate) fn read(composite: &CompositeFile, field: Field) -> io::Result<Self> {
        let mut storage = None;
        for mode in [Self::Legacy, Self::Posting, Self::Disabled] {
            if let Some(entry) = composite.open_read_with_idx(field, mode as usize) {
                if !entry.is_empty() || storage.replace(mode).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid norm storage metadata",
                    ));
                }
            }
        }
        Ok(storage.unwrap_or(Self::Legacy))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::Path;

    use super::*;
    use crate::directory::{CompositeWrite, Directory, RamDirectory};

    #[test]
    fn modes_are_explicit_and_validate_directory_entries() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        for (modes, payload, expected) in [
            (vec![], false, Some(NormStorage::Legacy)),
            (
                vec![NormStorage::Posting],
                false,
                Some(NormStorage::Posting),
            ),
            (
                vec![NormStorage::Disabled],
                false,
                Some(NormStorage::Disabled),
            ),
            (
                vec![NormStorage::Posting, NormStorage::Disabled],
                false,
                None,
            ),
            (vec![NormStorage::Posting], true, None),
        ] {
            let directory = RamDirectory::create();
            let path = Path::new("metadata");
            let mut writer = CompositeWrite::wrap(directory.open_write(path)?);
            for mode in modes {
                let entry = writer.for_field_with_idx(field, mode as usize);
                if payload {
                    entry.write_all(&[1])?;
                }
            }
            writer.for_field(field).write_all(&[0; 8])?;
            writer.close()?;
            let composite = CompositeFile::open(&directory.open_read(path)?)?;
            assert_eq!(NormStorage::read(&composite, field).ok(), expected);
        }
        Ok(())
    }
}

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::ops::Range;

use common::{BinarySerializable, CountingWriter, HasLen, VInt};

use crate::directory::{FileSlice, TerminatingWrite, WritePtr};
use crate::schema::{Field, Schema};
use crate::space_usage::{FieldUsage, PerFieldSpaceUsage};

#[derive(Eq, PartialEq, Hash, Copy, Ord, PartialOrd, Clone, Debug)]
pub struct FileAddr {
    field: Field,
    idx: usize,
}

impl FileAddr {
    fn new(field: Field, idx: usize) -> FileAddr {
        FileAddr { field, idx }
    }
}

impl BinarySerializable for FileAddr {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.field.serialize(writer)?;
        VInt(self.idx as u64).serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let field = Field::deserialize(reader)?;
        let idx = VInt::deserialize(reader)?.0 as usize;
        Ok(FileAddr { field, idx })
    }
}

/// A `CompositeWrite` is used to write a `CompositeFile`.
pub struct CompositeWrite<W = WritePtr> {
    write: CountingWriter<W>,
    offsets: Vec<(FileAddr, u64)>,
}

impl<W: TerminatingWrite + Write> CompositeWrite<W> {
    /// Crate a new API writer that writes a composite file
    /// in a given write.
    pub fn wrap(w: W) -> CompositeWrite<W> {
        CompositeWrite {
            write: CountingWriter::wrap(w),
            offsets: Vec::new(),
        }
    }

    /// Start writing a new field.
    pub fn for_field(&mut self, field: Field) -> &mut CountingWriter<W> {
        self.for_field_with_idx(field, 0)
    }

    /// Start writing a new field.
    pub fn for_field_with_idx(&mut self, field: Field, idx: usize) -> &mut CountingWriter<W> {
        let offset = self.write.written_bytes();
        let file_addr = FileAddr::new(field, idx);
        assert!(!self.offsets.iter().any(|el| el.0 == file_addr));
        self.offsets.push((file_addr, offset));
        &mut self.write
    }

    /// Pad the underlying file so the next field begins at an aligned
    /// absolute file offset.
    ///
    /// `prefix_len` is the number of bytes written before this composite
    /// writer was wrapped (for vector files, the fixed version header). The
    /// composite format represents adjacent start offsets rather than
    /// independent lengths, so the padding is a trailer on the preceding
    /// field. Readers of an aligned layout must validate and trim that
    /// at-most-`alignment - 1` trailer from the preceding logical payload.
    pub fn align_next_field(&mut self, alignment: usize, prefix_len: usize) -> io::Result<usize> {
        if !alignment.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "composite field alignment must be a non-zero power of two",
            ));
        }
        let written = usize::try_from(self.write.written_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "composite size does not fit in usize",
            )
        })?;
        let absolute_offset = prefix_len.checked_add(written).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "composite offset overflow")
        })?;
        let padding = (alignment - absolute_offset % alignment) % alignment;
        const ZEROES: [u8; 64] = [0; 64];
        let mut remaining = padding;
        while remaining != 0 {
            let chunk = remaining.min(ZEROES.len());
            self.write.write_all(&ZEROES[..chunk])?;
            remaining -= chunk;
        }
        Ok(padding)
    }

    /// Close the composite file
    ///
    /// An index of the different field offsets
    /// will be written as a footer.
    pub fn close(mut self) -> io::Result<()> {
        let footer_offset = self.write.written_bytes();
        VInt(self.offsets.len() as u64).serialize(&mut self.write)?;

        let mut prev_offset = 0;
        for (file_addr, offset) in self.offsets {
            VInt(offset - prev_offset).serialize(&mut self.write)?;
            file_addr.serialize(&mut self.write)?;
            prev_offset = offset;
        }

        let footer_len = (self.write.written_bytes() - footer_offset) as u32;
        footer_len.serialize(&mut self.write)?;
        self.write.terminate()
    }
}

/// A composite file is an abstraction to store a
/// file partitioned by field.
///
/// The file needs to be written field by field.
/// A footer describes the start and stop offsets
/// for each field.
#[derive(Clone)]
pub struct CompositeFile {
    data: FileSlice,
    offsets_index: HashMap<FileAddr, Range<usize>>,
}

impl std::fmt::Debug for CompositeFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeFile")
            .field("offsets_index", &self.offsets_index)
            .finish()
    }
}

impl CompositeFile {
    /// Opens a composite file stored in a given
    /// `FileSlice`.
    pub fn open(data: &FileSlice) -> io::Result<CompositeFile> {
        let end = data.len();
        let footer_len_data = data.slice_from(end - 4).read_bytes()?;
        let footer_len = u32::deserialize(&mut footer_len_data.as_slice())? as usize;
        let footer_start = end - 4 - footer_len;
        let footer_data = data
            .slice(footer_start..footer_start + footer_len)
            .read_bytes()?;
        let mut footer_buffer = footer_data.as_slice();
        let num_fields = VInt::deserialize(&mut footer_buffer)?.0 as usize;

        let mut file_addrs = vec![];
        let mut offsets = vec![];
        let mut field_index = HashMap::new();

        let mut offset = 0;
        for _ in 0..num_fields {
            offset += VInt::deserialize(&mut footer_buffer)?.0 as usize;
            let file_addr = FileAddr::deserialize(&mut footer_buffer)?;
            offsets.push(offset);
            file_addrs.push(file_addr);
        }
        offsets.push(footer_start);
        for i in 0..num_fields {
            let file_addr = file_addrs[i];
            let start_offset = offsets[i];
            let end_offset = offsets[i + 1];
            field_index.insert(file_addr, start_offset..end_offset);
        }

        Ok(CompositeFile {
            data: data.slice_to(footer_start),
            offsets_index: field_index,
        })
    }

    /// Returns a composite file that stores
    /// no fields.
    pub fn empty() -> CompositeFile {
        CompositeFile {
            offsets_index: HashMap::new(),
            data: FileSlice::empty(),
        }
    }

    /// Returns the `FileSlice` associated with
    /// a given `Field` and stored in a `CompositeFile`.
    pub fn open_read(&self, field: Field) -> Option<FileSlice> {
        self.open_read_with_idx(field, 0)
    }

    /// Returns the `FileSlice` associated with
    /// a given `Field` and stored in a `CompositeFile`.
    pub fn open_read_with_idx(&self, field: Field, idx: usize) -> Option<FileSlice> {
        self.offsets_index
            .get(&FileAddr { field, idx })
            .map(|byte_range| self.data.slice(byte_range.clone()))
    }

    /// Enumerates the `(field, index)` addresses declared by this composite's
    /// footer. Layout-specific readers can use this to reject indices outside
    /// their format without imposing one format's limits on other composite
    /// files.
    pub(crate) fn field_indices(&self) -> impl Iterator<Item = (Field, usize)> + '_ {
        self.offsets_index
            .keys()
            .map(|address| (address.field, address.idx))
    }

    /// Returns the space usage per field in this composite file.
    pub fn space_usage(&self, schema: &Schema) -> PerFieldSpaceUsage {
        let mut fields = Vec::new();
        for (&field_addr, byte_range) in &self.offsets_index {
            let field_name = schema.get_field_name(field_addr.field).to_string();
            let mut field_usage = FieldUsage::empty(field_name);
            field_usage.add_field_idx(field_addr.idx, byte_range.len().into());
            fields.push(field_usage);
        }
        PerFieldSpaceUsage::new(fields)
    }
}

#[cfg(test)]
mod test {

    use std::io::Write;
    use std::path::Path;

    use common::{BinarySerializable, HasLen, VInt};

    use super::{CompositeFile, CompositeWrite};
    use crate::directory::{Directory, RamDirectory};
    use crate::schema::Field;

    #[test]
    fn test_composite_file() -> crate::Result<()> {
        let path = Path::new("test_path");
        let directory = RamDirectory::create();
        {
            let w = directory.open_write(path).unwrap();
            let mut composite_write = CompositeWrite::wrap(w);
            let mut write_0 = composite_write.for_field(Field::from_field_id(0u32));
            VInt(32431123u64).serialize(&mut write_0)?;
            write_0.flush()?;
            let mut write_4 = composite_write.for_field(Field::from_field_id(4u32));
            VInt(2).serialize(&mut write_4)?;
            write_4.flush()?;
            composite_write.close()?;
        }
        {
            let r = directory.open_read(path)?;
            let composite_file = CompositeFile::open(&r)?;
            {
                let file0 = composite_file
                    .open_read(Field::from_field_id(0u32))
                    .unwrap()
                    .read_bytes()?;
                let mut file0_buf = file0.as_slice();
                let payload_0 = VInt::deserialize(&mut file0_buf)?.0;
                assert_eq!(file0_buf.len(), 0);
                assert_eq!(payload_0, 32431123u64);
            }
            {
                let file4 = composite_file
                    .open_read(Field::from_field_id(4u32))
                    .unwrap()
                    .read_bytes()?;
                let mut file4_buf = file4.as_slice();
                let payload_4 = VInt::deserialize(&mut file4_buf)?.0;
                assert_eq!(file4_buf.len(), 0);
                assert_eq!(payload_4, 2u64);
            }
        }
        Ok(())
    }

    #[test]
    fn test_composite_file_bug() -> crate::Result<()> {
        let path = Path::new("test_path");
        let directory = RamDirectory::create();
        {
            let w = directory.open_write(path).unwrap();
            let mut composite_write = CompositeWrite::wrap(w);
            let mut write = composite_write.for_field_with_idx(Field::from_field_id(1u32), 0);
            VInt(32431123u64).serialize(&mut write)?;
            write.flush()?;
            let write = composite_write.for_field_with_idx(Field::from_field_id(1u32), 1);
            write.flush()?;

            let mut write = composite_write.for_field_with_idx(Field::from_field_id(0u32), 0);
            VInt(1_000_000).serialize(&mut write)?;
            write.flush()?;

            composite_write.close()?;
        }
        {
            let r = directory.open_read(path)?;
            let composite_file = CompositeFile::open(&r)?;
            {
                let file = composite_file
                    .open_read_with_idx(Field::from_field_id(1u32), 0)
                    .unwrap()
                    .read_bytes()?;
                let mut file0_buf = file.as_slice();
                let payload_0 = VInt::deserialize(&mut file0_buf)?.0;
                assert_eq!(file0_buf.len(), 0);
                assert_eq!(payload_0, 32431123u64);
            }
            {
                let file = composite_file
                    .open_read_with_idx(Field::from_field_id(1u32), 1)
                    .unwrap()
                    .read_bytes()?;
                let file = file.as_slice();
                assert_eq!(file.len(), 0);
            }
            {
                let file = composite_file
                    .open_read_with_idx(Field::from_field_id(0u32), 0)
                    .unwrap()
                    .read_bytes()?;
                let file = file.as_slice();
                assert_eq!(file.len(), 3);
            }
        }
        Ok(())
    }

    #[test]
    fn test_align_next_field_accounts_for_file_prefix() -> crate::Result<()> {
        let mut bytes = vec![0_u8; 4];
        {
            let mut composite = CompositeWrite::wrap(&mut bytes);
            composite
                .for_field_with_idx(Field::from_field_id(0), 0)
                .write_all(&[7])?;
            let padding = composite.align_next_field(64, 4)?;
            assert_eq!(padding, 59);
            composite
                .for_field_with_idx(Field::from_field_id(0), 1)
                .write_all(&[9])?;
            composite.close()?;
        }

        assert_eq!(bytes[64], 9);
        let body = crate::directory::FileSlice::from(bytes[4..].to_vec());
        let composite = CompositeFile::open(&body)?;
        let preceding = composite
            .open_read_with_idx(Field::from_field_id(0), 0)
            .unwrap();
        assert_eq!(preceding.len(), 60);
        let aligned = composite
            .open_read_with_idx(Field::from_field_id(0), 1)
            .unwrap()
            .read_bytes()?;
        assert_eq!(aligned.as_slice(), &[9]);
        Ok(())
    }

    #[test]
    fn test_field_indices_enumerates_footer_addresses() -> crate::Result<()> {
        let mut bytes = Vec::new();
        {
            let mut composite = CompositeWrite::wrap(&mut bytes);
            composite
                .for_field_with_idx(Field::from_field_id(2), 7)
                .write_all(&[1])?;
            composite
                .for_field_with_idx(Field::from_field_id(0), 3)
                .write_all(&[2])?;
            composite.close()?;
        }

        let composite = CompositeFile::open(&crate::directory::FileSlice::from(bytes))?;
        let mut addresses = composite.field_indices().collect::<Vec<_>>();
        addresses.sort_unstable_by_key(|(field, idx)| (field.field_id(), *idx));
        assert_eq!(
            addresses,
            [(Field::from_field_id(0), 3), (Field::from_field_id(2), 7)]
        );
        Ok(())
    }
}

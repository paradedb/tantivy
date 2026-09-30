use std::cell::RefCell;
use std::cmp::min;
use std::io;
use std::ops::Range;

use super::file_slice::FileSlice;
use super::{HasLen, OwnedBytes};

const DEFAULT_BUFFER_MAX_SIZE: usize = 512 * 1024; // 512K

/// A buffered reader for a FileSlice.
///
/// Reads the underlying `FileSlice` in large, sequential chunks to amortize
/// the cost of `read_bytes` calls, while keeping peak memory usage under control.
///
/// TODO: Rather than wrapping a `FileSlice` in buffering, it will usually be better to adjust a
/// `FileHandle` to directly handle buffering itself.
/// TODO: See: https://github.com/paradedb/paradedb/issues/3374
#[derive(Clone)]
pub struct BufferedFileSlice {
    file_slice: FileSlice,
    buffer: RefCell<OwnedBytes>,
    buffer_range: RefCell<Range<u64>>,
    buffer_max_size: usize,
}

impl BufferedFileSlice {
    /// Creates a new `BufferedFileSlice`.
    ///
    /// The `buffer_max_size` is the amount of data that will be read from the
    /// `FileSlice` on a buffer miss.
    pub fn new(file_slice: FileSlice, buffer_max_size: usize) -> Self {
        Self {
            file_slice,
            buffer: RefCell::new(OwnedBytes::empty()),
            buffer_range: RefCell::new(0..0),
            buffer_max_size,
        }
    }

    /// Creates a new `BufferedFileSlice` with a default buffer max size.
    pub fn new_with_default_buffer_size(file_slice: FileSlice) -> Self {
        Self::new(file_slice, DEFAULT_BUFFER_MAX_SIZE)
    }

    /// Creates an empty `BufferedFileSlice`.
    pub fn empty() -> Self {
        Self::new(FileSlice::empty(), 0)
    }

    /// Reads a byte without cloning the retained buffer on a cache hit.
    #[inline]
    pub fn read_byte(&self, offset: u64) -> io::Result<u8> {
        let range = self.buffer_range.borrow();
        if range.contains(&offset) {
            return Ok(self.buffer.borrow()[(offset - range.start) as usize]);
        }
        drop(range);
        let end = offset.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::UnexpectedEof, "byte offset out of bounds")
        })?;
        Ok(self.get_bytes(offset..end)?[0])
    }

    /// Returns an `OwnedBytes` corresponding to the given `required_range`.
    ///
    /// If the requested range is not in the buffer, this will trigger a read
    /// from the underlying `FileSlice`.
    ///
    /// If the requested range is larger than the buffer_max_size, it will be read directly from the
    /// source without buffering.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying read fails or the range is
    /// out of bounds.
    pub fn get_bytes(&self, required_range: Range<u64>) -> io::Result<OwnedBytes> {
        let buffer_range = self.buffer_range.borrow();

        // Cache miss condition: the required range is not fully contained in the current buffer.
        if required_range.start < buffer_range.start || required_range.end > buffer_range.end {
            drop(buffer_range); // release borrow before mutating

            if required_range.end > self.file_slice.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Requested range extends beyond the end of the file slice.",
                ));
            }

            if (required_range.end - required_range.start) as usize > self.buffer_max_size {
                // This read is larger than our buffer max size.
                // Read it directly and bypass the buffer to avoid churning.
                return self
                    .file_slice
                    .read_bytes_slice(required_range.start as usize..required_range.end as usize);
            }

            let new_buffer_start = required_range.start;
            let new_buffer_end = min(
                new_buffer_start + self.buffer_max_size as u64,
                self.file_slice.len() as u64,
            );
            let read_range = new_buffer_start..new_buffer_end;

            let new_buffer = self
                .file_slice
                .read_bytes_slice(read_range.start as usize..read_range.end as usize)?;

            self.buffer.replace(new_buffer);
            self.buffer_range.replace(read_range);
        }

        // Now the data is guaranteed to be in the buffer.
        let buffer = self.buffer.borrow();
        let buffer_range = self.buffer_range.borrow();
        let local_start = (required_range.start - buffer_range.start) as usize;
        let local_end = (required_range.end - buffer_range.start) as usize;
        Ok(buffer.slice(local_start..local_end))
    }

    /// Reads through `read_ahead_end` on a cache miss, retaining any cached prefix.
    /// The entire refill is buffered regardless of the configured buffer size.
    pub fn get_bytes_with_read_ahead(
        &self,
        required_range: Range<u64>,
        read_ahead_end: u64,
    ) -> io::Result<OwnedBytes> {
        if required_range.start > required_range.end || required_range.end > read_ahead_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Read-ahead range does not contain the required range.",
            ));
        }
        if read_ahead_end > self.file_slice.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Read-ahead range extends beyond the end of the file slice.",
            ));
        }
        if required_range.is_empty() {
            return Ok(OwnedBytes::empty());
        }

        let buffer_range = self.buffer_range.borrow().clone();
        if required_range.start < buffer_range.start || required_range.end > buffer_range.end {
            let read_start = if buffer_range.contains(&required_range.start) {
                buffer_range.end
            } else {
                required_range.start
            };
            let suffix = self
                .file_slice
                .read_bytes_slice(read_start as usize..read_ahead_end as usize)?;
            let new_buffer = if read_start > required_range.start {
                let buffer = self.buffer.borrow();
                let prefix = &buffer[(required_range.start - buffer_range.start) as usize..];
                let mut bytes =
                    Vec::with_capacity((read_ahead_end - required_range.start) as usize);
                bytes.extend_from_slice(prefix);
                bytes.extend_from_slice(&suffix);
                OwnedBytes::new(bytes)
            } else {
                suffix
            };
            self.buffer.replace(new_buffer);
            self.buffer_range
                .replace(required_range.start..read_ahead_end);
        }

        self.get_bytes(required_range)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::file_slice::FileHandle;

    #[derive(Debug)]
    struct RecordingFile {
        reads: Mutex<Vec<Range<usize>>>,
        fail_from: usize,
    }

    impl HasLen for RecordingFile {
        fn len(&self) -> usize {
            64
        }
    }

    impl FileHandle for RecordingFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            if range.start >= self.fail_from {
                return Err(io::Error::other("injected read failure"));
            }
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(
                range.map(|byte| byte as u8).collect::<Vec<_>>(),
            ))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(16)
        }
    }

    fn recording_file() -> Arc<RecordingFile> {
        Arc::new(RecordingFile {
            reads: Mutex::default(),
            fail_from: usize::MAX,
        })
    }

    #[test]
    fn page_aligned_refills_retain_partial_blocks() -> io::Result<()> {
        let handle = recording_file();
        let file = FileSlice::new(handle.clone()).slice(3..61).slice(4..57);
        let buffer = BufferedFileSlice::new(file.clone(), 4);
        for range in [0..7, 7..12, 12..26, 26..29, 29..45, 45..53] {
            let reads_before = handle.reads.lock().unwrap().len();
            let end = file.storage_block_end(range.end).unwrap();
            let bytes = buffer
                .get_bytes_with_read_ahead(range.start as u64..range.end as u64, end as u64)?;
            let expected: Vec<u8> = (range.start + 7..range.end + 7)
                .map(|byte| byte as u8)
                .collect();
            assert_eq!(bytes.as_slice(), expected);
            if handle.reads.lock().unwrap().len() > reads_before {
                assert!(buffer.buffer.borrow().len() < range.len() + 16);
            }
        }
        assert_eq!(
            *handle.reads.lock().unwrap(),
            [7..16, 16..32, 32..48, 48..60]
        );
        Ok(())
    }

    #[test]
    fn read_ahead_only_refills_on_required_range_misses() -> io::Result<()> {
        let handle = recording_file();
        let buffer = BufferedFileSlice::new(FileSlice::new(handle.clone()), 4);
        let retained = buffer.get_bytes_with_read_ahead(0..8, 16)?;
        assert_eq!(
            buffer.get_bytes_with_read_ahead(8..16, 32)?.as_slice(),
            &(8..16).collect::<Vec<u8>>()
        );
        let clone = buffer.clone();
        assert_eq!(
            clone.get_bytes_with_read_ahead(12..20, 32)?.as_slice(),
            &(12..20).collect::<Vec<u8>>()
        );
        assert_eq!(*handle.reads.lock().unwrap(), [0..16, 16..32]);
        assert_eq!(retained.as_slice(), &(0..8).collect::<Vec<u8>>());
        assert_eq!(buffer.get_bytes_with_read_ahead(0..16, 16)?.len(), 16);
        assert_eq!(clone.buffer_range.borrow().clone(), 12..32);
        assert_eq!(buffer.buffer_range.borrow().clone(), 0..16);

        assert_eq!(
            buffer.get_bytes_with_read_ahead(40..48, 48)?.as_slice(),
            &(40..48).collect::<Vec<u8>>()
        );
        assert_eq!(
            buffer.get_bytes_with_read_ahead(2..4, 16)?.as_slice(),
            &[2, 3]
        );
        assert_eq!(
            *handle.reads.lock().unwrap(),
            [0..16, 16..32, 40..48, 2..16]
        );
        Ok(())
    }

    #[test]
    fn read_ahead_validates_ranges_without_reading() -> io::Result<()> {
        let handle = recording_file();
        let buffer = BufferedFileSlice::new(FileSlice::new(handle.clone()), 4);
        for offset in [0, 7, 64] {
            assert!(
                buffer
                    .get_bytes_with_read_ahead(offset..offset, offset)?
                    .is_empty()
            );
        }
        for (range, end, kind) in [
            (Range { start: 8, end: 4 }, 16, io::ErrorKind::InvalidInput),
            (0..8, 4, io::ErrorKind::InvalidInput),
            (0..8, 65, io::ErrorKind::UnexpectedEof),
            (63..65, 65, io::ErrorKind::UnexpectedEof),
            (u64::MAX..u64::MAX, u64::MAX, io::ErrorKind::UnexpectedEof),
        ] {
            assert_eq!(
                buffer
                    .get_bytes_with_read_ahead(range, end)
                    .unwrap_err()
                    .kind(),
                kind
            );
        }
        assert!(handle.reads.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn failed_continuation_preserves_cached_prefix() -> io::Result<()> {
        let mut handle = recording_file();
        Arc::get_mut(&mut handle).unwrap().fail_from = 16;
        let buffer = BufferedFileSlice::new(FileSlice::new(handle.clone()), 4);
        buffer.get_bytes_with_read_ahead(0..8, 16)?;
        assert!(buffer.get_bytes_with_read_ahead(12..20, 32).is_err());
        assert_eq!(
            buffer.get_bytes_with_read_ahead(12..16, 16)?.as_slice(),
            &[12, 13, 14, 15]
        );
        let reads = handle.reads.lock().unwrap();
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0], 0..16);
        assert_eq!(buffer.buffer_range.borrow().clone(), 0..16);
        Ok(())
    }

    #[test]
    fn fixed_size_reads_keep_existing_refill_and_bypass_behavior() -> io::Result<()> {
        let handle = recording_file();
        let buffer = BufferedFileSlice::new(FileSlice::new(handle.clone()), 4);
        assert_eq!(buffer.get_bytes(0..2)?.as_slice(), &[0, 1]);
        assert_eq!(buffer.get_bytes(2..5)?.as_slice(), &[2, 3, 4]);
        assert_eq!(buffer.get_bytes(10..20)?.len(), 10);
        assert_eq!(buffer.get_bytes(3..6)?.as_slice(), &[3, 4, 5]);
        assert_eq!(*handle.reads.lock().unwrap(), [0..4, 2..6, 10..20]);
        Ok(())
    }
}

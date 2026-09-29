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
    block_bounded: bool,
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
            block_bounded: false,
        }
    }

    /// Limits read-ahead to a storage block when the requested range fits within it.
    pub fn new_block_bounded(file_slice: FileSlice, buffer_max_size: usize) -> Self {
        Self {
            block_bounded: true,
            ..Self::new(file_slice, buffer_max_size)
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
            let mut new_buffer_end = min(
                new_buffer_start + self.buffer_max_size as u64,
                self.file_slice.len() as u64,
            );
            if self.block_bounded
                && !required_range.is_empty()
                && let Some(block) = self
                    .file_slice
                    .storage_block_range(required_range.start as usize)
            {
                new_buffer_end = if required_range.end <= block.end as u64 {
                    new_buffer_end.min(block.end as u64)
                } else {
                    required_range.end
                };
            }
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
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::file_slice::FileHandle;

    #[derive(Debug)]
    struct TrackedFile {
        data: Vec<u8>,
        block_len: Option<usize>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        fail: Arc<AtomicBool>,
    }

    impl HasLen for TrackedFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for TrackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            if self.fail.load(Ordering::Relaxed) {
                return Err(io::Error::other("read failed"));
            }
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.data[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            self.block_len
        }
    }

    #[test]
    fn block_bounded_reads_preserve_slices_and_clones() -> io::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let fail = Arc::new(AtomicBool::new(false));
        let file = FileSlice::new(Arc::new(TrackedFile {
            data: (0..100).collect(),
            block_len: Some(16),
            reads: reads.clone(),
            fail: fail.clone(),
        }));
        let file = file.slice(3..90).slice(10..80);
        let buffer = BufferedFileSlice::new_block_bounded(file.clone(), 17);
        assert_eq!(buffer.read_byte(0)?, 13);
        assert_eq!(buffer.get_bytes(1..3)?.as_slice(), &[14, 15]);
        assert_eq!(*reads.lock().unwrap(), vec![13..16]);
        assert_eq!(buffer.read_byte(3)?, 16);
        let clone = buffer.clone();
        assert_eq!(clone.get_bytes(4..8)?.as_slice(), &[17, 18, 19, 20]);
        assert_eq!(*reads.lock().unwrap(), vec![13..16, 16..32]);
        assert_eq!(buffer.read_byte(0)?, 13);
        assert_eq!(clone.read_byte(6)?, 19);
        assert_eq!(reads.lock().unwrap().len(), 3);
        assert_eq!(buffer.get_bytes(18..22)?.as_slice(), &[31, 32, 33, 34]);
        assert_eq!(reads.lock().unwrap().last(), Some(&(31..35)));
        assert_eq!(buffer.read_byte(22)?, 35);
        assert_eq!(reads.lock().unwrap().last(), Some(&(35..48)));
        assert_eq!(buffer.read_byte(69)?, 82);
        assert_eq!(reads.lock().unwrap().last(), Some(&(82..83)));
        assert!(buffer.get_bytes(70..70)?.is_empty());
        assert!(buffer.read_byte(70).is_err());
        assert!(buffer.read_byte(u64::MAX).is_err());
        assert_eq!(
            buffer.get_bytes(0..40)?.as_slice(),
            &(13..53).collect::<Vec<_>>()
        );
        assert_eq!(reads.lock().unwrap().last(), Some(&(13..53)));
        fail.store(true, Ordering::Relaxed);
        assert!(buffer.read_byte(5).is_err());
        assert_eq!(buffer.read_byte(69)?, 82);
        fail.store(false, Ordering::Relaxed);
        for offset in (0..70).rev().chain(0..70) {
            assert_eq!(buffer.read_byte(offset)?, (offset + 13) as u8);
        }
        let direct = BufferedFileSlice::new_block_bounded(file, 0);
        assert_eq!(direct.read_byte(0)?, 13);
        assert_eq!(reads.lock().unwrap().last(), Some(&(13..14)));
        Ok(())
    }

    #[test]
    fn block_bounded_reads_preserve_nonblock_read_ahead() -> io::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(TrackedFile {
            data: (0..100).collect(),
            block_len: None,
            reads: reads.clone(),
            fail: Arc::new(AtomicBool::new(false)),
        }));
        let buffer = BufferedFileSlice::new_block_bounded(file.slice(13..83), 17);
        assert_eq!(buffer.read_byte(0)?, 13);
        assert_eq!(*reads.lock().unwrap(), vec![13..30]);
        assert_eq!(buffer.read_byte(16)?, 29);
        assert_eq!(reads.lock().unwrap().len(), 1);
        Ok(())
    }
}

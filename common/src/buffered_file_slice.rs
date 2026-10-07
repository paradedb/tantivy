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
    buffer_by_storage_block: bool,
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
            buffer_by_storage_block: false,
        }
    }

    /// Aligns retained buffers to storage pages when the backend exposes them.
    pub fn with_storage_block_buffer(mut self) -> Self {
        self.buffer_by_storage_block = self.file_slice.storage_block_len().is_some();
        self
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

            if required_range.is_empty() {
                return Ok(OwnedBytes::empty());
            }

            if (required_range.end - required_range.start) as usize > self.buffer_max_size {
                // This read is larger than our buffer max size.
                // Read it directly and bypass the buffer to avoid churning.
                return self
                    .file_slice
                    .read_bytes_slice(required_range.start as usize..required_range.end as usize);
            }

            let read_range = if self.buffer_by_storage_block {
                let block = self
                    .file_slice
                    .storage_block_range(required_range.start as usize)
                    .unwrap();
                if required_range.end <= block.end as u64 {
                    block.start as u64..block.end as u64
                } else {
                    required_range.clone()
                }
            } else {
                let end = min(
                    required_range.start + self.buffer_max_size as u64,
                    self.file_slice.len() as u64,
                );
                required_range.start..end
            };

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
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::file_slice::FileHandle;

    #[derive(Debug)]
    struct PagedFile {
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        data: Vec<u8>,
    }

    impl HasLen for PagedFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for PagedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.data[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(8)
        }
    }

    #[test]
    fn storage_block_buffer_handles_slices_and_boundaries() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(PagedFile {
            reads: reads.clone(),
            data: (0..32).collect(),
        }));
        let buffer = BufferedFileSlice::new(file.slice(2..25), 8).with_storage_block_buffer();
        for range in [1..4, 0..1, 5..8, 8..9, 22..23, 23..23, 0..20] {
            let expected: Vec<u8> = range.clone().map(|i| (i + 2) as u8).collect();
            assert_eq!(buffer.get_bytes(range).unwrap().as_slice(), expected);
        }
        assert_eq!(
            *reads.lock().unwrap(),
            vec![2..8, 7..10, 8..16, 24..25, 2..22]
        );
        assert!(buffer.get_bytes(23..24).is_err());
    }
}

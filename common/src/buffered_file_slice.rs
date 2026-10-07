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
    block_aligned: bool,
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
            block_aligned: false,
        }
    }

    /// Aligns reads to storage blocks, or fixed-size buffers when storage geometry is unavailable.
    pub fn new_block_aligned(file_slice: FileSlice, buffer_max_size: usize) -> Self {
        assert!(buffer_max_size > 0);
        Self {
            block_aligned: true,
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
    #[inline(always)]
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

            let read_range = if self.block_aligned && !required_range.is_empty() {
                let first = required_range.start as usize;
                let last = required_range.end as usize - 1;
                let start = self.file_slice.storage_block_range(first).map_or_else(
                    || first / self.buffer_max_size * self.buffer_max_size,
                    |block| block.start,
                );
                let end = self.file_slice.storage_block_range(last).map_or_else(
                    || {
                        ((last / self.buffer_max_size + 1) * self.buffer_max_size)
                            .min(self.file_slice.len())
                    },
                    |block| block.end,
                );
                start as u64..end as u64
            } else {
                required_range.start
                    ..min(
                        required_range.start + self.buffer_max_size as u64,
                        self.file_slice.len() as u64,
                    )
            };
            let old_range = self.buffer_range.borrow();
            let overlap_start = read_range.start.max(old_range.start);
            let overlap_end = read_range.end.min(old_range.end);
            let new_buffer = if self.block_aligned && overlap_start < overlap_end {
                let mut bytes = Vec::with_capacity((read_range.end - read_range.start) as usize);
                if read_range.start < overlap_start {
                    bytes.extend_from_slice(
                        &self
                            .file_slice
                            .read_bytes_slice(read_range.start as usize..overlap_start as usize)?,
                    );
                }
                bytes.extend_from_slice(
                    &self.buffer.borrow()[(overlap_start - old_range.start) as usize
                        ..(overlap_end - old_range.start) as usize],
                );
                if overlap_end < read_range.end {
                    bytes.extend_from_slice(
                        &self
                            .file_slice
                            .read_bytes_slice(overlap_end as usize..read_range.end as usize)?,
                    );
                }
                OwnedBytes::new(bytes)
            } else {
                self.file_slice
                    .read_bytes_slice(read_range.start as usize..read_range.end as usize)?
            };
            drop(old_range);

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

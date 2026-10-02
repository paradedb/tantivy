use std::cell::RefCell;
use std::io;
use std::ops::Range;

use common::HasLen;
use once_cell::unsync::OnceCell;

use super::compression::{compressed_block_size, BlockDecoder, VIntDecoder};
use super::{BlockInfo, SkipReader};
use crate::directory::{FileSlice, OwnedBytes};
use crate::TERMINATED;

#[derive(Clone)]
pub(super) struct FrequencyReader {
    file: FileSlice,
    buffer: RefCell<(usize, OwnedBytes)>,
    decoder: OnceCell<Box<BlockDecoder>>,
    spare_decoder: RefCell<Option<Box<BlockDecoder>>>,
}

impl FrequencyReader {
    pub(super) fn new(file: FileSlice) -> Self {
        Self {
            file,
            buffer: RefCell::new((0, OwnedBytes::empty())),
            decoder: OnceCell::new(),
            spare_decoder: RefCell::default(),
        }
    }

    pub(super) fn invalidate(&mut self) {
        if let Some(decoder) = self.decoder.take() {
            *self.spare_decoder.get_mut() = Some(decoder);
        }
    }

    pub(super) fn reset(&mut self, file: FileSlice) {
        self.invalidate();
        if let Some(decoder) = self.spare_decoder.get_mut() {
            **decoder = BlockDecoder::with_val(1);
        }
        self.file = file;
        *self.buffer.get_mut() = (0, OwnedBytes::empty());
    }

    #[inline]
    pub(super) fn read(&self, skip_reader: &SkipReader) -> &BlockDecoder {
        self.decoder
            .get_or_try_init(|| {
                let mut decoder = self
                    .spare_decoder
                    .borrow_mut()
                    .take()
                    .unwrap_or_else(|| Box::new(BlockDecoder::with_val(1)));
                let start = skip_reader.freq_byte_offset();
                match skip_reader.block_info() {
                    BlockInfo::BitPacked {
                        tf_num_bits,
                        strict_delta_encoded,
                        ..
                    } => {
                        self.with_bytes(
                            start..start + compressed_block_size(tf_num_bits),
                            |bytes| {
                                decoder.uncompress_block_unsorted(
                                    bytes,
                                    tf_num_bits,
                                    strict_delta_encoded,
                                );
                            },
                        )?;
                    }
                    BlockInfo::VInt { num_docs } if num_docs > 0 && start < self.file.len() => {
                        self.with_bytes(start..self.file.len(), |bytes| {
                            decoder.uncompress_vint_unsorted(bytes, num_docs as usize, TERMINATED);
                        })?;
                    }
                    BlockInfo::VInt { .. } => {}
                }
                Ok::<_, io::Error>(decoder)
            })
            .expect("term frequencies became unreadable after the reader was opened")
    }

    fn with_bytes<T>(
        &self,
        range: Range<usize>,
        consume: impl FnOnce(&[u8]) -> T,
    ) -> io::Result<T> {
        if range.start > range.end || range.end > self.file.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "frequency range out of bounds",
            ));
        }
        if range.is_empty() {
            return Ok(consume(&[]));
        }
        if let Some(bytes) = self.file.as_slice() {
            return Ok(consume(&bytes[range]));
        }
        let buffer = self.buffer.borrow();
        if range.start >= buffer.0 && range.end <= buffer.0 + buffer.1.len() {
            return Ok(consume(
                &buffer.1[range.start - buffer.0..range.end - buffer.0],
            ));
        }
        drop(buffer);
        let read_range = if let Some(first) = self.file.storage_block_range(range.start) {
            let last = self.file.storage_block_range(range.end - 1).unwrap();
            first.start..last.end
        } else {
            range.start
                ..range
                    .start
                    .saturating_add(1024)
                    .max(range.end)
                    .min(self.file.len())
        };
        let bytes = self.file.read_bytes_slice(read_range.clone())?;
        *self.buffer.borrow_mut() = (read_range.start, bytes);
        let buffer = self.buffer.borrow();
        Ok(consume(
            &buffer.1[range.start - buffer.0..range.end - buffer.0],
        ))
    }
}

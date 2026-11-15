//! Storage reads of vector components, charged to the running search stage.
use std::cell::Cell;
use std::ops::Range;

use common::{HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::vector::telemetry::charge_read;

/// The vector data requested by the current storage read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VectorReadKind {
    /// Document addressing, metadata, or unclassified reads.
    #[default]
    Other,
    /// Original vector rows, including exact scans and reranking.
    FullPrecision,
    /// Quantized codes and their associated norms and sidecars.
    QuantizedLayer(u8),
}

thread_local! {
    static READ_KIND: Cell<VectorReadKind> = const { Cell::new(VectorReadKind::Other) };
}

/// The requested data kind, independent of the current search stage.
pub fn current_vector_read_kind() -> VectorReadKind {
    READ_KIND.get()
}

struct ReadKindGuard(VectorReadKind);

impl Drop for ReadKindGuard {
    fn drop(&mut self) {
        READ_KIND.set(self.0);
    }
}

/// Summable physical read requests; storage blocks count repetitions across requests.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct VectorIoStats {
    /// Successful byte-read requests.
    pub reads: u64,
    /// Total requested bytes, including padding in bands.
    pub bytes_read: u64,
    /// Sum of physical storage blocks touched per request, when geometry is known.
    pub storage_blocks: u64,
}

/// Where a scan's borrowed reads keep their bytes. Each read replaces the last, so a scan reading
/// through one scratch allocates when the buffer grows, not on every read.
#[derive(Default)]
pub(crate) struct ByteScratch {
    /// The assembled bytes of a read that spans storage blocks.
    buf: Vec<u8>,
    /// The owner of a read served from within one storage block.
    pinned: Option<OwnedBytes>,
}

/// Reads vector bytes and charges the request to the running search stage.
pub(crate) trait VectorRead {
    fn read_vector_bytes(&self, kind: VectorReadKind) -> std::io::Result<OwnedBytes>;
    fn read_vector_bytes_slice(
        &self,
        range: Range<usize>,
        kind: VectorReadKind,
    ) -> std::io::Result<OwnedBytes>;
    /// The bytes of `range` for as long as `scratch` stays borrowed, charged as
    /// [`Self::read_vector_bytes_slice`] charges them.
    ///
    /// Storage that holds its bytes in memory lends them. On block storage a range within one
    /// block is read as an owner, which such storage hands out without copying, and `scratch`
    /// keeps it alive. Any other range is assembled into the buffer `scratch` reuses.
    fn borrow_vector_bytes_slice<'a>(
        &'a self,
        range: Range<usize>,
        kind: VectorReadKind,
        scratch: &'a mut ByteScratch,
    ) -> std::io::Result<&'a [u8]>;
}
impl VectorRead for FileSlice {
    fn read_vector_bytes(&self, kind: VectorReadKind) -> std::io::Result<OwnedBytes> {
        self.read_vector_bytes_slice(0..self.len(), kind)
    }

    fn read_vector_bytes_slice(
        &self,
        range: Range<usize>,
        kind: VectorReadKind,
    ) -> std::io::Result<OwnedBytes> {
        assert!(range.start <= range.end, "invalid vector byte range");
        let _kind = ReadKindGuard(READ_KIND.replace(kind));
        let bytes = self.read_bytes_slice(range.clone())?;
        charge_range(self, &range);
        Ok(bytes)
    }

    fn borrow_vector_bytes_slice<'a>(
        &'a self,
        range: Range<usize>,
        kind: VectorReadKind,
        scratch: &'a mut ByteScratch,
    ) -> std::io::Result<&'a [u8]> {
        assert!(range.start <= range.end, "invalid vector byte range");
        let _kind = ReadKindGuard(READ_KIND.replace(kind));
        let blocks = storage_blocks(self, &range);
        let bytes = match self.as_slice() {
            Some(bytes) => &bytes[range.clone()],
            None if blocks == 1 => scratch
                .pinned
                .insert(self.read_bytes_slice(range.clone())?)
                .as_slice(),
            None => {
                scratch.pinned = None;
                scratch.buf.clear();
                self.read_bytes_slice_into(range.clone(), &mut scratch.buf)?;
                &scratch.buf
            }
        };
        charge_read(range.len() as u64, blocks);
        Ok(bytes)
    }
}

/// The storage blocks `range` spans in `slice`: 0 when it is empty or the storage has no block
/// geometry.
fn storage_blocks(slice: &FileSlice, range: &Range<usize>) -> u64 {
    if range.is_empty() {
        return 0;
    }
    match (
        slice.storage_block_ord(range.start),
        slice.storage_block_ord(range.end - 1),
    ) {
        (Some(first), Some(last)) => (last - first + 1) as u64,
        _ => 0,
    }
}

/// Charges one read of `range` in `slice`, with the storage blocks it spans.
fn charge_range(slice: &FileSlice, range: &Range<usize>) {
    charge_read(range.len() as u64, storage_blocks(slice, range));
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::test_support::BlockTrackedBytes;
    use super::*;
    use crate::vector::telemetry::{enter_vector_stage, measure};
    use crate::vector::VectorStage;

    #[test]
    fn ranged_reads_preserve_origin_and_charge_only_requested_bytes() -> std::io::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let slice = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: (0..64).collect(),
            reads: Arc::clone(&reads),
            block_len: 8,
        }))
        .slice(11..53);
        let (bytes, totals) = measure(|| {
            let _stage = enter_vector_stage(VectorStage::LayerScan(2));
            slice.read_vector_bytes_slice(3..16, VectorReadKind::QuantizedLayer(2))
        });
        assert_eq!(&*bytes?, &(14..27).collect::<Vec<u8>>());
        assert_eq!(*reads.lock().unwrap(), [14..27]);
        let io = totals.io(VectorStage::LayerScan(2));
        assert_eq!((io.reads, io.bytes_read, io.storage_blocks), (1, 13, 3));
        assert_eq!(current_vector_read_kind(), VectorReadKind::Other);

        let (bytes, totals) = measure(|| {
            let _stage = enter_vector_stage(VectorStage::LayerScan(2));
            slice.read_vector_bytes_slice(42..42, VectorReadKind::QuantizedLayer(2))
        });
        assert!(bytes?.is_empty());
        let io = totals.io(VectorStage::LayerScan(2));
        assert_eq!((io.reads, io.bytes_read, io.storage_blocks), (1, 0, 0));
        Ok(())
    }

    /// A borrowed read that spans storage blocks fills the scratch buffer, replacing what the
    /// previous read left, and is requested and charged exactly as an owned read is.
    #[test]
    fn borrowed_reads_reuse_the_scratch_and_charge_like_owned_reads() -> std::io::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let slice = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: (0..64).collect(),
            reads: Arc::clone(&reads),
            block_len: 8,
        }))
        .slice(11..53);
        let mut scratch = ByteScratch::default();
        let (read, totals) = measure(|| -> std::io::Result<()> {
            let _stage = enter_vector_stage(VectorStage::LayerScan(2));
            let kind = VectorReadKind::QuantizedLayer(2);
            let bytes = slice.borrow_vector_bytes_slice(3..16, kind, &mut scratch)?;
            assert_eq!(bytes, &(14..27).collect::<Vec<u8>>());
            let bytes = slice.borrow_vector_bytes_slice(20..22, kind, &mut scratch)?;
            assert_eq!(bytes, &[31, 32]);
            assert!(slice
                .borrow_vector_bytes_slice(42..42, kind, &mut scratch)?
                .is_empty());
            Ok(())
        });
        read?;
        assert!(
            scratch.buf.capacity() >= 13,
            "the scratch keeps its allocation"
        );
        assert!(scratch.pinned.is_none());
        assert_eq!(*reads.lock().unwrap(), [14..27, 31..33, 53..53]);
        let io = totals.io(VectorStage::LayerScan(2));
        assert_eq!((io.reads, io.bytes_read, io.storage_blocks), (3, 15, 5));
        assert_eq!(current_vector_read_kind(), VectorReadKind::Other);
        Ok(())
    }

    /// Storage that holds its bytes in memory lends them and leaves the scratch alone.
    #[test]
    fn borrowed_reads_of_in_memory_storage_skip_the_scratch() -> std::io::Result<()> {
        let slice = FileSlice::from((0..64).collect::<Vec<u8>>()).slice(11..53);
        let mut scratch = ByteScratch {
            buf: vec![7_u8; 3],
            pinned: None,
        };
        let bytes =
            slice.borrow_vector_bytes_slice(3..16, VectorReadKind::FullPrecision, &mut scratch)?;
        assert_eq!(bytes, &(14..27).collect::<Vec<u8>>());
        assert_eq!(scratch.buf, [7, 7, 7]);
        assert!(scratch.pinned.is_none());
        Ok(())
    }

    /// Bytes that log which entry point each read arrives through.
    #[derive(Debug)]
    struct EntryTracked {
        bytes: Vec<u8>,
        owned: Arc<Mutex<Vec<Range<usize>>>>,
        into: Arc<Mutex<Vec<Range<usize>>>>,
    }

    impl HasLen for EntryTracked {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl crate::directory::FileHandle for EntryTracked {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.owned.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }

        fn read_bytes_into(&self, range: Range<usize>, buf: &mut Vec<u8>) -> std::io::Result<()> {
            self.into.lock().unwrap().push(range.clone());
            buf.extend_from_slice(&self.bytes[range]);
            Ok(())
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(8)
        }
    }

    /// A read within one storage block takes the handle's owner, which block storage hands out
    /// without copying; a read that spans blocks is assembled in the scratch buffer. Either
    /// replaces the read before it.
    #[test]
    fn borrowed_reads_copy_only_ranges_that_span_storage_blocks() -> std::io::Result<()> {
        let (owned, into) = (Arc::default(), Arc::default());
        let slice = FileSlice::new(Arc::new(EntryTracked {
            bytes: (0..64).collect(),
            owned: Arc::clone(&owned),
            into: Arc::clone(&into),
        }))
        .slice(11..53);
        let mut scratch = ByteScratch::default();
        let (read, totals) = measure(|| -> std::io::Result<()> {
            let _stage = enter_vector_stage(VectorStage::LayerScan(1));
            let kind = VectorReadKind::QuantizedLayer(1);
            // Absolute 16..24 is exactly block 2.
            let bytes = slice.borrow_vector_bytes_slice(5..13, kind, &mut scratch)?;
            assert_eq!(bytes, &(16..24).collect::<Vec<u8>>());
            assert!(scratch.buf.is_empty() && scratch.pinned.is_some());
            // One byte further reaches block 3.
            let bytes = slice.borrow_vector_bytes_slice(5..14, kind, &mut scratch)?;
            assert_eq!(bytes, &(16..25).collect::<Vec<u8>>());
            assert!(scratch.pinned.is_none());
            // The last byte of block 2 alone.
            let bytes = slice.borrow_vector_bytes_slice(12..13, kind, &mut scratch)?;
            assert_eq!(bytes, &[23]);
            Ok(())
        });
        read?;
        assert_eq!(*owned.lock().unwrap(), [16..24, 23..24]);
        assert_eq!(*into.lock().unwrap(), [16..25]);
        let io = totals.io(VectorStage::LayerScan(1));
        assert_eq!((io.reads, io.bytes_read, io.storage_blocks), (3, 18, 4));
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::io;
    use std::ops::Range;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use common::{HasLen, OwnedBytes};

    use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
    use crate::directory::{
        Directory, FileHandle, InnerWritePtr, RamDirectory, TempFilePtr, WatchCallback, WatchHandle,
    };
    use crate::vector::{
        current_vector_read_kind, current_vector_stage, VectorReadKind, VectorStage,
    };

    /// Ranges read, in order.
    pub(crate) type TrackedReads = Arc<Mutex<Vec<Range<usize>>>>;

    /// In-memory bytes that log every range read and report a storage block length.
    #[derive(Debug)]
    pub(crate) struct BlockTrackedBytes {
        pub(crate) bytes: Vec<u8>,
        pub(crate) reads: TrackedReads,
        pub(crate) block_len: usize,
    }

    impl HasLen for BlockTrackedBytes {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for BlockTrackedBytes {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(self.block_len)
        }
    }

    pub(crate) const PAGE_BYTES: usize = 8192;

    type ReadLog = Arc<Mutex<Vec<(VectorStage, Range<usize>)>>>;

    thread_local! {
        static LOG_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }

    /// Builds trace fixtures before the probe's read log is armed, restoring nested state on exit.
    pub(crate) fn with_unarmed_log<T>(prepare: impl FnOnce() -> T) -> T {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                LOG_ARMED.set(self.0);
            }
        }
        let _restore = Restore(LOG_ARMED.replace(false));
        prepare()
    }

    /// Records vector read requests against a PostgreSQL-sized page geometry.
    #[derive(Clone, Debug, Default)]
    pub(crate) struct PagedDirectory {
        inner: RamDirectory,
        pub(crate) reads: ReadLog,
        pub(crate) centroid_reads: ReadLog,
        pub(crate) read_kinds: Arc<Mutex<Vec<VectorReadKind>>>,
    }
    #[derive(Debug)]
    struct PagedFile {
        inner: Arc<dyn FileHandle>,
        reads: ReadLog,
        read_kinds: Arc<Mutex<Vec<VectorReadKind>>>,
    }
    impl HasLen for PagedFile {
        fn len(&self) -> usize {
            self.inner.len()
        }
    }
    impl FileHandle for PagedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            let bytes = self.inner.read_bytes(range.clone())?;
            if LOG_ARMED.get() {
                self.read_kinds
                    .lock()
                    .unwrap()
                    .push(current_vector_read_kind());
                self.reads
                    .lock()
                    .unwrap()
                    .push((current_vector_stage(), range));
            }
            Ok(bytes)
        }
        fn storage_block_len(&self) -> Option<usize> {
            Some(PAGE_BYTES)
        }
    }
    impl Directory for PagedDirectory {
        fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
            let inner = self.inner.get_file_handle(path)?;
            if path.extension().is_some_and(|ext| ext == "vec") {
                Ok(Arc::new(PagedFile {
                    inner,
                    reads: self.reads.clone(),
                    read_kinds: self.read_kinds.clone(),
                }))
            } else if path.extension().is_some_and(|ext| ext == "centroids") {
                Ok(Arc::new(PagedFile {
                    inner,
                    reads: self.centroid_reads.clone(),
                    read_kinds: Arc::default(),
                }))
            } else {
                Ok(inner)
            }
        }
        fn delete(&self, path: &Path) -> Result<(), DeleteError> {
            self.inner.delete(path)
        }
        fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
            self.inner.exists(path)
        }
        fn open_write_inner(&self, path: &Path) -> Result<InnerWritePtr, OpenWriteError> {
            self.inner.open_write_inner(path)
        }
        fn open_temp_file(&self) -> io::Result<TempFilePtr> {
            self.inner.open_temp_file()
        }
        fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
            self.inner.atomic_read(path)
        }
        fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
            self.inner.atomic_write(path, data)
        }
        fn sync_directory(&self) -> io::Result<()> {
            self.inner.sync_directory()
        }
        fn watch(&self, callback: WatchCallback) -> crate::Result<WatchHandle> {
            self.inner.watch(callback)
        }
    }
}

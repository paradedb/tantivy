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
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct VectorIoStats {
    /// Successful byte-read requests.
    pub reads: u64,
    /// Total requested bytes, including padding in bands.
    pub bytes_read: u64,
    /// Sum of physical storage blocks touched per request, when geometry is known.
    pub storage_blocks: u64,
}

/// Reads vector bytes and charges the request to the running search stage.
pub(crate) trait VectorRead {
    fn read_vector_bytes(&self, kind: VectorReadKind) -> std::io::Result<OwnedBytes>;
    fn visit_vector_bytes(
        &self,
        ranges: &[Range<usize>],
        kind: VectorReadKind,
        visitor: &mut dyn FnMut(usize, OwnedBytes) -> std::io::Result<()>,
    ) -> std::io::Result<()>;
}
impl VectorRead for FileSlice {
    fn visit_vector_bytes(
        &self,
        ranges: &[Range<usize>],
        kind: VectorReadKind,
        visitor: &mut dyn FnMut(usize, OwnedBytes) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let previous = READ_KIND.replace(kind);
        let _kind = ReadKindGuard(previous);
        self.visit_bytes(ranges, &mut |i, bytes| {
            let range = &ranges[i];
            let blocks = if range.is_empty() {
                0
            } else {
                self.storage_block_ord(range.start)
                    .zip(self.storage_block_ord(range.end - 1))
                    .map_or(0, |(first, last)| (last - first + 1) as u64)
            };
            charge_read(range.len() as u64, blocks);
            let _kind = ReadKindGuard(READ_KIND.replace(previous));
            visitor(i, bytes)
        })
    }

    fn read_vector_bytes(&self, kind: VectorReadKind) -> std::io::Result<OwnedBytes> {
        let _kind = ReadKindGuard(READ_KIND.replace(kind));
        let bytes = self.read_bytes()?;
        let mut storage_blocks = 0;
        if self.len() != 0 {
            if let (Some(first), Some(last)) = (
                self.storage_block_ord(0),
                self.storage_block_ord(self.len() - 1),
            ) {
                storage_blocks = (last - first + 1) as u64;
            }
        }
        charge_read(self.len() as u64, storage_blocks);
        Ok(bytes)
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
        pub(crate) read_kinds: Arc<Mutex<Vec<VectorReadKind>>>,
        pub(crate) streaming: Arc<std::sync::atomic::AtomicBool>,
    }
    #[derive(Debug)]
    struct PagedFile {
        inner: Arc<dyn FileHandle>,
        reads: ReadLog,
        read_kinds: Arc<Mutex<Vec<VectorReadKind>>>,
        streaming: Arc<std::sync::atomic::AtomicBool>,
    }
    impl HasLen for PagedFile {
        fn len(&self) -> usize {
            self.inner.len()
        }
    }
    impl FileHandle for PagedFile {
        fn supports_streaming_reads(&self) -> bool {
            self.streaming.load(std::sync::atomic::Ordering::Relaxed)
        }
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
                    streaming: self.streaming.clone(),
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

//! Per-thread storage I/O counters for vector search stages.
//!
//! A segment probe runs on one thread. The backend attributes its reads by
//! subtracting snapshots taken before and after the probe. Each read updates
//! a few integer counters; storage blocks count repeated page visits.
use std::cell::Cell;

use common::{HasLen, OwnedBytes};

use super::{current_vector_stage, Stage};
use crate::directory::FileSlice;

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
impl VectorIoStats {
    pub(crate) fn since(self, before: Self) -> Self {
        Self {
            reads: self.reads - before.reads,
            bytes_read: self.bytes_read - before.bytes_read,
            storage_blocks: self.storage_blocks - before.storage_blocks,
        }
    }
}
/// Counter slots: layer scans 0..3, the final stage (serialized as `rerank_*`), exact reads
/// before the final stage, and location-map and centroid-row reads.
pub(crate) const IO_SLOTS: usize = 6;
pub(crate) const RERANK_SLOT: usize = 3;
pub(crate) const EXACT_SLOT: usize = 4;
pub(crate) const LOCATE_SLOT: usize = 5;
thread_local! {
    static COUNTERS: Cell<[VectorIoStats; IO_SLOTS]> = const { Cell::new([VectorIoStats { reads: 0, bytes_read: 0, storage_blocks: 0 }; IO_SLOTS]) };
}
pub(crate) fn snapshot() -> [VectorIoStats; IO_SLOTS] {
    COUNTERS.get()
}

/// Reads vector bytes while attributing the actual range to the active scan stage.
pub(crate) trait VectorRead {
    fn read_vector_bytes(&self) -> std::io::Result<OwnedBytes>;
}
impl VectorRead for FileSlice {
    fn read_vector_bytes(&self) -> std::io::Result<OwnedBytes> {
        let bytes = self.read_bytes()?;
        let slot = match current_vector_stage() {
            Stage::LayerScan(l) if l < 3 => Some(l as usize),
            Stage::RerankFetch => Some(RERANK_SLOT),
            Stage::Exact => Some(EXACT_SLOT),
            Stage::Locate => Some(LOCATE_SLOT),
            _ => None,
        };
        if let Some(slot) = slot {
            let mut counters = COUNTERS.get();
            let counter = &mut counters[slot];
            counter.reads += 1;
            counter.bytes_read += self.len() as u64;
            if self.len() != 0 {
                if let (Some(first), Some(last)) = (
                    self.storage_block_ord(0),
                    self.storage_block_ord(self.len() - 1),
                ) {
                    counter.storage_blocks += (last - first + 1) as u64;
                }
            }
            COUNTERS.set(counters);
        }
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
    use crate::vector::{current_vector_stage, Stage};

    pub(crate) const PAGE_BYTES: usize = 8192;

    type ReadLog = Arc<Mutex<Vec<(Stage, Range<usize>)>>>;

    thread_local! {
        static LOG_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
        static DOC_IDS_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Counts a cluster DocIds column read made while the probe's read log is armed.
    pub(crate) fn record_doc_ids_read() {
        if LOG_ARMED.get() {
            DOC_IDS_READS.set(DOC_IDS_READS.get() + 1);
        }
    }

    /// DocIds column reads recorded on this thread so far.
    pub(crate) fn doc_ids_reads() -> usize {
        DOC_IDS_READS.get()
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
    }
    #[derive(Debug)]
    struct PagedFile {
        inner: Arc<dyn FileHandle>,
        reads: ReadLog,
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

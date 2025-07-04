use std::any::Any;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::{fmt, io, thread};

use log::Level;

use crate::directory::directory_lock::Lock;
use crate::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use crate::directory::{
    FileHandle, FileSlice, TerminatingWrite, WatchCallback, WatchHandle, WritePtr,
};
use crate::index::SegmentMetaInventory;
use crate::IndexMeta;

/// Retry the logic of acquiring locks is pretty simple.
/// We just retry `n` times after a given `duratio`, both
/// depending on the type of lock.
struct RetryPolicy {
    num_retries: usize,
    wait_in_ms: u64,
}

impl RetryPolicy {
    fn no_retry() -> RetryPolicy {
        RetryPolicy {
            num_retries: 0,
            wait_in_ms: 0,
        }
    }

    fn wait_and_retry(&mut self) -> bool {
        if self.num_retries == 0 {
            false
        } else {
            self.num_retries -= 1;
            let wait_duration = Duration::from_millis(self.wait_in_ms);
            thread::sleep(wait_duration);
            true
        }
    }
}

/// The `DirectoryLock` is an object that represents a file lock.
///
/// It is associated with a lock file, that gets deleted on `Drop.`
#[expect(dead_code)]
pub struct DirectoryLock(Box<dyn Send + Sync + 'static>);

struct DirectoryLockGuard {
    directory: Box<dyn Directory>,
    path: PathBuf,
}

impl<T: Send + Sync + 'static> From<Box<T>> for DirectoryLock {
    fn from(underlying: Box<T>) -> Self {
        DirectoryLock(underlying)
    }
}

impl Drop for DirectoryLockGuard {
    fn drop(&mut self) {
        if let Err(e) = self.directory.delete(&self.path) {
            error!("Failed to remove the lock file. {:?}", e);
        }
    }
}

enum TryAcquireLockError {
    FileExists,
    IoError(Arc<io::Error>),
}
impl From<io::Error> for TryAcquireLockError {
    fn from(io_error: io::Error) -> Self {
        Self::IoError(Arc::new(io_error))
    }
}

fn try_acquire_lock(
    filepath: &Path,
    directory: &dyn Directory,
) -> Result<DirectoryLock, TryAcquireLockError> {
    let mut write = directory.open_write(filepath).map_err(|e| match e {
        OpenWriteError::FileAlreadyExists(_) => TryAcquireLockError::FileExists,
        OpenWriteError::IoError { io_error, .. } => TryAcquireLockError::IoError(io_error),
    })?;
    write.flush().map_err(TryAcquireLockError::from)?;
    Ok(DirectoryLock::from(Box::new(DirectoryLockGuard {
        directory: directory.box_clone(),
        path: filepath.to_owned(),
    })))
}

fn retry_policy(is_blocking: bool) -> RetryPolicy {
    if is_blocking {
        RetryPolicy {
            num_retries: 100,
            wait_in_ms: 100,
        }
    } else {
        RetryPolicy::no_retry()
    }
}

pub type DirectoryPanicHandler = Arc<dyn Fn(Box<dyn Any + Send>) + Send + Sync + 'static>;

/// Write-once read many (WORM) abstraction for where
/// tantivy's data should be stored.
///
/// There are currently two implementations of `Directory`
///
/// - The [`MMapDirectory`][crate::directory::MmapDirectory], this should be your default choice.
/// - The [`RamDirectory`][crate::directory::RamDirectory], which should be used mostly for tests.
pub trait Directory: DirectoryClone + fmt::Debug + Send + Sync + 'static {
    /// Opens a file and returns a boxed `FileHandle`.
    ///
    /// Users of `Directory` should typically call `Directory::open_read(...)`,
    /// while `Directory` implementor should implement `get_file_handle()`.
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError>;

    /// Once a virtual file is open, its data may not
    /// change.
    ///
    /// Specifically, subsequent writes or flushes should
    /// have no effect on the returned [`FileSlice`] object.
    ///
    /// You should only use this to read files create with [`Directory::open_write()`].
    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        Ok(FileSlice::new(file_handle))
    }

    /// Removes a file
    ///
    /// Removing a file will not affect an eventual
    /// existing [`FileSlice`] pointing to it.
    ///
    /// Removing a nonexistent file, returns a
    /// [`DeleteError::FileDoesNotExist`].
    fn delete(&self, path: &Path) -> Result<(), DeleteError>;

    /// Returns true if and only if the file exists
    fn exists(&self, path: &Path) -> Result<bool, OpenReadError>;

    /// Returns a boxed `TerminatingWrite` object, to be passed into `open_write`
    /// which wraps it in a `BufWriter`
    fn open_write_inner(&self, path: &Path) -> Result<Box<dyn TerminatingWrite>, OpenWriteError>;

    /// Opens a writer for the *virtual file* associated with
    /// a [`Path`].
    ///
    /// Right after this call, for the span of the execution of the program
    /// the file should be created and any subsequent call to
    /// [`Directory::open_read()`] for the same path should return
    /// a [`FileSlice`].
    ///
    /// However, depending on the directory implementation,
    /// it might be required to call [`Directory::sync_directory()`] to ensure
    /// that the file is durably created.
    /// (The semantics here are the same when dealing with
    /// a POSIX filesystem.)
    ///
    /// Write operations may be aggressively buffered.
    /// The client of this trait is responsible for calling flush
    /// to ensure that subsequent `read` operations
    /// will take into account preceding `write` operations.
    ///
    /// Flush operation should also be persistent.
    ///
    /// The user shall not rely on [`Drop`] triggering `flush`.
    /// Note that [`RamDirectory`][crate::directory::RamDirectory] will
    /// panic! if `flush` was not called.
    ///
    /// The file may not previously exist.
    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        Ok(io::BufWriter::with_capacity(
            self.bufwriter_capacity(),
            self.open_write_inner(path)?,
        ))
    }

    /// Reads the full content file that has been written using
    /// [`Directory::atomic_write()`].
    ///
    /// This should only be used for small files.
    ///
    /// You should only use this to read files create with [`Directory::atomic_write()`].
    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError>;

    /// Atomically replace the content of a file with data.
    ///
    /// This calls ensure that reads can never *observe*
    /// a partially written file.
    ///
    /// The file may or may not previously exist.
    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()>;

    /// Sync the directory.
    ///
    /// This call is required to ensure that newly created files are
    /// effectively stored durably.
    fn sync_directory(&self) -> io::Result<()>;

    /// Acquire a lock in the directory given in the [`Lock`].
    ///
    /// The method is blocking or not depending on the [`Lock`] object.
    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        let box_directory = self.box_clone();
        let mut retry_policy = retry_policy(lock.is_blocking);
        loop {
            match try_acquire_lock(&lock.filepath, &*box_directory) {
                Ok(result) => {
                    return Ok(result);
                }
                Err(TryAcquireLockError::FileExists) => {
                    if !retry_policy.wait_and_retry() {
                        return Err(LockError::LockBusy);
                    }
                }
                Err(TryAcquireLockError::IoError(io_error)) => {
                    return Err(LockError::IoError(io_error));
                }
            }
        }
    }

    /// Registers a callback that will be called whenever a change on the `meta.json`
    /// using the [`Directory::atomic_write()`] API is detected.
    ///
    /// The behavior when using `.watch()` on a file using [`Directory::open_write()`] is, on the
    /// other hand, undefined.
    ///
    /// The file will be watched for the lifetime of the returned `WatchHandle`. The caller is
    /// required to keep it.
    /// It does not override previous callbacks. When the file is modified, all callback that are
    /// registered (and whose [`WatchHandle`] is still alive) are triggered.
    ///
    /// Internally, tantivy only uses this API to detect new commits to implement the
    /// `OnCommitWithDelay` `ReloadPolicy`. Not implementing watch in a `Directory` only prevents
    /// the `OnCommitWithDelay` `ReloadPolicy` to work properly.
    fn watch(&self, watch_callback: WatchCallback) -> crate::Result<WatchHandle>;

    /// Allows the directory to list managed files, overriding the ManagedDirectory's default
    /// list_managed_files
    fn list_managed_files(&self) -> crate::Result<HashSet<PathBuf>> {
        Err(crate::TantivyError::InternalError(
            "list_managed_files not implemented".to_string(),
        ))
    }

    /// Allows the directory to register a file as managed, overriding the ManagedDirectory's
    /// default register_file_as_managed
    fn register_files_as_managed(
        &self,
        _files: Vec<PathBuf>,
        _overwrite: bool,
    ) -> crate::Result<()> {
        Err(crate::TantivyError::InternalError(
            "register_files_as_managed not implemented".to_string(),
        ))
    }

    /// Allows the directory to save IndexMeta, overriding the SegmentUpdater's default save_meta
    fn save_metas(
        &self,
        _metas: &IndexMeta,
        _previous_metas: &IndexMeta,
        _payload: &mut (dyn Any + '_),
    ) -> crate::Result<()> {
        Err(crate::TantivyError::InternalError(
            "save_meta not implemented".to_string(),
        ))
    }

    /// Allows the directory to load IndexMeta, overriding the SegmentUpdater's default load_meta
    fn load_metas(&self, _inventory: &SegmentMetaInventory) -> crate::Result<IndexMeta> {
        Err(crate::TantivyError::InternalError(
            "load_metas not implemented".to_string(),
        ))
    }

    /// Returns true if this directory supports garbage collection.  The default assumption is
    /// `true`
    fn supports_garbage_collection(&self) -> bool {
        true
    }

    /// Return a panic handler to be assigned to the various thread pools that may be created
    ///
    /// The default is [`None`], which indicates that an unhandled panic from a thread pool will
    /// abort the process
    fn panic_handler(&self) -> Option<DirectoryPanicHandler> {
        None
    }

    /// Returns true if this directory is in a position of requiring that tantivy cancel
    /// whatever operation(s) it might be doing  Typically this is just for the background
    /// merge processes but could be used for anything
    fn wants_cancel(&self) -> bool {
        false
    }

    /// Send a logging message to the Directory to handle in its own way
    fn log(&self, message: &str) {
        log!(Level::Info, "{message}");
    }

    fn bufwriter_capacity(&self) -> usize {
        8192
    }
}

/// DirectoryClone
pub trait DirectoryClone {
    /// Clones the directory and boxes the clone
    fn box_clone(&self) -> Box<dyn Directory>;
}

impl<T> DirectoryClone for T
where T: 'static + Directory + Clone
{
    fn box_clone(&self) -> Box<dyn Directory> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Directory> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

impl<T: Directory + 'static> From<T> for Box<dyn Directory> {
    fn from(t: T) -> Self {
        Box::new(t)
    }
}

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::Path;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::Duration;

use super::*;

#[cfg(feature = "mmap")]
mod mmap_directory_tests {
    use crate::directory::MmapDirectory;

    type DirectoryImpl = MmapDirectory;

    fn make_directory() -> DirectoryImpl {
        MmapDirectory::create_from_tempdir().unwrap()
    }

    #[test]
    fn test_simple() -> crate::Result<()> {
        let directory = make_directory();
        super::test_simple(&directory)
    }

    #[test]
    fn test_write_create_the_file() {
        let directory = make_directory();
        super::test_write_create_the_file(&directory);
    }

    #[test]
    fn test_rewrite_forbidden() -> crate::Result<()> {
        let directory = make_directory();
        super::test_rewrite_forbidden(&directory)?;
        Ok(())
    }

    #[test]
    fn test_directory_delete() -> crate::Result<()> {
        let directory = make_directory();
        super::test_directory_delete(&directory)?;
        Ok(())
    }

    #[test]
    fn test_lock_non_blocking() {
        let directory = make_directory();
        super::test_lock_non_blocking(&directory);
    }

    #[test]
    fn test_lock_blocking() {
        let directory = make_directory();
        super::test_lock_blocking(&directory);
    }

    #[test]
    fn test_watch() {
        let directory = make_directory();
        super::test_watch(&directory);
    }

    #[test]
    fn test_temp_file() -> std::io::Result<()> {
        let directory = make_directory();
        super::test_temp_file(&directory)
    }
}

mod ram_directory_tests {
    use crate::directory::RamDirectory;

    type DirectoryImpl = RamDirectory;

    fn make_directory() -> DirectoryImpl {
        RamDirectory::default()
    }

    #[test]
    fn test_simple() -> crate::Result<()> {
        let directory = make_directory();
        super::test_simple(&directory)
    }

    #[test]
    fn test_write_create_the_file() {
        let directory = make_directory();
        super::test_write_create_the_file(&directory);
    }

    #[test]
    fn test_rewrite_forbidden() -> crate::Result<()> {
        let directory = make_directory();
        super::test_rewrite_forbidden(&directory)?;
        Ok(())
    }

    #[test]
    fn test_directory_delete() -> crate::Result<()> {
        let directory = make_directory();
        super::test_directory_delete(&directory)?;
        Ok(())
    }

    #[test]
    fn test_lock_non_blocking() {
        let directory = make_directory();
        super::test_lock_non_blocking(&directory);
    }

    #[test]
    fn test_lock_blocking() {
        let directory = make_directory();
        super::test_lock_blocking(&directory);
    }

    #[test]
    fn test_watch() {
        let directory = make_directory();
        super::test_watch(&directory);
    }

    #[test]
    fn test_temp_file() -> std::io::Result<()> {
        let directory = make_directory();
        super::test_temp_file(&directory)
    }
}

mod managed_directory_tests {
    use crate::directory::{ManagedDirectory, RamDirectory};

    #[test]
    fn test_temp_file_delegation() -> std::io::Result<()> {
        let directory = ManagedDirectory::wrap(Box::new(RamDirectory::default())).unwrap();
        super::test_temp_file(&directory)
    }
}

#[derive(Clone, Debug)]
struct DirectoryWithoutTempFile;

impl Directory for DirectoryWithoutTempFile {
    fn get_file_handle(&self, _path: &Path) -> Result<Arc<dyn FileHandle>, error::OpenReadError> {
        unimplemented!()
    }

    fn delete(&self, _path: &Path) -> Result<(), error::DeleteError> {
        unimplemented!()
    }

    fn exists(&self, _path: &Path) -> Result<bool, error::OpenReadError> {
        unimplemented!()
    }

    fn open_write_inner(&self, _path: &Path) -> Result<InnerWritePtr, error::OpenWriteError> {
        unimplemented!()
    }

    fn atomic_read(&self, _path: &Path) -> Result<Vec<u8>, error::OpenReadError> {
        unimplemented!()
    }

    fn atomic_write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        unimplemented!()
    }

    fn sync_directory(&self) -> io::Result<()> {
        unimplemented!()
    }

    fn watch(&self, _watch_callback: WatchCallback) -> crate::Result<WatchHandle> {
        unimplemented!()
    }
}

#[test]
fn test_temp_file_default_is_unsupported() {
    let error = DirectoryWithoutTempFile.open_temp_file().err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::Unsupported);
}

fn test_temp_file(directory: &dyn Directory) -> io::Result<()> {
    let mut temp_file = directory.open_temp_file()?;
    temp_file.write_all(b"spill-data")?;
    assert_eq!(temp_file.seek(SeekFrom::Start(0))?, 0);
    let mut data = Vec::new();
    temp_file.read_to_end(&mut data)?;
    assert_eq!(data, b"spill-data");
    Ok(())
}

fn test_simple(directory: &dyn Directory) -> crate::Result<()> {
    let test_path: &'static Path = Path::new("some_path_for_test");
    let mut write_file = directory.open_write(test_path)?;
    assert!(directory.exists(test_path).unwrap());
    write_file.write_all(&[4])?;
    write_file.write_all(&[3])?;
    write_file.write_all(&[7, 3, 5])?;
    write_file.flush()?;
    let read_file = directory.open_read(test_path)?.read_bytes()?;
    assert_eq!(read_file.as_slice(), &[4u8, 3u8, 7u8, 3u8, 5u8]);
    mem::drop(read_file);
    assert!(directory.delete(test_path).is_ok());
    assert!(!directory.exists(test_path).unwrap());
    Ok(())
}

fn test_rewrite_forbidden(directory: &dyn Directory) -> crate::Result<()> {
    let test_path: &'static Path = Path::new("some_path_for_test");
    directory.open_write(test_path)?;
    assert!(directory.exists(test_path).unwrap());
    assert!(directory.open_write(test_path).is_err());
    assert!(directory.delete(test_path).is_ok());
    Ok(())
}

fn test_write_create_the_file(directory: &dyn Directory) {
    let test_path: &'static Path = Path::new("some_path_for_test");
    {
        assert!(directory.open_read(test_path).is_err());
        let _w = directory.open_write(test_path).unwrap();
        assert!(directory.exists(test_path).unwrap());
        assert!(directory.open_read(test_path).is_ok());
        assert!(directory.delete(test_path).is_ok());
    }
}

fn test_directory_delete(directory: &dyn Directory) -> crate::Result<()> {
    let test_path: &'static Path = Path::new("some_path_for_test");
    assert!(directory.open_read(test_path).is_err());
    let mut write_file = directory.open_write(test_path)?;
    write_file.write_all(&[1, 2, 3, 4])?;
    write_file.flush()?;
    {
        let read_handle = directory.open_read(test_path)?.read_bytes()?;
        assert_eq!(read_handle.as_slice(), &[1u8, 2u8, 3u8, 4u8]);
        // Mapped files can't be deleted on Windows
        if !cfg!(windows) {
            assert!(directory.delete(test_path).is_ok());
            assert_eq!(read_handle.as_slice(), &[1u8, 2u8, 3u8, 4u8]);
        }
        assert!(directory.delete(Path::new("SomeOtherPath")).is_err());
    }

    if cfg!(windows) {
        assert!(directory.delete(test_path).is_ok());
    }

    assert!(directory.open_read(test_path).is_err());
    assert!(directory.delete(test_path).is_err());
    Ok(())
}

fn test_watch(directory: &dyn Directory) {
    let counter: Arc<AtomicUsize> = Default::default();
    let (tx, rx) = crossbeam_channel::unbounded();
    let timeout = Duration::from_millis(500);

    let handle = directory
        .watch(WatchCallback::new(move || {
            let val = counter.fetch_add(1, SeqCst);
            tx.send(val + 1).unwrap();
        }))
        .unwrap();

    assert!(directory
        .atomic_write(Path::new("meta.json"), b"foo")
        .is_ok());
    assert_eq!(rx.recv_timeout(timeout), Ok(1));

    assert!(directory
        .atomic_write(Path::new("meta.json"), b"bar")
        .is_ok());
    assert_eq!(rx.recv_timeout(timeout), Ok(2));

    mem::drop(handle);

    assert!(directory
        .atomic_write(Path::new("meta.json"), b"qux")
        .is_ok());
    assert!(rx.recv_timeout(timeout).is_err());
}

fn test_lock_non_blocking(directory: &dyn Directory) {
    {
        let lock_a_res = directory.acquire_lock(&Lock {
            filepath: PathBuf::from("a.lock"),
            is_blocking: false,
        });
        assert!(lock_a_res.is_ok());
        let lock_b_res = directory.acquire_lock(&Lock {
            filepath: PathBuf::from("b.lock"),
            is_blocking: false,
        });
        assert!(lock_b_res.is_ok());
        let lock_a_res2 = directory.acquire_lock(&Lock {
            filepath: PathBuf::from("a.lock"),
            is_blocking: false,
        });
        assert!(lock_a_res2.is_err());
    }
    let lock_a_res = directory.acquire_lock(&Lock {
        filepath: PathBuf::from("a.lock"),
        is_blocking: false,
    });
    assert!(lock_a_res.is_ok());
}

fn test_lock_blocking(directory: &dyn Directory) {
    let lock_a_res = directory.acquire_lock(&Lock {
        filepath: PathBuf::from("a.lock"),
        is_blocking: true,
    });
    assert!(lock_a_res.is_ok());
    let in_thread = Arc::new(AtomicBool::default());
    let in_thread_clone = in_thread.clone();
    let (sender, receiver) = oneshot::channel();
    std::thread::spawn(move || {
        //< lock_a_res is sent to the thread.
        in_thread_clone.store(true, SeqCst);
        let _just_sync = receiver.recv();
        // explicitly dropping lock_a_res. It would have been sufficient to just force it
        // to be part of the move, but the intent seems clearer that way.
        drop(lock_a_res);
    });
    {
        // A non-blocking call should fail, as the thread is running and holding the lock.
        let lock_a_res = directory.acquire_lock(&Lock {
            filepath: PathBuf::from("a.lock"),
            is_blocking: false,
        });
        assert!(lock_a_res.is_err());
    }
    let directory_clone = directory.box_clone();
    let (sender2, receiver2) = oneshot::channel();
    let join_handle = std::thread::spawn(move || {
        assert!(sender2.send(()).is_ok());
        let lock_a_res = directory_clone.acquire_lock(&Lock {
            filepath: PathBuf::from("a.lock"),
            is_blocking: true,
        });
        assert!(in_thread.load(SeqCst));
        assert!(lock_a_res.is_ok());
    });
    assert!(receiver2.recv().is_ok());
    assert!(sender.send(()).is_ok());
    assert!(join_handle.join().is_ok());
}

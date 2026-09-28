use std::ops::{Deref, Range};
use std::sync::{Arc, LazyLock};
use std::{fmt, io};

pub use stable_deref_trait::StableDeref;

type BoxedDeref = Box<dyn Deref<Target = [u8]> + Send + Sync>;
type BoxedLoader = Box<dyn FnOnce() -> (BoxedDeref, &'static [u8]) + Send>;

/// An OwnedBytes simply wraps an object that owns a slice of data and exposes
/// this data as a slice.
///
/// The backing object is required to be `StableDeref`.
#[derive(Clone)]
pub struct OwnedBytes {
    inner: OwnedBytesInner,
}

#[derive(Clone)]
enum OwnedBytesInner {
    Eager {
        data: &'static [u8],
        box_stable_deref: Arc<dyn Deref<Target = [u8]> + Sync + Send>,
    },
    Lazy {
        range: Range<usize>,
        source: Arc<LazySource>,
    },
}

struct LazySource {
    inner: LazyLock<(BoxedDeref, &'static [u8]), BoxedLoader>,
}

impl LazySource {
    #[inline]
    fn get_slice(&self) -> &'static [u8] {
        self.inner.deref().1
    }
}

impl Deref for LazySource {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.get_slice()
    }
}

impl OwnedBytes {
    /// Creates an empty `OwnedBytes`.
    pub fn empty() -> OwnedBytes {
        OwnedBytes::new(&[][..])
    }

    /// Creates an `OwnedBytes` instance given a `StableDeref` object.
    pub fn new<T: StableDeref + Deref<Target = [u8]> + 'static + Send + Sync>(
        data_holder: T,
    ) -> OwnedBytes {
        let box_stable_deref = Arc::new(data_holder);
        let bytes: &[u8] = box_stable_deref.deref();
        let data = unsafe { &*(bytes as *const [u8]) };
        OwnedBytes {
            inner: OwnedBytesInner::Eager {
                data,
                box_stable_deref,
            },
        }
    }

    /// Creates a lazy `OwnedBytes` instance that will only invoke `loader()`
    /// when the underlying bytes are accessed for the first time.
    ///
    /// The `len` parameter must equal the length of the slice expected to be returned.
    pub fn new_lazy<T, F>(len: usize, loader: F) -> OwnedBytes
    where
        F: FnOnce() -> T + Send + 'static,
        T: StableDeref + Deref<Target = [u8]> + Send + Sync + 'static,
    {
        let source = Arc::new(LazySource {
            inner: LazyLock::new(Box::new(move || {
                let boxed: BoxedDeref = Box::new(loader());
                let bytes: &[u8] = boxed.deref();
                assert!(
                    bytes.len() >= len,
                    "lazy loader returned slice of unexpected length (expected at least {}, got \
                     {})",
                    len,
                    bytes.len(),
                );
                let slice = &bytes[..len];
                let slice: &'static [u8] = unsafe { &*(slice as *const [u8]) };
                (boxed, slice)
            })),
        });
        OwnedBytes {
            inner: OwnedBytesInner::Lazy {
                range: 0..len,
                source,
            },
        }
    }

    /// creates a fileslice that is just a view over a slice of the data.
    #[must_use]
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> Self {
        match &self.inner {
            OwnedBytesInner::Eager {
                data,
                box_stable_deref,
            } => OwnedBytes {
                inner: OwnedBytesInner::Eager {
                    data: &data[range],
                    box_stable_deref: box_stable_deref.clone(),
                },
            },
            OwnedBytesInner::Lazy {
                range: cur_range,
                source,
            } => {
                assert!(
                    range.start <= range.end,
                    "slice index starts at {} but ends at {}",
                    range.start,
                    range.end
                );
                assert!(
                    range.end <= cur_range.len(),
                    "range end {} exceeds slice length {}",
                    range.end,
                    cur_range.len()
                );
                let new_start = cur_range.start + range.start;
                let new_end = cur_range.start + range.end;
                OwnedBytes {
                    inner: OwnedBytesInner::Lazy {
                        range: new_start..new_end,
                        source: source.clone(),
                    },
                }
            }
        }
    }

    /// Returns the underlying slice of data.
    /// `Deref` and `AsRef` are also available.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            OwnedBytesInner::Eager { data, .. } => data,
            OwnedBytesInner::Lazy { range, source } => {
                let full = source.get_slice();
                &full[range.clone()]
            }
        }
    }

    /// Returns the len of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.inner {
            OwnedBytesInner::Eager { data, .. } => data.len(),
            OwnedBytesInner::Lazy { range, .. } => range.len(),
        }
    }

    /// Returns true iff this `OwnedBytes` is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Splits the OwnedBytes into two OwnedBytes `(left, right)`.
    ///
    /// Left will hold `split_len` bytes.
    ///
    /// This operation is cheap and does not require to copy any memory.
    /// On the other hand, both `left` and `right` retain a handle over
    /// the entire slice of memory. In other words, the memory will only
    /// be released when both left and right are dropped.
    #[inline]
    #[must_use]
    pub fn split(self, split_len: usize) -> (OwnedBytes, OwnedBytes) {
        match self.inner {
            OwnedBytesInner::Eager {
                data,
                box_stable_deref,
            } => {
                let (left_data, right_data) = data.split_at(split_len);
                let right_box_stable_deref = box_stable_deref.clone();
                let left = OwnedBytes {
                    inner: OwnedBytesInner::Eager {
                        data: left_data,
                        box_stable_deref,
                    },
                };
                let right = OwnedBytes {
                    inner: OwnedBytesInner::Eager {
                        data: right_data,
                        box_stable_deref: right_box_stable_deref,
                    },
                };
                (left, right)
            }
            OwnedBytesInner::Lazy { range, source } => {
                assert!(
                    split_len <= range.len(),
                    "split_len {} exceeds slice length {}",
                    split_len,
                    range.len()
                );
                let mid = range.start + split_len;
                let left = OwnedBytes {
                    inner: OwnedBytesInner::Lazy {
                        range: range.start..mid,
                        source: source.clone(),
                    },
                };
                let right = OwnedBytes {
                    inner: OwnedBytesInner::Lazy {
                        range: mid..range.end,
                        source,
                    },
                };
                (left, right)
            }
        }
    }

    /// Splits the OwnedBytes into two OwnedBytes `(left, right)`.
    ///
    /// Right will hold `split_len` bytes.
    ///
    /// This operation is cheap and does not require to copy any memory.
    /// On the other hand, both `left` and `right` retain a handle over
    /// the entire slice of memory. In other words, the memory will only
    /// be released when both left and right are dropped.
    #[inline]
    #[must_use]
    pub fn rsplit(self, split_len: usize) -> (OwnedBytes, OwnedBytes) {
        let data_len = self.len();
        assert!(
            split_len <= data_len,
            "split_len {} exceeds slice length {}",
            split_len,
            data_len
        );
        self.split(data_len - split_len)
    }

    /// Splits the right part of the `OwnedBytes` at the given offset.
    ///
    /// `self` is truncated to `split_len`, left with the remaining bytes.
    pub fn split_off(&mut self, split_len: usize) -> OwnedBytes {
        match &mut self.inner {
            OwnedBytesInner::Eager {
                data,
                box_stable_deref,
            } => {
                let (left, right) = data.split_at(split_len);
                let right_box_stable_deref = box_stable_deref.clone();
                let right_piece = OwnedBytes {
                    inner: OwnedBytesInner::Eager {
                        data: right,
                        box_stable_deref: right_box_stable_deref,
                    },
                };
                *data = left;
                right_piece
            }
            OwnedBytesInner::Lazy { range, source } => {
                assert!(
                    split_len <= range.len(),
                    "split_len {} exceeds slice length {}",
                    split_len,
                    range.len()
                );
                let mid = range.start + split_len;
                let right_piece = OwnedBytes {
                    inner: OwnedBytesInner::Lazy {
                        range: mid..range.end,
                        source: source.clone(),
                    },
                };
                *range = range.start..mid;
                right_piece
            }
        }
    }

    /// Drops the left most `advance_len` bytes.
    #[inline]
    pub fn advance(&mut self, advance_len: usize) -> &[u8] {
        if advance_len == 0 {
            return &[];
        }
        match &mut self.inner {
            OwnedBytesInner::Eager { data, .. } => {
                let (head, rest) = data.split_at(advance_len);
                *data = rest;
                head
            }
            OwnedBytesInner::Lazy { range, source } => {
                assert!(
                    advance_len <= range.len(),
                    "advance_len {} exceeds slice length {}",
                    advance_len,
                    range.len()
                );
                let full = source.get_slice();
                let start = range.start;
                let mid = start + advance_len;
                let end = range.end;
                let head = &full[start..mid];
                let rest = &full[mid..end];
                let box_stable_deref: Arc<dyn Deref<Target = [u8]> + Sync + Send> = source.clone();
                self.inner = OwnedBytesInner::Eager {
                    data: rest,
                    box_stable_deref,
                };
                head
            }
        }
    }

    /// Reads an `u8` from the `OwnedBytes` and advance by one byte.
    #[inline]
    pub fn read_u8(&mut self) -> u8 {
        self.advance(1)[0]
    }

    #[inline]
    fn read_n<const N: usize>(&mut self) -> [u8; N] {
        self.advance(N).try_into().unwrap()
    }

    /// Reads an `u32` encoded as little-endian from the `OwnedBytes` and advance by 4 bytes.
    #[inline]
    pub fn read_u32(&mut self) -> u32 {
        u32::from_le_bytes(self.read_n())
    }

    /// Reads an `u64` encoded as little-endian from the `OwnedBytes` and advance by 8 bytes.
    #[inline]
    pub fn read_u64(&mut self) -> u64 {
        u64::from_le_bytes(self.read_n())
    }
}

impl fmt::Debug for OwnedBytes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // We truncate the bytes in order to make sure the debug string
        // is not too long.
        let bytes_truncated: &[u8] = if self.len() > 10 {
            &self.as_slice()[..10]
        } else {
            self.as_slice()
        };
        write!(f, "OwnedBytes({bytes_truncated:?}, len={})", self.len())
    }
}

impl PartialEq for OwnedBytes {
    fn eq(&self, other: &OwnedBytes) -> bool {
        self.len() == other.len() && self.as_slice() == other.as_slice()
    }
}

impl Eq for OwnedBytes {}

impl PartialEq<[u8]> for OwnedBytes {
    fn eq(&self, other: &[u8]) -> bool {
        self.len() == other.len() && self.as_slice() == other
    }
}

impl PartialEq<str> for OwnedBytes {
    fn eq(&self, other: &str) -> bool {
        self.len() == other.len() && self.as_slice() == other.as_bytes()
    }
}

impl<'a, T: ?Sized> PartialEq<&'a T> for OwnedBytes
where OwnedBytes: PartialEq<T>
{
    fn eq(&self, other: &&'a T) -> bool {
        *self == **other
    }
}

impl Deref for OwnedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for OwnedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl io::Read for OwnedBytes {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let to_read = self.len().min(buf.len());
        if to_read == 0 {
            return Ok(0);
        }
        let data = self.advance(to_read);
        buf[..to_read].copy_from_slice(data);
        Ok(to_read)
    }

    #[inline]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let read_len = self.len();
        if read_len == 0 {
            return Ok(0);
        }
        let data = self.advance(read_len);
        buf.extend_from_slice(data);
        Ok(read_len)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let read_len = self.read(buf)?;
        if read_len != buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Read};

    use super::OwnedBytes;

    #[test]
    fn test_owned_bytes_debug() {
        let short_bytes = OwnedBytes::new(b"abcd".as_ref());
        assert_eq!(
            format!("{short_bytes:?}"),
            "OwnedBytes([97, 98, 99, 100], len=4)"
        );
        let medium_bytes = OwnedBytes::new(b"abcdefghi".as_ref());
        assert_eq!(
            format!("{medium_bytes:?}"),
            "OwnedBytes([97, 98, 99, 100, 101, 102, 103, 104, 105], len=9)"
        );
        let long_bytes = OwnedBytes::new(b"abcdefghijklmnopq".as_ref());
        assert_eq!(
            format!("{long_bytes:?}"),
            "OwnedBytes([97, 98, 99, 100, 101, 102, 103, 104, 105, 106], len=17)"
        );
    }

    #[test]
    fn test_owned_bytes_read() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"abcdefghiklmnopqrstuvwxyz".as_ref());
        {
            let mut buf = [0u8; 5];
            bytes.read_exact(&mut buf[..]).unwrap();
            assert_eq!(&buf, b"abcde");
            assert_eq!(bytes.as_slice(), b"fghiklmnopqrstuvwxyz")
        }
        {
            let mut buf = [0u8; 2];
            bytes.read_exact(&mut buf[..]).unwrap();
            assert_eq!(&buf, b"fg");
            assert_eq!(bytes.as_slice(), b"hiklmnopqrstuvwxyz")
        }
        Ok(())
    }

    #[test]
    fn test_owned_bytes_read_right_at_the_end() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"abcde".as_ref());
        let mut buf = [0u8; 5];
        assert_eq!(bytes.read(&mut buf[..]).unwrap(), 5);
        assert_eq!(&buf, b"abcde");
        assert_eq!(bytes.as_slice(), b"");
        assert_eq!(bytes.read(&mut buf[..]).unwrap(), 0);
        assert_eq!(&buf, b"abcde");
        Ok(())
    }
    #[test]
    fn test_owned_bytes_read_incomplete() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"abcde".as_ref());
        let mut buf = [0u8; 7];
        assert_eq!(bytes.read(&mut buf[..]).unwrap(), 5);
        assert_eq!(&buf[..5], b"abcde");
        assert_eq!(bytes.read(&mut buf[..]).unwrap(), 0);
        Ok(())
    }

    #[test]
    fn test_owned_bytes_read_to_end() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"abcde".as_ref());
        let mut buf = Vec::new();
        bytes.read_to_end(&mut buf)?;
        assert_eq!(buf.as_slice(), b"abcde".as_ref());
        Ok(())
    }

    #[test]
    fn test_owned_bytes_read_u8() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"\xFF".as_ref());
        assert_eq!(bytes.read_u8(), 255);
        assert_eq!(bytes.len(), 0);
        Ok(())
    }

    #[test]
    fn test_owned_bytes_read_u64() -> io::Result<()> {
        let mut bytes = OwnedBytes::new(b"\0\xFF\xFF\xFF\xFF\xFF\xFF\xFF".as_ref());
        assert_eq!(bytes.read_u64(), u64::MAX - 255);
        assert_eq!(bytes.len(), 0);
        Ok(())
    }

    #[test]
    fn test_owned_bytes_split() {
        let bytes = OwnedBytes::new(b"abcdefghi".as_ref());
        let (left, right) = bytes.split(3);
        assert_eq!(left.as_slice(), b"abc");
        assert_eq!(right.as_slice(), b"defghi");
    }

    #[test]
    fn test_owned_bytes_split_boundary() {
        let bytes = OwnedBytes::new(b"abcdefghi".as_ref());
        {
            let (left, right) = bytes.clone().split(0);
            assert_eq!(left.as_slice(), b"");
            assert_eq!(right.as_slice(), b"abcdefghi");
        }
        {
            let (left, right) = bytes.split(9);
            assert_eq!(left.as_slice(), b"abcdefghi");
            assert_eq!(right.as_slice(), b"");
        }
    }

    #[test]
    fn test_split_off() {
        let mut data = OwnedBytes::new(b"abcdef".as_ref());
        assert_eq!(data, "abcdef");
        assert_eq!(data.split_off(2), "cdef");
        assert_eq!(data, "ab");
        assert_eq!(data.split_off(1), "b");
        assert_eq!(data, "a");
    }

    struct CountingBytes {
        data: Vec<u8>,
        eval_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl std::ops::Deref for CountingBytes {
        type Target = [u8];
        fn deref(&self) -> &Self::Target {
            self.eval_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            &self.data
        }
    }

    #[test]
    fn test_lazy_owned_bytes_deferred_eval() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let eval_count = Arc::new(AtomicUsize::new(0));
        let holder = CountingBytes {
            data: b"hello world".to_vec(),
            eval_count: eval_count.clone(),
        };

        let lazy = OwnedBytes::new_lazy(11, move || holder);
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);
        assert_eq!(lazy.len(), 11);
        assert!(!lazy.is_empty());
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);

        // Slice without triggering eval
        let sub = lazy.slice(0..5);
        assert_eq!(sub.len(), 5);
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);

        // Sub-slice of sub-slice
        let sub_sub = sub.slice(1..4);
        assert_eq!(sub_sub.len(), 3);
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);

        // Now deref sub_sub: should trigger exactly once
        assert_eq!(sub_sub.as_slice(), b"ell");
        assert_eq!(eval_count.load(Ordering::SeqCst), 1);

        // Accessing the parent slices should NOT trigger another eval
        assert_eq!(sub.as_slice(), b"hello");
        assert_eq!(lazy.as_slice(), b"hello world");
        assert_eq!(eval_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_lazy_owned_bytes_split_and_advance() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let eval_count = Arc::new(AtomicUsize::new(0));
        let holder = CountingBytes {
            data: b"abcdefghi".to_vec(),
            eval_count: eval_count.clone(),
        };

        let mut lazy = OwnedBytes::new_lazy(9, move || holder);
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);

        let (left, right) = lazy.clone().split(3);
        assert_eq!(eval_count.load(Ordering::SeqCst), 0);
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 6);

        let advanced = lazy.advance(2);
        assert_eq!(advanced, b"ab");
        assert_eq!(eval_count.load(Ordering::SeqCst), 1);
        assert_eq!(lazy.len(), 7);
        assert_eq!(lazy.as_slice(), b"cdefghi");
        assert_eq!(eval_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[should_panic(expected = "lazy loader returned slice of unexpected length")]
    fn test_lazy_owned_bytes_length_mismatch() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let holder = CountingBytes {
            data: b"short".to_vec(),
            eval_count: Arc::new(AtomicUsize::new(0)),
        };
        // Claim length is 10, but data is only 5 bytes
        let lazy = OwnedBytes::new_lazy(10, move || holder);
        let _ = lazy.as_slice();
    }
}

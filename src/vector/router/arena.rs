//! Centroid arenas: dense `dim`-strided vector storage the routers score against.

use crate::schema::Metric;
use crate::vector::{Similarity, VectorElement};

/// A flat, `dim`-strided arena of vectors addressed by dense row index.
///
/// The arena owns its representation — typed slices via the blanket impl,
/// raw little-endian file bytes for reloaded storage — and scores a typed
/// query against a stored vector with the kernel matching that
/// representation. The [`Metric`] is a parameter: an arena never holds one.
pub trait VectorArena {
    /// Element type of the vectors; queries are `&[Elem]`.
    type Elem: VectorElement;

    /// The number of vectors held, at `dim` elements each.
    fn num_vectors(&self, dim: usize) -> usize;

    /// [`Similarity`] of `query` to the vector at dense row `index`.
    fn similarity(
        &self,
        metric: Metric,
        dim: usize,
        index: u32,
        query: &[Self::Elem],
    ) -> Similarity;
}

/// Any `[T]`-shaped storage (`&[T]`, `Vec<T>`, …), scored with the typed kernels.
impl<T: VectorElement, S: std::ops::Deref<Target = [T]>> VectorArena for S {
    type Elem = T;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        assert_eq!(self.len() % dim, 0, "arena not a multiple of dim");
        self.len() / dim
    }

    #[inline]
    fn similarity(&self, metric: Metric, dim: usize, index: u32, query: &[T]) -> Similarity {
        metric.similarity(query, &self[index as usize * dim..][..dim])
    }
}

/// A [`VectorArena`] over raw little-endian `T` rows behind a
/// [`FileSlice`](crate::directory::FileSlice): each [`similarity`] call reads only that row, one
/// stride-sized ranged read scored with the byte kernels ([`Metric::similarity_bytes`]).
///
/// [`similarity`]: VectorArena::similarity
pub struct FileSliceArena<T> {
    pub(crate) slice: crate::directory::FileSlice,
    _elem: std::marker::PhantomData<T>,
}

impl<T> FileSliceArena<T> {
    /// Wraps a slice of contiguous `dim`-strided little-endian `T` rows.
    pub fn new(slice: crate::directory::FileSlice) -> Self {
        FileSliceArena {
            slice,
            _elem: std::marker::PhantomData,
        }
    }
}

impl<T: VectorElement> FileSliceArena<T> {
    /// Row `index`'s little-endian bytes, one stride-sized ranged read.
    pub(crate) fn row_bytes(&self, dim: usize, index: u32) -> std::io::Result<common::OwnedBytes> {
        let stride = dim * T::SIZE_BYTES;
        self.slice
            .slice(index as usize * stride..(index as usize + 1) * stride)
            .read_bytes()
    }
}

impl<T: VectorElement> VectorArena for FileSliceArena<T> {
    type Elem = T;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        use common::HasLen;
        let stride = dim * T::SIZE_BYTES;
        assert_eq!(self.slice.len() % stride, 0, "arena not a multiple of dim");
        self.slice.len() / stride
    }

    /// # Panics
    ///
    /// The trait has no error channel, so a failed read panics. Callers must
    /// pass in-range row indices; an I/O failure here means the underlying
    /// `Directory` could not produce bytes it already promised via the slice.
    #[inline]
    fn similarity(&self, metric: Metric, dim: usize, index: u32, query: &[T]) -> Similarity {
        let bytes = self
            .row_bytes(dim, index)
            .expect("failed to read vector arena row");
        metric.similarity_bytes(query, &bytes)
    }
}

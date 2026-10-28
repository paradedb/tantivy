//! Centroid arenas: dense `dim`-strided vector storage the routers score against.

use std::ops::Range;

use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::VectorElement;

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

    /// Replaces `out` with similarities for `rows`, in row order.
    fn batch_similarity(
        &self,
        query: &[Self::Elem],
        metric: Metric,
        rows: Range<usize>,
        out: &mut Vec<Similarity>,
    ) {
        out.clear();
        out.extend(rows.map(|row| self.similarity(metric, query.len(), row as u32, query)));
    }
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
pub(crate) struct FileSliceArena<T> {
    pub(crate) slice: crate::directory::FileSlice,
    _elem: std::marker::PhantomData<T>,
}

impl<T> FileSliceArena<T> {
    /// Wraps a slice of contiguous `dim`-strided little-endian `T` rows.
    pub(crate) fn new(slice: crate::directory::FileSlice) -> Self {
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

    fn batch_similarity(
        &self,
        query: &[T],
        metric: Metric,
        rows: Range<usize>,
        out: &mut Vec<Similarity>,
    ) {
        out.clear();
        if rows.is_empty() {
            return;
        }
        let stride = query.len() * T::SIZE_BYTES;
        let bytes = self
            .slice
            .slice(rows.start * stride..rows.end * stride)
            .read_bytes()
            .expect("failed to read vector arena rows");
        out.extend(
            bytes
                .chunks_exact(stride)
                .map(|row| metric.similarity_bytes(query, row)),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use common::{HasLen, OwnedBytes};

    use super::*;
    use crate::directory::{FileHandle, FileSlice};

    #[derive(Debug)]
    struct TrackedFile {
        bytes: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
    }

    impl HasLen for TrackedFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for TrackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }
    }

    #[test]
    fn batch_scores_match_single_row_scores() {
        for dim in [1, 7, 16, 37, 1024] {
            let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 8.5) * 0.13).collect();
            let row: Vec<f32> = (0..dim).map(|i| (i as f32 + 2.5) * -0.07).collect();
            let zero = vec![0.0; dim];
            let rows = [query.as_slice(), row.as_slice(), zero.as_slice()].concat();
            let bytes = rows
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let file = FileSliceArena::<f32>::new(FileSlice::from(bytes));
            let mut typed_scores = Vec::new();
            let mut file_scores = Vec::new();
            for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
                for query in [&query, &zero] {
                    for range in [0..3, 1..3, 2..3, 3..3] {
                        let expected: Vec<_> = range
                            .clone()
                            .map(|row| rows.similarity(metric, dim, row as u32, query))
                            .collect();
                        let expected_file: Vec<_> = range
                            .clone()
                            .map(|row| file.similarity(metric, dim, row as u32, query))
                            .collect();
                        rows.batch_similarity(query, metric, range.clone(), &mut typed_scores);
                        file.batch_similarity(query, metric, range, &mut file_scores);
                        assert_eq!(typed_scores, expected);
                        assert_eq!(file_scores, expected_file);
                    }
                }
            }
        }
    }

    #[test]
    fn file_batch_reads_only_the_requested_rows_once() {
        const DIM: usize = 1024;
        let rows: Vec<f32> = (0..150 * DIM).map(|i| (i % 53) as f32 * 0.02).collect();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let arena = FileSliceArena::<f32>::new(FileSlice::new(Arc::new(TrackedFile {
            bytes: rows.iter().flat_map(|v| v.to_le_bytes()).collect(),
            reads: reads.clone(),
        })));
        let query = &rows[..DIM];
        let mut actual = Vec::new();
        arena.batch_similarity(query, Metric::Cosine, 3..140, &mut actual);
        let mut expected = Vec::new();
        rows.batch_similarity(query, Metric::Cosine, 3..140, &mut expected);
        assert_eq!(actual, expected);
        assert_eq!(*reads.lock().unwrap(), vec![3 * 4096..140 * 4096]);
        reads.lock().unwrap().clear();
        arena.batch_similarity(query, Metric::Cosine, 150..150, &mut actual);
        assert!(actual.is_empty());
        assert!(reads.lock().unwrap().is_empty());
    }
}

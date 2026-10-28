//! Centroid arenas: dense `dim`-strided vector storage the routers score against.

use std::ops::Range;

use crate::schema::Metric;
use crate::vector::distance::{dot, dot_bytes, norm_squared, norm_squared_bytes, Similarity};
use crate::vector::VectorElement;

/// Router scoring state shared by every centroid and level in a search.
pub struct ArenaQuery<'a, T: VectorElement> {
    values: &'a [T],
    metric: Metric,
    cosine_norm: Option<f32>,
}

impl<'a, T: VectorElement> ArenaQuery<'a, T> {
    pub fn new(values: &'a [T], metric: Metric) -> Self {
        Self {
            values,
            metric,
            cosine_norm: (metric == Metric::Cosine).then(|| norm_squared(values).sqrt()),
        }
    }

    #[inline]
    fn similarity(&self, row: &[T]) -> Similarity {
        match self.cosine_norm {
            Some(query_norm) => {
                let row_norm = norm_squared(row).sqrt();
                Similarity::new(if query_norm == 0.0 || row_norm == 0.0 {
                    0.0
                } else {
                    dot(self.values, row) / (query_norm * row_norm)
                })
            }
            None => self.metric.similarity(self.values, row),
        }
    }

    #[inline]
    fn similarity_bytes(&self, row: &[u8]) -> Similarity {
        match self.cosine_norm {
            Some(query_norm) => {
                let row_norm = norm_squared_bytes::<T>(row).sqrt();
                Similarity::new(if query_norm == 0.0 || row_norm == 0.0 {
                    0.0
                } else {
                    dot_bytes(self.values, row) / (query_norm * row_norm)
                })
            }
            None => self.metric.similarity_bytes(self.values, row),
        }
    }
}

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

    /// Scores a contiguous range in row order, without reading outside that range.
    fn score_range(
        &self,
        query: &ArenaQuery<'_, Self::Elem>,
        range: Range<usize>,
        emit: impl FnMut(u32, Similarity),
    );
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
    fn score_range(
        &self,
        query: &ArenaQuery<'_, T>,
        range: Range<usize>,
        mut emit: impl FnMut(u32, Similarity),
    ) {
        let dim = query.values.len();
        for index in range {
            emit(index as u32, query.similarity(&self[index * dim..][..dim]));
        }
    }
}

/// A [`VectorArena`] over raw little-endian `T` rows behind a
/// [`FileSlice`](crate::directory::FileSlice), read in bounded contiguous batches.
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
    fn score_range(
        &self,
        query: &ArenaQuery<'_, T>,
        range: Range<usize>,
        mut emit: impl FnMut(u32, Similarity),
    ) {
        const BATCH_BYTES: usize = 256 * 1024;
        let stride = query.values.len() * T::SIZE_BYTES;
        let batch_rows = (BATCH_BYTES / stride).max(1);
        for start in range.clone().step_by(batch_rows) {
            let end = (start + batch_rows).min(range.end);
            let bytes = self
                .slice
                .slice(start * stride..end * stride)
                .read_bytes()
                .expect("failed to read vector arena rows");
            for (offset, row) in bytes.chunks_exact(stride).enumerate() {
                emit((start + offset) as u32, query.similarity_bytes(row));
            }
        }
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
    fn prepared_scores_match_scalar_metrics() {
        for dim in [1, 7, 16, 37, 1024] {
            let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 8.5) * 0.13).collect();
            let row: Vec<f32> = (0..dim).map(|i| (i as f32 + 2.5) * -0.07).collect();
            let zero = vec![0.0; dim];
            for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
                for query in [&query, &zero] {
                    let prepared = ArenaQuery::new(query, metric);
                    for row in [&row, &zero] {
                        let bytes: Vec<_> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                        assert_eq!(prepared.similarity(row), metric.similarity(query, row));
                        assert_eq!(
                            prepared.similarity_bytes(&bytes),
                            metric.similarity_bytes(query, &bytes)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn file_ranges_are_batched_bounded_and_score_the_same_rows() {
        const DIM: usize = 1024;
        let rows: Vec<f32> = (0..150 * DIM).map(|i| (i % 53) as f32 * 0.02).collect();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let arena = FileSliceArena::<f32>::new(FileSlice::new(Arc::new(TrackedFile {
            bytes: rows.iter().flat_map(|v| v.to_le_bytes()).collect(),
            reads: reads.clone(),
        })));
        let query = ArenaQuery::new(&rows[..DIM], Metric::Cosine);
        let mut actual = Vec::new();
        arena.score_range(&query, 3..140, |id, sim| actual.push((id, sim)));
        let mut expected = Vec::new();
        rows.score_range(&query, 3..140, |id, sim| expected.push((id, sim)));
        assert_eq!(actual, expected);
        assert_eq!(
            *reads.lock().unwrap(),
            vec![
                3 * 4096..67 * 4096,
                67 * 4096..131 * 4096,
                131 * 4096..140 * 4096
            ]
        );
        reads.lock().unwrap().clear();
        arena.score_range(&query, 150..150, |_, _| panic!("empty range scored"));
        assert!(reads.lock().unwrap().is_empty());
    }

    #[test]
    fn a_row_larger_than_the_batch_is_read_whole() {
        let row = vec![0.25f32; 70_000];
        let bytes = row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let arena = FileSliceArena::<f32>::new(FileSlice::from(bytes));
        let query = ArenaQuery::new(&row, Metric::L2);
        let mut actual = Vec::new();
        arena.score_range(&query, 0..1, |id, sim| actual.push((id, sim)));
        assert_eq!(actual, vec![(0, Similarity::new(0.0))]);
    }
}

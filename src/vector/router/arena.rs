//! Centroid arenas: dense `dim`-strided vector storage the routers score against.

use std::ops::Range;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use lru::LruCache;

use crate::index::SegmentId;
use crate::schema::{Field, Metric};
use crate::vector::distance::{dot_bytes, norm_squared, norm_squared_bytes, Similarity};
use crate::vector::VectorElement;

type NormKey = (SegmentId, Field, usize, usize);
type Norms = Arc<[OnceLock<f32>]>;

struct NormCache {
    entries: LruCache<NormKey, Norms>,
    slots: usize,
    capacity: usize,
}

impl NormCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: LruCache::unbounded(),
            slots: 0,
            capacity,
        }
    }

    fn get(&mut self, key: NormKey) -> Option<Norms> {
        let count = key.3;
        if count == 0 || count > self.capacity {
            return None;
        }
        if let Some(norms) = self.entries.get(&key) {
            return Some(Arc::clone(norms));
        }
        while self.slots + count > self.capacity || self.entries.len() >= 256 {
            let (_, evicted) = self.entries.pop_lru().unwrap();
            self.slots -= evicted.len();
        }
        let norms: Norms = (0..count).map(|_| OnceLock::new()).collect();
        self.slots += count;
        self.entries.put(key, Arc::clone(&norms));
        Some(norms)
    }
}

// Segment UUIDs identify immutable centroid rows. Retain only scalar norms, never file handles.
static CENTROID_NORMS: LazyLock<Mutex<NormCache>> = LazyLock::new(|| {
    Mutex::new(NormCache::new(
        32 * 1024 * 1024 / size_of::<OnceLock<f32>>(),
    ))
});

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

    /// Replaces `out` with similarities for `rows`, using the caller's cached query norm.
    fn batch_similarity(
        &self,
        query: &[Self::Elem],
        metric: Metric,
        query_norm: f32,
        rows: Range<usize>,
        out: &mut Vec<Similarity>,
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
    fn similarity(&self, metric: Metric, dim: usize, index: u32, query: &[T]) -> Similarity {
        metric.similarity(query, &self[index as usize * dim..][..dim])
    }

    fn batch_similarity(
        &self,
        query: &[T],
        metric: Metric,
        query_norm: f32,
        rows: Range<usize>,
        out: &mut Vec<Similarity>,
    ) {
        let dim = query.len();
        out.clear();
        out.extend(rows.map(|row| {
            metric.similarity_with_query_norm(query, &self[row * dim..][..dim], query_norm)
        }));
    }
}

/// A [`VectorArena`] over raw little-endian `T` rows behind a
/// [`FileSlice`](crate::directory::FileSlice): each [`similarity`] call reads only that row, one
/// stride-sized ranged read scored with the byte kernels ([`Metric::similarity_bytes`]).
///
/// [`similarity`]: VectorArena::similarity
pub(crate) struct FileSliceArena<T> {
    pub(crate) slice: crate::directory::FileSlice,
    norms: Option<Norms>,
    _elem: std::marker::PhantomData<T>,
}

impl<T> FileSliceArena<T> {
    /// Wraps a slice of contiguous `dim`-strided little-endian `T` rows.
    pub(crate) fn new(slice: crate::directory::FileSlice) -> Self {
        FileSliceArena {
            slice,
            norms: None,
            _elem: std::marker::PhantomData,
        }
    }
}

impl FileSliceArena<f32> {
    pub(crate) fn cache_norms(&mut self, segment: SegmentId, field: Field, dim: usize) {
        let count = self.num_vectors(dim);
        self.norms = CENTROID_NORMS
            .lock()
            .unwrap()
            .get((segment, field, dim, count));
    }
}

impl<T: VectorElement> FileSliceArena<T> {
    #[inline]
    fn score(
        &self,
        query: &[T],
        metric: Metric,
        query_norm: f32,
        index: usize,
        bytes: &[u8],
    ) -> Similarity {
        if metric == Metric::Cosine {
            if let Some(norms) = &self.norms {
                let norm = *norms[index].get_or_init(|| norm_squared_bytes::<T>(bytes).sqrt());
                return Similarity::new(if query_norm == 0.0 || norm == 0.0 {
                    0.0
                } else {
                    dot_bytes(query, bytes) / (query_norm * norm)
                });
            }
        }
        metric.similarity_bytes_with_query_norm(query, bytes, query_norm)
    }

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
        let query_norm = if metric == Metric::Cosine {
            norm_squared(query).sqrt()
        } else {
            0.0
        };
        self.score(query, metric, query_norm, index as usize, &bytes)
    }

    fn batch_similarity(
        &self,
        query: &[T],
        metric: Metric,
        query_norm: f32,
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
                .enumerate()
                .map(|(i, row)| self.score(query, metric, query_norm, rows.start + i, row)),
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
    use crate::vector::search::prepared::VectorQuery;

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
                    let prepared = VectorQuery::new(metric, Arc::new(query.clone()));
                    let norm = prepared.routing_norm();
                    for range in [0..3, 1..3, 2..3, 3..3] {
                        let expected: Vec<_> = range
                            .clone()
                            .map(|row| rows.similarity(metric, dim, row as u32, query))
                            .collect();
                        let expected_file: Vec<_> = range
                            .clone()
                            .map(|row| file.similarity(metric, dim, row as u32, query))
                            .collect();
                        rows.batch_similarity(
                            query,
                            metric,
                            norm,
                            range.clone(),
                            &mut typed_scores,
                        );
                        file.batch_similarity(query, metric, norm, range, &mut file_scores);
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
        let prepared = VectorQuery::new(Metric::Cosine, Arc::new(query.to_vec()));
        let norm = prepared.routing_norm();
        let mut actual = Vec::new();
        arena.batch_similarity(query, Metric::Cosine, norm, 3..140, &mut actual);
        let mut expected = Vec::new();
        rows.batch_similarity(query, Metric::Cosine, norm, 3..140, &mut expected);
        assert_eq!(actual, expected);
        assert_eq!(*reads.lock().unwrap(), vec![3 * 4096..140 * 4096]);
        reads.lock().unwrap().clear();
        arena.batch_similarity(query, Metric::Cosine, norm, 150..150, &mut actual);
        assert!(actual.is_empty());
        assert!(reads.lock().unwrap().is_empty());
    }

    #[test]
    fn cached_centroid_norms_preserve_scores_across_queries_and_reopens() {
        for dim in [1, 7, 16, 37, 1024] {
            let segment = SegmentId::generate_random();
            let field = Field::from_field_id(0);
            let values: Vec<f32> = [0.0, 1.0, 1e-19, 1e19]
                .into_iter()
                .flat_map(|scale| (0..dim).map(move |i| (i as f32 - 3.5) * scale))
                .collect();
            let slice = FileSlice::from(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let control = FileSliceArena::<f32>::new(slice.clone());
            let mut cached = FileSliceArena::<f32>::new(slice.clone());
            cached.cache_norms(segment, field, dim);
            let norms = Arc::clone(cached.norms.as_ref().unwrap());
            assert!(norms.iter().all(|n| n.get().is_none()));
            for _ in 0..2 {
                for scale in [0.0, 0.7, 1e-19, 1e19] {
                    let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.5) * scale).collect();
                    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
                        let mut expected = Vec::new();
                        let mut actual = Vec::new();
                        let norm = norm_squared(&query).sqrt();
                        control.batch_similarity(&query, metric, norm, 1..4, &mut expected);
                        cached.batch_similarity(&query, metric, norm, 1..4, &mut actual);
                        assert_eq!(
                            expected
                                .iter()
                                .map(|v| v.score().to_bits())
                                .collect::<Vec<_>>(),
                            actual
                                .iter()
                                .map(|v| v.score().to_bits())
                                .collect::<Vec<_>>(),
                        );
                        for row in 0..4 {
                            assert_eq!(
                                control
                                    .similarity(metric, dim, row, &query)
                                    .score()
                                    .to_bits(),
                                cached
                                    .similarity(metric, dim, row, &query)
                                    .score()
                                    .to_bits(),
                            );
                        }
                    }
                }
                cached = FileSliceArena::<f32>::new(slice.clone());
                cached.cache_norms(segment, field, dim);
                assert!(Arc::ptr_eq(&norms, cached.norms.as_ref().unwrap()));
                assert!(norms.iter().all(|n| n.get().is_some()));
            }
        }
    }

    #[test]
    fn norm_cache_isolates_segments_and_fields_and_evicts_by_size() {
        let mut cache = NormCache::new(6);
        let segment = SegmentId::generate_random();
        let key = (segment, Field::from_field_id(0), 1024, 2);
        let original = cache.get(key).unwrap();
        original[0].set(3.0).unwrap();
        let other_field = (segment, Field::from_field_id(1), 1024, 2);
        let other_segment = (SegmentId::generate_random(), key.1, 1024, 2);
        assert!(cache.get(other_field).unwrap()[0].get().is_none());
        assert!(cache.get(other_segment).unwrap()[0].get().is_none());
        assert!(Arc::ptr_eq(&original, &cache.get(key).unwrap()));
        assert!(cache.get((segment, key.1, 1024, 7)).is_none());
        cache.get((segment, key.1, 1024, 4)).unwrap();
        assert_eq!(cache.slots, 6);
        assert!(!cache.entries.contains(&other_field));
        assert!(!cache.entries.contains(&other_segment));
        assert_eq!(original[0].get(), Some(&3.0));
    }
}

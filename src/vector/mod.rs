//! Vector storage, indexing, and scoring.
//! Includes flat and inverted-file segment layouts.

use std::io;

mod backend;
mod bounds;
mod collector;
mod distance;
mod header;
mod index_reader;
mod plugin;
mod prepared;
pub(crate) mod quantization;
mod tie_break;

/// Flat vector storage.
pub mod flat;
/// Inverted-file vector storage.
pub mod ivf;

#[cfg(test)]
pub(crate) mod tests;

/// Vector segment file extension.
pub(crate) const VEC_EXT: &str = "vec";

#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
pub use backend::{
    quantization_bench_layer0_cosine_cluster, quantization_bench_layer0_cosine_cluster_f16_scales,
};
pub use backend::{
    set_fixed_probe_cost_rows, ProbeStats, ProbeTermination, VectorBackend,
    DEFAULT_FIXED_PROBE_COST_ROWS,
};
pub use bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, residual_norm, to_bound_space,
    BoundKind, BoundStore, BoundsBuilder, HeapPeek, QueryBound, Verdict,
};
pub use collector::{SegmentVectorFruit, TopDocsByVectorSimilarity, VectorSimilarityFruit};
#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
pub use distance::quantization_bench_dot_bytes_f32;
pub use distance::{
    cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes, Similarity,
};
pub use flat::FlatVecWriter;
pub use index_reader::{VectorClusterStats, VectorIndexReader, VectorInfo, VectorStorageFormat};
pub use ivf::{
    BKTree, BKTreeNode, BKTreeSearchIterator, BktNodeId, BuiltRouter, Candidate, ClusterId,
    FlatStore, Graph, IvfCentroids, IvfClusterer, IvfConfig, IvfIndex, IvfIndexBuilder,
    IvfLevelClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfSearchMetrics,
    IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, MultiLevelIvf,
    NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId, PersistedStackedIvf,
    RelativeNeighborhoodGraph, ResumableSearchIterator, RoutingIndex, SearchIterator,
    SearchTerminationReason, SliceStore, StackedIvfIndex, SuperKMeansLevelClusterer, Workspace,
};
pub use plugin::VectorPlugin;
pub use prepared::PreparedQuery;
pub use quantization::{
    quantized_code_stride, VectorNormPolicy, VectorQuantizationConfig, VectorQuantizationGrid,
    VectorQuantizationLayer, GRID_FORMAT_VERSION, MAX_QUANTIZATION_LAYERS,
    QUANTIZED_CODE_ALIGNMENT, QUANTIZED_CONSTANT_STRIDE, QUANTIZED_ERROR_RATIO_STRIDE,
    QUANTIZED_GAMMA_STRIDE, QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE,
    QUANTIZED_SIDECAR_STRIDE, VECTOR_QUANTIZATION_FORMAT_VERSION,
};
pub use tie_break::NoTieBreak;

pub use crate::schema::{Metric, VectorDType, VectorOptions};

/// Accumulator operations used by reduction kernels.
pub trait Accumulator: Copy + Send + Sync + 'static {
    /// Additive identity.
    const ZERO: Self;
    /// Adds two accumulator values.
    fn add(self, rhs: Self) -> Self;
    /// Converts the accumulator to binary64.
    fn to_f64(self) -> f64;
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

/// A vector element supported by storage and distance kernels.
pub trait VectorElement: Copy + Send + Sync + 'static {
    /// Schema data type.
    const DTYPE: VectorDType;
    /// Serialized byte width.
    const SIZE_BYTES: usize;

    /// Accumulator for reduction kernels.
    type Acc: Accumulator;

    /// Writes one little-endian element.
    fn encode_le<W: io::Write + ?Sized>(&self, buf: &mut W) -> io::Result<()>;

    /// Decodes one little-endian element.
    fn decode_le(bytes: &[u8]) -> Self;

    /// Returns the squared difference as binary32.
    fn squared_diff(a: Self, b: Self) -> f32;

    /// Returns the product as binary32.
    fn product(a: Self, b: Self) -> f32;

    /// Multiplies after widening to the accumulator type.
    fn mul_wide(a: Self, b: Self) -> Self::Acc;

    /// Lossless widening to `f32`.
    fn to_f32(self) -> f32;

    /// Narrows a binary32 value.
    fn from_f32(v: f32) -> Self;
}

impl VectorElement for f32 {
    const DTYPE: VectorDType = VectorDType::F32;
    const SIZE_BYTES: usize = 4;

    type Acc = f64;

    #[inline(always)]
    fn encode_le<W: io::Write + ?Sized>(&self, buf: &mut W) -> io::Result<()> {
        buf.write_all(&self.to_le_bytes())
    }

    #[inline(always)]
    fn decode_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline(always)]
    fn squared_diff(a: Self, b: Self) -> f32 {
        let d = a - b;
        d * d
    }

    #[inline(always)]
    fn product(a: Self, b: Self) -> f32 {
        a * b
    }

    #[inline(always)]
    fn mul_wide(a: Self, b: Self) -> f64 {
        (a as f64) * (b as f64)
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
}

/// A flat, `dim`-strided arena of vectors addressed by dense row index.
///
/// The arena owns its representation — typed slices via the blanket impl,
/// raw little-endian file bytes for reloaded storage — and scores a typed
/// query against a stored vector with the kernel matching that
/// representation. The [`Metric`] is a parameter: an arena never holds one.
pub trait VectorArena {
    /// Stored vector element type.
    type Elem: VectorElement;

    /// Returns the number of stored vectors.
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

/// Implements vector storage for contiguous typed slices.
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
/// [`FileSlice`](crate::directory::FileSlice): each [`similarity`] call
/// fetches only that row with one stride-sized ranged read, scored with the
/// byte kernels ([`Metric::similarity_bytes`]). The arena is never
/// materialized whole — under `MmapDirectory` a read is a zero-copy view,
/// and under a copying `Directory` only the visited rows' bytes are fetched.
///
/// This is the search-time counterpart of the typed blanket impl: a
/// [`Graph`] reloaded from disk wraps its file-resident vectors in one of
/// these and is search-only (refinement needs typed storage).
///
/// [`similarity`]: VectorArena::similarity
pub struct FileSliceArena<T> {
    slice: crate::directory::FileSlice,
    _elem: std::marker::PhantomData<T>,
}

impl<T> FileSliceArena<T> {
    /// Wraps contiguous little-endian vector rows.
    pub fn new(slice: crate::directory::FileSlice) -> Self {
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
    fn similarity(&self, metric: Metric, dim: usize, index: u32, query: &[T]) -> Similarity {
        let stride = dim * T::SIZE_BYTES;
        let bytes = self
            .slice
            .slice(index as usize * stride..(index as usize + 1) * stride)
            .read_bytes()
            .expect("failed to read vector arena row");
        metric.similarity_bytes(query, &bytes)
    }
}

//! Vector fields: storage, clustering and top-k similarity search.
//!
//! A segment stores a field's vectors flat until a merge reaches
//! [`IndexSettings::vector_clustering_threshold`](crate::IndexSettings::vector_clustering_threshold);
//! from there the merge clusters them (IVF) with the index's clusterer and router, quantized when
//! the index requests it. [`TopDocsByVectorSimilarity`] runs the query over either layout.
//!
//! ```
//! use tantivy::collector::TopDocs;
//! use tantivy::query::TermQuery;
//! use tantivy::schema::{IndexRecordOption, Metric, Schema, VectorOptions, STRING};
//! use tantivy::{Index, IndexWriter, TantivyDocument, Term};
//!
//! # fn main() -> tantivy::Result<()> {
//! let mut schema_builder = Schema::builder();
//! let category = schema_builder.add_text_field("category", STRING);
//! let embedding =
//!     schema_builder.add_vector_field("embedding", VectorOptions::new(64, Metric::Cosine));
//! let index = Index::builder()
//!     .schema(schema_builder.build())
//!     .quantize(embedding, [1, 1])
//!     .create_in_ram()?;
//!
//! let mut writer: IndexWriter = index.writer(50_000_000)?;
//! for i in 0..100 {
//!     let mut doc = TantivyDocument::new();
//!     doc.add_text(category, if i % 2 == 0 { "even" } else { "odd" });
//!     let vector: Vec<f32> = (0..64).map(|d| ((i * 64 + d) as f32).sin()).collect();
//!     doc.add_vector(embedding, &vector);
//!     writer.add_document(doc)?;
//! }
//! writer.commit()?;
//!
//! let even = TermQuery::new(
//!     Term::from_field_text(category, "even"),
//!     IndexRecordOption::Basic,
//! );
//! let query = vec![0.5_f32; 64];
//! let searcher = index.reader()?.searcher();
//! let hits = searcher.search(
//!     &even,
//!     &TopDocs::with_limit(10).order_by_similarity(embedding, query),
//! )?;
//! assert_eq!(hits.results.len(), 10);
//! # Ok(())
//! # }
//! ```

use std::io;

mod backend;
mod blocks;
mod element;
pub use element::MAX_ELEM_BYTES;
/// Fixed alignment of Data entries and their block directories.
pub const ENTRY_ALIGN: usize = 8;
const _: () = assert!(MAX_ELEM_BYTES <= ENTRY_ALIGN);
mod bounds;
mod cluster_source;
mod collector;
mod distance;
pub(crate) mod header;
pub(crate) mod index_reader;
mod interval;
mod metadata;
mod scan;
mod segment;
mod source;
mod stats;
mod storage_io;
pub use storage_io::VectorIoStats;
mod telemetry;
pub(crate) use telemetry::enter_vector_stage;
pub use telemetry::{current_vector_stage, Stage};
mod plugin;
mod prepared;
pub(crate) mod quantization;
mod tie_break;

pub mod flat;
pub mod ivf;
pub mod router;

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) const VEC_EXT: &str = "vec";

pub use backend::VectorBackend;
pub use bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, residual_norm, to_bound_space,
    BoundKind, BoundStore, BoundsBuilder, HeapPeek, QueryBound, Verdict,
};
pub use collector::{SegmentVectorFruit, TopDocsByVectorSimilarity, VectorSimilarityFruit};
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use distance::quantization_bench_dot_bytes_f32;
pub use distance::{
    cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes, Similarity,
};
pub use flat::FlatVecWriter;
pub use header::VectorFileVersion;
pub use index_reader::{
    VectorAuditMoments, VectorClusterStats, VectorErrorAuditMeasurements,
    VectorErrorConeAuditMeasurements, VectorErrorConeDepthMeasurements,
    VectorErrorDepthMeasurements, VectorEstimatorMeasurements, VectorEstimatorMoments,
    VectorEstimatorQuery, VectorEstimatorSource, VectorIndexReader, VectorInfo,
    VectorStorageFormat,
};
pub use ivf::{
    Candidate, ClusterId, InMemoryStackedIvf, InMemoryStore, IvfCentroids, IvfClusterer, IvfConfig,
    IvfIndex, IvfIndexBuilder, IvfLevelClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings,
    IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, LazyStackedIvf, LazyStore,
    MultiLevelIvf, StackedSearchStats, SuperKMeansClusterer, SuperKMeansLevelClusterer,
    APS_MAX_DIM, DEFAULT_MAX_LEAF_SIZE, DEFAULT_TRAINING_SAMPLE_RATIO,
};
pub use metadata::{
    F64Bits, Grid, Partition, QuantizationSchedule, Quantizer, QuantizerKind, Rotation,
    VectorColMetadata, VectorFieldMeta,
};
pub use plugin::VectorPlugin;
pub use quantization::{
    quantized_code_stride, VectorNormPolicy, VectorQuantizationConfig, VectorQuantizationGrid,
    VectorQuantizationLayer, MAX_QUANTIZATION_LAYERS, QUANTIZED_CODE_ALIGNMENT,
    QUANTIZED_CONSTANT_STRIDE, QUANTIZED_ERROR_RATIO_STRIDE, QUANTIZED_GAMMA_STRIDE,
    QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE, QUANTIZED_SIDECAR_STRIDE,
    VECTOR_QUANTIZATION_FORMAT_VERSION,
};
pub use router::{RouterKind, RouterMetrics, RoutingParams};
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use scan::estimate::quantization_bench_layer0_cosine_cluster;
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use scan::{columns::Columns, layer_read::Decoded};
pub use source::controller::{ProbeTermination, DEFAULT_FIXED_PROBE_COST_ROWS};
pub use stats::ProbeStats;
pub use tie_break::NoTieBreak;

// The schema-level vector types are re-exported here so `crate::vector::{...}`
// resolves for callers and tests that work entirely within the vector module.
pub use crate::schema::{Metric, VectorDType, VectorOptions};

/// Wide accumulator used by the reduction kernels (norms). Element
/// types choose their accumulator via [`VectorElement::Acc`]; this
/// trait is the vocabulary a kernel needs to *use* that choice:
/// make a zero, add, read out as f64 at the fold.
///
/// Deliberately a custom trait rather than std bounds: the natural
/// accumulator for a future quantized element is an integer (e.g.
/// `u64` for `u8` elements — exact, matches integer-SIMD hardware),
/// and `Into<f64>` does not exist for `u64` because std reserves
/// `From`/`Into` for lossless conversions. `to_f64` is explicitly
/// the lossy read-out at the end of a reduction.
///
/// Kept separate from `VectorElement` because these operations never
/// mention the element type — how to sum two `f64`s is a fact about
/// `f64` — and because accumulator impls are shared: `f32` and a
/// future `f16` would both declare `type Acc = f64` and reuse this
/// one impl.
pub trait Accumulator: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn add(self, rhs: Self) -> Self;
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

/// A vector element type with the primitives needed by the storage
/// layer and the distance kernels.
///
/// Implemented for the element types supported by [`VectorDType`]. The
/// `DTYPE` associated constant lets callers reject mismatches between
/// the declared schema dtype and the type passed at runtime. The
/// arithmetic methods (`squared_diff`, `product`) return `f32` so that
/// kernels can use a uniform accumulator type across dtypes; the
/// reduction kernels (norms) instead accumulate in [`Self::Acc`] via
/// [`Self::mul_wide`].
pub trait VectorElement: Copy + Send + Sync + 'static {
    const DTYPE: VectorDType;
    const SIZE_BYTES: usize;

    /// Accumulator for this element's squares in reduction kernels.
    /// Each element type declares how much headroom it needs: f32 -> f64
    /// (squaring doubles the exponent, so |v| > ~1.8e19 overflows an f32
    /// product; f64 makes any sum of finite-f32 squares finite).
    type Acc: Accumulator;

    fn encode_le<W: io::Write + ?Sized>(&self, buf: &mut W) -> io::Result<()>;

    /// Decode one element from its little-endian byte representation.
    /// `bytes.len()` must be `SIZE_BYTES`.
    fn decode_le(bytes: &[u8]) -> Self;

    /// `(a - b)^2` promoted to `f32` for accumulator-friendly distance
    /// computation. For `f32` this is the obvious arithmetic; for
    /// quantized types it may promote through a wider integer first.
    fn squared_diff(a: Self, b: Self) -> f32;

    /// `a * b` promoted to `f32`. Same rationale as `squared_diff`.
    fn product(a: Self, b: Self) -> f32;

    /// Widen-THEN-multiply: the cast happens before the square so the
    /// product cannot overflow the narrow type. One operation, so the
    /// ordering invariant is unforgettable per element type.
    fn mul_wide(a: Self, b: Self) -> Self::Acc;

    /// Lossless widening to `f32`.
    fn to_f32(self) -> f32;

    /// Narrowing from `f32`; used at normalization write-back where
    /// values are already `<= 1`.
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
    slice: crate::directory::FileSlice,
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

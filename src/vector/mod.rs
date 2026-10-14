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

mod collector;
mod distance;
mod element;
mod params;
pub(crate) mod quantization;
mod stats;
mod telemetry;
mod tie_break;

pub mod clustering;
pub mod diagnostics;
pub mod router;
mod search;
pub(crate) mod storage;
mod write;

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod tests;

pub use clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfTrainingBatch,
    IvfTrainingVectors, IvfVectorBatch, IvfVectors, SuperKMeansClusterer, DEFAULT_MAX_LEAF_SIZE,
    DEFAULT_TRAINING_SAMPLE_RATIO,
};
pub use collector::{SegmentVectorFruit, TopDocsByVectorSimilarity, VectorSimilarityFruit};
pub use diagnostics::{
    VectorAuditMoments, VectorClusterStats, VectorErrorAuditMeasurements,
    VectorErrorConeAuditMeasurements, VectorErrorConeDepthMeasurements,
    VectorErrorDepthMeasurements, VectorEstimatorMeasurements, VectorEstimatorMoments,
    VectorEstimatorQuery, VectorEstimatorSource, VectorInfo, VectorStorageFormat,
};
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use distance::quantization_bench_dot_bytes_f32;
pub use distance::{
    cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes, Similarity,
};
pub use element::{Accumulator, VectorElement, ENTRY_ALIGN, MAX_ELEM_BYTES};
pub use params::{AdaptiveProbeParams, LocatedPath, Selectivity, WorkModel, DEFAULT_ROUTER_RECALL};
pub use quantization::{
    quantized_code_stride, VectorNormPolicy, VectorQuantizationConfig, VectorQuantizationGrid,
    VectorQuantizationLayer, MAX_QUANTIZATION_LAYERS, QUANTIZED_CODE_ALIGNMENT,
    QUANTIZED_CONSTANT_STRIDE, QUANTIZED_ERROR_RATIO_STRIDE, QUANTIZED_GAMMA_STRIDE,
    QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE, QUANTIZED_SIDECAR_STRIDE,
    VECTOR_QUANTIZATION_FORMAT_VERSION,
};
pub use router::arena::{FileSliceArena, VectorArena};
pub use router::candidate::Candidate;
pub use router::stacked::aps::APS_MAX_DIM;
pub use router::stacked::levels::{
    AddLevelError, ClusterId, InMemoryStackedIvf, InMemoryStore, IvfConfig,
    IvfIndex as MultiLevelIvf, IvfIndexBuilder, IvfLevelClusterer, LazyStackedIvf, LazyStore,
    StackedSearchStats, SuperKMeansLevelClusterer, PARENT_NPROBE_FRACTION,
};
pub use router::{RouterKind, RouterMetrics, RoutingParams};
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use search::scan::estimate::quantization_bench_layer0_cosine_cluster;
#[cfg(feature = "unstable")]
#[doc(hidden)]
pub use search::scan::{columns::Columns, layer_read::Decoded};
pub use search::source::controller::{ProbeTermination, DEFAULT_FIXED_PROBE_COST_ROWS};
pub use search::VectorBackend;
pub use stats::ProbeStats;
pub use storage::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, residual_norm, to_bound_space,
    BoundKind, BoundStore, BoundsBuilder, HeapPeek, QueryBound, Verdict,
};
pub use storage::centroids::IvfIndex;
pub use storage::header::VectorFileVersion;
pub use storage::io::VectorIoStats;
pub use storage::metadata::{
    F64Bits, Grid, Partition, QuantizationSchedule, Quantizer, QuantizerKind, Rotation,
    VectorColMetadata, VectorFieldMeta,
};
pub use storage::reader::VectorIndexReader;
pub(crate) use telemetry::enter_vector_stage;
pub use telemetry::{current_vector_stage, Stage};
pub use tie_break::NoTieBreak;
pub use write::flat::FlatVecWriter;
pub use write::VectorPlugin;

// The schema-level vector types are re-exported here so `crate::vector::{...}`
// resolves for callers and tests that work entirely within the vector module.
pub use crate::schema::{Metric, VectorDType, VectorOptions};

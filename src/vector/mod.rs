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
//!     &TopDocs::with_limit(10)
//!         .order_by_similarity(embedding, query)
//!         .build(),
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

#[cfg(feature = "unstable")]
#[doc(hidden)]
pub mod bench;
pub mod clustering;
pub mod diagnostics;
mod router;
mod search;
pub(crate) mod storage;
pub(crate) mod write;

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod tests;

pub use collector::{
    SegmentVectorFruit, TopDocsByVectorSimilarity, TopDocsByVectorSimilarityBuilder,
    VectorSimilarityFruit,
};
pub use element::{Accumulator, VectorElement, ENTRY_ALIGN};
pub use params::{
    LocatedPath, ProbeParams, Selectivity, DEFAULT_FIXED_PROBE_COST_ROWS, LOCATED_MAX_SELECTIVITY,
};
pub use quantization::{VectorQuantizationConfig, VectorQuantizationLayer};
pub use router::{RouterKind, RouterMetrics};
pub use search::source::controller::ProbeTermination;
pub use stats::ProbeStats;
pub use storage::io::{current_vector_read_kind, VectorReadKind};
pub use storage::{VectorIndexReader, VectorIoStats};
pub use telemetry::{current_vector_stage, VectorStage};
pub use tie_break::NoTieBreak;

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
//!         .build_for_searcher(&searcher, &even)?,
//! )?;
//! assert_eq!(hits.results.len(), 10);
//!
//! // The same search with explicit preparation, routing, and document collection.
//! use tantivy::collector::Collector;
//! use tantivy::query::{EnableScoring, Query};
//! use tantivy::vector::{
//!     ProbeParams, TopClustersBySimilarityCollector, VectorSearchBudget, VectorSegment,
//! };
//! let params = ProbeParams::default();
//! let query = vec![0.5_f32; 64];
//! let builder = TopDocs::with_limit(10)
//!     .order_by_similarity(embedding, query.clone())
//!     .with_probe_params(params.clone());
//! let weight = even.weight(EnableScoring::disabled_from_searcher(&searcher))?;
//! let segments: Vec<_> = searcher.segment_readers().iter().enumerate()
//!     .map(|(ord, reader)| VectorSegment::from((ord as u32, reader))).collect();
//! let budget = VectorSearchBudget::for_segments(embedding, &params, &segments)?;
//! let prepared = builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
//! let router = TopClustersBySimilarityCollector::new(
//!     embedding, query, params, prepared.match_fraction(),
//! )?.with_budget(prepared.budget());
//! let candidates = router.collect_segments(&prepared.routing_segments(), |_| {})?;
//! let inputs = prepared.with_candidates(candidates, None)?
//!     .with_routing_statistics(router.take_statistics());
//! let collector = builder.build(inputs);
//! let hits = collector.merge_fruits(
//!     collector.collect_segments(weight.as_ref(), &segments, |_| {})?,
//! )?;
//! assert_eq!(hits.results.len(), 10);
//! # Ok(())
//! # }
//! ```
//!
//! Parallel callers resolve a [`VectorSearchBudget`] over the complete snapshot and pass it
//! to each worker's [`TopDocsByVectorSimilarityBuilder::prepare`]. Distribute the coordinator's
//! ranked candidates by original segment ordinal and attach each assignment with
//! [`PreparedVectorSearch::with_candidates`], supplying the same [`SharedVectorWork`] counter
//! to every worker. [`AtomicVectorWork`] supports threads or caller-owned shared memory.
//! Workers charge actual cluster opens and scored rows to the query-wide ceiling, with at most
//! one final cluster's overshoot per active worker. Located scans use separate size-based caps
//! and need neither candidates nor routing publication. Reset the counter for each new query
//! or rescan; never copy its value into independent worker counters.

mod collector;
mod distance;
mod element;
mod params;
pub(crate) mod quantization;
mod rotation_plan;
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
    AdaptiveProbeParams, LocatedPath, ProbeParams, Selectivity, DEFAULT_FIXED_PROBE_COST_ROWS,
};
pub use quantization::{VectorQuantizationConfig, VectorQuantizationLayer};
pub use router::candidate::Candidate;
pub use router::collector::{TopClustersBySimilarityCollector, VectorRoutingStats};
pub use router::{RouterKind, RouterMetrics};
pub use search::input::{
    ClusterRef, PreparedVectorSearch, VectorScanInputs, VectorSearchBudget, VectorSegment,
};
pub use search::source::controller::{AtomicVectorWork, ProbeTermination, SharedVectorWork};
pub use stats::ProbeStats;
pub use storage::blocks::BlockAddress;
pub use storage::io::{current_vector_read_kind, VectorReadKind};
pub use storage::{VectorIndexReader, VectorIndexStats, VectorIoStats};
pub use telemetry::{current_vector_stage, VectorStage};
pub use tie_break::NoTieBreak;

//! Top-N vector-similarity collection.
//!
//! IVF drives its own cluster iteration using prepared filters and explicit routing candidates.
//! A secondary key is an ordinary `SortKeyComputer` composed into the heap's
//! `(similarity, tie_break)` order, the same rule as the pull-model path.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{
    compare_for_top_k, AtomicSharedThreshold, Collector, ComparableDoc, SegmentCollector,
    SegmentSortKeyComputer, SharedThresholdArc, SortKeyComputer,
};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, Metric, Schema};
use crate::vector::params::ProbeParams;
use crate::vector::search::cluster_source::ClusterSource;
use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
use crate::vector::search::scan::bound::{ExactMerge, PruningBound, TopK};
use crate::vector::search::scan::heap::TieBreakHits;
use crate::vector::search::scan::{ScanConfig, SegmentScan};
use crate::vector::search::segment::plan::ScanInput;
use crate::vector::search::source::controller::{
    ProbeTermination, WorkBudget, WorkMeter, WorkUnits,
};
use crate::vector::search::source::matches::MatchClusters;
use crate::vector::search::source::router::RouterClusters;
use crate::vector::search::source::SourceReport;
use crate::vector::search::SegmentSearcher;
use crate::vector::stats::{ProbeStats, SegmentReport};
use crate::vector::storage::metadata::{Quantizer, VectorColMetadata};
use crate::vector::storage::quantized::QuantizedFieldReader;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::tie_break::NoTieBreak;
use crate::vector::write::RowAddress;
use crate::vector::{
    BlockAddress, PreparedVectorSearch, TopClustersBySimilarityCollector, VectorElement,
    VectorScanInputs, VectorSearchBudget, VectorSegment, VectorStage,
};
use crate::{DocAddress, DocId, Result, Score, SegmentOrdinal, TantivyError};

// --- Query preparation ---

/// Query identity consists of dimension, metric tag and structural layer metadata.
#[derive(Clone, Debug)]
struct PreparedKey(Arc<VectorColMetadata>);
impl PreparedKey {
    fn query_fields(&self) -> Option<(u32, u8, &[Quantizer])> {
        match self.0.as_ref() {
            VectorColMetadata::Plain(_) => None,
            VectorColMetadata::Quantized { field, layers } => {
                let metric = match field.metric {
                    Metric::L2 => 0,
                    Metric::Dot => 1,
                    Metric::Cosine => 2,
                };
                Some((field.dim, metric, layers))
            }
        }
    }
}
impl PartialEq for PreparedKey {
    fn eq(&self, other: &Self) -> bool {
        self.query_fields() == other.query_fields()
    }
}
impl Eq for PreparedKey {}
impl Hash for PreparedKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query_fields().hash(state);
    }
}

/// Shared initialization cell so each metadata key prepares its query exactly once.
type PreparedCell = Arc<OnceLock<Arc<QuantizedQueryCtx>>>;

// --- Builder ---

/// Builds a [`TopDocsByVectorSimilarity`] collector.
#[derive(Clone)]
pub struct TopDocsByVectorSimilarityBuilder<T: VectorElement, S = NoTieBreak> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    probe_params: ProbeParams,
    max_scan_levels: usize,
    /// Optional slot shared with other collectors searching the same query.
    shared_threshold: Option<SharedThresholdArc<Score>>,
    tie_break: S,
}

impl<T: VectorElement> TopDocsByVectorSimilarityBuilder<T, NoTieBreak> {
    /// Starts a builder for a top-vector-similarity collector.
    pub fn new(field: Field, query: Vec<T>, limit: usize) -> Self {
        Self {
            field,
            query: Arc::new(query),
            limit,
            offset: 0,
            probe_params: ProbeParams::default(),
            max_scan_levels: usize::MAX,
            shared_threshold: None,
            tie_break: NoTieBreak,
        }
    }
}

impl<T: VectorElement, S> TopDocsByVectorSimilarityBuilder<T, S> {
    /// Skips the first `offset` results of the global ranking; each segment keeps
    /// `limit + offset`.
    pub fn and_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Sets the probe parameters; flat segments ignore them.
    pub fn with_probe_params(mut self, params: ProbeParams) -> Self {
        self.probe_params = params;
        self
    }

    /// Limits the quantized residual prefix.
    pub fn with_max_scan_levels(mut self, max_scan_levels: usize) -> Self {
        self.max_scan_levels = max_scan_levels;
        self
    }

    /// Shares every segment scan's pruning threshold through `slot` with the query's other
    /// participants, each searching its own segments for the same query, filter and
    /// `limit + offset`. Every score in the slot must be a lower bound on the k-th best score
    /// among those rows, k = `limit + offset`: each scan raises its threshold to the slot's score
    /// and publishes its own when that is higher. When omitted, `build` installs a process-local
    /// [`AtomicSharedThreshold`] shared by every segment of this search.
    pub fn with_shared_threshold(mut self, slot: SharedThresholdArc<Score>) -> Self {
        self.shared_threshold = Some(slot);
        self
    }

    /// Orders equal similarities by `tie_break`, as `ORDER BY embedding <=> $1, id` does.
    ///
    /// The tie-break takes part in each segment's eviction but never in routing. Segments cut on
    /// the segment-local key, so `convert_segment_sort_key` must preserve order within a segment.
    pub fn with_tie_break<S2: SortKeyComputer>(
        self,
        tie_break: S2,
    ) -> TopDocsByVectorSimilarityBuilder<T, S2> {
        TopDocsByVectorSimilarityBuilder {
            field: self.field,
            query: self.query,
            limit: self.limit,
            offset: self.offset,
            probe_params: self.probe_params,
            max_scan_levels: self.max_scan_levels,
            shared_threshold: self.shared_threshold,
            tie_break,
        }
    }

    /// Evaluates assigned segments' filters once using a budget resolved over the full search
    /// snapshot. The budget determines bypass eligibility and subsequent scan work.
    pub fn prepare(
        &self,
        weight: &dyn Weight,
        segments: &[VectorSegment],
        budget: VectorSearchBudget,
        on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<PreparedVectorSearch> {
        PreparedVectorSearch::prepare(
            self.field,
            self.limit.saturating_add(self.offset),
            &self.probe_params,
            weight,
            segments,
            budget,
            on_segment,
        )
    }

    /// Convenience orchestration: prepares filters, ranks only routed segments, then builds a
    /// document collector with complete inputs. The document collector never performs routing.
    pub fn build_for_segments(
        self,
        weight: &dyn Weight,
        segments: &[VectorSegment],
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<TopDocsByVectorSimilarity<T, S>> {
        let mut opening = Vec::with_capacity(segments.len());
        for segment in segments {
            let (result, totals) = measure(|| {
                let _stage = enter_vector_stage(VectorStage::QueryPrep);
                segment.reader.vector_index(self.field)
            });
            on_segment(segment.ordinal);
            result?;
            opening.push((segment.ordinal, totals));
        }
        let budget = VectorSearchBudget::for_segments(self.field, &self.probe_params, segments)?;
        let mut prepared = self.prepare(weight, segments, budget, &mut on_segment)?;
        for (ordinal, totals) in opening {
            prepared.scope.get_mut(&ordinal).unwrap().2.append(&totals);
        }
        let router = TopClustersBySimilarityCollector::new(
            self.field,
            self.query.as_ref().clone(),
            self.probe_params.clone(),
            prepared.match_fraction(),
        )?
        .with_budget(prepared.budget())
        .without_candidate_limit();
        let candidates = router.collect_segments(&prepared.routing_segments(), &mut on_segment)?;
        let inputs = prepared
            .with_candidates(candidates, None)?
            .with_routing_statistics(router.take_statistics());
        Ok(self.build(inputs))
    }

    /// Prepares a collector for all segments of a searcher and the supplied filter.
    pub fn build_for_searcher(
        self,
        searcher: &crate::Searcher,
        filter: &dyn crate::query::Query,
    ) -> Result<TopDocsByVectorSimilarity<T, S>> {
        let weight = filter.weight(crate::query::EnableScoring::disabled_from_searcher(
            searcher,
        ))?;
        let segments = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(i, reader)| VectorSegment::from((i as SegmentOrdinal, reader)))
            .collect::<Vec<_>>();
        self.build_for_segments(weight.as_ref(), &segments, |_| {})
    }

    /// Builds the collector. Configuration cannot be changed afterwards.
    pub fn build(self, inputs: VectorScanInputs) -> TopDocsByVectorSimilarity<T, S> {
        let segment_top_n = self.limit.saturating_add(self.offset);
        let shared_topk_threshold = self.shared_threshold.unwrap_or_else(|| {
            Arc::new(AtomicSharedThreshold::default()) as SharedThresholdArc<Score>
        });
        TopDocsByVectorSimilarity {
            field: self.field,
            query: self.query,
            limit: self.limit,
            offset: self.offset,
            probe_params: self.probe_params,
            max_scan_levels: self.max_scan_levels,
            inputs,
            quantized_queries: Mutex::new(HashMap::new()),
            prepared: Mutex::new(None),
            collector_topk: Arc::new(Mutex::new(TopK::new(segment_top_n))),
            shared_topk_threshold,
            tie_break: self.tie_break,
        }
    }
}

// --- Collector ---

/// Collects documents by descending vector similarity; docs without a vector are never returned.
///
/// `T` must match the field's dtype ([`Collector::check_schema`]). `S` orders equal similarities
/// and defaults to [`NoTieBreak`]: ascending `DocAddress`.
///
/// Construct via [`TopDocsByVectorSimilarityBuilder`] or [`TopDocsByVectorSimilarity::new`].
pub struct TopDocsByVectorSimilarity<T: VectorElement, S = NoTieBreak> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    probe_params: ProbeParams,
    max_scan_levels: usize,
    inputs: VectorScanInputs,
    /// Exactly one prepared query for each distinct segment encoding.
    quantized_queries: Mutex<HashMap<PreparedKey, PreparedCell>>,
    /// The query prepared for the field's metric on first use.
    prepared: Mutex<Option<Arc<VectorQuery<T>>>>,
    /// The top-k exact scores (with addresses) from segments this collector has finished. Later
    /// segments seed their pruning-bound TopK from these for partial-k accounting.
    collector_topk: Arc<Mutex<TopK>>,
    /// The shared top-k threshold every segment scan of this collector syncs and publishes.
    /// Either the caller's slot or a process-local [`AtomicSharedThreshold`] from `build`.
    shared_topk_threshold: SharedThresholdArc<Score>,
    tie_break: S,
}

impl<T: VectorElement> TopDocsByVectorSimilarity<T, NoTieBreak> {
    /// Creates a top-vector-similarity collector with default probe settings.
    pub fn new(field: Field, query: Vec<T>, limit: usize, inputs: VectorScanInputs) -> Self {
        TopDocsByVectorSimilarityBuilder::new(field, query, limit).build(inputs)
    }
}

impl<T: VectorElement, S> TopDocsByVectorSimilarity<T, S> {
    /// Replaces a distributed scan's inputs for the next assignment in the same query,
    /// preserving prepared queries and pruning state. Field, filter, top-k and shared work
    /// counter must be unchanged.
    pub fn set_inputs(&mut self, inputs: VectorScanInputs) -> Result<()> {
        if inputs.prepared.field != self.field
            || inputs.prepared.top_n != self.segment_top_n()
            || !inputs
                .shared_work
                .as_ref()
                .zip(self.inputs.shared_work.as_ref())
                .is_some_and(|(next, current)| Arc::ptr_eq(next, current))
        {
            return Err(TantivyError::InvalidArgument(
                "replacement inputs require the same field, top-k and shared work counter".into(),
            ));
        }
        self.inputs = inputs;
        Ok(())
    }

    fn segment_top_n(&self) -> usize {
        self.limit.saturating_add(self.offset)
    }

    /// The query prepared for `metric`, on first use.
    fn vector_query(&self, metric: Metric) -> Arc<VectorQuery<T>> {
        let mut prepared = self.prepared.lock().unwrap();
        let query = match prepared.as_ref() {
            Some(query) if query.metric() == metric => query,
            _ => prepared.insert(Arc::new(VectorQuery::new(metric, Arc::clone(&self.query)))),
        };
        Arc::clone(query)
    }

    /// Prepares `query` once per metadata key, releasing the map lock before expensive
    /// preparation.
    fn quantized_query(
        &self,
        field: &QuantizedFieldReader,
        query: &VectorQuery<T>,
    ) -> Arc<QuantizedQueryCtx> {
        let index_ctx = field.index_ctx();
        let prepare = || {
            let active_layers = self.max_scan_levels.min(index_ctx.specs.len());
            Arc::new(QuantizedQueryCtx::with_depth(
                Arc::clone(index_ctx),
                query,
                active_layers,
            ))
        };
        let cell = Arc::clone(
            self.quantized_queries
                .lock()
                .unwrap()
                .entry(PreparedKey(Arc::clone(&index_ctx.meta)))
                .or_default(),
        );
        Arc::clone(cell.get_or_init(prepare))
    }

    #[cfg(test)]
    pub(crate) fn quantized_query_count(&self) -> usize {
        self.quantized_queries.lock().unwrap().len()
    }
}

// --- Fruits ---

/// The global top-N and each searched segment's [`ProbeStats`].
#[derive(Debug, Default)]
pub struct VectorSimilarityFruit {
    /// Global results in descending-similarity order.
    pub results: Vec<(Score, DocAddress)>,
    /// One entry per collected segment, in segment order. Counters sum across segments;
    /// `termination` is per segment.
    pub stats: Vec<ProbeStats>,
}

/// One segment's results with their tie-break keys, before [`Collector::merge_fruits`].
///
/// The merge orders by the same composite key as the segment heaps, then drops the tie-break.
pub struct SegmentVectorFruit<K> {
    results: KeyedHits<K>,
    stats: ProbeStats,
}

/// One segment's hits with their tie-break keys lifted out of the segment.
type KeyedHits<K> = Vec<((Score, K), DocAddress)>;

// --- Collection ---

impl<T, S> Collector for TopDocsByVectorSimilarity<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    type Fruit = VectorSimilarityFruit;
    type Child = NoOpSegmentCollector<S::SortKey>;

    fn check_schema(&self, schema: &Schema) -> Result<()> {
        VectorQuery::check_schema(self.field, &self.query, schema)?;
        if self.tie_break.requires_scoring() {
            // No score is computed (`requires_scoring` is false), so every doc would tie.
            return Err(TantivyError::InvalidArgument(
                "vector similarity cannot be tie-broken by the relevance score: no score is \
                 computed when ordering by a vector field"
                    .to_string(),
            ));
        }
        self.tie_break.check_schema(schema)
    }

    fn for_segment(
        &self,
        _segment_local_id: SegmentOrdinal,
        _reader: &SegmentReader,
    ) -> Result<Self::Child> {
        // Unreachable: `collect_segment` is overridden; the child only satisfies the bound.
        Ok(NoOpSegmentCollector::default())
    }

    fn requires_scoring(&self) -> bool {
        // Similarity comes from the stored vectors, never from the filter's score.
        false
    }

    fn collect_segment(
        &self,
        _weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> Result<SegmentVectorFruit<S::SortKey>> {
        self.validate_input(segment_ord, reader)?;
        let (_, cap, preparation) = &self.inputs.prepared.scope[&segment_ord];
        if !self
            .inputs
            .prepared
            .segments
            .iter()
            .any(|s| s.ordinal == segment_ord)
        {
            return Ok(SegmentVectorFruit {
                results: Vec::new(),
                stats: ProbeStats::new(
                    SegmentReport::Empty { located_cap: *cap },
                    preparation.clone(),
                ),
            });
        }
        let (searched, mut totals) = measure(|| self.search_segment(segment_ord, reader));
        let (results, report, merge) = searched?;
        totals.append(preparation);
        if let Some(stats) = self.inputs.routing_stats.get(&segment_ord) {
            totals.append(&stats.totals);
        }
        let mut stats = ProbeStats::new(report, totals);
        stats.record_shared_offer(merge);
        Ok(SegmentVectorFruit { results, stats })
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<SegmentVectorFruit<S::SortKey>>,
    ) -> Result<Self::Fruit> {
        let (mut merged, totals) = measure(|| {
            let _assembly = enter_vector_stage(VectorStage::ResultAssembly);
            // Each fruit is already its segment's top `limit + offset` under this order, so the
            // global window is a sort of their union. Stats keep one entry per segment.
            let comparator = (NaturalComparator, self.tie_break.comparator());
            let mut stats = Vec::with_capacity(segment_fruits.len());
            let mut all: Vec<ComparableDoc<(Score, S::SortKey), DocAddress>> = Vec::new();
            for fruit in segment_fruits {
                stats.push(fruit.stats);
                all.extend(
                    fruit
                        .results
                        .into_iter()
                        .map(|(sort_key, doc)| ComparableDoc { sort_key, doc }),
                );
            }
            // A total order (ascending `DocAddress` last), so the unstable sort is deterministic.
            all.sort_unstable_by(|lhs, rhs| compare_for_top_k(&comparator, lhs, rhs));
            let results = all
                .into_iter()
                .skip(self.offset)
                .take(self.limit)
                .map(|cd| (cd.sort_key.0, cd.doc))
                .collect();
            VectorSimilarityFruit { results, stats }
        });
        if let Some(first) = merged.stats.first_mut() {
            first.add_assembly(&totals);
        }
        Ok(merged)
    }
}

impl<T, S> TopDocsByVectorSimilarity<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    /// Scans the supplied segments using prepared inputs and their explicit candidate order.
    /// Uses either the shared work counter or one meter over the global candidate order.
    /// Located filter matches are exhaustive and reserve their share of the budget.
    /// Calls `on_segment` after each slice of work for per-segment read accounting.
    pub fn collect_segments(
        &self,
        weight: &dyn Weight,
        segments: &[VectorSegment],
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<Vec<SegmentVectorFruit<S::SortKey>>> {
        if segments.is_empty() || self.inputs.shared_work.is_some() {
            return segments
                .iter()
                .map(|segment| {
                    let result = self.collect_segment(weight, segment.ordinal, &segment.reader);
                    on_segment(segment.ordinal);
                    result
                })
                .collect();
        }
        for segment in segments {
            self.validate_input(segment.ordinal, &segment.reader)?;
        }
        let mut vectors = Vec::with_capacity(segments.len());
        let mut tie_breaks = Vec::with_capacity(segments.len());
        let mut totals = Vec::with_capacity(segments.len());
        for segment in segments {
            let (result, measured) = measure(|| {
                let _init = enter_vector_stage(VectorStage::ScanInit);
                let vectors = {
                    let _prep = enter_vector_stage(VectorStage::QueryPrep);
                    let vectors = segment.reader.vector_index(self.field)?;
                    self.vector_query(vectors.options().metric());
                    vectors
                };
                Ok::<_, TantivyError>((
                    vectors,
                    self.tie_break.segment_sort_key_computer(&segment.reader)?,
                ))
            });
            on_segment(segment.ordinal);
            let (reader, tie_break) = result?;
            vectors.push(reader);
            tie_breaks.push(tie_break);
            let mut measured = measured;
            measured.append(&self.inputs.prepared.scope[&segment.ordinal].2);
            if let Some(stats) = self.inputs.routing_stats.get(&segment.ordinal) {
                measured.append(&stats.totals);
            }
            totals.push(measured);
        }
        let results = search_global(
            self,
            &vectors,
            segments,
            &mut tie_breaks,
            &mut totals,
            &mut |i| on_segment(segments[i].ordinal),
        )?;
        let fruits = results
            .into_iter()
            .zip(tie_breaks)
            .zip(totals)
            .map(|(((hits, report), tie_break), totals)| {
                let merge = self
                    .collector_topk
                    .lock()
                    .unwrap()
                    .merge_exact(hits.iter().map(|((score, _), address)| {
                        (
                            RowAddress {
                                segment_ord: address.segment_ord,
                                // Reserve the sentinel block for exact document seed keys.
                                block_address: BlockAddress::new(u32::MAX, address.doc_id),
                            },
                            *score,
                        )
                    }));
                let mut stats = ProbeStats::new(report, totals);
                stats.record_shared_offer(merge);
                let results = hits
                    .into_iter()
                    .map(|((score, key), address)| {
                        ((score, tie_break.convert_segment_sort_key(key)), address)
                    })
                    .collect();
                SegmentVectorFruit { results, stats }
            })
            .collect();
        Ok(fruits)
    }

    fn validate_input(&self, ordinal: SegmentOrdinal, reader: &SegmentReader) -> Result<()> {
        if self.inputs.prepared.field != self.field
            || self.inputs.prepared.top_n != self.segment_top_n()
        {
            return Err(TantivyError::InvalidArgument(
                "scan inputs belong to a different vector field or top-k window".into(),
            ));
        }
        match self.inputs.prepared.scope.get(&ordinal) {
            Some((id, _, _)) if *id == reader.segment_id() => Ok(()),
            _ => Err(TantivyError::InvalidArgument(
                "segment is missing from the prepared vector snapshot".into(),
            )),
        }
    }

    fn segment_searcher(&self, segment_ord: SegmentOrdinal) -> Result<SegmentSearcher<T>> {
        let segment = self
            .inputs
            .prepared
            .segments
            .iter()
            .find(|s| s.ordinal == segment_ord)
            .expect("prepared segment")
            .clone();
        let reader = &segment.reader;
        self.check_schema(reader.schema())?;
        let (query, quantized) = {
            let _prep = enter_vector_stage(VectorStage::QueryPrep);
            let vectors = reader.vector_index(self.field)?;
            let query = self.vector_query(vectors.options().metric());
            let quantized = if self.max_scan_levels > 0 {
                vectors
                    .quantization()?
                    .map(|field| self.quantized_query(field, &query))
            } else {
                None
            };
            (query, quantized)
        };
        let mut backend = SegmentSearcher::for_segment(
            segment,
            self.field,
            query,
            quantized,
            self.probe_params.clone(),
        )?;
        backend.shared_work = self.inputs.shared_work.clone();
        Ok(backend
            .with_collector_topk(
                Arc::clone(&self.collector_topk),
                Arc::clone(&self.shared_topk_threshold),
            )
            .with_candidates(
                self.inputs
                    .candidates
                    .iter()
                    .filter(|c| c.node.0 == segment_ord)
                    .map(|c| crate::vector::Candidate {
                        sim: c.sim,
                        node: c.node.1,
                    })
                    .collect(),
                self.inputs
                    .routing_stats
                    .get(&segment_ord)
                    .map(|s| s.metrics),
            ))
    }

    /// Searches one segment: prepares the query, runs the segment searcher, and lifts each hit's
    /// tie-break key out of the segment.
    fn search_segment(
        &self,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> Result<(KeyedHits<S::SortKey>, SegmentReport, ExactMerge)> {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let backend = self.segment_searcher(segment_ord)?;
        let mut tie_break = self.tie_break.segment_sort_key_computer(reader)?;
        drop(init);
        let (hits, report) = backend.top_n_by(
            self.segment_top_n(),
            &mut tie_break,
            self.tie_break.comparator(),
        )?;
        let merge = {
            let _assembly = enter_vector_stage(VectorStage::ResultAssembly);
            self.collector_topk
                .lock()
                .unwrap()
                .merge_exact(hits.iter().map(|((score, _), address)| {
                    (
                        RowAddress {
                            segment_ord: address.segment_ord,
                            // Real block indices are below u32::MAX; reserve it for seed keys.
                            block_address: BlockAddress::new(u32::MAX, address.doc_id),
                        },
                        *score,
                    )
                }))
        };
        // A segment key can be a term ordinal: lift it before it leaves the segment.
        let results = hits
            .into_iter()
            .map(|((score, segment_key), address)| {
                (
                    (score, tie_break.convert_segment_sort_key(segment_key)),
                    address,
                )
            })
            .collect();
        Ok((results, report, merge))
    }
}

#[allow(clippy::large_enum_variant)]
enum GlobalSource<'a> {
    Router(RouterClusters<'a>),
    RouterBypass(MatchClusters<'a>),
}

/// Searches multiple segments with a shared cluster order, work budget, and top-k score threshold.
/// Returns each segment's hits and statistics so the collector can combine the results.
fn search_global<T, S>(
    collector: &TopDocsByVectorSimilarity<T, S>,
    vectors: &[Arc<VectorIndexReader>],
    segments: &[VectorSegment],
    tie_breaks: &mut [S::Child],
    totals: &mut [StageTotals],
    on_segment: &mut impl FnMut(usize),
) -> Result<Vec<(TieBreakHits<S::Child>, SegmentReport)>>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    let top_n = collector.segment_top_n();
    let backends: Vec<OnceLock<Result<SegmentSearcher<T>>>> =
        (0..vectors.len()).map(|_| OnceLock::new()).collect();
    let backend = |i: usize| {
        backends[i]
            .get_or_init(|| collector.segment_searcher(segments[i].ordinal))
            .as_ref()
            .map_err(Clone::clone)
    };
    let mut global_budget = collector.inputs.prepared.budget.work;
    let ordinals: HashMap<_, _> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| (s.ordinal, i))
        .collect();
    let assigned_rows: usize = if collector
        .inputs
        .prepared
        .segments
        .iter()
        .all(|s| ordinals.contains_key(&s.ordinal))
    {
        collector.inputs.prepared.rows
    } else {
        vectors
            .iter()
            .filter_map(|reader| reader.index())
            .map(|index| index.num_rows())
            .sum()
    };
    global_budget.total = global_budget.total
        * (assigned_rows as f64 / collector.inputs.prepared.budget.rows.max(1) as f64);
    let mut topk = collector.collector_topk.lock().unwrap().clone();
    let mut plans = Vec::with_capacity(vectors.len());
    let mut finished = Vec::with_capacity(vectors.len());
    for (i, (segment, tie_break)) in segments.iter().zip(tie_breaks.iter_mut()).enumerate() {
        let prepared = collector
            .inputs
            .prepared
            .segments
            .iter()
            .find(|s| s.ordinal == segment.ordinal);
        let (result, measured) = measure(|| -> Result<_> {
            match prepared.map(|segment| segment.scan.as_ref().expect("prepared segment")) {
                Some(ScanInput::Flat { .. }) => {
                    let (hits, report) =
                        backend(i)?.top_n_by(top_n, tie_break, collector.tie_break.comparator())?;
                    topk.merge_exact(hits.iter().map(|((score, _), address)| {
                        (
                            RowAddress {
                                segment_ord: address.segment_ord,
                                block_address: BlockAddress::new(u32::MAX, address.doc_id),
                            },
                            *score,
                        )
                    }));
                    Ok((None, Some((hits, report))))
                }
                Some(input) => Ok((Some(input.clone()), None)),
                None => Ok((
                    None,
                    Some((
                        Vec::new(),
                        SegmentReport::Empty {
                            located_cap: collector.inputs.prepared.scope[&segment.ordinal].1,
                        },
                    )),
                )),
            }
        });
        totals[i].append(&measured);
        on_segment(i);
        let (plan, result) = result?;
        plans.push(plan);
        finished.push(result);
    }
    let candidates = collector.inputs.candidates.iter().filter_map(|c| {
        ordinals.get(&c.node.0).map(|&i| {
            (
                i,
                crate::vector::Candidate {
                    sim: c.sim,
                    node: c.node.1,
                },
            )
        })
    });
    let mut reserved = WorkUnits::ZERO;
    let mut scans: Vec<_> = (0..vectors.len()).map(|_| None).collect();
    let mut pending_scans = Vec::with_capacity(vectors.len());
    let mut sources = Vec::with_capacity(vectors.len());
    let mut reports = Vec::with_capacity(vectors.len());
    for (i, (((reader, segment), plan), tie_break)) in vectors
        .iter()
        .zip(segments)
        .zip(plans)
        .zip(tie_breaks.iter_mut())
        .enumerate()
    {
        let (result, measured) = measure(|| -> Result<_> {
            let _init = enter_vector_stage(VectorStage::ScanInit);
            let Some(plan) = plan else {
                return Ok((None, None, None));
            };
            let capacity = plan.capacity_hint();
            let report = match &plan {
                ScanInput::Routed { report, .. } | ScanInput::Located { report, .. } => *report,
                ScanInput::Flat { .. } => unreachable!(),
            };
            let config = ScanConfig {
                adaptive_probing: matches!(&plan, ScanInput::Routed { .. })
                    .then_some(collector.probe_params.adaptive_probing)
                    .flatten(),
                top_n,
                capacity,
                exact_plan: collector.probe_params.exact_plan,
                #[cfg(test)]
                plan_override: None,
                #[cfg(test)]
                admission_pruning_disabled: false,
            };
            let source = match plan {
                ScanInput::Routed { filter, budget, .. } => {
                    GlobalSource::Router(RouterClusters::for_global(
                        reader,
                        budget,
                        filter,
                        &segment.reader,
                        collector
                            .inputs
                            .routing_stats
                            .get(&segment.ordinal)
                            .map(|s| s.metrics)
                            .unwrap_or_else(|| {
                                reader
                                    .index()
                                    .unwrap()
                                    .opened_router()
                                    .kind()
                                    .empty_metrics()
                            }),
                    ))
                }
                ScanInput::Located { docs, budget, .. } => {
                    reserved += budget.total;
                    GlobalSource::RouterBypass(MatchClusters::locate(
                        reader,
                        docs,
                        &segment.reader,
                        budget,
                    )?)
                }
                ScanInput::Flat { .. } => unreachable!(),
            };
            Ok((Some((config, tie_break)), Some(source), Some(report)))
        });
        totals[i].append(&measured);
        on_segment(i);
        let (pending, source, report) = result?;
        pending_scans.push(pending);
        sources.push(source);
        reports.push(report);
    }
    let start_scan = |i: usize, (config, tie_break)| {
        let _init = enter_vector_stage(VectorStage::ScanInit);
        let backend = backend(i)?;
        SegmentScan::new(
            backend.reader(),
            backend.reader().index().unwrap(),
            backend.scoring_query(),
            config,
            PruningBound::new(top_n, Arc::clone(&collector.shared_topk_threshold)),
            backend.segment_ord(),
            tie_break,
            collector.tie_break.comparator(),
        )
    };
    let mut meter = WorkMeter::new(
        WorkBudget {
            total: WorkUnits::new((global_budget.total.get() - reserved.get()).max(0.0)),
            ..global_budget
        },
        None,
    );
    for (i, source) in sources.iter_mut().enumerate() {
        let Some(GlobalSource::RouterBypass(source)) = source else {
            continue;
        };
        let (result, measured) = measure(|| {
            let mut scan = start_scan(i, pending_scans[i].take().unwrap())?;
            scan.swap_topk(&mut topk);
            let _stage = enter_vector_stage(scan.shortlist_stage());
            while let Some(rows) = source.next()? {
                let scored = scan.shortlist(&rows)?;
                source.admitted(scored)?;
            }
            scan.swap_topk(&mut topk);
            scans[i] = Some(scan);
            Ok::<_, TantivyError>(())
        });
        totals[i].append(&measured);
        on_segment(i);
        result?;
    }
    let mut rate = collector
        .probe_params
        .adaptive_probing
        .filter(|_| top_n > 0)
        .map(|_| crate::vector::search::scan::RateApsState::default());
    let mut recall_reached = false;
    for (i, candidate) in candidates {
        if meter.exhausted() || recall_reached {
            break;
        }
        let Some(GlobalSource::Router(source)) = &mut sources[i] else {
            unreachable!()
        };
        let (result, measured) = measure(|| {
            if let Some(pending) = pending_scans[i].take() {
                scans[i] = Some(start_scan(i, pending)?);
            }
            let scan = scans[i].as_mut().unwrap();
            scan.swap_topk(&mut topk);
            scan.swap_rate(&mut rate);
            let _stage = enter_vector_stage(scan.shortlist_stage());
            meter.open();
            if let Some(rows) = source.probe_global(candidate)? {
                let scored = scan.shortlist(&rows)?;
                source.admitted(scored)?;
                meter.rows(scored);
                recall_reached = scan.recall_reached();
            }
            scan.swap_rate(&mut rate);
            scan.swap_topk(&mut topk);
            Ok::<_, TantivyError>(())
        });
        totals[i].append(&measured);
        on_segment(i);
        result?;
        if !recall_reached && rate.as_mut().is_some_and(|rate| rate.refinement_due()) {
            let mut candidates: Vec<_> = scans
                .iter_mut()
                .flatten()
                .flat_map(|scan| scan.refinement_candidates(top_n))
                .collect();
            if candidates.len() > top_n {
                candidates.select_nth_unstable_by(top_n, |a, b| {
                    b.bound
                        .estimate()
                        .total_cmp(&a.bound.estimate())
                        .then_with(|| a.address.cmp(&b.address))
                });
                candidates.truncate(top_n);
            }
            for (j, scan) in scans.iter_mut().enumerate() {
                let Some(scan) = scan else { continue };
                scan.swap_topk(&mut topk);
                let (result, measured) = measure(|| scan.refine_topk(&candidates));
                scan.swap_topk(&mut topk);
                totals[j].append(&measured);
                on_segment(j);
                result?;
            }
            let scan = scans[i].as_mut().unwrap();
            scan.swap_rate(&mut rate);
            recall_reached = scan.recall_reached();
            scan.swap_rate(&mut rate);
        }
    }
    for (i, scan) in scans.iter_mut().enumerate() {
        let Some(scan) = scan else {
            continue;
        };
        scan.swap_topk(&mut topk);
        let (result, measured) = measure(|| scan.refine_shortlist());
        scan.swap_topk(&mut topk);
        totals[i].append(&measured);
        on_segment(i);
        result?;
    }
    for (i, ((scan, source), report)) in scans.into_iter().zip(sources).zip(reports).enumerate() {
        let Some(source) = source else {
            continue;
        };
        let (result, measured) = measure(|| {
            let source = match source {
                GlobalSource::Router(source) => {
                    let mut report = source.finish();
                    report.probe.termination = if recall_reached {
                        ProbeTermination::QuantizedRecall
                    } else if meter.exhausted()
                        && report.counters.postings_row + report.counters.postings_skipped
                            < vectors[i].index().unwrap().num_clusters()
                    {
                        ProbeTermination::Ceiling
                    } else {
                        ProbeTermination::Exhausted
                    };
                    SourceReport::Router(report)
                }
                GlobalSource::RouterBypass(source) => SourceReport::Matches(source.finish()),
            };
            let (hits, scan) = match scan {
                Some(scan) => scan.finish(segments[i].ordinal)?,
                None => {
                    let layers = match vectors[i].metadata() {
                        Some(VectorColMetadata::Quantized { layers, .. }) => {
                            collector.max_scan_levels.min(layers.len())
                        }
                        _ => 0,
                    };
                    (
                        Vec::new(),
                        crate::vector::search::scan::ScanReport {
                            layers: vec![Default::default(); layers],
                            ..Default::default()
                        },
                    )
                }
            };
            Ok::<_, TantivyError>((
                hits,
                SegmentReport::Ivf {
                    plan: report.unwrap(),
                    source,
                    scan: Box::new(scan),
                },
            ))
        });
        totals[i].append(&measured);
        on_segment(i);
        finished[i] = Some(result?);
    }
    Ok(finished.into_iter().map(Option::unwrap).collect())
}

/// Satisfies `Collector::Child`; [`Collector::collect_segment`] is overridden, so it never runs.
pub struct NoOpSegmentCollector<K>(PhantomData<K>);

impl<K> Default for NoOpSegmentCollector<K> {
    fn default() -> Self {
        NoOpSegmentCollector(PhantomData)
    }
}

impl<K: 'static + Send> SegmentCollector for NoOpSegmentCollector<K> {
    type Fruit = SegmentVectorFruit<K>;
    fn collect(&mut self, _doc: DocId, _score: Score) {}
    fn harvest(self) -> Self::Fruit {
        SegmentVectorFruit {
            results: Vec::new(),
            stats: ProbeStats::default(),
        }
    }
}

#[cfg(test)]
mod ivf_e2e_tests {
    //! End-to-end coverage: drives the full
    //! `searcher.search → TopDocsByVectorSimilarity → collect_segment
    //! → IvfBackend::top_n → merge_fruits` path against the shared
    //! `TestVectorIndex` fixture and asserts the resulting global
    //! top-K matches `index.ground_truth(...)`. Built on the shared
    //! fixture so the manual flat/ivf scene construction the
    //! pre-consolidation tests carried is gone — `vector_storage_format`
    //! is the only knob.
    use std::ops::Range;
    use std::sync::Arc;

    use super::{ScanInput, VectorSegment, VectorSimilarityFruit};
    use crate::collector::sort_key::{SortBySimilarityScore, SortByStaticFastValue, SortByString};
    use crate::collector::{AtomicSharedThreshold, Collector, SharedThresholdArc, TopDocs};
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query};
    use crate::schema::{Field, Metric, Schema, VectorDType, VectorOptions, FAST, STORED, STRING};
    use crate::vector::diagnostics::VectorStorageFormat;
    use crate::vector::distance::l2_squared_bytes;
    use crate::vector::params::{LocatedPath, ProbeParams};
    use crate::vector::stats::AccessPath;
    use crate::vector::test_support::{FixtureSpec, Layout};
    use crate::vector::tests::{exhaustive_params, ground_truth, Grid2DClusterer, TestVectorIndex};
    use crate::vector::{
        Candidate, ClusterRef, RouterKind, TopClustersBySimilarityCollector, VectorSearchBudget,
    };
    use crate::{
        DocAddress, Index, IndexWriter, Order, Result, Score, TantivyDocument, TantivyError,
    };

    fn build_routed<T, S>(
        builder: super::TopDocsByVectorSimilarityBuilder<T, S>,
        weight: &dyn crate::query::Weight,
        segments: &[VectorSegment],
        distributed: bool,
    ) -> Result<super::TopDocsByVectorSimilarity<T, S>>
    where
        T: crate::vector::VectorElement,
        S: crate::collector::SortKeyComputer + Send + Sync + 'static,
    {
        let budget =
            VectorSearchBudget::for_segments(builder.field, &builder.probe_params, segments)?;
        let prepared = builder.prepare(weight, segments, budget, |_| {})?;
        let router = TopClustersBySimilarityCollector::new(
            builder.field,
            builder.query.as_ref().clone(),
            builder.probe_params.clone(),
            prepared.match_fraction(),
        )?
        .with_budget(budget);
        let candidates = router.collect_segments(&prepared.routing_segments(), |_| {})?;
        let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> =
            distributed.then(|| {
                Arc::new(crate::vector::AtomicVectorWork::default())
                    as Arc<dyn crate::vector::SharedVectorWork>
            });
        Ok(builder.build(
            prepared
                .with_candidates(candidates, shared_work.clone())?
                .with_routing_statistics(router.take_statistics()),
        ))
    }

    /// IVF + exhaustive probing matches the global oracle. Exact ranks
    /// every centroid; Stacked's flat-top 25% prune is approximate and is
    /// not expected to match. The shared fixture produces multiple IVF
    /// segments (it merges raw segments pairwise), so this single test
    /// already exercises cross-segment merge_fruits.
    #[test]
    fn e2e_ivf_matches_global_oracle() -> Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .router(RouterKind::Exact)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let params = exhaustive_params(9);
        for query in [[0.5_f32, 0.5], [9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let expected = index.ground_truth(query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(index.embedding_field(), query.to_vec())
                    .with_probe_params(params.clone())
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap();
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "IVF query={query:?} k={k}");
            }
        }
        Ok(())
    }

    #[test]
    fn global_search_preserves_filtered_quantized_results() -> Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let fixture = FixtureSpec::clustered(metric, 128, 8, 768)
                .segments(3)
                .schedule(&[1, 1])
                .label("rare", |i| i % 19 == 0)
                .deletes(&[0, 19, 201, 700])
                .without_vectors(|i| i % 23 == 0)
                .build()?;
            let searcher = fixture.index.reader()?.searcher();
            let mut readers: Vec<_> = searcher
                .segment_readers()
                .iter()
                .enumerate()
                .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                .collect();
            readers.reverse();
            let mut query = vec![0.01; 128];
            query[2] = 1.0;
            for filter in [
                &AllQuery as &dyn Query,
                &fixture.filter("rare"),
                &fixture.filter("absent"),
            ] {
                for (located, exact_plan) in [
                    (LocatedPath::Off, false),
                    (LocatedPath::Auto, false),
                    (LocatedPath::Off, true),
                    (LocatedPath::Auto, true),
                ] {
                    let params = ProbeParams {
                        located,
                        exact_plan,
                        ..exhaustive_params(8)
                    };
                    for batch_size in [1, 2, readers.len()] {
                        let threshold: SharedThresholdArc<Score> =
                            Arc::new(AtomicSharedThreshold::default());
                        let collector = TopDocs::with_limit(7)
                            .and_offset(3)
                            .order_by_similarity(fixture.field, query.clone())
                            .with_probe_params(params.clone())
                            .with_shared_threshold(Arc::clone(&threshold))
                            .build_for_searcher(&searcher, filter)?;
                        let weight = fixture.weight(filter)?;
                        let mut fruits = Vec::new();
                        for batch in readers.chunks(batch_size) {
                            fruits.extend(collector.collect_segments(
                                weight.as_ref(),
                                batch,
                                |_| {},
                            )?);
                        }
                        let fruit = collector.merge_fruits(fruits)?;
                        let expected: Vec<_> = fixture
                            .oracle(filter, &query, 10)?
                            .into_iter()
                            .skip(3)
                            .collect();
                        assert_eq!(
                            fruit.results, expected,
                            "{metric:?} {located:?} batch={batch_size}"
                        );
                        assert_eq!(fruit.stats.len(), readers.len());
                        assert_eq!(threshold.load().is_some(), !fruit.results.is_empty());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn global_search_only_prepares_probed_segments() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 128, 16, 1024)
            .segments(4)
            .schedule(&[1, 4])
            .build()?;
        let mut query = vec![0.0; 128];
        query[3] = 1.0;
        for max_scan_levels in [0, 2] {
            let searcher = fixture.index.reader()?.searcher();
            let segments: Vec<_> = searcher
                .segment_readers()
                .iter()
                .enumerate()
                .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                .collect();
            let vectors: Vec<_> = segments
                .iter()
                .map(|s| s.reader.vector_index(fixture.field))
                .collect::<Result<_>>()?;
            assert!(vectors
                .iter()
                .all(|v| v.has_quantized_storage() && !v.quantization_initialized()));
            let collector = TopDocs::with_limit(5)
                .order_by_similarity(fixture.field, query.clone())
                .with_max_scan_levels(max_scan_levels)
                .with_probe_params(ProbeParams {
                    max_probe: 1,
                    min_probe_clusters: 0,
                    exact_plan: false,
                    located: LocatedPath::Off,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap();
            let weight = fixture.all()?;
            let fruit = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &segments,
                |_| {},
            )?)?;
            assert_eq!(fruit.results.len(), 5);
            assert_eq!(
                fruit.stats.iter().filter(|s| s.work_charged > 0.0).count(),
                1
            );
            for (vectors, stats) in vectors.iter().zip(&fruit.stats) {
                assert_eq!(
                    vectors.quantization_initialized(),
                    max_scan_levels > 0 && stats.work_charged > 0.0
                );
                assert!(stats.layers.get(max_scan_levels).is_none());
                for layer in 0..max_scan_levels {
                    assert!(stats.layers.get(layer).is_some());
                }
            }
            let exhaustive = TopDocs::with_limit(5)
                .order_by_similarity(fixture.field, query.clone())
                .with_max_scan_levels(max_scan_levels)
                .with_probe_params(exhaustive_params(16))
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap();
            let fruit = exhaustive.merge_fruits(exhaustive.collect_segments(
                weight.as_ref(),
                &segments,
                |_| {},
            )?)?;
            assert_eq!(fruit.results, fixture.oracle(&AllQuery, &query, 5)?);
            assert!(vectors
                .iter()
                .all(|v| v.quantization_initialized() == (max_scan_levels > 0)));
        }
        let searcher = fixture.index.reader()?.searcher();
        let segments: Vec<_> = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
            .collect();
        let collector = TopDocs::with_limit(5)
            .order_by_similarity(fixture.field, query)
            .build_for_searcher(&searcher, &fixture.filter("absent"))?;
        let weight = fixture.weight(&fixture.filter("absent"))?;
        let fruit = collector.merge_fruits(collector.collect_segments(
            weight.as_ref(),
            &segments,
            |_| {},
        )?)?;
        assert!(fruit.results.is_empty());
        for segment in segments {
            assert!(!segment
                .reader
                .vector_index(fixture.field)?
                .quantization_initialized());
        }
        Ok(())
    }

    #[test]
    fn global_adaptive_stopping_shares_evidence_across_segments() -> Result<()> {
        use crate::vector::{AdaptiveProbeParams, ProbeTermination};

        let vector = |x| {
            let mut v = vec![0.0; 64];
            v[0] = x;
            v
        };
        let fixture = FixtureSpec::ivf(
            Metric::L2,
            (0..64).map(|c| vector(c as f32 * 10.0)).collect(),
            (0..19392)
                .map(|i| {
                    let mut v = vector((i % 64) as f32 * 10.0);
                    if i / 64 % 10 == 9 {
                        v[1] = 1000.0;
                    }
                    v
                })
                .collect(),
        )
        .segments(3)
        .label("half", |i| i % 3 != 0)
        .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let readers: Vec<_> = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
            .collect();
        for filter in [&AllQuery as &dyn Query, &fixture.filter("half")] {
            let query = vector(0.0);
            let weight = fixture.weight(filter)?;
            for target in [0.9, 1.0] {
                let collector = TopDocs::with_limit(10)
                    .order_by_similarity(fixture.field, query.clone())
                    .with_probe_params(ProbeParams {
                        adaptive_probing: (target < 1.0).then(|| AdaptiveProbeParams {
                            admission_fraction: (1.0_f64 - target).powi(4),
                        }),
                        located: LocatedPath::Off,
                        ..exhaustive_params(64)
                    })
                    .build_for_searcher(&searcher, filter)?;
                let result = collector.merge_fruits(collector.collect_segments(
                    weight.as_ref(),
                    &readers,
                    |_| {},
                )?)?;
                assert_eq!(result.results, fixture.oracle(filter, &query, 10)?);
                let probed: usize = result.stats.iter().map(|s| s.clusters_probed()).sum();
                if target < 1.0 {
                    assert!(result
                        .stats
                        .iter()
                        .all(|s| s.termination == ProbeTermination::QuantizedRecall));
                    assert!(
                        probed < 32,
                        "global admission history should stop across segment switches: {probed}"
                    );
                    assert!(result.stats.iter().all(|s| s.candidates_scored > 0));
                } else {
                    assert_eq!(probed, 192);
                    assert!(result
                        .stats
                        .iter()
                        .all(|s| s.termination != ProbeTermination::QuantizedRecall));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn global_search_refines_during_probing_only_with_adaptive() -> Result<()> {
        use crate::vector::VectorStage;

        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let fixture = FixtureSpec::clustered(metric, 128, 8, 768)
                .router(crate::vector::RouterKind::QuantizedStacked)
                .segments(3)
                .schedule(&[1, 1])
                .label("matches", |i| i % 3 == 0)
                .paged()
                .build()?;
            let searcher = fixture.index.reader()?.searcher();
            let mut readers: Vec<_> = searcher
                .segment_readers()
                .iter()
                .enumerate()
                .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                .collect();
            readers.reverse();
            let mut query = vec![0.01; 128];
            query[2] = 1.0;
            for filter in [&AllQuery as &dyn Query, &fixture.filter("matches")] {
                for (located, adaptive) in [
                    (LocatedPath::Off, false),
                    (LocatedPath::Auto, false),
                    (LocatedPath::Off, true),
                    (LocatedPath::Auto, true),
                ] {
                    let collector = TopDocs::with_limit(10)
                        .order_by_similarity(fixture.field, query.clone())
                        .with_probe_params(ProbeParams {
                            located,
                            exact_plan: false,
                            router_min_beam_fraction: 1.0,
                            router_aps: false,
                            adaptive_probing: adaptive.then_some(
                                crate::vector::AdaptiveProbeParams {
                                    admission_fraction: 1e-16,
                                },
                            ),
                            ..exhaustive_params(8)
                        })
                        .build_for_searcher(&searcher, filter)?;
                    let weight = fixture.weight(filter)?;
                    fixture.clear_reads();
                    let result = collector.merge_fruits(collector.collect_segments(
                        weight.as_ref(),
                        &readers,
                        |_| {},
                    )?)?;
                    let reads = fixture.reads();
                    let last_shortlist = reads
                        .iter()
                        .rposition(|(stage, _)| *stage == VectorStage::LayerScan(0))
                        .expect("layer-0 reads");
                    let first_refinement = reads
                        .iter()
                        .position(|(stage, _)| {
                            matches!(
                                stage,
                                VectorStage::LayerScan(1)
                                    | VectorStage::ExactScan
                                    | VectorStage::ExactFetch
                                    | VectorStage::ExactScore
                            )
                        })
                        .expect("refinement reads");
                    let refines_during_probing = adaptive
                        && result.stats.iter().any(|stats| {
                            stats.access_path == crate::vector::stats::AccessPath::Routed
                        });
                    assert_eq!(
                        first_refinement < last_shortlist,
                        refines_during_probing,
                        "{located:?}"
                    );
                    assert_eq!(result.results, fixture.oracle(filter, &query, 10)?);
                    let mut refined = 0;
                    for stats in &result.stats {
                        let trace = &stats.quantized_trace;
                        let rows: std::collections::HashSet<_> =
                            trace.refined_topk_rows.iter().copied().collect();
                        assert_eq!(rows.len(), trace.refined_topk_rows.len());
                        assert!(trace
                            .boundary_rows
                            .iter()
                            .flatten()
                            .all(|row| !rows.contains(row)));
                        assert!(trace.exact_phase_rows.iter().all(|row| !rows.contains(row)));
                        refined += rows.len();
                    }
                    assert_eq!(refined > 0, refines_during_probing);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn located_work_cannot_starve_a_routed_segment() -> Result<()> {
        let mut weak = FixtureSpec::clustered(Metric::L2, 32, 8, 128).label("matches", |i| i < 16);
        for vector in weak.vectors.iter_mut().flatten() {
            for value in vector {
                *value += 10.0;
            }
        }
        if let Layout::Ivf(centroids) = &mut weak.layout {
            for value in centroids.iter_mut().flatten() {
                *value += 10.0;
            }
        }
        let weak_a = weak.build()?;
        let weak_b = weak.build()?;
        let near = FixtureSpec::clustered(Metric::L2, 32, 8, 128)
            .label("matches", |_| true)
            .build()?;
        let searchers = [
            weak_a.index.reader()?.searcher(),
            weak_b.index.reader()?.searcher(),
            near.index.reader()?.searcher(),
        ];
        let readers: Vec<_> = searchers
            .iter()
            .enumerate()
            .map(|(i, s)| VectorSegment::from((i as u32, &s.segment_readers()[0])))
            .collect();
        let mut query = vec![0.0; 32];
        query[5] = 1.0;
        let params = ProbeParams {
            max_probe: 1,
            min_probe_clusters: 3,
            ..Default::default()
        };
        for planned in [false, true] {
            let builder = TopDocs::with_limit(3)
                .order_by_similarity(near.field, query.clone())
                .with_probe_params(params.clone());
            let weight = near.weight(&near.filter("matches"))?;
            let collector = build_routed(builder, weight.as_ref(), &readers, planned)?;
            let result = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &readers,
                |_| {},
            )?)?;
            assert_eq!(result.results.len(), 3);
            assert!(result
                .results
                .iter()
                .all(|(_, address)| address.segment_ord == 2));
            assert_eq!(result.stats[0].access_path, AccessPath::Located);
            assert_eq!(result.stats[1].access_path, AccessPath::Located);
            assert_eq!(result.stats[2].access_path, AccessPath::Routed);
            assert!(result.stats[2].work_charged > 0.0);
        }
        Ok(())
    }

    #[test]
    fn partitioned_parallel_search_preserves_serial_recall_and_work() -> Result<()> {
        let fixtures = (0..8)
            .map(|partition| {
                let centroids: Vec<_> = (0..8)
                    .map(|cluster| {
                        let mut vector = vec![0.0; 32];
                        vector[0] = (partition * 100 + cluster) as f32;
                        vector
                    })
                    .collect();
                let vectors = (0..256)
                    .map(|row| {
                        let mut vector = centroids[row % 8].clone();
                        vector[1] = row as f32 * 0.0001;
                        vector
                    })
                    .collect();
                FixtureSpec::ivf(Metric::L2, centroids, vectors)
                    .label("half", |row| (row / 8) % 2 == 0)
                    .build()
            })
            .collect::<Result<Vec<_>>>()?;
        let searchers = fixtures
            .iter()
            .map(|fixture| fixture.index.reader().map(|reader| reader.searcher()))
            .collect::<Result<Vec<_>>>()?;
        let segments: Vec<_> = searchers
            .iter()
            .enumerate()
            .map(|(ordinal, searcher)| {
                VectorSegment::from((ordinal as u32, searcher.segment_reader(0)))
            })
            .collect();
        let fixture = &fixtures[0];
        let mut query = vec![0.0f32; 32];
        query[0] = 1.5;
        let params = ProbeParams {
            max_probe: 4,
            located: LocatedPath::Off,
            ..Default::default()
        };
        for (filter, fraction) in [
            (&AllQuery as &dyn Query, 1.0),
            (&fixture.filter("half"), 0.5),
        ] {
            let weight = fixture.weight(filter)?;
            let builder = || {
                TopDocs::with_limit(10)
                    .order_by_similarity(fixture.field, query.clone())
                    .with_probe_params(params.clone())
            };
            let serial = builder().build_for_segments(weight.as_ref(), &segments, |_| {})?;
            let serial = serial.merge_fruits(serial.collect_segments(
                weight.as_ref(),
                &segments,
                |_| {},
            )?)?;
            let budget = VectorSearchBudget::for_segments(fixture.field, &params, &segments)?;
            let worker_builder = || {
                TopDocs::with_limit(10)
                    .order_by_similarity(fixture.field, query.clone())
                    .with_probe_params(ProbeParams {
                        max_probe: 1,
                        ..params.clone()
                    })
            };
            let prepared = segments
                .iter()
                .map(|segment| {
                    worker_builder().prepare(
                        weight.as_ref(),
                        std::slice::from_ref(segment),
                        budget,
                        |_| {},
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let routed: Vec<_> = prepared.iter().flat_map(|p| p.routing_segments()).collect();
            let router = TopClustersBySimilarityCollector::new(
                fixture.field,
                query.clone(),
                params.clone(),
                fraction,
            )?
            .with_budget(budget);
            let candidates = router.collect_segments(&routed, |_| {})?;
            let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> =
                Some(Arc::new(crate::vector::AtomicVectorWork::default()));
            let fruits = std::thread::scope(|scope| {
                segments
                    .iter()
                    .zip(prepared)
                    .map(|(segment, prepared)| {
                        let inputs = prepared.with_candidates(
                            candidates
                                .iter()
                                .filter(|c| c.node.0 == segment.ordinal)
                                .copied()
                                .collect(),
                            shared_work.clone(),
                        )?;
                        let collector = worker_builder().build(inputs);
                        let weight = &weight;
                        Ok(scope.spawn(move || {
                            collector.collect_segment(
                                weight.as_ref(),
                                segment.ordinal,
                                &segment.reader,
                            )
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|thread| thread.join().unwrap())
                    .collect::<Result<Vec<_>>>()
            })?;
            let parallel = builder()
                .build_for_segments(weight.as_ref(), &[], |_| {})?
                .merge_fruits(fruits)?;
            let expected = fixture.oracle(filter, &query, 10)?;
            let recall = |fruit: &VectorSimilarityFruit| {
                fruit
                    .results
                    .iter()
                    .filter(|(_, doc)| expected.iter().any(|(_, wanted)| wanted == doc))
                    .count() as f64
                    / 10.0
            };
            assert_eq!(recall(&serial), 1.0);
            assert_eq!(recall(&parallel), recall(&serial));
            let work = |fruit: &VectorSimilarityFruit| {
                fruit.stats.iter().map(|s| s.work_charged).sum::<f32>()
            };
            assert!(
                (work(&parallel) - work(&serial)).abs() < 1e-5,
                "fraction={fraction} parallel={} serial={} stats={:?}",
                work(&parallel),
                work(&serial),
                parallel
                    .stats
                    .iter()
                    .map(|s| (s.work_budget, s.work_charged, s.candidates_scored))
                    .collect::<Vec<_>>()
            );
            assert_eq!(parallel.stats[0].work_budget, 4.0);
            assert!(parallel.stats[1..]
                .iter()
                .all(|s| s.work_budget == 0.0 && s.work_charged == 0.0));
        }
        Ok(())
    }

    #[test]
    fn routing_candidates_preserves_results_across_parallel_segments() -> Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let fixture = FixtureSpec::clustered(metric, 64, 8, 384)
                .label("matches", |i| i % 3 == 0)
                .segments(3)
                .schedule(&[1, 1])
                .deletes(&[0, 3, 80])
                .build()?;
            let searcher = fixture.index.reader()?.searcher();
            let readers: Vec<_> = searcher
                .segment_readers()
                .iter()
                .enumerate()
                .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                .collect();
            let mut query = vec![0.01; 64];
            query[2] = 1.0;
            for filter in [&AllQuery as &dyn Query, &fixture.filter("matches")] {
                for located in [LocatedPath::Off, LocatedPath::Auto] {
                    let params = ProbeParams {
                        located,
                        ..exhaustive_params(8)
                    };
                    let builder = || {
                        TopDocs::with_limit(7)
                            .and_offset(3)
                            .order_by_similarity(fixture.field, query.clone())
                            .with_probe_params(params.clone())
                    };
                    let weight = fixture.weight(filter)?;
                    let budget =
                        VectorSearchBudget::for_segments(fixture.field, &params, &readers)?;
                    let prepared = readers
                        .iter()
                        .map(|segment| {
                            builder().prepare(
                                weight.as_ref(),
                                std::slice::from_ref(segment),
                                budget,
                                |_| {},
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let routed = prepared
                        .iter()
                        .flat_map(|p| p.routing_segments())
                        .collect::<Vec<_>>();
                    let router = TopClustersBySimilarityCollector::new(
                        fixture.field,
                        query.clone(),
                        params.clone(),
                        0.0,
                    )?
                    .with_budget(budget);
                    let candidates = router.collect_segments(&routed, |_| {})?;
                    let bytes = serde_json::to_vec(&candidates).unwrap();
                    let candidates: Vec<Candidate<ClusterRef>> =
                        serde_json::from_slice(&bytes).unwrap();
                    let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> =
                        Some(Arc::new(crate::vector::AtomicVectorWork::default()));
                    let threshold: SharedThresholdArc<Score> =
                        Arc::new(AtomicSharedThreshold::default());
                    let fruits = std::thread::scope(|scope| {
                        readers
                            .iter()
                            .zip(prepared)
                            .rev()
                            .map(|(segment, prepared)| {
                                let (ord, reader) = (segment.ordinal, &segment.reader);
                                let input = prepared
                                    .with_candidates(
                                        candidates
                                            .iter()
                                            .filter(|c| c.node.0 == ord)
                                            .copied()
                                            .collect(),
                                        shared_work.clone(),
                                    )
                                    .unwrap();
                                let collector = builder()
                                    .with_shared_threshold(Arc::clone(&threshold))
                                    .build(input);
                                let weight = &weight;
                                scope.spawn(move || {
                                    collector.collect_segment(weight.as_ref(), ord, reader)
                                })
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|thread| thread.join().unwrap())
                            .collect::<Result<Vec<_>>>()
                    })?;
                    let fruit = builder()
                        .build_for_segments(weight.as_ref(), &[], |_| {})?
                        .merge_fruits(fruits)?;
                    let expected: Vec<_> = fixture
                        .oracle(filter, &query, 10)?
                        .into_iter()
                        .skip(3)
                        .collect();
                    assert_eq!(fruit.results, expected, "{metric:?}");
                    assert_eq!(fruit.stats.len(), readers.len());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn routing_candidates_leaves_row_reads_and_budget_checks_to_consumers() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 64, 8, 768)
            .segments(3)
            .schedule(&[1, 1])
            .label("matches", |i| i % 3 == 0)
            .paged()
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let readers: Vec<_> = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
            .collect();
        for segment in &readers {
            segment.reader.vector_index(fixture.field)?;
        }
        let mut query = vec![0.0; 64];
        query[2] = 1.0;
        let params = ProbeParams {
            max_probe: 3,
            located: LocatedPath::Off,
            exact_plan: false,
            ..Default::default()
        };
        let builder = || {
            TopDocs::with_limit(5)
                .order_by_similarity(fixture.field, query.clone())
                .with_probe_params(params.clone())
        };
        fixture.clear_reads();
        let router = TopClustersBySimilarityCollector::new(
            fixture.field,
            query.clone(),
            params.clone(),
            0.0,
        )?;
        let candidates = router.collect_segments(&readers, |_| {})?;
        let planned = readers
            .iter()
            .map(|s| candidates.iter().filter(|c| c.node.0 == s.ordinal).count())
            .collect::<Vec<_>>();
        assert!(
            fixture.reads().is_empty(),
            "routing read vector rows or document IDs"
        );
        assert!(!fixture.centroid_reads().is_empty());
        let statistics = router.take_statistics();
        assert!(statistics.iter().all(|s| s
            .totals
            .time_ns(crate::vector::VectorStage::NonVectorSearch)
            == 0));
        let filter = fixture.filter("matches");
        let weight = fixture.weight(&filter)?;
        let budget = VectorSearchBudget::for_segments(fixture.field, &params, &readers)?;
        let prepared = builder().prepare(weight.as_ref(), &readers, budget, |_| {})?;
        let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> =
            Some(Arc::new(crate::vector::AtomicVectorWork::default()));
        let collector = builder().build(
            prepared
                .with_candidates(candidates, shared_work.clone())?
                .with_routing_statistics(statistics),
        );
        let fruit = collector.merge_fruits(collector.collect_segments(
            weight.as_ref(),
            &readers,
            |_| {},
        )?)?;
        assert!(!fixture.reads().is_empty());
        let charged = fruit.stats.iter().map(|s| s.work_charged).sum::<f32>();
        assert!((3.0..=4.0).contains(&charged));
        assert!((shared_work.as_ref().unwrap().charged() - f64::from(charged)).abs() < 1e-5);
        assert!(fruit.stats.iter().any(|s| s.pruned_filter > 0));
        assert!(fruit
            .stats
            .iter()
            .any(|s| s.termination == crate::vector::ProbeTermination::Ceiling));
        for (stats, count) in fruit.stats.iter().zip(planned) {
            assert_eq!(stats.planned_clusters, count);
        }
        assert!(fruit
            .stats
            .iter()
            .any(|s| s.postings_row + s.postings_skipped < s.planned_clusters));
        let exact = fixture.oracle(&filter, &query, 768)?;
        assert_eq!(fruit.results.len(), 5);
        assert!(fruit.results.iter().all(|hit| exact.contains(hit)));
        Ok(())
    }

    #[test]
    fn routing_candidates_shares_absolute_budget_across_unequal_segments() -> Result<()> {
        let fixtures = [4, 8, 16]
            .map(|clusters| FixtureSpec::clustered(Metric::L2, 32, clusters, clusters * 32).build())
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let searchers = fixtures
            .iter()
            .map(|fixture| fixture.index.reader().map(|reader| reader.searcher()))
            .collect::<Result<Vec<_>>>()?;
        let segments = searchers
            .iter()
            .enumerate()
            .map(|(ordinal, searcher)| {
                VectorSegment::from((ordinal as u32, searcher.segment_reader(0)))
            })
            .collect::<Vec<_>>();
        let field = fixtures[0].field;
        let weight = fixtures[0].all()?;
        for max_probe in [3, 8, 100] {
            let params = ProbeParams {
                max_probe,
                located: LocatedPath::Off,
                ..Default::default()
            };
            let builder = TopDocs::with_limit(3)
                .order_by_similarity(field, vec![0.0f32; 32])
                .with_probe_params(params.clone());
            let budget = VectorSearchBudget::for_segments(field, &params, &segments)?;
            let prepared = builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
            let total = max_probe.min(28) as f64;
            for (segment, clusters) in prepared.segments.iter().zip([4, 8, 16]) {
                let Some(ScanInput::Routed { budget, .. }) = &segment.scan else {
                    panic!("routed input")
                };
                assert!((budget.total.get() - total * clusters as f64 / 28.0).abs() < 1e-5);
            }
            let collector = build_routed(builder, weight.as_ref(), &segments, true)?;
            let fruit = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &segments,
                |_| {},
            )?)?;
            assert!(fruit
                .stats
                .iter()
                .all(|s| f64::from(s.work_budget) <= total + 1e-5));
            assert!(
                f64::from(fruit.stats.iter().map(|s| s.work_charged).sum::<f32>()) <= total + 1.0
            );
        }
        Ok(())
    }

    #[test]
    fn direct_scans_validate_schema_when_visited() -> Result<()> {
        let fixture = FixtureSpec::flat(Metric::L2, 2, vec![Some(vec![0.0, 1.0])]).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        let segments = [VectorSegment::from((0, reader))];
        for filter in [&AllQuery as &dyn Query, &fixture.filter("absent")] {
            let weight = fixture.weight(filter)?;
            let wrong_length = TopDocs::with_limit(1)
                .order_by_similarity(fixture.field, vec![0.0f32])
                .build_for_segments(weight.as_ref(), &segments, |_| {})?;
            for error in [
                wrong_length
                    .collect_segment(weight.as_ref(), 0, reader)
                    .err(),
                wrong_length
                    .collect_segments(weight.as_ref(), &segments, |_| {})
                    .err(),
            ] {
                if wrong_length.inputs.segments().is_empty() {
                    assert!(error.is_none());
                } else {
                    assert!(error.unwrap().to_string().contains("query vector length"));
                }
            }
            assert!(searcher.search(filter, &wrong_length).is_err());
        }
        Ok(())
    }

    #[test]
    fn partition_centroids_are_only_validated_for_routing() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 128)
            .label("matches", |i| i < 2)
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        for centroid in [vec![0.0f32; 31], vec![f32::NAN; 32]] {
            let segments = [VectorSegment {
                partition_centroid: Some(centroid.into()),
                ..VectorSegment::from((0, reader))
            }];
            for label in ["matches", "absent"] {
                let filter = fixture.filter(label);
                let weight = fixture.weight(&filter)?;
                let collector = TopDocs::with_limit(2)
                    .order_by_similarity(fixture.field, vec![0.0f32; 32])
                    .build_for_segments(weight.as_ref(), &segments, |_| {})?;
                assert!(collector.inputs.routing_stats.is_empty());
                let fruit = collector.merge_fruits(collector.collect_segments(
                    weight.as_ref(),
                    &segments,
                    |_| {},
                )?)?;
                assert_eq!(fruit.results, fixture.oracle(&filter, &[0.0; 32], 2)?);
            }
            let error = TopDocs::with_limit(2)
                .order_by_similarity(fixture.field, vec![0.0f32; 32])
                .with_probe_params(ProbeParams {
                    located: LocatedPath::Off,
                    ..Default::default()
                })
                .build_for_segments(fixture.all()?.as_ref(), &segments, |_| {})
                .err()
                .expect("routing must validate the centroid");
            assert!(error
                .to_string()
                .contains("invalid vector partition centroid"));
        }
        Ok(())
    }

    #[test]
    fn prepared_filters_are_not_evaluated_again() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::query::{Explanation, Scorer, Weight};
        use crate::vector::test_support::DocsWeight;
        use crate::{DocId, DocSet, SegmentReader, TERMINATED};

        struct TrackedFilter {
            weight: DocsWeight,
            consumed: Arc<AtomicUsize>,
            calls: Arc<AtomicUsize>,
        }
        struct TrackedScorer {
            inner: Box<dyn Scorer>,
            consumed: Arc<AtomicUsize>,
        }
        impl DocSet for TrackedScorer {
            fn advance(&mut self) -> DocId {
                if self.inner.doc() != TERMINATED {
                    self.consumed.fetch_add(1, Ordering::SeqCst);
                }
                self.inner.advance()
            }
            fn doc(&self) -> DocId {
                self.inner.doc()
            }
            fn size_hint(&self) -> u32 {
                self.inner.size_hint()
            }
        }
        impl Scorer for TrackedScorer {
            fn score(&mut self) -> Score {
                self.inner.score()
            }
        }
        impl Weight for TrackedFilter {
            fn scorer(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(TrackedScorer {
                    inner: self.weight.scorer(reader, boost)?,
                    consumed: Arc::clone(&self.consumed),
                }))
            }
            fn explain(&self, _: &SegmentReader, _: DocId) -> Result<Explanation> {
                unreachable!()
            }
        }
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 256).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        let docs: Vec<_> = (0..reader.max_doc()).step_by(3).collect();
        let consumed = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let filter = TrackedFilter {
            weight: DocsWeight {
                max_doc: reader.max_doc(),
                docs: docs.clone(),
            },
            consumed: Arc::clone(&consumed),
            calls: Arc::clone(&calls),
        };
        let params = ProbeParams {
            located: LocatedPath::Off,
            ..exhaustive_params(8)
        };
        let builder = || {
            TopDocs::with_limit(5)
                .order_by_similarity(fixture.field, vec![0.0; 32])
                .with_probe_params(params.clone())
        };
        let segments = [VectorSegment::from((0, reader))];
        let budget = VectorSearchBudget::for_segments(fixture.field, &params, &segments)?;
        let prepared = builder().prepare(&filter, &segments, budget, |_| {})?;
        assert_eq!(consumed.load(Ordering::SeqCst), docs.len());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let router = TopClustersBySimilarityCollector::new(
            fixture.field,
            vec![0.0; 32],
            params.clone(),
            prepared.match_fraction(),
        )?
        .with_budget(prepared.budget());
        let candidates = router.collect_segments(&prepared.routing_segments(), |_| {})?;
        let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> = None;
        let collector = builder().build(
            prepared
                .with_candidates(candidates, shared_work.clone())?
                .with_routing_statistics(router.take_statistics()),
        );
        let ready = builder().build_for_segments(&filter.weight, &segments, |_| {})?;
        let expected =
            ready.merge_fruits(vec![ready.collect_segment(&filter.weight, 0, reader)?])?;
        let result =
            collector.merge_fruits(vec![collector.collect_segment(&filter, 0, reader)?])?;
        assert_eq!(result.results, expected.results);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn routing_candidates_selects_competitive_segments() -> Result<()> {
        let mut distant = FixtureSpec::clustered(Metric::L2, 32, 8, 128);
        for vector in distant.vectors.iter_mut().flatten() {
            for value in vector {
                *value += 100.0;
            }
        }
        if let Layout::Ivf(centroids) = &mut distant.layout {
            for value in centroids.iter_mut().flatten() {
                *value += 100.0;
            }
        }
        let distant = distant.build()?;
        let near = FixtureSpec::clustered(Metric::L2, 32, 8, 128).build()?;
        let searchers = [
            distant.index.reader()?.searcher(),
            near.index.reader()?.searcher(),
        ];
        let readers: Vec<_> = searchers
            .iter()
            .enumerate()
            .map(|(ord, searcher)| {
                VectorSegment::from((ord as u32, &searcher.segment_readers()[0]))
            })
            .collect();
        let mut query = vec![0.0; 32];
        query[3] = 1.0;
        let params = ProbeParams {
            max_probe: 1,
            min_probe_clusters: 3,
            ..Default::default()
        };
        let builder = || {
            TopDocs::with_limit(3)
                .order_by_similarity(near.field, query.clone())
                .with_probe_params(params.clone())
        };
        let weight = near.all()?;
        let budget = VectorSearchBudget::for_segments(near.field, &params, &readers)?;
        let prepared = builder().prepare(weight.as_ref(), &readers, budget, |_| {})?;
        let router =
            TopClustersBySimilarityCollector::new(near.field, query.clone(), params.clone(), 1.0)?
                .with_budget(budget);
        let mut candidates = router.collect_segments(&prepared.routing_segments(), |_| {})?;
        candidates.truncate(3);
        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().all(|c| c.node.0 == 1));
        let shared_work: Option<Arc<dyn crate::vector::SharedVectorWork>> =
            Some(Arc::new(crate::vector::AtomicVectorWork::default()));
        let inputs = prepared.with_candidates(candidates.clone(), shared_work.clone())?;
        assert_eq!(inputs.segments().len(), 1);
        assert_eq!(inputs.segments()[0].ordinal, 1);
        let collector = builder().build(inputs.with_routing_statistics(router.take_statistics()));
        let fruit = collector.merge_fruits(collector.collect_segments(
            weight.as_ref(),
            &readers,
            |_| {},
        )?)?;
        assert_eq!(fruit.results.len(), 3);
        assert!(fruit
            .results
            .iter()
            .all(|(_, address)| address.segment_ord == 1));
        assert_eq!(fruit.stats[0].candidates_scored, 0);
        assert_eq!(fruit.stats[0].work_budget, 0.0);
        assert_eq!(fruit.stats[1].work_budget, 3.0);
        assert!(fruit.stats.iter().map(|s| s.work_charged).sum::<f32>() < 4.1);
        let budget = VectorSearchBudget::for_segments(near.field, &params, &readers)?;
        let prepared = builder().prepare(weight.as_ref(), &readers, budget, |_| {})?;
        let inputs = prepared.with_candidates(candidates, None)?;
        let collect = |segments: &[VectorSegment]| -> Result<_> {
            let collector = builder().build(inputs.clone());
            collector.merge_fruits(collector.collect_segments(weight.as_ref(), segments, |_| {})?)
        };
        let full = collect(&readers)?;
        let active = collect(inputs.segments())?;
        assert_eq!(active.results, full.results);
        assert_eq!(active.stats.len(), 1);
        assert_eq!(active.stats[0].work_budget, full.stats[1].work_budget);
        assert!(active.stats[0].work_charged > active.stats[0].work_budget);
        assert_eq!(active.stats[0].work_charged, full.stats[1].work_charged);
        Ok(())
    }

    #[test]
    fn global_search_has_one_budget_floor() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 16, 1024)
            .segments(4)
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let readers: Vec<_> = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
            .collect();
        let mut query = vec![0.0; 32];
        query[3] = 1.0;
        let new_collector = || {
            TopDocs::with_limit(10)
                .order_by_similarity(fixture.field, query.clone())
                .with_probe_params(ProbeParams {
                    max_probe: 1,
                    min_probe_clusters: 16,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap()
        };
        let collector = new_collector();
        let weight = fixture.all()?;
        let fruit = collector.merge_fruits(collector.collect_segments(
            weight.as_ref(),
            &readers,
            |_| {},
        )?)?;
        assert_eq!(fruit.stats.iter().map(|s| s.work_budget).sum::<f32>(), 16.0);
        assert!(fruit.stats.iter().map(|s| s.work_charged).sum::<f32>() < 18.0);
        let centroid = vec![0.0; 32];
        let single = [VectorSegment {
            partition_centroid: Some(centroid.into()),
            ..readers[2].clone()
        }];
        let legacy = new_collector().collect_segment(
            weight.as_ref(),
            single[0].ordinal,
            &single[0].reader,
        )?;
        let mut automatic = new_collector().collect_segments(weight.as_ref(), &single, |_| {})?;
        assert_eq!(automatic.len(), 1);
        let automatic = automatic.pop().unwrap();
        assert_eq!(automatic.results, legacy.results);
        let counters = |stats| {
            let mut value = serde_json::to_value(stats).unwrap();
            value
                .as_object_mut()
                .unwrap()
                .retain(|key, _| !key.ends_with("_ns"));
            value
        };
        assert_eq!(counters(automatic.stats), counters(legacy.stats));
        let empty =
            collector.merge_fruits(collector.collect_segments(weight.as_ref(), &[], |_| {})?)?;
        assert!(empty.results.is_empty());
        assert!(empty.stats.is_empty());
        Ok(())
    }

    #[test]
    fn adaptive_global_ceiling_charges_work_not_cluster_count() -> Result<()> {
        let centroids: Vec<_> = (0..16).map(|c| vec![c as f32 * 10.0, 0.0]).collect();
        let vectors = (0..4)
            .flat_map(|_| {
                centroids
                    .iter()
                    .enumerate()
                    .flat_map(|(c, v)| std::iter::repeat_n(v.clone(), if c < 8 { 1 } else { 32 }))
            })
            .collect();
        let fixture = FixtureSpec::ivf(Metric::L2, centroids, vectors)
            .segments(2)
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let readers: Vec<_> = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
            .collect();
        let weight = fixture.all()?;
        let mut results = Vec::new();
        for adaptive in [false, true] {
            let collector = TopDocs::with_limit(20)
                .order_by_similarity(fixture.field, vec![0.0_f32, 0.0])
                .with_probe_params(ProbeParams {
                    adaptive_probing: adaptive.then_some(crate::vector::AdaptiveProbeParams {
                        admission_fraction: 1e-20,
                    }),
                    max_probe: 8,
                    located: LocatedPath::Off,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap();
            let result = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &readers,
                |_| {},
            )?)?;
            assert_eq!(result.stats.iter().map(|s| s.work_budget).sum::<f32>(), 8.0);
            assert!(result.stats.iter().map(|s| s.work_charged).sum::<f32>() < 10.0);
            results.push(result.results);
        }
        assert_eq!(results[0].len(), 20);
        assert_eq!(results[0], results[1]);
        Ok(())
    }

    /// The production path: the fruit of a normal `searcher.search` carries
    /// one `ProbeStats` per IVF segment, each satisfying the counter
    /// invariant, so callers can aggregate probe metrics straight off the
    /// search result.
    #[test]
    fn e2e_ivf_fruit_carries_per_segment_probe_stats() -> Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let num_segments = searcher.segment_readers().len();

        let collector = TopDocs::with_limit(4)
            .order_by_similarity(index.embedding_field(), vec![0.5_f32, 0.5])
            .with_probe_params(exhaustive_params(9))
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap();
        let fruit = searcher.search(&AllQuery, &collector)?;

        // One ProbeStats per searched segment.
        assert_eq!(fruit.stats.len(), num_segments);
        let mut total_visited = 0usize;
        for s in &fruit.stats {
            assert_eq!(
                s.vectors_visited,
                s.pruned_filter + s.pruned_dead + s.candidates_scored,
                "invariant per segment: {s:?}"
            );
            total_visited += s.vectors_visited;
        }
        assert!(total_visited > 0, "exhaustive probe should visit docs");
        Ok(())
    }

    /// `and_offset(n)` returns the oracle's `[n, n+k)` slice.
    #[test]
    fn e2e_offset_window_matches_oracle_slice() -> Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .router(RouterKind::Exact)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let query = [0.5_f32, 0.5];
        let k = 3;
        let offset = 4;
        let full = index.ground_truth(query, offset + k)?;
        let expected = full[offset..].to_vec();
        let collector = TopDocs::with_limit(k)
            .and_offset(offset)
            .order_by_similarity(index.embedding_field(), query.to_vec())
            .with_probe_params(exhaustive_params(9))
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap();
        let actual = searcher.search(&AllQuery, &collector)?;
        assert_eq!(actual.results, expected);
        Ok(())
    }

    /// Flat-format build also matches the oracle. Pairs with
    /// `e2e_ivf_matches_global_oracle` to exercise the per-segment
    /// dispatch on both backend variants — `vector_storage_format`
    /// is the only thing that changes between them.
    #[test]
    fn e2e_flat_matches_global_oracle() -> Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Flat)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        for query in [[0.5_f32, 0.5], [9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let expected = index.ground_truth(query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(index.embedding_field(), query.to_vec())
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap();
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "Flat query={query:?} k={k}");
            }
        }
        Ok(())
    }

    fn tie_heavy_index(ids: &[u64]) -> Result<(Index, Field, Field)> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let id_field = schema_builder.add_u64_field("id", FAST);
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema_builder.build())
            .settings(settings)
            .ivf_clusterer(Arc::new(Grid2DClusterer {
                centroids: vec![[0.0, 0.0], [10.0, 10.0]],
            }))
            // Exact: this fixture's oracle assumes every cluster is probed.
            .ivf_router(RouterKind::Exact)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Only four distinct positions across all docs, so every doc shares its
        // distance with several others whichever query is asked.
        let positions = [[0.0_f32, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]];
        let add = |writer: &mut IndexWriter, range: Range<usize>| {
            for i in range {
                let mut doc = TantivyDocument::new();
                doc.add_vector(embedding_field, &positions[i % positions.len()]);
                doc.add_u64(id_field, ids[i]);
                writer.add_document(doc).unwrap();
            }
        };
        let third = ids.len() / 3;
        add(&mut writer, 0..third);
        writer.commit()?;
        add(&mut writer, third..third * 2);
        writer.commit()?;
        let mut targets = index.searchable_segment_ids()?;
        targets.sort();
        writer.merge(&targets).wait()?;
        add(&mut writer, third * 2..ids.len());
        writer.commit()?;
        writer.wait_merging_threads()?;

        // A fixture that ended up all-Flat would quietly stop testing the
        // probe loop at all.
        let searcher = index.reader()?.searcher();
        let ivf_segments = searcher
            .segment_readers()
            .iter()
            .filter(|reader| {
                reader
                    .vector_index(embedding_field)
                    .is_ok_and(|vectors| vectors.index().is_some())
            })
            .count();
        assert!(ivf_segments >= 1, "expected at least one Ivf segment");
        assert!(
            ivf_segments < searcher.segment_readers().len(),
            "expected at least one Flat segment"
        );
        Ok((index, embedding_field, id_field))
    }

    #[test]
    fn e2e_tie_break_matches_oracle_and_leaves_probing_untouched() -> Result<()> {
        let ids: Vec<u64> = (0..30).map(|i| (i * 11) % 30).collect();
        let (index, embedding_field, _) = tie_heavy_index(&ids)?;
        let searcher = index.reader()?.searcher();
        let tie_break = || (SortByStaticFastValue::<u64>::for_field("id"), Order::Asc);

        for query in [[0.0_f32, 0.0], [10.5, 9.5], [5.0, 5.0]] {
            // (score, id, address) for every doc, straight from the readers,
            // sorted descending score, then ascending id, then ascending
            // address — the same total order the composite heap applies.
            let mut expected: Vec<(Score, u64, DocAddress)> = Vec::new();
            for (segment_ord, reader) in searcher.segment_readers().iter().enumerate() {
                let id_column = reader.fast_fields().u64("id")?;
                let vector_reader = reader.vector_index(embedding_field)?;
                for doc_id in 0..reader.max_doc() {
                    let address = vector_reader.row_location(doc_id)?.unwrap();
                    let bytes = vector_reader.vector_bytes_for_row(address)?;
                    expected.push((
                        -l2_squared_bytes(&query, &bytes),
                        id_column.first(doc_id).unwrap(),
                        DocAddress::new(segment_ord as u32, doc_id),
                    ));
                }
            }
            expected.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            // Without ties straddling the k values below, the oracle check
            // asserts nothing the untie-broken path wouldn't already satisfy.
            let distinct_scores = expected
                .windows(2)
                .filter(|pair| pair[0].0 != pair[1].0)
                .count()
                + 1;
            assert!(
                distinct_scores < expected.len(),
                "fixture produced no distance ties for query={query:?}"
            );

            for k in [1usize, 3, 7, 12] {
                // Ordering: exhaustive probing so the IVF side is exact and
                // only the composite ordering + cross-segment merge is tested.
                let fruit = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k)
                        .order_by_similarity(embedding_field, query.to_vec())
                        .with_probe_params(exhaustive_params(9))
                        .with_tie_break(tie_break())
                        .build_for_searcher(&searcher, &crate::query::AllQuery)
                        .unwrap(),
                )?;
                let actual: Vec<DocAddress> =
                    fruit.results.iter().map(|(_, address)| *address).collect();
                let want: Vec<DocAddress> = expected.iter().take(k).map(|entry| entry.2).collect();
                assert_eq!(actual, want, "query={query:?} k={k}");

                let collector = TopDocs::with_limit(k)
                    .and_offset(2)
                    .order_by_similarity(embedding_field, query.to_vec())
                    .with_probe_params(exhaustive_params(9))
                    .with_tie_break(tie_break())
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap();
                let readers: Vec<_> = searcher
                    .segment_readers()
                    .iter()
                    .enumerate()
                    .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                    .collect();
                let weight =
                    Query::weight(&AllQuery, EnableScoring::disabled_from_searcher(&searcher))?;
                let global = collector.merge_fruits(collector.collect_segments(
                    weight.as_ref(),
                    &readers,
                    |_| {},
                )?)?;
                let want: Vec<_> = expected
                    .iter()
                    .skip(2)
                    .take(k)
                    .map(|entry| entry.2)
                    .collect();
                assert_eq!(
                    global
                        .results
                        .iter()
                        .map(|(_, address)| *address)
                        .collect::<Vec<_>>(),
                    want
                );

                // Probe invariance, under the default probe params so the
                // gate/ceiling logic actually runs.
                let collector =
                    || TopDocs::with_limit(k).order_by_similarity(embedding_field, query.to_vec());
                let mut untied = searcher.search(
                    &AllQuery,
                    &collector().build_for_searcher(&searcher, &AllQuery)?,
                )?;
                let mut tied = searcher.search(
                    &AllQuery,
                    &collector()
                        .with_tie_break(tie_break())
                        .build_for_searcher(&searcher, &AllQuery)?,
                )?;
                assert!(
                    untied.stats.iter().any(|s| s.candidates_scored > 0),
                    "no probe activity to compare for query={query:?} k={k}"
                );
                for stats in untied.stats.iter_mut().chain(&mut tied.stats) {
                    stats.clear_stage_timings();
                }
                assert_eq!(
                    format!("{:?}", untied.stats),
                    format!("{:?}", tied.stats),
                    "probe stats diverged for query={query:?} k={k}"
                );
            }
        }

        let err = searcher
            .search(
                &AllQuery,
                &TopDocs::with_limit(2)
                    .order_by_similarity(embedding_field, vec![0.0_f32, 0.0])
                    .with_tie_break(SortBySimilarityScore::new())
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap(),
            )
            .unwrap_err();
        assert!(
            matches!(err, TantivyError::InvalidArgument(ref msg) if msg.contains("relevance score")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }

    #[test]
    fn e2e_ivf_cluster_order_keeps_the_lowest_doc_of_a_tie() -> Result<()> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        // Exact ranks every centroid so both clusters are probed in distance
        // order. A Stacked flat top over two lists would keep only
        // ceil(0.25 × 2) = 1 list and never open the southern cluster.
        let index = Index::builder()
            .schema(schema_builder.build())
            .settings(settings)
            .ivf_clusterer(Arc::new(Grid2DClusterer {
                centroids: vec![[0.0, 10.0], [0.0, -10.0]],
            }))
            .ivf_router(RouterKind::Exact)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Query sits just north of the origin, so the northern centroid routes
        // first. Every doc is exactly distance 1 from it, so all scores tie and
        // ascending DocAddress alone decides the winner.
        let query = [0.0_f32, 0.1];
        // DocId 0 lands in the SOUTHERN cluster, probed second.
        writer.add_document({
            let mut doc = TantivyDocument::new();
            doc.add_vector(embedding_field, &[0.0_f32, -0.9]);
            doc
        })?;
        writer.commit()?;
        // DocIds 1..=3 land in the northern cluster, probed first, and are
        // enough to fill the heap and establish a threshold before DocId 0 is
        // ever scored.
        for v in [[0.0_f32, 1.1], [1.0, 0.1], [-1.0, 0.1]] {
            let mut doc = TantivyDocument::new();
            doc.add_vector(embedding_field, &v);
            writer.add_document(doc)?;
        }
        writer.commit()?;
        let mut targets = index.searchable_segment_ids()?;
        targets.sort();
        writer.merge(&targets).wait()?;
        writer.wait_merging_threads()?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let reader = searcher.segment_reader(0);
        let ivf = reader.vector_index(embedding_field)?;
        assert!(ivf.index().is_some(), "expected an Ivf segment");

        let fruit = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(1)
                .order_by_similarity(embedding_field, query.to_vec())
                .with_probe_params(exhaustive_params(2))
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap(),
        )?;
        // All four docs tie at distance 1, so the lowest DocAddress wins.
        let scores: Vec<Score> = fruit.results.iter().map(|(score, _)| *score).collect();
        assert_eq!(scores, vec![-1.0], "expected the shared distance");
        assert_eq!(
            fruit.results[0].1,
            DocAddress::new(0, 0),
            "cluster-order arrival dropped the lowest DocId of the tie"
        );
        Ok(())
    }

    #[test]
    fn e2e_tie_break_on_segment_local_term_ordinals() -> Result<()> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let city_field = schema_builder.add_text_field("city", STRING | FAST);
        let index = Index::builder()
            .schema(schema_builder.build())
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Every doc sits on the query point, so similarity ties globally and the
        // tie-break alone decides the order. Two commits give the same term two
        // different ordinals.
        for batch in [["b", "c"], ["a", "b"]] {
            for city in batch {
                let mut doc = TantivyDocument::new();
                doc.add_vector(embedding_field, &[0.0_f32, 0.0]);
                doc.add_text(city_field, city);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);

        // The premise: "b" must land on a different ordinal in each segment. If
        // the dictionaries happened to agree, comparing ordinals and comparing
        // strings would coincide and the assertions below would prove nothing.
        let ord_of_b = |segment_ord: u32| -> u64 {
            let column = searcher
                .segment_reader(segment_ord)
                .fast_fields()
                .str("city")
                .unwrap()
                .unwrap();
            let mut found = None;
            for doc_id in 0..searcher.segment_reader(segment_ord).max_doc() {
                for ord in column.term_ords(doc_id) {
                    let mut out = String::new();
                    column.ord_to_str(ord, &mut out).unwrap();
                    if out == "b" {
                        found = Some(ord);
                    }
                }
            }
            found.expect("every segment holds a \"b\"")
        };
        assert_ne!(
            ord_of_b(0),
            ord_of_b(1),
            "fixture failed to give \"b\" differing per-segment ordinals"
        );

        let cities = |fruit: &VectorSimilarityFruit| -> Vec<String> {
            fruit
                .results
                .iter()
                .map(|(_, address)| {
                    let column = searcher
                        .segment_reader(address.segment_ord)
                        .fast_fields()
                        .str("city")
                        .unwrap()
                        .unwrap();
                    let mut ords = column.term_ords(address.doc_id);
                    let ord = ords.next().unwrap();
                    let mut out = String::new();
                    column.ord_to_str(ord, &mut out).unwrap();
                    out
                })
                .collect()
        };

        // Ascending by string is a, b, b, c. Ascending by raw ordinal would be
        // b, a, c, b — so any ordinal leak shows up immediately.
        for (k, want) in [
            (4usize, vec!["a", "b", "b", "c"]),
            (2, vec!["a", "b"]),
            (1, vec!["a"]),
        ] {
            let fruit = searcher.search(
                &AllQuery,
                &TopDocs::with_limit(k)
                    .order_by_similarity(embedding_field, vec![0.0_f32, 0.0])
                    .with_tie_break((SortByString::for_field("city"), Order::Asc))
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap(),
            )?;
            assert_eq!(cities(&fruit), want, "k={k}");
            let collector = TopDocs::with_limit(k)
                .order_by_similarity(embedding_field, vec![0.0_f32, 0.0])
                .with_tie_break((SortByString::for_field("city"), Order::Asc))
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap();
            let readers: Vec<_> = searcher
                .segment_readers()
                .iter()
                .enumerate()
                .map(|(ord, reader)| VectorSegment::from((ord as u32, reader)))
                .collect();
            let weight =
                Query::weight(&AllQuery, EnableScoring::disabled_from_searcher(&searcher))?;
            let global = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &readers,
                |_| {},
            )?)?;
            assert_eq!(cities(&global), want, "global k={k}");
        }
        Ok(())
    }

    /// Single index containing both a Flat segment (un-merged commit) and
    /// an Ivf segment (merged commit under `vector_clustering_threshold=1`)
    /// so the collector has to dispatch `FlatBackend::top_n` on one and
    /// `IvfBackend::top_n` on the other in a single `searcher.search`.
    /// Hand-built — `TestVectorIndex` produces a single format index-wide
    /// — but uses the shared `Grid2DClusterer` and `ground_truth::top_k`
    /// so there's no parallel oracle / clusterer to drift.
    #[test]
    fn e2e_mixed_flat_and_ivf_matches_global_oracle() -> Result<()> {
        let centroids: Vec<[f32; 2]> = vec![[0.0, 0.0], [10.0, 10.0]];
        let metric = Metric::L2;
        let vector_options = VectorOptions::new(2, metric).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let label_field = schema_builder.add_text_field("label", STRING | STORED);
        let schema = schema_builder.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(Grid2DClusterer {
                centroids: centroids.clone(),
            }))
            // Exact so the IVF side can match the oracle; Stacked's flat-top
            // prune is approximate.
            .ivf_router(RouterKind::Exact)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Two commits → two flat segments; pairwise merge → one Ivf segment
        // (threshold=1 trips the format flip).
        let ivf_batches: [&[(&str, [f32; 2])]; 2] = [
            &[
                ("ivf0", [0.1, 0.1]),
                ("ivf1", [0.3, -0.2]),
                ("ivf2", [10.1, 9.9]),
            ],
            &[
                ("ivf3", [9.9, 10.1]),
                ("ivf4", [-0.2, 0.3]),
                ("ivf5", [10.4, 9.8]),
            ],
        ];
        for batch in ivf_batches {
            for (lbl, v) in batch {
                let mut doc = TantivyDocument::new();
                doc.add_vector(embedding_field, v);
                doc.add_text(label_field, *lbl);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        let mut ivf_targets = index.searchable_segment_ids()?;
        ivf_targets.sort();
        assert_eq!(ivf_targets.len(), 2, "expected two segments to merge");
        writer.merge(&ivf_targets).wait()?;

        // One more un-merged commit → flat segment.
        let flat_batch: [(&str, [f32; 2]); 3] = [
            ("flat0", [0.4, 0.4]),
            ("flat1", [10.3, 10.3]),
            ("flat2", [-0.1, 0.2]),
        ];
        for (lbl, v) in flat_batch {
            let mut doc = TantivyDocument::new();
            doc.add_vector(embedding_field, &v);
            doc.add_text(label_field, lbl);
            writer.add_document(doc)?;
        }
        writer.commit()?;
        writer.wait_merging_threads()?;

        // Confirm both formats are actually represented — the whole point
        // of this test is mixed dispatch, so a vacuous all-Flat or all-Ivf
        // index should fail loudly here.
        let searcher = index.reader()?.searcher();
        let mut flat_count = 0usize;
        let mut ivf_count = 0usize;
        for reader in searcher.segment_readers() {
            match reader.vector_index(embedding_field)?.index() {
                None => flat_count += 1,
                Some(_) => ivf_count += 1,
            }
        }
        assert!(
            flat_count >= 1 && ivf_count >= 1,
            "expected mixed segments, got {flat_count} flat / {ivf_count} ivf"
        );

        // Exhaustive probing on the Ivf side so the only thing being
        // tested here is per-segment dispatch + merge_fruits — not the
        // adaptive loop, which is covered separately.
        let params = exhaustive_params(9);
        for query in [[0.0_f32, 0.0], [10.0, 10.0], [5.0, 5.0]] {
            for k in [1usize, 3, 6] {
                let expected = ground_truth::top_k(&index, embedding_field, metric, &query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(embedding_field, query.to_vec())
                    .with_probe_params(params.clone())
                    .build_for_searcher(&searcher, &crate::query::AllQuery)
                    .unwrap();
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "mixed query={query:?} k={k}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod prepared_key_tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::quantization::{
        VectorNormPolicy, VectorQuantizationConfig, VectorQuantizationLayer,
    };
    use crate::vector::storage::metadata::{Grid, Partition, Quantizer, Rotation};

    #[test]
    fn concurrent_segments_share_prepared_query_state() {
        let mut schema = Schema::builder();
        let field = schema.add_vector_field("v", VectorOptions::new(3, Metric::Cosine));
        let index = crate::Index::create_in_ram(schema.build());
        let searcher = index.reader().unwrap().searcher();
        let collector = TopDocsByVectorSimilarityBuilder::new(field, vec![3.0f32, 4.0, 0.0], 1)
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap();
        let queries = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| collector.vector_query(Metric::Cosine)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for query in &queries {
            assert!(Arc::ptr_eq(query, &queries[0]));
            assert_eq!(query.routing_norm(), 5.0);
            assert!(std::ptr::eq(
                query.estimate_vector(),
                queries[0].estimate_vector()
            ));
        }
    }

    fn metadata(metric: Metric, schedule: &[u8]) -> VectorColMetadata {
        let opts = VectorOptions::new(100, metric);
        let config = VectorQuantizationConfig::materialize(
            "v".into(),
            &opts,
            schedule
                .iter()
                .map(|&bits| VectorQuantizationLayer { bits, seed: 17 })
                .collect(),
        )
        .unwrap();
        VectorColMetadata::build_ivf(&opts, Some(&config)).unwrap()
    }
    fn key(meta: &VectorColMetadata) -> PreparedKey {
        PreparedKey(Arc::new(meta.clone()))
    }
    fn hash(meta: &VectorColMetadata) -> u64 {
        let mut h = DefaultHasher::new();
        key(meta).hash(&mut h);
        h.finish()
    }
    // Storage geometry does not split prepared-query cache keys.
    #[test]
    fn query_identity_uses_only_semantic_bits() {
        let original = metadata(Metric::L2, &[1, 4]);
        let mut changed = original.clone();
        if let VectorColMetadata::Quantized { field, .. } = &mut changed {
            field.norm_policy = VectorNormPolicy::UnitL2;
            field.partition = Partition::Uniform { rows_per_block: 7 };
        }
        assert_eq!(key(&original), key(&changed));
        assert_eq!(hash(&original), hash(&changed));
        assert_ne!(original.to_bytes(), changed.to_bytes());
        for change in 0..8 {
            let mut changed = original.clone();
            if let VectorColMetadata::Quantized { field, layers } = &mut changed {
                match change {
                    0 => field.dim += 1,
                    1 => field.metric = Metric::Dot,
                    2 => {
                        if let Quantizer::SignPlane { rotation, .. } = &mut layers[0] {
                            *rotation = Rotation::None;
                        }
                    }
                    3 => {
                        if let Quantizer::SignPlane { rho_model, .. } = &mut layers[0] {
                            rho_model.0 += 1;
                        }
                    }
                    4 => {
                        if let Quantizer::GridPlane { bits, .. } = &mut layers[1] {
                            *bits = 3;
                        }
                    }
                    5 => {
                        if let Quantizer::GridPlane { grid, .. } = &mut layers[1] {
                            grid.points[0] = f32::from_bits(grid.points[0].to_bits() ^ 1);
                        }
                    }
                    6 => {
                        if let Quantizer::SignPlane { rotation, .. } = &mut layers[0] {
                            *rotation = Rotation::SeededFhtChaCha8 { seed: 18 };
                        }
                    }
                    _ => {
                        if let Quantizer::GridPlane { grid, .. } = &mut layers[1] {
                            grid.rho_model = f64::from_bits(grid.rho_model.to_bits() + 1);
                        }
                    }
                }
            }
            assert_ne!(key(&original), key(&changed));
        }
        let g = Grid {
            points: vec![0.0],
            rho_model: 1.0,
        };
        let mut other = g.clone();
        other.points[0] = -0.0;
        let quant = |grid| Quantizer::GridPlane {
            bits: 2,
            rotation: Rotation::None,
            grid,
        };
        assert_ne!(quant(g), quant(other));
    }
}

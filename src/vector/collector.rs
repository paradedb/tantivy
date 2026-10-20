//! Top-N vector-similarity collection.
//!
//! IVF drives its own cluster iteration, so [`Collector::collect_segment`] hands the filter
//! `Weight` to the segment's [`SegmentSearcher`](crate::vector::search::SegmentSearcher) instead of
//! pulling docs. A secondary key is an ordinary `SortKeyComputer` composed into the heap's
//! `(similarity, tie_break)` order, the same rule as the pull-model path.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{
    compare_for_top_k, Collector, ComparableDoc, SegmentCollector, SegmentSortKeyComputer,
    SortKeyComputer,
};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Metric, Schema};
use crate::vector::params::ProbeParams;
use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
use crate::vector::search::SegmentSearcher;
use crate::vector::stats::{ProbeStats, SegmentReport};
use crate::vector::storage::metadata::VectorColMetadata;
use crate::vector::storage::quantized::QuantizedFieldReader;
use crate::vector::telemetry::{enter_vector_stage, measure};
use crate::vector::tie_break::NoTieBreak;
use crate::vector::{VectorElement, VectorStage};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, TantivyError};

// --- Query preparation ---

/// Query identity consists of dimension, metric tag and structural layer metadata.
#[derive(Clone, Debug)]
struct PreparedKey(Arc<VectorColMetadata>);
impl PreparedKey {
    fn query_fields(&self) -> Option<(u32, u8, &[crate::vector::storage::metadata::Quantizer])> {
        match self.0.as_ref() {
            VectorColMetadata::Plain(_) => None,
            VectorColMetadata::Quantized { field, layers } => {
                let metric = match field.metric {
                    crate::schema::Metric::L2 => 0,
                    crate::schema::Metric::Dot => 1,
                    crate::schema::Metric::Cosine => 2,
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

// --- Collector ---

/// Collects documents by descending vector similarity; docs without a vector are never returned.
///
/// `T` must match the field's dtype ([`Collector::check_schema`]). `S` orders equal similarities
/// and defaults to [`NoTieBreak`]: ascending `DocAddress`.
pub struct TopDocsByVectorSimilarity<T: VectorElement, S = NoTieBreak> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    probe_params: ProbeParams,
    max_scan_levels: usize,
    /// Exactly one prepared query for each distinct segment encoding.
    quantized_queries: Mutex<HashMap<PreparedKey, PreparedCell>>,
    /// The query prepared for the field's metric on first use.
    prepared: Mutex<Option<Arc<VectorQuery<T>>>>,
    tie_break: S,
}

impl<T: VectorElement> TopDocsByVectorSimilarity<T, NoTieBreak> {
    /// Creates a top-vector-similarity collector.
    pub fn new(field: Field, query: Vec<T>, limit: usize) -> Self {
        Self {
            field,
            query: Arc::new(query),
            limit,
            offset: 0,
            probe_params: ProbeParams::default(),
            max_scan_levels: usize::MAX,
            quantized_queries: Mutex::new(HashMap::new()),
            prepared: Mutex::new(None),
            tie_break: NoTieBreak,
        }
    }
}

impl<T: VectorElement, S> TopDocsByVectorSimilarity<T, S> {
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

    /// Orders equal similarities by `tie_break`, as `ORDER BY embedding <=> $1, id` does.
    ///
    /// The tie-break takes part in each segment's eviction but never in routing. Segments cut on
    /// the segment-local key, so `convert_segment_sort_key` must preserve order within a segment.
    pub fn with_tie_break<S2: SortKeyComputer>(
        self,
        tie_break: S2,
    ) -> TopDocsByVectorSimilarity<T, S2> {
        TopDocsByVectorSimilarity {
            field: self.field,
            query: self.query,
            limit: self.limit,
            offset: self.offset,
            probe_params: self.probe_params,
            max_scan_levels: self.max_scan_levels,
            quantized_queries: self.quantized_queries,
            prepared: self.prepared,
            tie_break,
        }
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

    fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
        let entry = schema.get_field_entry(self.field);
        let opts = match entry.field_type() {
            FieldType::Vector(o) => o,
            _ => {
                return Err(TantivyError::SchemaError(format!(
                    "field {:?} is not a vector field",
                    entry.name(),
                )));
            }
        };
        if opts.dim() != self.query.len() {
            return Err(TantivyError::SchemaError(format!(
                "query vector length {} does not match field {:?} dim {}",
                self.query.len(),
                entry.name(),
                opts.dim(),
            )));
        }
        if opts.dtype() != T::DTYPE {
            return Err(TantivyError::SchemaError(format!(
                "query dtype {:?} does not match field {:?} dtype {:?}",
                T::DTYPE,
                entry.name(),
                opts.dtype(),
            )));
        }
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
    ) -> crate::Result<Self::Child> {
        // Unreachable: `collect_segment` is overridden; the child only satisfies the bound.
        Ok(NoOpSegmentCollector::default())
    }

    fn requires_scoring(&self) -> bool {
        // Similarity comes from the stored vectors, never from the filter's score.
        false
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        let (searched, totals) = measure(|| self.search_segment(weight, segment_ord, reader));
        let (results, report) = searched?;
        Ok(SegmentVectorFruit {
            results,
            stats: ProbeStats::new(report, totals),
        })
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<SegmentVectorFruit<S::SortKey>>,
    ) -> crate::Result<Self::Fruit> {
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
    /// Searches one segment: prepares the query, runs the segment searcher, and lifts each hit's
    /// tie-break key out of the segment.
    fn search_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<(KeyedHits<S::SortKey>, SegmentReport)> {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let (query, quantized) = {
            let _prep = enter_vector_stage(VectorStage::QueryPrep);
            let vectors = reader.vector_index(self.field)?;
            let query = self.vector_query(vectors.options().metric());
            let quantized = match vectors.quantization() {
                Some(field) if self.max_scan_levels > 0 => {
                    Some(self.quantized_query(field, &query))
                }
                _ => None,
            };
            (query, quantized)
        };
        let backend = SegmentSearcher::for_segment(
            reader,
            segment_ord,
            self.field,
            query,
            quantized,
            self.probe_params.clone(),
        )?;
        let mut tie_break = self.tie_break.segment_sort_key_computer(reader)?;
        drop(init);
        let (hits, report) = backend.top_n_by(
            weight,
            reader,
            self.segment_top_n(),
            &mut tie_break,
            self.tie_break.comparator(),
        )?;
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
        Ok((results, report))
    }
}

/// Satisfies `Collector::Child`; [`Collector::collect_segment`] is overridden, so it never runs.
pub struct NoOpSegmentCollector<K>(std::marker::PhantomData<K>);

impl<K> Default for NoOpSegmentCollector<K> {
    fn default() -> Self {
        NoOpSegmentCollector(std::marker::PhantomData)
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
    use std::sync::Arc;

    use super::VectorSimilarityFruit;
    use crate::collector::sort_key::{SortBySimilarityScore, SortByStaticFastValue};
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Field, Metric, Schema, VectorDType, VectorOptions, FAST, STORED, STRING};
    use crate::vector::diagnostics::VectorStorageFormat;
    use crate::vector::tests::{exhaustive_params, ground_truth, Grid2DClusterer, TestVectorIndex};
    use crate::vector::RouterKind;
    use crate::{DocAddress, Index, Order, Score, TantivyDocument, TantivyError};

    /// IVF + exhaustive probing matches the global oracle. The shared
    /// fixture produces multiple IVF segments (it merges raw segments
    /// pairwise), so this single test already exercises cross-segment
    /// merge_fruits.
    #[test]
    fn e2e_ivf_matches_global_oracle() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let params = exhaustive_params(9);
        for query in [[0.5_f32, 0.5], [9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let expected = index.ground_truth(query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(index.embedding_field(), query.to_vec())
                    .with_probe_params(params.clone());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "IVF query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// The production path: the fruit of a normal `searcher.search` carries
    /// one `ProbeStats` per IVF segment, each satisfying the counter
    /// invariant, so callers can aggregate probe metrics straight off the
    /// search result.
    #[test]
    fn e2e_ivf_fruit_carries_per_segment_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let num_segments = searcher.segment_readers().len();

        let collector = TopDocs::with_limit(4)
            .order_by_similarity(index.embedding_field(), vec![0.5_f32, 0.5])
            .with_probe_params(exhaustive_params(9));
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
    fn e2e_offset_window_matches_oracle_slice() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
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
            .with_probe_params(exhaustive_params(9));
        let actual = searcher.search(&AllQuery, &collector)?;
        assert_eq!(actual.results, expected);
        Ok(())
    }

    /// Flat-format build also matches the oracle. Pairs with
    /// `e2e_ivf_matches_global_oracle` to exercise the per-segment
    /// dispatch on both backend variants — `vector_storage_format`
    /// is the only thing that changes between them.
    #[test]
    fn e2e_flat_matches_global_oracle() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Flat)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        for query in [[0.5_f32, 0.5], [9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let expected = index.ground_truth(query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(index.embedding_field(), query.to_vec());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "Flat query={query:?} k={k}");
            }
        }
        Ok(())
    }

    fn tie_heavy_index(ids: &[u64]) -> crate::Result<(Index, Field, Field)> {
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
            .ivf_router(RouterKind::Stacked)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Only four distinct positions across all docs, so every doc shares its
        // distance with several others whichever query is asked.
        let positions = [[0.0_f32, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]];
        let add = |writer: &mut crate::IndexWriter, range: std::ops::Range<usize>| {
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
    fn e2e_tie_break_matches_oracle_and_leaves_probing_untouched() -> crate::Result<()> {
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
                    let row = vector_reader.row_id(doc_id)?.unwrap();
                    let bytes = vector_reader.vector_bytes_for_row(row)?;
                    expected.push((
                        -crate::vector::distance::l2_squared_bytes(&query, &bytes),
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
                        .with_tie_break(tie_break()),
                )?;
                let actual: Vec<DocAddress> =
                    fruit.results.iter().map(|(_, address)| *address).collect();
                let want: Vec<DocAddress> = expected.iter().take(k).map(|entry| entry.2).collect();
                assert_eq!(actual, want, "query={query:?} k={k}");

                // Probe invariance, under the default probe params so the
                // gate/ceiling logic actually runs.
                let collector =
                    || TopDocs::with_limit(k).order_by_similarity(embedding_field, query.to_vec());
                let mut untied = searcher.search(&AllQuery, &collector())?;
                let mut tied =
                    searcher.search(&AllQuery, &collector().with_tie_break(tie_break()))?;
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
                    .with_tie_break(SortBySimilarityScore::new()),
            )
            .unwrap_err();
        assert!(
            matches!(err, TantivyError::InvalidArgument(ref msg) if msg.contains("relevance score")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }

    #[test]
    fn e2e_ivf_cluster_order_keeps_the_lowest_doc_of_a_tie() -> crate::Result<()> {
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
                .with_probe_params(exhaustive_params(2)),
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
    fn e2e_tie_break_on_segment_local_term_ordinals() -> crate::Result<()> {
        use crate::collector::sort_key::SortByString;

        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let city_field = schema_builder.add_text_field("city", crate::schema::STRING | FAST);
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
                    .with_tie_break((SortByString::for_field("city"), Order::Asc)),
            )?;
            assert_eq!(cities(&fruit), want, "k={k}");
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
    fn e2e_mixed_flat_and_ivf_matches_global_oracle() -> crate::Result<()> {
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
            .ivf_router(RouterKind::Stacked)
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
                    .with_probe_params(params.clone());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "mixed query={query:?} k={k}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod prepared_key_tests {
    use super::*;
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::quantization::{
        VectorNormPolicy, VectorQuantizationConfig, VectorQuantizationLayer,
    };
    use crate::vector::storage::metadata::{Grid, Partition, Quantizer, Rotation};
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
        let mut h = std::collections::hash_map::DefaultHasher::new();
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
        let quant = |grid| crate::vector::storage::metadata::Quantizer::GridPlane {
            bits: 2,
            rotation: Rotation::None,
            grid,
        };
        assert_ne!(quant(g), quant(other));
    }
}

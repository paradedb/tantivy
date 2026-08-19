//! Top-N vector-similarity collector.
//!
//! Unlike the other `TopDocs::order_by_*` paths, collection here is not
//! per-segment at all: centroids are index-level and every segment shares
//! their cluster ids, so the search ranks the set's centroids once and
//! gathers each ranked cluster across ALL segments into one heap (see
//! [`search`](super::search)). That inverts the [`Collector`] trait's
//! per-segment pull model, so this collector implements
//! [`Collector::collect_global`] — the searcher-level hook — and must be
//! the TOP-LEVEL collector of its search: wrapped inside `MultiCollector`
//! or another combinator, the per-segment path would run instead, and
//! [`Collector::collect_segment`] fails loudly.
//!
//! A secondary key *is* an ordinary `SortKeyComputer` — see
//! [`TopDocsByVectorSimilarity::with_tie_break`]. The global heap sorts on
//! the composite `(similarity, tie_break)`; segment-local sort keys are
//! lifted to their global form at push time, for competitive candidates
//! only.

use std::sync::Arc;

use super::backend::ProbeStats;
use super::ivf::AdaptiveProbeParams;
use super::search::global_top_n_by;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::{Collector, SegmentCollector, SortKeyComputer};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::{DocAddress, DocId, Score, Searcher, SegmentOrdinal, TantivyError};

/// Top-N by vector similarity. Returns documents in descending
/// similarity order. Only docs that actually have a vector are
/// returned — docs that match the filter but lack a vector for `field`
/// are dropped (IVF storage can't see vectorless docs at all).
///
/// Generic over `T: VectorElement` — `T` must match the schema's
/// declared dtype, checked at [`Collector::check_schema`] time.
///
/// `S` orders documents that tie on similarity; it defaults to
/// [`NoTieBreak`], which leaves ties to ascending `DocAddress`. See
/// [`with_tie_break`](Self::with_tie_break).
pub struct TopDocsByVectorSimilarity<T: VectorElement, S = NoTieBreak> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    adaptive: AdaptiveProbeParams,
    tie_break: S,
}

impl<T: VectorElement> TopDocsByVectorSimilarity<T, NoTieBreak> {
    pub fn new(field: Field, query: Vec<T>, limit: usize) -> Self {
        Self {
            field,
            query: Arc::new(query),
            limit,
            offset: 0,
            adaptive: AdaptiveProbeParams::default(),
            tie_break: NoTieBreak,
        }
    }
}

impl<T: VectorElement, S> TopDocsByVectorSimilarity<T, S> {
    /// Drop the first `offset` results in the global ranking — used to
    /// paginate. The global heap keeps `limit + offset` candidates so the
    /// window is exact.
    pub fn and_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Override the adaptive probing parameters.
    pub fn with_adaptive_params(mut self, params: AdaptiveProbeParams) -> Self {
        self.adaptive = params;
        self
    }

    /// Order documents that tie on similarity by `tie_break`, as
    /// `ORDER BY embedding <=> $1, id` does.
    ///
    /// The tie-break takes part in the global heap's eviction, so it also
    /// decides *which* of a set of equally-distant documents survive, not
    /// only how the survivors are ordered. Similarity remains the primary
    /// key; the tie-break is only consulted between documents whose
    /// similarity is exactly equal.
    ///
    /// This does not change which clusters are probed: the probe loop's
    /// stopping rule reads the routed centroids and the work budget,
    /// never the tie-break.
    ///
    /// Candidates are keyed by the GLOBAL `SortKey` at push time — a
    /// segment-local `SegmentSortKey` (e.g. a term ordinal) means nothing
    /// beside another segment's, and the heap holds candidates from every
    /// segment at once. `convert_segment_sort_key` must therefore be
    /// order-preserving within a segment; the bundled computers satisfy
    /// this.
    pub fn with_tie_break<S2: SortKeyComputer>(
        self,
        tie_break: S2,
    ) -> TopDocsByVectorSimilarity<T, S2> {
        TopDocsByVectorSimilarity {
            field: self.field,
            query: self.query,
            limit: self.limit,
            offset: self.offset,
            adaptive: self.adaptive,
            tie_break,
        }
    }

    fn segment_top_n(&self) -> usize {
        self.limit.saturating_add(self.offset)
    }
}

/// What a [`TopDocsByVectorSimilarity`] search returns: the global top-N
/// plus the query's probe instrumentation.
#[derive(Debug, Default)]
pub struct VectorSimilarityFruit {
    /// Global top-N `(score, address)` pairs in descending-similarity order.
    pub results: Vec<(Score, DocAddress)>,
    /// The query's [`ProbeStats`] — one per query: the probe loop is
    /// global, so its counters are too.
    pub stats: ProbeStats,
}

/// Trait-bound placeholder: collection is global, so no per-segment fruit
/// is ever produced — the type exists only to satisfy the
/// `Child: SegmentCollector` bound.
pub struct SegmentVectorFruit;

impl<T, S> Collector for TopDocsByVectorSimilarity<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    type Fruit = VectorSimilarityFruit;
    type Child = NoOpSegmentCollector;

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
            // `requires_scoring` is false below, so the filter's BM25 score is
            // never computed and every doc would tie-break on the same
            // placeholder. Fail loudly rather than silently ordering by nothing.
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
        // Never called at runtime — collection is global. The child type
        // exists only to satisfy the trait bound.
        Ok(NoOpSegmentCollector)
    }

    fn requires_scoring(&self) -> bool {
        // Similarity is computed from the stored vectors, not from the
        // filter's BM25 score — let tantivy take the no-score fast path.
        false
    }

    fn collect_global(
        &self,
        weight: &dyn Weight,
        searcher: &Searcher,
    ) -> crate::Result<Option<Self::Fruit>> {
        let (hits, stats) = global_top_n_by(
            searcher,
            weight,
            self.field,
            &self.query,
            self.segment_top_n(),
            &self.adaptive,
            &self.tie_break,
        )?;
        let results = hits
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .map(|((score, _tie), address)| (score, address))
            .collect();
        Ok(Some(VectorSimilarityFruit { results, stats }))
    }

    fn collect_segment(
        &self,
        _weight: &dyn Weight,
        _segment_ord: SegmentOrdinal,
        _reader: &SegmentReader,
    ) -> crate::Result<SegmentVectorFruit> {
        Err(TantivyError::InvalidArgument(
            "TopDocsByVectorSimilarity collects across segments in one pass and must be the \
             top-level collector of its search; it cannot run inside MultiCollector or another \
             per-segment combinator"
                .to_string(),
        ))
    }

    fn merge_fruits(&self, _segment_fruits: Vec<SegmentVectorFruit>) -> crate::Result<Self::Fruit> {
        Err(TantivyError::InvalidArgument(
            "TopDocsByVectorSimilarity produces no per-segment fruits to merge; it must be the \
             top-level collector of its search"
                .to_string(),
        ))
    }
}

/// Trait-bound shim: the collector overrides [`Collector::collect_global`]
/// so the per-doc path never fires, but the `Child: SegmentCollector`
/// bound on `Collector` still has to be satisfied.
pub struct NoOpSegmentCollector;

impl SegmentCollector for NoOpSegmentCollector {
    type Fruit = SegmentVectorFruit;
    fn collect(&mut self, _doc: DocId, _score: Score) {}
    fn harvest(self) -> Self::Fruit {
        SegmentVectorFruit
    }
}

#[cfg(test)]
mod e2e_tests {
    //! End-to-end coverage of the production path: `searcher.search →
    //! Collector::collect_global → search::global_top_n_by`, asserted
    //! against `index.ground_truth(...)`.

    use std::sync::Arc;

    use super::VectorSimilarityFruit;
    use crate::collector::sort_key::{SortBySimilarityScore, SortByStaticFastValue};
    use crate::collector::TopDocs;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Field, Schema, FAST};
    use crate::vector::tests::{exhaustive_params, Grid2DCentroidIndex, TestVectorIndex};
    use crate::vector::{Metric, VectorDType, VectorOptions};
    use crate::{DocAddress, Index, Order, Score, TantivyDocument, TantivyError};

    /// Exhaustive probing matches the global oracle across the fixture's
    /// several segments — one routing pass, one heap.
    #[test]
    fn e2e_matches_global_oracle() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let params = exhaustive_params(9);
        for query in [[0.5_f32, 0.5], [9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let expected = index.ground_truth(query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(index.embedding_field(), query.to_vec())
                    .with_adaptive_params(params.clone());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// The fruit carries ONE global `ProbeStats` satisfying the counter
    /// invariant — the probe loop is global, so its counters are too.
    #[test]
    fn e2e_fruit_carries_global_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let num_segments = searcher.segment_readers().len();

        let collector = TopDocs::with_limit(4)
            .order_by_similarity(index.embedding_field(), vec![0.5_f32, 0.5])
            .with_adaptive_params(exhaustive_params(9));
        let fruit = searcher.search(&AllQuery, &collector)?;

        let s = &fruit.stats;
        assert_eq!(
            s.vectors_visited,
            s.pruned_filter + s.pruned_dead + s.pruned_seen + s.candidates_scored,
            "invariant: {s:?}"
        );
        assert!(s.routing.visited_count > 0, "one routing pass ran");
        assert_eq!(s.segments_searched as usize, num_segments);
        assert!(s.vectors_visited > 0);
        Ok(())
    }

    /// `and_offset(n)` returns the oracle's `[n, n+k)` slice.
    #[test]
    fn e2e_offset_window_matches_oracle_slice() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
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
            .with_adaptive_params(exhaustive_params(9));
        let actual = searcher.search(&AllQuery, &collector)?;
        assert_eq!(actual.results, expected);
        Ok(())
    }

    fn tie_heavy_index(ids: &[u64]) -> crate::Result<(Index, Field, Field)> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let id_field = schema_builder.add_u64_field("id", FAST);
        let index = Index::builder()
            .schema(schema_builder.build())
            .centroid_index(Arc::new(Grid2DCentroidIndex {
                centroids: vec![[0.0, 0.0], [10.0, 10.0]],
                version: 1,
            }))
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

        // The tie-break must cross a segment boundary to prove anything.
        let searcher = index.reader()?.searcher();
        assert!(
            searcher.segment_readers().len() >= 2,
            "fixture needs >= 2 segments"
        );
        Ok((index, embedding_field, id_field))
    }

    /// Tie-breaks match a brute-force total order, and leave the probe
    /// loop untouched: the same clusters are probed with or without one.
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
                    let row = vector_reader.row_id(doc_id).unwrap();
                    let bytes = vector_reader.vector_bytes_for_row(row)?;
                    expected.push((
                        -crate::vector::l2_squared_bytes(&query, &bytes),
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
                // Ordering: exhaustive probing so the ranking is exact and
                // only the composite ordering is tested.
                let fruit = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k)
                        .order_by_similarity(embedding_field, query.to_vec())
                        .with_adaptive_params(exhaustive_params(2))
                        .with_tie_break(tie_break()),
                )?;
                let actual: Vec<DocAddress> =
                    fruit.results.iter().map(|(_, address)| *address).collect();
                let want: Vec<DocAddress> = expected.iter().take(k).map(|entry| entry.2).collect();
                assert_eq!(actual, want, "query={query:?} k={k}");

                // Probe invariance, under the default adaptive params so the
                // gate/ceiling logic actually runs.
                let collector =
                    || TopDocs::with_limit(k).order_by_similarity(embedding_field, query.to_vec());
                let untied = searcher.search(&AllQuery, &collector())?;
                let tied = searcher.search(&AllQuery, &collector().with_tie_break(tie_break()))?;
                assert!(
                    untied.stats.candidates_scored > 0,
                    "no probe activity to compare for query={query:?} k={k}"
                );
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

    /// All four docs tie on distance; the lowest `DocAddress` must win
    /// even though its cluster is probed LAST — cluster-order arrival must
    /// not decide ties.
    #[test]
    fn e2e_cluster_order_keeps_the_lowest_doc_of_a_tie() -> crate::Result<()> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let index = Index::builder()
            .schema(schema_builder.build())
            .centroid_index(Arc::new(Grid2DCentroidIndex {
                centroids: vec![[0.0, 10.0], [0.0, -10.0]],
                version: 1,
            }))
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

        let fruit = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(1)
                .order_by_similarity(embedding_field, query.to_vec())
                .with_adaptive_params(exhaustive_params(2)),
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

    /// Segment-local term ordinals as tie-breaks: the global heap holds
    /// candidates from EVERY segment at once, so keys must be lifted to
    /// their global (string) form at push time — comparing raw ordinals
    /// across segments would order b, a, c, b.
    #[test]
    fn e2e_tie_break_on_segment_local_term_ordinals() -> crate::Result<()> {
        use crate::collector::sort_key::SortByString;

        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let city_field = schema_builder.add_text_field("city", crate::schema::STRING | FAST);
        let index = Index::builder()
            .schema(schema_builder.build())
            .centroid_index(Arc::new(Grid2DCentroidIndex {
                centroids: vec![[0.0, 0.0]],
                version: 1,
            }))
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

    /// `check_schema` failures surface with their own messages: a
    /// mismatched query dim and a non-vector field.
    #[test]
    fn e2e_check_schema_errors() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let searcher = index.index.reader()?.searcher();

        let wrong_dim =
            TopDocs::with_limit(2).order_by_similarity(index.embedding_field(), vec![0.0_f32; 3]);
        let err = searcher.search(&AllQuery, &wrong_dim).unwrap_err();
        assert!(
            matches!(err, TantivyError::SchemaError(ref msg) if msg.contains("does not match")),
            "unexpected error: {err:?}"
        );

        let not_a_vector =
            TopDocs::with_limit(2).order_by_similarity(index.label_field(), vec![0.0_f32, 0.0]);
        let err = searcher.search(&AllQuery, &not_a_vector).unwrap_err();
        assert!(
            matches!(err, TantivyError::SchemaError(ref msg) if msg.contains("not a vector field")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }
}

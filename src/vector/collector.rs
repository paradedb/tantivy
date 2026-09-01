//! Top-N vector-similarity collection.

use std::sync::Arc;

use super::backend::{ProbeStats, VectorBackend};
use super::ivf::AdaptiveProbeParams;
use super::prepared::QuantizedQueryCache;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{
    compare_for_top_k, Collector, ComparableDoc, SegmentCollector, SegmentSortKeyComputer,
    SortKeyComputer,
};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, TantivyError};

/// Collects documents by descending vector similarity.
pub struct TopDocsByVectorSimilarity<T: VectorElement, S = NoTieBreak> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    adaptive: AdaptiveProbeParams,
    max_scan_levels: usize,
    quantized_queries: QuantizedQueryCache,
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
            adaptive: AdaptiveProbeParams::default(),
            max_scan_levels: usize::MAX,
            quantized_queries: QuantizedQueryCache::default(),
            tie_break: NoTieBreak,
        }
    }
}

impl<T: VectorElement, S> TopDocsByVectorSimilarity<T, S> {
    /// Sets the global result offset.
    pub fn and_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Sets adaptive probing parameters.
    pub fn with_adaptive_params(mut self, params: AdaptiveProbeParams) -> Self {
        self.adaptive = params;
        self
    }

    /// Limits the quantized residual prefix.
    pub fn with_max_scan_levels(mut self, max_scan_levels: usize) -> Self {
        self.max_scan_levels = max_scan_levels;
        self
    }

    /// Sets a secondary ordering for equal similarities.
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
            max_scan_levels: self.max_scan_levels,
            quantized_queries: self.quantized_queries,
            tie_break,
        }
    }

    fn segment_top_n(&self) -> usize {
        self.limit.saturating_add(self.offset)
    }

    #[cfg(test)]
    pub(crate) fn cached_quantized_query_count(&self) -> usize {
        self.quantized_queries.len()
    }
}

/// Contains vector results and per-segment probe statistics.
#[derive(Debug, Default)]
pub struct VectorSimilarityFruit {
    /// Global results in descending-similarity order.
    pub results: Vec<(Score, DocAddress)>,
    /// Probe statistics in segment order.
    pub stats: Vec<ProbeStats>,
}

/// One segment's vector results with secondary sort keys.
pub struct SegmentVectorFruit<K> {
    results: Vec<((Score, K), DocAddress)>,
    stats: ProbeStats,
}

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
            // Relevance scores are unavailable on vector-ordered scans.
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
        Ok(NoOpSegmentCollector::default())
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        let backend = VectorBackend::for_segment(
            reader,
            segment_ord,
            self.field,
            Arc::clone(&self.query),
            &self.quantized_queries,
            self.adaptive.clone(),
            self.max_scan_levels,
        )?;
        let mut tie_break = self.tie_break.segment_sort_key_computer(reader)?;
        let (hits, stats) = backend.top_n_by(
            weight,
            reader,
            self.segment_top_n(),
            &mut tie_break,
            self.tie_break.comparator(),
        )?;
        let results = hits
            .into_iter()
            .map(|((score, segment_key), address)| {
                (
                    (score, tie_break.convert_segment_sort_key(segment_key)),
                    address,
                )
            })
            .collect();
        Ok(SegmentVectorFruit { results, stats })
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<SegmentVectorFruit<S::SortKey>>,
    ) -> crate::Result<Self::Fruit> {
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
        all.sort_unstable_by(|lhs, rhs| compare_for_top_k(&comparator, lhs, rhs));
        let results = all
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .map(|cd| (cd.sort_key.0, cd.doc))
            .collect();
        Ok(VectorSimilarityFruit { results, stats })
    }
}

/// Satisfies the collector's segment-child type requirement.
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
    use std::sync::Arc;

    use super::VectorSimilarityFruit;
    use crate::collector::sort_key::{SortBySimilarityScore, SortByStaticFastValue};
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Field, Schema, FAST, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, ground_truth, Grid2DClusterer, TestVectorIndex};
    use crate::vector::{Metric, VectorDType, VectorOptions, VectorStorageFormat};
    use crate::{DocAddress, Index, Order, Score, TantivyDocument, TantivyError};

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
                    .with_adaptive_params(params.clone());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "IVF query={query:?} k={k}");
            }
        }
        Ok(())
    }

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
            .with_adaptive_params(exhaustive_params(9));
        let fruit = searcher.search(&AllQuery, &collector)?;

        assert_eq!(fruit.stats.len(), num_segments);
        let mut total_visited = 0usize;
        for s in &fruit.stats {
            assert_eq!(
                s.vectors_visited,
                s.pruned_filter + s.pruned_dead + s.candidates_scored,
                "invariant per segment: {s:?}"
            );
            assert!(s.routing.visited_count > 0);
            total_visited += s.vectors_visited;
        }
        assert!(total_visited > 0, "exhaustive probe should visit docs");
        Ok(())
    }

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
            .with_adaptive_params(exhaustive_params(9));
        let actual = searcher.search(&AllQuery, &collector)?;
        assert_eq!(actual.results, expected);
        Ok(())
    }

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
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

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
                let fruit = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k)
                        .order_by_similarity(embedding_field, query.to_vec())
                        .with_adaptive_params(exhaustive_params(9))
                        .with_tie_break(tie_break()),
                )?;
                let actual: Vec<DocAddress> =
                    fruit.results.iter().map(|(_, address)| *address).collect();
                let want: Vec<DocAddress> = expected.iter().take(k).map(|entry| entry.2).collect();
                assert_eq!(actual, want, "query={query:?} k={k}");

                let collector =
                    || TopDocs::with_limit(k).order_by_similarity(embedding_field, query.to_vec());
                let untied = searcher.search(&AllQuery, &collector())?;
                let tied = searcher.search(&AllQuery, &collector().with_tie_break(tie_break()))?;
                assert!(
                    untied.stats.iter().any(|s| s.candidates_scored > 0),
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

    #[test]
    fn e2e_ivf_cluster_order_keeps_the_lowest_doc_of_a_tie() -> crate::Result<()> {
        let vector_options = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let mut schema_builder = Schema::builder();
        let embedding_field = schema_builder.add_vector_field("embedding", vector_options);
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema_builder.build())
            .settings(settings)
            .ivf_clusterer(Arc::new(Grid2DClusterer {
                centroids: vec![[0.0, 10.0], [0.0, -10.0]],
            }))
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        let query = [0.0_f32, 0.1];
        writer.add_document({
            let mut doc = TantivyDocument::new();
            doc.add_vector(embedding_field, &[0.0_f32, -0.9]);
            doc
        })?;
        writer.commit()?;
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
                .with_adaptive_params(exhaustive_params(2)),
        )?;
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
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

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

        let params = exhaustive_params(9);
        for query in [[0.0_f32, 0.0], [10.0, 10.0], [5.0, 5.0]] {
            for k in [1usize, 3, 6] {
                let expected = ground_truth::top_k(&index, embedding_field, metric, &query, k)?;
                let collector = TopDocs::with_limit(k)
                    .order_by_similarity(embedding_field, query.to_vec())
                    .with_adaptive_params(params.clone());
                let actual = searcher.search(&AllQuery, &collector)?;
                assert_eq!(actual.results, expected, "mixed query={query:?} k={k}");
            }
        }
        Ok(())
    }
}

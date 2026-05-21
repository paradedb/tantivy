//! Top-N vector-similarity collector.
//!
//! Unlike the other `TopDocs::order_by_*` paths, vector similarity is
//! not a [`SortKeyComputer`](crate::collector::sort_key::SortKeyComputer).
//! IVF needs to drain the filter `DocSet` into a bitmap upfront and
//! drive its own cluster iteration — that inverts the per-doc pull
//! model that sort-key computers assume. So this is its own
//! [`Collector`] with an overridden [`Collector::collect_segment`] that
//! hands the filter `Weight` down to the per-segment
//! [`VectorBackend`](super::backend::VectorBackend), which owns the
//! loop. Flat fits the model trivially; IVF gets to drive.

use std::cmp::Ordering;
use std::sync::Arc;

use super::backend::VectorBackend;
use super::ivf::AdaptiveProbeParams;
use super::options::VectorElement;
use crate::collector::{Collector, SegmentCollector};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, TantivyError};

/// Top-N by vector similarity. Returns documents in descending
/// similarity order. Only docs that actually have a vector are
/// returned — docs that match the filter but lack a vector for `field`
/// are dropped (this is required for IVF compatibility, which can't
/// see vectorless docs at all).
///
/// Generic over `T: VectorElement` — `T` must match the schema's
/// declared dtype, checked at [`Collector::check_schema`] time.
pub struct TopDocsByVectorSimilarity<T: VectorElement> {
    field: Field,
    query: Arc<Vec<T>>,
    limit: usize,
    offset: usize,
    adaptive: AdaptiveProbeParams,
}

impl<T: VectorElement> TopDocsByVectorSimilarity<T> {
    pub fn new(field: Field, query: Vec<T>, limit: usize) -> Self {
        Self {
            field,
            query: Arc::new(query),
            limit,
            offset: 0,
            adaptive: AdaptiveProbeParams::default(),
        }
    }

    /// Drop the first `offset` results in the global ranking — used to
    /// paginate. Each segment still produces its top `limit + offset`
    /// to ensure the global window has enough candidates.
    pub fn and_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Override the adaptive probing parameters (ignored by flat-only
    /// segments).
    pub fn with_adaptive_params(mut self, params: AdaptiveProbeParams) -> Self {
        self.adaptive = params;
        self
    }

    fn segment_top_n(&self) -> usize {
        self.limit.saturating_add(self.offset)
    }
}

impl<T: VectorElement> Collector for TopDocsByVectorSimilarity<T> {
    type Fruit = Vec<(Score, DocAddress)>;
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
        Ok(())
    }

    fn for_segment(
        &self,
        _segment_local_id: SegmentOrdinal,
        _reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        // Never called at runtime — we override `collect_segment`. The
        // child type exists only to satisfy the trait bound.
        Ok(NoOpSegmentCollector)
    }

    fn requires_scoring(&self) -> bool {
        // Similarity is computed from the stored vectors, not from the
        // filter's BM25 score — let tantivy take the no-score fast path.
        false
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let backend = VectorBackend::for_segment(
            reader,
            segment_ord,
            self.field,
            Arc::clone(&self.query),
            self.adaptive.clone(),
        )?;
        backend.top_n(weight, reader, self.segment_top_n())
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<Vec<(Score, DocAddress)>>,
    ) -> crate::Result<Self::Fruit> {
        // Per-segment fruits are each already top-(limit+offset);
        // flatten, sort descending, drop offset, truncate to limit.
        let mut all: Vec<(Score, DocAddress)> = segment_fruits.into_iter().flatten().collect();
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        if self.offset >= all.len() {
            return Ok(Vec::new());
        }
        all.drain(..self.offset);
        all.truncate(self.limit);
        Ok(all)
    }
}

/// Trait-bound shim: the collector overrides [`Collector::collect_segment`]
/// so the per-doc path never fires, but the `Child: SegmentCollector`
/// bound on `Collector` still has to be satisfied.
pub struct NoOpSegmentCollector;

impl SegmentCollector for NoOpSegmentCollector {
    type Fruit = Vec<(Score, DocAddress)>;
    fn collect(&mut self, _doc: DocId, _score: Score) {}
    fn harvest(self) -> Self::Fruit {
        Vec::new()
    }
}

#[cfg(test)]
mod ivf_e2e_tests {
    //! End-to-end coverage: drives the full
    //! `searcher.search → TopDocsByVectorSimilarity → collect_segment
    //! → IvfBackend::top_n → merge_fruits` path and asserts the
    //! resulting global top-K matches a brute-force oracle computed
    //! across every doc in every segment.

    use std::sync::Arc;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::SegmentId;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::vector::ivf::test_harness::{
        brute_force_global_oracle, build_ivf_segment, ParametricClusterer,
    };
    use crate::vector::reader::{VectorColumn, VectorColumnReader};
    use crate::vector::{Metric, VectorOptions, VectorReader, VECTOR_PLUGIN_NAME};
    use crate::{Index, IndexSettings, IndexWriter, TantivyDocument};

    /// Build an index with `vector_clustering_threshold = 1` and the
    /// supplied centroids, returning the `Index` + (category, vector)
    /// field handles. Caller orchestrates commits and merges through
    /// the returned writer.
    fn open_index_for_e2e(
        dim: usize,
        metric: Metric,
        centroids: Vec<Vec<f32>>,
    ) -> crate::Result<(Index, Field, Field)> {
        let mut sb = Schema::builder();
        let cat = sb.add_text_field("category", TEXT | STORED);
        let vec_f = sb.add_vector_field("embedding", VectorOptions::new(dim, metric));
        let schema = sb.build();
        let clusterer = ParametricClusterer::new(centroids);
        let index = Index::builder()
            .schema(schema)
            .settings(IndexSettings {
                vector_clustering_threshold: 1,
                ..IndexSettings::default()
            })
            .ivf_clusterer(clusterer)
            .create_in_ram()?;
        Ok((index, cat, vec_f))
    }

    fn add_batch(
        writer: &mut IndexWriter,
        cat_field: Field,
        vec_field: Field,
        batch: &[(&str, Vec<f32>)],
    ) -> crate::Result<()> {
        for (cat, v) in batch {
            let mut doc = TantivyDocument::new();
            doc.add_text(cat_field, cat);
            doc.add_vector(vec_field, v.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok(())
    }

    fn column_variant(
        index: &Index,
        vec_field: Field,
        seg_idx: usize,
    ) -> crate::Result<&'static str> {
        let searcher = index.reader()?.searcher();
        let seg = &searcher.segment_readers()[seg_idx];
        let vr: Arc<VectorReader> = seg
            .plugin_reader::<VectorReader>(VECTOR_PLUGIN_NAME)?
            .expect("plugin reader");
        Ok(match vr.open_column(vec_field)? {
            VectorColumn::Flat(_) => "Flat",
            VectorColumn::Ivf(_) => "Ivf",
        })
    }

    fn assert_addr_match(
        hits: &[(Score, DocAddress)],
        oracle: &[(Score, DocAddress)],
        label: &str,
    ) {
        assert_eq!(hits.len(), oracle.len(), "{label}: len");
        for (i, ((s1, a1), (s2, a2))) in hits.iter().zip(oracle.iter()).enumerate() {
            assert_eq!(a1, a2, "{label}: rank {i} addr mismatch");
            assert!(
                (s1 - s2).abs() < 1e-5,
                "{label}: rank {i} score {s1} vs {s2}",
            );
        }
    }

    // Single IVF segment → searcher.search → matches the brute-force
    // oracle. Validates the IvfBackend → collector wiring.
    #[test]
    fn e2e_single_ivf_segment_matches_oracle() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("a", vec![2.2, 2.7]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
            ("b", vec![8.2, 12.1]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let searcher = fixture.index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        for query in [vec![0.5_f32, 0.5], vec![9.7, 10.3], vec![5.0, 2.0]] {
            for k in [1usize, 3, 6] {
                let oracle = brute_force_global_oracle(
                    &fixture.index,
                    fixture.vec_field,
                    Metric::L2,
                    &query,
                    k,
                )?;
                let hits = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k).order_by_similarity(fixture.vec_field, query.clone()),
                )?;
                assert_addr_match(&hits, &oracle, &format!("single-IVF query={query:?} k={k}"));
            }
        }
        Ok(())
    }

    // Multi-segment: two IVF segments. The collector's `merge_fruits`
    // must flatten and re-sort to produce the global top-K matching
    // the cross-segment oracle.
    #[test]
    fn e2e_multi_ivf_segments_match_global_oracle() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let (index, cat, vec_f) = open_index_for_e2e(2, Metric::L2, centroids)?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // First IVF segment: two commits → merge.
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("a", vec![1.0_f32, 0.2]), ("a", vec![0.3, 1.1])],
        )?;
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("b", vec![9.4, 9.1]), ("b", vec![11.3, 10.6])],
        )?;
        let first_pair: Vec<SegmentId> = index.searchable_segment_ids()?.into_iter().collect();
        assert_eq!(first_pair.len(), 2);
        writer.merge(&first_pair).wait()?;
        let after_first = index
            .searchable_segment_ids()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            after_first.len(),
            1,
            "first merge collapses to one IVF segment"
        );

        // Second IVF segment: two more commits → merge only the new
        // ones, leaving the first IVF segment alone.
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("a", vec![2.2, 2.7]), ("b", vec![8.2, 12.1])],
        )?;
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("a", vec![0.7, 1.3]), ("b", vec![10.1, 9.8])],
        )?;
        let all = index
            .searchable_segment_ids()?
            .into_iter()
            .collect::<Vec<_>>();
        let new_ones: Vec<SegmentId> = all
            .into_iter()
            .filter(|s| !after_first.contains(s))
            .collect();
        assert_eq!(new_ones.len(), 2);
        writer.merge(&new_ones).wait()?;
        writer.wait_merging_threads()?;

        let final_count = index.searchable_segment_ids()?.len();
        assert_eq!(final_count, 2, "should now have two IVF segments");
        // Both segments must be IVF.
        assert_eq!(column_variant(&index, vec_f, 0)?, "Ivf");
        assert_eq!(column_variant(&index, vec_f, 1)?, "Ivf");

        let searcher = index.reader()?.searcher();
        for query in [vec![0.5_f32, 0.5], vec![9.7, 10.3]] {
            for k in [1usize, 4, 8] {
                let oracle = brute_force_global_oracle(&index, vec_f, Metric::L2, &query, k)?;
                let hits = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k).order_by_similarity(vec_f, query.clone()),
                )?;
                assert_addr_match(&hits, &oracle, &format!("multi-IVF query={query:?} k={k}"));
            }
        }
        Ok(())
    }

    // Offset: with `and_offset(n)`, the result is the global oracle's
    // [n, n+k) slice.
    #[test]
    fn e2e_offset_window_matches_oracle_slice() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("a", vec![2.2, 2.7]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
            ("b", vec![8.2, 12.1]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let searcher = fixture.index.reader()?.searcher();
        let query = vec![0.5_f32, 0.5];
        let k = 2;
        let offset = 2;
        let full = brute_force_global_oracle(
            &fixture.index,
            fixture.vec_field,
            Metric::L2,
            &query,
            offset + k,
        )?;
        let expected = full[offset..].to_vec();
        let hits = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(k)
                .and_offset(offset)
                .order_by_similarity(fixture.vec_field, query),
        )?;
        assert_addr_match(&hits, &expected, "offset window");
        Ok(())
    }

    // Mixed Flat + IVF segments in one index: the collector dispatches
    // per-segment to the correct backend and the merged result still
    // matches the global oracle.
    #[test]
    fn e2e_mixed_flat_and_ivf_match_global_oracle() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let (index, cat, vec_f) = open_index_for_e2e(2, Metric::L2, centroids)?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Two commits merged into one IVF segment.
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("a", vec![1.0_f32, 0.2]), ("b", vec![9.4, 9.1])],
        )?;
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[("a", vec![0.3, 1.1]), ("b", vec![11.3, 10.6])],
        )?;
        let first_pair: Vec<SegmentId> = index.searchable_segment_ids()?.into_iter().collect();
        writer.merge(&first_pair).wait()?;

        // One more commit that stays flat — no further merge. Sized
        // larger than the largest K below so the flat backend's
        // `TopNComputer` actually truncates (i.e., the test exercises
        // the truncation path on both formats, not just the IVF
        // side).
        add_batch(
            &mut writer,
            cat,
            vec_f,
            &[
                ("a", vec![0.5_f32, 0.5]),
                ("a", vec![0.7, 0.4]),
                ("a", vec![0.2, 0.9]),
                ("a", vec![1.4, 0.1]),
                ("b", vec![8.2, 12.1]),
                ("b", vec![10.4, 8.9]),
                ("b", vec![9.6, 10.7]),
                ("b", vec![11.7, 9.3]),
            ],
        )?;
        writer.wait_merging_threads()?;

        let segs = index.searchable_segment_ids()?.len();
        assert_eq!(segs, 2, "expect 1 IVF + 1 flat segment");
        // Confirm format variety.
        let mut variants: Vec<&str> = (0..segs)
            .map(|i| column_variant(&index, vec_f, i).expect("variant"))
            .collect();
        variants.sort();
        assert_eq!(variants, vec!["Flat", "Ivf"]);

        let searcher = index.reader()?.searcher();
        for query in [vec![0.4_f32, 0.4], vec![9.5, 10.1]] {
            for k in [1usize, 3, 6] {
                let oracle = brute_force_global_oracle(&index, vec_f, Metric::L2, &query, k)?;
                let hits = searcher.search(
                    &AllQuery,
                    &TopDocs::with_limit(k).order_by_similarity(vec_f, query.clone()),
                )?;
                assert_addr_match(
                    &hits,
                    &oracle,
                    &format!("mixed Flat+IVF query={query:?} k={k}"),
                );
            }
        }
        Ok(())
    }
}

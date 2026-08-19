//! Top-N vector-similarity collector.
//!
//! Unlike the other `TopDocs::order_by_*` paths, the *primary* sort key here is
//! not a [`SortKeyComputer`](crate::collector::sort_key::SortKeyComputer). IVF
//! needs to drain the filter `DocSet` into a bitmap upfront and drive its own
//! cluster iteration, which inverts the per-doc pull model that sort-key
//! computers assume. So this is its own [`Collector`] with an overridden
//! [`Collector::collect_segment`] that hands the filter `Weight` down to the
//! per-segment [`VectorBackend`](super::backend::VectorBackend), which owns the
//! loop. Flat fits the pull model trivially; IVF gets to drive.
//!
//! A secondary key *is* an ordinary `SortKeyComputer` — see
//! [`TopDocsByVectorSimilarity::with_tie_break`]. The heap sorts on the
//! composite `(similarity, tie_break)`, so `SortByStaticFastValue`,
//! `SortByString` and their `(key, Order)` tuples all compose here, and
//! [`TopNComputer`](crate::collector::TopNComputer) and `compare_for_top_k` are
//! shared verbatim with the pull-model path. Only the iteration driver differs,
//! never the ordering rule.

use std::sync::Arc;

use super::backend::{ProbeStats, VectorBackend};
use super::ivf::AdaptiveProbeParams;
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

/// Top-N by vector similarity. Returns documents in descending
/// similarity order. Only docs that actually have a vector are
/// returned — docs that match the filter but lack a vector for `field`
/// are dropped (this is required for IVF compatibility, which can't
/// see vectorless docs at all).
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

    /// Order documents that tie on similarity by `tie_break`, as
    /// `ORDER BY embedding <=> $1, id` does.
    ///
    /// The tie-break takes part in each segment's top-N eviction, so it also
    /// decides *which* of a set of equally-distant documents survive, not only
    /// how the survivors are ordered. Similarity remains the primary key; the
    /// tie-break is only consulted between documents whose similarity is
    /// exactly equal.
    ///
    /// This does not change which clusters an IVF segment probes: the probe
    /// loop's stopping rule reads the routed centroids and the filter, never
    /// the top-N heap.
    ///
    /// Each segment is cut to its own top-N under the segment-local
    /// `SegmentSortKey`, and only the survivors are lifted to `SortKey` for the
    /// cross-segment merge. `convert_segment_sort_key` must therefore be
    /// order-preserving within a segment, or a segment can discard a document
    /// that would have placed globally. The bundled computers satisfy this:
    /// term ordinals ascend with their terms, and `FastValue`'s `u64` encoding
    /// is monotonic.
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
/// plus each searched segment's [`ProbeStats`], so callers can inspect or
/// aggregate probe metrics without a side channel.
#[derive(Debug, Default)]
pub struct VectorSimilarityFruit {
    /// Global top-N `(score, address)` pairs in descending-similarity order.
    pub results: Vec<(Score, DocAddress)>,
    /// One [`ProbeStats`] per collected segment, in segment-ordinal order
    /// after [`Collector::merge_fruits`]. The counter fields are summable
    /// across segments; `termination` and `bound_armed_at_probe` only
    /// carry per-segment meaning.
    pub stats: Vec<ProbeStats>,
}

/// One segment's contribution, before [`Collector::merge_fruits`] cuts the
/// global window.
///
/// Carries the tie-break value alongside each score because the cross-segment
/// merge has to order by the same composite key the per-segment heaps used.
/// The value is dropped at merge time — callers order by similarity and read
/// their own columns back themselves, so it never reaches [`VectorSimilarityFruit`].
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
        // Never called at runtime — we override `collect_segment`. The
        // child type exists only to satisfy the trait bound.
        Ok(NoOpSegmentCollector::default())
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
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        let backend = VectorBackend::for_segment(
            reader,
            segment_ord,
            self.field,
            Arc::clone(&self.query),
            self.adaptive.clone(),
        )?;
        let mut tie_break = self.tie_break.segment_sort_key_computer(reader)?;
        let (hits, stats) = backend.top_n_by(
            weight,
            reader,
            self.segment_top_n(),
            &mut tie_break,
            self.tie_break.comparator(),
        )?;
        // Lift the segment-local tie-break key to its global form, but only
        // now: a `SegmentSortKey` can be a term ordinal, which means nothing
        // outside this segment and must never reach the cross-segment merge.
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
        // Per-segment fruits are each already top-(limit+offset) under this
        // same composite order, so the global window is a plain sort of their
        // union. Stats concatenate untouched — one entry per segment, kept
        // even when the offset swallows every result.
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
        // `compare_for_top_k` is the same rule the per-segment heaps used,
        // down to the trailing ascending-`DocAddress` tie-break, so it is a
        // total order and the unstable sort is deterministic.
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

/// Trait-bound shim: the collector overrides [`Collector::collect_segment`]
/// so the per-doc path never fires, but the `Child: SegmentCollector`
/// bound on `Collector` still has to be satisfied.
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
mod tests {
    //! Schema validation still runs ahead of any storage access, and the
    //! search itself is a hard TODO until the cross-segment probe loop
    //! over the index-level centroid set lands — these tests pin both.

    use crate::collector::sort_key::SortBySimilarityScore;
    use crate::collector::TopDocs;
    use crate::query::AllQuery;
    use crate::vector::tests::TestVectorIndex;
    use crate::vector::{Metric, VectorDType};
    use crate::TantivyError;

    /// TODO(cross-segment search): searches fail loudly instead of
    /// returning wrong results.
    #[test]
    fn vector_search_is_a_hard_todo() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let collector = TopDocs::with_limit(4)
            .order_by_similarity(index.embedding_field(), vec![0.5_f32, 0.5]);
        let err = searcher.search(&AllQuery, &collector).unwrap_err();
        assert!(
            matches!(err, TantivyError::InvalidArgument(ref msg) if msg.contains("not yet implemented")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }

    /// `check_schema` failures surface BEFORE the search-TODO error: a
    /// mismatched query dim, a non-vector field, and a score tie-break
    /// are each rejected with their own message.
    #[test]
    fn check_schema_errors_precede_the_todo() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let searcher = index.index.reader()?.searcher();

        let wrong_dim = TopDocs::with_limit(2)
            .order_by_similarity(index.embedding_field(), vec![0.0_f32; 3]);
        let err = searcher.search(&AllQuery, &wrong_dim).unwrap_err();
        assert!(
            matches!(err, TantivyError::SchemaError(ref msg) if msg.contains("does not match")),
            "unexpected error: {err:?}"
        );

        let not_a_vector = TopDocs::with_limit(2)
            .order_by_similarity(index.label_field(), vec![0.0_f32, 0.0]);
        let err = searcher.search(&AllQuery, &not_a_vector).unwrap_err();
        assert!(
            matches!(err, TantivyError::SchemaError(ref msg) if msg.contains("not a vector field")),
            "unexpected error: {err:?}"
        );

        let score_tie_break = TopDocs::with_limit(2)
            .order_by_similarity(index.embedding_field(), vec![0.0_f32, 0.0])
            .with_tie_break(SortBySimilarityScore::new());
        let err = searcher.search(&AllQuery, &score_tie_break).unwrap_err();
        assert!(
            matches!(err, TantivyError::InvalidArgument(ref msg) if msg.contains("relevance score")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }
}

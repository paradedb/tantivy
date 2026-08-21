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

use super::stats::{ProbeStats, ProbeTermination};
use super::tie_break::NoTieBreak;
use super::{global_top_n_by, VectorElement};
use crate::collector::{Collector, SegmentCollector, SortKeyComputer};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::vector::ivf::AdaptiveProbeParams;
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
    segments: Option<Vec<SegmentOrdinal>>,
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
            segments: None,
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

    /// Restrict the search to the given segment ordinals (the searcher
    /// snapshot's numbering); the default is every segment. The run
    /// behaves exactly like a snapshot holding only those segments —
    /// budget capacity included — so a consumer can drive disjoint
    /// subsets from parallel workers and combine the fruits with
    /// [`VectorSimilarityFruit::merge`]. Results are identical to the
    /// single full run at exhaustive budgets; under a tight ceiling a
    /// split loses the cross-subset kth seeding, so each worker may
    /// spend its budget slightly differently.
    pub fn for_segments(mut self, segments: Vec<SegmentOrdinal>) -> Self {
        self.segments = Some(segments);
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
            segments: self.segments,
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

impl VectorSimilarityFruit {
    /// Combine per-worker fruits — from disjoint
    /// [`for_segments`](TopDocsByVectorSimilarity::for_segments) runs of
    /// the SAME query — into one top-`limit` result.
    ///
    /// Orders by `(similarity desc, DocAddress asc)`: exactly the heap's
    /// order under the default [`NoTieBreak`]. A consumer that searched
    /// with a custom tie-break must merge on its own keys instead (e.g.
    /// an external merge node keyed on the real ORDER BY columns) — the
    /// fruit does not carry the erased tie keys.
    ///
    /// Stats combine additively; `bound_armed_at_probe` keeps the
    /// earliest arming and `termination` reports `Ceiling` if any worker
    /// hit its ceiling.
    pub fn merge(fruits: Vec<VectorSimilarityFruit>, limit: usize) -> VectorSimilarityFruit {
        let mut merged = VectorSimilarityFruit::default();
        let mut first = true;
        for fruit in fruits {
            merged.results.extend(fruit.results);
            let stats = fruit.stats;
            if first {
                merged.stats = stats;
                first = false;
                continue;
            }
            let acc = &mut merged.stats;
            acc.candidates_scored += stats.candidates_scored;
            acc.vectors_visited += stats.vectors_visited;
            acc.pruned_filter += stats.pruned_filter;
            acc.pruned_dead += stats.pruned_dead;
            acc.pruned_seen += stats.pruned_seen;
            acc.postings_row += stats.postings_row;
            acc.postings_skipped += stats.postings_skipped;
            acc.segment_opens += stats.segment_opens;
            acc.routing.visited_count += stats.routing.visited_count;
            acc.bounds_skips += stats.bounds_skips;
            acc.bound_armed_at_probe = match (acc.bound_armed_at_probe, stats.bound_armed_at_probe)
            {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if stats.termination == ProbeTermination::Ceiling {
                acc.termination = super::ProbeTermination::Ceiling;
            }
            acc.work_charged += stats.work_charged;
            acc.segments_searched += stats.segments_searched;
            acc.filters_built += stats.filters_built;
            acc.exact_rows_read += stats.exact_rows_read;
        }
        merged
            .results
            .sort_by(|(score_a, addr_a), (score_b, addr_b)| {
                score_b.total_cmp(score_a).then_with(|| addr_a.cmp(addr_b))
            });
        merged.results.truncate(limit);
        merged
    }
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
            self.segments.as_deref(),
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

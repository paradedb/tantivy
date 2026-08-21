//! Top-N vector-similarity collector.
//!
//! Collection is split by tier, matching what is actually coupled:
//! clustered segments share cluster ids with the index-level centroid
//! index, so they are collected in ONE multi-segment pass — one routing
//! pass, one shared-kth heap (see the [`search`](super) driver) — through
//! [`Collector::collect_multi_segment`]. Flat segments have no
//! cross-segment state at all and ride the ordinary per-segment path:
//! [`VectorSegmentCollector`] scores each filter match through the O(1)
//! flat doc→row rank, so a selective filter visits only its matches and
//! never materializes a bitset. Every fruit meets in
//! [`Collector::merge_fruits`].
//!
//! The collector must still be the TOP-LEVEL collector of its search:
//! wrapped inside `MultiCollector` or another per-segment combinator, the
//! multi-segment plan never runs and [`Collector::for_segment`] fails
//! loudly on the first clustered segment.
//!
//! A secondary key *is* an ordinary `SortKeyComputer` — see
//! [`TopDocsByVectorSimilarity::with_tie_break`]. The global heap sorts on
//! the composite `(similarity, tie_break)`; segment-local sort keys are
//! lifted to their global form at push time, for competitive candidates
//! only.

use std::sync::Arc;

use super::prepared::PreparedQuery;
use super::stats::ProbeStats;
use super::tie_break::NoTieBreak;
use super::{global_top_n_by, VectorElement};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{
    Collector, CollectorMode, SegmentCollector, SegmentSortKeyComputer, SortKeyComputer,
    TopNComputer,
};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::VectorIndexReader;
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
            if first {
                merged.stats = fruit.stats;
                first = false;
            } else {
                merged.stats.absorb(fruit.stats);
            }
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

/// One collection pass's keyed hits plus its probe stats: produced by the
/// multi-segment pass (clustered segments, one shared heap) and by each
/// flat segment's [`VectorSegmentCollector`]; merged in
/// [`Collector::merge_fruits`]. Keys are the GLOBAL `(similarity,
/// tie_break)` composite so fruits merge without re-reading anything.
pub struct SegmentVectorFruit<K> {
    pub hits: Vec<((Score, K), DocAddress)>,
    pub stats: ProbeStats,
}

/// Per-segment exact collection for one FLAT segment: the filter drives
/// `collect`, each match resolves doc→row through the flat id-map's O(1)
/// rank, and the row is fetched and scored into a local heap. A selective
/// filter therefore visits only its matches — no filter bitset, no full
/// row scan.
pub struct VectorSegmentCollector<T: VectorElement, S: SortKeyComputer> {
    ord: SegmentOrdinal,
    /// `limit + offset`; `0` collects nothing.
    limit: usize,
    vec: Arc<VectorIndexReader>,
    prepared: PreparedQuery<T>,
    tie: S::Child,
    topn: TopNComputer<(Score, S::SortKey), DocAddress, (NaturalComparator, S::Comparator)>,
    exact_rows: usize,
}

impl<T, S> SegmentCollector for VectorSegmentCollector<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    type Fruit = SegmentVectorFruit<S::SortKey>;

    fn collect(&mut self, doc: DocId, _score: Score) {
        if self.limit == 0 {
            return;
        }
        let Some(row) = self.vec.row_id(doc) else {
            // The doc matches the filter but has no vector: IVF-style
            // semantics, vectorless docs are never returned.
            return;
        };
        let bytes = self
            .vec
            .vector_bytes_for_row(row)
            .expect("failed to read a validated vector row");
        let score = self.prepared.score_doc_bytes(&bytes);
        self.exact_rows += 1;
        // Same competitive skip as the driver: a candidate losing on
        // similarity alone never pays the tie-key conversion.
        if let Some(((threshold_score, _), _)) = &self.topn.threshold {
            if score < *threshold_score {
                return;
            }
        }
        let segment_key = self.tie.segment_sort_key(doc, score);
        let global_key = self.tie.convert_segment_sort_key(segment_key);
        self.topn
            .push_unordered((score, global_key), DocAddress::new(self.ord, doc));
    }

    fn harvest(self) -> Self::Fruit {
        let hits = self
            .topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, cd.doc))
            .collect();
        let stats = ProbeStats {
            exact_rows_read: self.exact_rows,
            segments_searched: 1,
            ..ProbeStats::default()
        };
        SegmentVectorFruit { hits, stats }
    }
}

impl<T, S> Collector for TopDocsByVectorSimilarity<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    type Fruit = VectorSimilarityFruit;
    type Child = VectorSegmentCollector<T, S>;

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
        segment_local_id: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        let vec = reader.vector_index(self.field)?;
        if vec.clusters().is_some() {
            // The plan routes clustered segments through the
            // multi-segment pass; reaching here means a per-segment
            // combinator is driving this collector.
            return Err(TantivyError::InvalidArgument(
                "clustered vector segments collect through the multi-segment pass; \
                 TopDocsByVectorSimilarity must be the top-level collector of its search (it \
                 cannot run inside MultiCollector or another per-segment combinator)"
                    .to_string(),
            ));
        }
        let metric = vec.options().metric();
        Ok(VectorSegmentCollector {
            ord: segment_local_id,
            limit: self.segment_top_n(),
            prepared: PreparedQuery::new(metric, Arc::clone(&self.query)),
            tie: self.tie_break.segment_sort_key_computer(reader)?,
            topn: TopNComputer::new_with_comparator(
                self.segment_top_n().max(1),
                (NaturalComparator, self.tie_break.comparator()),
            ),
            exact_rows: 0,
            vec,
        })
    }

    fn requires_scoring(&self) -> bool {
        // Similarity is computed from the stored vectors, not from the
        // filter's BM25 score — let tantivy take the no-score fast path.
        false
    }

    fn collection_mode(&self, searcher: &Searcher) -> crate::Result<CollectorMode> {
        // Clustered segments are coupled — shared cluster ids, one
        // routing pass, one heap; flat segments are not and collect
        // per-segment.
        let mut clustered = Vec::new();
        for (ord, reader) in searcher.segment_readers().iter().enumerate() {
            let ord = ord as SegmentOrdinal;
            if let Some(allowed) = &self.segments {
                if !allowed.contains(&ord) {
                    continue;
                }
            }
            if reader.vector_index(self.field)?.clusters().is_some() {
                clustered.push(ord);
            }
        }
        if clustered.is_empty() {
            Ok(CollectorMode::SingleSegment)
        } else {
            Ok(CollectorMode::MultiSegment(clustered))
        }
    }

    /// Per-segment collection, i.e. the flat tier. Skipped without
    /// driving the filter when the segment is outside a
    /// [`for_segments`](TopDocsByVectorSimilarity::for_segments)
    /// restriction or has no vector rows.
    fn collect_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        let restricted_out = self
            .segments
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&segment_ord));
        if restricted_out || reader.vector_index(self.field)?.num_vectors() == 0 {
            return Ok(SegmentVectorFruit {
                hits: Vec::new(),
                stats: ProbeStats::default(),
            });
        }
        let mut segment_collector = self.for_segment(segment_ord, reader)?;
        crate::collector::default_collect_segment_impl(
            &mut segment_collector,
            weight,
            reader,
            self.requires_scoring(),
        )?;
        Ok(segment_collector.harvest())
    }

    fn collect_multi_segment(
        &self,
        weight: &dyn Weight,
        searcher: &Searcher,
        segments: &[SegmentOrdinal],
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        let (hits, stats) = global_top_n_by(
            searcher,
            weight,
            self.field,
            &self.query,
            self.segment_top_n(),
            &self.adaptive,
            &self.tie_break,
            Some(segments),
        )?;
        Ok(SegmentVectorFruit { hits, stats })
    }

    fn merge_fruits(
        &self,
        fruits: Vec<SegmentVectorFruit<S::SortKey>>,
    ) -> crate::Result<Self::Fruit> {
        let mut stats = ProbeStats::default();
        let mut first = true;
        let mut topn: TopNComputer<
            (Score, S::SortKey),
            DocAddress,
            (NaturalComparator, S::Comparator),
        > = TopNComputer::new_with_comparator(
            self.segment_top_n().max(1),
            (NaturalComparator, self.tie_break.comparator()),
        );
        for fruit in fruits {
            if first {
                stats = fruit.stats;
                first = false;
            } else {
                stats.absorb(fruit.stats);
            }
            for (key, doc) in fruit.hits {
                topn.push(key, doc);
            }
        }
        let results = if self.limit == 0 {
            Vec::new()
        } else {
            topn.into_sorted_vec()
                .into_iter()
                .skip(self.offset)
                .take(self.limit)
                .map(|cd| (cd.sort_key.0, cd.doc))
                .collect()
        };
        Ok(VectorSimilarityFruit { results, stats })
    }
}

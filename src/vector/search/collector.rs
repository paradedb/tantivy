//! Top-N vector-similarity collector.
//!
//! This collector does NOT run through `Searcher::search` — the
//! [`Collector`] trait's per-segment entry points error. Collection is
//! an explicit API on the struct, split by tier to match what is
//! actually coupled:
//!
//! * [`TopDocsByVectorSimilarity::collect_ivf`] — the clustered segments in ONE pass: they share
//!   cluster ids with the index-level centroid index, so one routing pass and one shared-kth heap
//!   cover them all (the [`search`](super) driver).
//! * [`TopDocsByVectorSimilarity::collect_flat`] — one flat segment, exactly: the filter drives the
//!   scan and each match resolves doc→row through the flat id-map's O(1) rank — no filter bitset,
//!   no full row walk.
//! * [`TopDocsByVectorSimilarity::merge`] — any number of either fruit, merged into the final
//!   top-N.
//!
//! A caller that already knows its segments' tiers (a parallel worker
//! that claimed the clustered chunk or a flat segment) calls the
//! matching method directly; [`TopDocsByVectorSimilarity::search`] is
//! the serial convenience that partitions and does all three.
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
    Collector, SegmentCollector, SegmentSortKeyComputer, SortKeyComputer, TopNComputer,
};
use crate::index::SegmentReader;
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
        _segment_local_id: SegmentOrdinal,
        _reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        Err(TantivyError::InvalidArgument(
            "TopDocsByVectorSimilarity does not run through Searcher::search; drive it with \
             collect_ivf / collect_flat / merge (or the search convenience) instead"
                .to_string(),
        ))
    }

    fn requires_scoring(&self) -> bool {
        // Similarity is computed from the stored vectors, not from the
        // filter's BM25 score — let tantivy take the no-score fast path.
        false
    }

    fn merge_fruits(
        &self,
        fruits: Vec<SegmentVectorFruit<S::SortKey>>,
    ) -> crate::Result<Self::Fruit> {
        Ok(self.merge(fruits))
    }
}

impl<T, S> TopDocsByVectorSimilarity<T, S>
where
    T: VectorElement,
    S: SortKeyComputer + Send + Sync + 'static,
{
    /// The coupled pass over `segments`, which must ALL be clustered: one
    /// routing pass over the centroid index, one shared-kth heap, one
    /// work budget — behaving exactly like a snapshot holding only those
    /// segments. The fruit merges with other passes' via
    /// [`Self::merge`].
    pub fn collect_ivf(
        &self,
        searcher: &Searcher,
        query: &dyn crate::query::Query,
        segments: &[SegmentOrdinal],
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        self.check_schema(searcher.schema())?;
        for &ord in segments {
            let reader = searcher
                .segment_readers()
                .get(ord as usize)
                .ok_or_else(|| {
                    TantivyError::InvalidArgument(format!("segment ordinal {ord} is out of range"))
                })?;
            if reader.vector_index(self.field)?.clusters().is_none() {
                return Err(TantivyError::InvalidArgument(format!(
                    "collect_ivf requires clustered segments; segment {ord} is flat"
                )));
            }
        }
        let weight = query.weight(crate::query::EnableScoring::disabled_from_searcher(
            searcher,
        ))?;
        let (hits, stats) = global_top_n_by(
            searcher,
            weight.as_ref(),
            self.field,
            &self.query,
            self.segment_top_n(),
            &self.adaptive,
            &self.tie_break,
            Some(segments),
        )?;
        Ok(SegmentVectorFruit { hits, stats })
    }

    /// Exact collection of ONE flat segment: the filter drives the scan,
    /// each match resolves doc→row through the flat id-map's O(1) rank —
    /// no filter bitset, no full row walk. The fruit merges with other
    /// passes' via [`Self::merge`].
    pub fn collect_flat(
        &self,
        searcher: &Searcher,
        query: &dyn crate::query::Query,
        segment: SegmentOrdinal,
    ) -> crate::Result<SegmentVectorFruit<S::SortKey>> {
        self.check_schema(searcher.schema())?;
        let reader = searcher
            .segment_readers()
            .get(segment as usize)
            .ok_or_else(|| {
                TantivyError::InvalidArgument(format!("segment ordinal {segment} is out of range"))
            })?;
        let vec = reader.vector_index(self.field)?;
        if vec.clusters().is_some() {
            return Err(TantivyError::InvalidArgument(format!(
                "collect_flat requires a flat segment; segment {segment} is clustered"
            )));
        }
        let mut segment_collector: VectorSegmentCollector<T, S> = VectorSegmentCollector {
            ord: segment,
            limit: self.segment_top_n(),
            prepared: PreparedQuery::new(vec.options().metric(), Arc::clone(&self.query)),
            tie: self.tie_break.segment_sort_key_computer(reader)?,
            topn: TopNComputer::new_with_comparator(
                self.segment_top_n().max(1),
                (NaturalComparator, self.tie_break.comparator()),
            ),
            exact_rows: 0,
            vec,
        };
        let weight = query.weight(crate::query::EnableScoring::disabled_from_searcher(
            searcher,
        ))?;
        crate::collector::default_collect_segment_impl(
            &mut segment_collector,
            weight.as_ref(),
            reader,
            false,
        )?;
        Ok(segment_collector.harvest())
    }

    /// Merge any number of [`Self::collect_ivf`] / [`Self::collect_flat`]
    /// fruits of the SAME query into the final top-N, applying the
    /// offset.
    pub fn merge(&self, fruits: Vec<SegmentVectorFruit<S::SortKey>>) -> VectorSimilarityFruit {
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
        VectorSimilarityFruit { results, stats }
    }

    /// The serial convenience: partition the snapshot by tier, run
    /// [`Self::collect_ivf`] over the clustered segments and
    /// [`Self::collect_flat`] over each flat one, and [`Self::merge`].
    pub fn search(
        &self,
        searcher: &Searcher,
        query: &dyn crate::query::Query,
    ) -> crate::Result<VectorSimilarityFruit> {
        self.check_schema(searcher.schema())?;
        let mut clustered = Vec::new();
        let mut flat = Vec::new();
        for (ord, reader) in searcher.segment_readers().iter().enumerate() {
            let vec = reader.vector_index(self.field)?;
            if vec.clusters().is_some() {
                clustered.push(ord as SegmentOrdinal);
            } else if vec.num_vectors() > 0 {
                flat.push(ord as SegmentOrdinal);
            }
        }
        let mut fruits = Vec::with_capacity(flat.len() + 1);
        if !clustered.is_empty() {
            fruits.push(self.collect_ivf(searcher, query, &clustered)?);
        }
        for ord in flat {
            fruits.push(self.collect_flat(searcher, query, ord)?);
        }
        Ok(self.merge(fruits))
    }
}

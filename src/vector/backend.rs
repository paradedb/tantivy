//! Per-segment vector search execution.
//!
//! Built once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity)
//! around the segment's cached [`VectorIndexReader`]. The search strategy
//! branches once, on whether the reader carries an [`IvfIndex`](super::ivf::IvfIndex): with it, the
//! filter is drained into a bitmap and the routed clusters are probed
//! adaptively; without it, the filter `Scorer` is iterated doc-by-doc and
//! every vector is scored exactly. Cluster scans read document ids together with
//! exact rows or layer-zero columns. Reranking reads selected full-precision rows.
//! Supports flat scans and routed quantized scans.

use std::sync::Arc;
use std::time::Instant;

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::query::Weight;
use crate::schema::Field;
use crate::vector::index_reader::VectorIndexReader;
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::prepared::{PreparedQuery, QuantizedQueryCtx, VectorQuery};
use crate::vector::scan::heap::{tie_break_key, TieBreakHits};
use crate::vector::stats::ProbeStats;
use crate::vector::tie_break::NoTieBreak;
use crate::vector::{enter_vector_stage, Stage, VectorElement};
use crate::{DocAddress, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// Per-segment vector search: the segment's [`VectorIndexReader`] plus the
/// per-query state. Build via [`VectorBackend::for_segment`].
/// Per-segment vector search state.
pub struct VectorBackend<T: VectorElement> {
    pub(super) reader: Arc<VectorIndexReader>,
    pub(super) query: Arc<PreparedQuery<T>>,
    quantized_query: Option<Arc<QuantizedQueryCtx>>,
    scan_init_ns: u64,
    query_prep_ns: u64,
    pub(super) adaptive: AdaptiveProbeParams,
    pub(super) segment_ord: SegmentOrdinal,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Prepares a segment vector backend.
    pub(crate) fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: VectorQuery<T>,
        adaptive: AdaptiveProbeParams,
    ) -> crate::Result<Self> {
        let reader = segment_reader.vector_index(field)?;
        let prep_start = Instant::now();
        let _query_prep_stage = enter_vector_stage(Stage::QueryPrep);
        let VectorQuery { raw, quantized } = query;
        let query = Arc::new(PreparedQuery::<T>::new(reader.options().metric(), raw));
        Ok(Self {
            reader,
            query,
            quantized_query: quantized,
            scan_init_ns: 0,
            query_prep_ns: prep_start.elapsed().as_nanos() as u64,
            adaptive,
            segment_ord,
        })
    }

    pub(crate) fn add_scan_init_ns(&mut self, elapsed_ns: u64) {
        self.scan_init_ns = self.scan_init_ns.saturating_add(elapsed_ns);
    }

    pub(crate) fn add_query_prep_ns(&mut self, elapsed_ns: u64) {
        self.query_prep_ns = self.query_prep_ns.saturating_add(elapsed_ns);
    }

    pub(crate) fn query_prep_ns(&self) -> u64 {
        self.query_prep_ns
    }

    /// Top-N within this segment: probe routed clusters when the reader has
    /// an index, exact-scan otherwise. Hits come back already tagged with
    /// `DocAddress`, so the collector doesn't need a second pass to attach
    /// the segment. The segment's [`ProbeStats`] ride along: the IVF path
    /// fills the probe-loop counters, the flat/exact path only
    /// `exact_rows_read`.
    /// Returns the segment's top vector matches and probe statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when segment vector data cannot be opened or scored.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let (hits, stats) = self.top_n_by(
            weight,
            segment_reader,
            top_n,
            &mut NoTieBreak,
            NaturalComparator,
        )?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), address)| (score, address))
                .collect(),
            stats,
        ))
    }

    /// [`Self::top_n`] ordered by `(similarity, tie_break)` rather than
    /// similarity alone.
    /// Returns top matches using a secondary sort key.
    ///
    /// The tie-break participates in the heap's eviction decision, not just in
    /// the ordering of what survives: candidates that tie on similarity at the
    /// k/k+1 boundary are separated by `tie_break` before `DocId` is consulted.
    /// Applying a secondary key to the returned rows instead would be too late,
    /// since the losing ties are already gone.
    /// # Errors
    ///
    /// The similarity component is always compared with [`NaturalComparator`]
    /// ("higher is better"); `tie_comparator` orders the tail alone.
    /// Returns an error when segment vector data cannot be opened or scored.
    pub fn top_n_by<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        #[cfg(test)]
        let trace_docs = super::storage_io::test_support::with_unarmed_log(|| {
            let mut mapping = Vec::new();
            if let Some(index) = self.reader.index() {
                let mut docs = Vec::new();
                for cluster in 0..index.num_clusters() {
                    self.reader.read_doc_ids(cluster, &mut docs)?;
                    mapping.extend_from_slice(&docs);
                }
            }
            Ok::<_, crate::TantivyError>(mapping)
        })?;
        let io_before = super::storage_io::snapshot();
        let fallbacks_before = cascade::sign_word_fallback_counts();
        let mut stats = ProbeStats {
            scan_init_ns: self.scan_init_ns,
            query_prep_ns: self.query_prep_ns,
            ..Default::default()
        };
        let hits = match self.reader.index() {
            None => self.exact_top_n(
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
                &mut stats,
            )?,
            Some(index) => match &self.quantized_query {
                Some(quantized_query) => self.quantized_top_n(
                    index,
                    quantized_query,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
                None => self.approximate_top_n(
                    index,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
            },
        };
        let io_after = super::storage_io::snapshot();
        let fallbacks_after = cascade::sign_word_fallback_counts();
        for (layer, stats) in stats.layers.0.iter_mut().enumerate() {
            stats.io = io_after[layer].since(io_before[layer]);
            stats.sign_word_fallbacks = fallbacks_after[layer] - fallbacks_before[layer];
        }
        stats.rerank_io = io_after[3].since(io_before[3]);
        #[cfg(test)]
        stats.quantized_trace.translate(&trace_docs);
        Ok((hits, stats))
    }

    /// Flat/exact scan: drain the filter DocSet doc-by-doc, scoring each
    /// survivor from one stride-sized row read. Fills only the
    /// `exact_rows_read` stat.
    fn exact_top_n<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = segment_reader.alive_bitset();
        let mut rows_read = 0usize;
        // Row reads are ranged and can fail; the `for_each` closure can't
        // return an error, so the first one is parked here and re-raised
        // after the walk.
        let mut read_err: Option<TantivyError> = None;
        drop(init_stage);
        stats.scan_init_ns = stats
            .scan_init_ns
            .saturating_add(init_start.elapsed().as_nanos() as u64);
        let scan_start = Instant::now();
        let exact_scan_stage = enter_vector_stage(Stage::ExactScan);
        weight.for_each_no_score(segment_reader, &mut |docs| {
            if read_err.is_some() {
                return;
            }
            for &doc in docs {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                match self.reader.vector_bytes(doc) {
                    Ok(None) => continue,
                    Ok(Some(vbytes)) => {
                        rows_read += 1;
                        let score = self.query.score_doc_bytes(&vbytes);
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push(key, doc);
                        }
                    }
                    Err(err) => {
                        read_err = Some(err);
                        return;
                    }
                }
            }
        })?;
        if let Some(err) = read_err {
            return Err(err);
        }
        drop(exact_scan_stage);
        stats.exact_scan_ns = Some(scan_start.elapsed().as_nanos() as u64);
        stats.exact_rows_read += rows_read;
        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }
}

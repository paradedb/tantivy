//! Per-segment vector search.
//!
//! A segment without a [`ClusterIndex`] iterates the filter doc by doc and scores every vector
//! exactly. An IVF segment is planned, then scanned over its cluster source.

pub(crate) mod cluster_source;
pub(crate) mod interval;
pub(crate) mod prepared;
pub(crate) mod scan;
pub(crate) mod segment;
pub(crate) mod source;

use std::sync::Arc;

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::query::Weight;
use crate::schema::Field;
use crate::vector::params::ProbeParams;
use crate::vector::search::prepared::{QuantizedQueryCtx, ScoringQuery, VectorQuery};
use crate::vector::search::scan::heap::{tie_break_key, TieBreakHits};
#[cfg(test)]
use crate::vector::search::scan::plan::PlanOverride;
use crate::vector::search::scan::{ScanConfig, SegmentScan};
use crate::vector::search::segment::plan::{SegmentPlan, Source};
use crate::vector::search::source::matches::MatchClusters;
use crate::vector::search::source::router::RouterClusters;
use crate::vector::search::source::SourceReport;
use crate::vector::stats::ProbeStats;
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::take_stage_totals;
use crate::vector::tie_break::NoTieBreak;
use crate::vector::{enter_vector_stage, VectorElement, VectorStage};
use crate::{DocAddress, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// Per-segment vector search state: the segment's [`VectorIndexReader`] and the prepared query.
/// Build via [`SegmentSearcher::for_segment`].
pub struct SegmentSearcher<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<VectorQuery<T>>,
    /// `None` when the segment is not quantized or quantized scanning is off.
    quantized: Option<Arc<QuantizedQueryCtx>>,
    probe_params: ProbeParams,
    segment_ord: SegmentOrdinal,
    #[cfg(test)]
    pub(crate) plan_override: Option<PlanOverride>,
}

impl<T: VectorElement> SegmentSearcher<T> {
    // --- Construction ---

    /// Opens `field`'s vector reader for one segment.
    pub(crate) fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: Arc<VectorQuery<T>>,
        quantized: Option<Arc<QuantizedQueryCtx>>,
        probe_params: ProbeParams,
    ) -> crate::Result<Self> {
        Ok(Self {
            reader: segment_reader.vector_index(field)?,
            query,
            quantized,
            probe_params,
            segment_ord,
            #[cfg(test)]
            plan_override: None,
        })
    }

    /// The query as this segment scores it.
    fn scoring_query(&self) -> ScoringQuery<'_, T> {
        ScoringQuery {
            query: &self.query,
            quantized: self.quantized.as_deref(),
        }
    }

    #[cfg(test)]
    pub(crate) fn params(&self) -> &ProbeParams {
        &self.probe_params
    }

    #[cfg(test)]
    pub(crate) fn reader(&self) -> &VectorIndexReader {
        &self.reader
    }

    #[cfg(test)]
    pub(crate) fn query(&self) -> &VectorQuery<T> {
        &self.query
    }

    // --- Search ---

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

    /// [`Self::top_n`] ordered by `(similarity, tie_break)`; the tie-break takes part in the heap's
    /// eviction, so it decides which documents tied at the k-th place survive.
    ///
    /// # Errors
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
        match self.reader.index() {
            None => {
                let (hits, rows_read) =
                    self.flat_top_n(weight, segment_reader, top_n, tie_break, tie_comparator)?;
                Ok((hits, ProbeStats::flat(rows_read, take_stage_totals())))
            }
            Some(index) => self.search_ivf(
                index,
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
            ),
        }
    }

    /// Searches one IVF segment: plans it, runs the scan over its cluster source, and assembles
    /// the statistics from every stage's report and the stage clock.
    fn search_ivf<K, C>(
        &self,
        index: &ClusterIndex,
        weight: &dyn Weight,
        segment: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: C,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        C: Comparator<K::SegmentSortKey>,
    {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let plan = SegmentPlan::build(&self.probe_params, index, weight, segment, top_n)?;
        let capacity = plan.capacity_hint();
        let SegmentPlan {
            budget,
            source,
            report,
        } = plan;
        let Some(source) = source else {
            drop(init);
            return Ok((
                Vec::new(),
                ProbeStats::empty(report.located_cap, take_stage_totals()),
            ));
        };
        let config = ScanConfig {
            top_n,
            capacity,
            exact_plan: self.probe_params.exact_plan,
            #[cfg(test)]
            plan_override: self.plan_override,
        };
        let mut scan = SegmentScan::new(
            &self.reader,
            index,
            self.scoring_query(),
            config,
            tie_break,
            tie_comparator,
        )?;
        drop(init);
        let source = match source {
            Source::Router(filter) => {
                let mut clusters = RouterClusters::new(
                    &self.reader,
                    &self.query,
                    budget,
                    filter,
                    segment,
                    &self.probe_params,
                );
                scan.run(&mut clusters)?;
                SourceReport::Router(clusters.finish())
            }
            Source::FilterMatches(docs) => {
                let mut clusters = MatchClusters::locate(&self.reader, docs, segment, budget)?;
                scan.run(&mut clusters)?;
                SourceReport::Matches(clusters.finish())
            }
        };
        let (hits, scan_report) = scan.finish(self.segment_ord)?;
        Ok((
            hits,
            ProbeStats::assemble(report, source, scan_report, take_stage_totals()),
        ))
    }

    // --- Flat scan ---

    /// Scores every filtered, alive document with a vector from one row read; returns the hits and
    /// the rows read.
    fn flat_top_n<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, usize)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = segment_reader.alive_bitset();
        let mut rows_read = 0usize;
        // The `for_each` closure can't return an error; the first one is re-raised after the walk.
        let mut read_err: Option<TantivyError> = None;
        drop(init);
        let scan = enter_vector_stage(VectorStage::ExactScan);
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
                        let score = self.query.score_row(&vbytes).score();
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
        drop(scan);
        let segment_ord = self.segment_ord;
        let _assembly = enter_vector_stage(VectorStage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        Ok((hits, rows_read))
    }
}

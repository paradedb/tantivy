//! Per-segment vector search.
//!
//! A segment without a [`ClusterIndex`] iterates the filter doc by doc and scores every vector
//! exactly. An IVF segment is planned, then scanned over its cluster source.

pub(crate) mod cluster_source;
pub(crate) mod input;
pub(crate) mod interval;
pub(crate) mod prepared;
pub(crate) mod scan;
pub(crate) mod segment;
pub(crate) mod source;

use std::sync::{Arc, Mutex};

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{
    AtomicSharedThreshold, SegmentSortKeyComputer, SharedThreshold, TopNComputer,
};
use crate::query::{ExcludedDocSet, ExclusionSet, Weight};
use crate::schema::Field;
use crate::vector::params::ProbeParams;
use crate::vector::router::RouterIter;
use crate::vector::search::input::VectorSegment;
use crate::vector::search::prepared::{QuantizedQueryCtx, ScoringQuery, VectorQuery};
use crate::vector::search::scan::bound::{PruningBound, TopK};
use crate::vector::search::scan::heap::{tie_break_key, TieBreakHits};
#[cfg(test)]
use crate::vector::search::scan::plan::PlanOverride;
use crate::vector::search::scan::{ScanConfig, SegmentScan};
use crate::vector::search::segment::plan::ScanInput;
use crate::vector::search::source::controller::SharedVectorWork;
use crate::vector::search::source::matches::MatchClusters;
use crate::vector::search::source::router::{DocFilter, RouterClusters};
use crate::vector::search::source::SourceReport;
#[cfg(test)]
use crate::vector::stats::ProbeStats;
use crate::vector::stats::SegmentReport;
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
#[cfg(test)]
use crate::vector::telemetry::measure;
#[cfg(test)]
use crate::vector::tie_break::NoTieBreak;
use crate::vector::{Candidate, RouterMetrics, VectorElement, VectorFilterStrategy, VectorStage};
use crate::{DocAddress, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// Per-segment vector search state: the segment's [`VectorIndexReader`] and the prepared query.
/// Build via [`SegmentSearcher::for_segment`].
pub(crate) struct SegmentSearcher<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<VectorQuery<T>>,
    /// `None` when the segment is not quantized or quantized scanning is off.
    quantized: Option<Arc<QuantizedQueryCtx>>,
    probe_params: ProbeParams,
    segment: VectorSegment,
    ranked: Vec<Candidate>,
    routing_metrics: RouterMetrics,
    pub(crate) shared_work: Option<Arc<dyn SharedVectorWork>>,
    /// Exact scores segments of this collector have finished. The scan seeds its TopK from them
    /// (partial k). `None` seeds nothing.
    pub(crate) collector_topk: Option<Arc<Mutex<TopK>>>,
    /// The threshold this scan publishes. Defaults to a process-local
    /// [`AtomicSharedThreshold`] until the collector installs its shared slot.
    pub(crate) shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
    #[cfg(test)]
    pub(crate) plan_override: Option<PlanOverride>,
    #[cfg(test)]
    pub(crate) admission_pruning_disabled: bool,
}

impl<T: VectorElement> SegmentSearcher<T> {
    // --- Construction ---

    /// Opens `field`'s vector reader for a prepared segment.
    pub(crate) fn for_segment(
        segment: VectorSegment,
        field: Field,
        query: Arc<VectorQuery<T>>,
        quantized: Option<Arc<QuantizedQueryCtx>>,
        probe_params: ProbeParams,
    ) -> crate::Result<Self> {
        if segment.scan.is_none() {
            return Err(TantivyError::InvalidArgument(
                "vector scan requires a prepared segment".into(),
            ));
        }
        Ok(Self {
            reader: segment.reader.vector_index(field)?,
            query,
            quantized,
            probe_params,
            segment,
            ranked: Vec::new(),
            routing_metrics: RouterMetrics::Exact { visited_count: 0 },
            collector_topk: None,
            shared_work: None,
            shared_topk_threshold: Arc::new(AtomicSharedThreshold::default()),
            #[cfg(test)]
            plan_override: None,
            #[cfg(test)]
            admission_pruning_disabled: false,
        })
    }

    /// Seeds the segment [`PruningBound`] from `collector_topk` and publishes through
    /// `shared_topk_threshold`.
    pub(crate) fn with_collector_topk(
        mut self,
        collector_topk: Arc<Mutex<TopK>>,
        shared_topk_threshold: Arc<dyn SharedThreshold<Score>>,
    ) -> Self {
        self.collector_topk = Some(collector_topk);
        self.shared_topk_threshold = shared_topk_threshold;
        self
    }

    pub(crate) fn with_candidates(
        mut self,
        ranked: Vec<Candidate>,
        metrics: Option<RouterMetrics>,
    ) -> Self {
        self.ranked = ranked;
        self.routing_metrics = metrics.unwrap_or_else(|| {
            self.reader
                .index()
                .map_or(RouterMetrics::Exact { visited_count: 0 }, |index| {
                    index.opened_router().kind().empty_metrics()
                })
        });
        self
    }

    /// The query as this segment scores it.
    pub(crate) fn scoring_query(&self) -> ScoringQuery<'_, T> {
        ScoringQuery {
            query: &self.query,
            quantized: self.quantized.as_deref(),
        }
    }

    pub(crate) fn post_filter<'a>(
        &'a self,
        weight: &'a dyn Weight,
    ) -> Option<(&'a dyn Weight, &'a SegmentReader, f64)> {
        if matches!(
            &self.segment.scan,
            Some(ScanInput::Routed {
                filter: DocFilter::Post,
                ..
            })
        ) {
            let VectorFilterStrategy::Post {
                estimated_selectivity,
            } = self.probe_params.filter_strategy().unwrap()
            else {
                unreachable!()
            };
            Some((weight, &self.segment.reader, estimated_selectivity))
        } else {
            None
        }
    }

    pub(crate) fn reader(&self) -> &VectorIndexReader {
        &self.reader
    }

    pub(crate) fn segment_ord(&self) -> SegmentOrdinal {
        self.segment.ordinal
    }

    // --- Search ---

    /// Returns the segment's top vector matches and the probe statistics of a measured search.
    ///
    /// # Errors
    ///
    /// Returns an error when segment vector data cannot be opened or scored.
    #[cfg(test)]
    pub(crate) fn top_n(
        &self,
        top_n: usize,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let (searched, totals) =
            measure(|| self.top_n_by(top_n, &mut NoTieBreak, NaturalComparator, weight));
        let (hits, report) = searched?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), address)| (score, address))
                .collect(),
            ProbeStats::new(report, totals),
        ))
    }

    /// The top matches by `(similarity, tie_break)` and the segment's report; the tie-break takes
    /// part in the heap's eviction, so it decides which documents tied at the k-th place survive.
    ///
    /// # Errors
    /// Returns an error when segment vector data cannot be opened or scored.
    pub(crate) fn top_n_by<K, CTail>(
        &self,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        weight: &dyn Weight,
    ) -> crate::Result<(TieBreakHits<K>, SegmentReport)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        match self.segment.scan.as_ref().expect("prepared segment") {
            ScanInput::Flat { filter } => {
                let (hits, rows_read) =
                    self.flat_top_n(filter, top_n, tie_break, tie_comparator, weight)?;
                Ok((hits, SegmentReport::Flat { rows_read }))
            }
            input => self.search_ivf(
                self.reader.index().expect("IVF input"),
                input,
                top_n,
                tie_break,
                tie_comparator,
                weight,
            ),
        }
    }

    /// Scans one prepared IVF segment and returns its hits and stage reports.
    fn search_ivf<K, C>(
        &self,
        index: &ClusterIndex,
        input: &ScanInput,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: C,
        weight: &dyn Weight,
    ) -> crate::Result<(TieBreakHits<K>, SegmentReport)>
    where
        K: SegmentSortKeyComputer,
        C: Comparator<K::SegmentSortKey>,
    {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let capacity = input.capacity_hint();
        let report = match input {
            ScanInput::Routed { report, .. } | ScanInput::Located { report, .. } => *report,
            ScanInput::Flat { .. } => unreachable!(),
        };
        let adaptive_probing = matches!(input, ScanInput::Routed { .. })
            .then_some(self.probe_params.adaptive_probing)
            .flatten();
        let config = ScanConfig {
            adaptive_probing,
            source_may_repeat_docs: matches!(input, ScanInput::Routed { .. }),
            top_n,
            capacity,
            exact_plan: self.probe_params.exact_plan,
            #[cfg(test)]
            plan_override: self.plan_override,
            #[cfg(test)]
            admission_pruning_disabled: self.admission_pruning_disabled,
        };
        let mut bound = PruningBound::new(top_n, Arc::clone(&self.shared_topk_threshold));
        if let Some(collector_topk) = &self.collector_topk {
            bound.seed(&collector_topk.lock().unwrap());
        }
        let mut scan = SegmentScan::new(
            &self.reader,
            index,
            self.scoring_query(),
            config,
            bound,
            self.segment.ordinal,
            tie_break,
            tie_comparator,
        )?
        .with_post_filter(self.post_filter(weight));
        drop(init);
        let source = match input.clone() {
            ScanInput::Routed { filter, budget, .. } => {
                let mut clusters = RouterClusters::from_ranked(
                    &self.reader,
                    RouterIter::new(
                        self.ranked
                            .iter()
                            .map(|c| Candidate {
                                sim: c.sim,
                                node: c.node.into(),
                            })
                            .collect(),
                        self.routing_metrics,
                    ),
                    budget,
                    filter,
                    &self.segment.reader,
                    self.shared_work.clone(),
                );
                scan.run(&mut clusters)?;
                let mut window = self.ranked.len().max(1);
                while scan.needs_more_matches()
                    && !clusters.exhausted()
                    && window < index.num_clusters()
                {
                    window = window.saturating_mul(2).min(index.num_clusters());
                    let ranking = crate::vector::router::global::GlobalRanking::from(
                        &[Some(index.opened_router())],
                        &[self.segment.partition_centroid.as_deref()],
                        &self.query,
                        crate::vector::router::RoutingParams {
                            k: window,
                            rerank_factor: self.probe_params.router_rerank_factor,
                            min_beam_fraction: self.probe_params.router_min_beam_fraction,
                            aps: self.probe_params.router_aps,
                            flat: self.probe_params.router_flat,
                        },
                        &mut [crate::vector::telemetry::StageTotals::default()],
                        &mut |_| {},
                    );
                    clusters.replace_ranking(RouterIter::new(
                        ranking
                            .clusters
                            .into_iter()
                            .map(|(_, c)| crate::vector::router::candidate::Candidate {
                                node: c.node.into(),
                                sim: c.sim,
                            })
                            .collect(),
                        ranking.metrics[0],
                    ));
                    scan.retried_router();
                    scan.run(&mut clusters)?;
                }
                SourceReport::Router(clusters.finish())
            }
            ScanInput::Located { docs, budget, .. } => {
                let mut clusters =
                    MatchClusters::locate(&self.reader, docs, &self.segment.reader, budget)?;
                scan.run(&mut clusters)?;
                SourceReport::Matches(clusters.finish())
            }
            ScanInput::Flat { .. } => unreachable!(),
        };
        let (hits, scan) = scan.finish(self.segment.ordinal)?;
        Ok((
            hits,
            SegmentReport::Ivf {
                plan: report,
                source,
                scan: Box::new(scan),
            },
        ))
    }

    // --- Flat scan ---

    /// Scores every filtered, alive document with a vector from one row read; returns the hits and
    /// the rows read.
    fn flat_top_n<K, CTail>(
        &self,
        filter: &DocFilter,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        weight: &dyn Weight,
    ) -> crate::Result<(TieBreakHits<K>, usize)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let init = enter_vector_stage(VectorStage::ScanInit);
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = self.segment.reader.alive_bitset();
        let mut rows_read = 0usize;
        drop(init);
        let scan = enter_vector_stage(VectorStage::ExactScan);
        let docs = match filter {
            DocFilter::All | DocFilter::Post => {
                itertools::Either::Left(0..self.segment.reader.max_doc())
            }
            DocFilter::Docs(docs) => itertools::Either::Right(
                (0..docs.max_value().div_ceil(64)).flat_map(move |bucket| {
                    docs.tinyset(bucket)
                        .into_iter()
                        .map(move |bit| bucket * 64 + bit)
                }),
            ),
        };
        let mut scorer = if matches!(filter, DocFilter::Post) {
            let _setup = enter_vector_stage(VectorStage::LateFilterSetup);
            Some(ExcludedDocSet::new(
                weight.scorer_for_candidates_unpositioned(
                    &self.segment.reader,
                    1.0,
                    self.segment.reader.max_doc(),
                )?,
            ))
        } else {
            None
        };
        for doc in docs {
            if let Some(bs) = alive {
                if !bs.is_alive(doc) {
                    continue;
                }
            }
            if let Some(scorer) = &mut scorer {
                let _predicate = enter_vector_stage(VectorStage::LateFilterPredicate);
                if !scorer.contains(doc) {
                    continue;
                }
            }
            let Some(vbytes) = self.reader.vector_bytes(doc)? else {
                continue;
            };
            rows_read += 1;
            let score = self.query.score_row(&vbytes).score();
            if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                topn.push(key, doc);
            }
        }
        drop(scan);
        let segment_ord = self.segment.ordinal;
        let _assembly = enter_vector_stage(VectorStage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        Ok((hits, rows_read))
    }
}

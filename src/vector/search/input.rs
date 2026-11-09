use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::index::{SegmentId, SegmentReader};
use crate::query::Weight;
use crate::schema::Field;
use crate::vector::router::candidate::Candidate;
use crate::vector::search::segment::plan::{
    collect_segment_filter, located_cap, PlanReport, ScanInput, Source,
};
use crate::vector::search::source::controller::{WorkBudget, WorkUnits};
#[cfg(test)]
use crate::vector::search::source::router::DocFilter;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::{ProbeParams, VectorStage};
use crate::{Result, SegmentOrdinal, TantivyError};

/// A segment handle shared by preparation, routing and scanning.
/// Construct from `(ordinal, &reader)`, then pass it to the builder's `prepare` method.
/// Preparation attaches scan state and omits segments with no work.
#[derive(Clone)]
pub struct VectorSegment {
    /// Ordinal in the caller's segment snapshot; never renumber after filtering.
    pub ordinal: SegmentOrdinal,
    /// Segment to search.
    pub reader: SegmentReader,
    /// Coarse partition centroid in the indexed vector field's coordinate space.
    pub partition_centroid: Option<Arc<[f32]>>,
    pub(crate) scan: Option<ScanInput>,
}

impl From<(SegmentOrdinal, &SegmentReader)> for VectorSegment {
    fn from((ordinal, reader): (SegmentOrdinal, &SegmentReader)) -> Self {
        Self {
            ordinal,
            reader: reader.clone(),
            partition_centroid: None,
            scan: None,
        }
    }
}

/// A cluster address within the segment snapshot used to prepare the search.
pub type ClusterRef = (SegmentOrdinal, u32);

/// Query-wide work accounting, independent of the serialized routing candidates.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorSearchBudget {
    pub(crate) clusters: usize,
    pub(crate) docs: usize,
    pub(crate) work: WorkBudget,
    pub(crate) distributed: bool,
}

impl VectorSearchBudget {
    /// Resolves the global ceiling and minimum once over the complete search snapshot.
    pub fn for_segments(
        field: Field,
        params: &ProbeParams,
        segments: &[VectorSegment],
    ) -> Result<Self> {
        let mut clusters = 0;
        let mut docs = 0;
        for segment in segments {
            let vectors = segment.reader.vector_index(field)?;
            if let Some(index) = vectors.index() {
                clusters += index.num_clusters();
                docs += index.num_docs();
            }
        }
        let (total, n_avg, x) = params.resolved_work_budget(clusters, docs)?;
        Ok(Self {
            clusters,
            docs,
            work: WorkBudget {
                total: WorkUnits::new(total),
                per_cluster: WorkUnits::new(x),
                per_row: WorkUnits::new(if n_avg > 0.0 { (1.0 - x) / n_avg } else { 0.0 }),
            },
            distributed: false,
        })
    }

    /// Uses independently scanned segments with budgets assigned from the global ranking.
    pub fn distributed(mut self) -> Self {
        self.distributed = true;
        self
    }

    /// Allocates routed work in global candidate order, before assigning candidates to workers.
    /// `located_budget` is the sum of all workers' [`PreparedVectorSearch::located_budget`].
    /// Use the same selectivity estimate as the cluster collector, or zero when unknown.
    pub fn allocate_routing(
        self,
        candidates: &[Candidate<ClusterRef>],
        match_fraction: f64,
        located_budget: f64,
    ) -> Result<VectorRoutingBudget> {
        if !(0.0..=1.0).contains(&match_fraction)
            || !located_budget.is_finite()
            || located_budget < 0.0
        {
            return Err(TantivyError::InvalidArgument(
                "invalid routing selectivity or located budget".into(),
            ));
        }
        let x = self.work.per_cluster.get();
        let cost = x + (1.0 - x) * match_fraction;
        let mut left = (self.work.total.get() - located_budget).max(0.0);
        let mut segments = HashMap::new();
        for candidate in candidates {
            let charge = cost.min(left);
            *segments.entry(candidate.node.0).or_insert(WorkUnits::ZERO) += WorkUnits::new(charge);
            left = (left - charge).max(0.0);
        }
        Ok(VectorRoutingBudget { segments })
    }
}

/// Producer-assigned work ceilings, transferable separately from the routing candidates.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorRoutingBudget {
    segments: HashMap<SegmentOrdinal, WorkUnits>,
}

impl VectorSegment {
    pub(crate) fn routed(&self) -> bool {
        matches!(self.scan, Some(ScanInput::Routed { .. }))
    }
}

/// Evaluated filters and resolved budgets. Empty segments are absent from the work list.
/// Keep this state on the worker that evaluated the filters; routing candidates and budgets
/// can cross processes. It is scoped to one field, filter, top-k window and segment snapshot.
#[derive(Clone)]
pub struct PreparedVectorSearch {
    pub(crate) field: Field,
    pub(crate) top_n: usize,
    pub(crate) budget: VectorSearchBudget,
    pub(crate) docs: usize,
    pub(crate) segments: Vec<VectorSegment>,
    pub(crate) scope: HashMap<SegmentOrdinal, (SegmentId, usize, StageTotals)>,
}

impl PreparedVectorSearch {
    pub(crate) fn prepare(
        field: Field,
        top_n: usize,
        params: &ProbeParams,
        weight: &dyn Weight,
        segments: &[VectorSegment],
        budget: VectorSearchBudget,
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<Self> {
        let mut prepared = Self {
            field,
            top_n,
            budget,
            docs: 0,
            segments: Vec::new(),
            scope: HashMap::new(),
        };
        for segment in segments {
            if prepared.scope.contains_key(&segment.ordinal) {
                return Err(TantivyError::InvalidArgument(
                    "duplicate vector segment ordinal".into(),
                ));
            }
            let (result, totals) = measure(|| -> Result<_> {
                let vectors = segment.reader.vector_index(field)?;
                let max_doc = segment.reader.max_doc();
                let (scan, cap) = if let Some(index) = vectors.index() {
                    prepared.docs += index.num_docs();
                    if top_n == 0 || max_doc == 0 || index.num_clusters() == 0 {
                        (None, 0)
                    } else {
                        let work = WorkBudget {
                            total: budget.work.total
                                * (index.num_clusters() as f64 / budget.clusters.max(1) as f64),
                            ..budget.work
                        };
                        let x = work.per_cluster.get();
                        let n_avg = (1.0 - x) / work.per_row.get();
                        let cap = located_cap(params.located, max_doc, work.total.get(), n_avg, x);
                        let report = PlanReport {
                            segment_rows: index.num_rows(),
                            segment_clusters: index.num_clusters(),
                            located_cap: cap,
                        };
                        let _filter = enter_vector_stage(VectorStage::NonVectorSearch);
                        let scan =
                            match collect_segment_filter(weight, &segment.reader, max_doc, cap)? {
                                Source::Router(filter) if !filter.is_empty() => {
                                    Some(ScanInput::Routed {
                                        filter,
                                        budget: work,
                                        report,
                                    })
                                }
                                Source::FilterMatches(docs) if !docs.is_empty() => {
                                    Some(ScanInput::Located {
                                        docs,
                                        budget: work,
                                        report,
                                    })
                                }
                                _ => None,
                            };
                        (scan, cap)
                    }
                } else if top_n == 0 || max_doc == 0 {
                    (None, 0)
                } else {
                    let _filter = enter_vector_stage(VectorStage::NonVectorSearch);
                    let Source::Router(filter) =
                        collect_segment_filter(weight, &segment.reader, max_doc, 0)?
                    else {
                        unreachable!()
                    };
                    (
                        (!filter.is_empty()).then_some(ScanInput::Flat { filter }),
                        0,
                    )
                };
                Ok((scan, cap))
            });
            on_segment(segment.ordinal);
            let (scan, cap) = result?;
            prepared
                .scope
                .insert(segment.ordinal, (segment.reader.segment_id(), cap, totals));
            if let Some(scan) = scan {
                let mut segment = segment.clone();
                segment.scan = Some(scan);
                prepared.segments.push(segment);
            }
        }
        Ok(prepared)
    }

    /// Nonempty segments, including located and flat scans.
    pub fn segments(&self) -> &[VectorSegment] {
        &self.segments
    }

    /// Only the segments that require centroid ranking.
    pub fn routing_segments(&self) -> Vec<VectorSegment> {
        self.segments
            .iter()
            .filter(|s| s.routed())
            .cloned()
            .collect()
    }

    /// Global accounting to use when ranking or preparing other workers' assignments.
    pub fn budget(&self) -> VectorSearchBudget {
        self.budget
    }

    /// Work reserved for this worker's exhaustive located scans.
    pub fn located_budget(&self) -> f64 {
        self.segments
            .iter()
            .filter_map(|segment| match &segment.scan {
                Some(ScanInput::Located { budget, .. }) => Some(budget.total.get()),
                _ => None,
            })
            .sum()
    }

    /// Observed selectivity across the segments that require routing.
    pub fn match_fraction(&self) -> f64 {
        let mut matched = 0.0;
        let mut docs = 0u64;
        for segment in &self.segments {
            if let Some(ScanInput::Routed { filter, .. }) = &segment.scan {
                matched += filter.match_fraction(segment.reader.max_doc())
                    * f64::from(segment.reader.max_doc());
                docs += u64::from(segment.reader.max_doc());
            }
        }
        matched / docs.max(1) as f64
    }

    /// Attaches this worker's ordered candidates. Routed segments with no candidates are
    /// omitted; located and flat scans need no candidates. Candidate ordinals must use the
    /// original prepared snapshot, and each cluster may occur only once. Independent scans use
    /// the producer's allocation; the shared scan keeps one global meter.
    pub fn with_candidates(
        mut self,
        candidates: Vec<Candidate<ClusterRef>>,
        routing_budget: &VectorRoutingBudget,
    ) -> Result<VectorScanInputs> {
        let cluster_counts: HashMap<_, _> = self
            .segments
            .iter()
            .filter_map(|s| match &s.scan {
                Some(ScanInput::Routed { report, .. }) => {
                    Some((s.ordinal, report.segment_clusters))
                }
                _ => None,
            })
            .collect();
        let mut seen = HashSet::new();
        let mut routed = HashSet::new();
        for candidate in &candidates {
            let (ordinal, cluster) = candidate.node;
            let clusters = cluster_counts.get(&ordinal).ok_or_else(|| {
                TantivyError::InvalidArgument(
                    "candidate belongs to a segment that was not prepared for routing".into(),
                )
            })?;
            if cluster as usize >= *clusters || !seen.insert(candidate.node) {
                return Err(TantivyError::InvalidArgument(
                    "invalid or duplicate routing cluster".into(),
                ));
            }
            routed.insert(ordinal);
        }
        self.segments
            .retain(|s| !s.routed() || routed.contains(&s.ordinal));
        if self.budget.distributed {
            for segment in &mut self.segments {
                if let Some(ScanInput::Routed { budget, .. }) = &mut segment.scan {
                    budget.total =
                        *routing_budget
                            .segments
                            .get(&segment.ordinal)
                            .ok_or_else(|| {
                                TantivyError::InvalidArgument(
                                    "missing routed segment budget".into(),
                                )
                            })?;
                }
            }
            self.segments.retain(|segment| {
                !matches!(
                    &segment.scan,
                    Some(ScanInput::Routed { budget, .. }) if budget.total == WorkUnits::ZERO
                )
            });
        }
        Ok(VectorScanInputs {
            prepared: self,
            candidates,
            routing_stats: HashMap::new(),
        })
    }
}

/// Complete scan work: evaluated filters, budgets and an explicit ranked candidate list.
/// Construct with [`PreparedVectorSearch::with_candidates`]. There is no implicit routing.
#[derive(Clone)]
pub struct VectorScanInputs {
    pub(crate) prepared: PreparedVectorSearch,
    pub(crate) candidates: Vec<Candidate<ClusterRef>>,
    pub(crate) routing_stats: HashMap<SegmentOrdinal, crate::vector::VectorRoutingStats>,
}

impl VectorScanInputs {
    /// Attaches optional producer telemetry separately from the candidate payload.
    pub fn with_routing_statistics(
        mut self,
        stats: Vec<crate::vector::VectorRoutingStats>,
    ) -> Self {
        self.routing_stats = stats.into_iter().map(|s| (s.segment_ordinal, s)).collect();
        self
    }

    /// Segments with executable work, retaining their original ordinals.
    pub fn segments(&self) -> &[VectorSegment] {
        self.prepared.segments()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{Collector, TopDocs};
    use crate::schema::Metric;
    use crate::vector::test_support::{DocsWeight, FixtureSpec};
    use crate::vector::{LocatedPath, TopClustersBySimilarityCollector};

    #[test]
    fn routing_budget_follows_global_order_and_reserves_located_work() -> Result<()> {
        let budget = VectorSearchBudget {
            clusters: 8,
            docs: 80,
            work: WorkBudget {
                total: WorkUnits::new(3.0),
                per_cluster: WorkUnits::new(0.25),
                per_row: WorkUnits::new(0.075),
            },
            distributed: true,
        };
        let candidates: Vec<_> = [0, 0, 1, 2, 2]
            .into_iter()
            .enumerate()
            .map(|(cluster, segment)| Candidate {
                sim: Metric::L2.similarity(&[0.0f32], &[cluster as f32]),
                node: (segment, cluster as u32),
            })
            .collect();
        for (fraction, reserved, expected) in [
            (1.0, 0.0, [2.0, 1.0, 0.0]),
            (1.0, 1.0, [2.0, 0.0, 0.0]),
            (0.5, 0.0, [1.25, 0.625, 1.125]),
            (0.0, 0.0, [0.5, 0.25, 0.5]),
            (1.0, 3.0, [0.0; 3]),
        ] {
            let allocation = budget.allocate_routing(&candidates, fraction, reserved)?;
            for (ordinal, expected) in expected.into_iter().enumerate() {
                assert_eq!(allocation.segments[&(ordinal as u32)].get(), expected);
            }
            assert!(allocation.segments.values().map(|b| b.get()).sum::<f64>() + reserved <= 3.0);
        }
        for fraction in [f64::NAN, -0.1, 1.1] {
            assert!(budget.allocate_routing(&candidates, fraction, 0.0).is_err());
        }
        for reserved in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(budget.allocate_routing(&candidates, 1.0, reserved).is_err());
        }
        assert!(budget.allocate_routing(&[], 1.0, 0.0)?.segments.is_empty());
        Ok(())
    }

    #[test]
    fn empty_and_located_segments_need_no_candidates() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 384)
            .segments(3)
            .label("few", |i| i % 128 < 2)
            .paged()
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segments = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(i, reader)| VectorSegment::from((i as u32, reader)))
            .collect::<Vec<_>>();
        for segment in &segments {
            segment.reader.vector_index(fixture.field)?;
        }
        for (label, empty) in [("absent", true), ("few", false)] {
            let filter = fixture.filter(label);
            let weight = fixture.weight(&filter)?;
            let query = vec![0.0f32; 32];
            let builder = TopDocs::with_limit(5).order_by_similarity(fixture.field, query.clone());
            let budget = VectorSearchBudget::for_segments(
                fixture.field,
                &ProbeParams::default(),
                &segments,
            )?;
            fixture.clear_reads();
            let prepared = builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
            assert!(prepared.routing_segments().is_empty());
            assert_eq!(prepared.segments().is_empty(), empty);
            assert!(fixture.centroid_reads().is_empty());
            let routing_budget = budget.allocate_routing(&[], 0.0, prepared.located_budget())?;
            let inputs = prepared.with_candidates(Vec::new(), &routing_budget)?;
            let active = inputs.segments();
            assert_eq!(active.is_empty(), empty);
            let active = active.iter().map(|s| s.ordinal).collect::<Vec<_>>();
            let collector = builder.build(inputs);
            let readers = segments
                .iter()
                .filter(|s| active.contains(&s.ordinal))
                .cloned()
                .collect::<Vec<_>>();
            let fruit = collector.merge_fruits(collector.collect_segments(
                weight.as_ref(),
                &readers,
                |_| {},
            )?)?;
            if empty {
                assert!(fixture.reads().is_empty());
            }
            assert_eq!(fruit.results, fixture.oracle(&filter, &query, 5)?);
        }
        Ok(())
    }

    #[test]
    fn distributed_budget_decides_bypass_before_routing() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 256)
            .segments(2)
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segments = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(i, reader)| VectorSegment::from((i as u32, reader)))
            .collect::<Vec<_>>();
        let params = ProbeParams {
            max_probe: 2,
            ..Default::default()
        };
        let builder = TopDocs::with_limit(5)
            .order_by_similarity(fixture.field, vec![0.0f32; 32])
            .with_probe_params(params.clone());
        let segment = &segments[0];
        let local_budget = VectorSearchBudget::for_segments(
            fixture.field,
            &params,
            std::slice::from_ref(segment),
        )?;
        let cap =
            (local_budget.work.total.get() / local_budget.work.per_row.get()).floor() as usize;
        let weight = DocsWeight {
            max_doc: segment.reader.max_doc(),
            docs: (0..cap as u32).collect(),
        };
        let local =
            builder.prepare(&weight, std::slice::from_ref(segment), local_budget, |_| {})?;
        assert!(local.routing_segments().is_empty());
        let budget =
            VectorSearchBudget::for_segments(fixture.field, &params, &segments)?.distributed();
        let distributed =
            builder.prepare(&weight, std::slice::from_ref(segment), budget, |_| {})?;
        assert_eq!(distributed.routing_segments().len(), 1);
        let Some(ScanInput::Routed {
            filter: DocFilter::Docs(matches),
            ..
        }) = &distributed.segments[0].scan
        else {
            panic!("routed filter")
        };
        assert_eq!(matches.len(), cap);
        Ok(())
    }

    #[test]
    fn candidate_addresses_and_snapshot_are_validated() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 128).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segments = [VectorSegment::from((7, searcher.segment_reader(0)))];
        let params = ProbeParams {
            located: LocatedPath::Off,
            ..Default::default()
        };
        let query = vec![0.0f32; 32];
        let builder = TopDocs::with_limit(5)
            .order_by_similarity(fixture.field, query.clone())
            .with_probe_params(params.clone());
        let weight = fixture.all()?;
        let budget = VectorSearchBudget::for_segments(fixture.field, &params, &segments)?;
        let prepared = builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
        let router = TopClustersBySimilarityCollector::new(fixture.field, query, params, 1.0)?;
        let candidates = router.collect_segments(&prepared.routing_segments(), |_| {})?;
        let routing_budget =
            budget.allocate_routing(&candidates, 1.0, prepared.located_budget())?;
        let distributed =
            builder.prepare(weight.as_ref(), &segments, budget.distributed(), |_| {})?;
        let missing = budget.allocate_routing(&[], 1.0, 0.0)?;
        assert!(distributed
            .with_candidates(candidates.clone(), &missing)
            .is_err());
        assert!(candidates.iter().all(|c| c.node.0 == 7));
        let mut invalid = candidates.clone();
        invalid[0].node.0 = 0;
        assert!(prepared
            .clone()
            .with_candidates(invalid, &routing_budget)
            .is_err());
        let mut invalid = candidates.clone();
        invalid[0].node.1 = u32::MAX;
        assert!(prepared
            .clone()
            .with_candidates(invalid, &routing_budget)
            .is_err());
        let mut invalid = candidates.clone();
        invalid.push(candidates[0]);
        assert!(prepared
            .clone()
            .with_candidates(invalid, &routing_budget)
            .is_err());
        assert!(prepared
            .clone()
            .with_candidates(Vec::new(), &routing_budget)?
            .segments()
            .is_empty());
        let collector = builder.build(prepared.with_candidates(candidates, &routing_budget)?);
        assert!(collector
            .collect_segment(weight.as_ref(), 0, &segments[0].reader)
            .is_err());
        let other = FixtureSpec::clustered(Metric::L2, 32, 8, 256)
            .segments(2)
            .build()?;
        let other_searcher = other.index.reader()?.searcher();
        let other_segment = other_searcher
            .segment_readers()
            .iter()
            .find(|s| s.segment_id() != segments[0].reader.segment_id())
            .unwrap();
        assert!(collector
            .collect_segment(weight.as_ref(), 7, other_segment)
            .is_err());
        let fruit = collector.collect_segment(weight.as_ref(), 7, &segments[0].reader)?;
        assert_eq!(collector.merge_fruits(vec![fruit])?.results.len(), 5);
        Ok(())
    }
}

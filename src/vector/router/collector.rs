use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::collector::{Collector, SegmentCollector};
use crate::index::SegmentReader;
use crate::query::Weight;
use crate::schema::{Field, Schema};
use crate::vector::router::candidate::Candidate;
use crate::vector::router::global::GlobalRanking;
use crate::vector::router::{RouterMetrics, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::{measure, StageTotals};
use crate::vector::{ClusterRef, ProbeParams, VectorElement, VectorSearchBudget, VectorSegment};
use crate::{DocId, Result, Score, SegmentOrdinal, TantivyError};

/// Per-segment routing telemetry, kept separate from the ranked candidate list.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorRoutingStats {
    /// Ordinal in the search snapshot.
    pub segment_ordinal: SegmentOrdinal,
    /// Router work counters.
    pub metrics: RouterMetrics,
    pub(crate) totals: StageTotals,
}

/// Ranks clusters globally without filtering or reading document rows. Callers supply only
/// the segments that need routing. Its fruit retains the global router's ordered candidates;
/// scanners enforce the work budget even when cluster sizes differ. The result is limited to
/// the requested candidate count, including when a router scores extra clusters.
pub struct TopClustersBySimilarityCollector<T: VectorElement> {
    field: Field,
    query: Arc<Vec<T>>,
    params: ProbeParams,
    match_fraction: f64,
    limit_candidates: bool,
    budget: Option<VectorSearchBudget>,
    partition_centroids: HashMap<SegmentOrdinal, Vec<f32>>,
    stats: Mutex<Vec<VectorRoutingStats>>,
}

impl<T: VectorElement> TopClustersBySimilarityCollector<T> {
    /// Creates a cluster collector. Use selectivity zero when it is unknown, to retain enough
    /// candidates for the cluster-open budget.
    pub fn new(
        field: Field,
        query: Vec<T>,
        params: ProbeParams,
        match_fraction: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&match_fraction) {
            return Err(TantivyError::InvalidArgument(
                "routing match fraction must be between zero and one".into(),
            ));
        }
        Ok(Self {
            field,
            query: Arc::new(query),
            params,
            match_fraction,
            limit_candidates: true,
            budget: None,
            partition_centroids: HashMap::new(),
            stats: Mutex::new(Vec::new()),
        })
    }

    /// Leaves the routed candidate list uncapped so one global scan can enforce its work budget.
    pub fn without_candidate_limit(mut self) -> Self {
        self.limit_candidates = false;
        self
    }

    /// Uses accounting from the full search, including segments bypassing routing.
    pub fn with_budget(mut self, budget: VectorSearchBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Adds a coarse partition centroid for use through the ordinary `Collector` interface.
    pub fn with_partition_centroid(mut self, ordinal: SegmentOrdinal, centroid: Vec<f32>) -> Self {
        self.partition_centroids.insert(ordinal, centroid);
        self
    }

    /// Takes the most recent routing telemetry without changing the candidate result.
    pub fn take_statistics(&self) -> Vec<VectorRoutingStats> {
        std::mem::take(&mut *self.stats.lock().unwrap())
    }

    fn prepare_segment(&self, segment: &VectorSegment) -> Result<ClusterSegmentCollector> {
        self.check_schema(segment.reader.schema())?;
        let centroid = segment.partition_centroid.as_deref().or_else(|| {
            self.partition_centroids
                .get(&segment.ordinal)
                .map(Vec::as_slice)
        });
        if centroid.is_some_and(|c| c.len() != self.query.len() || c.iter().any(|x| !x.is_finite()))
        {
            return Err(TantivyError::InvalidArgument(
                "invalid vector partition centroid".into(),
            ));
        }
        let (vectors, totals) = measure(|| segment.reader.vector_index(self.field));
        Ok(ClusterSegmentCollector {
            ordinal: segment.ordinal,
            vectors: vectors?,
            centroid: centroid.map(<[f32]>::to_vec),
            totals,
        })
    }

    /// Collects the supplied segments, preserving ordinals and partition centroids.
    pub fn collect_segments(
        &self,
        segments: &[VectorSegment],
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<Vec<Candidate<ClusterRef>>> {
        let fruits = segments
            .iter()
            .map(|segment| {
                let fruit = self.prepare_segment(segment);
                on_segment(segment.ordinal);
                fruit
            })
            .collect::<Result<Vec<_>>>()?;
        self.rank(fruits, on_segment)
    }

    fn rank(
        &self,
        mut segments: Vec<ClusterSegmentCollector>,
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<Vec<Candidate<ClusterRef>>> {
        if segments.is_empty() {
            self.stats.lock().unwrap().clear();
            return Ok(Vec::new());
        }
        segments.sort_unstable_by_key(|s| s.ordinal);
        let mut ordinals = std::collections::HashSet::new();
        if segments.iter().any(|s| !ordinals.insert(s.ordinal)) {
            return Err(TantivyError::InvalidArgument(
                "duplicate routing segment ordinal".into(),
            ));
        }
        let clusters = segments
            .iter()
            .filter_map(|s| s.vectors.index())
            .map(|i| i.num_clusters())
            .sum();
        let rows = segments
            .iter()
            .filter_map(|s| s.vectors.index())
            .map(|i| i.num_rows())
            .sum();
        let (total, _, x) = self.params.resolved_work_budget(clusters, rows)?;
        let (total, x, capacity) = self.budget.map_or((total, x, clusters), |b| {
            (b.work.total.get(), b.work.per_cluster.get(), b.clusters)
        });
        let routing = RoutingParams {
            k: self
                .params
                .router_k(total, x, self.match_fraction, capacity)
                .min(clusters),
            rerank_factor: self.params.router_rerank_factor,
            min_beam_fraction: self.params.router_min_beam_fraction,
            aps: self.params.router_aps,
            flat: self.params.router_flat,
        };
        let query = VectorQuery::new(
            segments[0].vectors.options().metric(),
            Arc::clone(&self.query),
        );
        let routers = segments
            .iter()
            .map(|s| s.vectors.index().map(|i| i.opened_router()))
            .collect::<Vec<_>>();
        let partitions = segments
            .iter()
            .map(|s| s.centroid.as_deref())
            .collect::<Vec<_>>();
        let mut totals = segments
            .iter()
            .map(|s| s.totals.clone())
            .collect::<Vec<_>>();
        let ranking = GlobalRanking::from(
            &routers,
            &partitions,
            &query,
            routing,
            &mut totals,
            &mut |i| on_segment(segments[i].ordinal),
        );
        *self.stats.lock().unwrap() = segments
            .iter()
            .zip(ranking.metrics)
            .zip(totals)
            .map(|((segment, metrics), totals)| VectorRoutingStats {
                segment_ordinal: segment.ordinal,
                metrics,
                totals,
            })
            .collect();
        Ok(ranking
            .clusters
            .into_iter()
            .take(if self.limit_candidates {
                routing.k
            } else {
                usize::MAX
            })
            .map(|(i, candidate)| Candidate {
                sim: candidate.sim,
                node: (segments[i].ordinal, candidate.node),
            })
            .collect())
    }
}

impl<T: VectorElement> Collector for TopClustersBySimilarityCollector<T> {
    type Fruit = Vec<Candidate<ClusterRef>>;
    type Child = ClusterSegmentCollector;

    fn check_schema(&self, schema: &Schema) -> Result<()> {
        VectorQuery::check_schema(self.field, &self.query, schema)
    }
    fn requires_scoring(&self) -> bool {
        false
    }
    fn for_segment(&self, ordinal: SegmentOrdinal, reader: &SegmentReader) -> Result<Self::Child> {
        self.prepare_segment(&VectorSegment::from((ordinal, reader)))
    }
    fn collect_segment(
        &self,
        _weight: &dyn Weight,
        ordinal: SegmentOrdinal,
        reader: &SegmentReader,
    ) -> Result<ClusterSegmentCollector> {
        self.for_segment(ordinal, reader)
    }
    fn merge_fruits(&self, fruits: Vec<ClusterSegmentCollector>) -> Result<Self::Fruit> {
        self.rank(fruits, |_| {})
    }
}

/// Segment-local router state collected for the global merge.
pub struct ClusterSegmentCollector {
    ordinal: SegmentOrdinal,
    vectors: Arc<VectorIndexReader>,
    centroid: Option<Vec<f32>>,
    totals: StageTotals,
}

impl SegmentCollector for ClusterSegmentCollector {
    type Fruit = Self;
    fn collect(&mut self, _doc: DocId, _score: Score) {}
    fn harvest(self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    #[test]
    fn collector_merges_global_candidates_without_scanning_documents() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 384)
            .segments(3)
            .paged()
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let mut segments = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(i, reader)| VectorSegment::from((i as u32, reader)))
            .collect::<Vec<_>>();
        for segment in &segments {
            segment.reader.vector_index(fixture.field)?;
        }
        let collector = TopClustersBySimilarityCollector::new(
            fixture.field,
            vec![0.0f32; 32],
            ProbeParams::default(),
            1.0,
        )?;
        fixture.clear_reads();
        let merged = searcher.search(&fixture.filter("absent"), &collector)?;
        assert!(!merged.is_empty());
        assert!(fixture.reads().is_empty());
        let statistics = collector.take_statistics();
        assert_eq!(statistics.len(), segments.len());
        assert!(statistics.iter().all(|s| s
            .totals
            .time_ns(crate::vector::VectorStage::NonVectorSearch)
            == 0));
        segments.reverse();
        assert_eq!(merged, collector.collect_segments(&segments, |_| {})?);
        let bytes = serde_json::to_vec(&merged).unwrap();
        let restored: Vec<Candidate<ClusterRef>> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(merged, restored);
        Ok(())
    }

    #[test]
    fn routing_validates_schema_and_partition_centroids() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 128).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segment = VectorSegment::from((0, searcher.segment_reader(0)));
        for (field, query, message) in [
            (fixture.label, vec![0.0f32; 32], "not a vector field"),
            (fixture.field, vec![0.0f32; 31], "query vector length"),
        ] {
            let collector =
                TopClustersBySimilarityCollector::new(field, query, ProbeParams::default(), 1.0)?;
            let error = collector
                .collect_segments(std::slice::from_ref(&segment), |_| {})
                .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
        for fraction in [-0.1, 1.1, f64::NAN] {
            assert!(TopClustersBySimilarityCollector::new(
                fixture.field,
                vec![0.0f32; 32],
                ProbeParams::default(),
                fraction
            )
            .is_err());
        }
        for centroid in [vec![0.0f32; 31], vec![f32::NAN; 32]] {
            let collector = TopClustersBySimilarityCollector::new(
                fixture.field,
                vec![0.0f32; 32],
                ProbeParams::default(),
                1.0,
            )?;
            assert!(collector
                .collect_segments(
                    &[VectorSegment {
                        partition_centroid: Some(centroid.clone().into()),
                        ..segment.clone()
                    }],
                    |_| {}
                )
                .is_err());
            assert!(collector
                .with_partition_centroid(0, centroid)
                .collect_segment(fixture.all()?.as_ref(), 0, &segment.reader)
                .is_err());
        }
        Ok(())
    }
}

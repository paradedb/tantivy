use std::collections::HashMap;
use std::sync::Arc;

use super::candidate::Candidate;
use super::global::GlobalRanking;
use super::{RouterMetrics, RoutingParams};
use crate::index::{SegmentId, SegmentReader};
use crate::schema::Field;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::{measure, StageTotals};
use crate::vector::{ProbeParams, VectorElement};
use crate::{Result, SegmentOrdinal, TantivyError};

/// A segment and its optional parent centroid in the global router.
#[derive(Clone, Copy)]
pub struct VectorSegment<'a> {
    /// Ordinal used in returned document addresses.
    pub ordinal: SegmentOrdinal,
    /// Segment to search.
    pub reader: &'a SegmentReader,
    /// Coarse partition centroid, in the indexed vector field's coordinate space.
    /// Segments without one join at their own topmost router layer.
    pub partition_centroid: Option<&'a [f32]>,
}

impl<'a> From<(SegmentOrdinal, &'a SegmentReader)> for VectorSegment<'a> {
    fn from((ordinal, reader): (SegmentOrdinal, &'a SegmentReader)) -> Self {
        Self {
            ordinal,
            reader,
            partition_centroid: None,
        }
    }
}

/// A global centroid ranking split by segment, for sharing routing across processes.
/// Consumers must use the same field, query, probe parameters, and segment snapshot.
/// Each consumer evaluates its filter and enforces its segment's probe budget.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorRoutingPlan {
    pub(crate) field: Field,
    pub(crate) num_clusters: usize,
    pub(crate) segments: HashMap<SegmentId, SegmentRoutingPlan>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentRoutingPlan {
    pub(crate) clusters: Vec<Candidate>,
    pub(crate) metrics: RouterMetrics,
    pub(crate) totals: StageTotals,
}

impl SegmentRoutingPlan {
    pub(crate) fn ranked_clusters(&self) -> crate::vector::router::RouterIter {
        crate::vector::router::RouterIter::new(
            self.clusters
                .iter()
                .map(|candidate| Candidate {
                    sim: candidate.sim,
                    node: candidate.node.into(),
                })
                .collect(),
            self.metrics,
        )
    }
}

/// Supplies a shared routing plan after a segment has evaluated its filter.
pub trait VectorRoutingPlanProvider: Send + Sync {
    /// May block until publication; every call must return the same plan.
    fn get(&self) -> Result<&VectorRoutingPlan>;
}

impl VectorRoutingPlanProvider for VectorRoutingPlan {
    fn get(&self) -> Result<&VectorRoutingPlan> {
        Ok(self)
    }
}

impl VectorRoutingPlan {
    /// Ranks centroids without evaluating filters or reading cluster rows.
    /// `match_fraction` estimates the filter's selectivity for sizing the ranking;
    /// use zero when unknown to retain enough candidates for the cluster-open budget.
    pub fn build<T: VectorElement>(
        field: Field,
        query: Vec<T>,
        probe_params: ProbeParams,
        match_fraction: f64,
        segments: &[VectorSegment<'_>],
        mut on_segment: impl FnMut(SegmentOrdinal),
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&match_fraction) {
            return Err(TantivyError::InvalidArgument(
                "routing match fraction must be between zero and one".into(),
            ));
        }
        let mut vectors = Vec::with_capacity(segments.len());
        let mut totals = Vec::with_capacity(segments.len());
        for segment in segments {
            let reader = segment.reader;
            VectorQuery::check_schema(field, &query, reader.schema())?;
            if let Some(centroid) = segment.partition_centroid {
                if centroid.len() != query.len() || centroid.iter().any(|x| !x.is_finite()) {
                    return Err(TantivyError::InvalidArgument(
                        "invalid vector partition centroid".into(),
                    ));
                }
            }
            let (vector, measured) = measure(|| reader.vector_index(field));
            on_segment(segment.ordinal);
            vectors.push(vector?);
            totals.push(measured);
        }
        let clusters = vectors
            .iter()
            .filter_map(|v| v.index())
            .map(|i| i.num_clusters())
            .sum();
        let docs = vectors
            .iter()
            .filter_map(|v| v.index())
            .map(|i| i.num_docs())
            .sum();
        let mut plan = Self {
            field,
            num_clusters: clusters,
            segments: HashMap::new(),
        };
        let Some(first) = vectors.first() else {
            return Ok(plan);
        };
        let query = VectorQuery::new(first.options().metric(), Arc::new(query));
        let (total, _, x) = probe_params.resolved_work_budget(clusters, docs)?;
        let routers = vectors
            .iter()
            .map(|reader| reader.index().map(|index| index.opened_router()))
            .collect::<Vec<_>>();
        let partitions: Vec<_> = segments.iter().map(|s| s.partition_centroid).collect();
        let routing = RoutingParams {
            k: probe_params.router_k(total, x, match_fraction, clusters),
            rerank_factor: probe_params.router_rerank_factor,
            min_beam_fraction: probe_params.router_min_beam_fraction,
            aps: probe_params.router_aps,
        };
        let ranking = GlobalRanking::from(
            &routers,
            &partitions,
            &query,
            routing,
            &mut totals,
            &mut |i| on_segment(segments[i].ordinal),
        );
        for (i, (segment, totals)) in segments.iter().zip(totals).enumerate() {
            plan.segments.insert(
                segment.reader.segment_id(),
                SegmentRoutingPlan {
                    clusters: Vec::new(),
                    metrics: ranking.metrics[i],
                    totals,
                },
            );
        }
        for (i, candidate) in ranking.clusters.into_iter().take(routing.k) {
            plan.segments
                .get_mut(&segments[i].reader.segment_id())
                .unwrap()
                .clusters
                .push(candidate);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    #[test]
    fn routing_validates_query_schema() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 128).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segments = [VectorSegment::from((0, searcher.segment_reader(0)))];
        for (field, query, message) in [
            (fixture.label, vec![0.0f32; 32], "not a vector field"),
            (fixture.field, vec![0.0f32; 31], "query vector length"),
        ] {
            let error = VectorRoutingPlan::build(
                field,
                query,
                ProbeParams::default(),
                1.0,
                &segments,
                |_| {},
            )
            .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
        Ok(())
    }

    #[test]
    fn routing_validates_selectivity_and_partition_centroids() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 32, 8, 128).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segment = VectorSegment::from((0, searcher.segment_reader(0)));
        for match_fraction in [-0.1, 1.1, f64::NAN] {
            let error = VectorRoutingPlan::build(
                fixture.field,
                vec![0.0f32; 32],
                ProbeParams::default(),
                match_fraction,
                &[segment],
                |_| {},
            )
            .unwrap_err();
            assert!(error.to_string().contains("match fraction"), "{error}");
        }
        for centroid in [vec![0.0f32; 31], vec![f32::NAN; 32]] {
            let error = VectorRoutingPlan::build(
                fixture.field,
                vec![0.0f32; 32],
                ProbeParams::default(),
                1.0,
                &[VectorSegment {
                    partition_centroid: Some(&centroid),
                    ..segment
                }],
                |_| {},
            )
            .unwrap_err();
            assert!(error.to_string().contains("partition centroid"), "{error}");
        }
        Ok(())
    }
}

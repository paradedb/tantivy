//! The stacked router: a stacked IVF over a segment's centroids, ranked with fixed nprobe
//! fractions or the Quake APS.

pub(crate) mod aps;
pub(crate) mod levels;

use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::IvfCentroids;
use crate::vector::distance::Similarity;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::aps::{
    supports_metric as aps_supports_metric, CandidateRows, RecallEstimator, APS_MAX_DIM,
};
use crate::vector::router::stacked::levels::{
    ClusterId, InMemoryStore, LazyStore, StackedIvf, StackedIvfBuilder, StackedIvfConfig,
    StackedSearchStats, SuperKMeansLevelClusterer, MEMBER_SCAN_PATIENCE, PARENT_NPROBE_FRACTION,
};
use crate::vector::router::{RouterMetrics, RoutingParams};
use crate::TantivyError;

/// The router's [`StackedIvfConfig`]. The segment's own IVF (`.vec` rows under
/// `.centroids`) is L0 and is probed by the caller's work budget; every
/// level of this router sits above it and is a parent, so the bottom router
/// level uses [`PARENT_NPROBE_FRACTION`] too rather than the standalone
/// L0 default. Each level uses `nlist = ceil(√n)`; parents are added
/// while the top has at least 256 lists. The config is not persisted, so
/// open must agree with build only on what search reads.
fn router_config() -> StackedIvfConfig {
    StackedIvfConfig {
        nprobe_fraction: PARENT_NPROBE_FRACTION,
        ..StackedIvfConfig::default()
    }
}

pub(super) fn build(
    options: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<StackedIvf<InMemoryStore>> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let clusterer = SuperKMeansLevelClusterer::for_metric(options.metric());
    let (index, permutation) = StackedIvfBuilder::new(
        matrix.values.clone(),
        matrix.rows,
        options.dim(),
        &clusterer,
        router_config(),
    )
    .build();
    let IvfCentroids::F32(matrix) = centroids;
    if permutation.len() != matrix.rows {
        return Err(TantivyError::InvalidArgument(format!(
            "stacked router returned a permutation over {} centroids, expected {}",
            permutation.len(),
            matrix.rows
        )));
    }
    let mut values = vec![0.0f32; matrix.values.len()];
    let mut seen = vec![false; matrix.rows];
    for (old, &new) in permutation.iter().enumerate() {
        let new = new as usize;
        if new >= matrix.rows || seen[new] {
            return Err(TantivyError::InvalidArgument(
                "stacked router centroid permutation is not a bijection".to_string(),
            ));
        }
        seen[new] = true;
        values[new * matrix.dims..(new + 1) * matrix.dims]
            .copy_from_slice(&matrix.values[old * matrix.dims..(old + 1) * matrix.dims]);
    }
    matrix.values = values;
    Ok(index)
}

pub(super) fn open(
    payload: FileSlice,
    centroids: FileSlice,
    options: &VectorOptions,
) -> crate::Result<StackedIvf<LazyStore>> {
    Ok(StackedIvf::<LazyStore>::open(
        payload,
        centroids,
        options.dim(),
        router_config(),
    )?)
}

/// The recall target the bottom router level runs with: the caller's,
/// unless APS is off (`recall >= 1.0`), the metric has no query ball
/// (Dot), or the dimension is past [`APS_MAX_DIM`], where the cap-volume
/// estimate is unreliable. The fixed nprobe path is used instead.
pub(crate) fn effective_recall(dim: usize, metric: Metric, recall: f32) -> f32 {
    if dim > APS_MAX_DIM || !aps_supports_metric(metric) || !(recall < 1.0) {
        1.0
    } else {
        recall.max(0.0)
    }
}

pub(super) fn rank(
    index: &StackedIvf<LazyStore>,
    query: &[f32],
    metric: Metric,
    params: RoutingParams,
) -> Ranking {
    let recall = effective_recall(query.len(), metric, params.recall);
    // Always return at most `params.k` L0 centroids. That `k` is the
    // clusters the caller's probe budget buys under its filter
    // (`router_k` ← `max_probe` and filter selectivity).
    //
    // Parents still select lists with their fixed fraction. This level scores those lists'
    // members into a size-`k` heap and may stop before the shortlist is exhausted. Capping
    // the heap is what ties the returned set to the probe budget.
    let k = params.k.clamp(1, index.vectors.len().max(1));
    // Geometric APS sizes the scan when it is reliable. Above that, stop after a run of lists
    // that miss the size-`k` heap. The shortlist is still the hard cap. `member_budget` is not
    // applied: a fixed `c × k` did not hold recall across probe targets.
    let _ = params.member_budget_multiplier;
    let patience = (recall >= 1.0).then_some(MEMBER_SCAN_PATIENCE);
    let (ranked, stats) = index.search_within(query, k, recall, metric, None, patience);
    let candidate_count = ranked.len();
    Ranking {
        ranked: ranked.into_iter(),
        candidate_count,
        stats,
        recall_target: recall,
    }
}

/// The APS estimator for the segment's own cluster scan over `ranking`,
/// or `None` when APS is off for `recall` at this dimension. Call before
/// the first pull: the estimator covers the whole candidate set. The
/// bottom router level's members are the segment centroids, row for row,
/// so candidate rows come from its member store, fetched only as the
/// estimator needs them.
pub(super) fn recall_estimator<'a>(
    index: &'a StackedIvf<LazyStore>,
    ranking: &Ranking,
    query: &[f32],
    metric: Metric,
    recall: f32,
) -> Option<RecallEstimator<'a>> {
    let candidates = ranking.ranked.as_slice();
    if effective_recall(query.len(), metric, recall) >= 1.0 || candidates.is_empty() {
        return None;
    }
    let sims: Vec<Similarity> = candidates.iter().map(|candidate| candidate.sim).collect();
    let rows = MemberRows {
        members: &index.vectors,
        clusters: candidates
            .iter()
            .map(|candidate| candidate.node.0)
            .collect(),
    };
    Some(RecallEstimator::new(query, &sims, Box::new(rows), metric))
}

/// Ranked candidates' centroid rows, read from the bottom level's members.
struct MemberRows<'a> {
    members: &'a LazyStore,
    clusters: Vec<u32>,
}

impl CandidateRows for MemberRows<'_> {
    fn append_row(&self, rank: usize, out: &mut Vec<f32>) -> std::io::Result<()> {
        self.members.extend_with_row(self.clusters[rank], out)
    }
}

pub(crate) struct Ranking {
    ranked: std::vec::IntoIter<Candidate<ClusterId>>,
    candidate_count: usize,
    stats: StackedSearchStats,
    recall_target: f32,
}

impl Ranking {
    pub(super) fn metrics(&self) -> RouterMetrics {
        RouterMetrics::Stacked {
            candidate_count: self.candidate_count,
            lists_scanned: self.stats.lists_scanned,
            members_scored: self.stats.members_scored,
            recall_target: self.recall_target,
        }
    }
}

impl Iterator for Ranking {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.ranked.next().map(|candidate| Candidate {
            sim: candidate.sim,
            node: candidate.node.0,
        })
    }
}

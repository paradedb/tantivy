use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::ivf::{
    Candidate, ClusterId, InMemoryStackedIvf, IvfConfig, IvfIndexBuilder, LazyStackedIvf,
    StackedSearchStats, SuperKMeansLevelClusterer, APS_MAX_DIM, LEAF_EXPANSION_SLACK,
    PARENT_NPROBE_FRACTION,
};
use crate::vector::router::{RouterMetrics, RoutingParams};
use crate::vector::IvfCentroids;
use crate::TantivyError;

/// The router's [`IvfConfig`]. The segment's own IVF (`.vec` rows under
/// `.centroids`) is L0 and is probed by the caller's work budget; every
/// level of this router sits above it and is a parent, so the bottom router
/// level uses [`PARENT_NPROBE_FRACTION`] too rather than the standalone
/// L0 default. Build-only knobs (`branching_factor`, `max_leaf_size`) keep
/// their defaults; the config is not persisted, so open must agree with
/// build only on what search reads.
fn router_config() -> IvfConfig {
    IvfConfig {
        nprobe_fraction: PARENT_NPROBE_FRACTION,
        ..IvfConfig::default()
    }
}

pub(super) fn build(
    options: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<InMemoryStackedIvf> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let clusterer = SuperKMeansLevelClusterer::default();
    let (index, permutation) = IvfIndexBuilder::new(
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
) -> crate::Result<LazyStackedIvf> {
    Ok(LazyStackedIvf::open(
        payload,
        centroids,
        options.dim(),
        router_config(),
    )?)
}

/// The recall target the bottom router level runs with: the caller's,
/// unless APS is off (`recall >= 1.0`) or the dimension is past
/// [`APS_MAX_DIM`], where the cap-volume estimate is unreliable and the
/// fixed nprobe path is used instead.
pub(crate) fn effective_recall(dim: usize, recall: f32) -> f32 {
    if dim >= APS_MAX_DIM || !(recall < 1.0) {
        1.0
    } else {
        recall.max(0.0)
    }
}

pub(super) fn rank(
    index: &LazyStackedIvf,
    query: &[f32],
    metric: Metric,
    params: RoutingParams,
) -> Ranking {
    let recall = effective_recall(query.len(), params.recall);
    // Always return at most `params.k` L0 centroids. That `k` tracks the
    // caller's probe budget (`router_k` ← `max_probe`), so easy queries
    // request fewer candidates and harder ones more.
    //
    // Parents still use Quake-fat nprobe and expand every list they select.
    // Only this leaf early-stops L0 expansion after `LEAF_EXPANSION_SLACK × k`
    // members have been scored (nearest lists first; each opened list is
    // fully scanned) so max_probe can cut routing work without thinning
    // parent list ranking.
    let k = params.k.clamp(1, index.vectors.len().max(1));
    let expansion_budget = k.saturating_mul(LEAF_EXPANSION_SLACK);
    let (ranked, stats) =
        index.search_limited(query, k, recall, metric, Some(expansion_budget));
    let candidate_count = ranked.len();
    Ranking {
        ranked: ranked.into_iter(),
        candidate_count,
        stats,
        recall_target: recall,
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

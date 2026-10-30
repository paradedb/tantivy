//! The stacked router: a stacked IVF over a segment's centroids, ranked with
//! [`StackedIvf::lists_to_open`] and stopped by [`aps`].

pub(crate) mod aps;
pub(crate) mod levels;

use crate::directory::FileSlice;
use crate::schema::VectorOptions;
use crate::vector::clustering::IvfCentroids;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::levels::{
    ClusterId, InMemoryStore, LazyStore, StackedIvf, StackedIvfBuilder, StackedIvfConfig,
    StackedSearchStats, SuperKMeansLevelClusterer, PARENT_NPROBE_FRACTION,
};
use crate::vector::router::{RouterMetrics, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;
use crate::TantivyError;

/// The router's [`StackedIvfConfig`]. The segment's own IVF (`.vec` rows under
/// `.centroids`) is L0 and is probed by the caller's work budget; every
/// level of this router sits above it and is a parent, so the bottom router
/// level uses [`PARENT_NPROBE_FRACTION`] too rather than the standalone
/// L0 default. Search opens [`StackedIvf::lists_to_open`] lists (25% floor,
/// ~2x `k` member beam, 100% cap). Each level uses `nlist = ceil(sqrt(n))`;
/// parents are added while the top has at least 256 lists. The config is
/// not persisted, so open must agree with build only on what search reads.
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

pub(super) fn rank<T: VectorElement>(
    index: &StackedIvf<LazyStore>,
    query: &VectorQuery<T>,
    params: RoutingParams,
) -> Ranking {
    // Always return at most `params.k` L0 centroids. That `k` is the
    // clusters the caller's probe budget buys under its filter
    // (`router_k` <- `max_probe` and filter selectivity).
    //
    // The search still opens the parent nprobe lists and scores every
    // member into a size-`k` heap: scanning all members of the selected
    // lists avoids the classic IVF boundary miss; capping the heap (and
    // thus `candidate_count`) is what ties the returned set to probe.
    let k = params.k.clamp(1, index.vectors.len().max(1));
    let (ranked, stats) = super::search::search(index, query, k, k, params.aps);
    let ranked = ranked.into_sorted();
    let candidate_count = ranked.len();
    Ranking {
        ranked: ranked.into_iter(),
        candidate_count,
        stats,
    }
}

pub(crate) struct Ranking {
    ranked: std::vec::IntoIter<Candidate<ClusterId>>,
    candidate_count: usize,
    stats: StackedSearchStats,
}

impl Ranking {
    pub(super) fn metrics(&self) -> RouterMetrics {
        RouterMetrics::Stacked {
            candidate_count: self.candidate_count,
            lists_scanned: self.stats.lists_scanned,
            members_scored: self.stats.members_scored,
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

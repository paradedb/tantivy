//! Cluster routing: the router kinds an IVF segment persists, built at merge and opened per
//! segment to rank its clusters for a query.

use std::fmt;
use std::io::{self, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::error::Incompatibility;
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::IvfCentroids;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::aps::RecallEstimator;
use crate::vector::router::stacked::levels::{InMemoryStore, LazyStore, StackedIvf};
use crate::TantivyError;

pub(crate) mod arena;
pub(crate) mod candidate;
mod exact;
pub(crate) mod stacked;

/// The routing structure new IVF segments build. Codes are persisted; 0 belonged to the removed
/// graph router and is never reused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum RouterKind {
    /// A stacked IVF over the centroids.
    #[default]
    Stacked = 1,
    /// Scores every centroid against the query.
    Exact = 2,
}

impl RouterKind {
    /// The kind persisted as `code`. A code this library doesn't build is an incompatible index
    /// (rebuild required), not corruption.
    pub(crate) fn from_code(code: u8) -> crate::Result<Self> {
        match code {
            1 => Ok(Self::Stacked),
            2 => Ok(Self::Exact),
            kind => Err(TantivyError::IncompatibleIndex(
                Incompatibility::VectorRouterMismatch { kind },
            )),
        }
    }

    pub(crate) fn build(
        self,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<BuiltRouter> {
        match self {
            Self::Stacked => Ok(Router::Stacked(stacked::build(options, centroids)?)),
            Self::Exact => Ok(Router::Exact(exact::build(options, centroids))),
        }
    }

    /// Opens the router persisted in `slot`, whichever kind it was built
    /// with. The configured router only decides what new segments build, so
    /// an index may hold segments routed by different kinds.
    pub(crate) fn open(
        slot: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<OpenedRouter> {
        if slot.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "router slot is missing its kind byte",
            )
            .into());
        }
        let persisted = Self::from_code(slot.read_byte(0)?)?;
        let payload = slot.slice_from(1);
        match persisted {
            Self::Stacked => Ok(Router::Stacked(stacked::open(payload, centroids, options)?)),
            Self::Exact => Ok(Router::Exact(exact::open(payload, centroids, options)?)),
        }
    }
}

impl fmt::Display for RouterKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Stacked => "stacked",
            Self::Exact => "exact",
        })
    }
}

pub(crate) enum Router<S: crate::vector::router::arena::VectorArena<Elem = f32>> {
    Stacked(StackedIvf<S>),
    Exact(exact::ExactRouter<S>),
}

pub(crate) type BuiltRouter = Router<InMemoryStore>;
pub(crate) type OpenedRouter = Router<LazyStore>;

impl BuiltRouter {
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
        }
    }

    pub(crate) fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        (self.kind() as u8).serialize(out)?;
        match self {
            Self::Stacked(router) => router.serialize_router_payload(out),
            Self::Exact(router) => router.serialize_payload(out),
        }
    }
}

/// Per-query routing knobs. Only the stacked router reads them; the exact router ranks the same
/// way regardless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RoutingParams {
    /// Clusters the caller expects to probe. The stacked router ranks at
    /// least this many members when APS is on; with APS off it ranks every
    /// member of the lists its nprobe fraction selects.
    pub(crate) k: usize,
    /// Stacked-router recall target in `(0, 1]`. `1.0` disables APS and
    /// routes with the fixed nprobe fractions. APS is also disabled above
    /// [`APS_MAX_DIM`](crate::vector::router::stacked::aps::APS_MAX_DIM) regardless of this
    /// value.
    pub(crate) recall: f32,
}

impl Default for RoutingParams {
    fn default() -> Self {
        Self {
            k: usize::MAX,
            recall: 1.0,
        }
    }
}

pub(crate) enum RouterIter {
    #[cfg(test)]
    Explicit(std::vec::IntoIter<Candidate>),
    Stacked(stacked::Ranking),
    Exact(exact::Ranking),
}

impl RouterIter {
    pub(crate) fn for_each_candidate(&self, mut visit: impl FnMut(Candidate)) {
        match self {
            #[cfg(test)]
            Self::Explicit(ranking) => ranking.as_slice().iter().copied().for_each(visit),
            Self::Exact(ranking) => ranking.ranked.as_slice().iter().copied().for_each(visit),
            Self::Stacked(ranking) => {
                for candidate in ranking.ranked.as_slice() {
                    visit(Candidate {
                        sim: candidate.sim,
                        node: candidate.node.0,
                    });
                }
            }
        }
    }

    pub(crate) fn metrics(&self) -> RouterMetrics {
        match self {
            #[cfg(test)]
            Self::Explicit(_) => RouterMetrics::Exact { visited_count: 0 },
            Self::Stacked(ranking) => ranking.metrics(),
            Self::Exact(ranking) => RouterMetrics::Exact {
                visited_count: ranking.visited_count(),
            },
        }
    }
}

impl Iterator for RouterIter {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            #[cfg(test)]
            Self::Explicit(ranking) => ranking.next(),
            Self::Stacked(ranking) => ranking.next(),
            Self::Exact(ranking) => ranking.next(),
        }
    }
}

impl OpenedRouter {
    #[cfg(test)]
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
        }
    }

    pub(crate) fn rank(&self, query: &[f32], metric: Metric, params: RoutingParams) -> RouterIter {
        match self {
            Self::Stacked(router) => {
                RouterIter::Stacked(stacked::rank(router, query, metric, params))
            }
            Self::Exact(router) => RouterIter::Exact(router.rank(query)),
        }
    }

    /// The APS estimator for the segment's own cluster scan over
    /// `ranking`, which must not have been pulled yet. Only the stacked
    /// router supports APS; `None` otherwise, or when APS is off.
    pub(crate) fn recall_estimator(
        &self,
        ranking: &RouterIter,
        query: &[f32],
        metric: Metric,
        recall: f32,
    ) -> Option<RecallEstimator<'_>> {
        match (self, ranking) {
            (Self::Stacked(router), RouterIter::Stacked(ranking)) => {
                stacked::recall_estimator(router, ranking, query, metric, recall)
            }
            _ => None,
        }
    }
}

/// Router-specific statistics captured after a ranking iterator stops.
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouterMetrics {
    /// The stacked router's work.
    Stacked {
        /// Ranked centroids handed to the probe loop.
        candidate_count: usize,
        /// Router lists opened, summed over every router level.
        lists_scanned: usize,
        /// Similarity computations spent routing, summed over every level.
        members_scored: usize,
        /// Recall target the bottom router level actually used; `1.0`
        /// means the fixed nprobe path (requested, or forced by dimension).
        recall_target: f32,
    },
    /// The exact router's work.
    Exact {
        /// Centroids scored.
        visited_count: usize,
    },
}

#[cfg(test)]
thread_local! {
    static TEST_CLUSTERS: std::cell::RefCell<Option<Vec<Candidate>>> = const { std::cell::RefCell::new(None) };
}

/// Scans the supplied cluster similarities without invoking a router.
#[cfg(test)]
pub(crate) fn with_test_clusters<T>(clusters: Vec<Candidate>, scan: impl FnOnce() -> T) -> T {
    struct Restore(Option<Vec<Candidate>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_CLUSTERS.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(TEST_CLUSTERS.with(|slot| slot.replace(Some(clusters))));
    scan()
}

#[cfg(test)]
pub(crate) fn test_clusters() -> Option<RouterIter> {
    TEST_CLUSTERS.with(|slot| {
        slot.borrow()
            .clone()
            .map(|rows| RouterIter::Explicit(rows.into_iter()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::clustering::IvfMatrix;

    fn centroids() -> IvfCentroids {
        IvfCentroids::F32(IvfMatrix {
            values: vec![0.0, 1.0, 2.0],
            rows: 3,
            dims: 1,
        })
    }

    #[test]
    fn exact_router_opens_its_payload() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let mut centroids = centroids();
        let built = RouterKind::Exact.build(&options, &mut centroids)?;
        let mut bytes = Vec::new();
        built.serialize(&mut bytes)?;
        let rows = match centroids {
            IvfCentroids::F32(matrix) => matrix
                .values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        };
        let opened = RouterKind::open(FileSlice::from(bytes), FileSlice::from(rows), &options)?;
        let mut ranking = opened.rank(&[1.1], Metric::L2, RoutingParams::default());
        assert_eq!(ranking.next().unwrap().node, 1);
        assert!(matches!(
            ranking.metrics(),
            RouterMetrics::Exact { visited_count: 3 }
        ));
        Ok(())
    }

    /// Two well-separated blobs of `dim`-d centroids, `n_per` each.
    fn blob_centroids(dim: usize, n_per: usize) -> IvfCentroids {
        let mut values = Vec::with_capacity(2 * n_per * dim);
        for blob in 0..2 {
            for i in 0..n_per {
                for d in 0..dim {
                    let base = if blob == 0 { 0.0 } else { 100.0 };
                    values.push(base + if d == 0 { i as f32 * 0.05 } else { 0.0 });
                }
            }
        }
        IvfCentroids::F32(IvfMatrix {
            values,
            rows: 2 * n_per,
            dims: dim,
        })
    }

    fn open_stacked(dim: usize, n_per: usize) -> crate::Result<OpenedRouter> {
        let options = VectorOptions::new(dim, Metric::L2);
        let mut centroids = blob_centroids(dim, n_per);
        let built = RouterKind::Stacked.build(&options, &mut centroids)?;
        let mut bytes = Vec::new();
        built.serialize(&mut bytes)?;
        let rows = match centroids {
            IvfCentroids::F32(matrix) => matrix
                .values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        };
        RouterKind::open(FileSlice::from(bytes), FileSlice::from(rows), &options)
    }

    fn stacked_metrics(metrics: RouterMetrics) -> (usize, usize, usize, f32) {
        match metrics {
            RouterMetrics::Stacked {
                candidate_count,
                lists_scanned,
                members_scored,
                recall_target,
            } => (
                candidate_count,
                lists_scanned,
                members_scored,
                recall_target,
            ),
            other => panic!("expected stacked metrics, got {other:?}"),
        }
    }

    /// Up to `APS_MAX_DIM` the requested recall target is honoured and the
    /// ranking is bounded by `k`; the metrics report the router's work.
    #[test]
    fn stacked_ranking_uses_requested_recall_within_dim_cap() -> crate::Result<()> {
        let opened = open_stacked(2, 64)?;
        let query = vec![0.0f32; 2];
        let params = RoutingParams { k: 8, recall: 0.5 };
        let ranking = opened.rank(&query, Metric::L2, params);
        let (candidates, lists, scored, recall) = stacked_metrics(ranking.metrics());
        assert_eq!(recall, 0.5);
        assert!(candidates >= 1 && candidates <= 8, "{candidates}");
        assert!(lists >= 1);
        assert!(scored >= candidates);
        let json = serde_json::to_value(ranking.metrics()).unwrap();
        assert_eq!(json["kind"], "stacked");
        assert!(json["lists_scanned"].as_u64().unwrap() >= 1);
        Ok(())
    }

    /// Above `APS_MAX_DIM` the recall target is forced to `1.0` (fixed
    /// nprobe list selection), but the returned set is still capped at `k`
    /// so the caller's probe budget limits `candidate_count`.
    #[test]
    fn stacked_ranking_falls_back_to_nprobe_above_dim_cap() -> crate::Result<()> {
        let dim = crate::vector::router::stacked::aps::APS_MAX_DIM + 1;
        let opened = open_stacked(dim, 32)?;
        let query = vec![0.0f32; dim];
        let params = RoutingParams { k: 2, recall: 0.5 };
        let ranking = opened.rank(&query, Metric::L2, params);
        let (candidates, lists, scored, recall) = stacked_metrics(ranking.metrics());
        assert_eq!(recall, 1.0, "dimension cap must force the nprobe path");
        assert!(
            candidates <= 2,
            "returned set must honor params.k, got {candidates}"
        );
        assert!(lists >= 1, "nprobe path must still open parent lists");
        assert!(
            scored >= candidates,
            "all members of selected lists are scored into the k-heap ({scored} < {candidates})"
        );
        Ok(())
    }

    #[test]
    fn effective_recall_guards() {
        assert_eq!(stacked::effective_recall(2, Metric::L2, 0.9), 0.9);
        assert_eq!(stacked::effective_recall(2, Metric::Cosine, 0.9), 0.9);
        assert_eq!(stacked::effective_recall(2, Metric::Dot, 0.9), 1.0);
        assert_eq!(stacked::effective_recall(2, Metric::L2, 1.0), 1.0);
        assert_eq!(stacked::effective_recall(2, Metric::L2, 1.5), 1.0);
        assert_eq!(stacked::effective_recall(2, Metric::L2, f32::NAN), 1.0);
        assert_eq!(
            stacked::effective_recall(
                crate::vector::router::stacked::aps::APS_MAX_DIM,
                Metric::L2,
                0.9
            ),
            0.9
        );
        assert_eq!(
            stacked::effective_recall(
                crate::vector::router::stacked::aps::APS_MAX_DIM + 1,
                Metric::L2,
                0.9
            ),
            1.0
        );
    }

    /// Codes 1 and 2 open as the stacked and exact routers; any other code, the graph router's 0
    /// included, is an incompatible index.
    #[test]
    fn router_kind_codes() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let open = |slot: Vec<u8>, centroids: IvfCentroids| {
            let IvfCentroids::F32(matrix) = centroids;
            let rows: Vec<u8> = matrix
                .values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            RouterKind::open(FileSlice::from(slot), FileSlice::from(rows), &options)
        };
        for kind in [RouterKind::Stacked, RouterKind::Exact] {
            let mut centroids = centroids();
            let mut slot = Vec::new();
            kind.build(&options, &mut centroids)?.serialize(&mut slot)?;
            assert_eq!(slot[0], kind as u8);
            assert_eq!(open(slot, centroids)?.kind(), kind);
        }
        for code in [0, 255] {
            let Err(error) = open(vec![code], centroids()) else {
                panic!("router kind {code} must not open");
            };
            assert!(
                matches!(
                    error,
                    TantivyError::IncompatibleIndex(Incompatibility::VectorRouterMismatch { kind })
                        if kind == code
                ),
                "{code}: {error:?}"
            );
        }
        Ok(())
    }
}

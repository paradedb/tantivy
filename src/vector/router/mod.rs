use std::fmt;
use std::io::{self, Write};

use common::{BinarySerializable, HasLen};

use super::ivf::graph::{
    Candidate, NeighborhoodGraphSearchMetrics, RelativeNeighborhoodGraph, ResumableSearchIterator,
    Workspace,
};
use super::ivf::{InMemoryStore, IvfCentroids, LazyStore, MultiLevelIvf};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::header::VectorFileVersion;

mod exact;
mod rng;
mod stacked;

/// The routing structure used for every IVF segment in an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RouterKind {
    Rng = 0,
    Stacked = 1,
    Exact = 2,
}

impl RouterKind {
    fn from_code(code: u8) -> io::Result<Self> {
        match code {
            0 => Ok(Self::Rng),
            1 => Ok(Self::Stacked),
            2 => Ok(Self::Exact),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown router kind: {other}"),
            )),
        }
    }

    pub(crate) fn build(
        self,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<BuiltRouter> {
        match self {
            Self::Rng => Ok(Router::Rng(rng::build(options, centroids)?)),
            Self::Stacked => Ok(Router::Stacked(stacked::build(options, centroids)?)),
            Self::Exact => Ok(Router::Exact(exact::build(options, centroids))),
        }
    }

    pub(crate) fn open(
        self,
        file_version: VectorFileVersion,
        slot: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<OpenedRouter> {
        if file_version != VectorFileVersion::V3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("router {self} requires vector file version V3, found {file_version:?}"),
            )
            .into());
        }
        if slot.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "router slot is missing its kind byte",
            )
            .into());
        }
        let persisted = Self::from_code(slot.read_byte(0)?)?;
        if persisted != self {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("configured router {self} does not match persisted router {persisted}"),
            )
            .into());
        }
        let payload = slot.slice_from(1);
        match self {
            Self::Rng => Ok(Router::Rng(rng::open(payload, centroids, options)?)),
            Self::Stacked => Ok(Router::Stacked(stacked::open(payload, centroids, options)?)),
            Self::Exact => Ok(Router::Exact(exact::open(payload, centroids, options)?)),
        }
    }
}

impl fmt::Display for RouterKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rng => "rng",
            Self::Stacked => "stacked",
            Self::Exact => "exact",
        })
    }
}

pub(crate) enum Router<S: super::VectorArena<Elem = f32>> {
    Rng(RelativeNeighborhoodGraph<S>),
    Stacked(MultiLevelIvf<S, S>),
    Exact(exact::ExactRouter<S>),
}

pub(crate) type BuiltRouter = Router<InMemoryStore>;
pub(crate) type OpenedRouter = Router<LazyStore>;

impl BuiltRouter {
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Rng(_) => RouterKind::Rng,
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
        }
    }

    pub(crate) fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        (self.kind() as u8).serialize(out)?;
        match self {
            Self::Rng(router) => router.serialize(out),
            Self::Stacked(router) => router.serialize_router_payload(out),
            Self::Exact(router) => router.serialize_payload(out),
        }
    }
}

#[derive(Default)]
pub(crate) struct RouterWorkspace {
    rng: Workspace,
}

/// Per-query routing knobs. Only the stacked router reads them; the RNG
/// and exact routers rank the same way regardless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutingParams {
    /// Clusters the caller expects to probe. The stacked router ranks at
    /// least this many members when APS is on; with APS off it ranks every
    /// member of the lists its nprobe fraction selects.
    pub k: usize,
    /// Stacked-router recall target in `(0, 1]`. `1.0` disables APS and
    /// routes with the fixed nprobe fractions. APS is also disabled above
    /// [`APS_MAX_DIM`](crate::vector::ivf::APS_MAX_DIM) regardless of this
    /// value.
    pub recall: f32,
}

impl Default for RoutingParams {
    fn default() -> Self {
        Self {
            k: usize::MAX,
            recall: 1.0,
        }
    }
}

pub(crate) enum RouterIter<'router, 'workspace> {
    Rng(ResumableSearchIterator<'router, 'workspace, LazyStore>),
    Stacked(stacked::Ranking),
    Exact(exact::Ranking),
}

impl RouterIter<'_, '_> {
    pub(crate) fn metrics(&self) -> RouterMetrics {
        match self {
            Self::Rng(ranking) => RouterMetrics::Rng(ranking.metrics()),
            Self::Stacked(ranking) => ranking.metrics(),
            Self::Exact(ranking) => RouterMetrics::Exact {
                visited_count: ranking.visited_count(),
            },
        }
    }
}

impl Iterator for RouterIter<'_, '_> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Rng(ranking) => ranking.next(),
            Self::Stacked(ranking) => ranking.next(),
            Self::Exact(ranking) => ranking.next(),
        }
    }
}

impl OpenedRouter {
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Rng(_) => RouterKind::Rng,
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
        }
    }

    pub(crate) fn rank<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
        metric: Metric,
        params: RoutingParams,
    ) -> RouterIter<'router, 'workspace> {
        match self {
            Self::Rng(router) => RouterIter::Rng(rng::rank(router, &mut workspace.rng, query)),
            Self::Stacked(router) => {
                RouterIter::Stacked(stacked::rank(router, query, metric, params))
            }
            Self::Exact(router) => RouterIter::Exact(router.rank(query)),
        }
    }
}

/// Router-specific statistics captured after a ranking iterator stops.
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouterMetrics {
    Rng(NeighborhoodGraphSearchMetrics),
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
    Exact {
        visited_count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::IvfMatrix;

    fn centroids() -> IvfCentroids {
        IvfCentroids::F32(IvfMatrix {
            values: vec![0.0, 1.0, 2.0],
            rows: 3,
            dims: 1,
        })
    }

    #[test]
    fn configured_router_opens_matching_payload() -> crate::Result<()> {
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
        let opened = RouterKind::Exact.open(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            FileSlice::from(rows),
            &options,
        )?;
        let mut workspace = RouterWorkspace::default();
        let mut ranking = opened.rank(&mut workspace, &[1.1], Metric::L2, RoutingParams::default());
        assert_eq!(ranking.next().unwrap().node, 1);
        assert!(matches!(
            ranking.metrics(),
            RouterMetrics::Exact { visited_count: 3 }
        ));
        Ok(())
    }

    #[test]
    fn rng_ranking_reuses_workspace_and_reports_graph_metrics() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let mut centroids = centroids();
        let built = RouterKind::Rng.build(&options, &mut centroids)?;
        let mut bytes = Vec::new();
        built.serialize(&mut bytes)?;
        assert_eq!(bytes[0], RouterKind::Rng as u8);
        let rows = match centroids {
            IvfCentroids::F32(matrix) => matrix
                .values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        };
        let opened = RouterKind::Rng.open(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            FileSlice::from(rows),
            &options,
        )?;
        let mut workspace = RouterWorkspace::default();
        for query in [[0.1], [1.9]] {
            let mut ranking =
                opened.rank(&mut workspace, &query, Metric::L2, RoutingParams::default());
            assert!(ranking.next().is_some());
            let metrics = ranking.metrics();
            match metrics {
                RouterMetrics::Rng(metrics) => {
                    assert!(metrics.visited_count > 0);
                    assert_eq!(metrics.result_count, 1);
                }
                metrics => panic!("expected RNG metrics, got {metrics:?}"),
            }
            let json = serde_json::to_value(metrics).unwrap();
            assert_eq!(json["kind"], "rng");
            assert!(json["visited_count"].as_u64().unwrap() > 0);
        }
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
        RouterKind::Stacked.open(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            FileSlice::from(rows),
            &options,
        )
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

    /// Below `APS_MAX_DIM` the requested recall target is honoured and the
    /// ranking is bounded by `k`; the metrics report the router's work.
    #[test]
    fn stacked_ranking_uses_requested_recall_below_dim_cap() -> crate::Result<()> {
        let opened = open_stacked(2, 64)?;
        let mut workspace = RouterWorkspace::default();
        let query = vec![0.0f32; 2];
        let params = RoutingParams { k: 8, recall: 0.5 };
        let ranking = opened.rank(&mut workspace, &query, Metric::L2, params);
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

    /// At or above `APS_MAX_DIM` the recall target is forced to `1.0` and
    /// the router ranks every member of its selected lists, not just `k`.
    #[test]
    fn stacked_ranking_falls_back_to_nprobe_at_dim_cap() -> crate::Result<()> {
        let dim = crate::vector::ivf::APS_MAX_DIM;
        let opened = open_stacked(dim, 32)?;
        let mut workspace = RouterWorkspace::default();
        let query = vec![0.0f32; dim];
        let params = RoutingParams { k: 2, recall: 0.5 };
        let ranking = opened.rank(&mut workspace, &query, Metric::L2, params);
        let (candidates, _, _, recall) = stacked_metrics(ranking.metrics());
        assert_eq!(recall, 1.0, "dimension cap must force the nprobe path");
        assert!(
            candidates > 2,
            "nprobe path ranks every member of the selected lists, got {candidates}"
        );
        Ok(())
    }

    #[test]
    fn effective_recall_guards() {
        assert_eq!(stacked::effective_recall(2, 0.9), 0.9);
        assert_eq!(stacked::effective_recall(2, 1.0), 1.0);
        assert_eq!(stacked::effective_recall(2, 1.5), 1.0);
        assert_eq!(stacked::effective_recall(2, f32::NAN), 1.0);
        assert_eq!(
            stacked::effective_recall(crate::vector::ivf::APS_MAX_DIM, 0.9),
            1.0
        );
        assert_eq!(
            stacked::effective_recall(crate::vector::ivf::APS_MAX_DIM - 1, 0.9),
            0.9
        );
    }

    #[test]
    fn configured_router_rejects_a_different_persisted_router() {
        let options = VectorOptions::new(1, Metric::L2);
        let error = RouterKind::Stacked
            .open(
                VectorFileVersion::V3,
                FileSlice::from(vec![RouterKind::Exact as u8]),
                FileSlice::empty(),
                &options,
            )
            .err()
            .expect("a different persisted router must fail");
        assert!(error
            .to_string()
            .contains("configured router stacked does not match persisted router exact"));
    }

    #[test]
    fn pre_v3_router_format_is_rejected() {
        let options = VectorOptions::new(1, Metric::L2);
        let error = RouterKind::Exact
            .open(
                VectorFileVersion::V2,
                FileSlice::empty(),
                FileSlice::empty(),
                &options,
            )
            .err()
            .expect("pre-V3 router formats must fail");
        assert!(error
            .to_string()
            .contains("requires vector file version V3"));
    }

    #[test]
    fn unknown_router_kind_is_rejected() {
        let options = VectorOptions::new(1, Metric::L2);
        let error = RouterKind::Exact
            .open(
                VectorFileVersion::V3,
                FileSlice::from(vec![u8::MAX]),
                FileSlice::empty(),
                &options,
            )
            .err()
            .expect("unknown router kinds must fail");
        assert!(error.to_string().contains("unknown router kind: 255"));
    }
}

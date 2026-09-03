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
    ) -> crate::Result<InMemoryRouter> {
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
    ) -> crate::Result<LazyRouter> {
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

pub(crate) type InMemoryRouter = Router<InMemoryStore>;
pub(crate) type LazyRouter = Router<LazyStore>;

impl InMemoryRouter {
    pub(crate) fn from(
        kind: RouterKind,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Self> {
        let IvfCentroids::F32(matrix) = &*centroids;
        let shape = (matrix.rows, matrix.dims, matrix.values.len());
        let router = kind.build(options, centroids)?;
        let IvfCentroids::F32(matrix) = &*centroids;
        if (matrix.rows, matrix.dims, matrix.values.len()) != shape {
            return Err(crate::TantivyError::InvalidArgument(
                "Router changed the centroid matrix shape while building".to_string(),
            ));
        }
        Ok(router)
    }

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

pub(crate) enum RouterIter<'router, 'workspace> {
    Rng(ResumableSearchIterator<'router, 'workspace, LazyStore>),
    Stacked(stacked::Ranking),
    Exact(exact::Ranking),
}

impl RouterIter<'_, '_> {
    pub(crate) fn metrics(&self) -> RouterMetrics {
        match self {
            Self::Rng(ranking) => RouterMetrics::Rng(ranking.metrics()),
            Self::Stacked(ranking) => RouterMetrics::Stacked {
                candidate_count: ranking.candidate_count(),
            },
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

impl LazyRouter {
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
    ) -> RouterIter<'router, 'workspace> {
        match self {
            Self::Rng(router) => RouterIter::Rng(rng::rank(router, &mut workspace.rng, query)),
            Self::Stacked(router) => RouterIter::Stacked(stacked::rank(router, query, metric)),
            Self::Exact(router) => RouterIter::Exact(router.rank(query)),
        }
    }
}

/// Router-specific statistics captured after a ranking iterator stops.
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouterMetrics {
    Rng(NeighborhoodGraphSearchMetrics),
    Stacked { candidate_count: usize },
    Exact { visited_count: usize },
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
        let mut ranking = opened.rank(&mut workspace, &[1.1], Metric::L2);
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
            let mut ranking = opened.rank(&mut workspace, &query, Metric::L2);
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

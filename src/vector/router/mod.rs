//! Cluster routing: the router kinds an IVF segment persists, built at merge and opened per
//! segment to rank its clusters for a query.

use std::fmt;
use std::io::{self, Write};

use common::{BinarySerializable, HasLen};

use crate::directory::error::Incompatibility;
use crate::directory::FileSlice;
use crate::schema::VectorOptions;
use crate::vector::clustering::IvfCentroids;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::levels::{InMemoryStore, LazyStore, StackedIvf};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;
use crate::TantivyError;

pub(crate) mod arena;
pub(crate) mod candidate;
mod exact;
mod quantized;
mod search;
mod sign;
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
    /// Opt-in one-bit stacked routing with a padded exact centroid rerank.
    /// Existing full-precision stacked routing remains the default.
    QuantizedStacked = 3,
}

impl RouterKind {
    /// The kind persisted as `code`. A code this library doesn't build is an incompatible index
    /// (rebuild required), not corruption.
    pub(crate) fn from_code(code: u8) -> crate::Result<Self> {
        match code {
            1 => Ok(Self::Stacked),
            2 => Ok(Self::Exact),
            3 => Ok(Self::QuantizedStacked),
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
            Self::QuantizedStacked => Ok(Router::QuantizedStacked(
                quantized::QuantizedRouter::build(options, centroids)?,
            )),
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
            Self::QuantizedStacked => Ok(Router::QuantizedStacked(
                quantized::QuantizedRouter::open(payload, centroids, options)?,
            )),
        }
    }
}

impl fmt::Display for RouterKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Stacked => "stacked",
            Self::Exact => "exact",
            Self::QuantizedStacked => "quantized_stacked",
        })
    }
}

pub(crate) enum Router<S: crate::vector::router::arena::VectorArena<Elem = f32>> {
    Stacked(StackedIvf<S>),
    Exact(exact::ExactRouter<S>),
    QuantizedStacked(quantized::QuantizedRouter<S>),
}

pub(crate) type BuiltRouter = Router<InMemoryStore>;
pub(crate) type OpenedRouter = Router<LazyStore>;

impl BuiltRouter {
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
            Self::QuantizedStacked(_) => RouterKind::QuantizedStacked,
        }
    }

    pub(crate) fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        (self.kind() as u8).serialize(out)?;
        match self {
            Self::Stacked(router) => router.serialize_router_payload(out),
            Self::Exact(router) => router.serialize_payload(out),
            Self::QuantizedStacked(router) => router.serialize(out),
        }
    }
}

/// Per-query routing knobs. Only the stacked router reads them; the exact router ranks the same
/// way regardless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RoutingParams {
    /// Clusters the caller expects to probe. The stacked router opens
    /// enough lists to feed a size-`k` member heap via
    /// [`StackedIvf::lists_to_open`](crate::vector::router::stacked::levels::StackedIvf::lists_to_open).
    pub(crate) k: usize,
    /// Quantized member retention only; never changes the list request.
    pub(crate) rerank_factor: usize,
    /// List-selection policy, independent of retention; quantized routing only.
    pub(crate) min_beam_fraction: f32,
    pub(crate) flat: bool,
    pub(crate) aps: bool,
}

impl Default for RoutingParams {
    fn default() -> Self {
        Self {
            k: usize::MAX,
            rerank_factor: 2,
            min_beam_fraction: 0.5,
            flat: false,
            aps: true,
        }
    }
}

pub(crate) enum RouterIter {
    #[cfg(test)]
    Explicit(std::vec::IntoIter<Candidate>),
    Stacked(stacked::Ranking),
    Exact(exact::Ranking),
    QuantizedStacked(quantized::Ranking),
}

impl RouterIter {
    pub(crate) fn metrics(&self) -> RouterMetrics {
        match self {
            #[cfg(test)]
            Self::Explicit(_) => RouterMetrics::Exact { visited_count: 0 },
            Self::Stacked(ranking) => ranking.metrics(),
            Self::QuantizedStacked(ranking) => ranking.metrics(),
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
            Self::QuantizedStacked(ranking) => ranking.next(),
        }
    }
}

impl OpenedRouter {
    pub(crate) fn kind(&self) -> RouterKind {
        match self {
            Self::Stacked(_) => RouterKind::Stacked,
            Self::Exact(_) => RouterKind::Exact,
            Self::QuantizedStacked(_) => RouterKind::QuantizedStacked,
        }
    }

    /// Centroids at each router level, bottom (the level whose members are the segment's
    /// centroids) to top. The exact router has no levels.
    pub(crate) fn level_sizes(&self) -> Vec<usize> {
        match self {
            Self::Stacked(router) => {
                let mut sizes = Vec::new();
                let mut level = Some(router);
                while let Some(current) = level {
                    sizes.push(current.nlist());
                    level = current.parent.as_deref();
                }
                sizes
            }
            Self::Exact(_) => Vec::new(),
            Self::QuantizedStacked(router) => router.level_sizes(),
        }
    }

    pub(crate) fn rank<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        params: RoutingParams,
    ) -> RouterIter {
        match self {
            Self::Stacked(router) => RouterIter::Stacked(stacked::rank(router, query, params)),
            Self::Exact(router) => RouterIter::Exact(router.rank(query)),
            Self::QuantizedStacked(router) => {
                RouterIter::QuantizedStacked(router.rank(query, params))
            }
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
    },
    /// Scalar-estimate routing and its separate exact output stage.
    QuantizedStacked {
        /// Whether this invocation bypassed the persisted hierarchy.
        flat: bool,
        /// Exact-ranked centroids handed to the document probe loop.
        candidate_count: usize,
        /// Lists opened across all router levels.
        lists_scanned: usize,
        /// One-bit scores across all levels, including root centroid ranking.
        approximate_scored: usize,
        /// One-bit member scores at the bottom router level only. Members
        /// retained from these scores are passed to the exact rerank.
        approximate_members_scored: usize,
        /// Full-precision scores during final reranking (or exact fallback).
        exact_scored: usize,
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
    use std::sync::Arc;

    use super::*;
    use crate::schema::Metric;
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
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![1.1]));
        let mut ranking = opened.rank(&query, RoutingParams::default());
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

    fn stacked_metrics(metrics: RouterMetrics) -> (usize, usize, usize) {
        match metrics {
            RouterMetrics::Stacked {
                candidate_count,
                lists_scanned,
                members_scored,
            } => (candidate_count, lists_scanned, members_scored),
            other => panic!("expected stacked metrics, got {other:?}"),
        }
    }

    /// Ranking is capped at `k`; metrics report the router's list work.
    #[test]
    fn stacked_ranking_honors_k() -> crate::Result<()> {
        let opened = open_stacked(2, 64)?;
        let query = vec![0.0f32; 2];
        let params = RoutingParams {
            k: 8,
            ..Default::default()
        };
        let query = VectorQuery::new(Metric::L2, Arc::new(query));
        let ranking = opened.rank(&query, params);
        let (candidates, lists, scored) = stacked_metrics(ranking.metrics());
        assert!(candidates >= 1 && candidates <= 8, "{candidates}");
        assert!(lists >= 1);
        assert!(scored >= candidates);
        let json = serde_json::to_value(ranking.metrics()).unwrap();
        assert_eq!(json["kind"], "stacked");
        assert!(json["lists_scanned"].as_u64().unwrap() >= 1);
        Ok(())
    }

    /// Codes 1, 2, and 3 open full-precision stacked, exact, and quantized stacked
    /// routers. Unknown codes, including the removed graph router's 0, are incompatible.
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
        for kind in [
            RouterKind::Stacked,
            RouterKind::Exact,
            RouterKind::QuantizedStacked,
        ] {
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

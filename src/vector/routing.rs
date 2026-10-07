use std::sync::Mutex;

use super::ivf::{Candidate, IvfIndex, RecallEstimator, RouterIndex};
use super::router::{RouterIter, RouterMetrics, RouterWorkspace, RoutingParams};

pub(crate) trait RankedClusters: Iterator<Item = Candidate> {
    fn metrics(&self) -> Option<RouterMetrics>;
}

impl RankedClusters for RouterIter<'_, '_> {
    fn metrics(&self) -> Option<RouterMetrics> {
        Some(RouterIter::metrics(self))
    }
}

pub(crate) struct ClusterRouting<'a> {
    pub ranked: Box<dyn RankedClusters + 'a>,
    pub estimator: Option<RecallEstimator<'a>>,
}

impl<'a> ClusterRouting<'a> {
    pub(crate) fn new(
        shared: Option<&'a SharedRouting<'_>>,
        index: &'a IvfIndex,
        workspace: &'a mut RouterWorkspace,
        query: &'a [f32],
        params: RoutingParams,
        recall: f32,
    ) -> Self {
        if let Some(shared) = shared {
            return shared.replay(recall);
        }
        let ranked = index.rank_clusters(workspace, query, params);
        let estimator = index.recall_estimator(&ranked, query, recall);
        Self {
            ranked: Box::new(ranked),
            estimator,
        }
    }
}

pub(crate) struct SharedRouting<'a> {
    router: &'a RouterIndex,
    pub query: &'a [f32],
    state: Mutex<RoutingState<'a>>,
}

struct RoutingState<'a> {
    ranked: RouterIter<'a, 'a>,
    cached: Vec<Candidate>,
}

impl<'a> SharedRouting<'a> {
    pub(crate) fn new(
        router: &'a RouterIndex,
        workspace: &'a mut RouterWorkspace,
        query: &'a [f32],
        params: RoutingParams,
    ) -> Self {
        Self {
            router,
            query,
            state: Mutex::new(RoutingState {
                ranked: router.rank_clusters(workspace, query, params),
                cached: Vec::new(),
            }),
        }
    }

    pub(crate) fn replay(&self, recall: f32) -> ClusterRouting<'_> {
        let estimator =
            self.router
                .recall_estimator(&self.state.lock().unwrap().ranked, self.query, recall);
        ClusterRouting {
            ranked: Box::new(RoutingCursor {
                shared: self,
                next: 0,
            }),
            estimator,
        }
    }

    pub(crate) fn metrics(&self) -> RouterMetrics {
        self.state.lock().unwrap().ranked.metrics()
    }
}

struct RoutingCursor<'a, 'router> {
    shared: &'a SharedRouting<'router>,
    next: usize,
}

impl Iterator for RoutingCursor<'_, '_> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        let mut state = self.shared.state.lock().unwrap();
        if self.next == state.cached.len() {
            let candidate = state.ranked.next()?;
            state.cached.push(candidate);
        }
        let candidate = state.cached[self.next];
        self.next += 1;
        Some(candidate)
    }
}

impl RankedClusters for RoutingCursor<'_, '_> {
    fn metrics(&self) -> Option<RouterMetrics> {
        None
    }
}

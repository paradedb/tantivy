use std::io::{self, Write};

use super::{IvfSearchMetrics, Router, RouterDescriptor};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::header::VectorFileVersion;
use crate::vector::ivf::graph::{
    NeighborhoodGraphConfig, RelativeNeighborhoodGraph, ResumableSearchIterator, Workspace,
};
use crate::vector::{Candidate, FileSliceArena, IvfCentroids, VectorArena};
use crate::Executor;

const GRAPH_ROUTER_ID: &str = "tantivy.relative-neighborhood-graph";

struct RngRanking<'g, 'w, S: VectorArena> {
    inner: ResumableSearchIterator<'g, 'w, S>,
    metrics: &'w mut IvfSearchMetrics,
}

impl<S: VectorArena> Iterator for RngRanking<'_, '_, S> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        let candidate = self.inner.next();
        let graph = self.inner.metrics();
        *self.metrics = IvfSearchMetrics {
            visited_count: graph.visited_count,
            graph: Some(graph),
        };
        candidate
    }
}

impl<S> Router for RelativeNeighborhoodGraph<S>
where S: VectorArena<Elem = f32> + Send + Sync + 'static
{
    fn router_descriptor() -> RouterDescriptor {
        RouterDescriptor::new(GRAPH_ROUTER_ID, VectorFileVersion::V3)
    }

    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        let IvfCentroids::F32(matrix) = centroids;
        let config = NeighborhoodGraphConfig::default();
        let mut graph = RelativeNeighborhoodGraph::new(
            matrix.values.as_slice(),
            options.dim(),
            options.metric(),
            config.clone(),
        );
        let num_threads = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        let executor = if num_threads > 1 {
            Executor::multi_thread(num_threads, "rng-build-")?
        } else {
            Executor::single_thread()
        };
        graph.build(&executor);

        let mut adjacency = Vec::new();
        RelativeNeighborhoodGraph::serialize(&graph, &mut adjacency)?;
        Ok(Box::new(RelativeNeighborhoodGraph::open(
            &adjacency,
            matrix.values.clone(),
            options.dim(),
            options.metric(),
            config,
        )?))
    }

    fn deserialize(
        payload: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Box<dyn Router>> {
        let vectors = match options.dtype() {
            VectorDType::F32 => FileSliceArena::<f32>::new(centroids),
        };
        let adjacency = payload.read_bytes()?;
        Ok(Box::new(RelativeNeighborhoodGraph::open(
            &adjacency,
            vectors,
            options.dim(),
            options.metric(),
            NeighborhoodGraphConfig::default(),
        )?))
    }

    fn rank<'a>(
        &'a self,
        workspace: &'a mut Workspace,
        query: &'a [f32],
        _metric: Metric,
        metrics: &'a mut IvfSearchMetrics,
    ) -> Box<dyn Iterator<Item = Candidate> + 'a> {
        let seeds = (0..self.len())
            .step_by((self.len() / 8).max(1))
            .take(8)
            .map(|node| node as u32)
            .collect::<Vec<_>>();
        let inner = self.search_iter(workspace, query, &seeds);
        let graph = inner.metrics();
        *metrics = IvfSearchMetrics {
            visited_count: graph.visited_count,
            graph: Some(graph),
        };
        Box::new(RngRanking { inner, metrics })
    }

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
        RelativeNeighborhoodGraph::serialize(self, out)
    }
}

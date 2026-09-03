use crate::directory::FileSlice;
use crate::schema::{VectorDType, VectorOptions};
use crate::vector::ivf::{
    InMemoryStore, LazyStore, NeighborhoodGraphConfig, RelativeNeighborhoodGraph,
    ResumableSearchIterator, Workspace,
};
use crate::vector::IvfCentroids;
use crate::Executor;

pub(super) fn build(
    options: &VectorOptions,
    centroids: &IvfCentroids,
) -> crate::Result<RelativeNeighborhoodGraph<InMemoryStore>> {
    let IvfCentroids::F32(matrix) = centroids;
    let config = NeighborhoodGraphConfig::default();
    let mut graph = RelativeNeighborhoodGraph::new(
        matrix.values.as_slice(),
        options.dim(),
        options.metric(),
        config,
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
    graph.serialize(&mut adjacency)?;
    Ok(RelativeNeighborhoodGraph::open(
        &adjacency,
        InMemoryStore::new(matrix.values.clone(), options.dim()),
        options.dim(),
        options.metric(),
        config,
    )?)
}

pub(super) fn open(
    payload: FileSlice,
    centroids: FileSlice,
    options: &VectorOptions,
) -> crate::Result<RelativeNeighborhoodGraph<LazyStore>> {
    let vectors = match options.dtype() {
        VectorDType::F32 => LazyStore::new(centroids, options.dim()),
    };
    let adjacency = payload.read_bytes()?;
    Ok(RelativeNeighborhoodGraph::open(
        &adjacency,
        vectors,
        options.dim(),
        options.metric(),
        NeighborhoodGraphConfig::default(),
    )?)
}

pub(super) fn rank<'router, 'workspace>(
    router: &'router RelativeNeighborhoodGraph<LazyStore>,
    workspace: &'workspace mut Workspace,
    query: &'router [f32],
) -> ResumableSearchIterator<'router, 'workspace, LazyStore> {
    let seeds = (0..router.len())
        .step_by((router.len() / 8).max(1))
        .take(8)
        .map(|node| node as u32)
        .collect::<Vec<_>>();
    router.search_iter(workspace, query, &seeds)
}

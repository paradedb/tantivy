//! Fixed-degree nearest-neighbor graph construction and search.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::io::{self, Write};
use std::ops::Deref;

use common::{BinarySerializable, BitSet};

use super::partition;
use crate::schema::Metric;
use crate::vector::{Similarity, VectorArena, VectorElement};
use crate::Executor;

/// Identifies a graph node.
pub type NodeId = u32;

/// Marks an unused neighbor slot.
pub const EMPTY: NodeId = NodeId::MAX;

/// A fixed-degree nearest-neighbor graph over a vector arena.
pub struct Graph<S> {
    /// Maximum out-degree per node.
    max_edges: usize,
    /// Vector dimensionality.
    dim: usize,
    /// Vector storage.
    vectors: S,
    /// Best-first, sentinel-padded adjacency runs.
    neighbors: Vec<NodeId>,
    /// Build-time edge similarities.
    sims: Vec<Similarity>,
}

impl<S: VectorArena> Graph<S> {
    /// Creates an empty build graph over a vector arena.
    ///
    /// # Panics
    ///
    /// Panics when the dimensions, degree, or node count are invalid.
    pub fn new(vectors: S, dim: usize, max_edges: usize) -> Self {
        let n = Self::node_count(&vectors, dim, max_edges);
        Graph {
            max_edges,
            dim,
            vectors,
            neighbors: vec![EMPTY; n * max_edges],
            sims: vec![Similarity::WORST; n * max_edges],
        }
    }

    /// Creates an empty search-only graph for deserialization.
    ///
    /// # Panics
    ///
    /// Panics when the dimensions, degree, or node count are invalid.
    pub fn for_reload(vectors: S, dim: usize, max_edges: usize) -> Self {
        let n = Self::node_count(&vectors, dim, max_edges);
        Graph {
            max_edges,
            dim,
            vectors,
            neighbors: vec![EMPTY; n * max_edges],
            sims: Vec::new(),
        }
    }

    /// Opens a serialized graph over a vector arena.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid adjacency payload.
    pub fn open(adjacency: &[u8], vectors: S, dim: usize) -> io::Result<Graph<S>> {
        let mut cursor = adjacency;
        let max_edges = u32::deserialize(&mut cursor)? as usize;
        if max_edges == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "serialized graph has zero max_edges",
            ));
        }
        let n = Self::node_count(&vectors, dim, max_edges);
        let expected = n * max_edges * std::mem::size_of::<NodeId>();
        if cursor.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "serialized graph adjacency is {} bytes, expected {expected} for {n} nodes",
                    cursor.len()
                ),
            ));
        }
        let neighbors: Vec<NodeId> = cursor
            .chunks_exact(std::mem::size_of::<NodeId>())
            .map(|chunk| NodeId::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Graph {
            max_edges,
            dim,
            vectors,
            neighbors,
            sims: Vec::new(),
        })
    }

    /// Validates the constructor arguments and derives the node count.
    fn node_count(vectors: &S, dim: usize, max_edges: usize) -> usize {
        assert!(max_edges > 0, "max_edges must be non-zero");
        assert!(dim > 0, "dim must be non-zero");
        let n = vectors.num_vectors(dim);
        assert!(n < NodeId::MAX as usize, "arena exceeds NodeId space");
        n
    }

    /// Inserts a directed edge into a bounded best-first adjacency.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, sim: Similarity) {
        debug_assert_eq!(
            self.sims.len(),
            self.neighbors.len(),
            "add_edge requires the build-time similarity buffer; use push_edge"
        );
        debug_assert!((from as usize) < self.len(), "from out of range");
        debug_assert!((to as usize) < self.len(), "to out of range");
        self.edge_list_mut(from).add_edge(to, sim);
    }

    /// Mutable view of `node`'s edge list.
    fn edge_list_mut(&mut self, node: NodeId) -> EdgeListMut<'_> {
        let k = self.max_edges;
        let start = node as usize * k;
        EdgeListMut {
            node,
            neighbors: &mut self.neighbors[start..start + k],
            sims: &mut self.sims[start..start + k],
        }
    }

    /// Iterates disjoint mutable edge lists in node order.
    pub(crate) fn edge_lists_mut(&mut self) -> impl Iterator<Item = EdgeListMut<'_>> {
        debug_assert_eq!(
            self.sims.len(),
            self.neighbors.len(),
            "edge_lists_mut requires the build-time similarity buffer"
        );
        let k = self.max_edges;
        self.neighbors
            .chunks_mut(k)
            .zip(self.sims.chunks_mut(k))
            .enumerate()
            .map(|(node, (neighbors, sims))| EdgeListMut {
                node: node as NodeId,
                neighbors,
                sims,
            })
    }

    /// Appends one ordered edge while deserializing a graph.
    ///
    /// # Panics
    ///
    /// Panics when the node is invalid or its adjacency is full.
    pub fn push_edge(&mut self, from: NodeId, to: NodeId) {
        debug_assert!((from as usize) < self.len(), "from out of range");
        let k = self.max_edges;
        let degree = self.degree(from);
        assert!(degree < k, "node already has max_edges neighbors");
        self.neighbors[from as usize * k + degree] = to;
    }

    /// Replaces one node's ordered neighbor list.
    ///
    /// # Panics
    ///
    /// Panics when the node is invalid or the adjacency exceeds the maximum degree.
    pub fn set_neighbors(&mut self, node: NodeId, neighbors: &[NodeId]) {
        let k = self.max_edges;
        assert!(neighbors.len() <= k, "too many neighbors for node");
        debug_assert!((node as usize) < self.len(), "node out of range");
        let base = node as usize * k;
        let run = &mut self.neighbors[base..base + k];
        run[..neighbors.len()].copy_from_slice(neighbors);
        run[neighbors.len()..].fill(EMPTY);
    }

    /// Returns a node's degree.
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        let base = node as usize * self.max_edges;
        self.neighbors[base..base + self.max_edges]
            .iter()
            .take_while(|&&n| n != EMPTY)
            .count()
    }

    /// Returns a node's best-first neighbors.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let base = node as usize * self.max_edges;
        &self.neighbors[base..base + self.degree(node)]
    }

    /// Returns the number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.num_vectors(self.dim)
    }

    /// Returns whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the maximum out-degree.
    #[inline]
    pub fn max_edges(&self) -> usize {
        self.max_edges
    }

    /// Serializes maximum degree and neighbor identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error when the output cannot be written or the degree exceeds `u32`.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        let max_edges = u32::try_from(self.max_edges)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "max_edges exceeds u32"))?;
        max_edges.serialize(out)?;
        for &neighbor in &self.neighbors {
            neighbor.serialize(out)?;
        }
        Ok(())
    }

    /// Returns the vector arena.
    #[inline]
    pub fn arena(&self) -> &S {
        &self.vectors
    }
}

/// Typed graph accessors.
impl<T, S: Deref<Target = [T]>> Graph<S> {
    /// Returns a node's vector.
    #[inline]
    pub fn payload(&self, node: NodeId) -> &[T] {
        let start = node as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Iterates vectors in node order.
    #[inline]
    pub fn iter(&self) -> std::slice::ChunksExact<'_, T> {
        self.vectors.chunks_exact(self.dim)
    }
}

impl<'a, T: 'a, S: Deref<Target = [T]>> IntoIterator for &'a Graph<S> {
    type Item = &'a [T];
    type IntoIter = std::slice::ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutable view of one node's edge list.
pub(crate) struct EdgeListMut<'a> {
    /// Owner of the edge list.
    node: NodeId,
    neighbors: &'a mut [NodeId],
    sims: &'a mut [Similarity],
}

impl EdgeListMut<'_> {
    /// Inserts an edge into the bounded best-first list.
    pub(crate) fn add_edge(&mut self, to: NodeId, sim: Similarity) {
        if to == self.node {
            return;
        }

        let last = self.sims.len() - 1;
        if sim <= self.sims[last] {
            return;
        }

        if let Some(pos) = self.neighbors.iter().position(|&n| n == to) {
            if sim <= self.sims[pos] {
                return;
            }
            self.sims[pos] = sim;
            let mut j = pos;
            while j > 0 && self.sims[j - 1] < self.sims[j] {
                self.neighbors.swap(j - 1, j);
                self.sims.swap(j - 1, j);
                j -= 1;
            }
            return;
        }

        let mut j = last;
        while j > 0 && self.sims[j - 1] < sim {
            self.neighbors[j] = self.neighbors[j - 1];
            self.sims[j] = self.sims[j - 1];
            j -= 1;
        }
        self.neighbors[j] = to;
        self.sims[j] = sim;
    }
}

/// Why a [`RelativeNeighborhoodGraph::search`] stopped expanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum SearchTerminationReason {
    /// The beam converged.
    SearchConverged,
    /// The reachable graph was exhausted.
    GraphExhausted,
}

/// Records graph-search work and termination.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct NeighborhoodGraphSearchMetrics {
    /// Number of scored nodes.
    pub visited_count: usize,
    /// Number of expanded candidates.
    pub expanded_count: usize,
    /// Number of scanned neighbor entries.
    pub edges_scanned: usize,
    /// Number of beam evictions.
    pub evictions: usize,
    /// Number of returned candidates.
    pub result_count: usize,
    /// Why the expansion loop stopped.
    pub termination_reason: SearchTerminationReason,
}

impl Default for NeighborhoodGraphSearchMetrics {
    fn default() -> Self {
        Self {
            visited_count: 0,
            expanded_count: 0,
            edges_scanned: 0,
            evictions: 0,
            result_count: 0,
            termination_reason: SearchTerminationReason::GraphExhausted,
        }
    }
}

/// Tuning knobs for a [`RelativeNeighborhoodGraph`].
#[derive(Clone, Copy, Debug)]
pub struct NeighborhoodGraphConfig {
    /// Maximum out-degree per node.
    pub max_edges: usize,
    /// Query beam width.
    pub ef: usize,
    /// Candidate count per node during refinement.
    pub num_candidates: usize,
    /// Number of partition trees used to seed the graph.
    pub num_trees: usize,
}

impl Default for NeighborhoodGraphConfig {
    fn default() -> Self {
        NeighborhoodGraphConfig {
            max_edges: 32,
            ef: 64,
            num_candidates: 256,
            num_trees: 32,
        }
    }
}

/// Yields graph-search candidates in converged beam batches.
pub struct SearchIterator<'g, 'w, S: VectorArena, const RESUMABLE: bool> {
    rng: &'g RelativeNeighborhoodGraph<S>,
    workspace: &'w mut Workspace,
    query: &'g [S::Elem],
    /// Beam width.
    ef: usize,
    /// Current converged batch.
    batch: Vec<Candidate>,
    /// Accumulated search metrics.
    metrics: NeighborhoodGraphSearchMetrics,
}

/// Search iterator that retains evictions across beam rounds.
pub type ResumableSearchIterator<'g, 'w, S> = SearchIterator<'g, 'w, S, true>;

/// Search iterator that drops evictions after one beam round.
type OneShotSearchIterator<'g, 'w, S> = SearchIterator<'g, 'w, S, false>;

impl<'g, 'w, S: VectorArena, const RESUMABLE: bool> SearchIterator<'g, 'w, S, RESUMABLE> {
    fn new(
        rng: &'g RelativeNeighborhoodGraph<S>,
        workspace: &'w mut Workspace,
        query: &'g [S::Elem],
        seeds: &[NodeId],
        ef: usize,
    ) -> Self {
        debug_assert_eq!(query.len(), rng.graph.dim(), "query dimension mismatch");
        let n = rng.graph.len();
        workspace.begin_query(n);

        let arena = rng.graph.arena();
        let dim = rng.graph.dim();
        let mut metrics = NeighborhoodGraphSearchMetrics::default();

        for &node_id in seeds {
            if node_id as usize >= n || workspace.visited.contains(node_id) {
                continue;
            }
            workspace.visited.insert(node_id);
            metrics.visited_count += 1;
            let sim = arena.similarity(rng.metric, dim, node_id, query);
            workspace.frontier.push(Candidate { sim, node: node_id });
        }

        SearchIterator {
            rng,
            workspace,
            query,
            ef,
            batch: Vec::new(),
            metrics,
        }
    }

    /// Returns accumulated search metrics.
    pub fn metrics(&self) -> NeighborhoodGraphSearchMetrics {
        self.metrics
    }

    /// Runs one beam round to convergence and drains it into `self.batch`.
    fn run_round(&mut self) {
        let graph = &self.rng.graph;
        let arena = graph.arena();
        let dim = graph.dim();
        let metric = self.rng.metric;
        let ws = &mut *self.workspace;

        self.metrics.termination_reason = SearchTerminationReason::GraphExhausted;

        while let Some(&candidate) = ws.frontier.peek() {
            // A strict candidate order prevents equal-score cycles.
            if ws.results.len() >= self.ef
                && ws.results.peek().is_some_and(|worst| candidate < worst.0)
            {
                self.metrics.termination_reason = SearchTerminationReason::SearchConverged;
                break;
            }

            ws.frontier.pop();
            if ws.results.len() < self.ef {
                ws.results.push(Reverse(candidate));
            } else if let Some(mut worst) = ws.results.peek_mut() {
                let evicted = std::mem::replace(&mut *worst, Reverse(candidate)).0;
                drop(worst);
                if RESUMABLE {
                    ws.frontier.push(evicted);
                }
                self.metrics.evictions += 1;
            }

            let neighbors = graph.neighbors(candidate.node);
            self.metrics.expanded_count += 1;
            self.metrics.edges_scanned += neighbors.len();

            for &neighbor in neighbors {
                if ws.visited.contains(neighbor) {
                    continue;
                }
                ws.visited.insert(neighbor);
                self.metrics.visited_count += 1;

                let sim = arena.similarity(metric, dim, neighbor, self.query);
                ws.frontier.push(Candidate {
                    sim,
                    node: neighbor,
                });
            }
        }

        self.batch.extend(ws.results.drain().map(|Reverse(c)| c));
        // Pop order is descending similarity with ascending node-id ties.
        self.batch
            .sort_unstable_by(|a, b| a.sim.cmp(&b.sim).then_with(|| b.node.cmp(&a.node)));
    }
}

impl<S: VectorArena, const RESUMABLE: bool> Iterator for SearchIterator<'_, '_, S, RESUMABLE> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch.is_empty() {
            self.run_round();
        }
        let candidate = self.batch.pop()?;
        self.metrics.result_count += 1;
        Some(candidate)
    }
}

/// Relative-neighborhood graph over vector storage.
pub struct RelativeNeighborhoodGraph<S> {
    /// Graph storage.
    graph: Graph<S>,
    /// Similarity metric.
    metric: Metric,
    /// Graph configuration.
    config: NeighborhoodGraphConfig,
}

impl<S: VectorArena> RelativeNeighborhoodGraph<S> {
    /// Creates an empty relative-neighborhood graph.
    ///
    /// # Panics
    ///
    /// Panics when the dimensions, degree, or node count are invalid.
    pub fn new(vectors: S, dim: usize, metric: Metric, params: NeighborhoodGraphConfig) -> Self {
        RelativeNeighborhoodGraph {
            graph: Graph::new(vectors, dim, params.max_edges),
            metric,
            config: params,
        }
    }

    /// Searches for the `k` most similar nodes.
    pub fn search(
        &self,
        ws: &mut Workspace,
        query: &[S::Elem],
        seeds: &[NodeId],
        k: usize,
    ) -> (Vec<Candidate>, NeighborhoodGraphSearchMetrics) {
        if self.graph.is_empty() || k == 0 {
            return (Vec::new(), NeighborhoodGraphSearchMetrics::default());
        }
        let mut iter = OneShotSearchIterator::new(self, ws, query, seeds, self.config.ef.max(k));
        let out: Vec<Candidate> = iter.by_ref().take(k).collect();
        let metrics = iter.metrics();
        (out, metrics)
    }

    /// Starts a resumable graph search.
    pub fn search_iter<'g, 'w>(
        &'g self,
        ws: &'w mut Workspace,
        query: &'g [S::Elem],
        seeds: &[NodeId],
    ) -> ResumableSearchIterator<'g, 'w, S> {
        ResumableSearchIterator::new(self, ws, query, seeds, self.config.ef)
    }

    /// Serializes the graph adjacency.
    ///
    /// # Errors
    ///
    /// Returns an error when the output cannot be written.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        self.graph.serialize(out)
    }

    /// Opens a serialized search-only graph.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid adjacency payload.
    pub fn open(
        adjacency: &[u8],
        vectors: S,
        dim: usize,
        metric: Metric,
        params: NeighborhoodGraphConfig,
    ) -> io::Result<Self> {
        Ok(RelativeNeighborhoodGraph {
            graph: Graph::open(adjacency, vectors, dim)?,
            metric,
            config: params,
        })
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Returns whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
}

/// Typed graph construction and refinement.
impl<T: VectorElement, S: Deref<Target = [T]>> RelativeNeighborhoodGraph<S> {
    /// Refines each node's relative-neighborhood adjacency.
    pub fn refine(&mut self, executor: &Executor)
    where S: Sync {
        let len = self.graph.len();
        if len == 0 {
            return;
        }

        // Phase 1: select neighbors from the shared snapshot.
        let chunk = (len / executor.num_threads()).max(1);
        let ranges = (0..len)
            .step_by(chunk)
            .map(|s| (s as NodeId, (s + chunk).min(len) as NodeId));
        let chunked_selected: Vec<Vec<Vec<NodeId>>> = {
            let rng = &*self;
            executor
                .map(
                    move |(start, end): (NodeId, NodeId)| {
                        let mut ws = Workspace::new();
                        let mut out = Vec::with_capacity((end - start) as usize);
                        for node in start..end {
                            let query = rng.graph.payload(node);
                            let (candidates, _) =
                                rng.search(&mut ws, query, &[node], rng.config.num_candidates);
                            out.push(rng.select_neighbors(node, &candidates));
                        }
                        Ok(out)
                    },
                    ranges,
                )
                .expect("refine search panicked")
        };

        // Phase 2: replace adjacency runs.
        let mut node: NodeId = 0;
        for chunk in &chunked_selected {
            for selected in chunk {
                self.graph.set_neighbors(node, selected);
                node += 1;
            }
        }
    }

    /// Selects a bounded relative-neighborhood adjacency.
    fn select_neighbors(&self, node: NodeId, candidates: &[Candidate]) -> Vec<NodeId> {
        let max_edges = self.config.max_edges;
        let mut selected: Vec<NodeId> = Vec::with_capacity(max_edges);
        for &Candidate { sim, node: cand } in candidates {
            if cand == node {
                continue;
            }
            if selected.len() >= max_edges {
                break;
            }
            let cand_vec = self.graph.payload(cand);
            let keep = selected
                .iter()
                .all(|&r| self.metric.similarity(self.graph.payload(r), cand_vec) <= sim);
            if keep {
                selected.push(cand);
            }
        }

        debug_assert!(!selected.is_empty(), "selected nodes should not be empty");
        selected
    }
}

/// Seed for deterministic graph initialization.
const KNN_INIT_TPT_SEED: u64 = 42;

/// Binary32 graph construction.
impl RelativeNeighborhoodGraph<&[f32]> {
    /// Builds a relative-neighborhood graph.
    pub fn build(&mut self, executor: &Executor) {
        self.build_init_knn(executor);
        self.refine(executor);
    }

    /// Seeds the graph from partition-tree leaves.
    fn build_init_knn(&mut self, executor: &Executor) {
        let vectors = *self.graph.arena();
        let dim = self.graph.dim();
        let n = self.graph.len();
        if n == 0 {
            return;
        }

        let metric = self.metric;
        let mut tpt = partition::TPTree::new(
            partition::TPTreeConfig::default(),
            dim,
            vectors,
            KNN_INIT_TPT_SEED,
        );
        let mut indices: Vec<NodeId> = (0..n as NodeId).collect();
        for _ in 0..self.config.num_trees {
            let leaves = tpt.partition(&mut indices);

            // `indices` is a permutation, so every mutable edge list is unique.
            let mut unclaimed: Vec<Option<EdgeListMut>> =
                self.graph.edge_lists_mut().map(Some).collect();
            let mut edge_lists: Vec<EdgeListMut> = indices
                .iter()
                .map(|&node| {
                    unclaimed[node as usize]
                        .take()
                        .expect("indices is a permutation")
                })
                .collect();

            let mut leaf_tasks: Vec<(&[NodeId], &mut [EdgeListMut])> =
                Vec::with_capacity(leaves.len());
            let mut rest = edge_lists.as_mut_slice();
            for leaf in &leaves {
                let (leaf_lists, tail) = std::mem::take(&mut rest).split_at_mut(leaf.len());
                leaf_tasks.push((&indices[leaf.clone()], leaf_lists));
                rest = tail;
            }
            debug_assert!(rest.is_empty(), "leaves must tile all of indices");

            executor
                .map(
                    move |(members, edge_lists): (&[NodeId], &mut [EdgeListMut])| {
                        for i in 0..members.len() {
                            let vec_a = &vectors[members[i] as usize * dim..][..dim];
                            for j in (i + 1)..members.len() {
                                let vec_b = &vectors[members[j] as usize * dim..][..dim];
                                let sim = metric.similarity(vec_a, vec_b);
                                edge_lists[i].add_edge(members[j], sim);
                                edge_lists[j].add_edge(members[i], sim);
                            }
                        }
                        Ok(())
                    },
                    leaf_tasks.into_iter(),
                )
                .expect("leaf KNN computation panicked");
        }
    }
}

/// Reusable buffers for graph search.
pub struct Workspace {
    /// Nodes scored in the current query, 1 bit per node.
    pub(crate) visited: BitSet,
    /// Max-heap by similarity: every scored candidate not currently committed
    /// to `results` — the pool the search pops from, best first.
    pub(crate) frontier: BinaryHeap<Candidate>,
    /// Min-heap by similarity (via `Reverse`): the current beam — the best
    /// `width` committed results, with the least-similar on top for eviction.
    pub(crate) results: BinaryHeap<Reverse<Candidate>>,
}

impl Default for Workspace {
    fn default() -> Self {
        Workspace {
            visited: BitSet::with_max_value(0),
            frontier: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }
}

impl Workspace {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Workspace::default()
    }

    /// Prepares the workspace for a query over `n` nodes: zeroes the visited
    /// bitset (growing it if needed) and clears the heaps.
    pub(crate) fn begin_query(&mut self, n: usize) {
        if (self.visited.max_value() as usize) < n {
            self.visited = BitSet::with_max_value(n as u32);
        } else {
            self.visited.clear();
        }
        self.frontier.clear();
        self.results.clear();
    }
}

/// A `(similarity, node)` pair ordered by similarity (ties broken by node id).
///
/// Higher similarity sorts greater, so a max-heap yields most-similar first.
/// Generic over the id type (`NodeId` by default).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Candidate<N = NodeId> {
    /// Similarity to the query (higher is more similar).
    pub sim: Similarity,
    pub node: N,
}

impl<N: Eq> Eq for Candidate<N> {}

impl<N: Ord> Ord for Candidate<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim
            .cmp(&other.sim)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl<N: Ord> PartialOrd for Candidate<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_nodes(n: NodeId, max_edges: usize) -> Graph<Vec<f32>> {
        Graph::new((0..n).map(|i| i as f32).collect(), 1, max_edges)
    }

    fn sim(score: f32) -> Similarity {
        Similarity::new(score)
    }

    #[test]
    fn edge_lists_mut_allows_disjoint_parallel_writes() {
        let mut g = graph_with_nodes(4, 2);
        let mut lists: Vec<EdgeListMut> = g.edge_lists_mut().collect();
        let (left, right) = lists.split_at_mut(2);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                left[0].add_edge(1, sim(1.0));
                left[1].add_edge(0, sim(1.0));
            });
            scope.spawn(move || {
                right[0].add_edge(3, sim(1.0));
                right[1].add_edge(2, sim(1.0));
            });
        });
        drop(lists);
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[0]);
        assert_eq!(g.neighbors(2), &[3]);
        assert_eq!(g.neighbors(3), &[2]);
    }

    #[test]
    fn construction_derives_nodes_from_the_arena() {
        let g: Graph<Vec<f32>> = Graph::new(vec![1.0, 2.0, 3.0, 4.0], 2, 8);
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty());
        assert_eq!(g.payload(0), &[1.0, 2.0]);
        assert_eq!(g.payload(1), &[3.0, 4.0]);
        assert_eq!(g.degree(0), 0);
        assert!(g.neighbors(0).is_empty());

        let empty: Graph<Vec<f32>> = Graph::new(Vec::new(), 2, 8);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn borrowed_storage_leaves_the_arena_with_the_caller() {
        let matrix: Vec<f32> = vec![0.0, 1.0, 2.0];
        let mut g: Graph<&[f32]> = Graph::new(&matrix, 1, 2);
        let vectors = *g.arena();
        g.add_edge(0, 1, sim(1.0));
        assert_eq!(vectors, matrix.as_slice());
        assert_eq!(g.neighbors(0), &[1]);
    }

    #[test]
    #[should_panic(expected = "arena not a multiple of dim")]
    fn construction_rejects_a_misaligned_arena() {
        let _ = Graph::new(vec![1.0f32, 2.0, 3.0], 2, 4);
    }

    #[test]
    fn edges_are_sorted_best_first() {
        let mut g = graph_with_nodes(5, 8);
        g.add_edge(0, 3, sim(0.1));
        g.add_edge(0, 1, sim(0.8));
        g.add_edge(0, 4, sim(0.5));
        g.add_edge(0, 2, sim(0.9));
        assert_eq!(g.neighbors(0), &[2, 1, 4, 3]);
        assert_eq!(g.degree(0), 4);
    }

    #[test]
    fn bounded_top_k_evicts_the_least_similar() {
        let mut g = graph_with_nodes(5, 2);
        g.add_edge(0, 1, sim(0.5));
        g.add_edge(0, 2, sim(0.6));
        g.add_edge(0, 3, sim(0.9));
        assert_eq!(g.neighbors(0), &[3, 2]);
        g.add_edge(0, 4, sim(0.1));
        assert_eq!(g.neighbors(0), &[3, 2]);
    }

    #[test]
    fn re_adding_keeps_the_more_similar_score() {
        let mut g = graph_with_nodes(4, 4);
        g.add_edge(0, 1, sim(0.2));
        g.add_edge(0, 2, sim(0.5));
        g.add_edge(0, 1, sim(0.9));
        assert_eq!(g.neighbors(0), &[1, 2]);
        g.add_edge(0, 1, sim(0.1));
        assert_eq!(g.neighbors(0), &[1, 2]);
    }

    #[test]
    fn edges_are_directed_and_self_edges_ignored() {
        let mut g = graph_with_nodes(3, 4);
        g.add_edge(0, 1, sim(0.3));
        assert_eq!(g.neighbors(0), &[1]);
        assert!(g.neighbors(1).is_empty());
        g.add_edge(2, 2, sim(1.0));
        assert!(g.neighbors(2).is_empty());
    }

    #[test]
    fn iter_yields_vectors_in_node_order() {
        let g: Graph<Vec<f32>> = Graph::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 4);

        let collected: Vec<&[f32]> = g.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], &[1.0, 2.0]);
        assert_eq!(collected[1], &[3.0, 4.0]);
        assert_eq!(collected[2], &[5.0, 6.0]);

        for (id, vector) in (&g).into_iter().enumerate() {
            assert_eq!(vector, g.payload(id as NodeId));
        }
    }

    #[test]
    fn for_reload_pushes_edges_in_stored_order() {
        let arena: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let mut g = Graph::for_reload(arena, 1, 4);
        g.push_edge(0, 1);
        g.push_edge(0, 2);
        g.push_edge(0, 3);
        assert_eq!(g.neighbors(0), &[1, 2, 3]);
        assert_eq!(g.degree(0), 3);
    }

    #[test]
    fn set_neighbors_overwrites_and_repads() {
        let mut g = graph_with_nodes(5, 4);
        g.set_neighbors(0, &[3, 1, 2]);
        assert_eq!(g.neighbors(0), &[3, 1, 2]);
        assert_eq!(g.degree(0), 3);
        g.set_neighbors(0, &[4]);
        assert_eq!(g.neighbors(0), &[4]);
        assert_eq!(g.degree(0), 1);
        g.set_neighbors(0, &[]);
        assert!(g.neighbors(0).is_empty());
        assert_eq!(g.degree(0), 0);
    }

    #[test]
    #[should_panic(expected = "too many neighbors")]
    fn set_neighbors_rejects_more_than_max_edges() {
        let mut g = graph_with_nodes(4, 2);
        g.set_neighbors(0, &[1, 2, 3]);
    }

    fn decode(bytes: &[u8]) -> (u32, Vec<NodeId>) {
        assert_eq!(
            bytes.len() % 4,
            0,
            "serialization is a whole number of u32s"
        );
        let mut words = bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()));
        let max_edges = words.next().expect("missing max_edges header");
        (max_edges, words.collect())
    }

    #[test]
    fn serialize_writes_max_edges_then_padded_adjacency() {
        let mut g = graph_with_nodes(3, 2);
        g.add_edge(0, 2, sim(0.2));
        g.add_edge(0, 1, sim(0.9));
        g.add_edge(1, 0, sim(0.9));
        let mut bytes = Vec::new();
        g.serialize(&mut bytes).unwrap();

        let (max_edges, neighbors) = decode(&bytes);
        assert_eq!(max_edges, 2);
        assert_eq!(neighbors, vec![1, 2, 0, EMPTY, EMPTY, EMPTY]);
    }

    #[test]
    fn reloaded_graph_serializes_byte_identically() {
        let mut built = graph_with_nodes(4, 3);
        built.add_edge(0, 1, sim(0.9));
        built.add_edge(0, 3, sim(0.6));
        built.add_edge(2, 0, sim(0.8));

        let mut bytes = Vec::new();
        built.serialize(&mut bytes).unwrap();

        let arena: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let mut reloaded = Graph::for_reload(arena, 1, built.max_edges());
        for node in 0..built.len() as NodeId {
            for &to in built.neighbors(node) {
                reloaded.push_edge(node, to);
            }
        }

        let mut reloaded_bytes = Vec::new();
        reloaded.serialize(&mut reloaded_bytes).unwrap();
        assert_eq!(bytes, reloaded_bytes);
    }
}

#[cfg(test)]
mod rng_tests {
    use super::*;

    fn line_index(n: NodeId) -> RelativeNeighborhoodGraph<Vec<f32>> {
        let params = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 1, Metric::L2, params);
        for i in 0..n as i64 {
            for off in [-2i64, -1, 1, 2] {
                let nb = i + off;
                if (0..n as i64).contains(&nb) {
                    let sim = Metric::L2.similarity(&[i as f32], &[nb as f32]);
                    rng.graph.add_edge(i as NodeId, nb as NodeId, sim);
                }
            }
        }
        rng
    }

    #[test]
    fn search_finds_nearest_neighbors() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let (res, metrics) = rng.search(&mut ws, &[4.2], &[0], 3);
        let ids: Vec<NodeId> = res.iter().map(|c| c.node).collect();
        assert_eq!(ids, vec![4, 5, 3]);
        assert_eq!(metrics.result_count, 3);
        assert!(metrics.visited_count >= 3);
        assert!(metrics.expanded_count >= 1);
        assert!(metrics.edges_scanned >= metrics.visited_count - 1);
        assert!(res[0].sim >= res[1].sim && res[1].sim >= res[2].sim);
    }

    #[test]
    fn search_handles_degenerate_inputs() {
        let rng = line_index(5);
        let mut ws = Workspace::new();
        assert!(rng.search(&mut ws, &[1.0], &[0], 0).0.is_empty());
        assert!(rng.search(&mut ws, &[1.0], &[], 3).0.is_empty());

        let empty: RelativeNeighborhoodGraph<Vec<f32>> = RelativeNeighborhoodGraph::new(
            Vec::new(),
            1,
            Metric::L2,
            NeighborhoodGraphConfig::default(),
        );
        assert!(empty.search(&mut ws, &[1.0], &[0], 3).0.is_empty());
    }

    #[test]
    fn search_reuses_workspace_deterministically() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let (a, _) = rng.search(&mut ws, &[4.2], &[0], 3);
        let (b, _) = rng.search(&mut ws, &[4.2], &[0], 3);
        assert_eq!(a, b);
    }

    #[test]
    fn search_iter_first_batch_matches_search() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let (batch, _) = rng.search(&mut ws, &[4.2], &[0], 3);
        let iterated: Vec<Candidate> = rng.search_iter(&mut ws, &[4.2], &[0]).take(3).collect();
        assert_eq!(batch, iterated);
    }

    #[test]
    fn search_iter_first_batch_is_the_true_top_ef_sorted() {
        let rng = line_index(20);
        let mut ws = Workspace::new();
        let first_batch: Vec<NodeId> = rng
            .search_iter(&mut ws, &[10.2], &[0])
            .take(8)
            .map(|c| c.node)
            .collect();
        assert_eq!(first_batch, vec![10, 11, 9, 12, 8, 13, 7, 14]);
    }

    #[test]
    fn search_iter_resumes_past_the_first_batch() {
        let rng = line_index(20);
        let mut ws = Workspace::new();
        let mut ids: Vec<NodeId> = rng
            .search_iter(&mut ws, &[10.2], &[0])
            .take(12)
            .map(|c| c.node)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (5..=16).collect::<Vec<NodeId>>());
    }

    #[test]
    fn search_iter_yields_every_reachable_node_exactly_once() {
        let n: NodeId = 20;
        let rng = line_index(n);
        let mut ws = Workspace::new();
        let mut iter = rng.search_iter(&mut ws, &[4.2], &[0]);
        let mut ids: Vec<NodeId> = iter.by_ref().map(|c| c.node).collect();

        ids.sort_unstable();
        assert_eq!(ids, (0..n).collect::<Vec<NodeId>>());

        let metrics = iter.metrics();
        assert_eq!(metrics.result_count, n as usize);
        assert_eq!(metrics.visited_count, n as usize);
        assert!(metrics.expanded_count >= n as usize);
        assert_eq!(
            metrics.termination_reason,
            SearchTerminationReason::GraphExhausted
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn search_iter_handles_degenerate_inputs() {
        let rng = line_index(5);
        let mut ws = Workspace::new();
        assert_eq!(rng.search_iter(&mut ws, &[1.0], &[]).next(), None);

        let empty: RelativeNeighborhoodGraph<Vec<f32>> = RelativeNeighborhoodGraph::new(
            Vec::new(),
            1,
            Metric::L2,
            NeighborhoodGraphConfig::default(),
        );
        assert_eq!(empty.search_iter(&mut ws, &[1.0], &[0]).next(), None);
    }

    #[test]
    fn search_iter_survives_duplicate_vectors() {
        let params = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 2,
            num_candidates: 4,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..6).map(|i| (i / 2) as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 1, Metric::L2, params);
        for i in 0..6u32 {
            for j in 0..6u32 {
                if i != j {
                    let sim = Metric::L2.similarity(&[(i / 2) as f32], &[(j / 2) as f32]);
                    rng.graph.add_edge(i, j, sim);
                }
            }
        }
        let mut ws = Workspace::new();
        let mut ids: Vec<NodeId> = rng
            .search_iter(&mut ws, &[0.9], &[0])
            .map(|c| c.node)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..6).collect::<Vec<NodeId>>());
    }

    #[test]
    fn search_from_a_node_returns_it_then_nearest() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let (res, _) = rng.search(&mut ws, &[4.0], &[4], 4);
        let ids: Vec<NodeId> = res.iter().map(|c| c.node).collect();
        assert_eq!(ids[0], 4);
        assert!(ids[1] == 3 || ids[1] == 5);
    }

    fn sorted_neighbors<S: VectorArena>(
        rng: &RelativeNeighborhoodGraph<S>,
        node: NodeId,
    ) -> Vec<NodeId> {
        let mut v = rng.graph.neighbors(node).to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn refine_applies_rng_occlusion() {
        let config = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 4,
            num_candidates: 4,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..3).map(|i| i as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 1, Metric::L2, config);
        for i in 0..3i64 {
            for j in 0..3i64 {
                if i != j {
                    let sim = Metric::L2.similarity(&[i as f32], &[j as f32]);
                    rng.graph.add_edge(i as NodeId, j as NodeId, sim);
                }
            }
        }

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1]);
        assert_eq!(sorted_neighbors(&rng, 2), vec![1]);
        assert_eq!(sorted_neighbors(&rng, 1), vec![0, 2]);
    }

    #[test]
    fn refine_prunes_full_mesh_to_the_optimal_path_graph() {
        const N: NodeId = 6;
        let config = NeighborhoodGraphConfig {
            max_edges: 8,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 1, Metric::L2, config);
        for i in 0..N as i64 {
            for j in 0..N as i64 {
                if i != j {
                    let sim = Metric::L2.similarity(&[i as f32], &[j as f32]);
                    rng.graph.add_edge(i as NodeId, j as NodeId, sim);
                }
            }
        }

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1]);
        assert_eq!(sorted_neighbors(&rng, N - 1), vec![N - 2]);
        for i in 1..N - 1 {
            assert_eq!(sorted_neighbors(&rng, i), vec![i - 1, i + 1]);
        }
    }

    fn fully_connect(rng: &mut RelativeNeighborhoodGraph<Vec<f32>>, pts: &[[f32; 2]]) {
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i != j {
                    let sim = Metric::L2.similarity(&pts[i], &pts[j]);
                    rng.graph.add_edge(i as NodeId, j as NodeId, sim);
                }
            }
        }
    }

    #[test]
    fn refine_keeps_duplicate_vector_edges() {
        let config = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 4,
            num_candidates: 4,
            num_trees: 1,
        };
        let pts = [[0.0f32, 0.0], [0.0, 0.0], [1.0, 0.0]];
        let vectors: Vec<f32> = pts.iter().flatten().copied().collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 2, Metric::L2, config);
        fully_connect(&mut rng, &pts);

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1, 2]);
    }

    #[test]
    fn refine_caps_selected_neighbors_at_max_edges() {
        let config = NeighborhoodGraphConfig {
            max_edges: 2,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let vectors: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0, -3.0, 0.0, 0.0, -4.0];
        let mut rng = RelativeNeighborhoodGraph::new(vectors, 2, Metric::L2, config);
        rng.graph.set_neighbors(0, &[1, 2]);
        rng.graph.set_neighbors(1, &[0, 3]);
        rng.graph.set_neighbors(2, &[0, 4]);
        rng.graph.set_neighbors(3, &[1, 0]);
        rng.graph.set_neighbors(4, &[2, 0]);

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1, 2]);
    }

    #[test]
    fn build_init_knn_seeds_reciprocal_edges() {
        let config = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors.as_slice(), 1, Metric::L2, config);

        rng.build_init_knn(&Executor::single_thread());

        for i in 0..6u32 {
            let nbrs = rng.graph.neighbors(i);
            assert!(!nbrs.is_empty(), "node {i} has no edges");
            assert!(
                nbrs[0] == i.wrapping_sub(1) || nbrs[0] == i + 1,
                "node {i}'s nearest edge {} is not adjacent",
                nbrs[0]
            );
        }
        assert!(rng.graph.neighbors(0).contains(&1));
        assert!(rng.graph.neighbors(1).contains(&0));
    }

    #[test]
    fn build_recovers_the_path_graph() {
        const N: NodeId = 6;
        let config = NeighborhoodGraphConfig {
            max_edges: 8,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let vectors: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let mut rng = RelativeNeighborhoodGraph::new(vectors.as_slice(), 1, Metric::L2, config);

        rng.build(&Executor::single_thread());

        assert_eq!(sorted_neighbors(&rng, 0), vec![1]);
        assert_eq!(sorted_neighbors(&rng, N - 1), vec![N - 2]);
        for i in 1..N - 1 {
            assert_eq!(sorted_neighbors(&rng, i), vec![i - 1, i + 1]);
        }
    }
}

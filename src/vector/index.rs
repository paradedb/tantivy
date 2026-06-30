//! The relative neighborhood graph index over a [`Graph`]: greedy beam
//! [`search`](RelativeNeighborhoodGraph::search), KNN
//! [`build`](RelativeNeighborhoodGraph::build), and RNG
//! [`refine`](RelativeNeighborhoodGraph::refine) (pruning).
//!
//! [`Graph`] is the pure storage layer; this type layers on the metric and the
//! search/build parameters. The per-query working buffers live in a separate
//! [`Workspace`] the caller owns and threads through `search`. Because the
//! workspace is a parameter rather than borrowed from the index, `search` only
//! needs `&self` — so a query can run while the caller still borrows the graph
//! (e.g. `refine` uses each node's vector as the query straight from the arena,
//! with no copy), one workspace can be reused across many queries, and many
//! queries can run *concurrently*. That last property is what lets
//! [`build`](RelativeNeighborhoodGraph::build) and
//! [`refine`](RelativeNeighborhoodGraph::refine) fan their per-leaf / per-node
//! search work across an [`Executor`](crate::Executor), serializing only the
//! cheap graph mutation.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::schema::Metric;
use crate::Executor;

use super::graph::{Graph, NodeId};
use super::VectorElement;

/// Tuning knobs for a [`RelativeNeighborhoodGraph`].
#[derive(Clone, Copy, Debug)]
pub struct NeighborhoodGraphConfig {
    /// Maximum out-degree per node (the *k* in *k*-NN graph).
    pub max_edges: usize,
    /// Beam width for query-time search (`>= k`).
    pub ef: usize,
    /// Size of the candidate pool gathered per node during [`refine`]: each
    /// node runs a top-`num_candidates` search of the current graph, and those
    /// candidates feed RNG edge selection.
    ///
    /// [`refine`]: RelativeNeighborhoodGraph::refine
    pub num_candidates: usize,
    /// Number of independent TPT partitions [`build`] unions to seed the initial
    /// KNN graph. Each tree splits along different random directions; unioning
    /// their per-leaf edges stitches across any single tree's split boundaries,
    /// so more trees means fewer missed neighbors (better init recall) at linear
    /// build cost.
    ///
    /// [`build`]: RelativeNeighborhoodGraph::build
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

/// A relative neighborhood graph (RNG) index over vectors of element type `T`.
///
/// Vectors live in the inner [`Graph`]'s flat `dim`-strided arena; queries are
/// `&[T]` of the same element type and dimension. This type owns the metric and
/// parameters; per-query scratch is supplied by the caller as a [`Workspace`].
pub struct RelativeNeighborhoodGraph<T: VectorElement> {
    /// Flat vector arena and directed adjacency.
    graph: Graph<T>,
    /// Similarity metric (higher is better). Search ranks by similarity; build
    /// orders edges by its negation, so smaller is closer.
    metric: Metric,
    /// Search, build, and refine tuning knobs.
    config: NeighborhoodGraphConfig,
}

impl<T: VectorElement> RelativeNeighborhoodGraph<T> {
    /// Creates an empty index with room for `capacity` nodes of `dim`-dimensional
    /// vectors, using `metric` and the given tuning `params`.
    pub fn new(
        capacity: usize,
        dim: usize,
        metric: Metric,
        params: NeighborhoodGraphConfig,
    ) -> Self {
        RelativeNeighborhoodGraph {
            graph: Graph::new(capacity, dim, params.max_edges),
            metric,
            config: params,
        }
    }

    /// Copies `vector` into the flat arena as a new node and returns its id.
    pub fn add_vector(&mut self, vector: &[T]) -> NodeId {
        self.graph.add_node(vector)
    }

    /// Greedy beam search for the `k` nodes most similar to `query`, expanding
    /// outward from `seeds`. Returns [`Candidate`]s most similar first, with a
    /// beam width of `max(ef, k)`.
    ///
    /// `ws` holds the per-query scratch; reuse one across queries to avoid
    /// reallocating. It is reset at the start of each call.
    pub fn search(
        &self,
        ws: &mut Workspace,
        query: &[T],
        seeds: &[NodeId],
        k: usize,
    ) -> Vec<Candidate> {
        debug_assert_eq!(query.len(), self.graph.dim(), "query dimension mismatch");
        if self.graph.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = self.config.ef.max(k);
        let n = self.graph.len();
        let epoch = ws.begin_query(n);

        let visited = &mut ws.visited;
        let frontier = &mut ws.frontier;
        let results = &mut ws.results;

        // Seeds are few (one per entry point) and unique after the visited
        // filter, so they always fit under the beam — push them blindly rather
        // than paying the bounded-insert path.
        debug_assert!(seeds.len() <= ef, "seed count exceeds beam width");
        for &node_id in seeds {
            let idx = node_id as usize;
            if idx >= n || visited[idx] == epoch {
                continue;
            }
            visited[idx] = epoch;
            let sim = self.metric.similarity(query, self.graph.payload(node_id));
            let c = Candidate { sim, node: node_id };
            frontier.push(c);
            results.push(Reverse(c));
        }

        while let Some(cand) = frontier.pop() {
            // Stop once the best unexpanded candidate can't beat the worst kept
            // result and the result set is already full.
            if results.len() >= ef && results.peek().is_some_and(|w| cand.sim < w.0.sim) {
                break;
            }
            for &nb in self.graph.neighbors(cand.node) {
                let idx = nb as usize;
                if visited[idx] == epoch {
                    continue;
                }
                visited[idx] = epoch;
                let sim = self.metric.similarity(query, self.graph.payload(nb));
                let c = Candidate { sim, node: nb };
                // A candidate earns a frontier slot exactly when it enters the
                // top-`ef` results, so the single comparison against the worst
                // kept result is admission test, in-place eviction, and frontier
                // gate at once. `peek_mut` only re-sifts if we mutate through it:
                // a loser costs one compare and no reheapify; a winner costs one
                // sift-down on drop (versus pop + push, which is two).
                if results.len() < ef {
                    results.push(Reverse(c));
                    frontier.push(c);
                } else if let Some(mut worst) = results.peek_mut() {
                    if c.sim > worst.0.sim {
                        *worst = Reverse(c);
                        frontier.push(c);
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.drain().map(|Reverse(c)| c).collect();
        out.sort_unstable_by(|a, b| b.sim.total_cmp(&a.sim).then_with(|| a.node.cmp(&b.node)));
        out.truncate(k);
        out
    }

    /// Refines every node against the current graph: each node searches from
    /// itself to gather a candidate pool, applies the RNG occlusion rule to
    /// reselect its edges, and the new adjacencies are written back. This pass is
    /// what turns a raw KNN graph into an RNG.
    ///
    /// The search-and-select phase is read-only over the graph, so it runs in
    /// parallel on the `executor`; the cheap write-back is applied serially
    /// afterward (a node's adjacency must not be mutated while other nodes are
    /// still reading it). Because every node reads the same pre-pass snapshot,
    /// this is a *synchronous* refinement — all nodes see the original edges, not
    /// each other's updates — which is the shape that parallelizes and is
    /// equivalent in quality on a single pass. Nodes are processed in chunks so
    /// each task reuses one [`Workspace`] instead of allocating a per-node
    /// `O(len)` visited buffer.
    pub fn refine(&mut self, executor: &Executor) {
        let len = self.graph.len();
        if len == 0 {
            return;
        }

        // Phase 1 (parallel, read-only): each node searches the snapshot and
        // RNG-selects its new neighbors. One chunk per executor thread, so a
        // single Workspace is reused across the chunk's nodes rather than
        // allocating a per-node visited buffer. `max(1)` guards the degenerate
        // case of more threads than nodes.
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
                            let candidates =
                                rng.search(&mut ws, query, &[node], rng.config.num_candidates);
                            out.push(rng.select_neighbors(node, &candidates));
                        }
                        Ok(out)
                    },
                    ranges,
                )
                .expect("refine search panicked")
        };

        // Phase 2 (serial): write each node's selection back. Disjoint per node
        // and a bounded copy each, so the serial cost is negligible.
        let mut node: NodeId = 0;
        for chunk in &chunked_selected {
            for selected in chunk {
                self.graph.set_neighbors(node, selected);
                node += 1;
            }
        }
    }

    /// Applies the relative-neighborhood-graph occlusion rule to `candidates`
    /// (nearest-first) and returns the survivors — `node`'s new adjacency, at most
    /// `max_edges`, skipping `node` itself. Read-only, so it can run concurrently
    /// across nodes; the caller writes the result back into the graph.
    ///
    /// Everything is in similarity space (higher is better): a candidate `c` is
    /// kept unless some already-selected neighbor `r` is *more* similar to `c`
    /// than `node` is — then `r` makes the direct `node -> c` edge redundant and
    /// occludes it (the classic RNG "lune" emptiness test). The comparison is
    /// non-strict (`<=`), so an `r` *exactly* as similar as `node` does not
    /// occlude — the canonical RNG definition, and what keeps duplicate vectors
    /// from wiping out a node's whole edge set.
    fn select_neighbors(&self, node: NodeId, candidates: &[Candidate]) -> Vec<NodeId> {
        let max_edges = self.config.max_edges;
        let mut selected: Vec<NodeId> = Vec::with_capacity(max_edges);
        for &Candidate { sim, node: cand } in candidates {
            if cand == node {
                continue; // the query node itself is never its own neighbor
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

/// Build is `f32`-only for now: the TPT partitioner does floating-point
/// variance/projection math over the vectors, so it needs a concrete float
/// element. The rest of the index ([`search`](RelativeNeighborhoodGraph::search),
/// [`refine`](RelativeNeighborhoodGraph::refine)) stays generic over
/// [`VectorElement`]; only this construction path is pinned to `f32`.
impl RelativeNeighborhoodGraph<f32> {
    /// Builds the RNG index over `vectors` — a flat `dim`-strided arena, the same
    /// layout [`TPTree`](partition::TPTree) consumes. Seeds a raw KNN graph with a
    /// TPT forest and then prunes it into an RNG, so a caller needs only this one
    /// call (no separate [`refine`](Self::refine)).
    ///
    /// `vectors` is passed in rather than read back out of the inner [`Graph`] so
    /// the partitioner can borrow the arena immutably while edge insertion borrows
    /// the graph mutably. The `executor` parallelizes the per-leaf distance work
    /// (see [`build_init_knn`](Self::build_init_knn)). Expects an empty graph
    /// sized for at least `vectors.len() / dim` nodes; `vectors.len()` must be a
    /// multiple of `dim`.
    pub fn build(&mut self, executor: &Executor, vectors: &[f32]) {
        self.build_init_knn(executor, vectors);
        self.refine(executor);
    }

    /// Seeds the raw KNN graph: adds every vector as a node, then unions a forest
    /// of [`num_trees`](NeighborhoodGraphConfig::num_trees) TPT partitions,
    /// brute-forcing exact KNN within each leaf and inserting the edges in both
    /// directions. The result is the raw (best-effort symmetric) KNN graph that
    /// [`build`](Self::build) hands to [`refine`](Self::refine).
    fn build_init_knn(&mut self, executor: &Executor, vectors: &[f32]) {
        let dim = self.graph.dim();
        debug_assert_eq!(vectors.len() % dim, 0, "arena not a multiple of dim");
        debug_assert!(self.graph.is_empty(), "build expects an empty graph");
        let n = vectors.len() / dim;
        if n == 0 {
            return;
        }

        for chunk in vectors.chunks_exact(dim) {
            self.graph.add_node(chunk);
        }

        // One TPTree is reused across trees: its RNG advances between partitions
        // so successive trees split along different directions, and each tree
        // starts from the previous one's in-place permutation, diversifying the
        // prefix sample.
        let metric = self.metric;
        let mut tpt = partition::TPTree::new(partition::TPTreeConfig::default(), dim, vectors);
        let mut indices: Vec<NodeId> = (0..n as NodeId).collect();
        for _ in 0..self.config.num_trees {
            let leaves = tpt.partition(&mut indices);

            let indices_ref: &[NodeId] = &indices;
            let per_leaf: Vec<Vec<(NodeId, NodeId, f32)>> = executor
                .map(
                    move |leaf: std::ops::Range<usize>| {
                        let members = &indices_ref[leaf];
                        let mut edges =
                            Vec::with_capacity(members.len() * members.len().saturating_sub(1) / 2);
                        for (i, &a) in members.iter().enumerate() {
                            let va = &vectors[a as usize * dim..][..dim];
                            for &b in &members[i + 1..] {
                                let vb = &vectors[b as usize * dim..][..dim];
                                edges.push((a, b, -metric.similarity(va, vb)));
                            }
                        }
                        Ok(edges)
                    },
                    leaves.into_iter(),
                )
                .expect("leaf KNN computation panicked");

            for edges in &per_leaf {
                for &(a, b, dist) in edges {
                    self.graph.add_edge(a, b, dist);
                    self.graph.add_edge(b, a, dist);
                }
            }
        }
    }
}

/// Reusable per-query working buffers. The caller owns it and passes `&mut` it to [`RelativeNeighborhoodGraph::search`];
/// reuse one across queries to avoid reallocating. It holds only data, no logic.
#[derive(Default)]
pub struct Workspace {
    /// `visited[node] == visited_epoch` marks a node seen in the current query.
    /// The epoch stamp resets the buffer in O(1) between queries (just a counter
    /// bump) instead of re-zeroing all `n` slots.
    visited: Vec<u32>,
    visited_epoch: u32,
    /// Max-heap by similarity: the frontier of candidates left to expand.
    frontier: BinaryHeap<Candidate>,
    /// Min-heap by similarity (via `Reverse`): the best `ef` results so far, with
    /// the least-similar on top for eviction.
    results: BinaryHeap<Reverse<Candidate>>,
}

impl Workspace {
    /// Creates an empty workspace. It grows to fit on first use.
    pub fn new() -> Self {
        Workspace::default()
    }

    /// Prepares the workspace for a query over `n` nodes: grows the visited
    /// buffer, advances the epoch (resetting stamps on wraparound), clears the
    /// heaps, and returns the epoch to stamp this query's visits with.
    fn begin_query(&mut self, n: usize) -> u32 {
        if self.visited.len() < n {
            self.visited.resize(n, 0);
        }
        self.visited_epoch = self.visited_epoch.wrapping_add(1);
        if self.visited_epoch == 0 {
            // Wrapped: clear so no stale stamp collides with the new epoch.
            self.visited.iter_mut().for_each(|v| *v = 0);
            self.visited_epoch = 1;
        }
        self.frontier.clear();
        self.results.clear();
        self.visited_epoch
    }
}

/// A `(similarity, node)` pair ordered by similarity (ties broken by node id for
/// determinism). Ordered ascending, so a plain max-heap yields most-similar
/// first and `Reverse<Candidate>` yields least-similar first. Also the element
/// type [`search`](RelativeNeighborhoodGraph::search) returns.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Candidate {
    /// Similarity to the query (higher is closer); `-sim` is the distance.
    pub sim: f32,
    /// The graph node this candidate refers to.
    pub node: NodeId,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim
            .total_cmp(&other.sim)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Trinary-projection-tree (TPT) partitioning, the candidate generator that
/// seeds the initial KNN graph before [`refine`](RelativeNeighborhoodGraph::refine).
///
/// A TPT recursively splits a slice of node ids along a sparse random
/// hyperplane — only the few highest-variance dimensions carry a weight, the
/// rest are implicitly zero (the "trinary" sparsity). Each split direction is
/// *fit on a sample* of the slice but *applied to the whole slice*, so the cost
/// is independent of node count. Recursion bottoms out at
/// [`leaf_size`](TPTreeConfig::leaf_size); the leaves are small contiguous index
/// ranges the builder brute-forces into exact KNN edges.
pub(crate) mod partition {
    use std::ops::Range;

    use super::NodeId;

    /// Tuning knobs for [`TPTree`].
    #[derive(Clone, Copy, Debug)]
    pub struct TPTreeConfig {
        /// Max points in a leaf before recursion stops. The per-leaf exact KNN is
        /// quadratic in this, so it trades build cost for init-graph recall.
        pub leaf_size: usize,
        /// How many points of each slice to sample when fitting a split
        /// direction. The split is fit on the sample, then applied to the whole
        /// slice.
        pub samples: usize,
        /// How many of the highest-variance dimensions carry a nonzero projection
        /// weight; every other dimension is weighted zero.
        pub top_dims: usize,
        /// Random unit-norm projections tried per split; the one that spreads the
        /// sample most (max projected variance) wins, with the single
        /// highest-variance axis as the baseline.
        pub iterations: usize,
    }

    impl Default for TPTreeConfig {
        fn default() -> Self {
            TPTreeConfig {
                leaf_size: 2000,
                samples: 1000,
                top_dims: 5,
                iterations: 100,
            }
        }
    }

    /// A single TPT over a flat, `dim`-strided vector arena. Borrows the arena
    /// and owns the RNG; [`partition`](TPTree::partition) permutes a caller-owned
    /// `indices` slice in place and returns the leaf ranges into it.
    pub struct TPTree<'a> {
        vectors: &'a [f32],
        dim: usize,
        config: TPTreeConfig,
        rng: fastrand::Rng,
    }

    impl<'a> TPTree<'a> {
        /// Wraps an arena for partitioning. `vectors` is the flat `dim`-strided
        /// buffer (its length must be a multiple of `dim`).
        pub fn new(config: TPTreeConfig, dim: usize, vectors: &'a [f32]) -> Self {
            debug_assert!(dim > 0, "dim must be non-zero");
            debug_assert_eq!(vectors.len() % dim, 0, "arena not a multiple of dim");
            TPTree {
                vectors,
                dim,
                config,
                rng: fastrand::Rng::new(),
            }
        }

        /// Partitions `indices` in place and returns the leaf ranges into it.
        /// Each returned range is a contiguous run of `indices` holding one
        /// leaf's node ids (at most [`leaf_size`](TPTreeConfig::leaf_size)).
        pub fn partition(&mut self, indices: &mut [NodeId]) -> Vec<Range<usize>> {
            let mut leaves = Vec::new();
            if !indices.is_empty() {
                self.subdivide(indices, 0, &mut leaves);
            }
            leaves
        }

        /// Coordinate `d` of `node`.
        #[inline]
        fn coord(&self, node: NodeId, d: usize) -> f32 {
            self.vectors[node as usize * self.dim + d]
        }

        /// Recursively splits `indices` (whose first element sits at absolute
        /// `offset` in the original array), appending leaf ranges to `leaves`.
        fn subdivide(
            &mut self,
            indices: &mut [NodeId],
            offset: usize,
            leaves: &mut Vec<Range<usize>>,
        ) {
            if indices.len() <= self.config.leaf_size {
                leaves.push(offset..offset + indices.len());
                return;
            }
            let split = self.choose_split(indices);
            let (left, right) = indices.split_at_mut(split);
            self.subdivide(left, offset, leaves);
            self.subdivide(right, offset + split, leaves);
        }

        /// Picks a split hyperplane for `indices` and partitions the slice around
        /// it in place, returning the boundary `split` (left = `[0, split)`,
        /// right = `[split, len)`). The boundary is always in `1..len`, so each
        /// child is strictly smaller and the recursion terminates.
        fn choose_split(&mut self, indices: &mut [NodeId]) -> usize {
            let n = indices.len();
            let dim = self.dim;
            let sample = n.min(self.config.samples);
            let top_dims = self.config.top_dims.min(dim).max(1);

            // Per-dimension mean over the sample.
            let mut mean = vec![0.0f32; dim];
            for &node in &indices[..sample] {
                for (d, m) in mean.iter_mut().enumerate() {
                    *m += self.coord(node, d);
                }
            }
            for m in &mut mean {
                *m /= sample as f32;
            }

            // Per-dimension variance (sum of squared deviations) over the sample.
            let mut variance = vec![0.0f32; dim];
            for &node in &indices[..sample] {
                for (d, var) in variance.iter_mut().enumerate() {
                    let diff = self.coord(node, d) - mean[d];
                    *var += diff * diff;
                }
            }

            // The top-`top_dims` highest-variance dimensions; only these carry a
            // projection weight.
            let mut dims: Vec<usize> = (0..dim).collect();
            dims.sort_unstable_by(|&a, &b| variance[b].total_cmp(&variance[a]));
            dims.truncate(top_dims);

            // Baseline: project onto the single highest-variance axis.
            let mut best_weight = vec![0.0f32; top_dims];
            best_weight[0] = 1.0;
            let mut best_mean = mean[dims[0]];
            let mut best_var = variance[dims[0]];

            // Try random unit-norm projections over the top dims; keep whichever
            // spreads the sample the most.
            let mut proj = vec![0.0f32; sample];
            let mut weight = vec![0.0f32; top_dims];
            for _ in 0..self.config.iterations {
                let mut norm = 0.0f32;
                for w in &mut weight {
                    *w = self.rng.f32() * 2.0 - 1.0; // [-1, 1)
                    norm += *w * *w;
                }
                let norm = norm.sqrt();
                if norm == 0.0 {
                    continue;
                }
                for w in &mut weight {
                    *w /= norm;
                }

                let mut m = 0.0f32;
                for (slot, &node) in proj.iter_mut().zip(&indices[..sample]) {
                    let mut v = 0.0f32;
                    for (k, &d) in dims.iter().enumerate() {
                        v += weight[k] * self.coord(node, d);
                    }
                    *slot = v;
                    m += v;
                }
                m /= sample as f32;

                let mut var = 0.0f32;
                for &p in &proj {
                    let diff = p - m;
                    var += diff * diff;
                }
                if var > best_var {
                    best_var = var;
                    best_mean = m;
                    best_weight.copy_from_slice(&weight);
                }
            }

            // Partition the WHOLE slice around the chosen hyperplane: below the
            // threshold stays left, at-or-above swaps to the tail (two-pointer,
            // in place). `i`/`j` are signed so the right pointer can cross zero.
            let mut i: isize = 0;
            let mut j: isize = n as isize - 1;
            while i <= j {
                let node = indices[i as usize];
                let mut val = 0.0f32;
                for (k, &d) in dims.iter().enumerate() {
                    val += best_weight[k] * self.coord(node, d);
                }
                if val < best_mean {
                    i += 1;
                } else {
                    indices.swap(i as usize, j as usize);
                    j -= 1;
                }
            }

            // Everything landed on one side (e.g. identical vectors): fall back to
            // a median split so the recursion still shrinks.
            let split = i as usize;
            if split == 0 || split == n {
                n / 2
            } else {
                split
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn arena(pts: &[[f32; 3]]) -> Vec<f32> {
            pts.iter().flatten().copied().collect()
        }

        #[test]
        fn partition_separates_two_far_clusters() {
            // Two clusters far apart in the x–z plane (y is low-variance noise).
            // One split must cleanly separate them: max-variance projection puts
            // the threshold in the gap, so no leaf mixes the two.
            let pts = [
                [1., 5., 1.],
                [2., 5., 0.],
                [0., 4., 2.],
                [1., 6., 1.], // cluster A: ids 0..4
                [9., 5., 10.],
                [10., 5., 9.],
                [8., 4., 11.],
                [9., 6., 10.], // cluster B: ids 4..8
            ];
            let v = arena(&pts);
            let config = TPTreeConfig {
                leaf_size: 4,
                samples: 8,
                top_dims: 2,
                iterations: 100,
            };
            let mut tpt = TPTree::new(config, 3, &v);
            let mut indices: Vec<NodeId> = (0..8).collect();

            let leaves = tpt.partition(&mut indices);

            assert_eq!(leaves.len(), 2, "8 points / leaf_size 4 → one split");
            for leaf in leaves {
                let ids = &indices[leaf];
                let all_a = ids.iter().all(|&id| id < 4);
                let all_b = ids.iter().all(|&id| id >= 4);
                assert!(all_a || all_b, "leaf mixes clusters: {ids:?}");
            }
        }

        #[test]
        fn partition_terminates_on_identical_vectors() {
            // Every vector identical → no projection separates anything. The
            // median-split fallback must still drive recursion to leaves rather
            // than loop forever.
            let v = vec![0.0f32; 3 * 8];
            let config = TPTreeConfig {
                leaf_size: 2,
                samples: 8,
                top_dims: 2,
                iterations: 8,
            };
            let mut tpt = TPTree::new(config, 3, &v);
            let mut indices: Vec<NodeId> = (0..8).collect();

            let leaves = tpt.partition(&mut indices);

            assert!(leaves.iter().all(|l| l.len() <= 2));
            assert_eq!(leaves.iter().map(|l| l.len()).sum::<usize>(), 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line of `n` 1-D points at positions `0..n`, each connected to its ±1 and
    /// ±2 neighbors (edge "distance" = squared gap, matching `-L2` similarity).
    fn line_index(n: NodeId) -> RelativeNeighborhoodGraph<f32> {
        let params = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(n as usize, 1, Metric::L2, params);
        for i in 0..n {
            rng.add_vector(&[i as f32]);
        }
        for i in 0..n as i64 {
            for off in [-2i64, -1, 1, 2] {
                let nb = i + off;
                if (0..n as i64).contains(&nb) {
                    let d = ((i - nb) * (i - nb)) as f32;
                    rng.graph.add_edge(i as NodeId, nb as NodeId, d);
                }
            }
        }
        rng
    }

    #[test]
    fn search_finds_nearest_neighbors() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        // Query at 4.2 → nearest points are 4, 5, 3, in that order.
        let res = rng.search(&mut ws, &[4.2], &[0], 3);
        let ids: Vec<NodeId> = res.iter().map(|c| c.node).collect();
        assert_eq!(ids, vec![4, 5, 3]);
        // Similarities are returned in descending order.
        assert!(res[0].sim >= res[1].sim && res[1].sim >= res[2].sim);
    }

    #[test]
    fn search_handles_degenerate_inputs() {
        let rng = line_index(5);
        let mut ws = Workspace::new();
        assert!(rng.search(&mut ws, &[1.0], &[0], 0).is_empty()); // k == 0
        assert!(rng.search(&mut ws, &[1.0], &[], 3).is_empty()); // no seeds

        let empty: RelativeNeighborhoodGraph<f32> =
            RelativeNeighborhoodGraph::new(4, 1, Metric::L2, NeighborhoodGraphConfig::default());
        assert!(empty.search(&mut ws, &[1.0], &[0], 3).is_empty()); // empty graph
    }

    #[test]
    fn search_reuses_workspace_deterministically() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let a = rng.search(&mut ws, &[4.2], &[0], 3);
        let b = rng.search(&mut ws, &[4.2], &[0], 3);
        assert_eq!(a, b); // epoch reset means repeated queries match exactly
    }

    #[test]
    fn search_from_a_node_returns_it_then_nearest() {
        let rng = line_index(8);
        let mut ws = Workspace::new();
        let res = rng.search(&mut ws, &[4.0], &[4], 4);
        let ids: Vec<NodeId> = res.iter().map(|c| c.node).collect();
        assert_eq!(ids[0], 4); // the query point itself ranks first
        assert!(ids[1] == 3 || ids[1] == 5); // then its nearest neighbors
    }

    fn sorted_neighbors(rng: &RelativeNeighborhoodGraph<f32>, node: NodeId) -> Vec<NodeId> {
        let mut v = rng.graph.neighbors(node).to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn refine_applies_rng_occlusion() {
        // Colinear points 0,1,2. The RNG must drop the 0–2 edge: node 1 sits
        // between them, so 1 occludes 2 from 0 (and 0 from 2).
        let config = NeighborhoodGraphConfig {
            max_edges: 4, // room for both edges; RNG, not capacity, does the pruning
            ef: 4,
            num_candidates: 4,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(3, 1, Metric::L2, config);
        for i in 0..3 {
            rng.add_vector(&[i as f32]);
        }
        // Start fully connected so each node's search sees every other node.
        for i in 0..3i64 {
            for j in 0..3i64 {
                if i != j {
                    let d = ((i - j) * (i - j)) as f32;
                    rng.graph.add_edge(i as NodeId, j as NodeId, d);
                }
            }
        }

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1]); // 0–2 occluded by 1
        assert_eq!(sorted_neighbors(&rng, 2), vec![1]); // 2–0 occluded by 1
        assert_eq!(sorted_neighbors(&rng, 1), vec![0, 2]); // middle keeps both
    }

    #[test]
    fn refine_prunes_full_mesh_to_the_optimal_path_graph() {
        // The exact RNG of n equally spaced colinear points is the path graph:
        // every node keeps only its immediate ±1 neighbors; ±2 and beyond are
        // occluded by the node in between. Starting from a full mesh, `refine`
        // must recover exactly that minimal, optimal edge set — proof the
        // occlusion rule prunes everything redundant and nothing it shouldn't.
        const N: NodeId = 6;
        let config = NeighborhoodGraphConfig {
            max_edges: 8, // far more room than the answer needs
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(N as usize, 1, Metric::L2, config);
        for i in 0..N {
            rng.add_vector(&[i as f32]);
        }
        for i in 0..N as i64 {
            for j in 0..N as i64 {
                if i != j {
                    rng.graph
                        .add_edge(i as NodeId, j as NodeId, ((i - j) * (i - j)) as f32);
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

    /// Fully connect every node to every other with `-L2` edge distances.
    fn fully_connect(rng: &mut RelativeNeighborhoodGraph<f32>, pts: &[[f32; 2]]) {
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i != j {
                    let d = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                    rng.graph.add_edge(i as NodeId, j as NodeId, d);
                }
            }
        }
    }

    #[test]
    fn refine_keeps_duplicate_vector_edges() {
        // Nodes 0 and 1 are identical. The occlusion is non-strict, so the
        // duplicate — exactly as similar to 2 as 0 is — must NOT occlude the
        // 0->2 edge. A strict `<` would wipe it, leaving 0 with just [1].
        let config = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 4,
            num_candidates: 4,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(3, 2, Metric::L2, config);
        let pts = [[0.0f32, 0.0], [0.0, 0.0], [1.0, 0.0]];
        for p in &pts {
            rng.add_vector(p);
        }
        fully_connect(&mut rng, &pts);

        rng.refine(&Executor::SingleThread);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1, 2]);
    }

    #[test]
    fn refine_caps_selected_neighbors_at_max_edges() {
        // Node 0 has four neighbors in four directions at distinct distances;
        // none occlude each other, so pure RNG would keep all four. With
        // max_edges = 2 the occlusion loop must stop after the two nearest.
        let config = NeighborhoodGraphConfig {
            max_edges: 2,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(5, 2, Metric::L2, config);
        rng.add_vector(&[0.0, 0.0]); // 0: origin
        rng.add_vector(&[1.0, 0.0]); // 1: dist 1  (nearest)
        rng.add_vector(&[0.0, 2.0]); // 2: dist 4  (2nd)
        rng.add_vector(&[-3.0, 0.0]); // 3: dist 9
        rng.add_vector(&[0.0, -4.0]); // 4: dist 16
                                      // Hand-wired connected init (max_edges = 2 each) so node 0's search can
                                      // still reach all four candidates despite the tight degree.
        rng.graph.set_neighbors(0, &[1, 2]);
        rng.graph.set_neighbors(1, &[0, 3]);
        rng.graph.set_neighbors(2, &[0, 4]);
        rng.graph.set_neighbors(3, &[1, 0]);
        rng.graph.set_neighbors(4, &[2, 0]);

        rng.refine(&Executor::SingleThread);

        // Nearest two kept; the farther two dropped despite being valid RNG edges.
        assert_eq!(sorted_neighbors(&rng, 0), vec![1, 2]);
    }

    #[test]
    fn build_init_knn_seeds_reciprocal_edges() {
        // The raw KNN seam before refine: 1-D line 0..6, single tree. The whole
        // set fits in one leaf, so build_init_knn does exact brute-force KNN —
        // each node's nearest is its ±1 neighbor, and every edge is inserted both
        // ways. (build() would then refine this down to the path graph.)
        let config = NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(6, 1, Metric::L2, config);
        let vectors: Vec<f32> = (0..6).map(|i| i as f32).collect();

        rng.build_init_knn(&Executor::single_thread(), &vectors);

        for i in 0..6u32 {
            let nbrs = rng.graph.neighbors(i);
            assert!(!nbrs.is_empty(), "node {i} has no edges");
            assert!(
                nbrs[0] == i.wrapping_sub(1) || nbrs[0] == i + 1,
                "node {i}'s nearest edge {} is not adjacent",
                nbrs[0]
            );
        }
        // Reciprocity: the 0–1 edge exists in both directions.
        assert!(rng.graph.neighbors(0).contains(&1));
        assert!(rng.graph.neighbors(1).contains(&0));
    }

    #[test]
    fn build_recovers_the_path_graph() {
        // Full pipeline through the single public call: build() seeds the init KNN
        // over a colinear line and refines it internally. The exact RNG of equally
        // spaced colinear points is the path graph — the same target as
        // refine_prunes_full_mesh_to_the_optimal_path_graph, but driven end-to-end
        // by build() with no separate refine().
        const N: NodeId = 6;
        let config = NeighborhoodGraphConfig {
            max_edges: 8,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        };
        let mut rng = RelativeNeighborhoodGraph::new(N as usize, 1, Metric::L2, config);
        let vectors: Vec<f32> = (0..N).map(|i| i as f32).collect();

        rng.build(&Executor::single_thread(), &vectors);

        assert_eq!(sorted_neighbors(&rng, 0), vec![1]);
        assert_eq!(sorted_neighbors(&rng, N - 1), vec![N - 2]);
        for i in 1..N - 1 {
            assert_eq!(sorted_neighbors(&rng, i), vec![i - 1, i + 1]);
        }
    }
}

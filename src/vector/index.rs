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
//! with no copy), and one workspace can be reused across many queries.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::schema::Metric;

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
}

impl Default for NeighborhoodGraphConfig {
    fn default() -> Self {
        NeighborhoodGraphConfig {
            max_edges: 32,
            ef: 64,
            num_candidates: 256,
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
    /// Tuning knobs (`max_edges`, `ef`).
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

    /// Builds the KNN graph by inserting reciprocal edges from a candidate
    /// source. Candidate generation (per-cluster vs TPTree) is still open.
    pub fn build(&mut self) {
        todo!("reciprocal init-KNN build")
    }

    /// Refines every node against the current graph: searches from the node
    /// itself to gather a candidate pool, applies the RNG occlusion rule to
    /// reselect its edges, and rewrites them in place. This pass is what turns a
    /// raw KNN graph into an RNG.
    pub fn refine(&mut self) {
        let mut ws = Workspace::new();
        for node_id in 0..self.graph.len() as NodeId {
            let query = self.graph.payload(node_id);
            let candidates = self.search(&mut ws, query, &[node_id], self.config.num_candidates);
            self.set_neighbors(&mut ws, node_id, &candidates);
        }
    }

    /// Applies SPTAG's RNG occlusion rule to `candidates` (nearest-first), writing
    /// the survivors straight into `node`'s adjacency (at most `max_edges`) and
    /// skipping `node` itself.
    ///
    /// Everything is in similarity space (higher is better): a candidate `c` is
    /// kept unless some already-selected neighbor `r` is *strictly more* similar to
    /// `c` than `node` is — then `r` makes the direct `node -> c` edge redundant and
    /// occludes it. The comparison is non-strict (`<=`), so an `r`
    /// *exactly* as similar as `node` does not occlude — matching SPTAG and keeping
    /// duplicate vectors from wiping out a node's whole edge set.
    fn set_neighbors(&mut self, ws: &mut Workspace, node: NodeId, candidates: &[Candidate]) {
        let max_edges = self.config.max_edges;
        let selected = &mut ws.selected;
        selected.clear();
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
        self.graph.set_neighbors(node, selected);
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
    /// RNG-selected neighbor ids for the node being refined.
    selected: Vec<NodeId>,
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

        rng.refine();

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

        rng.refine();

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
        };
        let mut rng = RelativeNeighborhoodGraph::new(3, 2, Metric::L2, config);
        let pts = [[0.0f32, 0.0], [0.0, 0.0], [1.0, 0.0]];
        for p in &pts {
            rng.add_vector(p);
        }
        fully_connect(&mut rng, &pts);

        rng.refine();

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

        rng.refine();

        // Nearest two kept; the farther two dropped despite being valid RNG edges.
        assert_eq!(sorted_neighbors(&rng, 0), vec![1, 2]);
    }
}

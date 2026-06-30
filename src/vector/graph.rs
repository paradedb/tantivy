//! A generic, single-threaded relative neighborhood graph, modeled on the RNG
//! in Microsoft's [SPTAG](https://github.com/microsoft/SPTAG). For now only the
//! plain *k*-nearest-neighbor graph is implemented; RNG edge pruning comes later.
//!
//! - Node ids are dense indices straight into the backing arrays.
//! - Adjacency is one flat array: node `i` owns the contiguous, nearest-first,
//!   [`EMPTY`]-padded run `neighbors[i * max_edges ..][.. max_edges]`.
//! - Edges store only ids. Distances drive bounded top-*k* insertion at build
//!   time but aren't durable — the order is baked in and search recomputes
//!   distances against the live query. A graph reconstructed from disk
//!   ([`Graph::for_reload`]) carries no distance buffer and is filled in stored
//!   order via [`Graph::push_edge`].
//!
//! `Graph<T>` owns its vectors in one flat, `dim`-strided arena — node `i`'s
//! vector is `vectors[i * dim ..][.. dim]`, contiguous and prefetchable rather
//! than a heap-scattered allocation per node. [`payload`](Graph::payload) hands
//! back that `&[T]` slice; the graph has no notion of a metric and never
//! computes a distance itself.

/// A dense node identifier, indexing straight into the backing arrays.
pub type NodeId = u32;

/// Sentinel marking an unused neighbor slot; node ids never reach [`NodeId::MAX`].
pub const EMPTY: NodeId = NodeId::MAX;

/// A single-threaded *k*-nearest-neighbor graph over `dim`-dimensional vectors
/// of element type `T`.
///
/// See the [module docs](self) for the layout and design rationale.
pub struct Graph<T> {
    /// Maximum out-degree per node (the *k* in *k*-NN).
    max_edges: usize,
    /// Vector dimensionality; the stride of the `vectors` arena.
    dim: usize,
    /// Flat vector arena: node `i`'s vector is `vectors[i * dim ..][.. dim]`.
    /// One allocation, contiguous, indexed by node id.
    vectors: Vec<T>,
    /// Flat adjacency: node `i` owns `neighbors[i * max_edges ..][.. max_edges]`,
    /// sorted nearest-first and [`EMPTY`]-padded. The durable search structure.
    neighbors: Vec<NodeId>,
    /// Per-edge distances driving top-*k* eviction during construction. Empty
    /// for a graph reconstructed via [`for_reload`](Graph::for_reload).
    dists: Vec<f32>,
}

impl<T> Graph<T> {
    /// Creates an empty graph with room for `capacity` nodes of `dim`-dimensional
    /// vectors and up to `max_edges` neighbors each. The flat edge arrays are
    /// allocated once here, so `capacity` is a hard cap: adding more than
    /// `capacity` nodes panics.
    pub fn new(capacity: usize, dim: usize, max_edges: usize) -> Self {
        assert!(max_edges > 0, "max_edges must be non-zero");
        assert!(dim > 0, "dim must be non-zero");
        Graph {
            max_edges,
            dim,
            vectors: Vec::with_capacity(capacity * dim),
            neighbors: vec![EMPTY; capacity * max_edges],
            dists: vec![f32::INFINITY; capacity * max_edges],
        }
    }

    /// Creates a graph for reconstruction from disk: same capacity as
    /// [`new`](Graph::new) but with no distance buffer. Edges are filled in their
    /// stored, nearest-first order via [`push_edge`](Graph::push_edge);
    /// [`add_edge`](Graph::add_edge) must not be used.
    pub fn for_reload(capacity: usize, dim: usize, max_edges: usize) -> Self {
        assert!(max_edges > 0, "max_edges must be non-zero");
        assert!(dim > 0, "dim must be non-zero");
        Graph {
            max_edges,
            dim,
            vectors: Vec::with_capacity(capacity * dim),
            neighbors: vec![EMPTY; capacity * max_edges],
            dists: Vec::new(),
        }
    }

    /// Copies `vector` in as a new node and returns its id. The node starts with
    /// no edges. Panics if `vector.len() != dim` or the graph is at `capacity`.
    pub fn add_node(&mut self, vector: &[T]) -> NodeId
    where
        T: Clone,
    {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        let id = self.len() as NodeId;
        assert!(
            (id as usize + 1) * self.max_edges <= self.neighbors.len(),
            "graph is at capacity"
        );
        self.vectors.extend_from_slice(vector);
        id
    }

    /// Considers the directed edge `from -> to`, keeping it only if `from` has a
    /// free slot or `dist` beats its farthest neighbor (which is evicted) — so
    /// each node retains its closest `max_edges`, nearest-first. Only `from`'s
    /// adjacency is touched; the builder adds the reverse edge for symmetry.
    ///
    /// Re-adding an existing `to` keeps the closer distance; self-edges are
    /// ignored. Only valid on a build graph ([`new`](Graph::new)); use
    /// [`push_edge`](Graph::push_edge) on a reloaded one.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, dist: f32) {
        debug_assert_eq!(
            self.dists.len(),
            self.neighbors.len(),
            "add_edge requires the build-time distance buffer; use push_edge"
        );
        debug_assert!((from as usize) < self.len(), "from out of range");
        debug_assert!((to as usize) < self.len(), "to out of range");
        if from == to {
            return;
        }

        let k = self.max_edges;
        let start = from as usize * k;
        let end = start + k - 1;

        // Reject when the list is full and this edge is no closer than the
        // farthest neighbor.
        if dist >= self.dists[end] {
            return;
        }

        // Deduplicate: if `to` is already a neighbor, keep only the closer copy
        // and let it bubble back into sorted position.
        if let Some(off) = self.neighbors[start..start + k]
            .iter()
            .position(|&n| n == to)
        {
            let pos = start + off;
            if dist >= self.dists[pos] {
                return;
            }
            self.dists[pos] = dist;
            let mut j = pos;
            while j > start && self.dists[j - 1] > self.dists[j] {
                self.neighbors.swap(j - 1, j);
                self.dists.swap(j - 1, j);
                j -= 1;
            }
            return;
        }

        // Insertion sort: slide `dist` into place from the back, shifting larger
        // distances up and dropping whatever falls off the last slot.
        let mut j = end;
        while j > start && self.dists[j - 1] > dist {
            self.neighbors[j] = self.neighbors[j - 1];
            self.dists[j] = self.dists[j - 1];
            j -= 1;
        }
        self.neighbors[j] = to;
        self.dists[j] = dist;
    }

    /// Blindly appends `to` as `from`'s next neighbor, with no top-*k* or
    /// distance rules. For reconstructing a graph whose edges are already stored
    /// in nearest-first order. Panics if `from` already has `max_edges` neighbors.
    pub fn push_edge(&mut self, from: NodeId, to: NodeId) {
        debug_assert!((from as usize) < self.len(), "from out of range");
        let k = self.max_edges;
        let degree = self.degree(from);
        assert!(degree < k, "node already has max_edges neighbors");
        self.neighbors[from as usize * k + degree] = to;
    }

    /// Overwrites `node`'s adjacency with `neighbors` (already in the desired,
    /// nearest-first order), padding the remaining slots with [`EMPTY`]. Used by
    /// the RNG rebuild to replace a node's edge set in one shot.
    ///
    /// Does not maintain the build-time distance buffer, so it must not be
    /// interleaved with [`add_edge`](Graph::add_edge) on the same node.
    pub fn set_neighbors(&mut self, node: NodeId, neighbors: &[NodeId]) {
        let k = self.max_edges;
        assert!(neighbors.len() <= k, "too many neighbors for node");
        debug_assert!((node as usize) < self.len(), "node out of range");
        let base = node as usize * k;
        let run = &mut self.neighbors[base..base + k];
        run[..neighbors.len()].copy_from_slice(neighbors);
        run[neighbors.len()..].fill(EMPTY);
    }

    /// Borrows `node`'s vector — a contiguous `dim`-length slice of the arena.
    #[inline]
    pub fn payload(&self, node: NodeId) -> &[T] {
        let start = node as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// The number of neighbors currently recorded for `node`.
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        let base = node as usize * self.max_edges;
        self.neighbors[base..base + self.max_edges]
            .iter()
            .take_while(|&&n| n != EMPTY)
            .count()
    }

    /// Borrows `node`'s neighbor ids, nearest-first. Excludes empty slots.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let base = node as usize * self.max_edges;
        &self.neighbors[base..base + self.degree(node)]
    }

    /// The number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    /// Whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The maximum out-degree (the *k* in *k*-NN).
    #[inline]
    pub fn max_edges(&self) -> usize {
        self.max_edges
    }

    /// Iterates every node's vector in id order — node `0`, then `1`, and so on.
    /// Each item is that node's contiguous `dim`-length slice of the arena; pair
    /// with [`Iterator::enumerate`] to recover the [`NodeId`]. This is the build
    /// loop's entry point for visiting every node.
    #[inline]
    pub fn iter(&self) -> std::slice::ChunksExact<'_, T> {
        self.vectors.chunks_exact(self.dim)
    }
}

impl<'a, T> IntoIterator for &'a Graph<T> {
    type Item = &'a [T];
    type IntoIter = std::slice::ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a graph of `n` 1-dimensional nodes (vector = `[id]`), for terse
    /// edge tests that only care about topology.
    fn graph_with_nodes(n: NodeId, max_edges: usize) -> Graph<f32> {
        let mut g = Graph::new(n as usize, 1, max_edges);
        for i in 0..n {
            assert_eq!(g.add_node(&[i as f32]), i);
        }
        g
    }

    #[test]
    fn add_node_returns_dense_ids_and_vectors() {
        let mut g: Graph<f32> = Graph::new(4, 2, 8);
        assert!(g.is_empty());
        assert_eq!(g.add_node(&[1.0, 2.0]), 0);
        assert_eq!(g.add_node(&[3.0, 4.0]), 1);
        assert_eq!(g.len(), 2);
        assert_eq!(g.payload(0), &[1.0, 2.0]);
        assert_eq!(g.payload(1), &[3.0, 4.0]);
        assert_eq!(g.degree(0), 0);
        assert!(g.neighbors(0).is_empty());
    }

    #[test]
    fn edges_are_sorted_nearest_first() {
        let mut g = graph_with_nodes(5, 8);
        g.add_edge(0, 3, 0.9);
        g.add_edge(0, 1, 0.2);
        g.add_edge(0, 4, 0.5);
        g.add_edge(0, 2, 0.1);
        assert_eq!(g.neighbors(0), &[2, 1, 4, 3]);
        assert_eq!(g.degree(0), 4);
    }

    #[test]
    fn bounded_top_k_evicts_the_farthest() {
        let mut g = graph_with_nodes(5, 2);
        g.add_edge(0, 1, 0.5);
        g.add_edge(0, 2, 0.4);
        // Full now with {2:0.4, 1:0.5}. A closer edge evicts the farthest (1).
        g.add_edge(0, 3, 0.1);
        assert_eq!(g.neighbors(0), &[3, 2]);
        // A farther edge than the current max is rejected outright.
        g.add_edge(0, 4, 0.9);
        assert_eq!(g.neighbors(0), &[3, 2]);
    }

    #[test]
    fn re_adding_keeps_the_closer_distance() {
        let mut g = graph_with_nodes(4, 4);
        g.add_edge(0, 1, 0.8);
        g.add_edge(0, 2, 0.5);
        // Re-add 1 closer: it must move ahead of 2 and not duplicate.
        g.add_edge(0, 1, 0.1);
        assert_eq!(g.neighbors(0), &[1, 2]);
        // Re-add 1 farther: ignored.
        g.add_edge(0, 1, 0.9);
        assert_eq!(g.neighbors(0), &[1, 2]);
    }

    #[test]
    fn edges_are_directed_and_self_edges_ignored() {
        let mut g = graph_with_nodes(3, 4);
        g.add_edge(0, 1, 0.3);
        assert_eq!(g.neighbors(0), &[1]);
        assert!(g.neighbors(1).is_empty());
        g.add_edge(2, 2, 0.0);
        assert!(g.neighbors(2).is_empty());
    }

    #[test]
    fn iter_yields_vectors_in_node_order() {
        let mut g: Graph<f32> = Graph::new(3, 2, 4);
        g.add_node(&[1.0, 2.0]);
        g.add_node(&[3.0, 4.0]);
        g.add_node(&[5.0, 6.0]);

        let collected: Vec<&[f32]> = g.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], &[1.0, 2.0]);
        assert_eq!(collected[1], &[3.0, 4.0]);
        assert_eq!(collected[2], &[5.0, 6.0]);

        // `&graph` works as IntoIterator; enumerate recovers the node id.
        for (id, vector) in (&g).into_iter().enumerate() {
            assert_eq!(vector, g.payload(id as NodeId));
        }
    }

    #[test]
    fn for_reload_pushes_edges_in_stored_order() {
        let mut g: Graph<f32> = Graph::for_reload(4, 1, 4);
        for i in 0..4 {
            g.add_node(&[i as f32]);
        }
        // Edges arrive already nearest-first; push them blindly, no distances.
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
        // Overwriting with a SHORTER list must re-empty the freed tail slots,
        // not leave a stale id behind — the path the RNG refine relies on each
        // pass when a node's edge set shrinks.
        g.set_neighbors(0, &[4]);
        assert_eq!(g.neighbors(0), &[4]);
        assert_eq!(g.degree(0), 1);
        // The empty slice clears the adjacency entirely.
        g.set_neighbors(0, &[]);
        assert!(g.neighbors(0).is_empty());
        assert_eq!(g.degree(0), 0);
    }

    #[test]
    #[should_panic(expected = "too many neighbors")]
    fn set_neighbors_rejects_more_than_max_edges() {
        let mut g = graph_with_nodes(4, 2);
        g.set_neighbors(0, &[1, 2, 3]); // 3 > max_edges 2
    }
}

//! A generic, single-threaded *k*-nearest-neighbor graph: a flat vector arena
//! plus fixed-degree adjacency. It is the storage substrate for graph-based
//! approximate-nearest-neighbor indexes — such as the relative neighborhood
//! graph built on top of it in the sibling `index` module — and carries no edge
//! semantics of its own beyond "node `i`'s nearest neighbors, in order".
//!
//! - Node ids are dense indices straight into the backing arrays. The node set
//!   is fixed at construction: the arena's length determines the node count,
//!   and every node starts with no edges.
//! - Adjacency is one flat array: node `i` owns the contiguous, best-first
//!   (most similar first), [`EMPTY`]-padded run
//!   `neighbors[i * max_edges ..][.. max_edges]`.
//! - Edges store only ids. [`Similarity`] scores drive bounded top-*k*
//!   insertion at build time but aren't durable — the order is baked in and
//!   search rescores against the live query. A graph reconstructed from disk
//!   ([`Graph::for_reload`]) carries no similarity buffer and is filled in
//!   stored order via [`Graph::push_edge`].
//!
//! `Graph<S>` never owns vector data of its own: `S` is any [`VectorArena`] —
//! a flat, `dim`-strided arena where node `i`'s vector is
//! `vectors[i * dim ..][.. dim]`. A build borrows the clusterer's matrix
//! (`S = &[f32]`); a reload can wrap owned or file-resident storage. Scoring
//! goes through [`VectorArena::similarity`]; the graph itself has no notion
//! of a metric and only ever *compares* the [`Similarity`] values handed to it.

use std::io::{self, Write};
use std::ops::Deref;

use common::BinarySerializable;

use super::{Similarity, VectorArena};

/// A dense node identifier, indexing straight into the backing arrays.
pub type NodeId = u32;

/// Sentinel marking an unused neighbor slot; node ids never reach [`NodeId::MAX`].
pub const EMPTY: NodeId = NodeId::MAX;

/// A single-threaded *k*-nearest-neighbor graph over `dim`-dimensional vectors
/// stored in the arena `S` (any [`VectorArena`], typed or byte-backed).
///
/// See the [module docs](self) for the layout and design rationale.
pub struct Graph<S> {
    /// Maximum out-degree per node (the *k* in *k*-NN).
    max_edges: usize,
    /// Vector dimensionality; the stride of the `vectors` arena.
    dim: usize,
    /// Flat vector arena: node `i`'s vector is `vectors[i * dim ..][.. dim]`.
    /// One contiguous buffer, indexed by node id, borrowed or owned via `S`.
    vectors: S,
    /// Flat adjacency: node `i` owns `neighbors[i * max_edges ..][.. max_edges]`,
    /// sorted best-first (most similar first) and [`EMPTY`]-padded. The durable
    /// search structure.
    neighbors: Vec<NodeId>,
    /// Per-edge similarities driving top-*k* eviction during construction.
    /// Empty for a graph reconstructed via [`for_reload`](Graph::for_reload).
    sims: Vec<Similarity>,
}

impl<S: VectorArena> Graph<S> {
    /// Creates a build graph over `vectors`, a flat `dim`-strided arena whose
    /// length fixes the node count. Every node starts with no edges; the flat
    /// edge arrays are allocated here, once. Panics if `vectors` is not a
    /// multiple of `dim` long.
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

    /// Creates a graph for reconstruction from disk: same shape as
    /// [`new`](Graph::new) but with no similarity buffer. Edges are filled in
    /// their stored, best-first order via [`push_edge`](Graph::push_edge);
    /// [`add_edge`](Graph::add_edge) must not be used.
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

    /// Reconstructs a graph serialized by [`serialize`](Graph::serialize)
    /// over `vectors` — the arena is persisted separately, and its length
    /// fixes the node count the adjacency is validated against. The stored
    /// adjacency is exactly the in-memory layout (best-first, EMPTY-padded),
    /// so this is a validate-and-decode, not a rebuild. Like
    /// [`for_reload`](Graph::for_reload), the result carries no similarity
    /// buffer and is search-only.
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
        // Ids must stay below the EMPTY sentinel (NodeId::MAX).
        assert!(n < NodeId::MAX as usize, "arena exceeds NodeId space");
        n
    }

    /// Considers the directed edge `from -> to`, keeping it only if `from` has a
    /// free slot or `sim` beats its least-similar neighbor (which is evicted) —
    /// so each node retains its `max_edges` most similar, best-first. Only
    /// `from`'s adjacency is touched; the builder adds the reverse edge for
    /// symmetry.
    ///
    /// Re-adding an existing `to` keeps the more similar score; self-edges are
    /// ignored. Only valid on a build graph ([`new`](Graph::new)); use
    /// [`push_edge`](Graph::push_edge) on a reloaded one.
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

    /// Mutable views of every node's edge list, in id order. The views are
    /// disjoint, so they can be split across threads and mutated concurrently
    /// without locks. Only valid on a build graph ([`new`](Graph::new)), which
    /// has the similarity buffer.
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

    /// Blindly appends `to` as `from`'s next neighbor, with no top-*k* or
    /// similarity rules. For reconstructing a graph whose edges are already
    /// stored in best-first order. Panics if `from` already has `max_edges`
    /// neighbors.
    pub fn push_edge(&mut self, from: NodeId, to: NodeId) {
        debug_assert!((from as usize) < self.len(), "from out of range");
        let k = self.max_edges;
        let degree = self.degree(from);
        assert!(degree < k, "node already has max_edges neighbors");
        self.neighbors[from as usize * k + degree] = to;
    }

    /// Overwrites `node`'s adjacency with `neighbors` (already in the desired,
    /// best-first order), padding the remaining slots with [`EMPTY`]. Used by
    /// the RNG rebuild to replace a node's edge set in one shot.
    ///
    /// Does not maintain the build-time similarity buffer, so it must not be
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

    /// The number of neighbors currently recorded for `node`.
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        let base = node as usize * self.max_edges;
        self.neighbors[base..base + self.max_edges]
            .iter()
            .take_while(|&&n| n != EMPTY)
            .count()
    }

    /// Borrows `node`'s neighbor ids, best-first. Excludes empty slots.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let base = node as usize * self.max_edges;
        &self.neighbors[base..base + self.degree(node)]
    }

    /// The number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.num_vectors(self.dim)
    }

    /// Whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Writes the durable part of the graph — `max_edges`, then the flat
    /// adjacency exactly as held in memory — as little-endian `u32`s:
    ///
    /// ```text
    /// max_edges (u32) + neighbors (u32[len · max_edges], best-first,
    ///                              EMPTY-padded runs of max_edges per node)
    /// ```
    ///
    /// Neither the vectors nor the node count are written: the arena is
    /// persisted (and the count derived) elsewhere, and a reload wraps it via
    /// [`for_reload`](Graph::for_reload). Similarities aren't durable at all —
    /// see the [module docs](self).
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        let max_edges = u32::try_from(self.max_edges)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "max_edges exceeds u32"))?;
        max_edges.serialize(out)?;
        for &neighbor in &self.neighbors {
            neighbor.serialize(out)?;
        }
        Ok(())
    }

    /// Borrows the arena storage. For `Copy` storage like `&[T]`, dereferencing
    /// the borrow yields a reference with the *arena's* lifetime, so the TPT
    /// build can read vectors while mutating edge lists.
    #[inline]
    pub fn arena(&self) -> &S {
        &self.vectors
    }
}

/// Typed-arena views: only `[T]`-shaped storage can hand out `&[T]` borrows
/// (file bytes have no alignment guarantee).
impl<T, S: Deref<Target = [T]>> Graph<S> {
    /// Borrows `node`'s vector — a contiguous `dim`-length slice of the arena.
    #[inline]
    pub fn payload(&self, node: NodeId) -> &[T] {
        let start = node as usize * self.dim;
        &self.vectors[start..start + self.dim]
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

impl<'a, T: 'a, S: Deref<Target = [T]>> IntoIterator for &'a Graph<S> {
    type Item = &'a [T];
    type IntoIter = std::slice::ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Mutable view of one node's edge list: its neighbor-id and similarity runs,
/// best-first. Views of different nodes are disjoint, so a set of them
/// (from [`Graph::edge_lists_mut`]) can be mutated by different threads
/// without locks.
pub(crate) struct EdgeListMut<'a> {
    /// The node this list belongs to; self-edges to it are rejected.
    node: NodeId,
    neighbors: &'a mut [NodeId],
    sims: &'a mut [Similarity],
}

impl EdgeListMut<'_> {
    /// Considers the directed edge `self.node -> to` — the same bounded
    /// best-first insert as [`Graph::add_edge`], which delegates here.
    pub(crate) fn add_edge(&mut self, to: NodeId, sim: Similarity) {
        if to == self.node {
            return;
        }

        // Reject when the list is full and this edge is no more similar than
        // the least-similar neighbor. (Empty slots hold `Similarity::WORST`,
        // which any real score beats.)
        let last = self.sims.len() - 1;
        if sim <= self.sims[last] {
            return;
        }

        // Deduplicate: if `to` is already a neighbor, keep only the more
        // similar copy and let it bubble back into sorted position.
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

        // Insertion sort: slide `sim` into place from the back, shifting less
        // similar entries down and dropping whatever falls off the last slot.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a graph of `n` 1-dimensional nodes (vector = `[id]`), for terse
    /// edge tests that only care about topology.
    fn graph_with_nodes(n: NodeId, max_edges: usize) -> Graph<Vec<f32>> {
        Graph::new((0..n).map(|i| i as f32).collect(), 1, max_edges)
    }

    /// Shorthand for a raw similarity score (higher is better).
    fn sim(score: f32) -> Similarity {
        Similarity::new(score)
    }

    #[test]
    fn edge_lists_mut_allows_disjoint_parallel_writes() {
        // Two threads each own half the edge lists. The borrows are disjoint,
        // so this compiles without locks and behaves like serial add_edge.
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
    fn new_derives_nodes_from_the_arena() {
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
        // The merge-time shape: the graph borrows the caller's matrix, and
        // `arena` hands back a reference independent of the graph borrow — so
        // the vectors stay readable while edge lists are mutated, which is
        // exactly what the TPT build needs.
        let matrix: Vec<f32> = vec![0.0, 1.0, 2.0];
        let mut g: Graph<&[f32]> = Graph::new(&matrix, 1, 2);
        let vectors = *g.arena();
        g.add_edge(0, 1, sim(1.0)); // mutate while `vectors` is still borrowed
        assert_eq!(vectors, matrix.as_slice());
        assert_eq!(g.neighbors(0), &[1]);
    }

    #[test]
    #[should_panic(expected = "arena not a multiple of dim")]
    fn new_rejects_a_misaligned_arena() {
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
        // Full now with {2:0.6, 1:0.5}. A better edge evicts the worst (1).
        g.add_edge(0, 3, sim(0.9));
        assert_eq!(g.neighbors(0), &[3, 2]);
        // An edge worse than the current minimum is rejected outright.
        g.add_edge(0, 4, sim(0.1));
        assert_eq!(g.neighbors(0), &[3, 2]);
    }

    #[test]
    fn re_adding_keeps_the_more_similar_score() {
        let mut g = graph_with_nodes(4, 4);
        g.add_edge(0, 1, sim(0.2));
        g.add_edge(0, 2, sim(0.5));
        // Re-add 1 with a better score: it must move ahead of 2 and not duplicate.
        g.add_edge(0, 1, sim(0.9));
        assert_eq!(g.neighbors(0), &[1, 2]);
        // Re-add 1 with a worse score: ignored.
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

        // `&graph` works as IntoIterator; enumerate recovers the node id.
        for (id, vector) in (&g).into_iter().enumerate() {
            assert_eq!(vector, g.payload(id as NodeId));
        }
    }

    #[test]
    fn for_reload_pushes_edges_in_stored_order() {
        let arena: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let mut g = Graph::for_reload(arena, 1, 4);
        // Edges arrive already best-first; push them blindly, no similarities.
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

    /// Decodes a serialized graph back into (max_edges, neighbors) u32s.
    fn decode(bytes: &[u8]) -> (u32, Vec<NodeId>) {
        assert_eq!(bytes.len() % 4, 0, "serialization is a whole number of u32s");
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
        g.add_edge(0, 1, sim(0.9)); // more similar: sorts ahead of 2
        g.add_edge(1, 0, sim(0.9));
        // node 2 keeps an all-EMPTY run

        let mut bytes = Vec::new();
        g.serialize(&mut bytes).unwrap();

        let (max_edges, neighbors) = decode(&bytes);
        assert_eq!(max_edges, 2);
        assert_eq!(neighbors, vec![1, 2, 0, EMPTY, EMPTY, EMPTY]);
    }

    #[test]
    fn reloaded_graph_serializes_byte_identically() {
        // The durable invariant behind slot reuse across merges: serialize →
        // reload (push edges in stored order) → serialize must be a fixed
        // point, so nothing drifts however many times a graph round-trips.
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

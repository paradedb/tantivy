//! A hierarchical (HNSW-style) routing index over the centroid
//! [`RelativeNeighborhoodGraph`].
//!
//! Layer 0 is the plain RNG over every centroid — unchanged build, format,
//! and resumable beam search. On top sits a stack of RNGs over exponentially
//! sampled subsets (`~n/max_edges` nodes per step up). A query descends
//! greedily from the top layer to find good entry points, then runs the
//! base layer's resumable [`search_iter`] seeded from the descent — replacing
//! the fixed strided-seed heuristic.
//!
//! Storage stays lazily materialized like the single-layer RNG: only
//! adjacency and the per-layer id maps are decoded at open; vectors are
//! scored through the caller's [`VectorArena`] (upper layers via a
//! [`SubsetArena`] that remaps layer-local ids), so file-resident centroids
//! are never copied.
//!
//! [`search_iter`]: RelativeNeighborhoodGraph::search_iter

use std::io::{self, Write};
use std::sync::Arc;

use common::BinarySerializable;

use super::graph::{
    Candidate, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, Workspace,
};
use crate::schema::Metric;
use crate::vector::{Similarity, VectorArena};
use crate::Executor;

/// Fixed seed for level assignment: builds must be reproducible; the value
/// doesn't matter.
const LEVEL_SEED: u64 = 42;

/// Seeds handed to the base layer: the descent's best candidates, or the
/// strided fallback when there are no upper layers. Matches the historical
/// strided-seed count.
const BASE_SEED_COUNT: usize = 8;

/// Beam width for the greedy descent through upper layers. Layers are tiny,
/// so a beam a little wider than greedy-1 buys entry quality for negligible
/// cost.
const DESCENT_EF: usize = 16;

/// An upper layer smaller than this adds nothing over entering the layer
/// below directly and is dropped.
const MIN_LAYER_NODES: usize = 2;

/// A [`VectorArena`] view of a subset of another arena: layer-local node `i`
/// scores as base node `ids[i]`. This is what lets upper layers search
/// against file-resident centroids without materializing their vectors.
struct SubsetArena<S> {
    base: S,
    /// Layer-local id → base node id, strictly ascending.
    ids: Arc<[NodeId]>,
}

impl<S: VectorArena> VectorArena for SubsetArena<S> {
    type Elem = S::Elem;

    #[inline]
    fn num_vectors(&self, _dim: usize) -> usize {
        self.ids.len()
    }

    #[inline]
    fn similarity(
        &self,
        metric: Metric,
        dim: usize,
        node: NodeId,
        query: &[Self::Elem],
    ) -> Similarity {
        self.base.similarity(metric, dim, self.ids[node as usize], query)
    }
}

/// One upper layer: its sorted base-id map and the RNG over the subset.
struct UpperLayer<S> {
    /// Layer-local id → base node id, strictly ascending. Shared with the
    /// graph's [`SubsetArena`].
    ids: Arc<[NodeId]>,
    graph: RelativeNeighborhoodGraph<SubsetArena<S>>,
}

/// The hierarchical routing index. See the [module docs](self).
pub struct CentroidHnsw<S: VectorArena> {
    base: RelativeNeighborhoodGraph<S>,
    /// Bottom-up: `upper[0]` is layer 1. Every layer's id set is a subset of
    /// the layer below (validated at [`open`](Self::open)).
    upper: Vec<UpperLayer<S>>,
}

/// Upper layers share the base graph's shape knobs but search with the
/// narrow descent beam.
fn upper_config(config: NeighborhoodGraphConfig) -> NeighborhoodGraphConfig {
    NeighborhoodGraphConfig {
        ef: DESCENT_EF,
        ..config
    }
}

/// Geometric level assignment (`P(level >= l) = max_edges^-l`), returning the
/// upper layers' sorted id sets, bottom-up. Seeded, so deterministic.
fn assign_upper_layers(n: usize, max_edges: usize) -> Vec<Vec<NodeId>> {
    let ml = 1.0 / (max_edges as f64).ln();
    let mut rng = fastrand::Rng::with_seed(LEVEL_SEED);
    let mut layers: Vec<Vec<NodeId>> = Vec::new();
    for node in 0..n as NodeId {
        // In (0, 1]: keeps ln() finite.
        let u = 1.0 - rng.f64();
        let level = (-u.ln() * ml) as usize;
        for l in 1..=level {
            if layers.len() < l {
                layers.push(Vec::new());
            }
            layers[l - 1].push(node);
        }
    }
    while layers
        .last()
        .is_some_and(|layer| layer.len() < MIN_LAYER_NODES)
    {
        layers.pop();
    }
    layers
}

/// The historical seed heuristic, used when there are no upper layers:
/// [`BASE_SEED_COUNT`] evenly strided node ids.
fn strided_seeds(n: usize) -> Vec<NodeId> {
    (0..n)
        .step_by((n / BASE_SEED_COUNT).max(1))
        .take(BASE_SEED_COUNT)
        .map(|node| node as NodeId)
        .collect()
}

fn accumulate(
    total: &mut NeighborhoodGraphSearchMetrics,
    layer: NeighborhoodGraphSearchMetrics,
) {
    total.visited_count += layer.visited_count;
    total.expanded_count += layer.expanded_count;
    total.edges_scanned += layer.edges_scanned;
    total.evictions += layer.evictions;
    total.result_count += layer.result_count;
    total.termination_reason = layer.termination_reason;
}

impl CentroidHnsw<&[f32]> {
    /// Builds the hierarchy over the borrowed centroid arena: the base RNG
    /// via [`RelativeNeighborhoodGraph::build`], then one RNG per sampled
    /// upper layer (built over a transient contiguous copy of the subset —
    /// the TPT partitioner needs flat float access — and reopened over a
    /// [`SubsetArena`] borrow, so nothing vector-sized outlives the build).
    pub fn build<'a>(
        vectors: &'a [f32],
        dim: usize,
        metric: Metric,
        config: NeighborhoodGraphConfig,
        executor: &Executor,
    ) -> io::Result<CentroidHnsw<&'a [f32]>> {
        let mut base = RelativeNeighborhoodGraph::new(vectors, dim, metric, config);
        base.build(executor);

        let n = vectors.len() / dim;
        let mut upper = Vec::new();
        for ids in assign_upper_layers(n, config.max_edges) {
            let subset: Vec<f32> = ids
                .iter()
                .flat_map(|&id| vectors[id as usize * dim..][..dim].iter().copied())
                .collect();
            let mut built =
                RelativeNeighborhoodGraph::new(subset.as_slice(), dim, metric, upper_config(config));
            built.build(executor);
            let mut adjacency = Vec::new();
            built.serialize(&mut adjacency)?;

            let ids: Arc<[NodeId]> = ids.into();
            let arena = SubsetArena {
                base: vectors,
                ids: Arc::clone(&ids),
            };
            let graph = RelativeNeighborhoodGraph::open(
                &adjacency,
                arena,
                dim,
                metric,
                upper_config(config),
            )?;
            upper.push(UpperLayer { ids, graph });
        }
        Ok(CentroidHnsw { base, upper })
    }
}

impl<S: VectorArena> CentroidHnsw<S> {
    /// The number of base-layer nodes (= centroids).
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// Whether the base layer has no nodes.
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Total layer count, including the base layer.
    pub fn num_layers(&self) -> usize {
        1 + self.upper.len()
    }

    /// Greedy descent from the top layer down to layer 1: each layer is
    /// beam-searched from the entry carried down from above, and the last
    /// layer's best [`BASE_SEED_COUNT`] hits become the base layer's seeds
    /// (in base id space). Without upper layers this falls back to the
    /// strided heuristic. Descent cost is accumulated into `metrics`.
    fn base_seeds(
        &self,
        ws: &mut Workspace,
        query: &[S::Elem],
        metrics: &mut NeighborhoodGraphSearchMetrics,
    ) -> Vec<NodeId> {
        if self.upper.is_empty() {
            return strided_seeds(self.base.len());
        }
        let mut seeds_local: Vec<NodeId> = vec![0];
        for (idx, layer) in self.upper.iter().enumerate().rev() {
            let k = if idx == 0 { BASE_SEED_COUNT } else { 1 };
            let (candidates, layer_metrics) = layer.graph.search(ws, query, &seeds_local, k);
            accumulate(metrics, layer_metrics);
            let base_ids: Vec<NodeId> = candidates
                .iter()
                .map(|candidate| layer.ids[candidate.node as usize])
                .collect();
            if idx == 0 {
                return base_ids;
            }
            // Layer idx's ids are a subset of layer idx-1's (validated at
            // open), so every carried-down entry resolves.
            let below = &self.upper[idx - 1];
            seeds_local = base_ids
                .iter()
                .map(|id| {
                    below
                        .ids
                        .binary_search(id)
                        .expect("upper layer ids must be a subset of the layer below")
                        as NodeId
                })
                .collect();
        }
        unreachable!("descent returns at layer 1");
    }

    /// Resumable routing search: descends the hierarchy for entry points,
    /// then hands off to the base layer's
    /// [`search_iter`](RelativeNeighborhoodGraph::search_iter) — same pull
    /// semantics and exact arena-scored similarities as the single-layer
    /// RNG. Returns the iterator plus the descent's accumulated cost (the
    /// iterator's own [`metrics`](ResumableSearchIterator::metrics) cover
    /// only the base layer).
    pub fn search_iter<'g, 'w>(
        &'g self,
        ws: &'w mut Workspace,
        query: &'g [S::Elem],
    ) -> (
        ResumableSearchIterator<'g, 'w, S>,
        NeighborhoodGraphSearchMetrics,
    ) {
        let mut descent = NeighborhoodGraphSearchMetrics::default();
        let seeds = self.base_seeds(ws, query, &mut descent);
        (self.base.search_iter(ws, query, &seeds), descent)
    }

    /// One-shot top-`k` over the hierarchy (descent + base
    /// [`search`](RelativeNeighborhoodGraph::search)); the replica
    /// selector's entry point. Metrics cover descent and base together.
    pub fn search(
        &self,
        ws: &mut Workspace,
        query: &[S::Elem],
        k: usize,
    ) -> (Vec<Candidate>, NeighborhoodGraphSearchMetrics) {
        let mut metrics = NeighborhoodGraphSearchMetrics::default();
        let seeds = self.base_seeds(ws, query, &mut metrics);
        let (out, base_metrics) = self.base.search(ws, query, seeds.as_slice(), k);
        accumulate(&mut metrics, base_metrics);
        (out, metrics)
    }

    /// Writes the durable hierarchy:
    ///
    /// ```text
    /// num_layers (u32, >= 1)
    /// layer 0:      Graph adjacency block (see Graph::serialize; node count
    ///               comes from the centroid arena)
    /// layer l >= 1: count (u32) + ids (u32[count], strictly ascending base
    ///               node ids) + Graph adjacency block over count nodes
    /// ```
    ///
    /// As with the single-layer RNG, the metric and tuning knobs are
    /// configuration, not data.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        let num_layers = u32::try_from(self.num_layers())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "layer count exceeds u32"))?;
        num_layers.serialize(out)?;
        self.base.serialize(out)?;
        for layer in &self.upper {
            let count = u32::try_from(layer.ids.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "layer node count exceeds u32")
            })?;
            count.serialize(out)?;
            for &id in layer.ids.iter() {
                id.serialize(out)?;
            }
            layer.graph.serialize(out)?;
        }
        Ok(())
    }

    /// Opens a serialized hierarchy (see [`serialize`](Self::serialize)) over
    /// `vectors` — typically a
    /// [`FileSliceArena`](crate::vector::FileSliceArena), which upper layers
    /// share through [`SubsetArena`] views, so only adjacency and id maps are
    /// materialized. Validates each layer's ids (strictly ascending, in
    /// range, subset of the layer below) so descent id mapping is
    /// infallible. Search-only, like [`RelativeNeighborhoodGraph::open`].
    pub fn open(
        bytes: &[u8],
        vectors: S,
        dim: usize,
        metric: Metric,
        config: NeighborhoodGraphConfig,
    ) -> io::Result<Self>
    where
        S: Clone,
    {
        let invalid = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
        let mut cursor = bytes;
        let num_layers = u32::deserialize(&mut cursor)? as usize;
        if num_layers == 0 {
            return Err(invalid("serialized hierarchy has zero layers"));
        }
        let n = vectors.num_vectors(dim);
        let base_block = take_graph_block(&mut cursor, n)?;
        let base = RelativeNeighborhoodGraph::open(base_block, vectors.clone(), dim, metric, config)?;

        let mut upper: Vec<UpperLayer<S>> = Vec::with_capacity(num_layers - 1);
        for _ in 1..num_layers {
            let count = u32::deserialize(&mut cursor)? as usize;
            if count == 0 || count > n {
                return Err(invalid("upper layer node count out of range"));
            }
            let mut ids: Vec<NodeId> = Vec::with_capacity(count);
            for _ in 0..count {
                let id = NodeId::deserialize(&mut cursor)?;
                if id as usize >= n || ids.last().is_some_and(|&prev| prev >= id) {
                    return Err(invalid("upper layer ids must be strictly ascending node ids"));
                }
                if let Some(below) = upper.last() {
                    if below.ids.binary_search(&id).is_err() {
                        return Err(invalid("upper layer ids must be a subset of the layer below"));
                    }
                }
                ids.push(id);
            }
            let ids: Arc<[NodeId]> = ids.into();
            let block = take_graph_block(&mut cursor, count)?;
            let graph = RelativeNeighborhoodGraph::open(
                block,
                SubsetArena {
                    base: vectors.clone(),
                    ids: Arc::clone(&ids),
                },
                dim,
                metric,
                upper_config(config),
            )?;
            upper.push(UpperLayer { ids, graph });
        }
        if !cursor.is_empty() {
            return Err(invalid("trailing bytes after hierarchy"));
        }
        Ok(CentroidHnsw { base, upper })
    }
}

/// Splits the next [`Graph`](super::graph::Graph) adjacency block (for `n`
/// nodes) off the cursor: peeks the leading `max_edges` word to size the
/// block, which [`RelativeNeighborhoodGraph::open`] then validates in full.
fn take_graph_block<'a>(cursor: &mut &'a [u8], n: usize) -> io::Result<&'a [u8]> {
    let invalid = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
    let Some(word) = cursor.get(..4) else {
        return Err(invalid("truncated graph block header"));
    };
    let max_edges = u32::from_le_bytes(word.try_into().unwrap()) as usize;
    let size = n
        .checked_mul(max_edges)
        .and_then(|words| words.checked_mul(std::mem::size_of::<NodeId>()))
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(|| invalid("graph block size overflow"))?;
    if cursor.len() < size {
        return Err(invalid("truncated graph adjacency block"));
    }
    let (block, rest) = cursor.split_at(size);
    *cursor = rest;
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_vectors(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// `max_edges: 4` makes upper-layer sampling dense (`4^-l`), so modest
    /// node counts still grow a real hierarchy.
    fn dense_sampling_config() -> NeighborhoodGraphConfig {
        NeighborhoodGraphConfig {
            max_edges: 4,
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
        }
    }

    fn build_line(n: usize, config: NeighborhoodGraphConfig) -> (Vec<f32>, CentroidHnsw<&'static [f32]>) {
        // Leak the arena: `CentroidHnsw<&[f32]>` borrows it, and tests want
        // to move the index around freely.
        let vectors: &'static [f32] = line_vectors(n).leak();
        let hnsw = CentroidHnsw::build(vectors, 1, Metric::L2, config, &Executor::single_thread())
            .unwrap();
        (vectors.to_vec(), hnsw)
    }

    #[test]
    fn level_assignment_is_deterministic_and_nested() {
        let a = assign_upper_layers(5000, 4);
        let b = assign_upper_layers(5000, 4);
        assert_eq!(a, b);
        assert!(!a.is_empty(), "5000 nodes at 4^-l sampling must stack layers");
        // Expected layer 1 occupancy is n/4; allow generous slack.
        assert!((800..=1700).contains(&a[0].len()), "layer 1 size {}", a[0].len());
        for (below, above) in a.iter().zip(a.iter().skip(1)) {
            assert!(above.len() >= MIN_LAYER_NODES);
            assert!(above.len() < below.len());
            for id in above {
                assert!(below.binary_search(id).is_ok(), "layers must nest");
            }
        }
        for layer in &a {
            assert!(layer.windows(2).all(|w| w[0] < w[1]), "ids strictly ascending");
        }
    }

    #[test]
    fn small_build_falls_back_to_strided_seeds() {
        // 8 nodes at 32^-1 sampling: the fixed seed lifts no pair of nodes,
        // so there are no upper layers and routing must match the plain RNG
        // searched from the strided seeds.
        let config = NeighborhoodGraphConfig {
            ef: 8,
            num_candidates: 8,
            num_trees: 1,
            ..Default::default()
        };
        let (vectors, hnsw) = build_line(8, config);
        assert_eq!(hnsw.num_layers(), 1);

        let mut rng = RelativeNeighborhoodGraph::new(vectors.as_slice(), 1, Metric::L2, config);
        rng.build(&Executor::single_thread());

        let mut ws = Workspace::new();
        let (via_hnsw, _) = hnsw.search(&mut ws, &[4.2], 3);
        let (direct, _) = rng.search(&mut ws, &[4.2], &strided_seeds(8), 3);
        assert_eq!(via_hnsw, direct);
    }

    #[test]
    fn descent_routes_to_the_true_nearest() {
        let (_, hnsw) = build_line(200, dense_sampling_config());
        assert!(hnsw.num_layers() >= 2, "200 nodes at 4^-l sampling must stack layers");

        let mut ws = Workspace::new();
        for target in [3usize, 57, 100, 199] {
            let query = [target as f32 + 0.2];
            let (res, metrics) = hnsw.search(&mut ws, &query, 3);
            let ids: Vec<NodeId> = res.iter().map(|c| c.node).collect();
            assert_eq!(ids[0], target as NodeId, "query near {target}");
            assert!(
                metrics.visited_count < 200,
                "descent must beat a full scan, visited {}",
                metrics.visited_count
            );
        }
    }

    #[test]
    fn search_iter_still_drains_every_node_exactly_once() {
        let n = 200usize;
        let (_, hnsw) = build_line(n, dense_sampling_config());
        let mut ws = Workspace::new();
        let (iter, _) = hnsw.search_iter(&mut ws, &[57.2]);
        let mut ids: Vec<NodeId> = iter.map(|c| c.node).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..n as NodeId).collect::<Vec<NodeId>>());
    }

    #[test]
    fn serialized_hierarchy_round_trips() {
        let (vectors, built) = build_line(200, dense_sampling_config());
        let mut bytes = Vec::new();
        built.serialize(&mut bytes).unwrap();

        let opened = CentroidHnsw::open(
            &bytes,
            vectors.as_slice(),
            1,
            Metric::L2,
            dense_sampling_config(),
        )
        .unwrap();
        assert_eq!(opened.num_layers(), built.num_layers());

        let mut reserialized = Vec::new();
        opened.serialize(&mut reserialized).unwrap();
        assert_eq!(bytes, reserialized, "open/serialize must be lossless");

        let mut ws = Workspace::new();
        let (from_built, _) = built.search(&mut ws, &[123.4], 5);
        let (from_opened, _) = opened.search(&mut ws, &[123.4], 5);
        assert_eq!(from_built, from_opened);
    }

    #[test]
    fn build_is_deterministic() {
        let (_, a) = build_line(200, dense_sampling_config());
        let (_, b) = build_line(200, dense_sampling_config());
        let mut bytes_a = Vec::new();
        let mut bytes_b = Vec::new();
        a.serialize(&mut bytes_a).unwrap();
        b.serialize(&mut bytes_b).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn open_rejects_corrupt_hierarchies() {
        let vectors = line_vectors(3);
        let config = dense_sampling_config();
        let open =
            |bytes: &[u8]| CentroidHnsw::open(bytes, vectors.as_slice(), 1, Metric::L2, config);

        assert!(open(&[]).is_err(), "empty");
        assert!(open(&0u32.to_le_bytes()).is_err(), "zero layers");

        // A valid single-layer hierarchy over 3 nodes, max_edges 1.
        let mut valid = Vec::new();
        valid.extend_from_slice(&1u32.to_le_bytes());
        valid.extend_from_slice(&1u32.to_le_bytes());
        for neighbor in [1u32, 0, 1] {
            valid.extend_from_slice(&neighbor.to_le_bytes());
        }
        assert!(open(&valid).is_ok());

        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(trailing.len() % 4 != 0 || open(&trailing).is_err());
        let mut trailing_word = valid.clone();
        trailing_word.extend_from_slice(&0u32.to_le_bytes());
        assert!(open(&trailing_word).is_err(), "trailing bytes");

        // Two layers, with the upper layer's ids out of range / unsorted.
        let upper_layer = |ids: [u32; 2]| {
            let mut bytes = valid.clone();
            bytes[0..4].copy_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&2u32.to_le_bytes());
            for id in ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            bytes.extend_from_slice(&1u32.to_le_bytes());
            for neighbor in [1u32, 0] {
                bytes.extend_from_slice(&neighbor.to_le_bytes());
            }
            bytes
        };
        assert!(open(&upper_layer([0, 2])).is_ok(), "valid two-layer form");
        assert!(open(&upper_layer([0, 5])).is_err(), "id out of range");
        assert!(open(&upper_layer([2, 0])).is_err(), "ids must ascend");
    }
}

//! Balanced k-means tree (BKT) over IVF centroids.
//!
//! Leaves hold IVF / RNG graph [`NodeId`](super::NodeId)s for seed generation.
//! Tree topology uses this module's [`NodeId`].

use std::collections::{BinaryHeap, VecDeque};
use std::io::{self, Write};
use std::mem;
use std::ops::Deref;

use common::{BinarySerializable, HasLen};

use super::graph::Candidate;
use super::NodeId as GraphNodeId;
use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::{FileSliceArena, Similarity, VectorArena};

/// Default per-round member budget for [`BKTree::search_iter`], and the
/// initial seed count when routing through a BKT. Matches SPTAG's
/// `NumberOfInitialDynamicPivots`.
pub(crate) const DEFAULT_MAX_LEAVES: usize = 50;

/// Seeds injected when the BKT frontier outranks the RNG mid-search.
/// Matches SPTAG's `NumberOfOtherDynamicPivots`.
pub(crate) const DEFAULT_REFILL_SEEDS: usize = 4;

const NODE_INTERNAL: u8 = 0;
const NODE_LEAF: u8 = 1;

/// Index into [`BKTree::nodes`]. Distinct from graph [`NodeId`](super::NodeId).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Index into [`BKTree::nodes`].
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(value: u32) -> Self {
        NodeId(value)
    }
}

impl From<NodeId> for u32 {
    #[inline]
    fn from(value: NodeId) -> Self {
        value.0
    }
}

/// A node in a [`BKTree`].
///
/// An internal node's children are contiguous in [`BKTree::nodes`].
/// A leaf's members are contiguous in [`BKTree::members`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BKTreeNode {
    Internal {
        /// Row in [`BKTree::centers`].
        centroid_id: u32,
        /// First child index in [`BKTree::nodes`].
        children_offset: u32,
        /// Number of direct children.
        children_size: u32,
    },
    Leaf {
        /// Row in [`BKTree::centers`].
        centroid_id: u32,
        /// Start of this leaf's members in [`BKTree::members`].
        members_offset: u32,
        /// Number of members in this leaf.
        members_size: u32,
    },
}

/// Balanced k-means tree over IVF centroids.
///
/// `S` is the center store: typically `Vec<f32>` at build time, or a
/// [`FileSliceArena`](crate::vector::FileSliceArena) after reload.
#[derive(Clone, Debug)]
pub struct BKTree<S: VectorArena> {
    pub dim: usize,
    /// Distance / similarity metric. Not persisted; supply again on open.
    pub metric: Metric,
    /// Tree nodes. Root is [`NodeId(0)`] when non-empty.
    pub nodes: Vec<BKTreeNode>,
    /// IVF / RNG graph node ids. Each leaf owns a contiguous slice.
    pub members: Vec<GraphNodeId>,
    /// One center vector per tree node (`len == nodes.len() * dim`).
    pub centers: S,
}

impl<S: VectorArena> BKTree<S> {
    /// Number of tree nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the tree has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root node id ([`NodeId(0)`]). Panics in debug if the tree is empty.
    #[inline]
    pub fn root(&self) -> NodeId {
        debug_assert!(!self.nodes.is_empty());
        NodeId(0)
    }

    /// Row index of `node`'s center in [`BKTree::centers`].
    #[inline]
    fn center_row(&self, node: NodeId) -> u32 {
        match &self.nodes[node.index()] {
            BKTreeNode::Internal { centroid_id, .. } | BKTreeNode::Leaf { centroid_id, .. } => {
                *centroid_id
            }
        }
    }

    /// Best-first search yielding graph [`NodeId`](super::NodeId)s nearest the
    /// query, in approximate nearest-leaf order. Continue pulling to request
    /// more seeds; the iterator resumes from where it left off.
    pub fn search_iter<'g>(&'g self, query: &'g [S::Elem]) -> BKTreeSearchIterator<'g, S> {
        self.search_iter_n(query, DEFAULT_MAX_LEAVES)
    }

    /// Same as [`search_iter`](Self::search_iter) with a custom per-round
    /// member budget of `max_leaves`.
    pub fn search_iter_n<'g>(
        &'g self,
        query: &'g [S::Elem],
        max_leaves: usize,
    ) -> BKTreeSearchIterator<'g, S> {
        BKTreeSearchIterator::new(self, query, max_leaves.max(1))
    }
}

/// Yields graph [`NodeId`](super::NodeId)s from a [`BKTree`] search.
///
/// Produced by [`BKTree::search_iter`] / [`BKTree::search_iter_n`].
pub struct BKTreeSearchIterator<'g, S: VectorArena> {
    tree: &'g BKTree<S>,
    query: &'g [S::Elem],
    max_leaves: usize,
    frontier: BinaryHeap<Candidate<NodeId>>,
    results: VecDeque<GraphNodeId>,
}

impl<'g, S: VectorArena> BKTreeSearchIterator<'g, S> {
    fn new(tree: &'g BKTree<S>, query: &'g [S::Elem], max_leaves: usize) -> Self {
        debug_assert_eq!(query.len(), tree.dim, "query dimension mismatch");

        let mut frontier = BinaryHeap::new();
        if !tree.is_empty() {
            let root = tree.root();
            let sim = tree
                .centers
                .similarity(tree.metric, tree.dim, tree.center_row(root), query);
            frontier.push(Candidate { sim, node: root });
        }

        BKTreeSearchIterator {
            tree,
            query,
            max_leaves,
            frontier,
            results: VecDeque::new(),
        }
    }

    /// Similarity of the best unexpanded tree node, if any.
    #[inline]
    pub(crate) fn frontier_best(&self) -> Option<Similarity> {
        self.frontier.peek().map(|c| c.sim)
    }

    fn run_round(&mut self) {
        debug_assert!(self.results.is_empty());

        let tree = self.tree;
        let centers = &tree.centers;
        let dim = tree.dim;
        let metric = tree.metric;

        while let Some(candidate) = self.frontier.pop() {
            match &tree.nodes[candidate.node.index()] {
                BKTreeNode::Internal {
                    children_offset,
                    children_size,
                    ..
                } => {
                    let child_start = *children_offset as usize;
                    let child_count = *children_size as usize;
                    for child in child_start..child_start + child_count {
                        let child_id = NodeId(child as u32);
                        let sim =
                            centers.similarity(metric, dim, tree.center_row(child_id), self.query);
                        self.frontier.push(Candidate {
                            sim,
                            node: child_id,
                        });
                    }
                }
                BKTreeNode::Leaf {
                    members_offset,
                    members_size,
                    ..
                } => {
                    let start = *members_offset as usize;
                    let end = start + *members_size as usize;
                    self.results
                        .extend(tree.members[start..end].iter().copied());
                    if self.results.len() > self.max_leaves {
                        break;
                    }
                }
            }
        }
    }
}

impl<S: VectorArena> Iterator for BKTreeSearchIterator<'_, S> {
    type Item = GraphNodeId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.results.is_empty() {
            if self.frontier.is_empty() {
                return None;
            }
            self.run_round();
        }
        let node = self.results.pop_front()?;
        Some(node)
    }
}

impl<S> BKTree<S>
where
    S: VectorArena<Elem = f32> + Deref<Target = [f32]>,
{
    /// Serialize the tree for `.centroids` slot `[4]`.
    ///
    /// ```text
    /// centers_byte_offset: u64
    /// num_nodes: u32
    /// nodes… (tag u8 + fields)
    /// num_members: u32
    /// members: u32[num_members]
    /// centers: f32[num_nodes · dim]
    /// ```
    ///
    /// `dim` and `metric` are not written; pass them to the reader from field
    /// options.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        let num_nodes = self.nodes.len();
        let expected_centers = num_nodes.checked_mul(self.dim).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "BKT center length overflow")
        })?;
        if self.centers.len() != expected_centers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BKT centers length {} != num_nodes ({num_nodes}) * dim ({})",
                    self.centers.len(),
                    self.dim
                ),
            ));
        }
        if self.centers.num_vectors(self.dim) != num_nodes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT centers arena row count does not match num_nodes",
            ));
        }

        let mut topology = Vec::new();
        self.serialize_topology(&mut topology)?;
        let centers_byte_offset = (mem::size_of::<u64>() + topology.len()) as u64;

        centers_byte_offset.serialize(out)?;
        out.write_all(&topology)?;
        for &value in self.centers.iter() {
            value.serialize(out)?;
        }
        Ok(())
    }

    fn serialize_topology<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        if self.nodes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot serialize an empty BKTree",
            ));
        }

        u32::try_from(self.nodes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "BKT node count exceeds u32"))?
            .serialize(out)?;

        for node in &self.nodes {
            match node {
                BKTreeNode::Internal {
                    centroid_id,
                    children_offset,
                    children_size,
                } => {
                    NODE_INTERNAL.serialize(out)?;
                    centroid_id.serialize(out)?;
                    children_offset.serialize(out)?;
                    children_size.serialize(out)?;
                    let start = *children_offset as usize;
                    let end = start.checked_add(*children_size as usize).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "BKT children range overflow")
                    })?;
                    if end > self.nodes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "BKT children range out of bounds",
                        ));
                    }
                }
                BKTreeNode::Leaf {
                    centroid_id,
                    members_offset,
                    members_size,
                } => {
                    NODE_LEAF.serialize(out)?;
                    centroid_id.serialize(out)?;
                    members_offset.serialize(out)?;
                    members_size.serialize(out)?;
                    let start = *members_offset as usize;
                    let end = start.checked_add(*members_size as usize).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "BKT members range overflow")
                    })?;
                    if end > self.members.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "BKT members range out of bounds",
                        ));
                    }
                }
            }
        }

        u32::try_from(self.members.len())
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "BKT member count exceeds u32")
            })?
            .serialize(out)?;
        for &member in &self.members {
            member.serialize(out)?;
        }
        Ok(())
    }
}

fn deserialize_nodes_and_members(
    mut bytes: &[u8],
) -> io::Result<(Vec<BKTreeNode>, Vec<GraphNodeId>)> {
    let num_nodes = u32::deserialize(&mut bytes)? as usize;
    if num_nodes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "BKT has zero nodes",
        ));
    }

    let mut nodes = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let tag = u8::deserialize(&mut bytes)?;
        let centroid_id = u32::deserialize(&mut bytes)?;
        match tag {
            NODE_INTERNAL => {
                let children_offset = u32::deserialize(&mut bytes)?;
                let children_size = u32::deserialize(&mut bytes)?;
                nodes.push(BKTreeNode::Internal {
                    centroid_id,
                    children_offset,
                    children_size,
                });
            }
            NODE_LEAF => {
                let members_offset = u32::deserialize(&mut bytes)?;
                let members_size = u32::deserialize(&mut bytes)?;
                nodes.push(BKTreeNode::Leaf {
                    centroid_id,
                    members_offset,
                    members_size,
                });
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown BKT node tag {tag}"),
                ));
            }
        }
    }

    let num_members = u32::deserialize(&mut bytes)? as usize;
    let mut members = Vec::with_capacity(num_members);
    for _ in 0..num_members {
        members.push(u32::deserialize(&mut bytes)?);
    }
    if !bytes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "BKT topology has trailing bytes before centers",
        ));
    }

    for node in &nodes {
        match node {
            BKTreeNode::Internal {
                children_offset,
                children_size,
                ..
            } => {
                let start = *children_offset as usize;
                let end = start.saturating_add(*children_size as usize);
                if end > nodes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "BKT children range out of bounds",
                    ));
                }
            }
            BKTreeNode::Leaf {
                members_offset,
                members_size,
                ..
            } => {
                let start = *members_offset as usize;
                let end = start.saturating_add(*members_size as usize);
                if end > members.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "BKT members range out of bounds",
                    ));
                }
            }
        }
    }

    Ok((nodes, members))
}

fn expected_center_bytes(num_nodes: usize, dim: usize) -> io::Result<usize> {
    num_nodes
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(mem::size_of::<f32>()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "BKT center length overflow"))
}

impl BKTree<FileSliceArena<f32>> {
    /// Open a serialized BKT (`.centroids` slot `[4]`).
    ///
    /// Pins nodes and members; centers stay behind a [`FileSliceArena`] for
    /// lazy per-row reads. `dim` and `metric` must match the field options
    /// used at write time.
    pub fn open(slice: FileSlice, dim: usize, metric: Metric) -> io::Result<Self> {
        let offset_len = mem::size_of::<u64>();
        if slice.len() < offset_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT payload shorter than centers_byte_offset",
            ));
        }
        let header = slice.slice_to(offset_len).read_bytes()?;
        let mut header_reader = header.as_slice();
        let centers_byte_offset = u64::deserialize(&mut header_reader)? as usize;
        if centers_byte_offset > slice.len() || centers_byte_offset < offset_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT centers_byte_offset out of range",
            ));
        }

        let topology = slice.slice(offset_len..centers_byte_offset).read_bytes()?;
        let (nodes, members) = deserialize_nodes_and_members(&topology)?;

        let centers_slice = slice.slice_from(centers_byte_offset);
        let expected = expected_center_bytes(nodes.len(), dim)?;
        if centers_slice.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BKT centers blob is {} bytes, expected {expected}",
                    centers_slice.len()
                ),
            ));
        }

        Ok(BKTree {
            dim,
            metric,
            nodes,
            members,
            centers: FileSliceArena::new(centers_slice),
        })
    }
}

impl BKTree<Vec<f32>> {
    /// Deserialize a payload from [`BKTree::serialize`] into an owned tree.
    ///
    /// `dim` and `metric` must match the field options used at write time.
    pub fn deserialize_owned(bytes: &[u8], dim: usize, metric: Metric) -> io::Result<Self> {
        if bytes.len() < mem::size_of::<u64>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT payload shorter than centers_byte_offset",
            ));
        }
        let mut cursor = bytes;
        let centers_byte_offset = u64::deserialize(&mut cursor)? as usize;
        if centers_byte_offset > bytes.len() || centers_byte_offset < mem::size_of::<u64>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT centers_byte_offset out of range",
            ));
        }
        let (nodes, members) =
            deserialize_nodes_and_members(&bytes[mem::size_of::<u64>()..centers_byte_offset])?;
        let centers_bytes = &bytes[centers_byte_offset..];

        let expected = expected_center_bytes(nodes.len(), dim)?;
        if centers_bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BKT centers blob is {} bytes, expected {expected}",
                    centers_bytes.len()
                ),
            ));
        }
        let mut center_cursor = centers_bytes;
        let mut centers = Vec::with_capacity(nodes.len() * dim);
        for _ in 0..nodes.len() * dim {
            centers.push(f32::deserialize(&mut center_cursor)?);
        }

        Ok(BKTree {
            dim,
            metric,
            nodes,
            members,
            centers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;

    /// Tiny tree: root internal → two leaves, four IVF member ids.
    ///
    /// Centers: root at origin, leaf 1 at `(1,0)`, leaf 2 at `(0,1)`.
    fn sample_tree() -> BKTree<Vec<f32>> {
        let dim = 2;
        BKTree {
            dim,
            metric: Metric::L2,
            nodes: vec![
                BKTreeNode::Internal {
                    centroid_id: 0,
                    children_offset: 1,
                    children_size: 2,
                },
                BKTreeNode::Leaf {
                    centroid_id: 1,
                    members_offset: 0,
                    members_size: 2,
                },
                BKTreeNode::Leaf {
                    centroid_id: 2,
                    members_offset: 2,
                    members_size: 2,
                },
            ],
            members: vec![10, 11, 20, 21],
            // 3 nodes × 2 dims
            centers: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn serialize_round_trips_owned_tree() {
        let tree = sample_tree();
        let mut bytes = Vec::new();
        tree.serialize(&mut bytes).unwrap();
        let decoded = BKTree::deserialize_owned(&bytes, tree.dim, tree.metric).unwrap();
        assert_eq!(decoded.root(), NodeId(0));
        assert_eq!(decoded.dim, tree.dim);
        assert_eq!(decoded.metric, tree.metric);
        assert_eq!(decoded.nodes, tree.nodes);
        assert_eq!(decoded.members, tree.members);
        assert_eq!(decoded.centers, tree.centers);
    }

    #[test]
    fn open_round_trips_via_file_slice() {
        let tree = sample_tree();
        let mut bytes = Vec::new();
        tree.serialize(&mut bytes).unwrap();
        let opened = BKTree::open(FileSlice::from(bytes), tree.dim, tree.metric).unwrap();
        assert_eq!(opened.nodes, tree.nodes);
        assert_eq!(opened.members, tree.members);
        assert_eq!(opened.dim, tree.dim);
        let mut members: Vec<_> = opened.search_iter(&[1.0, 0.0]).collect();
        members.sort_unstable();
        assert_eq!(members, vec![10, 11, 20, 21]);
    }

    #[test]
    fn centers_byte_offset_points_at_center_tail() {
        let tree = sample_tree();
        let mut bytes = Vec::new();
        tree.serialize(&mut bytes).unwrap();

        let mut cursor = bytes.as_slice();
        let centers_byte_offset = u64::deserialize(&mut cursor).unwrap() as usize;
        let center_bytes = &bytes[centers_byte_offset..];
        let expected = tree.nodes.len() * tree.dim * mem::size_of::<f32>();
        assert_eq!(center_bytes.len(), expected);
        assert_eq!(
            centers_byte_offset,
            bytes.len() - expected,
            "centers must be a trailing contiguous blob"
        );
    }

    #[test]
    fn search_iter_yields_graph_member_ids() {
        let tree = sample_tree();
        // Query near leaf 1's center `(1, 0)` → members 10, 11 first.
        let first = tree
            .search_iter(&[1.0, 0.0])
            .next()
            .expect("expected a graph member");
        assert!(first == 10 || first == 11);
    }

    #[test]
    fn search_iter_yields_each_member_exactly_once() {
        let tree = sample_tree();
        let mut members: Vec<GraphNodeId> = tree.search_iter(&[0.5, 0.5]).collect();
        members.sort_unstable();
        assert_eq!(members, vec![10, 11, 20, 21]);
    }

    #[test]
    fn search_iter_empty_tree_yields_nothing() {
        let tree = BKTree {
            dim: 1,
            metric: Metric::L2,
            nodes: vec![],
            members: vec![],
            centers: vec![],
        };
        assert_eq!(tree.search_iter(&[0.0]).next(), None);
    }

    #[test]
    fn search_iter_n_respects_result_budget() {
        let tree = sample_tree();
        // max_leaves=1 → stop once results.len() > 1 (first leaf has 2 members).
        let first_round: Vec<_> = tree.search_iter_n(&[1.0, 0.0], 1).take(2).collect();
        assert_eq!(first_round.len(), 2);
        assert!(first_round.iter().all(|&n| n == 10 || n == 11));
    }

    #[test]
    fn search_iter_take_preserves_frontier_across_batches() {
        let tree = sample_tree();
        let mut iter = tree.search_iter_n(&[1.0, 0.0], 1);
        assert!(iter.frontier_best().is_some());
        let first: Vec<_> = (&mut iter).take(2).collect();
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|&n| n == 10 || n == 11));
        // A second batch should still be available from the remaining frontier.
        assert!(iter.frontier_best().is_some());
        let second: Vec<_> = (&mut iter).take(2).collect();
        assert_eq!(second.len(), 2);
        assert!(second.iter().all(|&n| n == 20 || n == 21));
    }
}

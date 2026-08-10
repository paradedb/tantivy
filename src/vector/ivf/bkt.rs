//! Balanced k-means tree (BKT) over IVF centroids — topology for seed
//! generation into the centroid RNG.
//!
//! Build-time trees own center rows (`BKTree<Vec<f32>>`). Query-time reload
//! (deferred) pins nodes/members and wraps the trailing center blob in a
//! [`FileSliceArena`](crate::vector::FileSliceArena), matching the RNG.
use std::io::{self, Write};
use std::mem;
use std::ops::Deref;

use common::BinarySerializable;

use super::NodeId;
use crate::schema::Metric;
use crate::vector::VectorArena;

const NODE_INTERNAL: u8 = 0;
const NODE_LEAF: u8 = 1;

/// One node in a [`BKTree`].
///
/// Direct children of an internal node are contiguous in [`BKTree::nodes`].
/// Leaf members are contiguous in [`BKTree::members`] and name IVF / RNG
/// [`NodeId`]s (centroid indices into `.centroids` slot `[0]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BKTreeNode {
    Internal {
        /// Row in the centers arena.
        centroid_id: u32,
        /// Index of the first child in [`BKTree::nodes`].
        children_offset: u32,
        /// Number of direct children.
        children_size: u32,
    },
    Leaf {
        /// Row in the centers arena (leaf mean / center).
        centroid_id: u32,
        /// Start of this leaf's members in [`BKTree::members`].
        members_offset: u32,
        /// Number of IVF centroid ids owned by this leaf.
        members_size: u32,
    },
}

/// Balanced k-means tree over IVF centroids.
///
/// `S` is the center store: owned / borrowed floats at build time, a
/// [`FileSliceArena`](crate::vector::FileSliceArena) after reload.
#[derive(Clone, Debug)]
pub struct BKTree<S: VectorArena> {
    pub dim: usize,
    /// Scoring family; not persisted — supplied again at open from field options.
    pub metric: Metric,
    /// Tree nodes; index `0` is the root.
    pub nodes: Vec<BKTreeNode>,
    /// IVF / RNG node ids; each leaf owns a contiguous slice.
    pub members: Vec<NodeId>,
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

    /// Root node id — always `0` while the tree is non-empty.
    #[inline]
    pub fn root(&self) -> NodeId {
        debug_assert!(!self.nodes.is_empty());
        0
    }

    /// Contiguous member slice for a leaf, or empty for an internal node.
    pub fn leaf_members(&self, node: &BKTreeNode) -> &[NodeId] {
        match node {
            BKTreeNode::Leaf {
                members_offset,
                members_size,
                ..
            } => {
                let start = *members_offset as usize;
                let end = start + *members_size as usize;
                &self.members[start..end]
            }
            BKTreeNode::Internal { .. } => &[],
        }
    }
}

impl<S> BKTree<S>
where
    S: VectorArena<Elem = f32> + Deref<Target = [f32]>,
{
    /// Writes the durable BKT payload for `.centroids` slot `[4]`:
    ///
    /// ```text
    /// centers_byte_offset: u64
    /// num_nodes: u32
    /// nodes… (tag u8 + fields)
    /// num_members: u32
    /// members: u32[num_members]
    /// centers: f32[num_nodes · dim]   // trailing blob for FileSliceArena
    /// ```
    ///
    /// The root is implicit (`nodes[0]`). Metric and `dim` are not written;
    /// the reader takes them from [`VectorOptions`](crate::schema::VectorOptions).
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

impl BKTree<Vec<f32>> {
    /// Decode a payload produced by [`BKTree::serialize`] into an owned tree.
    ///
    /// `dim` and `metric` are not on the wire (same as the RNG); the caller
    /// supplies the field options. Used for write-side round-trip tests and as
    /// a reference for the future `FileSlice` open path.
    pub fn deserialize_owned(bytes: &[u8], dim: usize, metric: Metric) -> io::Result<Self> {
        if bytes.len() < mem::size_of::<u64>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT payload shorter than centers_byte_offset",
            ));
        }
        let mut cursor = bytes;
        let centers_byte_offset = u64::deserialize(&mut cursor)? as usize;
        if centers_byte_offset > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT centers_byte_offset past end of payload",
            ));
        }
        let topology_end = centers_byte_offset;
        let topology_bytes = &bytes[mem::size_of::<u64>()..topology_end];
        let centers_bytes = &bytes[centers_byte_offset..];

        let mut topo = topology_bytes;
        let num_nodes = u32::deserialize(&mut topo)? as usize;
        if num_nodes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT has zero nodes",
            ));
        }

        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let tag = u8::deserialize(&mut topo)?;
            let centroid_id = u32::deserialize(&mut topo)?;
            match tag {
                NODE_INTERNAL => {
                    let children_offset = u32::deserialize(&mut topo)?;
                    let children_size = u32::deserialize(&mut topo)?;
                    nodes.push(BKTreeNode::Internal {
                        centroid_id,
                        children_offset,
                        children_size,
                    });
                }
                NODE_LEAF => {
                    let members_offset = u32::deserialize(&mut topo)?;
                    let members_size = u32::deserialize(&mut topo)?;
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

        let num_members = u32::deserialize(&mut topo)? as usize;
        let mut members = Vec::with_capacity(num_members);
        for _ in 0..num_members {
            members.push(u32::deserialize(&mut topo)?);
        }
        if !topo.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BKT topology has trailing bytes before centers",
            ));
        }

        let expected_centers = num_nodes.checked_mul(dim).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "BKT center length overflow")
        })?;
        let expected_center_bytes = expected_centers
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "BKT center byte length overflow",
                )
            })?;
        if centers_bytes.len() != expected_center_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BKT centers blob is {} bytes, expected {expected_center_bytes}",
                    centers_bytes.len()
                ),
            ));
        }
        let mut center_cursor = centers_bytes;
        let mut centers = Vec::with_capacity(expected_centers);
        for _ in 0..expected_centers {
            centers.push(f32::deserialize(&mut center_cursor)?);
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
        assert_eq!(decoded.root(), 0);
        assert_eq!(decoded.dim, tree.dim);
        assert_eq!(decoded.metric, tree.metric);
        assert_eq!(decoded.nodes, tree.nodes);
        assert_eq!(decoded.members, tree.members);
        assert_eq!(decoded.centers, tree.centers);
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
    fn leaf_members_slices_match_offsets() {
        let tree = sample_tree();
        assert_eq!(tree.leaf_members(&tree.nodes[1]), &[10, 11]);
        assert_eq!(tree.leaf_members(&tree.nodes[2]), &[20, 21]);
        assert!(tree.leaf_members(&tree.nodes[0]).is_empty());
    }
}

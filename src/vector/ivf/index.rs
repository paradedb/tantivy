//! The `.centroids` file and its reader, [`IvfIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`IvfIndex::open`] that parses them
//! back sit side by side.
//!
//! Written per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `Explicit`). A [`CompositeFile`](crate::directory::CompositeFile) with
//! three slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride)
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] RNG over the centroids (see `Graph::serialize` for the layout;
//!     absent for degenerate centroid counts — routing then falls back to a
//!     linear scan of the centroids)
//! ```
//!
//! One dense `centroid_id = 0..N` indexes all three: `cluster_offsets[c]` is
//! the first row of cluster `c` in the parallel `.vec` rows/`IdMap`, and graph
//! node `c` is centroid `c` (its vector is row `c` of slot `[0]`, which is why
//! the graph slot stores no vectors of its own).

use std::cmp::Ordering;
use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::graph::{
    evenly_spaced_seeds, NeighborhoodGraphConfig, NodeId, RelativeNeighborhoodGraph, Workspace,
};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::{FileSliceArena, VectorArena};

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and
/// the RNG adjacency (edges only, ~`k × max_edges × 4` bytes). The centroid
/// vectors stay behind a [`FileSliceArena`] and are fetched one node at a
/// time as routing visits them. Everything row-scale stays out entirely (the
/// rows and id-map live on
/// [`VectorIndexReader`](crate::vector::VectorIndexReader)).
pub struct IvfIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field — the segment's logical
    /// vector count, written at merge time. Rows including replicas are
    /// [`Self::num_rows`].
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words), deferred:
    /// routing fetches per-node ranges rather than materializing
    /// `num_centroids × stride` bytes.
    centroids_slice: FileSlice,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    dim: usize,
    metric: Metric,
    /// The persisted RNG over the centroids (slot `[2]`), reloaded
    /// search-only over the lazy centroid arena. `None` for degenerate
    /// centroid counts, where the write side skips the slot and routing
    /// falls back to a linear scan of the (few) centroids.
    graph: Option<RelativeNeighborhoodGraph<FileSliceArena<f32>>>,
    /// Evenly spaced routing entry points — the same formula the write-side
    /// replica selector searches with.
    seeds: Vec<NodeId>,
}

impl IvfIndex {
    /// Write slot `[0]` (num_centroids + num_docs + centroid bytes) of the
    /// `.centroids` composite for a field. `num_docs` is the number of
    /// distinct docs assigned — NOT the posting-row total, which replication
    /// can multiply and which slot `[1]`'s offsets already encode.
    pub(crate) fn serialize_centroids<W: Write + ?Sized>(
        num_centroids: usize,
        num_docs: usize,
        centroid_bytes: &[u8],
        options: &VectorOptions,
        out: &mut W,
    ) -> io::Result<()> {
        let expected = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if centroid_bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IVF centroid byte length",
            ));
        }
        u32::try_from(num_centroids)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32"))?
            .serialize(out)?;
        u32::try_from(num_docs)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "doc count exceeds u32"))?
            .serialize(out)?;
        out.write_all(centroid_bytes)
    }

    /// Write slot `[1]` (cluster offsets prefix sum) of the `.centroids`
    /// composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Parse a field's `.centroids` slots. Only the count words, the offsets,
    /// and the graph adjacency are materialized; the centroid rows stay
    /// behind a [`FileSlice`] for lazy per-node reads.
    pub(crate) fn open(
        options: &VectorOptions,
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        graph_slice: Option<FileSlice>,
    ) -> crate::Result<Self> {
        let count_words = 2 * mem::size_of::<u32>();
        if centroids_slice.len() < count_words {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroids slot is smaller than its count words",
            )
            .into());
        }
        let header = centroids_slice.slice_to(count_words).read_bytes()?;
        let mut reader = header.as_slice();
        let num_centroids = u32::deserialize(&mut reader)? as usize;
        let num_docs = u32::deserialize(&mut reader)? as usize;
        let centroid_len = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        let centroids_slice = centroids_slice.slice_from(count_words);
        if centroids_slice.len() != centroid_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroid byte length mismatch",
            )
            .into());
        }

        let cluster_offsets = offsets_slice.read_bytes()?;
        let expected_offsets = (num_centroids + 1)
            .checked_mul(mem::size_of::<u64>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "cluster offset length overflow")
            })?;
        if cluster_offsets.len() != expected_offsets {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF cluster offset byte length mismatch",
            )
            .into());
        }

        let graph = match graph_slice {
            Some(slice) => {
                let vectors = match options.dtype() {
                    VectorDType::F32 => FileSliceArena::<f32>::new(centroids_slice.clone()),
                };
                // The adjacency is pinned; the centroid rows behind the
                // arena are not. Node count and adjacency length are
                // cross-validated against the arena inside `Graph::open`.
                let adjacency = slice.read_bytes()?;
                Some(RelativeNeighborhoodGraph::open(
                    &adjacency,
                    vectors,
                    options.dim(),
                    options.metric(),
                    NeighborhoodGraphConfig::default(),
                )?)
            }
            None => None,
        };

        let index = IvfIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            cluster_offsets,
            dim: options.dim(),
            metric: options.metric(),
            graph,
            seeds: evenly_spaced_seeds(num_centroids),
        };
        // Every distinct doc owns at least its primary row, so a doc count
        // above the row total means a corrupt (or differently-framed) file.
        if index.num_docs > index.num_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF doc count exceeds the posting-row total",
            )
            .into());
        }
        Ok(index)
    }

    pub fn num_clusters(&self) -> usize {
        self.num_centroids
    }

    /// Distinct docs with a vector (the persisted count; replication inflates
    /// the row total, [`Self::num_rows`]).
    pub(crate) fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Total posting rows across all clusters — memberships, counting a
    /// replicated doc once per cell it lives in.
    pub fn num_rows(&self) -> usize {
        self.cluster_offset(self.num_centroids) as usize
    }

    fn cluster_offset(&self, cluster: usize) -> u64 {
        let start = cluster * mem::size_of::<u64>();
        let end = start + mem::size_of::<u64>();
        u64::from_le_bytes(self.cluster_offsets[start..end].try_into().unwrap())
    }

    /// The contiguous row range of `cluster` within the `.vec` rows.
    #[inline]
    pub fn cluster_range(&self, cluster: usize) -> Range<usize> {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        self.cluster_offset(cluster) as usize..self.cluster_offset(cluster + 1) as usize
    }

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_centroids).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests. The search path never calls this; routing fetches per-node
    /// ranges through the lazy arena instead.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    /// Clusters to probe for `query`, best routing score first, as
    /// `(score, cluster)` pairs, plus the number of centroids scored to
    /// produce them (the navigation cost, surfaced as
    /// `ProbeStats::centroids_ranked`).
    ///
    /// With a persisted RNG this is a beam search
    /// ([`RelativeNeighborhoodGraph::search`]) with a beam of
    /// `max(ef, limit)`, returning at most `limit` clusters. Without one
    /// (degenerate centroid counts) every centroid is scored exactly. Both
    /// paths fetch centroid rows lazily through the same [`FileSliceArena`]
    /// and score with the same kernels, so their rankings agree.
    pub(crate) fn rank_clusters(&self, query: &[f32], limit: usize) -> (Vec<(f32, u32)>, usize) {
        match &self.graph {
            Some(graph) => {
                let mut ws = Workspace::new();
                let candidates = graph.search(&mut ws, query, &self.seeds, limit);
                let ranked = candidates
                    .into_iter()
                    .map(|candidate| (candidate.sim.score(), candidate.node))
                    .collect();
                (ranked, ws.num_visited())
            }
            None => {
                let arena = FileSliceArena::<f32>::new(self.centroids_slice.clone());
                let mut ranked: Vec<(f32, u32)> = (0..self.num_centroids)
                    .map(|cluster| {
                        let sim = arena.similarity(self.metric, self.dim, cluster as NodeId, query);
                        (sim.score(), cluster as u32)
                    })
                    .collect();
                ranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                ranked.truncate(limit);
                (ranked, self.num_centroids)
            }
        }
    }
}

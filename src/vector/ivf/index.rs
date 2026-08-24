//! The `.centroids` file and its reader, [`IvfIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`IvfIndex::open`] that parses them
//! back sit side by side.
//!
//! The on-disk file is a 4-byte format-version stamp (see `vector::header`)
//! followed by a [`CompositeFile`](crate::directory::CompositeFile). Written
//! per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `Explicit`). The composite has five slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride)
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] RNG over the centroids (see `Graph::serialize` for the layout);
//!     OPTIONAL — when present, a beam search refines BKT ranking; ignored
//!     when no BKT is present
//! [3] centroid bounds, REQUIRED: a segment-level BoundKind byte, then
//!     N · stride(kind) f32s in cluster order — for Ball, one f32 per
//!     cluster: max ||x - c|| over the cluster's NATIVE members' stored
//!     rows against the stored centroid (the merge documents the
//!     metric-uniform fold; replica spill is excluded per the stored
//!     `bounds_scope = native`)
//! [4] BKT over the centroids (see `BKTree::serialize`); OPTIONAL — absence
//!     means routing falls back to a linear scan of the centroids
//! ```
//!
//! Slot presence is the compatibility mechanism WITHIN a generation: the
//! composite footer maps `(field, slot)` to ranges, so a reader probes an
//! optional slot and an older segment simply lacks it. Slots `[2]` and `[4]`
//! work that way. Slot `[3]` does not, which is why it costs a generation:
//! absence would have to mean "no bounds", and a silently absent bound is
//! indistinguishable from a zero one. So `.centroids` stamps `V2`, a
//! pre-V2 file is refused at open with a REINDEX message, and a V2 file
//! missing the slot is corrupt rather than old.
use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::bkt::{BKTree, BKTreeSearchIterator, DEFAULT_MAX_LEAVES, DEFAULT_REFILL_SEEDS};
use super::encode_vector;
use super::graph::{
    Candidate, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, Workspace,
};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::{BoundKind, BoundStore, FileSliceArena, VectorArena};

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets,
/// RNG adjacency, and BKT topology. Centroid / BKT center vectors stay behind
/// [`FileSliceArena`]s and are fetched one row at a time. Everything
/// row-scale (the rows and id-map) lives on
/// [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub struct IvfIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field. Rows including
    /// replicas are [`Self::num_rows`].
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words).
    centroids_slice: FileSlice,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    dim: usize,
    metric: Metric,
    /// Balanced k-means tree over the centroids (slot `[4]`). `None` falls
    /// back to an exact scan of the centroids.
    bkt: Option<BKTree<FileSliceArena<f32>>>,
    /// The persisted RNG over the centroids (slot `[2]`). When present with
    /// a BKT, a beam search refines BKT ranking; ignored without a BKT.
    graph: Option<RelativeNeighborhoodGraph<FileSliceArena<f32>>>,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
}

impl IvfIndex {
    /// Write slot `[0]` of the `.centroids` composite for a field. `num_docs`
    /// is the number of distinct docs assigned — NOT the posting-row total,
    /// which replication can multiply.
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

    /// Write slot `[1]` of the `.centroids` composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Write slot `[3]` of the `.centroids` composite for a field: the
    /// segment-level kind byte, then the per-cluster payload.
    ///
    /// * `kind` (`BoundKind`) — the segment-level bound kind.
    /// * `values` (`&[f32]`) — `num_centroids * kind.stride(dim)` values in cluster order; the
    ///   caller's [`BoundsBuilder`] output.
    /// * `out` (`&mut W`) — the slot writer.
    ///
    /// Returns (`io::Result<()>`): write errors only — the payload length
    /// is validated at open, against the count words of slot `[0]`.
    ///
    /// [`BoundsBuilder`]: crate::vector::BoundsBuilder
    pub(crate) fn serialize_bounds<W: Write + ?Sized>(
        kind: BoundKind,
        values: &[f32],
        out: &mut W,
    ) -> io::Result<()> {
        (kind as u8).serialize(out)?;
        for value in values {
            value.serialize(out)?;
        }
        Ok(())
    }

    /// In-memory routing index over a centroid matrix, for ranking quality
    /// tests and benches.
    ///
    /// `centroids` is row-major `n × dim`. Posting offsets and bounds are
    /// dummies — [`rank_clusters`](Self::rank_clusters) never reads them.
    /// Pass `bkt` / `graph` to select the routing mode: neither is exact
    /// scan; `bkt` alone is BKT ranking; both is BKT seeded with RNG refill.
    pub fn from_centroids(
        options: &VectorOptions,
        centroids: &[f32],
        bkt: Option<&BKTree<Vec<f32>>>,
        graph: Option<&RelativeNeighborhoodGraph<Vec<f32>>>,
    ) -> crate::Result<Self> {
        let dim = options.dim();
        if dim == 0 || centroids.len() % dim != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "centroid matrix is not a multiple of dim",
            )
            .into());
        }
        let n = centroids.len() / dim;
        let mut centroid_bytes = Vec::with_capacity(n * options.bytes_per_vector());
        for row in centroids.chunks_exact(dim) {
            centroid_bytes.extend(encode_vector(row, dim)?);
        }
        let mut centroid_slot = Vec::new();
        Self::serialize_centroids(n, n, &centroid_bytes, options, &mut centroid_slot)?;
        let offsets: Vec<u64> = (0..=n as u64).collect();
        let mut offset_slot = Vec::new();
        Self::serialize_offsets(&offsets, &mut offset_slot)?;
        let mut bounds_slot = Vec::new();
        Self::serialize_bounds(BoundKind::Ball, &vec![0.0; n], &mut bounds_slot)?;

        let graph_slice = match graph {
            Some(graph) => {
                let mut buf = Vec::new();
                graph.serialize(&mut buf)?;
                Some(FileSlice::from(buf))
            }
            None => None,
        };
        let bkt_slice = match bkt {
            Some(tree) => {
                let mut buf = Vec::new();
                tree.serialize(&mut buf)?;
                Some(FileSlice::from(buf))
            }
            None => None,
        };

        Self::open(
            options,
            FileSlice::from(centroid_slot),
            FileSlice::from(offset_slot),
            graph_slice,
            FileSlice::from(bounds_slot),
            bkt_slice,
        )
    }

    /// Parse a field's `.centroids` slots. Only the count words, the offsets,
    /// the bounds, the graph adjacency, and the BKT topology are
    /// materialized; centroid and BKT center rows stay behind
    /// [`FileSlice`]s for lazy per-row reads.
    /// `bounds_slice` is required: the caller has already refused any file
    /// old enough to lack it (the V2 check in `VectorIndexReader::open`).
    pub(crate) fn open(
        options: &VectorOptions,
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        graph_slice: Option<FileSlice>,
        bounds_slice: FileSlice,
        bkt_slice: Option<FileSlice>,
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
                // Adjacency length is validated against the arena's node
                // count inside `Graph::open`.
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

        // P1: bounds slot — one kind byte, then the stride-derived payload.
        let (bound_kind, bounds) = {
            let bytes = bounds_slice.read_bytes()?;
            let Some((&kind_code, payload)) = bytes.as_slice().split_first() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds slot is missing its kind byte",
                )
                .into());
            };
            let kind = BoundKind::from_code(kind_code)?;
            let expected = num_centroids
                .checked_mul(kind.stride(options.dim()))
                .and_then(|values| values.checked_mul(mem::size_of::<f32>()))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bounds byte length overflow")
                })?;
            if payload.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds byte length mismatch",
                )
                .into());
            }
            let mut reader = payload;
            let values: Vec<f32> = (0..num_centroids * kind.stride(options.dim()))
                .map(|_| f32::deserialize(&mut reader))
                .collect::<io::Result<_>>()?;
            // A negative bound is corrupt, never produced: the fold is a
            // max of norms seeded at 0.0. NaN / +inf are NOT rejected —
            // they fail open arithmetically at the margin comparisons.
            if values.iter().any(|&value| value < 0.0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds slot holds a negative bound",
                )
                .into());
            }
            (kind, values)
        };

        let bkt = match bkt_slice {
            Some(slice) => Some(BKTree::open(slice, options.dim(), options.metric())?),
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
            bound_kind,
            bounds,
            bkt,
        };
        // Every distinct doc owns at least its primary row, so a doc count
        // above the row total means a corrupt file.
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

    /// Distinct docs with a vector; replication inflates the row total,
    /// [`Self::num_rows`].
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

    /// The stored centroid bounds of this segment's clusters.
    ///
    /// Returns (`BoundStore`): a view over the pinned slot `[3]` payload —
    /// segment-level kind plus per-cluster values; `f32::INFINITY` =
    /// SATURATED (always probes).
    #[inline]
    pub fn bounds(&self) -> BoundStore<'_> {
        BoundStore::new(self.bound_kind, &self.bounds)
    }

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_centroids).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests only. Routing fetches per-node ranges through the lazy arena.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    /// Nearest-centroid search for `query`, ranked lazily — a [`ClusterRanking`]
    /// yielding [`Candidate`]s best similarity first (graph node `c` *is*
    /// cluster `c`, so `Candidate::node` is the cluster id).
    ///
    /// With a BKT this is approximate k-NN over the centroids via
    /// [`BKTree::search_iter`]. When a persisted RNG is also present, ranking
    /// is a resumable beam search ([`RelativeNeighborhoodGraph::search_iter`])
    /// seeded from the BKT; each beam round may pull additional BKT members
    /// whenever the tree frontier outranks the graph frontier. Without a BKT
    /// every centroid is scored exactly, up front — a graph alone is not used
    /// for routing. That exact ranking is the ground truth for BKT / BKT+RNG.
    ///
    /// `ws` holds the routing search's scratch and is borrowed for the
    /// ranking's lifetime; [`ClusterRanking::metrics`] reports the cost
    /// incurred so far (surfaced as `ProbeStats::routing`).
    pub fn rank_clusters<'a>(
        &'a self,
        ws: &'a mut Workspace,
        query: &'a [f32],
    ) -> ClusterRanking<'a> {
        match &self.bkt {
            Some(bkt) => {
                let mut bkt_iter = bkt.search_iter(query);
                let graph = match &self.graph {
                    Some(graph) => {
                        let seeds: Vec<NodeId> = (&mut bkt_iter).take(DEFAULT_MAX_LEAVES).collect();
                        let seeds = if seeds.is_empty() {
                            Self::strided_seeds(graph.len())
                        } else {
                            seeds
                        };
                        Some(graph.search_iter(ws, query, &seeds))
                    }
                    None => None,
                };
                ClusterRanking::Graph {
                    bkt: bkt_iter,
                    graph,
                    centroids: FileSliceArena::<f32>::new(self.centroids_slice.clone()),
                    bkt_scored: 0,
                }
            }
            None => {
                let arena = FileSliceArena::<f32>::new(self.centroids_slice.clone());
                let mut ranked: Vec<Candidate> = (0..self.num_centroids)
                    .map(|cluster| Candidate {
                        sim: arena.similarity(self.metric, self.dim, cluster as u32, query),
                        node: cluster as NodeId,
                    })
                    .collect();
                ranked.sort_unstable_by(|a, b| b.cmp(a));
                ClusterRanking::Exact {
                    ranked: ranked.into_iter(),
                    num_centroids: self.num_centroids,
                }
            }
        }
    }

    fn strided_seeds(graph_len: usize) -> Vec<NodeId> {
        (0..graph_len)
            .step_by((graph_len / 8).max(1))
            .take(8)
            .map(|node| node as NodeId)
            .collect()
    }
}

/// Lazily ranked clusters for one query, yielded best routing score first;
/// returned by [`IvfIndex::rank_clusters`], which documents the two paths.
pub enum ClusterRanking<'a> {
    /// BKT-ranked routing. When `graph` is set, a beam search over the
    /// persisted centroid RNG is seeded (and refilled) from the BKT. When
    /// `graph` is `None`, clusters are yielded from the BKT iterator, each
    /// scored against its stored IVF centroid.
    Graph {
        bkt: BKTreeSearchIterator<'a, FileSliceArena<f32>>,
        graph: Option<ResumableSearchIterator<'a, 'a, FileSliceArena<f32>>>,
        /// IVF centroid rows, used to score BKT members when `graph` is `None`.
        centroids: FileSliceArena<f32>,
        /// IVF centroids scored on the BKT-only path so far.
        bkt_scored: usize,
    },
    /// Exact fallback for BKT-less segments: every centroid scored and
    /// sorted up front.
    Exact {
        ranked: std::vec::IntoIter<Candidate>,
        num_centroids: usize,
    },
}

impl ClusterRanking<'_> {
    /// The routing cost incurred so far: fixed for the exact path, growing
    /// with each pull that resumes the beam search on the graph path — so
    /// take the snapshot after the last pull.
    pub fn metrics(&self) -> IvfSearchMetrics {
        match self {
            ClusterRanking::Graph {
                graph: Some(iter), ..
            } => IvfSearchMetrics {
                visited_count: iter.metrics().visited_count,
                graph: Some(iter.metrics()),
            },
            ClusterRanking::Graph { bkt_scored, .. } => IvfSearchMetrics {
                visited_count: *bkt_scored,
                graph: None,
            },
            ClusterRanking::Exact { num_centroids, .. } => IvfSearchMetrics {
                visited_count: *num_centroids,
                graph: None,
            },
        }
    }

    /// One beam round, refilling from the BKT whenever its frontier outranks
    /// the RNG frontier before an expansion.
    fn run_seeded_round(
        iter: &mut ResumableSearchIterator<'_, '_, FileSliceArena<f32>>,
        bkt: &mut BKTreeSearchIterator<'_, FileSliceArena<f32>>,
    ) {
        while {
            let should_refill = match (iter.frontier_best(), bkt.frontier_best()) {
                (_, None) => false,
                (None, Some(_)) => true,
                (Some(rng_best), Some(bkt_best)) => bkt_best > rng_best,
            };
            if should_refill {
                let seeds: Vec<NodeId> = bkt.take(DEFAULT_REFILL_SEEDS).collect();
                iter.inject(&seeds);
            }
            iter.expand_one()
        } {}
        iter.commit_round();
    }
}

impl Iterator for ClusterRanking<'_> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Candidate> {
        match self {
            ClusterRanking::Graph {
                graph,
                bkt,
                centroids,
                bkt_scored,
            } => {
                if let Some(graph_iter) = graph {
                    if graph_iter.batch_is_empty() {
                        Self::run_seeded_round(graph_iter, bkt);
                    }
                    graph_iter.pop_batch()
                } else {
                    let candidate = bkt.next_candidate(centroids)?;
                    *bkt_scored += 1;
                    Some(candidate)
                }
            }
            ClusterRanking::Exact { ranked, .. } => ranked.next(),
        }
    }
}

/// Routing cost of one [`IvfIndex::rank_clusters`] ranking (a
/// [`ClusterRanking::metrics`] snapshot): how many centroids were scored to
/// pick the probe order, and — when routing went through the centroid RNG —
/// the beam search's full [`NeighborhoodGraphSearchMetrics`].
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct IvfSearchMetrics {
    /// Centroids scored to route the query (the navigation cost):
    /// `num_centroids` on the exact path, the beam-visited count when routed
    /// via the RNG, the BKT members scored when routed from the tree alone.
    pub visited_count: usize,
    /// The centroid-graph beam search's counters; `None` when routing did
    /// not go through the RNG (exact scan or BKT-only).
    pub graph: Option<NeighborhoodGraphSearchMetrics>,
}

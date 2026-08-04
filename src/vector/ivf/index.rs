//! The `.centroids` file and its reader, [`IvfIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`IvfIndex::open`] that parses them
//! back sit side by side.
//!
//! The on-disk file is a 4-byte format-version stamp (see `vector::header`)
//! followed by a [`CompositeFile`](crate::directory::CompositeFile). Written
//! per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `Explicit`). The composite has four slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + codec (u8, V3+) + payload:
//!       codec 0 (f32, and the implicit V2 layout): N · dim · 4 bytes
//!       codec 1 (SQ4, what the writer emits): Sq4Params (2 · dim f32s)
//!               + row norms (N f32s, ||recon||² per centroid — the
//!               row-side constant of the ADC scorer)
//!               + N packed rows of sq4_stride(dim) bytes
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] RNG over the centroids (see `Graph::serialize` for the layout;
//!     absent for degenerate centroid counts — routing then falls back to a
//!     linear scan of the centroids)
//! [3] centroid bounds, REQUIRED: a segment-level BoundKind byte, then
//!     N · stride(kind) f32s in cluster order — for Ball, one f32 per
//!     cluster: max ||x - c|| over the cluster's NATIVE members' stored
//!     rows against the stored centroid (the merge documents the
//!     metric-uniform fold; replica spill is excluded per the stored
//!     `bounds_scope = native`)
//! ```
//!
//! Slot presence is the compatibility mechanism WITHIN a generation: the
//! composite footer maps `(field, slot)` to ranges, so a reader probes an
//! optional slot and an older segment simply lacks it. Slot `[2]` works
//! that way. Slot `[3]` does not, which is why it costs a generation:
//! absence would have to mean "no bounds", and a silently absent bound is
//! indistinguishable from a zero one. So `.centroids` stamps `V2`, a
//! pre-V2 file is refused at open with a REINDEX message, and a V2 file
//! missing the slot is corrupt rather than old.
//!
//! One dense `centroid_id = 0..N` indexes all four: `cluster_offsets[c]` is
//! the first row of cluster `c` in the parallel `.vec` rows/`IdMap`, and graph
//! node `c` is centroid `c` (its vector is row `c` of slot `[0]`, which is why
//! the graph slot stores no vectors of its own).

use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::graph::{
    Candidate, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, Workspace,
};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::header::VectorFileVersion;
use crate::vector::{
    sq4_stride, ArenaScorer, BoundKind, BoundStore, FileSliceArena, FileSliceScorer, Similarity,
    Sq4Params, Sq4Scorer, Sq4SliceArena, VectorArena,
};

/// Slot `[0]` codec byte (V3+): plain little-endian f32 rows.
const CODEC_F32: u8 = 0;
/// Slot `[0]` codec byte (V3+): SQ4 params + packed 4-bit rows.
const CODEC_SQ4: u8 = 1;

/// The centroid storage a segment routes over: f32 rows (V2 files, or a V3
/// f32 codec) or SQ4 codes (what the writer emits from V3 on). One enum so
/// [`IvfIndex`] and its routing graph stay non-generic; both variants score
/// f32 queries through their native kernels.
#[derive(Clone)]
pub(crate) enum CentroidArena {
    F32(FileSliceArena<f32>),
    Sq4(Sq4SliceArena),
}

impl VectorArena for CentroidArena {
    type Elem = f32;
    type Scorer<'a>
        = CentroidScorer<'a>
    where Self: 'a;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        match self {
            CentroidArena::F32(arena) => arena.num_vectors(dim),
            CentroidArena::Sq4(arena) => arena.num_vectors(dim),
        }
    }

    #[inline]
    fn scorer<'a>(&'a self, metric: Metric, dim: usize, query: &'a [f32]) -> CentroidScorer<'a> {
        match self {
            CentroidArena::F32(arena) => CentroidScorer::F32(arena.scorer(metric, dim, query)),
            CentroidArena::Sq4(arena) => CentroidScorer::Sq4(arena.scorer(metric, dim, query)),
        }
    }
}

/// [`CentroidArena`]'s per-query scorer: the codec dispatch, hoisted out
/// of the per-node loop.
pub(crate) enum CentroidScorer<'a> {
    F32(FileSliceScorer<'a, f32>),
    Sq4(Sq4Scorer<'a>),
}

impl ArenaScorer for CentroidScorer<'_> {
    #[inline]
    fn score(&self, node: NodeId) -> Similarity {
        match self {
            CentroidScorer::F32(scorer) => scorer.score(node),
            CentroidScorer::Sq4(scorer) => scorer.score(node),
        }
    }
}

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and
/// the RNG adjacency (edges only, `num_centroids × max_edges × 4` bytes). The
/// centroid vectors stay behind a [`FileSliceArena`] and are fetched one node
/// at a time as routing visits them. Everything row-scale (the rows and
/// id-map) lives on [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub struct IvfIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field. Rows including
    /// replicas are [`Self::num_rows`].
    num_docs: usize,
    /// The centroid payload (slot `[0]` past the count words and, from V3,
    /// the codec byte and quantization params): the rows in the codec's own
    /// representation, kept for [`Self::centroid_bytes`] introspection.
    centroids_slice: FileSlice,
    /// The codec-aware view over `centroids_slice` that routing scores
    /// through; the graph holds its own clone.
    arena: CentroidArena,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    dim: usize,
    metric: Metric,
    /// The persisted RNG over the centroids (slot `[2]`). `None` for
    /// degenerate centroid counts, where routing falls back to a linear scan.
    graph: Option<RelativeNeighborhoodGraph<CentroidArena>>,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
}

impl IvfIndex {
    /// Write slot `[0]` of the `.centroids` composite for a field, in the V3
    /// SQ4 codec: count words, codec byte, quantization params, per-row
    /// norms ([`sq4_row_norms`](crate::vector::sq4_row_norms)), then the
    /// packed rows. `num_docs` is the number of distinct docs assigned — NOT
    /// the posting-row total, which replication can multiply.
    pub(crate) fn serialize_centroids<W: Write + ?Sized>(
        num_centroids: usize,
        num_docs: usize,
        params: &Sq4Params,
        norms_sq: &[f32],
        codes: &[u8],
        options: &VectorOptions,
        out: &mut W,
    ) -> io::Result<()> {
        if params.dim() != options.dim() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SQ4 params dimension mismatch",
            ));
        }
        if norms_sq.len() != num_centroids {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SQ4 row norm count mismatch",
            ));
        }
        let expected = num_centroids
            .checked_mul(sq4_stride(options.dim()))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if codes.len() != expected {
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
        CODEC_SQ4.serialize(out)?;
        params.serialize(out)?;
        for norm in norms_sq {
            norm.serialize(out)?;
        }
        out.write_all(codes)
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

    /// Parse a field's `.centroids` slots. Only the count words, the codec
    /// header, the offsets, the bounds, and the graph adjacency are
    /// materialized; the centroid rows stay behind a [`FileSlice`] for lazy
    /// per-node reads. `bounds_slice` is required: the caller has already
    /// refused any file old enough to lack it (the V2 check in
    /// `VectorIndexReader::open`); `version` selects the slot `[0]` layout —
    /// V2 has no codec byte and holds f32 rows.
    pub(crate) fn open(
        options: &VectorOptions,
        version: VectorFileVersion,
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        graph_slice: Option<FileSlice>,
        bounds_slice: FileSlice,
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
        let mut centroids_slice = centroids_slice.slice_from(count_words);

        let codec = if version >= VectorFileVersion::V3 {
            if centroids_slice.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF centroids slot is missing its codec byte",
                )
                .into());
            }
            let byte = centroids_slice.slice_to(1).read_bytes()?[0];
            centroids_slice = centroids_slice.slice_from(1);
            byte
        } else {
            CODEC_F32
        };

        let (arena, stride) = match codec {
            CODEC_F32 => (
                CentroidArena::F32(FileSliceArena::<f32>::new(centroids_slice.clone())),
                options.bytes_per_vector(),
            ),
            CODEC_SQ4 => {
                let params_len = Sq4Params::serialized_len(options.dim());
                let norms_len = num_centroids
                    .checked_mul(mem::size_of::<f32>())
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "row norm length overflow")
                    })?;
                if centroids_slice.len() < params_len + norms_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "IVF centroids slot is smaller than its SQ4 params and norms",
                    )
                    .into());
                }
                let params_bytes = centroids_slice.slice_to(params_len).read_bytes()?;
                let params = Sq4Params::deserialize(params_bytes.as_slice(), options.dim())?;
                centroids_slice = centroids_slice.slice_from(params_len);
                let norms_bytes = centroids_slice.slice_to(norms_len).read_bytes()?;
                let mut norms_reader = norms_bytes.as_slice();
                let norms_sq: Vec<f32> = (0..num_centroids)
                    .map(|_| f32::deserialize(&mut norms_reader))
                    .collect::<io::Result<_>>()?;
                centroids_slice = centroids_slice.slice_from(norms_len);
                // Validate before construction — the arena asserts the
                // norm/row correspondence, and a corrupt file must error,
                // not panic.
                let stride = sq4_stride(options.dim());
                let expected = num_centroids.checked_mul(stride).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
                })?;
                if centroids_slice.len() != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "IVF centroid byte length mismatch",
                    )
                    .into());
                }
                (
                    CentroidArena::Sq4(Sq4SliceArena::new(
                        centroids_slice.clone(),
                        params,
                        norms_sq,
                    )),
                    stride,
                )
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown IVF centroid codec: {other}"),
                )
                .into());
            }
        };

        let centroid_len = num_centroids.checked_mul(stride).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
        })?;
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
                // Adjacency length is validated against the arena's node
                // count inside `Graph::open`.
                let adjacency = slice.read_bytes()?;
                Some(RelativeNeighborhoodGraph::open(
                    &adjacency,
                    arena.clone(),
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

        let index = IvfIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            arena,
            cluster_offsets,
            dim: options.dim(),
            metric: options.metric(),
            graph,
            bound_kind,
            bounds,
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

    /// The centroid rows in their stored codec (packed SQ4 codes from V3
    /// on, f32 rows for V2 files), materialized in one read — for
    /// introspection and tests only. Routing fetches per-node ranges
    /// through the lazy arena.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    /// The exact routing similarity of `query` to `cluster`'s stored
    /// centroid, scored through the same codec-aware arena and kernel the
    /// ranked routing stream uses.
    pub fn centroid_similarity(&self, cluster: usize, query: &[f32]) -> Similarity {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        self.arena
            .similarity(self.metric, self.dim, cluster as NodeId, query)
    }

    /// Decodes `cluster`'s stored centroid into `out` (`dim` f32s) — the
    /// exact values routing scores: a byte decode for f32 codecs, the SQ4
    /// reconstruction otherwise.
    pub fn decode_centroid(&self, cluster: usize, out: &mut [f32]) -> crate::Result<()> {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        assert_eq!(out.len(), self.dim, "output dimension mismatch");
        match &self.arena {
            CentroidArena::F32(_) => {
                let stride = self.dim * mem::size_of::<f32>();
                let bytes = self
                    .centroids_slice
                    .slice(cluster * stride..(cluster + 1) * stride)
                    .read_bytes()?;
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            CentroidArena::Sq4(arena) => {
                let stride = sq4_stride(self.dim);
                let bytes = self
                    .centroids_slice
                    .slice(cluster * stride..(cluster + 1) * stride)
                    .read_bytes()?;
                arena.params().decode_row_into(&bytes, out);
            }
        }
        Ok(())
    }

    /// Clusters to probe for `query`, ranked lazily — a [`ClusterRanking`]
    /// yielding [`Candidate`]s best routing score first (graph node `c` *is*
    /// cluster `c`, so `Candidate::node` is the cluster id).
    ///
    /// With a persisted RNG this is a resumable beam search
    /// ([`RelativeNeighborhoodGraph::search_iter`]): the first batch is one
    /// converged round at the configured
    /// [`ef`](NeighborhoodGraphConfig::ef), and pulling past it resumes the
    /// search, so routing cost is paid only as far as probing actually
    /// reaches. Without one every centroid is scored exactly, up front. Both
    /// paths score through the same [`FileSliceArena`], so their rankings
    /// agree.
    ///
    /// `ws` holds the routing search's scratch and is borrowed for the
    /// ranking's lifetime; [`ClusterRanking::metrics`] reports the cost
    /// incurred so far (surfaced as `ProbeStats::routing`).
    pub(crate) fn rank_clusters<'a>(
        &'a self,
        ws: &'a mut Workspace,
        query: &'a [f32],
    ) -> ClusterRanking<'a> {
        match &self.graph {
            Some(graph) => {
                // TODO: Replace with proper seed generation
                let seeds: Vec<NodeId> = {
                    (0..graph.len())
                        .step_by((graph.len() / 8).max(1))
                        .take(8)
                        .map(|node| node as NodeId)
                        .collect()
                };
                ClusterRanking::Graph(graph.search_iter(ws, query, &seeds))
            }
            None => {
                let scorer = self.arena.scorer(self.metric, self.dim, query);
                let mut ranked: Vec<Candidate> = (0..self.num_centroids)
                    .map(|cluster| Candidate {
                        sim: scorer.score(cluster as NodeId),
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
}

/// Lazily ranked clusters for one query, yielded best routing score first;
/// returned by [`IvfIndex::rank_clusters`], which documents the two paths.
pub(crate) enum ClusterRanking<'a> {
    /// Beam-searched routing over the persisted centroid RNG; pulling past a
    /// converged batch resumes the search.
    Graph(ResumableSearchIterator<'a, 'a, CentroidArena>),
    /// Exact fallback for graph-less segments: every centroid scored and
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
    pub(crate) fn metrics(&self) -> IvfSearchMetrics {
        match self {
            ClusterRanking::Graph(iter) => IvfSearchMetrics {
                visited_count: iter.metrics().visited_count,
                graph: Some(iter.metrics()),
            },
            ClusterRanking::Exact { num_centroids, .. } => IvfSearchMetrics {
                visited_count: *num_centroids,
                graph: None,
            },
        }
    }
}

impl Iterator for ClusterRanking<'_> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Candidate> {
        match self {
            ClusterRanking::Graph(iter) => iter.next(),
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
    /// via the RNG.
    pub visited_count: usize,
    /// The centroid-graph beam search's counters; `None` when routing fell
    /// back to a linear scan of the centroids.
    pub graph: Option<NeighborhoodGraphSearchMetrics>,
}

//! Assignments: computing a vector's cells against the index-level
//! centroid index, and the persisted per-segment result,
//! [`SegmentClusters`].
//!
//! LONG TERM: assignments should not stay scattered across segments.
//! Today every segment persists its own cluster → row-range map, so
//! probing one cluster costs one ranged read per segment that holds it
//! (`ProbeStats::segment_opens` vs `clusters_probed` is exactly that
//! fragmentation). The centroids already made this jump — hoisted to one
//! index-level artifact — and the postings should follow: an index-level
//! store keyed by cluster id, so a probed cluster is one contiguous read
//! however many segments contributed rows to it.
//!
//! WRITE SIDE — both write paths (per-commit serialize and merge) assign
//! every vector against the same frozen centroid index, taking the
//! primary cell and the `replicas - 1` next-nearest cells from ONE k-NN
//! call per vector ([`assign_cells`]). The selector mirrors the
//! query-time router's ranking per metric, so cells predict where a
//! query would look. Everything runs on the CALLING thread: assignment
//! reads centroid rows through the index's `Directory`, and an embedder
//! like pg_search runs inside a Postgres backend, where FFI from a
//! spawned thread aborts the transaction — so no path in the vector
//! write pipeline spawns.
//!
//! READ SIDE — a segment keeps only what is genuinely per-segment: which
//! rows landed in which cluster, and the residual geometry of those
//! rows. This module owns the wire format of those `.vec` slots end to
//! end: the serializers the write paths call and the
//! [`SegmentClusters::open`] that parses them back sit side by side.
//!
//! The `.vec` composite slots (see `vector::header::vec_slot`):
//!
//! ```text
//! [2] cluster_offsets (u64[C+1], prefix sum over the cluster-sorted rows)
//! [3] centroid bounds, REQUIRED: a segment-level BoundKind byte, then
//!     C · stride(kind) f32s in cluster order — for Ball, one f32 per
//!     cluster: max ||x - c|| over the cluster's NATIVE members' stored
//!     rows against the centroid index's stored centroid (replica spill is excluded
//!     per the stored `bounds_scope = native`)
//! [4] IVF meta: num_docs (u32) + num_centroids (u32)
//! ```
//!
//! One dense `centroid_id = 0..C` indexes the set file's rows and these
//! slots alike: `cluster_offsets[c]` is the first row of cluster `c` in the
//! parallel `.vec` rows/`IdMap`.

use std::cmp::Ordering;
use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::graph::{NeighborhoodGraphSearchMetrics, NodeId, RelativeNeighborhoodGraph, Workspace};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::distance::{cosine, dot, l2_squared};
use crate::vector::ivf::centroid_index::{FieldCentroids, UnitNormRowsArena};
use crate::vector::{BoundKind, BoundStore};
use crate::Executor;

/// The per-segment IVF remainder for one field: which contiguous row
/// ranges of the `.vec` rows form each cluster, plus the per-cluster
/// bounds.
///
/// Everything here is small and pinned; the row-scale payload (rows,
/// id-map) lives on [`VectorIndexReader`](crate::vector::VectorIndexReader),
/// and the centroid rows live in the index-level set file.
pub struct SegmentClusters {
    num_centroids: usize,
    /// Distinct documents with a vector in this field. Rows including
    /// replicas are [`Self::num_rows`].
    num_docs: usize,
    /// Slot `[2]`: the `u64[C+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
    /// Derived at open (one pass over the offsets): bit `c` set ⟺ cluster
    /// `c` has rows in THIS segment. The probe loop's presence check
    /// touches one bit instead of two random u64s of the offsets array.
    non_empty: Vec<u64>,
    /// Count of set bits in [`Self::non_empty`] — this segment's share of
    /// the index's open-charge capacity.
    num_non_empty: usize,
}

impl SegmentClusters {
    /// Write slot `[2]` of the `.vec` composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Write slot `[3]` of the `.vec` composite for a field: the
    /// segment-level kind byte, then the per-cluster payload.
    ///
    /// * `kind` (`BoundKind`) — the segment-level bound kind.
    /// * `values` (`&[f32]`) — `num_centroids * kind.stride(dim)` values in cluster order; the
    ///   caller's [`BoundsBuilder`] output.
    /// * `out` (`&mut W`) — the slot writer.
    ///
    /// Returns (`io::Result<()>`): write errors only — the payload length
    /// is validated at open, against the count words of slot `[4]`.
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

    /// Write slot `[4]` of the `.vec` composite for a field. `num_docs` is
    /// the number of distinct docs assigned — NOT the posting-row total,
    /// which replication can multiply.
    pub(crate) fn serialize_ivf_meta<W: Write + ?Sized>(
        num_docs: usize,
        num_centroids: usize,
        out: &mut W,
    ) -> io::Result<()> {
        u32::try_from(num_docs)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "doc count exceeds u32"))?
            .serialize(out)?;
        u32::try_from(num_centroids)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32"))?
            .serialize(out)
    }

    /// Parse a field's per-segment IVF slots. Everything is materialized and
    /// pinned — offsets, bounds, and the meta words are all `O(C)`.
    pub(crate) fn open(
        options: &VectorOptions,
        offsets_slice: FileSlice,
        bounds_slice: FileSlice,
        meta_slice: FileSlice,
    ) -> crate::Result<Self> {
        let meta_len = 2 * mem::size_of::<u32>();
        if meta_slice.len() != meta_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF meta slot has the wrong length",
            )
            .into());
        }
        let meta = meta_slice.read_bytes()?;
        let mut reader = meta.as_slice();
        let num_docs = u32::deserialize(&mut reader)? as usize;
        let num_centroids = u32::deserialize(&mut reader)? as usize;

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

        // Bounds slot — one kind byte, then the stride-derived payload.
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

        let mut index = SegmentClusters {
            num_centroids,
            num_docs,
            cluster_offsets,
            bound_kind,
            bounds,
            non_empty: vec![0u64; num_centroids.div_ceil(64)],
            num_non_empty: 0,
        };
        for cluster in 0..num_centroids {
            if index.cluster_offset(cluster + 1) > index.cluster_offset(cluster) {
                index.non_empty[cluster / 64] |= 1u64 << (cluster % 64);
                index.num_non_empty += 1;
            }
        }
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

    /// Presence: `true` iff `cluster` has rows in this segment. One bit of
    /// pinned state; the probe loop's cheapest gate.
    #[inline]
    pub fn has_cluster(&self, cluster: usize) -> bool {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        (self.non_empty[cluster / 64] >> (cluster % 64)) & 1 == 1
    }

    /// The non-empty row range of `cluster`, or `None` when the cluster has
    /// no rows here — the driver-facing form, so the offsets encoding can
    /// change underneath (a sparse layout is the flagged follow-up).
    #[inline]
    pub fn non_empty_cluster_range(&self, cluster: usize) -> Option<Range<usize>> {
        self.has_cluster(cluster)
            .then(|| self.cluster_range(cluster))
    }

    /// Number of clusters with at least one row in this segment.
    pub fn num_non_empty_clusters(&self) -> usize {
        self.num_non_empty
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
}

/// Routing cost of ranking the clusters to probe for one query: how many
/// centroids were scored to pick the probe order, and — when routing went
/// through the centroid RNG — the beam search's full
/// [`NeighborhoodGraphSearchMetrics`].
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

// ---- computing assignments ------------------------------------------

/// Rows assigned per [`Executor`] work item when a batch is split across
/// threads. Small enough to load-balance, large enough to amortize the
/// per-item `Workspace` allocation.
const ASSIGN_CHUNK_ROWS: usize = 256;

/// Centroid ids of the `knn` nearest centroids to `query`, nearest first —
/// the exact counterpart of [`RelativeNeighborhoodGraph::nearest`], same
/// distance family per metric. Ties break on centroid id so selection is
/// deterministic.
fn exact_nearest_centroids(
    metric: Metric,
    centroids: &[f32],
    dim: usize,
    query: &[f32],
    knn: usize,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = centroids
        .chunks_exact(dim)
        .enumerate()
        .map(|(id, centroid)| {
            // Each arm is the negated `Metric::similarity` ordering the
            // graph selector ranks by.
            let d = match metric {
                // `1 - cosine` orders identically to descending cosine.
                Metric::Cosine => 1.0 - cosine(query, centroid),
                // Negated raw dot: the query-time router ranks by dot, and
                // dot ordering is ||q||-invariant — a v-directed query
                // ranks cells by dot(v, c), so raw dot IS the
                // query-consistent criterion.
                Metric::Dot => -dot(query, centroid),
                // Squared L2 orders identically to L2.
                Metric::L2 => l2_squared(query, centroid),
            };
            // A NaN score (a zero-norm centroid under Cosine) must rank
            // WORST, not tie: `partial_cmp`'s `Equal` fallback would let
            // the id tie-break route real vectors into a degenerate cell.
            let d = if d.is_nan() { f32::INFINITY } else { d };
            (d, id)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(knn);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// How a vector's cells are picked from the stored centroids. Exact
/// k-NN scan for a small centroid index — anything the search's own `ef` visit
/// budget would cover wholesale anyway, where the brute scan is at most
/// as expensive and exact (an approximate graph over a handful of points
/// can return fewer than `knn` neighbours, silently under-assigning) —
/// and, for large ones, the set's PERSISTED router: the same pinned
/// graph queries route through, cached on `Index`, so a segment build
/// never constructs a selector structure of its own.
pub(crate) enum CentroidSelector<'a> {
    Exact { centroids: Vec<f32> },
    Router(&'a RelativeNeighborhoodGraph<UnitNormRowsArena>),
}

impl<'a> CentroidSelector<'a> {
    /// Selector over `set`'s rows, sized for `cells_per_vector`-deep
    /// selection. `router` is the set's persisted routing graph (absent
    /// for degenerate sets and consumer-defined routers); without it a
    /// large set falls back to the exact scan — correct, just O(C) per
    /// vector.
    pub(crate) fn for_set(
        set: &FieldCentroids,
        router: Option<&'a RelativeNeighborhoodGraph<UnitNormRowsArena>>,
        options: &VectorOptions,
        cells_per_vector: usize,
    ) -> crate::Result<Self> {
        let ef_search = (cells_per_vector * 4).max(64);
        if set.num_centroids() <= ef_search {
            return Ok(CentroidSelector::Exact {
                centroids: set.values_f32(options)?,
            });
        }
        match router {
            Some(graph) => Ok(CentroidSelector::Router(graph)),
            None => {
                log::warn!(
                    "assigning against {} centroids with no readable router; falling back to an \
                     exact per-vector scan",
                    set.num_centroids(),
                );
                Ok(CentroidSelector::Exact {
                    centroids: set.values_f32(options)?,
                })
            }
        }
    }

    /// The `knn` nearest centroid ids to `v`, nearest first. The graph arm
    /// may return fewer than `knn` (approximate recall), never duplicates.
    fn nearest(
        &self,
        metric: Metric,
        dim: usize,
        ws: &mut Workspace,
        v: &[f32],
        knn: usize,
    ) -> Vec<usize> {
        match self {
            CentroidSelector::Exact { centroids } => {
                exact_nearest_centroids(metric, centroids, dim, v, knn)
            }
            CentroidSelector::Router(graph) => {
                // The router's Cosine arm is a raw dot over unit-norm rows
                // (`UnitNormRowsArena`); assignment inputs are stored rows,
                // normalized at ingest, so the contract holds.
                // TODO: Replace with proper seed generation
                let seeds: Vec<NodeId> = (0..graph.len())
                    .step_by((graph.len() / 8).max(1))
                    .take(8)
                    .map(|node| node as NodeId)
                    .collect();
                graph
                    .search(ws, v, &seeds, knn)
                    .0
                    .into_iter()
                    .map(|candidate| candidate.node as usize)
                    .collect()
            }
        }
    }
}

/// Assign a batch of vectors to their cells: per vector, up to
/// `cells_per_vector` distinct centroid ids, nearest first — index 0 is the
/// primary, the rest are replica cells. `values` is `dim`-strided,
/// row-parallel output order. Chunks the batch across `executor`.
pub(crate) fn assign_cells(
    selector: &CentroidSelector<'_>,
    metric: Metric,
    dim: usize,
    values: &[f32],
    cells_per_vector: usize,
    executor: &Executor,
) -> crate::Result<Vec<Vec<usize>>> {
    debug_assert_eq!(values.len() % dim.max(1), 0);
    let num_rows = values.len() / dim.max(1);
    let assign_chunk = |range: std::ops::Range<usize>| -> Vec<Vec<usize>> {
        // One scratch reused across the chunk's graph lookups — never
        // per vector.
        let mut ws = Workspace::new();
        range
            .map(|row| {
                let v = &values[row * dim..(row + 1) * dim];
                selector.nearest(metric, dim, &mut ws, v, cells_per_vector)
            })
            .collect()
    };
    if executor.num_threads() <= 1 || num_rows <= ASSIGN_CHUNK_ROWS {
        return Ok(assign_chunk(0..num_rows));
    }
    let chunk_starts = (0..num_rows).step_by(ASSIGN_CHUNK_ROWS);
    let per_chunk = executor.map(
        |start| {
            Ok(assign_chunk(
                start..(start + ASSIGN_CHUNK_ROWS).min(num_rows),
            ))
        },
        chunk_starts,
    )?;
    Ok(per_chunk.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the Dot selection semantics: cells follow RAW dot — the
    /// query-time router's ranking — not angular order. Centroid norms are
    /// deliberately unequal so the two orderings disagree.
    #[test]
    fn dot_selector_uses_raw_dot_not_angular() {
        let centroids: Vec<f32> = vec![
            10.0, 0.0, // long, off-direction: dot 10, cosine 0.45
            0.0, 1.0, // short, near-direction: dot 2, cosine 0.89
            7.0, 7.0, // long, near-direction: dot 21, cosine 0.95
        ];
        let query = [1.0_f32, 2.0];
        let picked = exact_nearest_centroids(Metric::Dot, &centroids, 2, &query, 3);
        // Raw-dot order: [7,7] (21), then [10,0] (10), then [0,1] (2).
        // Angular order would put [0,1] ahead of [10,0].
        assert_eq!(picked, vec![2, 0, 1], "must rank by raw dot");
    }

    /// Assignment yields nearest-first distinct cells for every row, and the
    /// parallel chunking preserves row order.
    #[test]
    fn assign_cells_is_nearest_first_and_order_preserving() -> crate::Result<()> {
        let selector = CentroidSelector::Exact {
            centroids: vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        };
        let values: Vec<f32> = vec![
            1.0, 0.0, // nearest 0, then 1
            9.0, 1.0, // nearest 1, then 0
            0.5, 9.0, // nearest 2, then 0
        ];
        let cells = assign_cells(
            &selector,
            Metric::L2,
            2,
            &values,
            2,
            &Executor::single_thread(),
        )?;
        assert_eq!(cells, vec![vec![0, 1], vec![1, 0], vec![2, 0]]);
        Ok(())
    }
}

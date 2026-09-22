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
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride),
//!     rows in the router's canonical cluster-sorted order when a stacked
//!     router was built
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] a router-kind byte followed by the selected router's payload, REQUIRED
//! [3] centroid bounds, REQUIRED: a segment-level BoundKind byte,
//!     then N · stride(kind) f32s in cluster order — for Ball, one f32 per
//!     cluster: max ||x - c|| over the cluster's members' stored rows against
//!     the stored centroid (the merge documents the metric-uniform fold).
//! ```
//!
//! Persisted IVF routers use the V3 layout. Earlier bare-router and
//! router-less layouts are not supported.
use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::header::VectorFileVersion;
use crate::vector::router::{OpenedRouter, RouterIter, RouterKind, RouterWorkspace};
use crate::vector::{BoundKind, BoundStore, Similarity};

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
    /// Distinct documents with a vector in this field.
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words).
    centroids_slice: FileSlice,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    metric: Metric,
    router: OpenedRouter,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
}

impl IvfIndex {
    /// Write slot `[0]` of the `.centroids` composite for a field. `num_docs`
    /// is the number of distinct docs assigned, not the posting-row total.
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

    /// Parse a field's `.centroids` slots. Only the count words, the offsets,
    /// the bounds, and the router topology are materialized; the centroid
    /// rows stay behind a [`FileSlice`] for lazy per-node reads.
    /// The persisted router kind must match the configured router.
    pub(crate) fn open(
        version: VectorFileVersion,
        options: &VectorOptions,
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        router_slice: FileSlice,
        router: RouterKind,
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

        let router = router.open(version, router_slice, centroids_slice.clone(), options)?;

        let bytes = bounds_slice.read_bytes()?;
        let Some((&kind_code, payload)) = bytes.as_slice().split_first() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF bounds slot is missing its kind byte",
            )
            .into());
        };
        let bound_kind = BoundKind::from_code(kind_code)?;
        let expected = num_centroids
            .checked_mul(bound_kind.stride(options.dim()))
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
        let bounds: Vec<f32> = (0..num_centroids * bound_kind.stride(options.dim()))
            .map(|_| f32::deserialize(&mut reader))
            .collect::<io::Result<_>>()?;
        // A negative bound is corrupt, never produced: the fold is a max of
        // norms seeded at 0.0. NaN / +inf fail open in margin comparisons.
        if bounds.iter().any(|&value| value < 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF bounds slot holds a negative bound",
            )
            .into());
        }

        let index = IvfIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            cluster_offsets,
            metric: options.metric(),
            router,
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

    pub fn router(&self) -> RouterKind {
        self.router.kind()
    }

    /// Distinct docs with a vector.
    pub(crate) fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Total posting rows across all clusters.
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

    pub(crate) fn centroid_scores(&self, query: &[f32]) -> crate::Result<Vec<Similarity>> {
        let mut scores = Vec::with_capacity(self.num_centroids);
        if self.num_centroids == 0 {
            return Ok(scores);
        }
        let stride = self.centroids_slice.len() / self.num_centroids;
        for start in (0..self.num_centroids).step_by(64) {
            let end = (start + 64).min(self.num_centroids);
            let bytes = self
                .centroids_slice
                .slice(start * stride..end * stride)
                .read_bytes()?;
            if bytes.len() != (end - start) * stride {
                return Err(
                    io::Error::new(io::ErrorKind::UnexpectedEof, "short centroid read").into(),
                );
            }
            scores.extend(
                bytes
                    .chunks_exact(stride)
                    .map(|row| self.metric.similarity_bytes(query, row)),
            );
        }
        Ok(scores)
    }

    pub(crate) fn rank_clusters_with_scores<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
        scores: &'router [Similarity],
    ) -> RouterIter<'router, 'workspace> {
        self.router
            .rank_with_scores(workspace, query, self.metric, Some(scores))
    }

    pub(crate) fn rank_clusters<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
    ) -> RouterIter<'router, 'workspace> {
        self.router.rank(workspace, query, self.metric)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::directory::FileHandle;
    use crate::vector::ivf::{LazyStore, NeighborhoodGraphConfig, RelativeNeighborhoodGraph};

    #[derive(Debug)]
    struct ScoreFile {
        bytes: OwnedBytes,
        reads: AtomicUsize,
        bytes_read: AtomicUsize,
        fault: AtomicUsize,
        copy: bool,
    }

    impl HasLen for ScoreFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for ScoreFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(range.len(), Ordering::Relaxed);
            match self.fault.load(Ordering::Relaxed) {
                1 => return Err(io::Error::other("centroid read failed")),
                2 => return Ok(self.bytes.slice(range.start..range.end - 1)),
                _ => {}
            }
            if self.copy {
                Ok(OwnedBytes::new(self.bytes[range].to_vec()))
            } else {
                Ok(self.bytes.slice(range))
            }
        }
    }

    fn score_fixture(
        n: usize,
        dim: usize,
        metric: Metric,
        copy: bool,
    ) -> (IvfIndex, Arc<ScoreFile>) {
        let mut state = 17u64;
        let bytes: Vec<u8> = (0..n * dim)
            .flat_map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5).to_le_bytes()
            })
            .collect();
        let file = Arc::new(ScoreFile {
            bytes: OwnedBytes::new(bytes),
            reads: AtomicUsize::new(0),
            bytes_read: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
            copy,
        });
        let centroids_slice = FileSlice::new(file.clone());
        let mut adjacency = 8u32.to_le_bytes().to_vec();
        for node in 0..n {
            for step in [1, 7, 19, 43, 101, 251, 509, 1021] {
                adjacency.extend_from_slice(&(((node + step) % n) as u32).to_le_bytes());
            }
        }
        let router = OpenedRouter::Rng(
            RelativeNeighborhoodGraph::open(
                &adjacency,
                LazyStore::new(centroids_slice.clone(), dim),
                dim,
                metric,
                NeighborhoodGraphConfig::default(),
            )
            .unwrap(),
        );
        let cluster_offsets = OwnedBytes::new(
            (0..=n)
                .flat_map(|i| (i as u64).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        (
            IvfIndex {
                num_centroids: n,
                num_docs: n,
                centroids_slice,
                cluster_offsets,
                metric,
                router,
                bound_kind: BoundKind::Ball,
                bounds: vec![0.0; n],
            },
            file,
        )
    }

    #[test]
    fn centroid_scores_preserve_bytes_and_batch_boundaries() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for n in [0, 1, 63, 64, 65, 129] {
                for dim in [2, 17, 1024] {
                    let (index, file) = score_fixture(n, dim, metric, false);
                    let query: Vec<f32> = (0..dim).map(|i| (i % 13) as f32 / 13.0).collect();
                    let scores = index.centroid_scores(&query)?;
                    assert_eq!(scores.len(), n);
                    assert_eq!(file.reads.load(Ordering::Relaxed), n.div_ceil(64));
                    assert_eq!(file.bytes_read.load(Ordering::Relaxed), n * dim * 4);
                    for (score, bytes) in scores.iter().zip(file.bytes.chunks_exact(dim * 4)) {
                        assert_eq!(
                            score.score().to_bits(),
                            metric.similarity_bytes(&query, bytes).score().to_bits()
                        );
                    }
                    if n > 0 {
                        let mut lazy_ws = RouterWorkspace::default();
                        let mut scored_ws = RouterWorkspace::default();
                        let mut lazy = index.rank_clusters(&mut lazy_ws, &query);
                        let mut scored =
                            index.rank_clusters_with_scores(&mut scored_ws, &query, &scores);
                        for _ in 0..=n {
                            assert_eq!(
                                lazy.next().map(|c| (c.node, c.sim.score().to_bits())),
                                scored.next().map(|c| (c.node, c.sim.score().to_bits()))
                            );
                            assert_eq!(
                                serde_json::to_value(lazy.metrics())?,
                                serde_json::to_value(scored.metrics())?
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn centroid_scores_propagate_read_failures() {
        let (index, file) = score_fixture(65, 17, Metric::Dot, false);
        for fault in [1, 2] {
            file.fault.store(fault, Ordering::Relaxed);
            assert!(index.centroid_scores(&[0.5; 17]).is_err());
        }
    }

    #[test]
    #[ignore = "routing-only timing screen; run in release on an otherwise idle host"]
    fn benchmark_centroid_precompute() -> crate::Result<()> {
        use std::hint::black_box;
        use std::time::Instant;

        for n in [32usize, 512, 4096] {
            for copy in [false, true] {
                let (index, file) = score_fixture(n, 1024, Metric::Dot, copy);
                let query = vec![0.25; 1024];
                for fraction in [0.01f64, 0.3, 1.0] {
                    let prefix = ((n as f64 * fraction).ceil() as usize).max(16).min(n);
                    let mut elapsed = [0u128; 2];
                    let mut reads = [0; 2];
                    let mut bytes = [0; 2];
                    for iteration in 0..22 {
                        for mode in [iteration % 2, 1 - iteration % 2] {
                            file.reads.store(0, Ordering::Relaxed);
                            file.bytes_read.store(0, Ordering::Relaxed);
                            let start = Instant::now();
                            let scores = if mode == 1 {
                                Some(index.centroid_scores(&query)?)
                            } else {
                                None
                            };
                            let mut workspace = RouterWorkspace::default();
                            let mut ranked = match scores.as_deref() {
                                Some(scores) => {
                                    index.rank_clusters_with_scores(&mut workspace, &query, scores)
                                }
                                None => index.rank_clusters(&mut workspace, &query),
                            };
                            for candidate in ranked.by_ref().take(prefix) {
                                black_box(candidate);
                            }
                            black_box(ranked.metrics());
                            let ns = start.elapsed().as_nanos();
                            if iteration >= 2 {
                                elapsed[mode] += ns;
                                reads[mode] += file.reads.load(Ordering::Relaxed);
                                bytes[mode] += file.bytes_read.load(Ordering::Relaxed);
                            }
                        }
                    }
                    eprintln!(
                        "centroid_precompute n={n} dim=1024 copy={copy} prefix={prefix} \
                         fraction={fraction} lazy_ns={} precomputed_ns={} lazy_reads={} \
                         precomputed_reads={} lazy_bytes={} precomputed_bytes={}",
                        elapsed[0] / 20,
                        elapsed[1] / 20,
                        reads[0] / 20,
                        reads[1] / 20,
                        bytes[0] / 20,
                        bytes[1] / 20
                    );
                }
            }
        }
        Ok(())
    }
}

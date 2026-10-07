//! Per-segment IVF postings and bounds, referencing the shared centroid router.
//! The V5 `.centroids` layout is described in `vector/FORMAT.md`.
use std::io::{self, Write};
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use common::{BinarySerializable, OwnedBytes};

use crate::directory::FileSlice;
use crate::index::CentroidIndexMeta;
use crate::schema::{Metric, VectorOptions};
use crate::vector::ivf::RecallEstimator;
use crate::vector::router::{OpenedRouter, RouterIter, RouterKind, RouterWorkspace, RoutingParams};
use crate::vector::{BoundKind, BoundStore};

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and
/// the RNG adjacency (edges only, `num_centroids × max_edges × 4` bytes). The
/// centroid vectors stay behind a [`FileSliceArena`] and are fetched one node
/// at a time as routing visits them. Everything row-scale (the rows and
/// id-map) lives on [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub struct IvfIndex {
    centroid_index: CentroidIndexMeta,
    routing: Arc<RouterIndex>,
    /// Distinct documents with a vector in this field.
    num_docs: usize,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
}

/// Centroid data and routing state, independent of segment postings.
pub(crate) struct RouterIndex {
    num_centroids: usize,
    /// Canonically ordered centroid rows, without metadata.
    centroids_slice: FileSlice,
    metric: Metric,
    router: OpenedRouter,
}

impl IvfIndex {
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
    /// The payload length is validated against the shared router at open.
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

    pub(crate) fn open_postings(
        options: &VectorOptions,
        routing: Arc<RouterIndex>,
        centroid_index: CentroidIndexMeta,
        num_docs: usize,
        offsets_slice: FileSlice,
        bounds_slice: FileSlice,
    ) -> crate::Result<Self> {
        let num_centroids = routing.num_clusters();
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
            centroid_index,
            routing,
            num_docs,
            cluster_offsets,
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

    pub(crate) fn centroid_index_meta(&self) -> &CentroidIndexMeta {
        &self.centroid_index
    }

    pub fn num_clusters(&self) -> usize {
        self.routing.num_clusters()
    }

    pub fn router(&self) -> RouterKind {
        self.routing.router()
    }

    /// Distinct docs with a vector.
    pub(crate) fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Total posting rows across all clusters.
    pub fn num_rows(&self) -> usize {
        self.cluster_offset(self.num_clusters()) as usize
    }

    fn cluster_offset(&self, cluster: usize) -> u64 {
        let start = cluster * mem::size_of::<u64>();
        let end = start + mem::size_of::<u64>();
        u64::from_le_bytes(self.cluster_offsets[start..end].try_into().unwrap())
    }

    /// The contiguous row range of `cluster` within the `.vec` rows.
    #[inline]
    pub fn cluster_range(&self, cluster: usize) -> Range<usize> {
        debug_assert!(cluster < self.num_clusters(), "cluster out of bounds");
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
        (0..self.num_clusters()).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests only. Routing fetches per-node ranges through the lazy arena.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        self.routing.centroid_bytes()
    }

    /// Rank this segment's clusters for `query`, nearest first. `params`
    /// steers the stacked router only (how many clusters the caller will
    /// probe and its recall target); other routers ignore it.
    pub(crate) fn rank_clusters<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
        params: RoutingParams,
    ) -> RouterIter<'router, 'workspace> {
        self.routing.rank_clusters(workspace, query, params)
    }

    /// The APS estimator for scanning `ranked` (from
    /// [`Self::rank_clusters`]) toward `recall`. `None`
    /// unless the stacked router ranked it and APS is on.
    pub(crate) fn recall_estimator(
        &self,
        ranked: &RouterIter<'_, '_>,
        query: &[f32],
        recall: f32,
    ) -> Option<RecallEstimator<'_>> {
        self.routing.recall_estimator(ranked, query, recall)
    }
}

impl RouterIndex {
    pub(crate) fn open(
        options: &VectorOptions,
        num_centroids: usize,
        centroids_slice: FileSlice,
        router_slice: FileSlice,
    ) -> crate::Result<Self> {
        let router = RouterKind::open(router_slice, centroids_slice.clone(), options)?;
        Ok(Self {
            num_centroids,
            centroids_slice,
            metric: options.metric(),
            router,
        })
    }

    pub(crate) fn num_clusters(&self) -> usize {
        self.num_centroids
    }

    pub(crate) fn router(&self) -> RouterKind {
        self.router.kind()
    }

    pub(crate) fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    pub(crate) fn rank_clusters<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
        params: RoutingParams,
    ) -> RouterIter<'router, 'workspace> {
        #[cfg(test)]
        if let Some(clusters) = crate::vector::router::test_clusters() {
            return clusters;
        }
        self.router.rank(workspace, query, self.metric, params)
    }

    pub(crate) fn recall_estimator(
        &self,
        ranked: &RouterIter<'_, '_>,
        query: &[f32],
        recall: f32,
    ) -> Option<RecallEstimator<'_>> {
        self.router
            .recall_estimator(ranked, query, self.metric, recall)
    }
}

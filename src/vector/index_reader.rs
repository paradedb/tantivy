//! Per-(segment, field) vector reader, modeled on
//! [`InvertedIndexReader`](crate::index::InvertedIndexReader).
//!
//! One [`VectorIndexReader`] serves one vector field of one segment. It is
//! opened (and cached) via
//! [`SegmentReader::vector_index`](crate::SegmentReader::vector_index), the
//! same shape as `SegmentReader::inverted_index`: small routing state is
//! parsed once and pinned in memory, while the bulk payload stays behind
//! [`FileSlice`]s and is fetched with ranged reads at query time — under a
//! copying [`Directory`](crate::directory::Directory) (block storage rather
//! than mmap), only the bytes a query actually touches are materialized.
//!
//! The reader is "store + optional index":
//! - The **store** is the segment's `.vec` composite: slot `[0]` is the
//!   row→doc_id [`IdMap`], slot `[1]` the dense vector rows (deferred).
//! - The **index** ([`IvfIndex`]) exists iff the segment was merged with IVF
//!   clustering, and is loaded from the sibling `.centroids` composite:
//!   centroids + `num_docs` in slot `[0]`, the cluster offsets prefix sum in
//!   slot `[1]`, and (for segments with enough centroids to need routing)
//!   the RNG routing graph in slot `[2]`. Its job is to tell a query which
//!   clusters — contiguous row ranges of slot `[1]` — to probe. Without it,
//!   search falls back to an exact scan.
//!
//! The pairing is an invariant of the write path (`VectorPlugin`): the IVF
//! merge writes cluster-sorted rows + an `Explicit` id-map + `.centroids`;
//! every other writer produces doc-ordered rows + `Identity`/`Bitmap` and no
//! sidecar. [`VectorIndexReader::open`] validates the two signals agree and
//! everything downstream relies on it.

use std::cmp::Ordering;
use std::ops::Range;

use common::{HasLen, OwnedBytes};

use super::flat::IdMap;
use super::header::read_header;
use super::ivf::{CentroidsMeta, CENTROIDS_EXT};
use super::{
    evenly_spaced_seeds, FileSliceArena, NeighborhoodGraphConfig, NodeId,
    RelativeNeighborhoodGraph, VectorArena, Workspace, VEC_EXT,
};
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, Metric, VectorDType, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

/// Which on-disk layout a segment's vector data uses. Purely descriptive —
/// derived from whether the field has an [`IvfIndex`] — and surfaced through
/// [`VectorInfo`] for tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    Flat,
    Ivf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    pub format: VectorStorageFormat,
    /// Distinct documents with a vector in this field — the same quantity for
    /// both storage formats. The IVF per-cluster numbers (`cluster_stats`
    /// here, [`SegmentReader::vector_cluster_sizes`]) count memberships —
    /// posting rows, one per cell a doc lives in — so with replication their
    /// sum exceeds `num_vectors`.
    pub num_vectors: usize,
    pub num_centroids: Option<usize>,
    pub cluster_stats: Option<VectorClusterStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_cluster_size: f64,
    pub empty_clusters: usize,
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct docs with a vector — resolved once at open (`IdMap` row count
    /// for flat storage, the persisted doc count for IVF, whose row total is
    /// inflated by replication).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    /// `.vec` slot `[0]`: row→doc_id map. `Identity`/`Bitmap` ⟺ doc-ordered
    /// rows and no index; `Explicit` ⟺ cluster-sorted rows and `Some` index.
    id_map: IdMap,
    index: Option<IvfIndex>,
}

impl VectorIndexReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// [`empty`](Self::empty) placeholder when the segment carries no vector
    /// data for the field (no `.vec` file, or the field has no slots in it —
    /// e.g. the segment predates the field), mirroring how
    /// `SegmentReader::inverted_index` returns an empty reader.
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment_reader.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(opts) => opts.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )))
            }
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (_version, body) = read_header(&vec_file)?;
        let vec_composite = CompositeFile::open(&body)?;
        let (Some(id_map_slice), Some(rows_slice)) = (
            vec_composite.open_read_with_idx(field, 0),
            vec_composite.open_read_with_idx(field, 1),
        ) else {
            return Ok(Self::empty(options));
        };
        let id_map = IdMap::open(id_map_slice, segment_reader.max_doc())?;

        // TODO: Add a version header to the centroids file
        let centroid_slots =
            match segment_reader.open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string())) {
                Ok(file) => {
                    let composite = CompositeFile::open(&file)?;
                    match (
                        composite.open_read_with_idx(field, 0),
                        composite.open_read_with_idx(field, 1),
                    ) {
                        // Slot [2] (the routing graph) is optional: the
                        // write side skips it for degenerate centroid
                        // counts, which route by linear scan instead.
                        (Some(centroids), Some(offsets)) => {
                            Some((centroids, offsets, composite.open_read_with_idx(field, 2)))
                        }
                        _ => None,
                    }
                }
                Err(OpenReadError::FileDoesNotExist(_)) => None,
                Err(err) => return Err(err.into()),
            };

        // The id-map variant and the `.centroids` sidecar are two signals of
        // one write-path decision; a mismatch means a corrupt segment, never a
        // fallback.
        let index = match (&id_map, centroid_slots) {
            (IdMap::Explicit(_), Some((centroids, offsets, graph))) => {
                Some(IvfIndex::open(centroids, offsets, graph, &options)?)
            }
            (IdMap::Explicit(_), None) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has cluster-sorted rows but no `.centroids` data",
                    entry.name()
                )))
            }
            (_, Some(_)) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has `.centroids` data but doc-ordered rows",
                    entry.name()
                )))
            }
            (_, None) => None,
        };

        let num_rows = id_map.num_rows() as usize;
        if let Some(index) = &index {
            if index.num_rows() != num_rows {
                return Err(TantivyError::InternalError(
                    "IVF id-map length does not match the cluster offsets".to_string(),
                ));
            }
        }
        // Cheap (length-only) corruption check on the deferred payload.
        if rows_slice.len() != num_rows * options.bytes_per_vector() {
            return Err(TantivyError::InternalError(format!(
                "vector rows length {} does not match {} rows of {} bytes",
                rows_slice.len(),
                num_rows,
                options.bytes_per_vector()
            )));
        }

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
        })
    }

    /// The no-data placeholder: zero vectors, no index. Every accessor
    /// behaves as an empty column, so callers never branch on presence.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            rows_slice: FileSlice::empty(),
            id_map: IdMap::Identity { num_docs: 0 },
            index: None,
        }
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered.
    /// `None` means search must scan the rows exactly.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
    pub(crate) fn info(&self) -> Option<VectorInfo> {
        if !self.present {
            return None;
        }
        let Some(index) = &self.index else {
            return Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: self.num_vectors,
                num_centroids: None,
                cluster_stats: None,
            });
        };
        let mut empty_clusters = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;
        let mut total_cluster_size = 0;
        for cluster_size in index.cluster_sizes() {
            empty_clusters += usize::from(cluster_size == 0);
            min_cluster_size = min_cluster_size.min(cluster_size);
            max_cluster_size = max_cluster_size.max(cluster_size);
            total_cluster_size += cluster_size;
        }
        let num_centroids = index.num_clusters();
        let avg_cluster_size = if num_centroids == 0 {
            0.0
        } else {
            total_cluster_size as f64 / num_centroids as f64
        };
        let min_cluster_size = if num_centroids == 0 {
            0
        } else {
            min_cluster_size
        };
        Some(VectorInfo {
            format: VectorStorageFormat::Ivf,
            num_vectors: self.num_vectors,
            num_centroids: Some(num_centroids),
            cluster_stats: Some(VectorClusterStats {
                min_cluster_size,
                max_cluster_size,
                avg_cluster_size,
                empty_clusters,
            }),
        })
    }

    /// Raw per-cluster posting-list sizes in cluster order; `None` when the
    /// field's storage is not IVF.
    pub(crate) fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// The raw little-endian bytes of `doc_id`'s vector, fetched with one
    /// stride-sized ranged read; `None` if the doc has no vector.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        let stride = self.options.bytes_per_vector();
        let bytes = self
            .rows_slice
            .slice(row * stride..(row + 1) * stride)
            .read_bytes()?;
        Ok(Some(bytes))
    }

    /// The raw vector rows of `cluster`, contiguous in `.vec` row order,
    /// fetched with one ranged read. IVF storage only.
    pub fn cluster_vector_bytes(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        let Some(index) = &self.index else {
            return Err(TantivyError::InvalidArgument(
                "vector storage is not clustered".to_string(),
            ));
        };
        if cluster >= index.num_clusters() {
            return Err(TantivyError::InvalidArgument(format!(
                "cluster {cluster} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let rows = index.cluster_range(cluster);
        let bytes = self
            .rows_slice
            .slice(rows.start * stride..rows.end * stride)
            .read_bytes()?;
        Ok(bytes)
    }

    /// The doc id stored at `row` of the cluster-sorted permutation, decoded
    /// on demand from the pinned `Explicit` id-map. IVF storage only.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let IdMap::Explicit(bytes) = &self.id_map else {
            unreachable!("doc_id_at is only meaningful for cluster-sorted (IVF) storage");
        };
        let start = row * std::mem::size_of::<DocId>();
        DocId::from_le_bytes(
            bytes[start..start + std::mem::size_of::<DocId>()]
                .try_into()
                .unwrap(),
        )
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if the storage is
    /// not IVF or `cluster` is out of bounds.
    pub fn cluster_doc_ids(&self, cluster: usize) -> Option<Vec<DocId>> {
        let index = self.index.as_ref()?;
        if cluster >= index.num_clusters() {
            return None;
        }
        Some(
            index
                .cluster_range(cluster)
                .map(|row| self.doc_id_at(row))
                .collect(),
        )
    }

    /// Doc → dense row. For clustered storage, rows are cluster-sorted and
    /// ascending by doc id within each cluster, so this scans clusters and
    /// binary-searches each one over the pinned id-map bytes.
    fn row_id(&self, doc_id: DocId) -> Option<usize> {
        match &self.id_map {
            IdMap::Identity { num_docs } => (doc_id < *num_docs).then_some(doc_id as usize),
            IdMap::Bitmap(_) => self.id_map.rank_if_exists(doc_id).map(|row| row as usize),
            IdMap::Explicit(_) => {
                let index = self.index.as_ref()?;
                for cluster in 0..index.num_clusters() {
                    let rows = index.cluster_range(cluster);
                    let mut lo = rows.start;
                    let mut hi = rows.end;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        match self.doc_id_at(mid).cmp(&doc_id) {
                            Ordering::Less => lo = mid + 1,
                            Ordering::Greater => hi = mid,
                            Ordering::Equal => return Some(mid),
                        }
                    }
                }
                None
            }
        }
    }
}

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and
/// the RNG adjacency (edges only, ~`k × max_edges × 4` bytes). The centroid
/// vectors stay behind a [`FileSliceArena`] and are fetched one node at a
/// time as routing visits them. Everything row-scale stays out entirely (the
/// rows and id-map live on [`VectorIndexReader`]).
pub struct IvfIndex {
    meta: CentroidsMeta,
    dim: usize,
    metric: Metric,
    /// The persisted RNG over the centroids (`.centroids` slot `[2]`),
    /// reloaded search-only over the lazy centroid arena. `None` for
    /// degenerate centroid counts, where the write side skips the slot and
    /// routing falls back to a linear scan of the (few) centroids.
    graph: Option<RelativeNeighborhoodGraph<FileSliceArena<f32>>>,
    /// Evenly spaced routing entry points — the same formula the write-side
    /// replica selector searches with.
    seeds: Vec<NodeId>,
}

impl IvfIndex {
    fn open(
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        graph_slice: Option<FileSlice>,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        let meta = CentroidsMeta::open(centroids_slice, offsets_slice, options)?;
        match options.dtype() {
            VectorDType::F32 => {}
        } // exhaustive: a new dtype must pick its arena element type here
        let graph = match graph_slice {
            Some(slice) => {
                // The adjacency is pinned; the centroid rows behind the
                // arena are not. Node count and adjacency length are
                // cross-validated against the arena inside `Graph::open`.
                let adjacency = slice.read_bytes()?;
                Some(RelativeNeighborhoodGraph::open(
                    &adjacency,
                    FileSliceArena::<f32>::new(meta.centroids_slice.clone()),
                    options.dim(),
                    options.metric(),
                    NeighborhoodGraphConfig::default(),
                )?)
            }
            None => None,
        };
        let seeds = evenly_spaced_seeds(meta.num_centroids);
        Ok(IvfIndex {
            meta,
            dim: options.dim(),
            metric: options.metric(),
            graph,
            seeds,
        })
    }

    pub fn num_clusters(&self) -> usize {
        self.meta.num_centroids
    }

    /// Distinct docs with a vector (the persisted count; replication inflates
    /// the row total, [`Self::num_rows`]).
    pub(crate) fn num_docs(&self) -> usize {
        self.meta.num_docs
    }

    /// Total posting rows across all clusters — memberships, counting a
    /// replicated doc once per cell it lives in.
    pub fn num_rows(&self) -> usize {
        self.meta.num_rows()
    }

    /// The contiguous row range of `cluster` within the `.vec` rows.
    #[inline]
    pub fn cluster_range(&self, cluster: usize) -> Range<usize> {
        debug_assert!(cluster < self.meta.num_centroids, "cluster out of bounds");
        self.meta.cluster_offset(cluster) as usize..self.meta.cluster_offset(cluster + 1) as usize
    }

    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.meta.cluster_sizes()
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests. The search path never calls this; routing fetches per-node
    /// ranges through the graph's lazy arena instead.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.meta.centroids_slice.read_bytes()?)
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
    /// paths fetch centroid rows lazily through the same
    /// [`FileSliceArena`] and score with the same kernels, so their
    /// rankings agree.
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
                let num_centroids = self.meta.num_centroids;
                let arena = FileSliceArena::<f32>::new(self.meta.centroids_slice.clone());
                let mut ranked: Vec<(f32, u32)> = (0..num_centroids)
                    .map(|cluster| {
                        let sim =
                            arena.similarity(self.metric, self.dim, cluster as NodeId, query);
                        (sim.score(), cluster as u32)
                    })
                    .collect();
                ranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                ranked.truncate(limit);
                (ranked, num_centroids)
            }
        }
    }
}

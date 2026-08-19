//! The index-level centroid set: the consumer-provided [`CentroidIndex`]
//! trait and the immutable `centroids.<version>` file it is serialized into.
//!
//! Centroids are an index-level artifact, installed at index creation like
//! the schema and settings: the consumer trains them externally (over the
//! whole corpus, not a per-segment shard) and hands them over through
//! [`CentroidIndex`]. The file is written once, before the first
//! `meta.json` references it, and never mutates — segments record which
//! set version they assigned against, so a future re-publish (background
//! reclustering, SPFresh-style maintenance) is an additive operation.
//!
//! On-disk layout: the 4-byte vector format header, the set's `u64`
//! version, then a [`CompositeFile`] with per-field slots (see
//! `header::centroid_set_slot`):
//!
//! ```text
//! [0] num_centroids (u32) + centroid rows (C · stride, normalized here
//!     for Cosine so every downstream consumer scores the same bytes)
//! [1] router payload (OPTIONAL, absent for C <= 1): one router-kind tag
//!     byte, then the structure — tag 0 is tantivy's serialized
//!     RelativeNeighborhoodGraph, tags >= 128 are consumer-defined
//! ```

use std::io::Write;
use std::path::PathBuf;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::distance::{maybe_normalize_bytes, NormalizeOutcome};
use super::header::{centroid_set_slot, read_header, write_header, VectorFileVersion};
use super::ivf::assign::build_executor;
use super::ivf::{
    decode_row, encode_vector, Candidate, IvfCentroids, IvfSearchMetrics, NeighborhoodGraphConfig,
    NodeId, RelativeNeighborhoodGraph, ResumableSearchIterator, Workspace,
};
use super::{FileSliceArena, VectorArena};
use crate::directory::{CompositeFile, CompositeWrite, Directory, FileSlice};
use crate::schema::{Field, FieldType, Metric, Schema, VectorDType, VectorOptions};
use crate::TantivyError;

/// Router-kind tag for tantivy's own RNG payload (the
/// [`CentroidIndex::serialize_router`] default). Consumer-defined router
/// structures must tag themselves `>= 128`.
pub const ROUTER_KIND_RNG: u8 = 0;

/// The consumer-provided centroid index, pulled once at index creation.
///
/// Implementors own centroid *training* entirely — tantivy never trains.
/// Segments assign against the serialized set with tantivy's internal
/// selector, so the only knobs here are the data itself: the rows, the
/// MVCC version, and (optionally) a custom routing structure.
pub trait CentroidIndex: Send + Sync + 'static {
    /// The set's version stamp. Recorded in `meta.json`, in the file name,
    /// and in every segment that assigns against this set — the hook for
    /// consumer-side MVCC once re-publishing exists.
    fn version(&self) -> u64;

    /// The centroids for `field`. Required for every vector field in the
    /// schema; erroring here fails index creation.
    fn centroids(&self, field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids>;

    /// Serialize the routing structure over the centroids into the router
    /// slot. `centroids` are the normalized rows exactly as stored in the
    /// set file. The payload's first byte must be a router-kind tag; the
    /// default writes [`ROUTER_KIND_RNG`] followed by a serialized
    /// [`RelativeNeighborhoodGraph`](super::RelativeNeighborhoodGraph).
    /// Never called for degenerate sets (`C <= 1`), which route by linear
    /// scan and own no router slot.
    fn serialize_router(
        &self,
        _field: Field,
        options: &VectorOptions,
        centroids: &IvfCentroids,
        out: &mut dyn Write,
    ) -> crate::Result<()> {
        let IvfCentroids::F32(matrix) = centroids;
        out.write_all(&[ROUTER_KIND_RNG])?;
        let mut graph = RelativeNeighborhoodGraph::new(
            matrix.values.as_slice(),
            options.dim(),
            options.metric(),
            NeighborhoodGraphConfig::default(),
        );
        graph.build(&build_executor("router-rng-")?);
        graph.serialize(out)?;
        Ok(())
    }
}

/// The managed file name of the version-`version` centroid set.
pub(crate) fn centroid_set_filename(version: u64) -> PathBuf {
    PathBuf::from(format!("centroids.{version}"))
}

/// Pull every vector field's centroids from `provider`, validate and
/// normalize them, and write the `centroids.<version>` file. Called at
/// index creation, BEFORE the first `meta.json` references the file.
/// Returns the written file name.
pub(crate) fn write_centroid_set(
    directory: &dyn Directory,
    schema: &Schema,
    provider: &dyn CentroidIndex,
) -> crate::Result<PathBuf> {
    let path = centroid_set_filename(provider.version());
    let mut write = directory.open_write(&path)?;
    write_header(&mut write)?;
    provider.version().serialize(&mut write)?;
    let mut composite = CompositeWrite::wrap(write);

    for (field, entry) in schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let centroids = provider.centroids(field, opts)?;
        let IvfCentroids::F32(matrix) = &centroids;
        if matrix.dims != opts.dim() {
            return Err(TantivyError::InvalidArgument(format!(
                "CentroidIndex produced centroids with {} dimensions for field '{}', expected {}",
                matrix.dims,
                entry.name(),
                opts.dim()
            )));
        }
        if matrix.values.len() != matrix.rows * matrix.dims {
            return Err(TantivyError::InvalidArgument(format!(
                "CentroidIndex produced {} centroid values for {} rows x {} dimensions in field \
                 '{}'",
                matrix.values.len(),
                matrix.rows,
                matrix.dims,
                entry.name()
            )));
        }
        if matrix.rows == 0 {
            return Err(TantivyError::InvalidArgument(format!(
                "CentroidIndex produced no centroids for field '{}'",
                entry.name()
            )));
        }
        u32::try_from(matrix.rows).map_err(|_| {
            TantivyError::InvalidArgument(format!(
                "CentroidIndex produced more than u32::MAX centroids for field '{}'",
                entry.name()
            ))
        })?;

        // Normalize INTO the stored bytes for Cosine, so the segment
        // bounds folds and the future search path all score the exact
        // bytes written here. Non-finite centroids are a hard creation
        // error — this is consumer input at its validation boundary. A
        // zero-norm row under Cosine stays as-is; assignment tolerates it
        // and the segment bounds fold saturates its cluster.
        let mut normalized_values: Vec<f32> = Vec::with_capacity(matrix.values.len());
        let mut centroid_bytes = Vec::with_capacity(matrix.rows * opts.bytes_per_vector());
        for (centroid_ord, centroid) in matrix.values.chunks_exact(opts.dim()).enumerate() {
            let mut bytes = encode_vector(centroid, opts.dim())?;
            if maybe_normalize_bytes(opts, &mut bytes) == NormalizeOutcome::NonFinite {
                return Err(TantivyError::InvalidArgument(format!(
                    "CentroidIndex produced a non-finite centroid (ord {centroid_ord}) for field \
                     '{}'",
                    entry.name()
                )));
            }
            normalized_values.extend_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
            centroid_bytes.extend_from_slice(&bytes);
        }

        {
            let centroids_w = composite.for_field_with_idx(field, centroid_set_slot::CENTROIDS);
            (matrix.rows as u32).serialize(centroids_w)?;
            centroids_w.write_all(&centroid_bytes)?;
            centroids_w.flush()?;
        }

        // The router slot: skipped for degenerate centroid counts, where
        // routing is a linear scan a structure cannot beat.
        if matrix.rows > 1 {
            let normalized = IvfCentroids::F32(super::ivf::IvfMatrix {
                values: normalized_values,
                rows: matrix.rows,
                dims: matrix.dims,
            });
            let router_w = composite.for_field_with_idx(field, centroid_set_slot::ROUTER);
            let mut sink = RouterSlotWriter {
                out: router_w,
                written: 0,
            };
            provider.serialize_router(field, opts, &normalized, &mut sink)?;
            if sink.written == 0 {
                return Err(TantivyError::InvalidArgument(format!(
                    "CentroidIndex::serialize_router wrote no router payload for field '{}'; the \
                     payload must start with a router-kind tag byte",
                    entry.name()
                )));
            }
            sink.out.flush()?;
        }
    }
    composite.close()?;
    Ok(path)
}

/// `Write` adapter counting the router payload so an empty (tag-less)
/// payload is caught at creation instead of at open.
struct RouterSlotWriter<W: Write> {
    out: W,
    written: usize,
}

impl<W: Write> Write for RouterSlotWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.out.write(buf)?;
        self.written += n;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// Copy the source index's newest centroid set file VERBATIM into
/// `directory` — same version, same normalized rows, same router payload —
/// so a sibling index (e.g. an in-memory index staging a mutable segment)
/// assigns against byte-identical centroids and stamps the same version.
/// Returns the copied set's meta entry inputs `(version, filename)`.
pub(crate) fn copy_centroid_set(
    source: &crate::Index,
    directory: &dyn Directory,
    schema: &Schema,
) -> crate::Result<(u64, PathBuf)> {
    let metas = source.load_metas()?;
    let newest = metas
        .centroid_sets
        .iter()
        .max_by_key(|set| set.version)
        .ok_or_else(|| {
            TantivyError::InvalidArgument(
                "the source index has no centroid set to share".to_string(),
            )
        })?;
    let source_path = std::path::Path::new(&newest.filename);
    let bytes = source.directory().open_read(source_path)?.read_bytes()?;
    let path = centroid_set_filename(newest.version);
    let mut write = directory.open_write(&path)?;
    write.write_all(&bytes)?;
    common::TerminatingWrite::terminate(write)?;
    // The copy is verbatim, but the DESTINATION schema decides what must be
    // in it: every vector field needs a slot (field ids must line up, which
    // sharing between same-schema siblings guarantees).
    let reader = CentroidSetReader::open(directory, &path)?;
    for (field, entry) in schema.fields() {
        if let FieldType::Vector(opts) = entry.field_type() {
            reader.field_rows(field, opts)?;
        }
    }
    Ok((newest.version, path))
}

/// Reader over one `centroids.<version>` file.
pub(crate) struct CentroidSetReader {
    version: u64,
    composite: CompositeFile,
}

impl CentroidSetReader {
    /// Open the set file named `filename` (from the meta's
    /// `centroid_sets` record) in `directory`.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
    ) -> crate::Result<Self> {
        let file = directory.open_read(filename)?;
        let (version_stamp, body) = read_header(&file)?;
        if version_stamp < VectorFileVersion::V3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file {} is not a centroid set file", filename.display()),
            )
            .into());
        }
        let set_version_len = std::mem::size_of::<u64>();
        let version_bytes = body.slice_to(set_version_len).read_bytes()?;
        let version = u64::deserialize(&mut version_bytes.as_slice())?;
        let composite = CompositeFile::open(&body.slice_from(set_version_len))?;
        Ok(CentroidSetReader { version, composite })
    }

    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// The stored centroids of `field`. Every vector field is validated to
    /// have a slot at creation, so absence here is corruption or a
    /// schema/set mismatch, not an old file.
    pub(crate) fn field_centroids(
        &self,
        field: Field,
        options: &VectorOptions,
    ) -> crate::Result<FieldCentroids> {
        let Some(slice) = self
            .composite
            .open_read_with_idx(field, centroid_set_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid set v{} has no centroids for field {field:?}; the set does not match \
                 the schema",
                self.version
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid set slot is smaller than its count word",
            )
            .into());
        }
        let count_bytes = slice.slice_to(count_len).read_bytes()?;
        let num_centroids = u32::deserialize(&mut count_bytes.as_slice())? as usize;
        let stride = options.bytes_per_vector();
        let rows = slice.slice_from(count_len).read_bytes()?;
        if rows.len() != num_centroids * stride {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid set byte length mismatch",
            )
            .into());
        }
        Ok(FieldCentroids {
            num_centroids,
            stride,
            rows,
        })
    }

    /// The raw router slot of `field`, if the set carries one (absent for
    /// degenerate `C <= 1` sets). First byte is the router-kind tag.
    pub(crate) fn router_slice(&self, field: Field) -> Option<FileSlice> {
        self.composite
            .open_read_with_idx(field, centroid_set_slot::ROUTER)
    }

    /// The centroid count and the rows as a lazy [`FileSlice`] (past the
    /// count word) — the search path's view, which never materializes the
    /// rows whole.
    pub(crate) fn field_rows(
        &self,
        field: Field,
        options: &VectorOptions,
    ) -> crate::Result<(usize, FileSlice)> {
        let Some(slice) = self
            .composite
            .open_read_with_idx(field, centroid_set_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid set v{} has no centroids for field {field:?}; the set does not match \
                 the schema",
                self.version
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid set slot is smaller than its count word",
            )
            .into());
        }
        let count_bytes = slice.slice_to(count_len).read_bytes()?;
        let num_centroids = u32::deserialize(&mut count_bytes.as_slice())? as usize;
        let rows = slice.slice_from(count_len);
        if rows.len() != num_centroids * options.bytes_per_vector() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid set byte length mismatch",
            )
            .into());
        }
        Ok((num_centroids, rows))
    }
}

/// One field's centroids, materialized from the set file: the stored bytes
/// (for residual folds against the exact on-disk rows) plus a decoded f32
/// view (for the assignment selector).
pub(crate) struct FieldCentroids {
    num_centroids: usize,
    stride: usize,
    rows: OwnedBytes,
}

impl FieldCentroids {
    pub(crate) fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// The stored bytes of centroid `c`.
    pub(crate) fn centroid_bytes(&self, c: usize) -> &[u8] {
        &self.rows[c * self.stride..(c + 1) * self.stride]
    }

    /// All centroid rows decoded to a flat `dim`-strided f32 arena.
    pub(crate) fn values_f32(&self, options: &VectorOptions) -> crate::Result<Vec<f32>> {
        match options.dtype() {
            VectorDType::F32 => Ok(self
                .rows
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()),
        }
    }
}

/// A [`VectorArena`] over the set's stored rows, which are UNIT-NORM under
/// Cosine by construction (normalized at set creation): Cosine similarity
/// collapses to a raw dot product — no per-row norm recomputation, which
/// profiled at roughly half of routing cost — PROVIDED the query side is
/// unit-norm too. Callers uphold the query half: the search driver
/// normalizes the routing query once per query, and assignment vectors are
/// stored rows, normalized at ingest. L2/Dot delegate untouched.
pub(crate) struct UnitNormRowsArena(FileSliceArena<f32>);

impl VectorArena for UnitNormRowsArena {
    type Elem = f32;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        self.0.num_vectors(dim)
    }

    #[inline]
    fn similarity(
        &self,
        metric: Metric,
        dim: usize,
        node: NodeId,
        query: &[f32],
    ) -> crate::vector::Similarity {
        match metric {
            // dot(q̂, ĉ) == cosine(q, c) for unit-norm q̂ and ĉ.
            Metric::Cosine => self.0.similarity(Metric::Dot, dim, node, query),
            Metric::L2 | Metric::Dot => self.0.similarity(metric, dim, node, query),
        }
    }
}

/// The search-time view of one centroid set: per vector field, the lazy
/// centroid rows plus the parsed router. Opened once per set version and
/// cached on [`Index`](crate::Index) — the router adjacency alone is
/// `C × max_edges × 4` bytes, far too heavy to parse per query.
pub(crate) struct SetSearchIndex {
    /// The set version this view was opened for — the cache key, kept for
    /// assertions and diagnostics.
    #[allow(dead_code)]
    version: u64,
    fields: std::collections::HashMap<Field, FieldRouter>,
}

impl SetSearchIndex {
    /// Open the set file and parse every vector field's router.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
        schema: &Schema,
    ) -> crate::Result<Self> {
        let reader = CentroidSetReader::open(directory, filename)?;
        let mut fields = std::collections::HashMap::new();
        for (field, entry) in schema.fields() {
            let opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };
            let (num_centroids, rows_slice) = reader.field_rows(field, opts)?;
            let graph = match reader.router_slice(field) {
                Some(slice) if slice.len() >= 1 => {
                    let tag = slice.slice_to(1).read_bytes()?[0];
                    if tag == ROUTER_KIND_RNG {
                        let adjacency = slice.slice_from(1).read_bytes()?;
                        let arena = match opts.dtype() {
                            VectorDType::F32 => {
                                UnitNormRowsArena(FileSliceArena::<f32>::new(rows_slice.clone()))
                            }
                        };
                        Some(RelativeNeighborhoodGraph::open(
                            &adjacency,
                            arena,
                            opts.dim(),
                            opts.metric(),
                            NeighborhoodGraphConfig::default(),
                        )?)
                    } else if tag >= 128 {
                        // A consumer-defined router tantivy cannot read:
                        // route by exact scan, which is always correct.
                        log::warn!(
                            "field '{}' carries a consumer router (tag {tag}) in centroid set \
                             v{}; routing falls back to an exact centroid scan",
                            entry.name(),
                            reader.version(),
                        );
                        None
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("unknown reserved router-kind tag {tag}"),
                        )
                        .into());
                    }
                }
                // Degenerate sets (C <= 1) own no router slot; a linear
                // scan needs no structure. An empty slot is corrupt but
                // routes fine the same way.
                _ => None,
            };
            fields.insert(
                field,
                FieldRouter {
                    num_centroids,
                    dim: opts.dim(),
                    metric: opts.metric(),
                    rows_slice,
                    graph,
                },
            );
        }
        Ok(SetSearchIndex {
            version: reader.version(),
            fields,
        })
    }

    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    pub(crate) fn field_router(&self, field: Field) -> Option<&FieldRouter> {
        self.fields.get(&field)
    }
}

/// One field's routing state within a [`SetSearchIndex`]: says which
/// clusters a query should probe, index-wide — every segment shares these
/// cluster ids.
pub(crate) struct FieldRouter {
    num_centroids: usize,
    dim: usize,
    metric: Metric,
    /// The set's centroid rows, fetched per node through the lazy arena.
    rows_slice: FileSlice,
    /// The persisted RNG over the centroids. `None` for degenerate
    /// centroid counts or consumer routers, where routing falls back to a
    /// linear scan.
    graph: Option<RelativeNeighborhoodGraph<UnitNormRowsArena>>,
}

impl FieldRouter {
    pub(crate) fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// The persisted routing graph, shared with the assignment selector so
    /// segment builds never construct one of their own. `None` for
    /// degenerate sets and consumer-defined routers.
    pub(crate) fn graph(&self) -> Option<&RelativeNeighborhoodGraph<UnitNormRowsArena>> {
        self.graph.as_ref()
    }

    /// Clusters to probe for `query`, ranked lazily — a [`ClusterRanking`]
    /// yielding [`Candidate`]s best routing score first (graph node `c` *is*
    /// cluster `c`, so `Candidate::node` is the cluster id).
    ///
    /// Under Cosine, `query` MUST be unit-norm: both paths score through
    /// [`UnitNormRowsArena`], whose Cosine arm is a raw dot over the
    /// unit-norm stored rows.
    ///
    /// With a persisted RNG this is a resumable beam search
    /// ([`RelativeNeighborhoodGraph::search_iter`]): the first batch is one
    /// converged round at the configured `ef`, and pulling past it resumes
    /// the search, so routing cost is paid only as far as probing actually
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
                let arena = UnitNormRowsArena(FileSliceArena::<f32>::new(self.rows_slice.clone()));
                let mut ranked: Vec<Candidate> = (0..self.num_centroids)
                    .map(|cluster| Candidate {
                        sim: arena.similarity(self.metric, self.dim, cluster as NodeId, query),
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
/// returned by [`FieldRouter::rank_clusters`], which documents the two
/// paths.
pub(crate) enum ClusterRanking<'a> {
    /// Beam-searched routing over the persisted centroid RNG; pulling past a
    /// converged batch resumes the search.
    Graph(ResumableSearchIterator<'a, 'a, UnitNormRowsArena>),
    /// Exact fallback for router-less sets: every centroid scored and
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

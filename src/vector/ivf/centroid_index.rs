//! The index-level centroid index: the consumer-provided [`CentroidProducer`]
//! trait and the immutable `centroids` file it is serialized into.
//!
//! Centroids are an index-level artifact, installed at index creation like
//! the schema and settings: the consumer trains them externally (over the
//! whole corpus, not a per-segment shard) and hands them over through
//! [`CentroidProducer`]. The file is written once, before the first
//! `meta.json` references it, and never mutates — one set per index, for
//! the index's whole life. A future re-publish (background reclustering,
//! SPFresh-style maintenance) will need its own versioning design; none
//! exists today.
//!
//! On-disk layout: the 4-byte vector format header, then a
//! [`CompositeFile`] with per-field slots (see
//! `header::centroid_index_slot`):
//!
//! ```text
//! [0] num_centroids (u32) + centroid rows (C · stride, normalized here
//!     for Cosine so every downstream consumer scores the same bytes)
//! [1] required router-kind discriminant followed by the router payload
//! ```

use std::io::Write;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::{decode_row, encode_vector, IvfCentroids};
use crate::core::CENTROIDS_FILEPATH;
use crate::directory::{CompositeFile, CompositeWrite, Directory, FileSlice};
use crate::schema::{Field, FieldType, Metric, Schema, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::header::{centroid_index_slot, read_header, write_header, VectorFileVersion};
use crate::vector::router::{InMemoryRouter, LazyRouter, RouterIter, RouterKind, RouterWorkspace};
use crate::TantivyError;

/// The consumer-provided centroid producer, pulled once at index creation.
///
/// Implementors own centroid *training* entirely — tantivy never trains.
/// Segments assign against the serialized set through the index's selected
/// router, so this trait owns only the centroid data.
pub trait CentroidProducer: Send + Sync + 'static {
    /// The centroids for `field`. Required for every vector field in the
    /// schema; erroring here fails index creation.
    fn centroids(&self, field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids>;
}

impl dyn CentroidProducer {
    /// Pull every vector field's centroids, validate and normalize them,
    /// build the selected router, and write the [`CENTROIDS_FILEPATH`] file.
    pub(crate) fn serialize(
        &self,
        directory: &dyn Directory,
        schema: &Schema,
        router: RouterKind,
    ) -> crate::Result<()> {
        let mut write = directory.open_write(&CENTROIDS_FILEPATH)?;
        write_header(&mut write)?;
        let mut composite = CompositeWrite::wrap(write);

        for (field, entry) in schema.fields() {
            let opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };
            let mut centroids = self.centroids(field, opts)?;
            {
                let IvfCentroids::F32(matrix) = &centroids;
                if matrix.dims != opts.dim() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced centroids with {} dimensions for field '{}', \
                         expected {}",
                        matrix.dims,
                        entry.name(),
                        opts.dim()
                    )));
                }
                if matrix.values.len() != matrix.rows * matrix.dims {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced {} centroid values for {} rows x {} dimensions \
                         in field '{}'",
                        matrix.values.len(),
                        matrix.rows,
                        matrix.dims,
                        entry.name()
                    )));
                }
                if matrix.rows == 0 {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced no centroids for field '{}'",
                        entry.name()
                    )));
                }
                u32::try_from(matrix.rows).map_err(|_| {
                    TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced more than u32::MAX centroids for field '{}'",
                        entry.name()
                    ))
                })?;
            }

            // Normalize INTO the stored bytes for Cosine, so the segment
            // bounds folds and the future search path all score the exact
            // bytes written here. Non-finite centroids are a hard creation
            // error — this is consumer input at its validation boundary. A
            // zero-norm row under Cosine stays as-is; assignment tolerates it
            // and the segment bounds fold saturates its cluster.
            {
                let IvfCentroids::F32(matrix) = &mut centroids;
                for (centroid_ord, centroid) in
                    matrix.values.chunks_exact_mut(opts.dim()).enumerate()
                {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    if maybe_normalize_bytes(opts, &mut bytes) == NormalizeOutcome::NonFinite {
                        return Err(TantivyError::InvalidArgument(format!(
                            "CentroidProducer produced a non-finite centroid (ord {centroid_ord}) \
                             for field '{}'",
                            entry.name()
                        )));
                    }
                    centroid.copy_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                }
            }

            let router = InMemoryRouter::from(router, &router_options(opts), &mut centroids)?;
            let IvfCentroids::F32(matrix) = &centroids;

            let mut centroid_bytes = Vec::with_capacity(matrix.rows * opts.bytes_per_vector());
            for centroid in matrix.values.chunks_exact(opts.dim()) {
                centroid_bytes.extend_from_slice(&encode_vector(centroid, opts.dim())?);
            }
            {
                let centroids_w =
                    composite.for_field_with_idx(field, centroid_index_slot::CENTROIDS);
                (matrix.rows as u32).serialize(centroids_w)?;
                centroids_w.write_all(&centroid_bytes)?;
                centroids_w.flush()?;
            }
            let router_w = composite.for_field_with_idx(field, centroid_index_slot::ROUTER);
            router.serialize(router_w)?;
            router_w.flush()?;
        }
        composite.close()?;
        Ok(())
    }
}

fn router_options(options: &VectorOptions) -> VectorOptions {
    VectorOptions::new(options.dim(), routing_metric(options)).with_dtype(options.dtype())
}

pub(crate) fn routing_metric(options: &VectorOptions) -> Metric {
    match options.metric() {
        Metric::Cosine => Metric::Dot,
        metric => metric,
    }
}

/// Reader over one `centroids` file.
pub(crate) struct CentroidIndexReader {
    version: VectorFileVersion,
    composite: CompositeFile,
}

impl CentroidIndexReader {
    /// Open the set file named `filename` (from the meta's
    /// `centroid_index` record) in `directory`.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
    ) -> crate::Result<Self> {
        let file = directory.open_read(filename)?;
        let (version, body) = read_header(&file)?;
        if version < VectorFileVersion::V3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file {} is not a centroid index file", filename.display()),
            )
            .into());
        }
        let composite = CompositeFile::open(&body)?;
        Ok(CentroidIndexReader { version, composite })
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
            .open_read_with_idx(field, centroid_index_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid index has no centroids for field {field:?}; the set does not match the \
                 schema"
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index slot is smaller than its count word",
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
                "centroid index byte length mismatch",
            )
            .into());
        }
        Ok(FieldCentroids {
            num_centroids,
            stride,
            rows,
        })
    }

    pub(crate) fn router_slice(&self, field: Field) -> crate::Result<FileSlice> {
        self.composite
            .open_read_with_idx(field, centroid_index_slot::ROUTER)
            .ok_or_else(|| {
                TantivyError::InternalError(format!(
                    "centroid index has no router for field {field:?}"
                ))
            })
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
            .open_read_with_idx(field, centroid_index_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid index has no centroids for field {field:?}; the set does not match the \
                 schema"
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index slot is smaller than its count word",
            )
            .into());
        }
        let count_bytes = slice.slice_to(count_len).read_bytes()?;
        let num_centroids = u32::deserialize(&mut count_bytes.as_slice())? as usize;
        let rows = slice.slice_from(count_len);
        if rows.len() != num_centroids * options.bytes_per_vector() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index byte length mismatch",
            )
            .into());
        }
        Ok((num_centroids, rows))
    }
}

/// One field's centroids, materialized from the set file.
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
}

/// The search-time view of the centroid index: per vector field, the lazy
/// centroid rows plus the parsed router. Opened once and cached on
/// [`Index`](crate::Index) — the router adjacency alone is
/// `C × max_edges × 4` bytes, far too heavy to parse per query.
pub(crate) struct CachedCentroidIndex {
    fields: std::collections::HashMap<Field, FieldRouter>,
}

impl CachedCentroidIndex {
    /// Open the set file and parse every vector field's router.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
        schema: &Schema,
        router: RouterKind,
    ) -> crate::Result<Self> {
        let reader = CentroidIndexReader::open(directory, filename)?;
        let mut fields = std::collections::HashMap::new();
        for (field, entry) in schema.fields() {
            let opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };
            let (num_centroids, rows_slice) = reader.field_rows(field, opts)?;
            let routing_options = router_options(opts);
            let router = router.open(
                reader.version,
                reader.router_slice(field)?,
                rows_slice,
                &routing_options,
            )?;
            fields.insert(
                field,
                FieldRouter {
                    num_centroids,
                    routing_options,
                    router,
                },
            );
        }
        Ok(CachedCentroidIndex { fields })
    }

    pub(crate) fn field_router(&self, field: Field) -> Option<&FieldRouter> {
        self.fields.get(&field)
    }
}

/// One field's routing state within a [`CachedCentroidIndex`]: says which
/// clusters a query should probe, index-wide — every segment shares these
/// cluster ids.
pub(crate) struct FieldRouter {
    num_centroids: usize,
    routing_options: VectorOptions,
    router: LazyRouter,
}

impl FieldRouter {
    pub(crate) fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    pub(crate) fn router(&self) -> &LazyRouter {
        &self.router
    }

    pub(crate) fn routing_options(&self) -> &VectorOptions {
        &self.routing_options
    }

    pub(crate) fn rank_clusters<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
    ) -> RouterIter<'router, 'workspace> {
        self.router
            .rank(workspace, query, self.routing_options.metric())
    }
}

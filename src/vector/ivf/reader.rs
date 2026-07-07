//! Reader for the IVF vector format.
//!
//! Opens the segment's `.vec` composite (slot `[0]` = `IdMap::Explicit`
//! row→doc_id map, slot `[1]` = cluster-sorted rows) together with the
//! sibling `.centroids` composite (slot `[0]` = centroids, slot `[1]` =
//! cluster offsets), and hands out per-field [`IvfVectorColumn`] views.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::OnceLock;

use common::OwnedBytes;

use super::{CentroidsMeta, CENTROIDS_EXT};
use crate::directory::CompositeFile;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::vector::flat::IdMap;
use crate::vector::header::read_header;
use crate::vector::reader::VectorColumnReader;
use crate::vector::VEC_EXT;
use crate::{DocId, SegmentReader, TantivyError};

pub struct IvfVecReader {
    vec: CompositeFile,
    centroids: CompositeFile,
    field_options: BTreeMap<Field, VectorOptions>,
    max_doc: u32,
}

impl IvfVecReader {
    pub(crate) fn open(segment_reader: &SegmentReader) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let mut field_options = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                field_options.insert(field, opts.clone());
            }
        }
        let vec_file = segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?;
        let (_version, body) = read_header(&vec_file)?;
        let centroids_file =
            segment_reader.open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))?;
        Ok(Self {
            vec: CompositeFile::open(&body)?,
            centroids: CompositeFile::open(&centroids_file)?,
            field_options,
            max_doc: segment_reader.max_doc(),
        })
    }

    fn field_options(&self, field: Field) -> crate::Result<&VectorOptions> {
        self.field_options.get(&field).ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })
    }

    fn centroids_meta(&self, field: Field, options: &VectorOptions) -> crate::Result<CentroidsMeta> {
        let missing = || {
            TantivyError::InternalError(format!(
                "no IVF centroids for vector field {field:?} in segment"
            ))
        };
        let centroids_slice = self.centroids.open_read_with_idx(field, 0).ok_or_else(missing)?;
        let offsets_slice = self.centroids.open_read_with_idx(field, 1).ok_or_else(missing)?;
        Ok(CentroidsMeta::open(centroids_slice, offsets_slice, options)?)
    }
}

impl VectorColumnReader for IvfVecReader {
    type Column = IvfVectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<IvfVectorColumn> {
        let options = self.field_options(field)?.clone();
        let missing = || {
            TantivyError::InternalError(format!(
                "no IVF vector data for vector field {field:?} in segment"
            ))
        };
        let id_map_slice = self.vec.open_read_with_idx(field, 0).ok_or_else(missing)?;
        let rows_slice = self.vec.open_read_with_idx(field, 1).ok_or_else(missing)?;
        // Held as raw little-endian bytes, not a decoded `Vec`: the row→doc_id
        // map is only read for clusters a query actually probes.
        let row_doc_id_bytes = match IdMap::open(id_map_slice, self.max_doc)? {
            IdMap::Explicit(bytes) => bytes,
            _ => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {field:?} is not IVF (id-map is not Explicit)"
                )))
            }
        };
        let centroids = self.centroids_meta(field, &options)?;
        let num_vectors = centroids.num_vectors();
        if row_doc_id_bytes.len() != num_vectors * size_of::<DocId>() {
            return Err(TantivyError::InternalError(
                "IVF id-map length does not match the cluster offsets".to_string(),
            ));
        }
        let row_bytes = rows_slice.read_bytes()?;
        let cluster_assignments = (0..centroids.num_centroids).map(|_| OnceLock::new()).collect();
        Ok(IvfVectorColumn {
            row_doc_id_bytes,
            cluster_assignments,
            row_bytes,
            centroids,
            options,
        })
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        let options = self.field_options(field)?;
        Ok(self.centroids_meta(field, options)?.num_vectors())
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        Ok(self.field_options(field)?.dim())
    }
}

/// Per-segment, per-field IVF column view.
///
/// `row_doc_id_bytes` and `row_bytes` are parallel arrays in cluster-sorted
/// row order; `centroids.cluster_offset(c)` slices the row range of cluster
/// `c` out of both. The row→doc_id map is decoded lazily, one cluster at a
/// time, and cached in `cluster_assignments`.
pub struct IvfVectorColumn {
    row_doc_id_bytes: OwnedBytes,
    cluster_assignments: Vec<OnceLock<Vec<DocId>>>,
    row_bytes: OwnedBytes,
    centroids: CentroidsMeta,
    options: VectorOptions,
}

impl IvfVectorColumn {
    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    pub fn len(&self) -> usize {
        self.centroids.num_vectors()
    }

    pub fn is_empty(&self) -> bool {
        self.centroids.num_vectors() == 0
    }

    pub fn num_clusters(&self) -> usize {
        self.centroids.num_centroids
    }

    pub fn centroid_bytes(&self) -> &[u8] {
        &self.centroids.centroid_bytes
    }

    /// The doc id stored at `row`, decoded on demand (no upfront pass).
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let start = row * size_of::<DocId>();
        DocId::from_le_bytes(
            self.row_doc_id_bytes[start..start + size_of::<DocId>()]
                .try_into()
                .unwrap(),
        )
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if out of bounds.
    /// Decoded from the raw id-map on first access and cached per cluster, so a
    /// query that probes a few clusters never materializes the rest.
    pub fn cluster_doc_ids(&self, cluster: usize) -> crate::Result<Option<&[DocId]>> {
        if cluster >= self.centroids.num_centroids {
            return Ok(None);
        }
        let docs = self.cluster_assignments[cluster].get_or_init(|| {
            let start = self.centroids.cluster_offset(cluster) as usize;
            let end = self.centroids.cluster_offset(cluster + 1) as usize;
            (start..end).map(|row| self.doc_id_at(row)).collect()
        });
        Ok(Some(docs.as_slice()))
    }

    /// The raw vector rows of `cluster`, contiguous in `.vec` row order.
    pub fn cluster_vector_bytes(&self, cluster: usize) -> crate::Result<&[u8]> {
        if cluster >= self.centroids.num_centroids {
            return Err(TantivyError::InvalidArgument(format!(
                "cluster {cluster} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let start = (self.centroids.cluster_offset(cluster) as usize) * stride;
        let end = (self.centroids.cluster_offset(cluster + 1) as usize) * stride;
        self.row_bytes.get(start..end).ok_or_else(|| {
            TantivyError::InternalError("IVF cluster vector byte range out of bounds".to_string())
        })
    }

    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        let row_id = self.row_id(doc_id)?;
        let stride = self.options.bytes_per_vector();
        let start = row_id.checked_mul(stride)?;
        let end = start.checked_add(stride)?;
        self.row_bytes.get(start..end)
    }

    /// Doc → row. Rows are cluster-sorted, so we scan clusters and binary
    /// search each cluster's ascending doc-id slice (decoded lazily as
    /// scanned).
    fn row_id(&self, doc_id: DocId) -> Option<usize> {
        for cluster in 0..self.centroids.num_centroids {
            let start = self.centroids.cluster_offset(cluster) as usize;
            let docs = self.cluster_doc_ids(cluster).ok()??;
            if let Ok(local_row) = docs.binary_search(&doc_id) {
                return Some(start + local_row);
            }
        }
        None
    }
}

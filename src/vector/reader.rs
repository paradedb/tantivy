//! Per-segment vector storage dispatch.
//!
//! Reads the segment's `.vecmeta` to learn which storage format was written
//! (flat or IVF), opens the matching reader, and exposes a field's column via
//! [`VectorColumnReader::open_column`].

use std::collections::BTreeMap;

use super::flat::{FlatVecReader, FlatVectorColumn};
use super::ivf::{IvfVecReader, IvfVectorColumn};
use super::meta::{VectorSegmentMeta, VectorStorageFormat, VECMETA_EXT};
use crate::directory::error::OpenReadError;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType};
use crate::{DocId, SegmentReader, TantivyError};

pub trait VectorColumnReader {
    type Column;

    fn open_column(&self, field: Field) -> crate::Result<Self::Column>;

    fn count(&self, field: Field) -> crate::Result<usize>;

    fn dim(&self, field: Field) -> crate::Result<usize>;
}

pub struct VectorReader {
    storage: VectorStorageReader,
    vector_dims: BTreeMap<Field, usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    pub format: VectorStorageFormat,
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

enum VectorStorageReader {
    None,
    Flat(FlatVecReader),
    Ivf(IvfVecReader),
}

impl VectorReader {
    pub(crate) fn open(segment_reader: &SegmentReader) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let mut vector_dims = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                vector_dims.insert(field, opts.dim());
            }
        }
        let meta_slice =
            match segment_reader.open_read(SegmentComponent::Custom(VECMETA_EXT.to_string())) {
                Ok(file_slice) => Some(file_slice),
                Err(OpenReadError::FileDoesNotExist(_)) => None,
                Err(err) => return Err(err.into()),
            };
        let storage = if let Some(file_slice) = meta_slice {
            let meta = VectorSegmentMeta::open(file_slice)?;
            match meta.format {
                VectorStorageFormat::Flat => {
                    VectorStorageReader::Flat(FlatVecReader::open(segment_reader)?)
                }
                VectorStorageFormat::Ivf => {
                    VectorStorageReader::Ivf(IvfVecReader::open(segment_reader, meta.payload)?)
                }
            }
        } else {
            VectorStorageReader::None
        };
        Ok(Self {
            storage,
            vector_dims,
        })
    }

    pub fn info(&self, field: Field) -> crate::Result<Option<VectorInfo>> {
        if !self.vector_dims.contains_key(&field) {
            return Ok(None);
        }
        match &self.storage {
            VectorStorageReader::None => Ok(None),
            VectorStorageReader::Flat(reader) => Ok(Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: reader.count(field)?,
                num_centroids: None,
                cluster_stats: None,
            })),
            VectorStorageReader::Ivf(reader) => {
                let meta = reader.field_meta(field)?;
                let mut empty_clusters = 0;
                let mut min_cluster_size = usize::MAX;
                let mut max_cluster_size = 0;
                let mut total_cluster_size = 0;
                for cluster_size in meta.cluster_sizes() {
                    empty_clusters += usize::from(cluster_size == 0);
                    min_cluster_size = min_cluster_size.min(cluster_size);
                    max_cluster_size = max_cluster_size.max(cluster_size);
                    total_cluster_size += cluster_size;
                }
                let avg_cluster_size = if meta.num_centroids == 0 {
                    0.0
                } else {
                    total_cluster_size as f64 / meta.num_centroids as f64
                };
                let min_cluster_size = if meta.num_centroids == 0 {
                    0
                } else {
                    min_cluster_size
                };
                Ok(Some(VectorInfo {
                    format: VectorStorageFormat::Ivf,
                    num_vectors: meta.num_vectors(),
                    num_centroids: Some(meta.num_centroids),
                    cluster_stats: Some(VectorClusterStats {
                        min_cluster_size,
                        max_cluster_size,
                        avg_cluster_size,
                        empty_clusters,
                    }),
                }))
            }
        }
    }
}

impl VectorColumnReader for VectorReader {
    type Column = VectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<VectorColumn> {
        if !self.vector_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        match &self.storage {
            VectorStorageReader::Flat(reader) => reader.open_column(field).map(VectorColumn::Flat),
            VectorStorageReader::Ivf(reader) => reader.open_column(field).map(VectorColumn::Ivf),
            VectorStorageReader::None => Err(TantivyError::InternalError(format!(
                "no vector data for vector field {field:?} in segment"
            ))),
        }
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        if !self.vector_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        match &self.storage {
            VectorStorageReader::Flat(reader) => reader.count(field),
            VectorStorageReader::Ivf(reader) => reader.count(field),
            VectorStorageReader::None => Err(TantivyError::InternalError(format!(
                "no vector data for vector field {field:?} in segment"
            ))),
        }
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        self.vector_dims.get(&field).copied().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })
    }
}

pub enum VectorColumn {
    Flat(FlatVectorColumn),
    Ivf(IvfVectorColumn),
}

impl VectorColumn {
    pub fn dim(&self) -> usize {
        match self {
            Self::Flat(column) => column.dim(),
            Self::Ivf(column) => column.dim(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Flat(column) => column.len(),
            Self::Ivf(column) => column.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Flat(column) => column.is_empty(),
            Self::Ivf(column) => column.is_empty(),
        }
    }

    pub fn contains(&self, doc_id: DocId) -> bool {
        match self {
            Self::Flat(column) => column.contains(doc_id),
            Self::Ivf(column) => column.contains(doc_id),
        }
    }

    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        match self {
            Self::Flat(column) => column.vector_bytes_at(doc_id),
            Self::Ivf(column) => column.vector_bytes_at(doc_id),
        }
    }
}

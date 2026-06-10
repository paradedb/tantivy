//! Per-segment vector storage dispatch.
//!
//! Reads the segment's `.vecmeta` to learn which storage format was written
//! (flat or IVF), opens the matching reader, and exposes a field's column via
//! [`VectorColumnReader::open_column`].

use std::collections::BTreeMap;

use super::flat::{FlatVecReader, FlatVectorColumn, FLATVEC_EXT};
use super::ivf::{IvfVecReader, IvfVectorColumn, ASSIGNMENTS_EXT, IVFVEC_EXT};
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
        let exists = |ext: &str| -> crate::Result<bool> {
            match segment_reader.open_read(SegmentComponent::Custom(ext.to_string())) {
                Ok(_) => Ok(true),
                Err(OpenReadError::FileDoesNotExist(_)) => Ok(false),
                Err(err) => Err(err.into()),
            }
        };
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
                    if exists(ASSIGNMENTS_EXT)? || exists(IVFVEC_EXT)? {
                        return Err(TantivyError::InternalError(
                            "segment is marked flat but has IVF vector components".to_string(),
                        ));
                    }
                    VectorStorageReader::Flat(FlatVecReader::open(segment_reader)?)
                }
                VectorStorageFormat::Ivf => {
                    if exists(FLATVEC_EXT)? {
                        return Err(TantivyError::InternalError(
                            "segment is marked IVF but has flat vector data".to_string(),
                        ));
                    }
                    if !exists(ASSIGNMENTS_EXT)? || !exists(IVFVEC_EXT)? {
                        return Err(TantivyError::InternalError(
                            "segment is marked IVF but is missing IVF vector components"
                                .to_string(),
                        ));
                    }
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

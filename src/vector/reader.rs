//! Per-segment vector storage dispatch.
//!
//! Opens the segment's `.vec` file (if present), learns the storage mode from
//! its self-describing `IdMap` header, and exposes a field's column via
//! [`VectorColumnReader::open_column`].

use std::collections::BTreeMap;

use super::flat::{FlatVecReader, FlatVectorColumn};
use super::VEC_EXT;
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
        // A `.vec` file is present iff this segment carries vector data. The
        // flat backend is the only mode for now; once IVF lands, the `IdMap`
        // header (`Explicit` ⟺ IVF) selects the reader.
        let storage =
            match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string())) {
                Ok(_) => VectorStorageReader::Flat(FlatVecReader::open(segment_reader)?),
                Err(OpenReadError::FileDoesNotExist(_)) => VectorStorageReader::None,
                Err(err) => return Err(err.into()),
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
}

impl VectorColumn {
    pub fn dim(&self) -> usize {
        match self {
            Self::Flat(column) => column.dim(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Flat(column) => column.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Flat(column) => column.is_empty(),
        }
    }

    pub fn contains(&self, doc_id: DocId) -> bool {
        match self {
            Self::Flat(column) => column.contains(doc_id),
        }
    }

    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        match self {
            Self::Flat(column) => column.vector_bytes_at(doc_id),
        }
    }
}

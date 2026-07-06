//! Per-segment dispatch over vector storage formats.
//!
//! Opens the segment's `.vec` file and learns the storage mode from its
//! self-describing `IdMap` header — `Explicit` ⟺ IVF (and a sibling
//! `.centroids` file), `Identity`/`Bitmap` ⟺ flat — then opens the matching
//! reader and exposes a field's column via [`VectorColumnReader::open_column`].

use std::collections::BTreeMap;

use super::flat::{FlatVecReader, FlatVectorColumn, IdMap};
use super::header::read_header;
use super::ivf::{IvfVecReader, IvfVectorColumn};
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, Schema};
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
        // A `.vec` file is present iff this segment carries vector data. The
        // `IdMap` variant of the first vector field is the mode discriminator:
        // `Explicit` ⟹ IVF (with a sibling `.centroids`), otherwise flat.
        let storage = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file_slice) => {
                if is_ivf(&file_slice, schema)? {
                    VectorStorageReader::Ivf(IvfVecReader::open(segment_reader)?)
                } else {
                    VectorStorageReader::Flat(FlatVecReader::open(segment_reader)?)
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => VectorStorageReader::None,
            Err(err) => return Err(err.into()),
        };
        Ok(Self {
            storage,
            vector_dims,
        })
    }
}

/// Peek the first vector field's `IdMap` tag in `.vec` to decide flat vs IVF.
fn is_ivf(vec_file: &FileSlice, schema: &Schema) -> crate::Result<bool> {
    let (_version, body) = read_header(vec_file)?;
    let composite = CompositeFile::open(&body)?;
    for (field, entry) in schema.fields() {
        if !matches!(entry.field_type(), FieldType::Vector(_)) {
            continue;
        }
        if let Some(id_map_slice) = composite.open_read_with_idx(field, 0) {
            return Ok(IdMap::peek_is_explicit(&id_map_slice)?);
        }
    }
    Ok(false)
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

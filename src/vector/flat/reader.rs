use std::any::Any;
use std::collections::BTreeMap;

use common::OwnedBytes;

use super::presence::Presence;
use crate::directory::CompositeFile;
use crate::index::SegmentComponent;
use crate::plugin::{PluginReader, PluginReaderContext};
use crate::schema::{Field, FieldType};
use crate::vector::reader::VectorColumnReader;
use crate::{DocId, TantivyError};

pub struct FlatVecReader {
    composite: CompositeFile,
    field_dims: BTreeMap<Field, usize>,
    max_doc: u32,
}

impl FlatVecReader {
    pub(crate) fn open(ctx: &PluginReaderContext) -> crate::Result<Self> {
        let mut field_dims = BTreeMap::new();
        for (field, entry) in ctx.schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                field_dims.insert(field, opts.dim());
            }
        }
        let composite = if field_dims.is_empty() {
            CompositeFile::empty()
        } else {
            match ctx
                .segment_reader
                .open_read(SegmentComponent::Custom(super::FLATVEC_EXT.to_string()))
            {
                Ok(file_slice) => CompositeFile::open(&file_slice)?,
                Err(_) => CompositeFile::empty(),
            }
        };
        Ok(Self {
            composite,
            field_dims,
            max_doc: ctx.segment_reader.max_doc(),
        })
    }

    pub(crate) fn has_column(&self, field: Field) -> bool {
        self.field_dims.contains_key(&field) && self.composite.open_read_with_idx(field, 0).is_some()
    }
}

impl VectorColumnReader for FlatVecReader {
    type Column = FlatVectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<FlatVectorColumn> {
        let dim = self.field_dims.get(&field).copied().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })?;
        let presence_slice = self.composite.open_read_with_idx(field, 0).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no flat vector data for vector field {field:?} in segment"
            ))
        })?;
        let rows_slice = self.composite.open_read_with_idx(field, 1).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no flat vector data for vector field {field:?} in segment"
            ))
        })?;
        let presence = Presence::open(presence_slice, self.max_doc)?;
        let row_bytes = rows_slice.read_bytes()?;
        Ok(FlatVectorColumn {
            presence,
            row_bytes,
            dim,
        })
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        self.field_dims.get(&field).ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })?;
        let presence_slice = self.composite.open_read_with_idx(field, 0).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no flat vector data for vector field {field:?} in segment"
            ))
        })?;
        let presence = Presence::open(presence_slice, self.max_doc)?;
        Ok(presence.num_non_null() as usize)
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        self.field_dims.get(&field).copied().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })
    }
}

/// A view over one vector field's data within a single segment.
///
/// Layout:
/// - `presence`: [`Presence::Full`] for dense columns (no bitmap stored, `row_id == doc_id`) or
///   [`Presence::Optional`] for sparse columns (rank-supporting bitmap).
/// - `row_bytes`: dense `f32` LE blob, exactly `presence.num_non_null()` rows of `dim` f32s each.
///
/// Lookup is `presence.rank_if_exists(doc) -> row_idx`, then
/// `&row_bytes[row_idx * dim * 4 ..]`. In the `Full` case the rank step
/// is the identity map — no bitmap consulted.
pub struct FlatVectorColumn {
    presence: Presence,
    row_bytes: OwnedBytes,
    dim: usize,
}

impl FlatVectorColumn {
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of docs that actually have a vector value.
    pub fn len(&self) -> usize {
        self.presence.num_non_null() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.presence.num_non_null() == 0
    }

    /// `true` if `doc_id` has a stored vector.
    #[inline]
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.presence.contains(doc_id)
    }

    /// Borrow the raw little-endian f32 bytes for a single document.
    ///
    /// Returns `None` if `doc_id` has no vector. The returned slice is a
    /// zero-copy borrow into the column's `OwnedBytes` (mmap page cache
    /// for `MmapDirectory`, refcounted in-memory blob for `RamDirectory`,
    /// or whatever the backing directory provides).
    #[inline]
    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        let row_id = self.presence.rank_if_exists(doc_id)? as usize;
        let stride = self.dim * 4;
        let start = row_id * stride;
        let end = start + stride;
        if end > self.row_bytes.len() {
            return None;
        }
        Some(&self.row_bytes[start..end])
    }
}

impl PluginReader for FlatVecReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

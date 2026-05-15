use std::any::Any;
use std::collections::BTreeMap;

use common::OwnedBytes;

use super::presence::Presence;
use crate::directory::CompositeFile;
use crate::index::SegmentComponent;
use crate::plugin::{PluginReader, PluginReaderContext};
use crate::schema::{Field, FieldType};
use crate::DocId;

pub struct FlatVecReader {
    composite: CompositeFile,
    field_dims: BTreeMap<Field, usize>,
    /// Segment's `max_doc`, captured at open time. Used to source
    /// `num_docs` for [`Presence::Full`] without storing it redundantly
    /// in the on-disk presence section.
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

    pub fn dim(&self, field: Field) -> Option<usize> {
        self.field_dims.get(&field).copied()
    }

    /// Open a field's vector data for sequential scan. The returned
    /// [`VectorColumn`] parses the field's presence bitmap and holds a
    /// zero-copy view of the dense row bytes. Returns `None` if the
    /// field isn't a vector field or the segment didn't write any data
    /// for it.
    pub fn open_column(&self, field: Field) -> Option<VectorColumn> {
        let dim = *self.field_dims.get(&field)?;
        let presence_slice = self.composite.open_read_with_idx(field, 0)?;
        let rows_slice = self.composite.open_read_with_idx(field, 1)?;
        let presence = Presence::open(presence_slice, self.max_doc).ok()?;
        let row_bytes = rows_slice.read_bytes().ok()?;
        Some(VectorColumn {
            presence,
            row_bytes,
            dim,
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
pub struct VectorColumn {
    presence: Presence,
    row_bytes: OwnedBytes,
    dim: usize,
}

impl VectorColumn {
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

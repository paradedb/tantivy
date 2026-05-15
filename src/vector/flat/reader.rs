//! Stub reader for the flat vector format.
//!
//! Until the writer lands, the unified [`VectorPlugin`](crate::vector::VectorPlugin)
//! isn't registered in `Index::default_plugins`, so this type is
//! never instantiated at runtime. `open_column` returns `None`, which
//! sends [`VectorBackend::for_segment`](crate::vector::VectorBackend::for_segment)
//! to an error path on the no-vector-data branch.

use std::any::Any;

use crate::plugin::{PluginReader, PluginReaderContext};
use crate::schema::Field;
use crate::DocId;

pub struct FlatVecReader {
    // TODO: composite file handle, per-field presence sections, dense
    // row blob, per-field dimension table.
}

impl FlatVecReader {
    pub(crate) fn open(_ctx: &PluginReaderContext) -> crate::Result<Self> {
        Ok(Self {})
    }

    pub fn dim(&self, _field: Field) -> Option<usize> {
        todo!("flat vector reader dim")
    }

    /// Open a per-field flat vector column. Stub returns `None`.
    pub fn open_column(&self, _field: Field) -> Option<VectorColumn> {
        None
    }
}

impl PluginReader for FlatVecReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Per-segment, per-field flat-vector column view.
///
/// Methods are `todo!()` placeholders — they document the surface that
/// [`FlatBackend`](crate::vector::VectorBackend) needs from the reader
/// once the writer/plugin land.
pub struct VectorColumn {
    // TODO: presence section (Full / Optional cardinality), zero-copy
    // borrow of the dense little-endian row bytes, per-row stride.
}

impl VectorColumn {
    pub fn dim(&self) -> usize {
        todo!("flat vector column dim")
    }

    pub fn len(&self) -> usize {
        todo!("flat vector column len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("flat vector column is_empty")
    }

    pub fn contains(&self, _doc_id: DocId) -> bool {
        todo!("flat vector column contains")
    }

    /// Borrow the raw little-endian element bytes for a single document.
    /// Returns `None` if `doc_id` has no vector.
    pub fn vector_bytes_at(&self, _doc_id: DocId) -> Option<&[u8]> {
        todo!("flat vector column lookup")
    }
}

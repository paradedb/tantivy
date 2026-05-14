//! Stub reader for the flat vector plugin.
//!
//! Until the writer lands, the plugin isn't registered, so
//! `segment_reader.plugin_reader::<FlatVecReader>("flat_vec")` always
//! returns `None`. The types here exist so that
//! [`VectorBackend`](super::super::backend::VectorBackend) can dispatch
//! over `Flat | Ivf` at compile time.

use std::any::Any;

use crate::plugin::PluginReader;
use crate::schema::Field;
use crate::DocId;

pub struct FlatVecReader {
    // TODO: composite file handle, per-field presence sections, dense
    // row blob, per-field dimension table.
}

impl FlatVecReader {
    pub(crate) fn stub() -> Self {
        Self {}
    }

    /// Open a per-field flat vector column. Stub returns `None` so the
    /// backend always errors out before exercising the flat path.
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
/// [`FlatBackend`](super::super::backend::FlatBackend) needs from the
/// reader once the writer/plugin land.
pub struct VectorColumn {
    // TODO: presence section (Full / Optional cardinality), zero-copy
    // borrow of the dense little-endian row bytes, per-row stride.
}

impl VectorColumn {
    /// Declared vector dimension for the field.
    pub fn dim(&self) -> usize {
        todo!("flat vector column dim")
    }

    /// Number of docs that actually have a vector value.
    pub fn len(&self) -> usize {
        todo!("flat vector column len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("flat vector column is_empty")
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, _doc_id: DocId) -> bool {
        todo!("flat vector column contains")
    }

    /// Borrow the raw little-endian element bytes for a single document.
    /// Returns `None` if `doc_id` has no vector.
    pub fn vector_bytes_at(&self, _doc_id: DocId) -> Option<&[u8]> {
        todo!("flat vector column lookup")
    }
}

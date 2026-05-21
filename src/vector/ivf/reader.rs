//! Stub reader for the IVF plugin.
//!
//! The plugin is registered (so backend dispatch and GC accounting
//! pick it up), but until the merge body lands no segment actually
//! has an `.ivfvec` file. `open_column` returns `None` accordingly,
//! which sends the backend to the flat fallback.

use std::any::Any;
use std::collections::BTreeMap;

use crate::plugin::PluginReader;
use crate::schema::{Field, FieldType, Schema};
use crate::vector::reader::VectorColumnReader;
use crate::{DocId, TantivyError};

pub struct IvfVecReader {
    // TODO: handles to the centroid index, assignment file, vector file,
    // and the per-cluster offsets metadata.
    field_dims: BTreeMap<Field, usize>,
}

impl IvfVecReader {
    pub(crate) fn stub(schema: &Schema) -> Self {
        let mut field_dims = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                field_dims.insert(field, opts.dim());
            }
        }
        Self { field_dims }
    }

}

impl VectorColumnReader for IvfVecReader {
    type Column = IvfVectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<IvfVectorColumn> {
        if !self.field_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        Err(TantivyError::InternalError(format!(
            "no IVF vector data for vector field {field:?} in segment"
        )))
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        if !self.field_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        Err(TantivyError::InternalError(format!(
            "no IVF vector data for vector field {field:?} in segment"
        )))
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        self.field_dims.get(&field).copied().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })
    }
}

impl PluginReader for IvfVecReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Per-segment, per-field IVF column view.
///
/// Methods are `todo!()` placeholders — they document the surface that
/// [`IvfBackend`](super::super::backend::IvfBackend) needs from the
/// reader once the writer/plugin land.
pub struct IvfVectorColumn {
    // TODO: centroid table, per-cluster doc-id postings, per-cluster
    // vector blob, cluster offset table.
}

impl IvfVectorColumn {
    pub fn dim(&self) -> usize {
        todo!("IVF vector column reader")
    }

    pub fn len(&self) -> usize {
        todo!("IVF vector column reader")
    }

    pub fn is_empty(&self) -> bool {
        todo!("IVF vector column reader")
    }

    pub fn contains(&self, _doc_id: DocId) -> bool {
        todo!("IVF vector column reader")
    }

    pub fn vector_bytes_at(&self, _doc_id: DocId) -> Option<&[u8]> {
        todo!("IVF vector column reader")
    }
}

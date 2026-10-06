use std::collections::BTreeMap;

use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::{DocId, TantivyError};

/// Buffers one vector field before serialization.
pub(super) struct FieldBuffer {
    pub(super) present_doc_ids: Vec<DocId>,
    pub(super) row_bytes: Vec<u8>,
    pub(super) opts: VectorOptions,
}

impl FieldBuffer {
    fn push_bytes(&mut self, doc_id: DocId, bytes: &[u8]) -> NormalizeOutcome {
        let stride = self.opts.bytes_per_vector();
        debug_assert_eq!(bytes.len(), stride);
        self.present_doc_ids.push(doc_id);
        let start = self.row_bytes.len();
        self.row_bytes.extend_from_slice(bytes);
        maybe_normalize_bytes(&self.opts, &mut self.row_bytes[start..start + stride])
    }

    fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.present_doc_ids.capacity() * std::mem::size_of::<DocId>()
            + self.row_bytes.capacity()
    }
}

pub(super) struct VectorBuffer {
    pub(super) fields: BTreeMap<Field, FieldBuffer>,
    pub(super) num_docs: DocId,
}

impl VectorBuffer {
    pub(super) fn for_schema(schema: &Schema) -> Self {
        let mut fields = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                fields.insert(
                    field,
                    FieldBuffer {
                        present_doc_ids: Vec::new(),
                        row_bytes: Vec::new(),
                        opts: opts.clone(),
                    },
                );
            }
        }
        Self {
            fields,
            num_docs: 0,
        }
    }

    pub(super) fn add_document(
        &mut self,
        doc_id: DocId,
        doc: &dyn ErasedDocument,
        schema: &Schema,
    ) -> crate::Result<()> {
        if self.fields.is_empty() {
            return Ok(());
        }
        self.num_docs = doc_id + 1;
        for (field, value) in doc.erased_fields() {
            let Some(buf) = self.fields.get_mut(&field) else {
                continue;
            };
            if buf.present_doc_ids.last() == Some(&doc_id) {
                continue;
            }
            let ErasedValue::Leaf(ReferenceValueLeaf::Bytes(bytes)) = value else {
                return Err(TantivyError::SchemaError(format!(
                    "Expected vector bytes for field {:?}",
                    schema.get_field_entry(field).name()
                )));
            };
            let stride = buf.opts.bytes_per_vector();
            if bytes.len() != stride {
                return Err(TantivyError::SchemaError(format!(
                    "vector byte length mismatch for field {:?}: expected {} bytes, got {}",
                    schema.get_field_entry(field).name(),
                    stride,
                    bytes.len(),
                )));
            }
            if buf.push_bytes(doc_id, bytes) == NormalizeOutcome::NonFinite {
                return Err(TantivyError::InvalidArgument(format!(
                    "non-finite element in vector field '{}' (doc {doc_id}): vectors must contain \
                     only finite values",
                    schema.get_field_entry(field).name(),
                )));
            }
        }
        Ok(())
    }

    pub(super) fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .fields
                .values()
                .map(FieldBuffer::mem_usage)
                .sum::<usize>()
    }
}

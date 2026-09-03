use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;

use super::id_map::IdMap;
use crate::directory::CompositeWrite;
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::PluginWriter;
use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::header::write_vector_header;
use crate::vector::VEC_EXT;
use crate::{DocId, TantivyError};

/// Buffers one vector field before serialization.
struct FieldBuffer {
    present_doc_ids: Vec<DocId>,
    row_bytes: Vec<u8>,
    opts: VectorOptions,
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

/// Writes full-precision vector rows.
pub struct FlatVecWriter {
    fields: BTreeMap<Field, FieldBuffer>,
    /// Number of documents represented by the presence map.
    num_docs: DocId,
}

impl FlatVecWriter {
    /// Creates a writer for all vector fields in a schema.
    pub fn for_schema(schema: &Schema) -> Self {
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
}

impl PluginWriter for FlatVecWriter {
    fn add_document(
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

    fn serialize(
        self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        if self.fields.is_empty() {
            return Ok(());
        }
        let mut write = segment.open_write(SegmentComponent::Custom(VEC_EXT.to_string()))?;
        write_vector_header(&mut write)?;
        let mut composite = CompositeWrite::wrap(write);

        for (field, buf) in &self.fields {
            let stride = buf.opts.bytes_per_vector();
            let (present, row_bytes): (Vec<DocId>, Vec<u8>) = if let Some(map) = doc_id_map {
                let mut p = Vec::new();
                let mut r = Vec::new();
                for (target_doc_id, source_doc_id) in map.iter_source_doc_ids().enumerate() {
                    if let Ok(row_idx) = buf.present_doc_ids.binary_search(&source_doc_id) {
                        p.push(target_doc_id as DocId);
                        let start = row_idx * stride;
                        r.extend_from_slice(&buf.row_bytes[start..start + stride]);
                    }
                }
                (p, r)
            } else {
                (buf.present_doc_ids.clone(), buf.row_bytes.clone())
            };

            let id_map_w = composite.for_field_with_idx(*field, 0);
            IdMap::serialize(&present, self.num_docs, id_map_w)?;
            id_map_w.flush()?;

            let rows_w = composite.for_field_with_idx(*field, 1);
            rows_w.write_all(&row_bytes)?;
            rows_w.flush()?;
        }
        composite.close()?;
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .fields
                .values()
                .map(FieldBuffer::mem_usage)
                .sum::<usize>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

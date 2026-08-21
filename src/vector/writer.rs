//! Per-commit vector writer: buffers raw vector bytes per doc and, at
//! segment finalize, writes the segment's `.vec` in the layout the index
//! dictates — clustered (assigned against the index-level centroid set)
//! when the index has one, flat (doc-ordered, searched exhaustively) when
//! it does not. The mutable/staging tier is exactly the no-set case.

use std::any::Any;
use std::collections::BTreeMap;

use super::centroid_set::CentroidSetReader;
use super::distance::{maybe_normalize_bytes, NormalizeOutcome};
use super::flat::write_flat_field;
use super::header::write_header;
use super::ivf::{write_ivf_field, IvfFieldWriteParams};
use super::VEC_EXT;
use crate::directory::CompositeWrite;
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::PluginWriter;
use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::{DocId, Executor, TantivyError};

/// Per-field in-memory state: the doc ids that have a value (ascending),
/// plus a dense byte array — analogous to fast-fields' `Optional`
/// cardinality. Docs without a vector occupy zero bytes of storage.
///
/// Rows are kept as raw little-endian bytes rather than decoded `T`
/// values. The bytes go in via `add_document` and come back out at
/// serialize time unchanged — no decode/re-encode round-trip, no
/// dependency on `T: VectorElement` in the writer.
struct FieldBuffer {
    /// Doc ids that have a value for this field, in insertion order
    /// (which equals ascending old-doc-id order since `add_document<D>`
    /// is called sequentially).
    present_doc_ids: Vec<DocId>,
    /// Dense byte blob: `row_bytes[i*stride..(i+1)*stride]` is the
    /// vector for `present_doc_ids[i]`.
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

pub struct VecWriter {
    fields: BTreeMap<Field, FieldBuffer>,
    /// Set by [`SegmentWriter::finalize`] before [`serialize`]. Used for
    /// the ascending-doc-order invariant checks.
    num_docs: DocId,
}

impl VecWriter {
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

impl PluginWriter for VecWriter {
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
            // Only the first value per field counts, matching `get_first` semantics.
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
            // NonFinite is a hard ingest error: bad data is rejected at the
            // boundary so merge and query never have to re-classify it. The
            // offending row is still in the buffer, but an add_document error
            // aborts the segment build — it is never serialized.
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
        if self
            .fields
            .values()
            .all(|buf| buf.present_doc_ids.is_empty())
        {
            // No rows anywhere: no `.vec` at all — the reader treats a
            // missing file as "no vector data".
            return Ok(());
        }

        let index = segment.index();
        let meta = index.load_metas()?;
        let set = match meta.centroid_set.as_ref() {
            Some(centroid_set) => {
                let set_search = index.centroid_set_search_index()?;
                let set_reader = CentroidSetReader::open(
                    index.directory(),
                    std::path::Path::new(&centroid_set.filename),
                )?;
                Some((set_search, set_reader))
            }
            None => None,
        };

        let mut write = segment.open_write(SegmentComponent::Custom(VEC_EXT.to_string()))?;
        write_header(&mut write)?;
        let mut composite = CompositeWrite::wrap(write);
        // Per-commit segments are small; assignment cost is dominated by
        // the merge path, which parallelizes. No thread pool here.
        let executor = Executor::single_thread();
        let cancel = || false;
        let schema = segment.schema();

        for (field, buf) in &self.fields {
            // Compute (present, row_bytes) in target doc-id order. For
            // the no-remap case the writer already accumulates in
            // ascending insertion (= target) order.
            let stride = buf.opts.bytes_per_vector();
            let (present, row_bytes): (Vec<DocId>, Vec<u8>) = if let Some(map) = doc_id_map {
                let mut p = Vec::new();
                let mut r = Vec::new();
                for (new_doc_id, old_doc_id) in map.iter_old_doc_ids().enumerate() {
                    if let Ok(row_idx) = buf.present_doc_ids.binary_search(&old_doc_id) {
                        p.push(new_doc_id as DocId);
                        let start = row_idx * stride;
                        r.extend_from_slice(&buf.row_bytes[start..start + stride]);
                    }
                }
                (p, r)
            } else {
                (buf.present_doc_ids.clone(), buf.row_bytes.clone())
            };
            if present.is_empty() {
                continue;
            }

            let Some((set_search, set_reader)) = &set else {
                write_flat_field(&mut composite, *field, &present, &row_bytes, self.num_docs)?;
                continue;
            };

            let field_centroids = set_reader.field_centroids(*field, &buf.opts)?;
            let params = IvfFieldWriteParams {
                router: set_search
                    .field_router(*field)
                    .and_then(|router| router.graph()),
                field: *field,
                opts: &buf.opts,
                set: &field_centroids,
                replicas: index.settings().vector_replicas,
                bounds_scope: index.settings().vector_bounds_scope,
                executor: &executor,
                cancel: &cancel,
                field_name: schema.get_field_entry(*field).name(),
            };
            write_ivf_field(
                &mut composite,
                &params,
                &mut |sink| {
                    for (row_idx, &doc_id) in present.iter().enumerate() {
                        sink(
                            doc_id,
                            row_idx as u64,
                            &row_bytes[row_idx * stride..(row_idx + 1) * stride],
                        )?;
                    }
                    Ok(())
                },
                &mut |handle, sink| {
                    let row_idx = handle as usize;
                    sink(&row_bytes[row_idx * stride..(row_idx + 1) * stride])
                },
            )?;
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

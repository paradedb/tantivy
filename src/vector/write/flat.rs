//! Flat vector segments: the per-document writer and the flat merge.

use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;

use common::CountingWriter;

use crate::directory::{CompositeWrite, Directory};
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::{PluginMergeContext, PluginWriter};
use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::distance::{maybe_normalize_bytes, row_is_finite, NormalizeOutcome};
use crate::vector::storage::blocks::{
    align_up, block_align, pad, write_metadata, BlockOffsetsWriter,
};
use crate::vector::storage::header::{write_vector_header, VectorEntry, HEADER_LEN, VEC_EXT};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::{VectorColMetadata, FLAT_ROWS_PER_BLOCK};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::RowAddress;
use crate::vector::ENTRY_ALIGN;
use crate::{DocId, TantivyError};

// --- Writer ---

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
        let row = &mut self.row_bytes[start..start + stride];
        // Every metric rejects NaN and infinities; normalization detects them through the norm.
        if !self.opts.needs_normalization() && !row_is_finite(&self.opts, row) {
            return NormalizeOutcome::NonFinite;
        }
        maybe_normalize_bytes(&self.opts, row)
    }

    fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.present_doc_ids.capacity() * std::mem::size_of::<DocId>()
            + self.row_bytes.capacity()
    }
}

/// Writes full-precision vector rows.
pub(crate) struct FlatVecWriter {
    fields: BTreeMap<Field, FieldBuffer>,
    /// Number of documents represented by the presence map.
    num_docs: DocId,
}

impl FlatVecWriter {
    /// Creates a writer for all vector fields in a schema.
    pub(crate) fn for_schema(schema: &Schema) -> Self {
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

        let mut id_maps = Vec::new();
        for (field, buf) in self.fields {
            // Compute (present, row_bytes) in target doc-id order. For
            // the no-remap case the writer already accumulates in
            // ascending insertion (= target) order.
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
                (buf.present_doc_ids, buf.row_bytes)
            };

            // Data entries precede IdMaps so their exact lengths exclude alignment padding.
            id_maps.push((field, present));
            let meta = VectorColMetadata::build_flat(&buf.opts);
            let align = block_align(&meta.slots());
            composite.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
            let data = composite.for_field_with_idx(field, VectorEntry::Data.index());
            let start = data.written_bytes();
            write_metadata(data, &meta)?;
            let mut block_offsets = BlockOffsetsWriter::new(data.written_bytes() - start);
            // A full row group or the final partial group forms one aligned Rows column.
            for block in row_bytes.chunks(FLAT_ROWS_PER_BLOCK as usize * stride) {
                data.write_all(block)?;
                pad(data, align_up(block.len(), align) - block.len())?;
                block_offsets.push(data.written_bytes() - start, block.len() / stride);
            }
            block_offsets.finish(data)?;
            assert_eq!((data.written_bytes() - start) as usize % ENTRY_ALIGN, 0);
            data.flush()?;
        }
        for (field, present) in id_maps {
            IdMap::serialize(
                &present,
                self.num_docs,
                composite.for_field_with_idx(field, VectorEntry::IdMap.index()),
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

// --- Merge ---

/// Merges source vectors into a flat target segment.
pub(crate) fn merge_flat(ctx: &PluginMergeContext) -> crate::Result<()> {
    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }
    if ctx.cancel.wants_cancel() {
        return Err(crate::TantivyError::Cancelled);
    }
    let path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let mut write = ctx.target_segment.index().directory().open_write(&path)?;
    write_vector_header(&mut write)?;
    let mut composite = CompositeWrite::wrap(write);

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum::<u32>();

    let mut id_maps = Vec::new();
    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };

        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;

        let source_rows = crate::vector::write::merge_source_rows(ctx, &field_readers)?;
        debug_assert_eq!(source_rows.len(), num_target_docs as usize);
        composite.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
        let rows_w = composite.for_field_with_idx(field, VectorEntry::Data.index());
        let target_present = write_flat_rows(
            rows_w,
            opts,
            &source_rows,
            &field_readers,
            RowBytes::Stored,
            ctx.cancel,
        )?;
        id_maps.push((field, target_present));
    }
    for (field, present) in id_maps {
        IdMap::serialize(
            &present,
            num_target_docs,
            composite.for_field_with_idx(field, VectorEntry::IdMap.index()),
        )?;
    }
    composite.close()?;
    Ok(())
}

/// The bytes [`write_flat_rows`] writes for each source row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowBytes {
    /// The row as stored, for a flat merge.
    Stored,
    /// The row re-normalized under the field's rule, as the encode pass normalizes it before
    /// encoding, so a quantized field's exact rows are the bytes its codes were built from.
    Normalized,
}

/// Streams the rows of `source_rows`, in target document order, into `rows_w` as one flat entry:
/// metadata, row groups of [`FLAT_ROWS_PER_BLOCK`] rows and the block offsets, each row's bytes as
/// `row_bytes` says. Row groups stream without knowing the final vector count. Returns the target
/// documents with a row, ascending.
pub(crate) fn write_flat_rows<W: Write>(
    rows_w: &mut CountingWriter<W>,
    opts: &VectorOptions,
    source_rows: &[Option<RowAddress>],
    readers: &[Arc<VectorIndexReader>],
    row_bytes: RowBytes,
    cancel: &dyn CancelSentinel,
) -> crate::Result<Vec<DocId>> {
    let normalize = row_bytes == RowBytes::Normalized && opts.needs_normalization();
    let mut normalized = Vec::new();
    let meta = VectorColMetadata::build_flat(opts);
    let align = block_align(&meta.slots());
    let start = rows_w.written_bytes();
    write_metadata(rows_w, &meta)?;
    let mut block_offsets = BlockOffsetsWriter::new(rows_w.written_bytes() - start);
    let mut block_bytes = 0;
    let mut present: Vec<DocId> = Vec::new();
    for (target_doc_id, source) in source_rows.iter().enumerate() {
        if cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        let Some(source) = source else {
            continue;
        };
        let bytes =
            readers[source.segment_ord as usize].vector_bytes_for_row(source.block_address)?;
        let row: &[u8] = if normalize {
            normalized.clear();
            normalized.extend_from_slice(&bytes);
            maybe_normalize_bytes(opts, &mut normalized);
            &normalized
        } else {
            &bytes
        };
        present.push(target_doc_id as DocId);
        rows_w.write_all(row)?;
        block_bytes += row.len();
        if present.len() % FLAT_ROWS_PER_BLOCK as usize == 0 {
            pad(rows_w, align_up(block_bytes, align) - block_bytes)?;
            block_offsets.push(rows_w.written_bytes() - start, FLAT_ROWS_PER_BLOCK as usize);
            block_bytes = 0;
        }
    }
    pad(rows_w, align_up(block_bytes, align) - block_bytes)?;
    if block_bytes != 0 {
        block_offsets.push(
            rows_w.written_bytes() - start,
            present.len() % FLAT_ROWS_PER_BLOCK as usize,
        );
    }
    block_offsets.finish(rows_w)?;
    assert_eq!((rows_w.written_bytes() - start) as usize % ENTRY_ALIGN, 0);
    rows_w.flush()?;
    Ok(present)
}

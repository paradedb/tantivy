//! Flat vector segments: the per-document writer and the flat merge.

use std::any::Any;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use common::CountingWriter;

use crate::directory::{CompositeWrite, Directory};
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::{PluginMergeContext, PluginWriter};
use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::distance::{
    admit, maybe_normalize_bytes, normalize_row_with_norm_squared, row_norm_squared, InvalidVector,
    MAX_NORM_SQUARED,
};
use crate::vector::quantization::VectorNormPolicy;
use crate::vector::storage::blocks::{
    align_up, block_align, pad, write_metadata, BlockOffsetsWriter,
};
use crate::vector::storage::header::{write_vector_header, VectorEntry, HEADER_LEN, VEC_EXT};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::{VectorColMetadata, FLAT_ROWS_PER_BLOCK};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::rows::StreamedRows;
use crate::vector::write::RowAddress;
use crate::vector::ENTRY_ALIGN;
use crate::{DocId, TantivyError};

// --- Admission ---

/// The error for a document whose vector the admission rule refuses.
fn refused_row(invalid: InvalidVector, field: &str, doc_id: DocId) -> TantivyError {
    TantivyError::InvalidArgument(match invalid {
        InvalidVector::NonFinite => format!(
            "non-finite element in vector field '{field}' (doc {doc_id}): vectors must contain \
             only finite values"
        ),
        InvalidVector::NormTooLarge => format!(
            "vector in field '{field}' (doc {doc_id}) has a squared norm above the limit of \
             {MAX_NORM_SQUARED:e}"
        ),
    })
}

/// The rule every row of a field passes on its way into storage: its byte length, the admission
/// of its squared norm, and unit normalization when the field stores unit rows.
pub(crate) struct RowAdmission {
    opts: VectorOptions,
    norm_policy: VectorNormPolicy,
}

impl RowAdmission {
    pub(crate) fn new(opts: &VectorOptions) -> Self {
        Self {
            opts: opts.clone(),
            norm_policy: VectorNormPolicy::for_options(opts),
        }
    }

    pub(crate) fn options(&self) -> &VectorOptions {
        &self.opts
    }

    /// Admits `doc_id`'s row of the field named `field` and appends it to `out`, normalized when
    /// the field stores unit rows. The row's one norm pass serves both the admission and the
    /// normalization; a refused row leaves `out` unchanged.
    pub(crate) fn admit(
        &self,
        field: &str,
        doc_id: DocId,
        bytes: &[u8],
        out: &mut Vec<u8>,
    ) -> crate::Result<()> {
        let stride = self.opts.bytes_per_vector();
        if bytes.len() != stride {
            return Err(TantivyError::SchemaError(format!(
                "vector byte length mismatch for field {:?}: expected {} bytes, got {}",
                field,
                stride,
                bytes.len(),
            )));
        }
        let norm_squared = row_norm_squared(self.opts.dtype(), bytes);
        admit(norm_squared, self.norm_policy)
            .map_err(|invalid| refused_row(invalid, field, doc_id))?;
        let start = out.len();
        out.extend_from_slice(bytes);
        match self.norm_policy {
            VectorNormPolicy::UnitL2 => {
                normalize_row_with_norm_squared(self.opts.dtype(), &mut out[start..], norm_squared);
            }
            VectorNormPolicy::None => {}
        }
        Ok(())
    }
}

// --- Flat entry ---

/// Writes one flat entry: the metadata, the rows in groups of [`FLAT_ROWS_PER_BLOCK`], each group
/// padded to the block alignment, and the block offsets. Rows stream without the final count.
pub(crate) struct FlatEntryWriter {
    stride: usize,
    align: usize,
    block_offsets: BlockOffsetsWriter,
    /// Bytes of the entry written so far.
    written: u64,
    num_rows: usize,
}

impl FlatEntryWriter {
    /// Starts an entry of `opts` rows on `out` with its metadata.
    pub(crate) fn start(out: &mut impl Write, opts: &VectorOptions) -> io::Result<Self> {
        let meta = VectorColMetadata::build_flat(opts);
        let align = block_align(&meta.slots());
        let written = write_metadata(out, &meta)? as u64;
        Ok(Self {
            stride: opts.bytes_per_vector(),
            align,
            block_offsets: BlockOffsetsWriter::new(written),
            written,
            num_rows: 0,
        })
    }

    /// Rows appended so far.
    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Appends `rows`, a whole number of rows, closing each group it fills.
    pub(crate) fn push_rows(&mut self, out: &mut impl Write, mut rows: &[u8]) -> io::Result<()> {
        debug_assert_eq!(rows.len() % self.stride, 0);
        let group = FLAT_ROWS_PER_BLOCK as usize;
        while !rows.is_empty() {
            let room = group - self.num_rows % group;
            let (head, tail) = rows.split_at(room.min(rows.len() / self.stride) * self.stride);
            out.write_all(head)?;
            self.written += head.len() as u64;
            self.num_rows += head.len() / self.stride;
            if self.num_rows % group == 0 {
                self.end_group(out, group)?;
            }
            rows = tail;
        }
        Ok(())
    }

    /// Pads a group of `rows` rows to the block alignment and records its end.
    fn end_group(&mut self, out: &mut impl Write, rows: usize) -> io::Result<()> {
        let bytes = rows * self.stride;
        let padding = align_up(bytes, self.align) - bytes;
        pad(out, padding)?;
        self.written += padding as u64;
        self.block_offsets.push(self.written, rows);
        Ok(())
    }

    /// Closes the last, partial group and writes the block offsets. Returns the entry's length, a
    /// multiple of [`ENTRY_ALIGN`].
    pub(crate) fn finish(mut self, out: &mut impl Write) -> io::Result<u64> {
        let partial = self.num_rows % FLAT_ROWS_PER_BLOCK as usize;
        if partial != 0 {
            self.end_group(out, partial)?;
        }
        let len = self.block_offsets.finish(out)?;
        assert_eq!(len % ENTRY_ALIGN, 0);
        Ok(len as u64)
    }
}

// --- Writer ---

/// Buffers one vector field before serialization.
struct FieldBuffer {
    present_doc_ids: Vec<DocId>,
    row_bytes: Vec<u8>,
    admission: RowAdmission,
}

impl FieldBuffer {
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
                        admission: RowAdmission::new(opts),
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
            let name = schema.get_field_entry(field).name();
            buf.admission
                .admit(name, doc_id, bytes, &mut buf.row_bytes)?;
            buf.present_doc_ids.push(doc_id);
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
            let opts = buf.admission.options();
            let stride = opts.bytes_per_vector();
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
            composite.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
            let data = composite.for_field_with_idx(field, VectorEntry::Data.index());
            let mut entry = FlatEntryWriter::start(data, opts)?;
            entry.push_rows(data, &row_bytes)?;
            entry.finish(data)?;
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

/// Merges source vectors into a flat target segment. With `streamed` rows, the target's `.vec`
/// already holds them as its Data entry and only the IdMap is written.
pub(crate) fn merge_flat(
    ctx: &PluginMergeContext,
    streamed: Option<(CompositeWrite, StreamedRows)>,
) -> crate::Result<()> {
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
    if let Some((mut composite, rows)) = streamed {
        let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum::<u32>();
        IdMap::serialize(
            &rows.present_docs(),
            num_target_docs,
            composite.for_field_with_idx(rows.field, VectorEntry::IdMap.index()),
        )?;
        composite.close()?;
        return Ok(());
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
    let mut entry = FlatEntryWriter::start(rows_w, opts)?;
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
        entry.push_rows(rows_w, row)?;
    }
    entry.finish(rows_w)?;
    rows_w.flush()?;
    Ok(present)
}

#[cfg(test)]
mod tests {
    use common::CountingWriter;

    use super::*;
    use crate::directory::FileSlice;
    use crate::schema::Metric;
    use crate::vector::storage::blocks::Blocks;
    use crate::vector::test_support::FixtureSpec;
    use crate::vector::BlockAddress;

    /// Rows pushed one at a time form the entry [`write_flat_rows`] writes from a source segment
    /// holding the same rows, and the entry reads back as those rows.
    #[test]
    fn flat_entry_writer_matches_write_flat_rows() -> crate::Result<()> {
        let opts = VectorOptions::new(2, Metric::L2);
        let group = FLAT_ROWS_PER_BLOCK as usize;
        for num_rows in [0, 1, group, group + 1] {
            let vectors: Vec<Vec<f32>> = (0..num_rows)
                .map(|row| vec![row as f32, -(row as f32)])
                .collect();
            let rows: Vec<u8> = vectors
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect();

            let mut pushed = Vec::new();
            let mut entry = FlatEntryWriter::start(&mut pushed, &opts)?;
            for row in rows.chunks_exact(opts.bytes_per_vector()) {
                entry.push_rows(&mut pushed, row)?;
            }
            assert_eq!(entry.finish(&mut pushed)? as usize, pushed.len());

            let fixture = FixtureSpec::flat(Metric::L2, 2, vectors.into_iter().map(Some).collect())
                .build()?;
            let searcher = fixture.index.reader()?.searcher();
            let readers = searcher
                .segment_readers()
                .iter()
                .map(|reader| reader.vector_index(fixture.field))
                .collect::<crate::Result<Vec<_>>>()?;
            let source_rows: Vec<Option<RowAddress>> = (0..num_rows as u32)
                .map(|row| {
                    Some(RowAddress {
                        segment_ord: 0,
                        block_address: BlockAddress::new(
                            row / FLAT_ROWS_PER_BLOCK,
                            row % FLAT_ROWS_PER_BLOCK,
                        ),
                    })
                })
                .collect();
            let mut merged = CountingWriter::wrap(Vec::new());
            let present = write_flat_rows(
                &mut merged,
                &opts,
                &source_rows,
                &readers,
                RowBytes::Stored,
                &(|| false),
            )?;
            assert_eq!(present.len(), num_rows);
            assert_eq!(pushed, merged.finish(), "{num_rows} rows");

            let blocks = Blocks::open(FileSlice::from(pushed), &opts, false)?;
            assert_eq!(blocks.num_rows(), num_rows);
            assert_eq!(blocks.num_blocks(), num_rows.div_ceil(group));
            assert_eq!(
                blocks.flat_rows().unwrap().read_bytes()?.as_slice(),
                rows.as_slice()
            );
        }
        Ok(())
    }
}

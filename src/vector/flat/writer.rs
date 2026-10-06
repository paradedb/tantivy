use std::any::Any;
use std::io::Write;

use super::id_map::IdMap;
use crate::directory::CompositeWrite;
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::PluginWriter;
use crate::schema::document::ErasedDocument;
use crate::schema::Schema;
use crate::vector::blocks::{align_up, block_align, pad, write_metadata, BlockDirectory};
use crate::vector::buffer::VectorBuffer;
use crate::vector::header::{write_vector_header, VectorEntry, HEADER_LEN};
use crate::vector::metadata::{VectorColMetadata, FLAT_ROWS_PER_BLOCK};
use crate::vector::{ENTRY_ALIGN, VEC_EXT};
use crate::DocId;

/// Writes full-precision vector rows.
pub struct FlatVecWriter {
    buffer: VectorBuffer,
}

impl FlatVecWriter {
    /// Creates a writer for all vector fields in a schema.
    pub fn for_schema(schema: &Schema) -> Self {
        Self {
            buffer: VectorBuffer::for_schema(schema),
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
        self.buffer.add_document(doc_id, doc, schema)
    }

    fn serialize(
        self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let VectorBuffer { fields, num_docs } = self.buffer;
        if fields.is_empty() {
            return Ok(());
        }
        let mut write = segment.open_write(SegmentComponent::Custom(VEC_EXT.to_string()))?;
        write_vector_header(&mut write)?;
        let mut composite = CompositeWrite::wrap(write);

        let mut id_maps = Vec::new();
        for (field, buf) in fields {
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
            let mut directory = BlockDirectory::new(data.written_bytes() - start);
            let mut rows = 0u32;
            // A full row group or the final partial group forms one aligned Rows column.
            for block in row_bytes.chunks(FLAT_ROWS_PER_BLOCK as usize * stride) {
                data.write_all(block)?;
                pad(data, align_up(block.len(), align) - block.len())?;
                rows += (block.len() / stride) as u32;
                directory.push(data.written_bytes() - start, rows);
            }
            directory.finish(data)?;
            assert_eq!((data.written_bytes() - start) as usize % ENTRY_ALIGN, 0);
            data.flush()?;
        }
        for (field, present) in id_maps {
            IdMap::serialize(
                &present,
                num_docs,
                composite.for_field_with_idx(field, VectorEntry::IdMap.index()),
            )?;
        }
        composite.close()?;
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.buffer.mem_usage()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

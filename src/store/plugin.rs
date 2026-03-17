//! Store as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `StoreWriter` and `StoreReader` types behind the plugin
//! interface so that the document store participates in the unified plugin lifecycle.

use std::any::Any;
use std::sync::Arc;

use measure_time::debug_time;

use crate::directory::Directory;
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::schema::document::Document;
use crate::schema::Schema;
use crate::store::{StoreReader, StoreWriter};
use crate::Segment;

/// Built-in plugin for the document store (row-oriented compressed storage).
///
/// The document store holds the stored fields for each document. It is accessed
/// when retrieving full documents after search. During indexing, documents are
/// written incrementally as they arrive.
///
/// For sorted indexes, documents are first written to a temporary store, then
/// reordered according to the sort field during serialization.
pub struct StorePlugin;

impl SegmentPlugin for StorePlugin {
    fn name(&self) -> &str {
        "store"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["store", "temp"]
    }

    fn write_phase(&self) -> u32 {
        2
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let settings = ctx.settings;
        let remapping_required = settings.sort_by_field.is_some() && !ctx.is_in_merge;

        // During merge, the merge() method handles file creation directly.
        // Only open the file during normal indexing.
        let store_writer = if !ctx.is_in_merge {
            let store_writer = if remapping_required {
                let path = ctx.segment.relative_path(SegmentComponent::TempStore);
                let store_write = ctx.directory.open_write(&path)?;
                StoreWriter::new(
                    store_write,
                    crate::store::Compressor::None,
                    // We want fast random access on the docs, so we choose a small block size.
                    // If this is zero, the skip index will contain too many checkpoints and
                    // therefore will be relatively slow.
                    16000,
                    settings.docstore_compress_dedicated_thread,
                )?
            } else {
                let path = ctx.segment.relative_path(SegmentComponent::Store);
                let store_write = ctx.directory.open_write(&path)?;
                StoreWriter::new(
                    store_write,
                    settings.docstore_compression,
                    settings.docstore_blocksize,
                    settings.docstore_compress_dedicated_thread,
                )?
            };
            Some(store_writer)
        } else {
            None
        };

        Ok(Box::new(StorePluginWriter {
            store_writer,
            remapping_required,
        }))
    }

    fn open_reader(&self, _ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        // The StoreReader is opened on-demand via SegmentReader::get_store_reader()
        // because it requires a cache_num_blocks parameter. We provide a no-op reader here.
        Ok(Arc::new(StorePluginReader))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        debug_time!("write-storable-fields");
        debug!("write-storable-fields");

        let path = ctx.target_segment.relative_path(SegmentComponent::Store);
        let store_write = ctx.target_segment.index().directory().open_write(&path)?;
        let settings = ctx.settings;
        let mut store_writer = StoreWriter::new(
            store_write,
            settings.docstore_compression,
            settings.docstore_blocksize,
            settings.docstore_compress_dedicated_thread,
        )?;

        if !ctx.doc_id_mapping.is_trivial() {
            debug!("non-trivial-doc-id-mapping");

            let store_readers: Vec<_> = ctx
                .readers
                .iter()
                .map(|reader| reader.get_store_reader(50))
                .collect::<Result<_, _>>()?;

            let mut document_iterators: Vec<_> = store_readers
                .iter()
                .enumerate()
                .map(|(i, store)| store.iter_raw(ctx.readers[i].alive_bitset()))
                .collect();

            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let doc_bytes_it = &mut document_iterators[old_doc_addr.segment_ord as usize];
                if let Some(doc_bytes_res) = doc_bytes_it.next() {
                    let doc_bytes = doc_bytes_res?;
                    store_writer.store_bytes(&doc_bytes)?;
                } else {
                    return Err(crate::error::DataCorruption::comment_only(format!(
                        "unexpected missing document in docstore on merge, doc address \
                         {old_doc_addr:?}",
                    ))
                    .into());
                }
            }
        } else {
            debug!("trivial-doc-id-mapping");
            for reader in ctx.readers {
                if ctx.cancel.wants_cancel() {
                    return Err(crate::TantivyError::Cancelled);
                }
                let store_reader = reader.get_store_reader(1)?;
                if reader.has_deletes()
                    // If there is not enough data in the store, we avoid stacking in order to
                    // avoid creating many small blocks in the doc store. Once we have 5 full blocks,
                    // we start stacking. In the worst case 2/7 of the blocks would be very small.
                    || store_reader.block_checkpoints().take(7).count() < 6
                    || store_reader.decompressor() != store_writer.compressor().into()
                {
                    for doc_bytes_res in store_reader.iter_raw(reader.alive_bitset()) {
                        if ctx.cancel.wants_cancel() {
                            return Err(crate::TantivyError::Cancelled);
                        }
                        let doc_bytes = doc_bytes_res?;
                        store_writer.store_bytes(&doc_bytes)?;
                    }
                } else {
                    store_writer.stack(store_reader)?;
                }
            }
        }
        store_writer.close()?;
        Ok(())
    }
}

/// Plugin writer wrapping [`StoreWriter`].
///
/// Exposes `store()` and `store_bytes()` for the `SegmentWriter` to call via downcast.
pub struct StorePluginWriter {
    store_writer: Option<StoreWriter>,
    remapping_required: bool,
}

impl StorePluginWriter {
    /// Store a document. Called by `SegmentWriter` via downcast.
    pub fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> crate::Result<()> {
        if let Some(ref mut writer) = self.store_writer {
            writer
                .store(document, schema)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    /// Store raw document bytes. Called during TempStore-to-Store reordering.
    pub fn store_bytes(&mut self, serialized_document: &[u8]) -> crate::Result<()> {
        if let Some(ref mut writer) = self.store_writer {
            writer
                .store_bytes(serialized_document)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }
}

impl PluginWriter for StorePluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        // For sorted indexes (remapping_required), we need to close the temp store
        // and reorder documents into the final store.
        if self.remapping_required {
            if let Some(doc_id_map) = doc_id_map {
                debug!("resort-docstore");
                // Close the temp store writer first
                if let Some(old_writer) = self.store_writer.take() {
                    old_writer
                        .close()
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                }

                // Open the final store writer
                let settings = segment.index().settings().clone();
                let store_write = segment.open_write(SegmentComponent::Store)?;
                let mut store_writer = StoreWriter::new(
                    store_write,
                    settings.docstore_compression,
                    settings.docstore_blocksize,
                    settings.docstore_compress_dedicated_thread,
                )
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;

                // Read from temp store and write in new order
                let store_read = StoreReader::open(
                    segment.open_read(SegmentComponent::TempStore)?,
                    1, /* The docstore is configured to have one doc per block, and each doc is
                        * accessed only once: we don't need caching. */
                )?;
                for old_doc_id in doc_id_map.iter_old_doc_ids() {
                    let doc_bytes = store_read.get_document_bytes(old_doc_id)?;
                    store_writer
                        .store_bytes(&doc_bytes)
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                }

                // Replace the writer with the final store writer
                self.store_writer = Some(store_writer);
            }
        }
        // For non-sorted indexes, data was already written incrementally. Nothing to do.
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        if let Some(writer) = self.store_writer {
            writer
                .close()
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.store_writer.as_ref().map_or(0, |w| w.mem_usage())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Plugin reader for the document store.
///
/// The actual store reading is done via `SegmentReader::get_store_reader()` which
/// requires a `cache_num_blocks` parameter, so this reader is a placeholder.
pub struct StorePluginReader;

impl PluginReader for StorePluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

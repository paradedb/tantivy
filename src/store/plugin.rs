//! Store as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `StoreWriter` and `StoreReader` types behind the plugin
//! interface so that the document store participates in the unified plugin lifecycle.

use std::any::Any;
use std::collections::BTreeMap;

use measure_time::debug_time;

use crate::directory::Directory;
use crate::index::{SegmentComponent, SegmentReader};
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::schema::document::{Document, TantivyDocument};
use crate::schema::Schema;
use crate::space_usage::{ComponentSpaceUsage, STORE};
use crate::store::StoreWriter;
use crate::{DocId, Segment};

/// The built-in [`SegmentPlugin`] that manages the document store.
pub struct StorePlugin;

impl SegmentPlugin for StorePlugin {
    fn extensions(&self) -> &[&str] {
        &["store"]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let settings = ctx.segment.index().settings();
        let directory = ctx.segment.index().directory();

        let path = ctx.segment.relative_path(SegmentComponent::Store);
        let store_write = directory.open_write(&path)?;
        let store_writer = StoreWriter::new(
            store_write,
            settings.docstore_compression,
            settings.docstore_blocksize,
            settings.docstore_compress_dedicated_thread,
        )?;

        Ok(Box::new(StorePluginWriter {
            store_writer: Some(store_writer),
            ignore_store: ctx.ignore_store,
        }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        if ctx.ignore_store {
            return Ok(());
        }
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

        for reader in ctx.readers {
            let store_reader = reader.get_store_reader(1)?;
            if reader.has_deletes()
                // If there is not enough data in the store, we avoid stacking in order to
                // avoid creating many small blocks in the doc store. Once we have 5 full blocks,
                // we start stacking. In the worst case 2/7 of the blocks would be very small.
                || store_reader.block_checkpoints().take(7).count() < 6
                || store_reader.decompressor() != store_writer.compressor().into()
            {
                for doc_bytes_res in store_reader.iter_raw(reader.alive_bitset()) {
                    let doc_bytes = doc_bytes_res?;
                    store_writer.store_bytes(&doc_bytes)?;
                }
            } else {
                store_writer.stack(store_reader)?;
            }
        }
        store_writer.close()?;
        Ok(())
    }

    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let store = segment_reader.get_store_reader(0)?;
        Ok(BTreeMap::from([(
            STORE.to_string(),
            ComponentSpaceUsage::Store(store.space_usage()),
        )]))
    }
}

/// The [`PluginWriter`] for the document store plugin.
pub struct StorePluginWriter {
    store_writer: Option<StoreWriter>,
    ignore_store: bool,
}

impl StorePluginWriter {
    /// Stores a single document in the document store.
    pub fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> crate::Result<()> {
        if let Some(ref mut writer) = self.store_writer {
            writer
                .store(document, schema)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    /// Stores an already serialized document in the document store.
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
    fn add_document(
        &mut self,
        _doc_id: DocId,
        doc: &TantivyDocument,
        schema: &Schema,
    ) -> crate::Result<()> {
        if self.ignore_store {
            return Ok(());
        }
        self.store(doc, schema)
    }

    fn serialize(&mut self, _segment: &Segment) -> crate::Result<()> {
        // Documents were already written incrementally via `add_document`. Nothing to do.
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

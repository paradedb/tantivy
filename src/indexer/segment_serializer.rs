use std::sync::Arc;

use common::TerminatingWrite;

use crate::directory::WritePtr;
use crate::fieldnorm::FieldNormsSerializer;
use crate::index::{Segment, SegmentComponent};
use crate::plugin::{PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::postings::InvertedIndexSerializer;
use crate::store::StoreWriter;

/// Segment serializer is in charge of laying out on disk
/// the data accumulated and sorted by the `SegmentWriter`.
pub struct SegmentSerializer {
    segment: Segment,
    pub(crate) store_writer: StoreWriter,
    fast_field_write: WritePtr,
    fieldnorms_serializer: Option<FieldNormsSerializer>,
    postings_serializer: InvertedIndexSerializer,
    /// Plugin writers, stored as (name, writer) pairs.
    plugin_writers: Vec<(String, Box<dyn PluginWriter>)>,
}

impl SegmentSerializer {
    /// Creates a new `SegmentSerializer`.
    pub fn for_segment(
        segment: Segment,
        is_in_merge: bool,
    ) -> crate::Result<SegmentSerializer> {
        let plugins: Vec<Arc<dyn SegmentPlugin>> = segment.index().plugins().to_vec();
        Self::for_segment_with_plugins(segment, is_in_merge, &plugins)
    }

    /// Creates a new `SegmentSerializer` with explicit plugins.
    pub fn for_segment_with_plugins(
        mut segment: Segment,
        is_in_merge: bool,
        plugins: &[Arc<dyn SegmentPlugin>],
    ) -> crate::Result<SegmentSerializer> {
        // If the segment is going to be sorted, we stream the docs first to a temporary file.
        // In the merge case this is not necessary because we can kmerge the already sorted
        // segments
        let remapping_required = segment.index().settings().sort_by_field.is_some() && !is_in_merge;
        let settings = segment.index().settings().clone();
        let store_writer = if remapping_required {
            let store_write = segment.open_write(SegmentComponent::TempStore)?;
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
            let store_write = segment.open_write(SegmentComponent::Store)?;
            StoreWriter::new(
                store_write,
                settings.docstore_compression,
                settings.docstore_blocksize,
                settings.docstore_compress_dedicated_thread,
            )?
        };

        let fast_field_write = segment.open_write(SegmentComponent::FastFields)?;

        let fieldnorms_write = segment.open_write(SegmentComponent::FieldNorms)?;
        let fieldnorms_serializer = FieldNormsSerializer::from_write(fieldnorms_write)?;

        let postings_serializer = InvertedIndexSerializer::open(&mut segment)?;

        // Create plugin writers
        let schema = segment.schema();
        let plugin_writers = plugins
            .iter()
            .map(|p| {
                let ctx = PluginWriterContext {
                    segment: &segment,
                    schema: &schema,
                    settings: &settings,
                    is_in_merge,
                };
                Ok((p.name().to_string(), p.create_writer(&ctx)?))
            })
            .collect::<crate::Result<Vec<_>>>()?;

        Ok(SegmentSerializer {
            segment,
            store_writer,
            fast_field_write,
            fieldnorms_serializer: Some(fieldnorms_serializer),
            postings_serializer,
            plugin_writers,
        })
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.store_writer.mem_usage()
            + self
                .plugin_writers
                .iter()
                .map(|(_, w)| w.mem_usage())
                .sum::<usize>()
    }

    pub fn segment(&self) -> &Segment {
        &self.segment
    }

    pub fn segment_mut(&mut self) -> &mut Segment {
        &mut self.segment
    }

    /// Accessor to the `PostingsSerializer`.
    pub fn get_postings_serializer(&mut self) -> &mut InvertedIndexSerializer {
        &mut self.postings_serializer
    }

    /// Accessor to the `FastFieldSerializer`.
    pub fn get_fast_field_write(&mut self) -> &mut WritePtr {
        &mut self.fast_field_write
    }

    /// Extract the field norm serializer.
    ///
    /// Note the fieldnorms serializer can only be extracted once.
    pub fn extract_fieldnorms_serializer(&mut self) -> Option<FieldNormsSerializer> {
        self.fieldnorms_serializer.take()
    }

    /// Accessor to the `StoreWriter`.
    pub fn get_store_writer(&mut self) -> &mut StoreWriter {
        &mut self.store_writer
    }

    /// Get a plugin writer by name and downcast to the expected type.
    pub fn get_plugin_writer<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.plugin_writers
            .iter_mut()
            .find(|(n, _)| n == name)
            .and_then(|(_, w)| w.as_any_mut().downcast_mut::<T>())
    }

    /// Access the plugin writers for iteration.
    pub fn plugin_writers_mut(&mut self) -> &mut Vec<(String, Box<dyn PluginWriter>)> {
        &mut self.plugin_writers
    }

    /// Finalize the segment serialization.
    pub fn close(mut self) -> crate::Result<()> {
        if let Some(fieldnorms_serializer) = self.extract_fieldnorms_serializer() {
            fieldnorms_serializer.close()?;
        }
        self.fast_field_write.terminate()?;
        self.postings_serializer.close()?;
        self.store_writer.close()?;
        for (_, writer) in self.plugin_writers {
            writer.close()?;
        }
        Ok(())
    }
}

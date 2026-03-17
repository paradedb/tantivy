use std::sync::Arc;

use crate::index::Segment;
use crate::plugin::{PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::postings::InvertedIndexSerializer;

/// Segment serializer is in charge of laying out on disk
/// the data accumulated and sorted by the `SegmentWriter`.
pub struct SegmentSerializer {
    segment: Segment,
    postings_serializer: InvertedIndexSerializer,
    /// Plugin writers, stored as (name, writer) pairs.
    /// Includes built-in plugins (fieldnorms, fast_fields, store, etc.) and custom plugins.
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
        let settings = segment.index().settings().clone();

        let postings_serializer = InvertedIndexSerializer::open(&mut segment)?;

        // Create plugin writers (includes built-in plugins like FieldNormsPlugin, StorePlugin)
        let schema = segment.schema();
        let directory: &dyn crate::Directory = segment.index().directory();
        let plugin_writers = plugins
            .iter()
            .map(|p| {
                let ctx = PluginWriterContext {
                    segment: &segment,
                    schema: &schema,
                    settings: &settings,
                    is_in_merge,
                    directory,
                };
                Ok((p.name().to_string(), p.create_writer(&ctx)?))
            })
            .collect::<crate::Result<Vec<_>>>()?;

        Ok(SegmentSerializer {
            segment,
            postings_serializer,
            plugin_writers,
        })
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.plugin_writers
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

    /// Get a plugin writer by name and downcast to the expected type (mutable).
    pub fn get_plugin_writer<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.plugin_writers
            .iter_mut()
            .find(|(n, _)| n == name)
            .and_then(|(_, w)| w.as_any_mut().downcast_mut::<T>())
    }

    /// Get a plugin writer by name and downcast to the expected type (immutable).
    pub fn get_plugin_writer_ref<T: 'static>(&self, name: &str) -> Option<&T> {
        self.plugin_writers
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, w)| w.as_any().downcast_ref::<T>())
    }

    /// Access the plugin writers for iteration.
    pub fn plugin_writers_mut(&mut self) -> &mut Vec<(String, Box<dyn PluginWriter>)> {
        &mut self.plugin_writers
    }

    /// Finalize the segment serialization.
    pub fn close(self) -> crate::Result<()> {
        self.postings_serializer.close()?;
        for (_, writer) in self.plugin_writers {
            writer.close()?;
        }
        Ok(())
    }
}

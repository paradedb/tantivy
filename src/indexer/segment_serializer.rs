use std::sync::Arc;

use crate::index::Segment;
use crate::plugin::{PluginWriter, PluginWriterContext, SegmentPlugin};

/// Segment serializer is in charge of laying out on disk
/// the data accumulated and sorted by the `SegmentWriter`.
pub struct SegmentSerializer {
    segment: Segment,
    /// Plugin writers for the built-in plugins (fieldnorms, postings, fast_fields,
    /// store, etc.) and custom plugins, each paired with its plugin's write phase
    /// (captured at creation, used to order serialization). Looked up by concrete
    /// writer type.
    plugin_writers: Vec<(u32, Box<dyn PluginWriter>)>,
}

impl SegmentSerializer {
    /// Creates a new `SegmentSerializer`.
    pub fn for_segment(
        segment: Segment,
        is_in_merge: bool,
        ignore_store: bool,
    ) -> crate::Result<SegmentSerializer> {
        let plugins: Vec<Arc<dyn SegmentPlugin>> = segment.index().plugins().to_vec();
        Self::for_segment_with_plugins(segment, is_in_merge, ignore_store, &plugins)
    }

    /// Creates a new `SegmentSerializer` with explicit plugins.
    pub fn for_segment_with_plugins(
        segment: Segment,
        is_in_merge: bool,
        ignore_store: bool,
        plugins: &[Arc<dyn SegmentPlugin>],
    ) -> crate::Result<SegmentSerializer> {
        let settings = segment.index().settings().clone();

        // Create plugin writers (includes built-in plugins like FieldNormsPlugin,
        // PostingsPlugin, FastFieldsPlugin, StorePlugin)
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
                    ignore_store,
                    directory,
                };
                Ok((p.write_phase(), p.create_writer(&ctx)?))
            })
            .collect::<crate::Result<Vec<_>>>()?;

        Ok(SegmentSerializer {
            segment,
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

    /// Get the plugin writer of the given concrete type (mutable).
    pub fn get_plugin_writer<T: 'static>(&mut self) -> Option<&mut T> {
        self.plugin_writers
            .iter_mut()
            .find_map(|(_, w)| w.as_any_mut().downcast_mut::<T>())
    }

    /// Get the plugin writer of the given concrete type (immutable).
    pub fn get_plugin_writer_ref<T: 'static>(&self) -> Option<&T> {
        self.plugin_writers
            .iter()
            .find_map(|(_, w)| w.as_any().downcast_ref::<T>())
    }

    /// Access the plugin writers (paired with their write phase) for iteration.
    pub fn plugin_writers_mut(&mut self) -> &mut Vec<(u32, Box<dyn PluginWriter>)> {
        &mut self.plugin_writers
    }

    /// Finalize the segment serialization.
    pub fn close(self) -> crate::Result<()> {
        for (_, writer) in self.plugin_writers {
            writer.close()?;
        }
        Ok(())
    }
}

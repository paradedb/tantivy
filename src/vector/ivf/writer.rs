//! Stub writer for the IVF vector plugin.
//!
//! IVF is built at merge time, never per-doc during indexing. So this
//! writer carries no state and every [`PluginWriter`] method is a
//! trivial no-op. The interesting work lives in
//! [`IvfVecPlugin::merge`](super::plugin::IvfVecPlugin::merge).

use std::any::Any;

use crate::index::Segment;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::PluginWriter;

pub struct IvfVecPluginWriter;

impl IvfVecPluginWriter {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl PluginWriter for IvfVecPluginWriter {
    fn serialize(
        &mut self,
        _segment: &mut Segment,
        _doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

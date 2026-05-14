//! Stub plugin for the flat vector format.
//!
//! Until the flat writer/reader land, the plugin isn't registered in
//! [`Index::default_plugins`](crate::Index::default_plugins). The type
//! exists so that [`VectorBackend`](super::super::backend::VectorBackend)
//! has something to reference and so the public surface
//! ([`FlatVecPlugin`], [`FlatVecPluginWriter`], [`FlatVecReader`],
//! [`VectorColumn`]) is stable.

use std::sync::Arc;

use super::reader::FlatVecReader;
use super::writer::FlatVecPluginWriter;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};

pub struct FlatVecPlugin;

impl SegmentPlugin for FlatVecPlugin {
    fn name(&self) -> &str {
        "flat_vec"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["flatvec"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(FlatVecPluginWriter::stub()))
    }

    fn open_reader(&self, _ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        Ok(Arc::new(FlatVecReader::stub()))
    }

    fn merge(&self, _ctx: PluginMergeContext) -> crate::Result<()> {
        todo!("flat vector merge")
    }
}

//! Stub reader for the IVF plugin.
//!
//! Until the IVF writer lands, the plugin isn't registered, so
//! `segment_reader.plugin_reader::<IvfVecReader>("ivf_vec")` always
//! returns `None`. The types here exist so that
//! [`VectorBackend`](super::super::backend::VectorBackend) can dispatch
//! over `Flat | Ivf` at compile time.

use std::any::Any;

use crate::plugin::PluginReader;
use crate::schema::Field;
use crate::vector::options::{Metric, VectorElement};
use crate::DocId;

pub struct IvfVecReader {
    // TODO: handles to the centroid index, assignment file, vector file,
    // and the per-cluster offsets metadata.
}

impl IvfVecReader {
    /// Open a per-field IVF column. Stub returns `None` so the backend
    /// always falls through to flat.
    pub fn open_column(&self, _field: Field) -> Option<IvfVectorColumn> {
        None
    }
}

impl PluginReader for IvfVecReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Per-segment, per-field IVF column view.
///
/// Methods are `todo!()` placeholders — they document the surface that
/// [`IvfBackend`](super::super::backend::IvfBackend) needs from the
/// reader once the writer/plugin land.
pub struct IvfVectorColumn {
    // TODO: centroid table, per-cluster doc-id postings, per-cluster
    // vector blob, cluster offset table.
}

impl IvfVectorColumn {
    /// Rank centroids by distance to the query (closest first). Drives
    /// the probe order in adaptive cluster scanning.
    pub fn rank_centroids<T: VectorElement>(&self, _query: &[T], _metric: Metric) -> Vec<u32> {
        todo!("IVF centroid ranking")
    }

    /// Sorted doc ids belonging to a cluster (within-cluster order).
    pub fn cluster_doc_ids(&self, _cluster: u32) -> &[DocId] {
        todo!("IVF cluster doc-id postings")
    }

    /// Raw little-endian vector bytes for a doc in a given cluster.
    pub fn vector_bytes_in_cluster(&self, _cluster: u32, _doc: DocId) -> &[u8] {
        todo!("IVF intra-cluster vector lookup")
    }
}

//! The write step: a field's `.centroids` slots, an empty field's entries, and the IdMap entries.

use std::io::Write;

use super::{no_centroids, train, FieldMerge};
use crate::directory::CompositeWrite;
use crate::schema::Field;
use crate::vector::router::{BuiltRouter, RouterKind};
use crate::vector::storage::blocks::{write_metadata, BlockOffsetsWriter};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::header::{CentroidSlot, VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::metadata::VectorColMetadata;
use crate::vector::{BlockAddress, ENTRY_ALIGN};
use crate::{DocId, TantivyError};

/// One field's addressing entries, written after every field's Data and ExactRows entries.
pub(super) struct FieldIdMaps {
    pub(super) field: Field,
    /// Each target document's primary cluster row.
    pub(super) locations: Vec<BlockAddress>,
    /// The documents with a row in `ExactRows`, ascending; `None` for a plain field, which has no
    /// such entry.
    pub(super) exact_present: Option<Vec<DocId>>,
    pub(super) num_target_docs: u32,
}

impl FieldIdMaps {
    /// The entries of a field without vectors: every location absent, no exact rows.
    pub(super) fn empty(merge: &FieldMerge) -> Self {
        Self {
            field: merge.field,
            locations: IdMap::empty_locations(merge.num_target_docs),
            exact_present: merge.quantization.map(|_| Vec::new()),
            num_target_docs: merge.num_target_docs,
        }
    }
}

/// Writes a field's `.centroids` slots: centroids and the router.
pub(super) fn centroid_slots(
    centroids_write: &mut CompositeWrite,
    merge: &FieldMerge,
    num_present_docs: usize,
    centroid_bytes: &[u8],
    num_centroids: usize,
    router: &BuiltRouter,
) -> crate::Result<()> {
    let field = merge.field;
    {
        let centroids_w =
            centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
        ClusterIndex::serialize_centroids(
            num_centroids,
            num_present_docs,
            centroid_bytes,
            merge.opts,
            centroids_w,
        )?;
        centroids_w.flush()?;
    }

    if merge.ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }
    let router_w = centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
    router.serialize(router_w)?;
    router_w.flush()?;
    Ok(())
}

/// Writes an empty IVF field to both vector composites, with a `router` built over no centroids:
/// a Data entry of metadata and no blocks and, for a quantized field whose rows were not streamed
/// into an ExactRows entry, an ExactRows entry alike.
pub(super) fn empty_field(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    merge: &FieldMerge,
    router: RouterKind,
) -> crate::Result<()> {
    let (field, opts) = (merge.field, merge.opts);
    let router = train::build_router(router, opts, &mut no_centroids(opts))?;
    let meta = VectorColMetadata::build_ivf(opts, merge.quantization)?;
    let mut entries = vec![(VectorEntry::Data, meta)];
    if merge.quantization.is_some() && !merge.streamed {
        entries.push((VectorEntry::ExactRows, VectorColMetadata::build_flat(opts)));
    }
    for (entry, meta) in entries {
        vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
        let data = vec_write.for_field_with_idx(field, entry.index());
        let start = data.written_bytes();
        write_metadata(data, &meta)?;
        BlockOffsetsWriter::new(data.written_bytes() - start).finish(data)?;
        assert_eq!((data.written_bytes() - start) as usize % ENTRY_ALIGN, 0);
    }
    {
        let centroids_w =
            centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
        ClusterIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let router_w = centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
        router.serialize(router_w)?;
        router_w.flush()?;
    }
    Ok(())
}

/// Writes each field's IdMap entries after every Data and ExactRows entry.
pub(super) fn id_maps(
    vec_write: &mut CompositeWrite,
    id_maps: Vec<FieldIdMaps>,
) -> crate::Result<()> {
    for FieldIdMaps {
        field,
        locations,
        exact_present,
        num_target_docs,
    } in id_maps
    {
        let id_map = vec_write.for_field_with_idx(field, VectorEntry::IdMap.index());
        IdMap::serialize_locations(&locations, id_map)?;
        id_map.flush()?;
        if let Some(present) = exact_present {
            let exact_id_map = vec_write.for_field_with_idx(field, VectorEntry::ExactIdMap.index());
            IdMap::serialize(&present, num_target_docs, exact_id_map)?;
            exact_id_map.flush()?;
        }
    }
    Ok(())
}

//! The write step: a field's `.centroids` slots, an empty field's entries, and the IdMap entries
//! with the build report.

use std::io::Write;
use std::time::{Duration, Instant};

use super::{no_centroids, train, FieldMerge, IvfBuildTimings};
use crate::directory::CompositeWrite;
use crate::schema::Field;
use crate::vector::router::{BuiltRouter, RouterKind};
use crate::vector::storage::blocks::{write_metadata, BlockDirectory};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::header::{CentroidSlot, VectorEntry, HEADER_LEN};
use crate::vector::storage::id_map::{DocLocation, IdMap};
use crate::vector::storage::metadata::VectorColMetadata;
use crate::vector::{BoundKind, ENTRY_ALIGN};
use crate::TantivyError;

/// Writes a field's `.centroids` slots: centroids, offsets, bounds, then the router.
pub(super) fn centroid_slots(
    centroids_write: &mut CompositeWrite,
    merge: &FieldMerge,
    num_present_docs: usize,
    centroid_bytes: &[u8],
    cluster_offsets: &[u64],
    bounds: &[f32],
    router: &BuiltRouter,
) -> crate::Result<()> {
    let field = merge.field;
    let num_centroids = cluster_offsets.len() - 1;
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
    {
        let offsets_w = centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
        ClusterIndex::serialize_offsets(cluster_offsets, offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
        ClusterIndex::serialize_bounds(BoundKind::Ball, bounds, bounds_w)?;
        bounds_w.flush()?;
    }

    if merge.ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }
    let router_w = centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
    router.serialize(router_w)?;
    router_w.flush()?;
    Ok(())
}

/// Writes an empty IVF field to both vector composites, with a `router` built over no centroids.
pub(super) fn empty_field(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    merge: &FieldMerge,
    router: RouterKind,
) -> crate::Result<()> {
    let (field, opts) = (merge.field, merge.opts);
    let router = train::build_router(router, opts, &mut no_centroids(opts))?;
    let meta = VectorColMetadata::build_ivf(opts, merge.quantization)?;
    vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
    let data = vec_write.for_field_with_idx(field, VectorEntry::Data.index());
    let start = data.written_bytes();
    write_metadata(data, &meta)?;
    BlockDirectory::new(data.written_bytes() - start).finish(data)?;
    assert_eq!((data.written_bytes() - start) as usize % ENTRY_ALIGN, 0);
    {
        let centroids_w =
            centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
        ClusterIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let offsets_w = centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
        ClusterIndex::serialize_offsets(&[0u64], offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
        ClusterIndex::serialize_bounds(BoundKind::Ball, &[], bounds_w)?;
        bounds_w.flush()?;
    }
    {
        let router_w = centroids_write.for_field_with_idx(field, CentroidSlot::Router.index());
        router.serialize(router_w)?;
        router_w.flush()?;
    }
    Ok(())
}

/// One field's build report: timings, total time, centroid and vector counts.
pub(super) type BuildReport = (Field, IvfBuildTimings, Duration, usize, usize);

/// Writes each field's IdMap entry after every Data entry, and logs each IVF field's build
/// timings on `paradedb::ivf_build`.
pub(super) fn id_maps(
    vec_write: &mut CompositeWrite,
    id_maps: Vec<(Field, Vec<DocLocation>)>,
    build_reports: &mut [BuildReport],
) -> crate::Result<()> {
    for (field, docs) in id_maps {
        let started = Instant::now();
        let id_map_start = vec_write.written_bytes();
        let id_map = vec_write.for_field_with_idx(field, VectorEntry::IdMap.index());
        IdMap::serialize_locations(&docs, id_map)?;
        id_map.flush()?;
        if let Some((_, timings, total, num_centroids, vector_count)) =
            build_reports.iter_mut().find(|(f, ..)| *f == field)
        {
            timings.vec_bytes += vec_write.written_bytes() - id_map_start;
            let elapsed = started.elapsed();
            timings.id_map_write += elapsed;
            *total += elapsed;
            log::info!(
                target: "paradedb::ivf_build",
                "ivf_build timings_ms train={} assign={} id_map_write={} encode={} total={} \
                 centroids={} vectors={} source_reads={} spill_bytes={} vec_bytes={} pad_bytes={} encode_ms={}",
                timings.train.as_millis(),
                timings.assign.as_millis(),
                timings.id_map_write.as_millis(),
                timings.encode.as_millis(),
                total.as_millis(),
                num_centroids,
                vector_count,
                timings.source_reads, timings.spill_bytes, timings.vec_bytes, timings.pad_bytes,
                timings.encode.as_millis(),
            );
        }
    }
    Ok(())
}

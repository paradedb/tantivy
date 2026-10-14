//! IVF merge: train centroids, assign every vector, encode the cluster blocks, and write the
//! `.vec` entries and `.centroids` slots.

use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
use quant::cascade::prepare_centroid;
#[cfg(test)]
use quant::cascade::LayerSpec;
#[cfg(test)]
use quant::model::Grid;

use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
#[cfg(test)]
use crate::schema::Metric;
use crate::schema::{Field, FieldType, VectorDType, VectorOptions};
#[cfg(test)]
use crate::vector::clustering::decode_row;
use crate::vector::clustering::{IvfCentroids, IvfClusterer, IvfMatrix};
#[cfg(test)]
use crate::vector::distance::l2_squared;
use crate::vector::router::RouterKind;
use crate::vector::storage::header::{
    write_centroid_header, write_vector_header, CENTROIDS_EXT, VEC_EXT,
};
use crate::vector::storage::id_map::DocLocation;
#[cfg(test)]
use crate::vector::storage::metadata::VectorColMetadata;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::{merge_source_rows, RowAddress};
use crate::vector::VectorQuantizationConfig;
use crate::{DocId, TantivyError};

mod assign;
#[cfg(test)]
mod build_tests;
mod encode;
#[cfg(test)]
mod tests;
mod train;
mod write;

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source: RowAddress,
}

/// Per-field IVF build counters and timings, reported on `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    /// Source vector reads across the training, assign, and encode passes.
    source_reads: usize,
    spill_bytes: usize,
    pad_bytes: usize,
    /// Bytes written to this field's vector entries, including entry padding.
    /// File headers and the composite footer are excluded.
    vec_bytes: u64,
    train: Duration,
    assign: Duration,
    id_map_write: Duration,
    encode: Duration,
}

/// One vector field's merge inputs.
struct FieldMerge<'a> {
    ctx: &'a PluginMergeContext<'a>,
    field: Field,
    name: &'a str,
    opts: &'a VectorOptions,
    quantization: Option<&'a VectorQuantizationConfig>,
    /// The field's reader in each source segment.
    readers: Vec<Arc<VectorIndexReader>>,
    num_target_docs: u32,
}

#[cfg(test)]
fn quantization_runtime(
    config: &VectorQuantizationConfig,
    opts: &VectorOptions,
) -> crate::Result<(Vec<LayerSpec>, Vec<Grid>)> {
    Ok(VectorColMetadata::build_ivf(opts, Some(config))?.runtime())
}

/// Writes `ctx`'s target segment as IVF: centroids trained by `clusterer`, every vector assigned,
/// and a new `router` over the centroids.
pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: &dyn IvfClusterer,
    router: RouterKind,
) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }

    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
    if num_target_docs == 0 {
        return Ok(());
    }

    let settings = clusterer.merge_settings(num_target_docs as usize)?;
    let directory = ctx.target_segment.index().directory();
    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let centroids_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(CENTROIDS_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_vector_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_file = directory.open_write(&centroids_path)?;
    write_centroid_header(&mut centroids_file)?;
    let mut centroids_write = CompositeWrite::wrap(centroids_file);

    let mut id_maps = Vec::new();
    let mut build_reports = Vec::new();
    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let quantization = ctx
            .settings
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name());
        if let Some(config) = quantization {
            config.validate(opts)?;
        }
        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let field_build_start = Instant::now();
        let source_rows = merge_source_rows(ctx, &field_readers)?;
        let vector_count = field_readers
            .iter()
            .map(|reader| reader.num_vectors())
            .sum::<usize>();
        let merge = FieldMerge {
            ctx,
            field,
            name: entry.name(),
            opts,
            quantization,
            readers: field_readers,
            num_target_docs,
        };
        if vector_count == 0 {
            write::empty_field(&mut vec_write, &mut centroids_write, &merge, router)?;
            id_maps.push((field, vec![DocLocation::ABSENT; num_target_docs as usize]));
            continue;
        }

        match opts.dtype() {
            VectorDType::F32 | VectorDType::F16 => {
                let mut timings = IvfBuildTimings::default();
                let Some(trained) = train::train(
                    &merge,
                    clusterer,
                    &settings,
                    &source_rows,
                    vector_count,
                    &mut timings,
                )?
                else {
                    // Every vector-bearing doc was deleted. Write the same empty slots as the
                    // no-vectors path: the reader errors on slots missing from composites the
                    // other fields still write.
                    write::empty_field(&mut vec_write, &mut centroids_write, &merge, router)?;
                    id_maps.push((field, vec![DocLocation::ABSENT; num_target_docs as usize]));
                    continue;
                };
                let mut centroids = trained.centroids;
                let router = train::build_router(router, opts, &mut centroids)?;
                let num_centroids = trained.num_centroids;

                let mut assigned_vectors = assign::assign(
                    &merge,
                    clusterer,
                    &settings,
                    source_rows,
                    &centroids,
                    vector_count,
                    &mut timings,
                )?;
                debug_assert_eq!(assigned_vectors.len(), trained.present_vectors);
                // The `.centroids` doc count: one posting row per document.
                let num_present_docs = assigned_vectors.len();
                let cluster_offsets = assign::cluster_offsets(&mut assigned_vectors, num_centroids);

                let (centroid_bytes, mut bounds_builder) =
                    encode::encode_centroids(&merge, &centroids, num_centroids)?;
                let locations = encode::write_data_entry(
                    &merge,
                    &mut vec_write,
                    &assigned_vectors,
                    &cluster_offsets,
                    &centroid_bytes,
                    &mut bounds_builder,
                    &mut timings,
                )?;
                id_maps.push((field, locations));

                write::centroid_slots(
                    &mut centroids_write,
                    &merge,
                    num_present_docs,
                    &centroid_bytes,
                    &cluster_offsets,
                    &bounds_builder.finish(),
                    &router,
                )?;

                build_reports.push((
                    field,
                    timings,
                    field_build_start.elapsed(),
                    num_centroids,
                    vector_count,
                ));
            }
        }
    }

    write::id_maps(&mut vec_write, id_maps, &mut build_reports)?;
    vec_write.close()?;
    centroids_write.close()?;
    Ok(())
}

/// An empty centroid matrix of `opts`'s dimension.
fn no_centroids(opts: &VectorOptions) -> IvfCentroids {
    IvfCentroids::F32(IvfMatrix {
        values: Vec::new(),
        rows: 0,
        dims: opts.dim(),
    })
}

//! The encode step: the stored centroids, and a quantized cluster's rows encoded into its block's
//! layer columns.

use std::io::Write;

use quant::cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, EncodedBatch, LayerSpec,
    PreparedCentroidWorkspace,
};
use quant::model::Grid;

use super::FieldMerge;
use crate::vector::clustering::{decode_row_append, encode_vector, IvfCentroids};
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::storage::metadata::{Slot, SlotType};
use crate::vector::VectorElement;

/// The stored centroid rows, normalized as the field's vectors are.
pub(super) fn encode_centroids(
    merge: &FieldMerge,
    centroids: &IvfCentroids,
    num_centroids: usize,
) -> crate::Result<Vec<u8>> {
    let opts = merge.opts;
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let centroid_rows: Vec<Vec<f32>> = centroid_matrix
        .values
        .chunks_exact(opts.dim())
        .map(|centroid| centroid.to_vec())
        .collect();
    let mut centroid_bytes = Vec::with_capacity(num_centroids * opts.bytes_per_vector());
    for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
        let mut bytes = encode_vector(centroid, opts.dim())?;
        let outcome = maybe_normalize_bytes(opts, &mut bytes);
        if outcome == NormalizeOutcome::NonFinite {
            log::warn!(
                "non-finite centroid {centroid_ord} in field '{}' written un-normalized during \
                 merge",
                merge.name,
            );
        }
        centroid_bytes.extend_from_slice(&bytes);
    }
    Ok(centroid_bytes)
}

/// Decoded rows one group of clusters holds between its read and its write.
const GROUP_DECODED_BYTES: usize = 256 << 20;

/// Test override of [`GROUP_DECODED_BYTES`]; 0 keeps the constant. Grouping never changes a written
/// byte, so a test may set it while other tests merge.
#[cfg(test)]
pub(super) static GROUP_DECODED_BYTES_FOR_TESTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(super) fn group_decoded_bytes() -> usize {
    #[cfg(test)]
    {
        let bytes = GROUP_DECODED_BYTES_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed);
        if bytes > 0 {
            return bytes;
        }
    }
    GROUP_DECODED_BYTES
}

/// One quantized cluster between its read and its write.
pub(super) struct PendingCluster {
    pub(super) cluster: usize,
    pub(super) rows: usize,
    /// The decoded rows, row-major, encoded in place.
    pub(super) values: Vec<f32>,
    /// One buffer per slot: the doc ids from the read, then the encoded columns.
    pub(super) bufs: Vec<Vec<u8>>,
}

/// One pool thread's encode state.
pub(super) struct EncodeWorkspace {
    pub(super) centroid: Vec<f32>,
    pub(super) prepared: PreparedCentroidWorkspace,
    pub(super) batch: BatchEncodeWorkspace,
}

/// What every cluster's encode shares.
pub(super) struct EncodeSpec<'a> {
    pub(super) centroid_bytes: &'a [u8],
    pub(super) dim: usize,
    pub(super) specs: &'a [LayerSpec],
    pub(super) grids: &'a [Grid],
    pub(super) slots: &'a [Slot],
    pub(super) tile_rows: usize,
    pub(super) scores_by_distance: bool,
}

impl PendingCluster {
    /// Encodes the rows in tiles of `tile_rows` against the cluster's prepared centroid and appends
    /// each tile's columns, then frees the decoded rows.
    pub(super) fn encode(
        &mut self,
        spec: &EncodeSpec,
        workspace: &mut EncodeWorkspace,
    ) -> crate::Result<()> {
        let stride = spec.dim * f32::SIZE_BYTES;
        workspace.centroid.clear();
        decode_row_append::<f32>(
            &spec.centroid_bytes[self.cluster * stride..][..stride],
            spec.dim,
            &mut workspace.centroid,
        )?;
        let prepared = workspace.prepared.prepare(&workspace.centroid);
        for tile in self.values.chunks_mut(spec.tile_rows * spec.dim) {
            let rows = tile.len() / spec.dim;
            let batch = encode_batch_in_place_with_workspace(
                tile,
                rows,
                prepared,
                spec.specs,
                spec.grids,
                &mut workspace.batch,
                spec.scores_by_distance,
            );
            debug_assert!(
                batch.residual_norms_squared.iter().all(|n| n.is_finite())
                    && batch
                        .layers
                        .iter()
                        .all(|layer| layer.scales.iter().all(|s| s.is_finite())),
                "a row outside the admitted range reached encode: non-finite residual norm or \
                 scale"
            );
            append_encoded_columns(&mut self.bufs, spec.slots, batch)?;
        }
        self.values = Vec::new();
        Ok(())
    }
}

/// Appends one encoded tile's columns to the cluster's column buffers, in slot order; rows and
/// document ids are gathered elsewhere.
fn append_encoded_columns(
    bufs: &mut [Vec<u8>],
    slots: &[Slot],
    batch: &EncodedBatch,
) -> std::io::Result<()> {
    for (idx, slot) in slots.iter().enumerate() {
        match &slot.slot_type {
            SlotType::ResidualNorms => {
                write_f32_run(&mut bufs[idx], &batch.residual_norms_squared)?
            }
            SlotType::QuantLayerCodes { layer, .. } => {
                bufs[idx].extend_from_slice(&batch.layers[*layer as usize].codes)
            }
            SlotType::QuantLayerScales { layer } => {
                write_f32_run(&mut bufs[idx], &batch.layers[*layer as usize].scales)?
            }
            SlotType::QuantLayerGammas { layer } => {
                write_u16_run(&mut bufs[idx], &batch.layers[*layer as usize].gammas)?
            }
            SlotType::QuantLayerErrors { layer } => write_u16_run(
                &mut bufs[idx],
                &batch.layers[*layer as usize].corrected_error_ratios,
            )?,
            SlotType::QuantLayerConstants { layer } => {
                write_f32_run(&mut bufs[idx], &batch.layers[*layer as usize].constants)?
            }
            SlotType::DocIds | SlotType::Rows { .. } => (),
        }
    }
    Ok(())
}

fn write_u16_run(writer: &mut impl Write, values: &[u16]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_run(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

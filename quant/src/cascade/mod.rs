//! Residual quantization cascade and layer-boundary operations: the shared schedule types,
//! encoding, query preparation and the error-model audits.

use std::borrow::Cow;

use crate::model::Grid;
use crate::rotation::Rotation as FhtRotation;

mod audit;
mod encode;
mod query;

pub use audit::{
    audit_prefix_error_model, audit_split_query_layer_error_squared,
    audit_split_query_layer_error_squared_with_plan, PrefixErrorAudit, PrefixErrorAuditResult,
};
pub use encode::{
    encode_batch_in_place, encode_batch_in_place_with_residual_observer,
    encode_batch_in_place_with_workspace, encode_layers, prepare_centroid,
    prepare_centroid_with_plan, BatchEncodeWorkspace, CorrectedErrorRatioRoundTrip, Encoded,
    EncodedBatch, EncodedLayerBatch, GammaRoundTrip, PreparedCentroid, PreparedCentroidWorkspace,
};
pub use query::{
    estimate_prepared_fp, estimate_prepared_fp_split, prepare_fp_query, prepare_fp_query_with_plan,
    prepare_split_query, prepare_split_query_with_plan, reconstruct_first_space, PreparedFpQuery,
    PreparedSplitQuery, QueryRotationPlan,
};

/// Packed GridPlane row width, including its tail word.
pub use crate::grid::packed_len as grid_code_stride;

/// Minimum stored gamma coefficient.
pub const GAMMA_MIN: f32 = 1.0;

/// Maximum stored gamma coefficient.
pub const GAMMA_MAX: f32 = 4.0;

/// Tagged encoder/scorer contract, independent of the numeric code width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Sign,
    Grid,
}

/// Pins the complete rotation construction; semantic changes require a new variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rotation {
    None,
    SeededFhtChaCha8 { seed: u64 },
}

impl Rotation {
    fn build(self, dim: usize) -> Option<FhtRotation> {
        match self {
            Self::None => None,
            Self::SeededFhtChaCha8 { seed } => Some(FhtRotation::new(dim, seed)),
        }
    }
}

/// Quantization parameters resolved from a segment's explicit tags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerSpec {
    /// Encoder and scorer contract.
    pub kind: LayerKind,
    /// Grid code width, or one for Sign.
    pub bits: u8,
    /// Transform applied before encoding and query preparation.
    pub rotation: Rotation,
}

/// Select the k-th largest finite score, with `k` one-indexed.
///
/// # Panics
///
/// Panics when `k` is out of range or the score count exceeds `u32::MAX`.
pub fn kth(scores: &[f32], k: usize) -> (usize, f32) {
    assert!(k > 0 && k <= scores.len());
    debug_assert!(scores.iter().all(|score| score.is_finite()));
    assert!(u32::try_from(scores.len()).is_ok());
    let mut indices: Vec<u32> = (0..scores.len() as u32).collect();
    let (_, selected, _) = indices.select_nth_unstable_by(k - 1, |&a, &b| {
        scores[b as usize].total_cmp(&scores[a as usize])
    });
    let index = *selected as usize;
    (index, scores[index])
}

/// Retain candidates whose optimistic score reaches the supplied pessimistic threshold.
///
/// # Panics
///
/// Panics when score and sigma lengths differ or the score count exceeds `u32::MAX`.
pub fn band_filter(scores: &[f32], sigmas: &[f32], kappa: f32, kth_pess: f32) -> Vec<u32> {
    assert_eq!(scores.len(), sigmas.len());
    assert!(u32::try_from(scores.len()).is_ok());
    debug_assert!(kappa.is_finite() && kth_pess.is_finite());
    debug_assert!(scores.iter().chain(sigmas).all(|value| value.is_finite()));
    scores
        .iter()
        .zip(sigmas)
        .enumerate()
        .filter_map(|(i, (&score, &sigma))| (score + kappa * sigma >= kth_pess).then_some(i as u32))
        .collect()
}

fn validate(d: usize, specs: &[LayerSpec], grids: &[Grid]) {
    validate_specs(d, specs);
    assert_eq!(specs.len(), grids.len());
    for (spec, grid) in specs.iter().zip(grids) {
        assert_eq!(spec.bits, grid.bits);
        assert!(matches!(spec.bits, 1..=4));
    }
}

fn validate_specs(d: usize, specs: &[LayerSpec]) {
    assert!(d > 0);
    assert!((1..=3).contains(&specs.len()));
    for spec in specs {
        match spec.kind {
            LayerKind::Sign => assert_eq!(spec.bits, 1),
            LayerKind::Grid => assert!(matches!(spec.bits, 2..=4)),
        }
    }
}

fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[inline]
fn aligned_le_words(bytes: &[u8]) -> Cow<'_, [u64]> {
    assert_eq!(bytes.len() % std::mem::size_of::<u64>(), 0);

    #[cfg(target_endian = "little")]
    {
        // SAFETY: `align_to` validates alignment; empty prefix and suffix gate borrowing.
        let (prefix, words, suffix) = unsafe { bytes.align_to::<u64>() };
        if prefix.is_empty()
            && suffix.is_empty()
            && words.len() == bytes.len() / std::mem::size_of::<u64>()
        {
            return Cow::Borrowed(words);
        }
    }

    Cow::Owned(decode_le_words(bytes))
}

fn decode_le_words(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks(8)
        .map(|chunk| {
            let bytes: [u8; 8] = chunk
                .try_into()
                .expect("chunk size is fixed at eight bytes");
            u64::from_le_bytes(bytes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn aligned_le_words_borrows_aligned_input() {
        let expected = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        // SAFETY: Every `u64` byte representation is valid as `u8`.
        let (prefix, bytes, suffix) = unsafe { expected.as_slice().align_to::<u8>() };
        assert!(prefix.is_empty() && suffix.is_empty());

        let words = aligned_le_words(bytes);
        assert_eq!(words.as_ref(), expected);
        assert!(matches!(words, Cow::Borrowed(_)));
    }

    #[test]
    fn aligned_le_words_decodes_deliberately_unaligned_input() {
        let expected = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let encoded = words_to_bytes(&expected);
        let mut storage = vec![0_u8; encoded.len() + 8];
        let offset = (0..8)
            .find(|offset| (storage.as_ptr() as usize + offset) % std::mem::align_of::<u64>() != 0)
            .expect("at least seven of eight consecutive byte offsets are unaligned");
        storage[offset..offset + encoded.len()].copy_from_slice(&encoded);

        let words = aligned_le_words(&storage[offset..offset + encoded.len()]);
        assert_eq!(words.as_ref(), expected);
        assert!(matches!(words, Cow::Owned(_)));
    }

    #[test]
    fn kth_matches_sort_and_edges() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut scores: Vec<f32> = (0..1_001)
            .map(|i| {
                if i % 17 == 0 {
                    0.5
                } else {
                    rng.next_u32() as f32
                }
            })
            .collect();
        scores[3] = 12.0;
        scores[4] = -9.0;
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for k in [1, 2, 17, 500, scores.len()] {
            let (index, value) = kth(&scores, k);
            assert_eq!(value, sorted[k - 1]);
            assert_eq!(scores[index], value);
        }
    }

    #[test]
    fn band_filter_retains_true_top_k_and_is_ascending() {
        let truths = [10.0_f32, 9.0, 8.0, 7.0, 6.0, 5.0];
        let scores = [8.5_f32, 10.0, 7.75, 7.1, 5.8, 5.2];
        let sigmas: Vec<f32> = scores
            .iter()
            .zip(truths)
            .map(|(&score, truth)| (score - truth).abs())
            .collect();
        let k = 3;
        let (index, value) = kth(&scores, k);
        let retained = band_filter(&scores, &sigmas, 1.0, value - sigmas[index]);
        assert!((0..k as u32).all(|top| retained.contains(&top)));
        assert!(retained.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use rand_chacha::ChaCha8Rng;
    use rand_core::RngCore;

    pub(crate) fn random_unit(rng: &mut ChaCha8Rng, d: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(d);
        while values.len() < d {
            let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            values.push((radius * angle.cos()) as f32);
            if values.len() < d {
                values.push((radius * angle.sin()) as f32);
            }
        }
        let norm = values
            .iter()
            .map(|&value| value * value)
            .sum::<f32>()
            .sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    pub(crate) fn assert_relative_eq(actual: f32, expected: f32) {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            absolute <= 1e-7 || relative <= 1e-5,
            "actual={actual}, expected={expected}, absolute={absolute}, relative={relative}"
        );
    }
}

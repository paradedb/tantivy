//! RaBitQ vector quantization.
//!
//! Stateless functions for encoding f32 vectors into compact binary-quantized
//! records and estimating distances from those records at query time.
//!
//! This module is used alongside [`BqVecPlugin`](crate::vector::bqvec::BqVecPlugin)
//! which handles the segment storage lifecycle.

pub mod distance;
pub mod math;
pub mod quantizer;
pub mod record;
pub mod rotation;
pub mod simd;

pub use distance::RaBitQQuery;
pub use quantizer::{QuantizedVector, RabitqConfig};
pub use rotation::{DynamicRotator, RotatorType};

/// Distance metric supported by RaBitQ quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Euclidean distance (L2).
    L2,
    /// Inner product (maximum similarity).
    InnerProduct,
}

/// Errors that can occur in RaBitQ operations.
#[derive(Debug)]
pub enum RabitqError {
    /// Returned when the persisted bytes are inconsistent or corrupt.
    InvalidPersistence(&'static str),
    /// Generic I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for RabitqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RabitqError::InvalidPersistence(msg) => write!(f, "invalid persisted data: {}", msg),
            RabitqError::Io(err) => write!(f, "i/o error: {}", err),
        }
    }
}

impl std::error::Error for RabitqError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RabitqError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RabitqError {
    fn from(err: std::io::Error) -> Self {
        RabitqError::Io(err)
    }
}

/// Encode a full-precision vector into a quantized byte record.
///
/// Rotates the vector, quantizes against zero centroid, and packs into
/// a fixed-size byte record suitable for [`BqVecPlugin`](crate::vector::bqvec::BqVecPlugin).
pub fn encode(
    rotator: &DynamicRotator,
    config: &RabitqConfig,
    metric: Metric,
    vector: &[f32],
) -> Vec<u8> {
    let rotated = rotator.rotate(vector);
    let zero_centroid = vec![0.0f32; rotated.len()];
    let qv = quantizer::quantize_with_centroid(&rotated, &zero_centroid, config, metric);
    record::pack(&qv)
}

/// Prepare a query for distance estimation.
///
/// Rotates the query vector and precomputes constants used by
/// [`RaBitQQuery::estimate_distance`].
pub fn prepare_query(
    rotator: &DynamicRotator,
    query: &[f32],
    ex_bits: usize,
    metric: Metric,
) -> RaBitQQuery {
    RaBitQQuery::new(query, rotator, ex_bits, metric)
}

/// Bytes needed per record for the given padded dimensionality and extended bits.
pub fn bytes_per_record(padded_dims: usize, ex_bits: usize) -> usize {
    record::bytes_per_record(padded_dims, ex_bits)
}

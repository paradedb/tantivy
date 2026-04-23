//! Vector search primitives: TurboQuant quantization, the cluster
//! index that owns per-doc record storage on disk (`cluster`), and
//! the shared rotation + scalar math utilities.

pub mod cluster;
pub mod math;
pub mod rotation;
pub mod turboquant;

/// Distance metric supported by the vector pipeline.
///
/// Lives here (rather than in a quantizer module) because both the
/// cluster plugin and the query-side collector need it without
/// knowing which quantizer produced the underlying codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Euclidean distance (L2).
    L2,
    /// Inner product (maximum similarity).
    InnerProduct,
}

/// Errors produced by vector-subsystem persistence code (rotator
/// deserialization, etc). Kept minimal — most vector code uses
/// `crate::Result` / `TantivyError` instead.
#[derive(Debug)]
pub enum VectorError {
    /// Returned when persisted bytes are inconsistent or corrupt.
    InvalidPersistence(&'static str),
    /// Generic I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::InvalidPersistence(msg) => write!(f, "invalid persisted data: {msg}"),
            VectorError::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl std::error::Error for VectorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VectorError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VectorError {
    fn from(err: std::io::Error) -> Self {
        VectorError::Io(err)
    }
}

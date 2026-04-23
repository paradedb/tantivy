//! TurboQuant: data-oblivious vector quantization.
//!
//! Implements the algorithm from
//! [TurboQuant: Online Vector Quantization with Near-optimal Distortion
//! Rate](https://arxiv.org/abs/2504.19874). Core idea: a Haar-random
//! rotation concentrates the coordinates of any unit-norm vector onto
//! a Beta distribution; scalar Lloyd-Max quantization of those
//! coordinates is then near-optimal without any training data.
//!
//! The motivating property for our use here is that encoding is
//! **per-vector and stateless** (the rotator + codebook are shared
//! globals). Encoded bytes from one segment are usable verbatim in a
//! merged segment, so merges do not need to re-encode — they can
//! byte-copy records.
//!
//! This module was ported (f64 → f32, nalgebra removed, process cache
//! added) from <https://github.com/abdelstark/turboquant> (MIT), with
//! the QJL Gaussian projection swapped for an SRHT-style FhtKac
//! rotation to reuse our existing fast orthogonal-transform
//! infrastructure.

pub mod bitpack;
pub mod codebook;
pub mod distance;
pub mod quantizer;
pub mod record;
pub mod transposed;

pub use codebook::Codebook;
pub use distance::TurboQuantQuery;
pub use quantizer::TurboQuantizer;
pub use record::bytes_per_record;

//! Quantization math and scoring kernels for tantivy's vector index. Every function takes and
//! returns slices; no tantivy type crosses the boundary.
//!
//! [`model`] holds the grids, the error model and binary16; [`rotation`] the seeded Hadamard
//! rotation; [`sign`] and [`grid`] the two code shapes and their kernels; [`cascade`] composes
//! them into layered residual encoding, query preparation and layer boundaries.

pub mod cascade;
pub mod grid;
pub mod model;
pub mod rotation;
pub mod sign;

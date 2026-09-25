# quant-model

Numerical oracle for the scalar quantizers used by the sibling kernel crates.
Grids, the closed-form error model, and binary16 conversion. No dependencies
on the other kernels.

## Grids

`build_grid(d, bits) -> Grid` runs Lloyd-Max on the exact marginal density of
one coordinate of a uniform point on the `d`-sphere, scaled by `√d` so the
coordinate has unit variance. The density is integrated on a fixed 2¹⁶-point
quadrature; iteration stops after 200 rounds or when no point moves by more
than 1e-12. Grids are forced symmetric on every round.

```rust
pub struct Grid {
    pub bits: u8,          // 1..=8
    pub points: Vec<f32>,  // 2^bits reconstruction points, ascending
    pub rho_model: f64,    // normalized RMSE of this grid on the marginal
}
```

Requires `d ≥ 64` and `bits ∈ 1..=8`. `rho_model_for_points(d, points)`
recomputes ρ for persisted points (power-of-two count, ≥ 2) so a reader can
check a stored grid against the model without rebuilding it.

Codes are assigned by nearest reconstruction point; the sibling crates
compute boundaries as midpoints between adjacent points.

## Error model

- `isotropic_sigma(rho, d) = rho / √d`: predicted dot-product error for
  unit-norm operands when the quantization error is isotropic.
- `kappa_miss(kappa) = ½·erfc(κ/√2)`: one-sided standard-normal tail, the
  probability a true value falls outside a `κσ` band.
- `empirical_sigma(estimates, truths)`: RMSE between two equal-length slices,
  for checking the model against measurement.

ρ is normalized RMSE, not an MSE ratio. The 1-bit anchor is
`√(1 − 2/π) ≈ 0.603`; the tests pin that convention.

## binary16

`f16::{f32_to_f16, f16_to_f32}` implement IEEE-754 binary16 conversion with
round-to-nearest-ties-to-even, including subnormals and NaN payloads. Row
scales, γ, and E are serialized through these, so their rounding is part of
the stored format. The `half` crate is a dev-dependency only, used to check
them.

## Tests

- `rho_table`: ρ goldens for `d ∈ {128, 768, 1536}` × `bits ∈ {1, 2, 4}`,
  measured on random unit vectors with an f16-rounded scale (≈ 0.60, 0.34,
  0.097).
- `sign_convention_anchor`: 1-bit ρ matches the Gaussian limit and is > 0.5.
- `kappa_table`: `kappa_miss` at κ = 2, 3, 4 to 2e-6.

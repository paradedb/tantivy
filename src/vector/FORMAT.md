# Quantized scan constants and numerical contract

The `.vec` identifier is V3; this change does not alter stored codes, sidecars,
or constants. Earlier `.vec` versions require rebuilding the index.

## Threshold and policy

All scores use higher-is-better score space. At each layer, and while admitting
clusters, the threshold is the **k-th largest lower endpoint**
`estimate - kappa * sigma`. It is not the lower endpoint of the k-th point
estimate. A candidate survives when its upper endpoint reaches the threshold.
Consequently, enclosing intervals preserve the true top-k; widening any interval
or increasing kappa cannot remove an existing survivor. Statistical intervals
are not guaranteed to enclose every true score, so exact reranking alone does
not guarantee recall parity.

The policy constants are uniform `kappa = 2.5`, statistical-width multiplier
`1.15`, and serialized gamma clamp `[1, 4]`. None is fitted per dataset, layer,
or query. The lower-endpoint threshold correction is independent of kappa.

## Arithmetic uncertainty

For separately rounded terms combined as a sum or difference, the additional
analytical term is `delta = c * f32::EPSILON * (abs(a) + abs(b))`. Independent
contributions are accumulated in quadrature, separately from the statistical
model; there is no empirical multiplier. The rounding-count constants are
documented at the expressions in `prepared.rs`:

- Initial L2 split subtraction: `c = 2` for operand rounding and fused subtraction.
- L2 refinement also includes the separate `prefix - constant` subtraction with
  `c = 1`.
- Initial dot/cosine scaling uses `c = 1`; refinement scaling and addition use
  `c = 2`.
- The L2 exact-base subtraction and final fused base-plus-corrected-residual
  expression each use `c = 1`, including cosine's base-plus-residual sum.

Raw-prefix variance is propagated through the current gamma correction before
combining with the model width. Refinement preserves previous arithmetic error.
Thus L2 at `query = centroid` still has positive uncertainty when cancellation
is possible, even though its data-model width vanishes.

Large common offsets can widen L2 arithmetic bounds and retain extra candidates.
Translation tests use exactly representable input translations and require the
same final ranking, true top-k survival at every boundary, and a translated
survivor set containing the original set. Centroid-relative L2 scoring remains
a follow-up to reduce this conservative cost on offset-heavy data.

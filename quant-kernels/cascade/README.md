# cascade

Layered residual quantization. Composes `fht`, `sign-plane`, and `grid-plane`
into a schedule of 1–3 layers, produces the per-row metadata the `.vec`
format stores, and provides the query side (rotation plan, split-form
scoring) and the layer-boundary selection operators.

## Schedule

```rust
pub struct LayerSpec { pub bits: u8, pub seed: u64, pub rotate: bool }
```

- 1–3 layers, `bits ∈ 1..=4`, any order. `specs[0].rotate` must be `true`.
- Each layer takes a matching `quant_model::Grid` with `grid.bits == spec.bits`.
  1-bit layers use `sign-plane` and ignore the grid's points.
- A layer with `rotate = true` applies `fht::Rotation::new(d, seed)` to the
  residual before quantizing. Later layers therefore live in a different
  coordinate space from earlier ones; the query is rotated the same way.

## Encoding

Every layer quantizes what the previous layers missed: rotate the residual
if the spec says so, encode it, subtract the reconstruction, continue.

Per row and per layer the encoder also records:

- `scale`: the exact `f32` row scale from the kernel crate.
- `γ` (binary16): `‖r‖² / ⟨r, r̂⟩`, where `r` is the original residual and
  `r̂` the cumulative prefix reconstruction. It is the scalar that makes the
  prefix carry the row's energy, clamped to `[GAMMA_MIN, GAMMA_MAX] = [1, 4]`.
- `E` (binary16): `‖r − γ_f16 · r̂‖² / ‖r‖²`, the error ratio left after
  applying the *serialized* γ, so the scan-side error model sees the same
  number the writer stored.
- `constant` (`f32`, optional): `⟨c, r̂_layer⟩` in the layer's coordinate
  space, for split-form scoring.

Entry points:

| Function | Use |
| --- | --- |
| `encode_batch_in_place_with_workspace(vectors, rows, centroid, specs, grids, ws, compute_constants)` | Cluster tile, row-major, encoded in place. Reuses `BatchEncodeWorkspace` across tiles; memory is bounded by one tile. |
| `encode_batch_in_place(..)` | Same with a fresh workspace. |
| `encode_batch_in_place_with_residual_observer(.., observer)` | Same, calling `observer(layer, residuals, scales)` after each layer, for audits. |
| `encode_layers(r, centroid, specs, grids) -> Encoded` | One residual, allocating. The reference the batch path is tested against byte-for-byte. |
| `audit_prefix_error_model(r0, specs, grids)` | Raw and serialized γ/E at every prefix depth. Test and tooling oracle; nothing here persists. |

Output is struct-of-arrays per layer, the shape the writer streams:

```rust
pub struct EncodedBatch {
    pub rows: usize,
    pub residual_norms_squared: Vec<f32>,   // radius² per row
    pub layers: Vec<EncodedLayerBatch>,
}
pub struct EncodedLayerBatch {
    pub codes: Vec<u8>,                     // rows × fixed stride
    pub scales: Vec<f32>,
    pub gammas: Vec<u16>,                   // binary16
    pub corrected_error_ratios: Vec<u16>,   // binary16 E
    pub constants: Vec<f32>,                // empty unless compute_constants
}
```

Code stride is `ceil(d/64)·8` bytes for sign layers and
`grid_plane::packed_len(d, bits)` for grid layers.

Centroids are prepared once per cluster with `prepare_centroid_with_plan` or
`PreparedCentroidWorkspace`, which rotates the centroid into every layer's
space; the rotation is applied once per cluster, not once per row (a debug
counter in `fht` enforces this in tests).

## Query side

`QueryRotationPlan::new(d, specs)` expands the schedule's seeds into
rotations once; share it with `Arc` across centroid preparation and query
preparation.

Two forms:

- **Full-precision** (`prepare_fp_query*`, `estimate_prepared_fp*`): the
  query rotated into each layer's space, scored with `sign_plane::estimate_fp`
  or a fresh grid LUT. Reference path.
- **Split** (`prepare_split_query_with_plan(query, plan, grids, sign_query_bits)`
  → `PreparedSplitQuery`): sign layers become `bq`-bit bitplanes, grid layers
  become LUTs (plus the packed byte LUT at 4 bits). Prepared once per segment.
  Per-layer scoring:
  - `score_layer(layer, codes, scale, constant, spec)` returns
    `scale · ⟨q_layer, codes⟩ − constant`, i.e. `⟨q − c, r̂⟩` without a
    per-cluster query.
  - `score_layer_batch_unscaled(..)` and `score_layer_batch_unscaled_indexed(..)`
    dispatch to the dense or indexed kernel of the layer's crate.
  - `query_error_squared(layer)` exposes the query-side quantization error
    (nonzero only for sign layers).

`estimate_prepared_fp_split` is the split form written out longhand and is
what the batch path is tested against.

## Layer boundaries

- `kth(scores, k) -> (index, value)`: k-th largest finite score, one-indexed,
  via `select_nth_unstable_by`.
- `band_filter(scores, sigmas, kappa, kth_pess) -> Vec<u32>`: indices with
  `score + kappa · sigma ≥ kth_pess`, ascending. Survivors feed the next
  layer's `_indexed` kernel.

`reconstruct_first_space(encoded, specs, grids, d)` undoes every layer's
rotation and sums reconstructions in the original space, for tests and
debugging.

## Safety

One `unsafe`: `aligned_le_words` borrows a code row as `&[u64]` through
`align_to::<u64>()` when the row is 8-byte aligned and the platform is
little-endian, and otherwise decodes a copy. Both branches are tested.

## Tests and benches

- Batch encoder matches `encode_layers` byte-for-byte, including γ/E, and
  reuses every workspace allocation across tiles.
- Serialized γ/E decode back to the reconstruction error and shared radius
  the writer will store; zero residuals are canonical.
- Layered ρ goldens (`layered_golden_and_sigma`, odd-`d` variant), the
  Gaussian γ distribution, and the exact-first-layer → zero-second-layer
  identity.
- Split form equals direct form per layer and summed; batch and indexed
  scoring equal single-row calls for every width.
- `kth` against sort; `band_filter` retains the true top-k.

`cargo bench -p cascade --bench baselines` gives fp-dot and decode-then-dot
references; `--bench cascade` times `kth` + `band_filter` at 50k and 500k.

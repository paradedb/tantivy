# tantivy-quant

Quantization math and scoring kernels for tantivy's vector index, imported by tantivy as
`quant`. No tantivy type crosses the boundary: everything here takes slices and returns slices.
`src/vector` consumes it through the format writer and the quantized scan.

| Module | Owns | Depends on |
| --- | --- | --- |
| [`model`](docs/model.md) | Lloyd-Max grids, closed-form error model, binary16 conversion | – |
| [`rotation`](docs/rotation.md) | Seeded, format-stable randomized Hadamard rotation | – |
| [`sign`](docs/sign.md) | 1-bit codes and popcount scoring kernels | `model` |
| [`grid`](docs/grid.md) | 2–4-bit packed codes and LUT scoring kernels | `model` |
| [`cascade`](docs/cascade.md) | Layered residual encoding, error metadata, split-form scoring, layer boundaries | all of the above |

Each module's page describes its API, on-disk shapes, and tests. This file covers only what
crosses module boundaries.

## How the pieces fit

**Build.** A cluster tile of row-major `f32` vectors is encoded against its
centroid:

1. Subtract the centroid. The residual's squared norm is the row's radius².
2. For each layer in the schedule, optionally rotate the residual with the
   layer's seeded `rotation::Rotation` (layer 0 always rotates), quantize it to
   1 bit (`sign`) or 2–4 bits (`grid`), and subtract the
   reconstruction. The next layer sees what this one missed.
3. Per row and per layer, record the exact `f32` scale, a binary16 γ and E
   for the error model, and optionally a split-form constant.

`cascade::encode_batch_in_place_with_workspace` does this for a whole cluster
with reusable buffers. `cascade::encode_layers` is the single-row reference
it is tested against byte-for-byte.

**Query.** A query is rotated once per layer into that layer's coordinate
space (`cascade::QueryRotationPlan` caches the expanded rotations for a
schedule). Sign layers quantize the query to affine bitplanes and score by
popcount; grid layers build a per-coordinate lookup table. Kernels return
unscaled scores; the caller applies the row's stored scale.

**Split form.** The query is prepared once per segment, not once per cluster.
Instead of scoring `⟨q − c, r̂⟩` with a per-cluster query, the kernels score
`⟨q, r̂⟩` and subtract a stored per-row constant `⟨c, r̂⟩` computed at build
time in the layer's coordinate space. `cascade::PreparedSplitQuery` is the
scan-side entry point; `cascade::estimate_prepared_fp_split` is the reference.

**Layer boundaries.** `cascade::kth` selects the k-th largest score and
`cascade::band_filter` keeps every candidate whose optimistic score `s + κσ`
reaches the pessimistic k-th. The `_indexed` batch kernels in all three
scoring modules then score only surviving row offsets from a contiguous
posting range, so a later layer never touches rows an earlier one ruled out.

## Invariants that cross modules

- **Seeds are part of the format.** `rand_chacha` and `rand_core` are pinned
  to exact versions in the workspace `Cargo.toml` because the raw ChaCha8
  stream decides rotations and permutations. Bumping them changes every
  stored code.
- **Build is build.** Nothing in the encode path samples rows, runs queries,
  or calibrates. γ and E are computed from the row itself.
- **Packed rows are 8-byte aligned with zero tail bits.** Sign rows are
  `ceil(d/64)` little-endian `u64` words; grid rows are `ceil(d·bits/64)·8`
  bytes. Bits past `d` are always zero, so byte comparison is code equality.
- **Zero residuals serialize canonically:** zero codes, zero scale, γ = 1,
  E = 0.
- **No `unsafe`** except one aligned `align_to::<u64>` cast in `cascade`
  with a decoding fallback when the input is unaligned or big-endian.

## Checks and benchmarks

```sh
./quant/check.sh           # nightly fmt --check, clippy -D warnings, tests in --release
./quant/check_asm_x86.sh   # sign emits popcnt; rotation emits packed addps/subps
```

CI (`.github/workflows/quant-kernels.yml`) runs both on x86 with
`-C target-cpu=native` and thin LTO, checks for `cnt` in `sign`'s
assembly on an ARM runner, and runs every Criterion target:

```sh
cargo bench -p tantivy-quant --bench rotation
cargo bench -p tantivy-quant --bench sign_score
cargo bench -p tantivy-quant --bench grid_kernels
cargo bench -p tantivy-quant --bench baselines
cargo bench -p tantivy-quant --bench cascade
```

The workspace release profile gives `tantivy-quant` `codegen-units = 1` and
`debug = true` so kernel symbols stay inspectable with `cargo-asm`. LTO is
set by the CI job, since Cargo has no per-package LTO.

## Reading order

`model` (constants and the error model) → `rotation` → `sign` and
`grid` (the two code shapes and their kernels) → `cascade`, starting at
`LayerSpec` and `encode_batch_in_place_reusing`, then `PreparedSplitQuery`.

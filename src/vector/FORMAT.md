# Vector quantization V3 format

Status: frozen for Phase B v1 on 2026-08-24.

Changelog:

- On 2026-08-25, before shipment, calibration v3 superseded the pre-release
  slot-15 calibration amendments. Slot 15 is retired and reserved: writers
  omit it and readers ignore it. Per-field, per-prefix `(bias, spread,
  sample_count, source)` calibration now lives only in
  `IndexSettings.vector_quantization`. There is no built-in numeric fallback
  or production-query floor.
- On 2026-08-25, before shipment, amendment 6 persisted the exact-density
  model rho alongside every used width's grid, including the sign plane.
  Nonzero quantized-query scorer resolution consumes these entries directly
  and is cached across query-driven segment reopens; query preparation never
  runs the Lloyd-Max solver. Level 0 does not resolve quantized scorer state.
- V3 amendment 3: quantized fields require `d >= 64`, matching the validated
  error-model envelope and the packed-word width; unquantized fields remain
  unrestricted.

- On 2026-08-24, before shipment, amendment 1 added the metric-gated
  residual-squared-norm slot 14 required for L2 split-form distance assembly.
- On 2026-08-24, before shipment, amendment 2 lifted the dimension divisibility
  restriction and defined word-rounded, zero-tail code rows for general `d`.

This document describes the byte contract introduced by
`VectorFileVersion::V3`. Seed expansion remains governed by the quant-kernels
golden tests. Codes use I7 coordinate packing (coordinate 0 in the least
significant bits), serialized as little-endian `u64` words.

## Container and compatibility

- Every newly written `.vec` and `.centroids` file starts with the existing
  four-byte little-endian version header, now set to `3`, followed by the
  existing `CompositeFile` body.
- V1 and V2 `.vec` files remain readable. They have no quantized slots. Flat
  segments use the exhaustive exact path; IVF segments retain centroid routing
  and the probe budget, then score the routed rows exactly. V1 `.centroids`
  remains rejected by the existing V2 bounds requirement; V2 `.centroids`
  remains readable.
- A V3 flat segment has no centroid context and therefore has no quantized
  slots. Absence of all quantized slots means quantized scoring is off; flat
  segments scan exhaustively and IVF segments use routed exact scoring.
- A V3 IVF field whose per-index configuration enables quantization must carry
  the complete configured slot prefix. A partial layer or a layer count that
  differs from the index configuration is corruption, not a fallback.

## `.vec` slots per field

Composite slot numbers are scoped by Tantivy field, so the same table applies
independently to every vector field.

| Slot | Logical payload |
|---:|---|
| 0 | Existing row-to-doc `IdMap` |
| 1 | Existing fp32 vector rows |
| `2 + 3*l` | Packed codes for zero-based layer `l` |
| `3 + 3*l` | Little-endian binary16 scale for layer `l` |
| `4 + 3*l` | Little-endian binary32 split-form constant for layer `l` |
| 14 | Little-endian binary32 `‖x-c‖²`, present iff metric assembly requires a per-row norm (L2 in v1) |
| 15 | Retired pre-release calibration slot; reserved, never written, ignored if present |

With the v1 maximum of four layers, quantized slots occupy 2 through 13. Each
layer owns three separate SoA byte ranges; layers never interleave.

Every logical array has one entry per IVF posting-membership row in the same
order as slot 1 and the explicit `IdMap`. Replicated documents therefore have
one quantized entry for every centroid membership, and storage grows by the
same replication factor as the existing fp32 rows.

For dimension `d` and layer width `b`, logical lengths for `n` posting rows are:

- codes: `n * ceil(d * b / 64) * 8` bytes;
- scales: `n * 2` bytes;
- constants: `n * 4` bytes.

`d >= 64` for quantized fields; unquantized fields are unrestricted. `b` is in
1 through 4. Coordinates are packed densely, and
every bit after the true `d * b` payload in a row's final little-endian `u64`
word is zero. Writers force this zero tail for both stored codes and prepared
query bitplanes; readers may validate it. Whole-word XOR/AND plus popcount is
therefore exact without masking, while formulas use the true `d`. Dimensions
divisible by 64 retain byte-identical V3 rows. At `d=768`, the
`[1,4]` schedule is `96 + 384 + 2*2 + 2*4 = 492` bytes per posting row for
Dot/Cosine, and 496 bytes with the L2 residual norm.

The norm is computed from the stored row and its posting centroid before any
rotation. Readers must not derive it from quantization scales or reconstructed
layers. L2 assembles `dist² = ‖q-c‖² - 2·est + ‖r‖²`, where `est` is the
split-form estimate of `⟨q-c,r⟩`; the score buffer stores `-dist²`.

Slot 15 has no active payload or decoder. Its number remains reserved so a
future format cannot reinterpret bytes written by a pre-release build. Current
writers never open this slot. Current readers neither require nor decode it;
if physical slot-15 bytes are present, they have no effect on open or scoring.

## Code alignment

The absolute file offset of each code slot is 64-byte aligned. The writer calls
`CompositeWrite::align_next_field(64, HEADER_LEN)` immediately before opening
each code slot.

`CompositeFile` records adjacent section starts and cannot represent an
unowned gap. Consequently alignment bytes are zero-valued trailers on the
preceding physical slot. A reader validates the preceding logical payload
length and permits at most 63 zero trailer bytes, then slices to the logical
length. Code, scale, and constant row strides never include alignment bytes.

## Per-index metadata

`IndexSettings.vector_quantization` is a list keyed by vector field name. This
is one per-index object shared by all segments, while allowing an index to have
vector fields with different dimensions and metrics. Each entry persists:

- quantization format version (`3`), field name, dimension, metric, and norm
  policy;
- one validity tuple per layer: width, quantizer tag, and `u64` rotation seed;
- one exact-density model entry per used width: grid version (`1`), ordered
  fp32 points, and normalized-RMSE `rho_model` (including the sign width);
- an optional calibration array with exactly one entry per active prefix
  depth. Each entry is `{ bias: f32, spread: f32, sample_count: u32, source:
  u8 }`, where source `0` is `held_out` and source `1` is `real_query`.

Calibration is field-scoped and shared by every segment of the index. All
depth entries must use one source; `bias` must be finite, `spread` finite and
non-negative, and `sample_count` nonzero. Array entry `l` calibrates the
cumulative estimate after scoring layers `0..=l`, not layer `l` in isolation.

`paradedb.vector_calibrate` supplies production queries to the production
quantized-query estimator, using at most the first 256 queries. It samples a
deterministic index-wide target of 1,000 live IVF posting-membership rows;
replicas remain separate memberships and every membership of a deleted
document is excluded. Caller queries are external to the indexed corpus, so
there is no query-cluster exclusion. For every row and prefix, exact dot,
cumulative prefix estimate, and base model sigma stay in `f32`; only the
finite normalized error
`(exact_dot - prefix_estimate) / (f16_scale[l] * rho[l] * query_norm)` is
widened into the aggregate moments. `bias` is its mean, `spread` its population
standard deviation, and `sample_count` counts accepted query-row errors. The
result is persisted with source `real_query`.

At prefix `l`, the reader centers the raw estimate exactly once by adding
`bias[l] * f16_scale[l] * rho[l] * query_norm`; the κ band uses
`spread[l] * f16_scale[l] * rho[l] * query_norm`. L2 applies its score-space
factor of two to both corrections. Selecting a deeper prefix replaces the
active bias and spread with that depth's entry; it does not add calibration
from shallower prefixes.

If the calibration array is absent, quantized scorer resolution is unavailable:
IVF search falls back to routed exact scoring and pg_search emits a notice;
there is no silent bias or spread constant. A held-out entry may initialize or
replace held-out calibration, but it cannot replace an existing real-query
entry. Only the explicit real-query update path used by
`paradedb.vector_calibrate` may replace real-query calibration.

Configuration is validated by `IndexBuilder` before any index metadata is
written: field exists and is vector/f32; dimension is at least 64 and matches;
layer count is 1 through 4; widths are 1 through 4; RaBitQ is one bit;
TurboQuant grids are present exactly once per used width with `2^b` finite,
strictly increasing points; newly materialized metadata also carries the sign
entry and finite non-negative model rho for every width; metric and
normalization agree with the schema. When calibration is present, its array
length, field types, nonzero sample counts, and uniform source obey the rules
above.

An empty configuration is the backward-compatible default. pg_search
materializes configured field entries at index creation, and IVF merges emit
the corresponding quantized slots. Segments reference the persisted per-index
seeds, grids, and optional calibration; they never fork them.

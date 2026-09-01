# Vector quantization V3 format

Status: frozen for Phase B v1 on 2026-08-24.

Changelog:

- On 2026-08-25, before shipment, amendment 8 replaced scalar slot-15 CAL
  with one entry per active prefix depth. The merge observes every prefix in
  the existing sampled encode pass, and the reader applies entry `l` after
  scoring layers `0..=l`. This removes schedule-depth dependence from prefix
  scans. Metadata payload version 2 records the layer count and `(CAL,
  sample_count)` for each prefix; local pre-amendment version-1 segments remain
  readable long enough to re-merge.
- On 2026-08-25, before shipment, amendment 7 made slot 15 honor the measured
  production-query CAL envelope. The current floor is `2.2659182`, measured
  from 1,000 Cohere scan rows; a larger field-local held-out measurement wins.
- On 2026-08-25, before shipment, amendment 6 persisted the exact-density
  model rho alongside every used width's grid, including the sign plane.
  Segment-open scorer resolution consumes these entries directly and is
  cached across query-driven segment reopens; query preparation never runs the
  Lloyd-Max solver.
- On 2026-08-25, before shipment, amendment 4 added field-scoped measured-CAL
  metadata slot 15. Each IVF merge measures the final-layer empirical/model
  dot-error sigma ratio from up to 1,024 encoded posting rows; readers apply
  the stored factor to every layer's sigma chain.
- On 2026-08-25, before shipment, amendment 5 changed the CAL query sample to
  held-out stored vectors. A pseudo-query never measures rows from its own IVF
  cluster, avoiding self-match bias while preserving the field's real vector
  distribution.
- V3 amendment 3: quantized fields require `d >= 64`, matching the validated
  error-model envelope and the packed-word width; unquantized fields remain unrestricted.

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
- V1 and V2 `.vec` files remain readable. They have no quantized slots and use
  the exact path. V1 `.centroids` remains rejected by the existing V2 bounds
  requirement; V2 `.centroids` remains readable.
- A V3 flat segment has no centroid context and therefore has no quantized
  slots. Absence of all quantized slots means quantization is off/exact.
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
| 15 | Quantization calibration metadata, present for every quantized IVF field |

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

Slot 15 is a variable-length, little-endian field payload: metadata version
`u32` (`2`), layer count `u32`, then one `(CAL f32, sample_count u32)` pair for
each prefix depth, for a total of `8 + 8*L` bytes. It is scoped to one field in
one segment because its values are measured from that merge's encoded posting
rows. The merge selects up to 64 deterministically spaced stored vectors as
pseudo-queries and samples up to 1,024 encoded rows at a deterministic
interval. Each row uses a pseudo-query whose primary assignment is a different
cluster. In the same encode pass, after every layer `l`, it observes the exact
dot product between that prefix's transformed query and remaining
reconstruction error; the denominator is the matching
`f16_scale[l] * rho[l] * query_norm` model sigma. Entry `l` is the larger of
that prefix's field-local empirical/model RMS ratio and the production-query
envelope (`2.2659182` from 1,000 measured Cohere scan rows); its sample count
identifies the selected source. L2's factor of two multiplies both sides and
therefore does not change the ratio. Empty fields store one CAL `1.0`, sample
count zero entry per built layer. After scoring prefix depth `l+1`, readers
multiply the normal f16 scale × rho × query-norm sigma chain by entry `l`.

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
  fp32 points, and normalized-RMSE `rho_model` (including the sign width).

Configuration is validated by `IndexBuilder` before any index metadata is
written: field exists and is vector/f32; dimension is at least 64 and matches;
layer count is 1 through 4; widths are 1 through 4; RaBitQ is one bit;
TurboQuant grids are present exactly once per used width with `2^b` finite,
strictly increasing points; newly materialized metadata also carries the sign
entry and finite non-negative model rho for every width; metric and
normalization agree with the schema.

An empty configuration is the backward-compatible default. pg_search's build
option will materialize the configured field entries once the Phase B writer
lands; segments only reference these persisted entries and never fork seeds or
grids.

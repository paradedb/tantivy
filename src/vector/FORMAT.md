# Vector quantization V3 format

Status: frozen for Phase B v1 on 2026-08-24.

Changelog:

- On 2026-08-29, before shipment, amendment 7 added the exact cumulative
  prefix correction `gamma` to every leading-sign layer sidecar. Slot
  `3 + 3*l` is now a cluster-blocked pair of binary16 runs: all scales for a
  cluster, then all gammas for that cluster. Gamma is clamped to `[1,4]`
  before its binary16 round trip. The build path now resolves grids and model
  rho exclusively from persisted metadata and streams slot payloads through
  merge-local temporary files outside the persistent segment namespace.
  PostgreSQL supplies resource-owned `BufFile` spills and mmap directories use
  OS temporary files, so process failure cannot publish or orphan a segment
  component.
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
| `3 + 3*l` | Cluster-blocked little-endian binary16 scale/gamma sidecar for layer `l` |
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
- scale/gamma sidecar: `n * 4` bytes;
- constants: `n * 4` bytes.

The sidecar is blocked by the existing IVF cluster ranges. For a cluster with
row range `[a,b)` and `m = b-a`, its byte block is `[4a,4b)`: the first `2m`
bytes are scales in row order and the next `2m` bytes are cumulative-prefix
gammas in the same row order. Empty clusters contribute no bytes. The block
layout is deterministic from `cluster_offsets`; no second offset table is
stored. A reader pins and decodes the cluster block, never treats the sidecar
as a global four-byte row stride.

`d >= 64` for quantized fields; unquantized fields are unrestricted. `b` is in
1 through 4. Coordinates are packed densely, and
every bit after the true `d * b` payload in a row's final little-endian `u64`
word is zero. Writers force this zero tail for both stored codes and prepared
query bitplanes; readers may validate it. Whole-word XOR/AND plus popcount is
therefore exact without masking, while formulas use the true `d`. Dimensions
divisible by 64 retain byte-identical V3 rows. At `d=768`, the
`[1,4]` schedule is `96 + 384 + 2*4 + 2*4 = 496` bytes per posting row for
Dot/Cosine, and 500 bytes with the L2 residual norm.

For an original residual `r` and the raw, pre-binary16 cumulative
reconstruction `rhat_l` through layer `l`, the encoder stores
`gamma_l = ||r||^2 / <r,rhat_l>`, clamped to `[1,4]` and rounded to binary16.
A zero residual stores exactly `1`. Production scoring multiplies the raw
cumulative estimate by the decoded gamma exactly once. Gamma is
construction-exact from the pre-binary16 encode values; the authorized
binary16 sidecar round trip is measured independently in band units and
protected by regression tests. It is a per-row correction, never a
dataset-fitted aggregate.

The cumulative effective scale is
`s_eff,l^2 = <r,rhat_l>/d = ||r||^2/(d*gamma_l)`. For a leading-sign
`[1,...]` schedule, deeper prefixes derive it as
`s_eff,l^2 = s1^2 * gamma1/gamma_l`. A grid-first `[2]` or `[4]` schedule
additionally requires a stored effective scale because a grid's RMS scale does
not determine `<r,rhat>`. That grid-first sidecar extension is intentionally
out of scope: these measured-dead schedules remain format-legal, but the
current merge writer returns a clear error rather than substituting the grid
RMS scale or a fitted constant.

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
length. Code, sidecar, and constant logical lengths never include alignment
bytes.

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
non-negative, and `sample_count` nonzero. Array entry `l` measures the
cumulative estimate after scoring layers `0..=l`, not layer `l` in isolation.
These entries are verification/regression evidence only. They are never
applied to an estimate or a band width. A gamma-corrected build must measure
an absolute residual normalized bias no larger than `0.3`; the stored bias is
the tripwire for violating that construction invariant, not a correction.

`paradedb.vector_calibrate` supplies production queries to the production
quantized-query estimator, using at most the first 256 queries. It samples a
deterministic index-wide target of 1,000 live IVF posting-membership rows;
replicas remain separate memberships and every membership of a deleted
document is excluded. Caller queries are external to the indexed corpus, so
there is no query-cluster exclusion. `bias` is the normalized error mean,
`spread` its population standard deviation, and `sample_count` counts accepted
query-row errors. The result is persisted with source `real_query`; source and
replacement precedence remain audit provenance, not scorer inputs. Absence of
the calibration array does not create a numeric fallback.

V3 sign-query preparation uses exactly `B_q=4` bitplanes. This is a fixed
format/kernel parameter, not a dataset-fitted scorer policy. At query
preparation, every sign layer measures the exact constant
`B_j = ||R_j(q) - Q_Bq(R_j(q))||^2` from the f32 rotated query and the values
reconstructed by that layer's prepared bitplanes. A grid layer scores an exact
f32-query LUT and therefore contributes exactly `B_j = 0`. For prefix `l`, let
`u_c = q-c` for L2 and `u_c = q` for Dot/Cosine. The analytical dot-error
variance is

`s_eff,l^2 * gamma_l * (gamma_l - 1) * ||u_c||^2
 + gamma_l^2 * sum_{j<=l, sign}(s_j^2 * B_j)`.

The band sigma is the square root of that value multiplied by the declared
analytical safety factor `1.15`; L2 multiplies both estimate-space error and
sigma by two. No measured aggregate is part of this chain.

Boundary confidence levels remain policy, not fitted calibration. Under the
normalized analytical model `S=1`, a true-member miss at one boundary is at
most `Q(kappa)`: `Q(2) ~= 2.3e-2` and `Q(4) ~= 3.2e-5`. The first boundary uses
`kappa=2`; a terminal sign boundary uses `kappa=2`, and a terminal grid
boundary uses `kappa=4`. Real misses are expected to be rarer because this
bound ignores the true member's score margin. A larger terminal kappa buys
caution cheaply after earlier filtering, whereas early kappa is paid in
survivors. Any kappa change requires an explicit design ruling; it is never
derived silently from a dataset measurement.

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

## Construction and scan invariants

- **R1 — scorer constants.** No dataset-fitted constant is ever applied in
  scoring. Corrections are per-row construction-exact (`gamma`, subject only
  to its declared binary16 sidecar representation), per-query exact (`e_q`),
  or closed-form theory. A situation that appears to require an applied
  aggregate is a stop-and-report condition. The complete declared policy
  constants are kappa per boundary, analytical safety `1.15`, and the gamma
  clamp `[1,4]`.
- **R2 — quantization build purity.** The quantization subgraph reachable
  after IVF assignment consists only of resolving persisted format constants,
  encoding the assigned rows, and writing their slots. Calibration sampling,
  pseudo-queries, diagnostic estimators, and analysis passes are forbidden on
  that subgraph. IVF centroid-training sampling is an intrinsic part of the
  preceding clustering construction and is the explicit exemption; it does
  not calibrate or alter quantization constants. A source-level regression
  test rejects known calibration and analysis entry points in the production
  merge module.
- **R3 — scan granularity.** Scan uses whole-cluster batch operations. There
  is no per-row branch, filtering decision, or candidate-struct
  materialization inside a scoring loop. Decisions occur only at cluster
  admission or a layer boundary.

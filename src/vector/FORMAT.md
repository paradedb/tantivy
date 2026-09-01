# Vector quantization V3 format

Status: frozen for Phase B v1 on 2026-08-24.

This document describes the byte contract introduced by
`VectorFileVersion::V3`. Kernel packing and seed expansion remain governed by
the quant-kernels golden tests.

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

With the v1 maximum of four layers, quantized slots occupy 2 through 13. Each
layer owns three separate SoA byte ranges; layers never interleave.

Every logical array has one entry per IVF posting-membership row in the same
order as slot 1 and the explicit `IdMap`. Replicated documents therefore have
one quantized entry for every centroid membership, and storage grows by the
same replication factor as the existing fp32 rows.

For dimension `d` and layer width `b`, logical lengths for `n` posting rows are:

- codes: `n * d * b / 8` bytes;
- scales: `n * 2` bytes;
- constants: `n * 4` bytes.

`d` is non-zero and divisible by 64; `b` is in 1 through 4. At `d=768`, the
`[1,4]` schedule is `96 + 384 + 2*2 + 2*4 = 492` bytes per posting row.

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
- one exact-density grid per TurboQuant width: grid version (`1`) and ordered
  fp32 points.

Configuration is validated by `IndexBuilder` before any index metadata is
written: field exists and is vector/f32; dimension matches and is divisible by
64; layer count is 1 through 4; widths are 1 through 4; RaBitQ is one bit;
TurboQuant grids are present exactly once per used width with `2^b` finite,
strictly increasing points; metric and normalization agree with the schema.

An empty configuration is the backward-compatible default. pg_search's build
option will materialize the configured field entries once the Phase B writer
lands; segments only reference these persisted entries and never fork seeds or
grids.

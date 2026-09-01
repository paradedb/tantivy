# Vector storage format

This document is the normative contract for vector storage, residual
quantization, construction, and quantized scanning.

## File framing

The sole accepted vector-format identifier is the little-endian `u32` value
`4`. Every `.vec` and `.centroids` file begins with that identifier followed by
one composite-file body. A different identifier is an error whose remedy is
`rebuild required`. A reader has one decoder and performs no migration.

Composite slots have logical lengths. A writer may append at most 63 zero
bytes to a physical slot so the following code slot begins at an absolute
64-byte boundary. Readers slice each slot to its logical length and validate
every alignment trailer byte as zero.

## Index settings

Quantization is configured independently for each vector field. Absence of a
field configuration disables quantized storage and scoring for that field. A
configuration contains:

- field name, dimension, metric, and normalization policy;
- one to three residual layers, each with a width and `u64` seed;
- one exact-density model/grid entry for every width used by the schedule,
  containing its width, reconstruction points, and normalized model value
  (sign encoding ignores the points at runtime);
- optional caller-query diagnostic results, stored per active prefix as bias,
  spread, sample count, source tier, and a required non-empty protocol tag.

A quantized field has dimension at least 64 and binary32 rows. Every layer
width is in `1..=4`. Width one selects the sign quantizer; widths two through
four select the grid quantizer. Layers may appear in any order, including a
grid layer first. Seeds and grids are generated once per index and shared by
every segment and merge.

Diagnostic results are written only by the explicit calibration function.
Construction, query preparation, uncertainty calculation, filtering, and
scoring never read them. Every persisted depth carries the same source and
protocol tag so results from different measurement procedures cannot be
silently combined.

## `.vec` slot map

There are twelve field-scoped slots:

| Slot | Contents | Presence |
|---:|---|---|
| 0 | row-to-document map | always |
| 1 | binary32 vector rows | always |
| 2 | binary32 residual squared radius `R0² = ||x-c||²` | every quantized field |
| 3 | layer 0 packed codes | every quantized field |
| 4 | layer 0 scale/gamma/corrected-error sidecar | every quantized field |
| 5 | layer 0 binary32 split constants | L2 quantization only |
| 6 | layer 1 packed codes | when layer 1 exists |
| 7 | layer 1 scale/gamma/corrected-error sidecar | when layer 1 exists |
| 8 | layer 1 binary32 split constants | L2 quantization only, when layer 1 exists |
| 9 | layer 2 packed codes | when layer 2 exists |
| 10 | layer 2 scale/gamma/corrected-error sidecar | when layer 2 exists |
| 11 | layer 2 binary32 split constants | L2 quantization only, when layer 2 exists |

Every stored entry is per IVF posting-membership row in storage order. A
document assigned to several clusters therefore has one entry for each
membership, computed against that membership's centroid. Codes, sidecars, and
constants from different layers never share a byte range.

Slot 2 is mandatory for a quantized field under every metric. Constants are
mandatory for quantized L2 and absent for quantized dot and cosine. The reader
represents constants as an optional typed payload; it never substitutes a
number for an absent slot. A presence mismatch, incomplete configured layer,
extra layer, or quantized payload without matching field settings is
corruption.

## Packed codes

For dimension `d` and layer width `b`, one code row occupies

```text
ceil(d * b / 64) * 8 bytes.
```

Codes form a least-significant-bit-first stream serialized as little-endian
`u64` words. Coordinate zero begins at bit zero. A sign code of `1` means a
positive rotated coordinate. A grid coordinate occupies `b` consecutive bits
and stores its reconstruction-point index. Every bit after the `d * b` payload
is zero. Packers and query-bitplane preparation force the zero tail, and a
reader validates it once per pinned range. Whole-word scoring kernels therefore
need no tail mask.

Each code slot begins at an absolute 64-byte-aligned file position and has a
fixed row stride. Posting ranges come from the IVF cluster-offset table; no
second posting-offset structure exists.

## Cluster-blocked sidecars

For a cluster with `n` posting rows, each layer sidecar is one blocked
structure-of-arrays record:

```text
[little-endian f32 scale; n]
[little-endian f16 gamma; n]
[little-endian f16 corrected-error E; n]
```

The record occupies `8n` bytes. A reader pins the cluster range once and splits
the three contiguous runs. Values from different runs are never interleaved by
row, and no other sidecar layout is valid.

The encoder stores each scale in binary32 without narrowing. A sign layer uses
the mean absolute value of its rotated input residual. A grid layer uses the
root-mean-square value of its rotated input residual.

For original residual `r = x-c` and cumulative reconstruction `h_l` through
layer `l`, let

```text
R0²         = ||r||²
gamma_raw_l = R0² / <r, h_l>
gamma_l     = f16_decode(f16_encode(clamp(gamma_raw_l, 1, 4)))
E_raw_l     = ||r - gamma_l * h_l||² / R0²
E_l         = f16_decode(f16_encode(E_raw_l)).
```

The `gamma_l` used to compute `E_raw_l` is exactly the serialized, clamped,
binary16 value used by scoring. The encoder serializes the ratio as `E_l`, and
the reader uses that decoded value directly. When `R0² = 0`, the canonical
values are `gamma_l = 1` and `E_l = 0` for every layer. Sidecar gamma must be
finite and in `[1,4]`; corrected error must be finite and non-negative.

These definitions give the corrected residual estimator, exact error with
respect to the stored correction coefficient, and stored uncertainty term:

```text
corrected residual estimate = gamma_l * raw_prefix_estimate_l
exact corrected data-error energy = R0² * E_raw_l
stored uncertainty data term = R0² * E_l.
```

Slot 2 stores `R0²` in binary32 for every quantized metric. L2 also uses it
for exact distance assembly; dot and cosine use it only in the uncertainty
calculation.

## L2 split constants

Each present constants slot contains one little-endian binary32 value per
posting row. It removes the centroid cross term from the corresponding
residual-layer dot estimate. The higher-is-better L2 score is

```text
-distance² = -||q-c||² + 2 * corrected_residual_dot - R0².
```

Routing supplies `||q-c||²`; slot 2 supplies `R0²`. Dot and cosine require
no split constant, so construction and scanning omit those slots and their
per-row constant work.

## Scoring and uncertainty

The sign kernel consumes a four-bit affine quantization of the rotated query.
Grid LUT kernels consume the rotated binary32 query directly. Query preparation
records the exact squared sign-query reconstruction error `B_j` from the
bitplanes used by the kernel. A grid layer contributes zero query error.

At active prefix `l`, define the dot-space variance for posting row `i` as

```text
variance_l,i = (R0²_i * E_l,i / d) * ||u_c||²
             + gamma_l,i²
               * sum_{j <= l, sign layer j}(scale_j,i² * B_j).
```

For L2, `u_c = q-c`; for dot and cosine, `u_c = q`. Query-error terms attach
only to the sign layers whose kernels consume quantized queries, at each
layer's own stored scale. The same cumulative gamma corrects the complete raw
prefix, so its square multiplies the accumulated query-error term.

The confidence width is

```text
sigma_l = 1.15 * sqrt(variance_l,i)
```

for dot and cosine. L2 multiplies this width by two exactly once, after the
square root, matching the factor on its residual dot. Each boundary retains a
row when its optimistic endpoint reaches the pessimistic k-th endpoint.

The complete declared scoring-policy constants are:

- uniform boundary width `kappa = 2` at every checkpoint and for every
  schedule;
- analytical safety multiplier `1.15` applied once to `sigma`;
- serialized gamma clamp `[1,4]`.

**R1 — scoring inputs.** No dataset-fitted aggregate is applied in scoring.
Every correction is stored per row, computed per query, or closed-form theory.
A situation that appears to require an applied aggregate is an error to report,
not a reason to introduce one. A policy change requires the design owner's
approval. Under the unit-spread normal model, one true
member's per-checkpoint miss probability is at most
`Q(2) = 0.0227501` (approximately `2.3e-2`). Without an independence
assumption, the union-bound worst case for `L` checkpoints is
`min(1, L * Q(2))`: `0.0227501`, `0.0455002`, and `0.0682503` for one, two, and
three checkpoints. Score margin is absent from this bound. A recall shortfall
is addressed only by probe fraction or a uniform-kappa policy change, never a
per-boundary exception.

## Construction contract

**R2 — build purity.** Construction consists only of constants, encoding, and
writing:

1. Resolve rotation plans and seeds, persisted grids and model values, IVF
   centroids and assignments, and a deterministic byte-layout table for every
   present slot and cluster.
2. Encode one cluster at a time in storage order. Reused cluster scratch holds
   rotations, reconstructions, packed codes, scales, gamma, corrected error,
   radius, and L2 constants. Each present slot streams to its own temporary
   file. Closing the field splices those streams into the composite slots.

Memory is proportional to one cluster rather than the segment. No build or
merge entry point performs sampling, pseudo-query generation, calibration,
diagnostic analysis, or data-fitted model work. Merge consumes the grids and
model values persisted in index settings.

## Scan-shape contract

**R3 — scan granularity.** Routing and admission make decisions only at cluster
granularity. Scoring is batch-shaped: an admitted cluster pins each required
contiguous slot range, decodes the three sidecar runs into reused
structure-of-arrays scratch, invokes one batch kernel over its logical row
batch, and executes fissioned combine and uncertainty loops. A scoring loop
contains no per-row allocation, branch, filter decision, or candidate-structure
construction.

At each layer boundary, one segment-wide pass chooses the k-th score, applies
the uniform confidence band, and constructs storage-ordered survivors. A later
layer consumes survivors through indexed batch kernels and density-adaptive
range reads without scalar scoring helpers or gathered code copies. Cosine may
batch survivors across cluster boundaries; L2 remains cluster-local because
its metric base is cluster-specific. Final survivors are deduplicated by
document, fetched once in storage order, rescored exactly, and reduced to
top-k.

Setting scan depth to zero disables quantized scoring while preserving IVF
routing and the probe budget; routed rows are scored exactly. Flat segments
use their exact scan path.

## Diagnostics contract

Diagnostics are explicit post-build operations. They may compare quantized
prefix estimates with exact scores and report bias, spread, gamma clamp and
round-trip statistics, corrected-error statistics, survivor fractions, and
recall. The SQL calibration operation is the only diagnostic allowed to write
protocol-tagged metadata. Error-model and confidence-cone verification
operations are read-only. Readers and scorers do not consume diagnostic
metadata, and construction never invokes diagnostics.

Design history and measurements are recorded in `quant-kernels/RESULTS.md`.

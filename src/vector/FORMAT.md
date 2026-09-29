# Vector storage format

## File headers and entries

`.vec` uses a little-endian u32 version header, with current and supported version
**4**. Other versions fail with “rebuild required.” The `.centroids` grammar and
its version **3** describe routing, posting offsets, and cluster bounds.
`VectorQuantizationConfig.format_version = 3` identifies the independent index
settings grammar; settings specify the target for future builds.

Each vector field declares exactly these composite entries:

| Index | Entry | Contents |
|---|---|---|
| 0 | IdMap | Identity, Bitmap, or DocLocations document addressing |
| 1 | Data | Field metadata, aligned blocks, and a stored block directory |

Missing entries or any additional entry index are corruption. All Data entries
precede all IdMaps. Data entries start and end on `ENTRY_ALIGN` (8-byte) boundaries
in absolute file offsets. IdMaps follow all Data entries and need no alignment.
The composite directory derives each entry's length from its successor's start;
entry-end padding belongs to Data, so no gap is added to an IdMap. The first
Data entry may be preceded by padding outside any entry.

An IdMap begins with a u8 tag: 0 for Identity (document count supplied by the
segment), 1 for the columnar OptionalIndex Bitmap encoding, or 3 for
DocLocations. All other tags, including 2, are rejected. Identity and Bitmap
have document-ordered rows.

DocLocations contains exactly `max_doc` records of `cluster:u32, local:u32`,
both little-endian, with no padding. Record `doc` starts at body offset `8*doc`.
Absent documents store `(u32::MAX, 0)`; the cluster count must be less than
`u32::MAX`. Opening the entry checks its body length is exactly `8*max_doc`.
Each lookup reads one record and checks `cluster < K` and `local < rows_in(cluster)`.
The entry opens lazily for document lookups. Scans, reranking and result assembly
use Data columns and do not open it.

## Data entry and blocks

```
meta_len:u32
meta:bytes[meta_len]
zero padding to block_align(slots), relative to the entry start
block[0] ... block[B-1]
zero padding to 8, relative to the entry start
byte_starts:u64[B+1]
row_starts:u32[B+1]
zero padding to 8, relative to the entry start
num_blocks:u64                 // B, the entry's last 8 bytes
```

Each slot declares its decoder element: U8, F16, F32, U32 or U64. `ElemType::size()`
uses `size_of` on the corresponding fixed-width representation (u16 for F16),
never ABI alignment. A const assertion requires every size to be a power of two.
`MAX_ELEM_BYTES` is the maximum over `ElemType::ALL` and is publicly exported
for storage-provider alignment checks. `ENTRY_ALIGN` is fixed at 8 bytes and must
be at least `MAX_ELEM_BYTES`; it governs entry and directory padding. There is no
stored alignment field.

A column starts at the next multiple of its own element size, relative to the
block start, and contains `n * stride` bytes. Every stride must be a multiple
of its element size. `block_align(slots)` is the maximum element size in that
field. The writer pads each block to that alignment; an empty block consumes
no column bytes. The writer records actual byte positions and cumulative row
counts as it streams blocks. Before the directory it pads the block area to
8 with zeros and sets `byte_starts[B]` to that aligned directory start.

The terminal `num_blocks` locates the directory without reconstructing block
lengths. With entry-relative `entry_end`, its start is exactly:

```
directory_start = entry_end - 8 - align8(4 * (B + 1)) - 8 * (B + 1)
```

All directory values are little-endian. Search initialization loads both arrays
whole and validates their framing before column access. Byte boundaries are
nondecreasing and aligned to the field's decoder elements; nonempty blocks
advance. Row boundaries are nondecreasing, begin at zero, and end at the field's
row count. Equal boundaries represent empty IVF clusters. The final byte
boundary equals `directory_start`. The last block's column end bounds the zero
padding before the directory; zero padding after the row array is also checked.
Column reads cannot exceed their stored block boundary. The writer asserts that
every Data entry length is a multiple of `ENTRY_ALIGN`. All padding is zero.

`Clusters` requires `row_starts` to equal the posting offsets in `.centroids`,
a DocLocations IdMap, and centroid data. `Uniform { rows_per_block }` requires
an Identity or Bitmap IdMap, no centroid data, and a nonzero block size. Its
stored boundaries must describe `[b*r, min((b+1)*r, num_rows))`. Zero flat rows
means zero blocks; both directory arrays still contain one sentinel.
The flat writer stores `Plain` with `rows_per_block = 16384`. Plain blocks align
to F32; blocks containing SignPlane codes align to U64.

Column order and widths are part of the contract:

| Column | Decoder element | Row stride | Band |
|---|---|---:|---|
| Rows | F32 | dim * dtype width | none |
| DocIds, clustered only | U32 | 4 | none |
| ResidualNorms | F32 | 4 | 0 |
| QuantLayerCodes(l) | U64 for SignPlane, U8 for GridPlane | quantizer code stride | l |
| QuantLayerScales(l) | F32 | 4 | l |
| QuantLayerGammas(l) | F16 | 2 | l |
| QuantLayerErrors(l) | F16 | 2 | l |
| QuantLayerConstants(l), L2 only | F32 | 4 | l |

Flat Plain has only Rows; clustered Plain has Rows followed by DocIds.
DocIds ascend within each block. Quantized adds ResidualNorms after DocIds
when clustered, then each layer's columns in
ascending layer order. Norms, scales and constants are binary32; gammas and
errors are binary16. SignPlane code stride is `ceil(dim/64)*8`; GridPlane stride
is `ceil(dim*bits/64)*8` as defined by `grid_plane::packed_len`. Code tail bits
must be zero. Rows come first to permit streaming source bytes immediately;
encoded columns are buffered for one block, then flushed in column order.

Band 0 includes ResidualNorms through the last layer-0 column. Higher bands run
from that layer's codes through its final column. `layer_span(b, l)` is band l;
each band uses one read with columns exposed as views.

DocIds are read separately and validated as strictly ascending and below the
segment's `max_doc`. Filters and deleted-document visibility select rows before
any payload read; a cluster with no survivors reads only DocIds. With no filter
or deletions, quantized scans read no DocIds until rerank resolves the final
candidates, once per candidate-bearing cluster. Exact scans without a row gate
read Rows alone and resolve DocIds only when a score reaches heap admission.
Filtered exact scans plan reads over survivor rows in the Rows column.

Sparse code reads group
selected rows only while their storage-page spans overlap; adjacent disjoint
pages start a new group. Each range ends at its last selected row. Sidecars from
scales through the last sidecar or constants column use one span per touched
cluster. Codes are never coalesced with sidecars. Multi-page requests copy under
paged storage, so equal page counts do not make wider reads free. Column row
ranges cannot cross blocks; row fetching splits sorted requests into blocks.
File-offset alignment does not imply aligned memory. Decoders handle unaligned
bytes and do not assume a particular storage-page layout.

## Metadata grammar

All multibyte fields are little-endian. Tags use explicit maps, independent of
Rust discriminants. The `.vec` version selects the grammar; metadata has no
additional version field.

```
meta       := repr:u8 field [n_layers:u8 quantizer{n_layers} if repr=1]
field      := dim:u32 dtype:u8 metric:u8 norm_policy:u8 partition
partition  := kind:u8 [rows_per_block:u32 if kind=1]
quantizer  := kind:u8 payload
  SignPlane := rotation rho_model:f64
  GridPlane := bits:u8 rotation grid_version:u32 rho_model:f64
               n_points:u16 points:f32{n_points}
rotation   := kind:u8 [seed:u64 if kind=1]
```

| Tag domain | Frozen values |
|---|---|
| repr | 0 Plain, 1 Quantized |
| dtype | 0 F32 |
| metric | 0 L2, 1 Dot, 2 Cosine |
| norm_policy | 0 None, 1 UnitL2 |
| partition | 0 Clusters, 1 Uniform |
| quantizer | 0 SignPlane, 1 GridPlane |
| rotation | 0 None, 1 SeededFhtChaCha8 |

Unknown tags fail at open. SignPlane is the one-bit popcount contract without
grid points. GridPlane permits bits 2 through 4 and exactly `1 << bits` points.
SeededFhtChaCha8 pins FHT construction, ChaCha8, and `rand_core::seed_from_u64`.
Quantized fields contain one to three layers. Dimension, dtype, metric and
normalization must match the schema. Parsing rejects truncated records,
trailing metadata bytes, invalid geometry, and entry-length mismatches as
DataCorruption before payload access.

## Prepared-query identity

Quantized metadata compares and hashes `(dim, metric, layers)`. A quantizer's
identity includes its tag, width, rotation tag and seed, and model bits. Grid
identity is `(rho_model.to_bits(), points.map(to_bits))`; its descriptive version
is excluded. Signed zero and all floating-point bit distinctions are preserved.
Equality and hashing share the same field helper.

Plain metadata compares by variant only and is never a prepared-query key.
Normalization, dtype and partition do not affect preparation. Full
storage identity, including round trips, compares serialized metadata bytes.
A collector caches one OnceLock cell per metadata key and releases its map lock
before preparation. Read paths use the segment's metadata, never global build
settings.

## Versioning rules

1. Stored values need no version: grid points, rho_model, block size,
   and IdMap data are self-describing inputs.
2. The `.vec` version alone versions byte grammar, entry framing, and metadata
   encoding. New tag values need no bump; adding fields to an existing record does.
3. Quantizer contracts and seed-derived rotation semantics are pinned by enum
   tags. Semantic changes add a variant and retain the existing decode arm.
4. `slots()` and `layer_slots()` are format semantics, including ordering,
   element widths, strides, and band ownership. Golden tests pin these lists.
5. Golden bytes pin encoder columns and transforms for each quantizer and rotation.
6. Query policy remains code: kappa, statistical-width multiplier and SIGN_QUERY_BITS.
7. Retiring a decode arm is deliberate; segments using retired tags fail with
   “rebuild required.”

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

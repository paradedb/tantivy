# grid-plane

2–4-bit scalar quantization against a `quant-model` grid, with packed code
rows and lookup-table scoring.

## Row layout

`packed_len(d, bits) = ceil(d · bits / 64) · 8` bytes. Codes are little-endian
bit fields: 2- and 4-bit codes sit at `bits · (i % (8/bits))` inside byte
`i / (8/bits)` and never straddle a byte; 3-bit codes straddle and use a
separate pack/unpack path (`code_at_3`). Rows are padded to a multiple of
8 bytes and every bit past `d · bits` is zero.

The row scale is RMS, `‖y‖ / √d`, matching the unit-variance sphere
marginal the grid was built for. Codes are the nearest grid point of
`y / scale`, found by `partition_point` over midpoint boundaries. An all-zero
input encodes to zero bytes and scale `0.0`.

```rust
let grid = quant_model::build_grid(d, bits);      // bits in 2..=4 here
let mut codes = vec![0u8; packed_len(d, bits)];
let scale_f16 = encode(&y, &grid.points, bits, &mut codes);
let scale_f32 = encode_f32(&y, &grid.points, bits, &mut codes);
// *_with_scratch variants take a d-byte code buffer to avoid allocating
let y_hat = decode(&codes, &grid.points, d, bits, scale_f16);
decode_into(&codes, &grid.points, bits, scale_f16, &mut out);
```

`validate_grid` asserts `bits ∈ 2..=4` and `grid.len() == 2^bits`.

## Query preparation

- `build_lut(u, grid, bits)`: coordinate-major table of `u[i] · grid[j]`,
  `d × 2^bits` entries. A row's unscaled score is one lookup per coordinate.
- `build_packed_lut_4(lut, d)`: for 4-bit codes only. Folds each pair of
  adjacent coordinate tables into one 256-entry table indexed by the packed
  code byte, so scoring does one lookup per byte instead of two nibble
  extractions. An odd `d` keeps its last 16-entry table unfolded.

## Scoring

Single row:

- `score(codes, lut, d, bits)`: unscaled.
- `estimate(codes, scale_f16, lut, d, bits)` / `estimate_f32(..)`: scaled.

Batch, fixed stride, one output per row:

- `score_batch(codes, code_stride, lut, d, bits, out)`
- `score_batch_indexed(codes, code_stride, row_offsets, lut, d, bits, out)`
- `score_batch_packed_4(codes, code_stride, packed_lut, d, out)`
- `score_batch_packed_4_indexed(codes, code_stride, row_offsets, packed_lut, d, out)`

The `packed_4` kernels process eight rows per inner batch. The `_indexed`
variants score only the rows named in `row_offsets`, for candidates that
survived an earlier layer.

## Tests and benches

Tests cover packing round trips and bit order for every width, odd-`d`
zero tails, scratch-vs-allocating equality, LUT scores against a direct sum,
batch and indexed kernels against the scalar path for all widths, canonical
zero rows, and a golden for the packed 4-bit pipeline (`build_packed_lut_4`
→ `score_batch_packed_4`) so a change to the fold or the byte order fails
before it reaches a stored file.

`cargo bench -p grid-plane --bench kernels` times encode, scalar score, and
the batch kernels at `d = 768` for 2, 3, and 4 bits.

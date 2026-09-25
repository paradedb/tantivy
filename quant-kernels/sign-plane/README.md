# sign-plane

1-bit encoding and popcount scoring. A row is the sign of each coordinate
plus one scale; a query is a small number of bitplanes so scoring is
AND + popcount.

## Row layout

`d` sign bits packed little-endian into `ceil(d/64)` `u64` words (bit `i` of
word `i/64` is set when `y[i] > 0`). Bits past `d` are always zero; every
kernel checks this in debug builds, and `pack`/`unpack` are the only
functions that touch the layout directly.

The row scale is the mean absolute value, `Σ|y_i| / d`, which minimizes
squared reconstruction error for sign codes. `encode` returns it as binary16;
`encode_f32` returns the exact value. An all-zero input encodes to zero words
and scale `0.0`.

## Query preparation

```rust
let q = prepare_query(&u, bq);   // bq in 1..=8
```

Quantizes the query affinely: `code_i = round((u_i − lo) / delta)` with
`lo = min(u)`, `delta = (max − min) / (2^bq − 1)`. Bit `b` of every code goes
into plane `b`, so `QueryPlanes` holds `bq` word vectors plus `lo`, `delta`,
and `sum_codes`. `error_squared()` is the query's own quantization error,
which the error model adds to the row-side term.

## Scoring

With `P` = number of positive signs and `S` = `Σ_b 2^b · popcount(x & plane_b)`
(the query-code sum over positive coordinates), the dot-product estimate is

```
Σ_i sign_i · (lo + delta · code_i)  =  lo · (2P − d)  +  delta · (2S − sum_codes)
```

- `score_asym(x, q) -> (P, S)`: the raw counts.
- `estimate_asym_unscaled(x, q)`: the formula above, no row scale.
- `estimate_asym(x, scale_f16, q)` / `estimate_asym_f32(x, scale, q)`: scaled.
- `estimate_fp*(x, query)`: against a full-precision query, for references
  and the non-split path.
- `score_sym(x, q)`: Hamming distance between two sign rows.

Batch kernels take contiguous fixed-stride rows and write one score per row:

- `estimate_asym_batch_unscaled(rows, words_per_row, q, out)`
- `estimate_asym_batch_unscaled_indexed(rows, words_per_row, row_offsets, q, out)`
  scores only the rows named in `row_offsets`, for candidates that survived
  an earlier layer.

Both have a specialized path for `bq = 4` (the four planes unrolled in one
loop) and a generic plane loop otherwise.

## Tests and benches

Tests cover packing round trips at even and odd `d` with zero tails, the
Hamming score, formula exactness against decode-then-dot, canonical zero
rows, the f16 scale against a reference reconstruction, indexed-vs-dense
batch equality on sparse selections, and statistical σ against
`quant_model::isotropic_sigma`.

`cargo bench -p sign-plane --bench score` times the single-row and batch
kernels. `check_asm_x86.sh` and the ARM CI job assert the release build uses
hardware `popcnt` / `cnt`.

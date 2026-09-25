# fht

Seeded, format-stable, block-diagonal randomized Hadamard rotation. Applied
to residuals before quantization so coordinates look isotropic to a scalar
quantizer, and to queries so they land in the same space.

## API

```rust
let rotation = Rotation::new(d, seed);   // d > 0, d <= u32::MAX
rotation.apply(&mut x);                  // in place, allocates d scratch
rotation.apply_inverse(&mut x);
rotation.apply_with_scratch(&mut x, &mut scratch);          // caller-owned scratch
rotation.apply_inverse_with_scratch(&mut x, &mut scratch);
```

Use the `_with_scratch` variants in loops; the allocating ones are for
one-off calls and tests.

## Construction

`Rotation::new` draws three rounds from a `ChaCha8Rng` seeded with `seed`.
Each round is:

1. random sign flips (one bit per coordinate),
2. a random permutation (Fisher-Yates, with an unbiased `uniform_below` on
   the raw `u32` stream),
3. a block-diagonal Hadamard transform, each block normalized by `1/√n`.

Blocks are the power-of-two decomposition of `d`, so any dimension works:
`768 → [512, 256]`, `100 → [64, 32, 4]`, `769 → [512, 256, 1]`. The
permutation mixes across blocks between rounds, which is why odd dimensions
still mix well (the tests check impulse spread at `d = 65, 100, 300, 769`).

The transform is orthogonal: it preserves norms and dot products to fp32
precision, and `apply_inverse` round-trips.

## Format stability

The seed and the exact ChaCha8 stream determine the rotation, so the stored
codes depend on `rand_chacha = "=0.3.1"` and `rand_core = "=0.6.4"` (pinned in
the workspace `Cargo.toml`). Two determinism snapshots
(`d = 768, seed = 42` and `d = 100, seed = 42`) compare the first eight
outputs bit-for-bit and will fail on any change to the RNG, the sampler, or
the round structure.

## Debug counter

Under `debug_assertions`, `debug_reset_apply_count()` and
`debug_apply_count()` expose a thread-local count of `apply*` calls. `cascade`
uses it to assert that a cluster's rotation runs once per layer, not once per
row.

## Tests and benches

Tests cover the block decomposition, the Hadamard against a naive matrix,
norm/dot preservation, inverse round trip, scratch-vs-allocating equality,
sampler uniformity, and the two snapshots.

`cargo bench -p fht --bench rotation` times `apply` at the common dimensions.
`check_asm_x86.sh` asserts the release build emits packed `addps`/`subps`
for the butterfly.

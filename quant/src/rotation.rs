//! Seeded, format-stable, block-diagonal randomized Hadamard rotation.

#[cfg(debug_assertions)]
use std::cell::Cell;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

const ROUNDS: usize = 3;

/// The IEEE-754 binary32 sign bit; XORing it into a float's bits negates the float.
const SIGN_BIT: u32 = 1 << 31;

#[cfg(debug_assertions)]
thread_local! {
    static APPLY_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug)]
struct Round {
    /// One XOR mask per coordinate: `SIGN_BIT` to negate it, zero to keep it.
    signs: Vec<u32>,
    /// `signs` in gather order, `perm_signs[t] = signs[perm[t]]`, so the forward round flips each
    /// sign while it gathers.
    perm_signs: Vec<u32>,
    perm: Vec<u32>,
    inv_perm: Vec<u32>,
}

#[derive(Clone, Debug)]
/// A seeded orthogonal transform.
pub struct Rotation {
    d: usize,
    rounds: Vec<Round>,
    blocks: Vec<usize>,
}

impl Rotation {
    /// Creates a rotation for `d` coordinates.
    ///
    /// # Panics
    ///
    /// Panics when `d` is zero or exceeds `u32::MAX`.
    pub fn new(d: usize, seed: u64) -> Self {
        assert!(d > 0);
        assert!(u32::try_from(d).is_ok());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut rounds = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let signs = sign_masks(&mut rng, d);
            let mut perm: Vec<u32> = (0..d as u32).collect();
            for i in (1..d).rev() {
                let j = uniform_below(&mut rng, (i + 1) as u32) as usize;
                perm.swap(i, j);
            }
            let mut inv_perm = vec![0; d];
            for (i, &source) in perm.iter().enumerate() {
                inv_perm[source as usize] = i as u32;
            }
            // The gathers index without bounds checks, so both tables must be permutations of 0..d.
            assert!(
                inv_perm
                    .iter()
                    .enumerate()
                    .all(|(i, &position)| perm[position as usize] as usize == i),
                "rotation permutation is not a bijection"
            );
            let perm_signs = perm.iter().map(|&source| signs[source as usize]).collect();
            rounds.push(Round {
                signs,
                perm_signs,
                perm,
                inv_perm,
            });
        }
        Self {
            d,
            rounds,
            blocks: blocks(d),
        }
    }

    /// Applies the rotation in place.
    ///
    /// # Panics
    ///
    /// Panics when `x` does not match the rotation dimension.
    pub fn apply(&self, x: &mut [f32]) {
        let mut scratch = vec![0.0; self.d];
        self.apply_with_scratch(x, &mut scratch);
    }

    /// Applies the rotation with caller-owned scratch storage.
    ///
    /// # Panics
    ///
    /// Panics when either slice does not match the rotation dimension.
    pub fn apply_with_scratch(&self, x: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(x.len(), self.d);
        assert_eq!(scratch.len(), self.d);
        #[cfg(debug_assertions)]
        APPLY_COUNT.with(|count| count.set(count.get() + 1));
        for round in &self.rounds {
            // Negating then permuting equals permuting with the signs in gather order.
            for ((target, &source), &sign) in
                scratch.iter_mut().zip(&round.perm).zip(&round.perm_signs)
            {
                // SAFETY: `Rotation::new` asserts `perm` is a permutation of `0..d`, and
                // `x.len() == d` is asserted above.
                let value = unsafe { *x.get_unchecked(source as usize) };
                *target = f32::from_bits(value.to_bits() ^ sign);
            }
            x.copy_from_slice(scratch);
            block_hadamard(x, &self.blocks);
        }
    }

    /// Applies the inverse rotation in place.
    ///
    /// # Panics
    ///
    /// Panics when `x` does not match the rotation dimension.
    pub fn apply_inverse(&self, x: &mut [f32]) {
        let mut scratch = vec![0.0; self.d];
        self.apply_inverse_with_scratch(x, &mut scratch);
    }

    /// Applies the inverse rotation with caller-owned scratch storage.
    ///
    /// # Panics
    ///
    /// Panics when either slice does not match the rotation dimension.
    pub fn apply_inverse_with_scratch(&self, x: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(x.len(), self.d);
        assert_eq!(scratch.len(), self.d);
        for round in self.rounds.iter().rev() {
            block_hadamard(x, &self.blocks);
            // Permuting back then negating flips each gathered coordinate by its own sign.
            for ((target, &source), &sign) in
                scratch.iter_mut().zip(&round.inv_perm).zip(&round.signs)
            {
                // SAFETY: `Rotation::new` asserts `inv_perm` is a permutation of `0..d`, and
                // `x.len() == d` is asserted above.
                let value = unsafe { *x.get_unchecked(source as usize) };
                *target = f32::from_bits(value.to_bits() ^ sign);
            }
            x.copy_from_slice(scratch);
        }
    }
}

/// Reset the current thread's debug-only `Rotation::apply` counter.
#[cfg(debug_assertions)]
pub fn debug_reset_apply_count() {
    APPLY_COUNT.with(|count| count.set(0));
}

/// Read the current thread's debug-only `Rotation::apply` counter.
#[cfg(debug_assertions)]
pub fn debug_apply_count() -> usize {
    APPLY_COUNT.with(Cell::get)
}

fn blocks(mut d: usize) -> Vec<usize> {
    let mut result = Vec::new();
    while d != 0 {
        let block = 1usize << (usize::BITS - 1 - d.leading_zeros());
        result.push(block);
        d -= block;
    }
    result
}

/// Unbiased uniform in `[0, n)` from the stable raw ChaCha stream.
fn uniform_below(rng: &mut ChaCha8Rng, n: u32) -> u32 {
    debug_assert!(n > 0);
    let zone = (u32::MAX / n) * n;
    loop {
        let x = rng.next_u32();
        if x < zone {
            return x % n;
        }
    }
}

/// Draws one sign bit per coordinate and expands each into an XOR mask.
///
/// The stream is consumed in whole `u64` words, bit `i % 64` of word `i / 64` deciding
/// coordinate `i`, so the stored codes do not depend on the in-memory mask layout.
fn sign_masks(rng: &mut ChaCha8Rng, d: usize) -> Vec<u32> {
    let mut masks = Vec::with_capacity(d);
    for _ in 0..d.div_ceil(64) {
        let word = rng.next_u64();
        let remaining = d - masks.len();
        masks.extend((0..remaining.min(64)).map(|bit| {
            if word & (1_u64 << bit) != 0 {
                SIGN_BIT
            } else {
                0
            }
        }));
    }
    masks
}

fn block_hadamard(x: &mut [f32], blocks: &[usize]) {
    let mut offset = 0;
    for &block_len in blocks {
        hadamard(&mut x[offset..offset + block_len]);
        offset += block_len;
    }
}

/// Normalized Hadamard transform of one power-of-two block, on the widest kernel the CPU has.
fn hadamard(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    #[cfg(target_arch = "x86_64")]
    if x.len() >= x86::MIN_LEN && crate::cpu::tier() == crate::cpu::Tier::Avx512 {
        // SAFETY: the tier is only reported when the CPU has avx512f.
        unsafe { x86::hadamard_avx512(x) };
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if x.len() >= neon::MIN_LEN {
        // SAFETY: this branch only compiles when NEON is enabled for the whole build.
        unsafe { neon::hadamard(x) };
        return;
    }
    hadamard_portable(x);
}

/// Portable transform, and the reference every SIMD kernel must match bit-for-bit: one butterfly
/// level per pass in ascending width, then one normalization multiply.
fn hadamard_portable(x: &mut [f32]) {
    debug_assert!(x.len().is_power_of_two());
    let mut width = 1;
    while width < x.len() {
        for chunk in x.chunks_exact_mut(width * 2) {
            let (left, right) = chunk.split_at_mut(width);
            for (a, b) in left.iter_mut().zip(right) {
                let left_value = *a;
                let right_value = *b;
                *a = left_value + right_value;
                *b = left_value - right_value;
            }
        }
        width *= 2;
    }
    let normalization = 1.0 / (x.len() as f32).sqrt();
    for value in x {
        *value *= normalization;
    }
}

/// Runtime-dispatched x86 Hadamard kernel.
///
/// Every coordinate goes through the same additions, subtractions and final multiply, in the same
/// level order, as [`hadamard_portable`], so the output is bit-identical and the stored codes do
/// not depend on the CPU.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::{
        __m512, _mm512_add_ps, _mm512_loadu_ps, _mm512_mask_sub_ps, _mm512_mul_ps,
        _mm512_permute_ps, _mm512_set1_ps, _mm512_shuffle_f32x4, _mm512_storeu_ps, _mm512_sub_ps,
    };

    /// Floats per 512-bit register.
    const LANES: usize = 16;

    /// Shortest block the kernel accepts.
    pub(super) const MIN_LEN: usize = LANES;

    /// # Safety
    ///
    /// The CPU must have AVX-512 F.
    ///
    /// # Panics
    ///
    /// Panics unless `x.len()` is a power of two of at least `MIN_LEN`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn hadamard_avx512(x: &mut [f32]) {
        assert!(x.len().is_power_of_two() && x.len() >= MIN_LEN);
        let len = x.len();
        let normalization = _mm512_set1_ps(1.0 / (len as f32).sqrt());
        let base = x.as_mut_ptr();

        // Widths 1, 2, 4 and 8 pair lanes of the same register.
        for i in (0..len).step_by(LANES) {
            // SAFETY: `len` is a multiple of `LANES`, so `i + LANES <= len`.
            unsafe {
                let lanes = base.add(i);
                _mm512_storeu_ps(lanes, lane_levels(_mm512_loadu_ps(lanes)));
            }
        }

        // Wider levels pair whole registers, two levels per pass while two remain.
        let mut width = LANES;
        while width * 2 < len {
            for chunk in (0..len).step_by(width * 4) {
                for i in (chunk..chunk + width).step_by(LANES) {
                    // SAFETY: `width` is a multiple of `LANES` and `chunk + 4 * width <= len`, so
                    // all four registers lie inside the slice and do not overlap.
                    unsafe {
                        let a = _mm512_loadu_ps(base.add(i));
                        let b = _mm512_loadu_ps(base.add(i + width));
                        let c = _mm512_loadu_ps(base.add(i + 2 * width));
                        let d = _mm512_loadu_ps(base.add(i + 3 * width));
                        let (ab_sum, ab_diff) = (_mm512_add_ps(a, b), _mm512_sub_ps(a, b));
                        let (cd_sum, cd_diff) = (_mm512_add_ps(c, d), _mm512_sub_ps(c, d));
                        _mm512_storeu_ps(base.add(i), _mm512_add_ps(ab_sum, cd_sum));
                        _mm512_storeu_ps(base.add(i + width), _mm512_add_ps(ab_diff, cd_diff));
                        _mm512_storeu_ps(base.add(i + 2 * width), _mm512_sub_ps(ab_sum, cd_sum));
                        _mm512_storeu_ps(base.add(i + 3 * width), _mm512_sub_ps(ab_diff, cd_diff));
                    }
                }
            }
            width *= 4;
        }
        if width < len {
            debug_assert_eq!(width * 2, len);
            for i in (0..width).step_by(LANES) {
                // SAFETY: `i + LANES <= width` and `2 * width == len`.
                unsafe {
                    let a = _mm512_loadu_ps(base.add(i));
                    let b = _mm512_loadu_ps(base.add(i + width));
                    _mm512_storeu_ps(base.add(i), _mm512_add_ps(a, b));
                    _mm512_storeu_ps(base.add(i + width), _mm512_sub_ps(a, b));
                }
            }
        }

        for i in (0..len).step_by(LANES) {
            // SAFETY: `len` is a multiple of `LANES`, so `i + LANES <= len`.
            unsafe {
                let lanes = base.add(i);
                _mm512_storeu_ps(lanes, _mm512_mul_ps(_mm512_loadu_ps(lanes), normalization));
            }
        }
    }

    /// Runs the width 1, 2, 4 and 8 butterfly levels inside one register.
    ///
    /// Each level swaps the two halves of every pair, adds to get `left + right` in the lower
    /// half, and overwrites the upper half with `left - right`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn lane_levels(v: __m512) -> __m512 {
        let swapped = _mm512_permute_ps::<0b10_11_00_01>(v);
        let v = _mm512_mask_sub_ps(_mm512_add_ps(v, swapped), 0xAAAA, swapped, v);
        let swapped = _mm512_permute_ps::<0b01_00_11_10>(v);
        let v = _mm512_mask_sub_ps(_mm512_add_ps(v, swapped), 0xCCCC, swapped, v);
        let swapped = _mm512_shuffle_f32x4::<0b10_11_00_01>(v, v);
        let v = _mm512_mask_sub_ps(_mm512_add_ps(v, swapped), 0xF0F0, swapped, v);
        let swapped = _mm512_shuffle_f32x4::<0b01_00_11_10>(v, v);
        _mm512_mask_sub_ps(_mm512_add_ps(v, swapped), 0xFF00, swapped, v)
    }
}

/// NEON Hadamard kernel, bit-identical to [`hadamard_portable`].
///
/// The levels run in the portable order. Inside a register a butterfly is computed as
/// `[right, left] + [left, -right]`, which is `left + right` and `left - right` because IEEE-754
/// subtraction is addition of the negated operand.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use std::arch::aarch64::{
        float32x4_t, uint32x4_t, vaddq_f32, vdupq_n_f32, veorq_u32, vextq_f32, vld1q_f32,
        vld1q_u32, vmulq_f32, vreinterpretq_f32_u32, vreinterpretq_u32_f32, vrev64q_f32, vst1q_f32,
        vsubq_f32,
    };

    use super::SIGN_BIT;

    /// Floats per 128-bit register.
    const LANES: usize = 4;

    /// Shortest block the kernel accepts: the first pass transforms four registers at a time.
    pub(super) const MIN_LEN: usize = 4 * LANES;

    /// Sign masks selecting the `right` lanes of the width 1 and width 2 butterflies.
    const ODD_LANES: [u32; LANES] = [0, SIGN_BIT, 0, SIGN_BIT];
    const UPPER_LANES: [u32; LANES] = [0, 0, SIGN_BIT, SIGN_BIT];

    /// # Panics
    ///
    /// Panics unless `x.len()` is a power of two of at least `MIN_LEN`.
    #[target_feature(enable = "neon")]
    pub(super) fn hadamard(x: &mut [f32]) {
        assert!(x.len().is_power_of_two() && x.len() >= MIN_LEN);
        let len = x.len();
        let normalization = vdupq_n_f32(1.0 / (len as f32).sqrt());
        let base = x.as_mut_ptr();
        // SAFETY: both arrays hold exactly one register of lanes.
        let (odd, upper) = unsafe {
            (
                vld1q_u32(ODD_LANES.as_ptr()),
                vld1q_u32(UPPER_LANES.as_ptr()),
            )
        };

        // Widths 1 and 2 pair lanes of the same register; widths 4 and 8 pair the four registers
        // of one 16-float group.
        for i in (0..len).step_by(MIN_LEN) {
            // SAFETY: `len` is a multiple of `MIN_LEN`, so `i + 4 * LANES <= len`.
            unsafe {
                let group = base.add(i);
                let [a, b, c, d] = two_levels(
                    lane_levels(vld1q_f32(group), odd, upper),
                    lane_levels(vld1q_f32(group.add(LANES)), odd, upper),
                    lane_levels(vld1q_f32(group.add(2 * LANES)), odd, upper),
                    lane_levels(vld1q_f32(group.add(3 * LANES)), odd, upper),
                );
                vst1q_f32(group, a);
                vst1q_f32(group.add(LANES), b);
                vst1q_f32(group.add(2 * LANES), c);
                vst1q_f32(group.add(3 * LANES), d);
            }
        }

        // Wider levels pair whole registers, two levels per pass while two remain.
        let mut width = MIN_LEN;
        while width * 2 < len {
            for chunk in (0..len).step_by(width * 4) {
                for i in (chunk..chunk + width).step_by(LANES) {
                    // SAFETY: `width` is a multiple of `LANES` and `chunk + 4 * width <= len`, so
                    // all four registers lie inside the slice and do not overlap.
                    unsafe {
                        let [a, b, c, d] = two_levels(
                            vld1q_f32(base.add(i)),
                            vld1q_f32(base.add(i + width)),
                            vld1q_f32(base.add(i + 2 * width)),
                            vld1q_f32(base.add(i + 3 * width)),
                        );
                        vst1q_f32(base.add(i), a);
                        vst1q_f32(base.add(i + width), b);
                        vst1q_f32(base.add(i + 2 * width), c);
                        vst1q_f32(base.add(i + 3 * width), d);
                    }
                }
            }
            width *= 4;
        }
        if width < len {
            debug_assert_eq!(width * 2, len);
            for i in (0..width).step_by(LANES) {
                // SAFETY: `i + LANES <= width` and `2 * width == len`.
                unsafe {
                    let a = vld1q_f32(base.add(i));
                    let b = vld1q_f32(base.add(i + width));
                    vst1q_f32(base.add(i), vaddq_f32(a, b));
                    vst1q_f32(base.add(i + width), vsubq_f32(a, b));
                }
            }
        }

        for i in (0..len).step_by(LANES) {
            // SAFETY: `len` is a multiple of `LANES`, so `i + LANES <= len`.
            unsafe {
                let lanes = base.add(i);
                vst1q_f32(lanes, vmulq_f32(vld1q_f32(lanes), normalization));
            }
        }
    }

    /// Runs the width 1 and width 2 butterfly levels inside one register.
    #[inline]
    #[target_feature(enable = "neon")]
    fn lane_levels(v: float32x4_t, odd: uint32x4_t, upper: uint32x4_t) -> float32x4_t {
        let v = vaddq_f32(vrev64q_f32(v), negate(v, odd));
        vaddq_f32(vextq_f32::<2>(v, v), negate(v, upper))
    }

    /// Flips the sign of the lanes whose mask holds `SIGN_BIT`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn negate(v: float32x4_t, mask: uint32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), mask))
    }

    /// Runs two consecutive butterfly levels over four registers spaced one width apart.
    #[inline]
    #[target_feature(enable = "neon")]
    fn two_levels(
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        d: float32x4_t,
    ) -> [float32x4_t; 4] {
        let (ab_sum, ab_diff) = (vaddq_f32(a, b), vsubq_f32(a, b));
        let (cd_sum, cd_diff) = (vaddq_f32(c, d), vsubq_f32(c, d));
        [
            vaddq_f32(ab_sum, cd_sum),
            vaddq_f32(ab_diff, cd_diff),
            vsubq_f32(ab_sum, cd_sum),
            vsubq_f32(ab_diff, cd_diff),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    fn input(d: usize, phase: f32) -> Vec<f32> {
        (0..d)
            .map(|i| ((i as f32 + 1.0) * 0.017 + phase).sin())
            .collect()
    }

    #[test]
    fn block_decomposition() {
        assert_eq!(blocks(768), [512, 256]);
        assert_eq!(blocks(1536), [1024, 512]);
        assert_eq!(blocks(128), [128]);
        assert_eq!(blocks(1024), [1024]);
        assert_eq!(blocks(65), [64, 1]);
        assert_eq!(blocks(100), [64, 32, 4]);
        assert_eq!(blocks(300), [256, 32, 8, 4]);
        assert_eq!(blocks(769), [512, 256, 1]);
    }

    #[test]
    fn sampler_sanity() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut buckets = [0_u32; 6];
        for _ in 0..60_000 {
            buckets[uniform_below(&mut rng, 6) as usize] += 1;
        }
        for count in buckets {
            assert!((9_500..=10_500).contains(&count), "{buckets:?}");
        }
    }

    #[test]
    fn hadamard_matches_naive_matrix() {
        for len in [2, 4, 8, 16] {
            let original: Vec<f32> = (0..len).map(|i| i as f32 * 0.3 - 0.7).collect();
            let normalization = 1.0 / (len as f32).sqrt();
            let expected: Vec<f32> = (0..len)
                .map(|row| {
                    original
                        .iter()
                        .enumerate()
                        .map(|(column, &value)| {
                            let sign = if (row & column).count_ones() % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            sign * value * normalization
                        })
                        .sum()
                })
                .collect();
            let mut actual = original;
            hadamard(&mut actual);
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "len={len}: {actual} != {expected}"
                );
            }
        }
    }

    /// Checks a kernel against the portable transform at every power-of-two length it accepts.
    fn assert_bit_identical_to_portable(name: &str, min_len: usize, kernel: impl Fn(&mut [f32])) {
        let mut len = min_len;
        while len <= 4096 {
            // Signed zeros, a wide dynamic range and inexact sums exercise every rounding path.
            let original: Vec<f32> = (0..len)
                .map(|i| match i % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.0e-30 * (i as f32 + 1.0),
                    3 => -3.0e20 * (i as f32).cos(),
                    _ => ((i as f32 + 1.0) * 0.731).sin(),
                })
                .collect();
            let mut expected = original.clone();
            hadamard_portable(&mut expected);
            let mut actual = original;
            kernel(&mut actual);
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{name} len={len}, i={i}: {actual} != {expected}"
                );
            }
            len *= 2;
        }
    }

    #[test]
    fn simd_hadamard_is_bit_identical_to_portable() {
        assert_bit_identical_to_portable("dispatched", 1, hadamard);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f was just detected.
            assert_bit_identical_to_portable("avx512", x86::MIN_LEN, |x| unsafe {
                x86::hadamard_avx512(x)
            });
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        // SAFETY: this line only compiles when NEON is enabled for the whole build.
        assert_bit_identical_to_portable("neon", neon::MIN_LEN, |x| unsafe { neon::hadamard(x) });
    }

    #[test]
    fn preserves_norm_dot_and_round_trips() {
        for d in [65, 100, 128, 300, 768, 769, 1536] {
            let rotation = Rotation::new(d, 99);
            let original_x = input(d, 0.2);
            let original_y = input(d, 1.1);
            let mut x = original_x.clone();
            let mut y = original_y.clone();
            rotation.apply(&mut x);
            rotation.apply(&mut y);
            let norm_ratio = dot(&x, &x).sqrt() / dot(&original_x, &original_x).sqrt();
            assert!((norm_ratio - 1.0).abs() < 1e-4, "d={d}: {norm_ratio}");
            let original_dot = dot(&original_x, &original_y);
            assert!((dot(&x, &y) - original_dot).abs() / original_dot.abs().max(1.0) < 1e-4);
            rotation.apply_inverse(&mut x);
            for (&actual, &expected) in x.iter().zip(&original_x) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "d={d}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn caller_scratch_matches_allocating_wrappers() {
        for d in [65, 100, 768, 769] {
            let rotation = Rotation::new(d, 99);
            let original = input(d, 0.73);
            let mut allocating = original.clone();
            rotation.apply(&mut allocating);

            let mut reused = original.clone();
            let mut scratch = vec![f32::NAN; d];
            rotation.apply_with_scratch(&mut reused, &mut scratch);
            assert_eq!(reused, allocating, "forward mismatch at d={d}");

            rotation.apply_inverse_with_scratch(&mut reused, &mut scratch);
            let mut expected = allocating;
            rotation.apply_inverse(&mut expected);
            assert_eq!(reused, expected, "inverse mismatch at d={d}");
        }
    }

    #[test]
    fn determinism_snapshot_and_mixing() {
        let mut x = vec![0.0; 768];
        x[0] = 1.0;
        Rotation::new(768, 42).apply(&mut x);
        let expected = [
            0.008_543_869_f32,
            0.014_860_673,
            -0.086_899_504,
            0.033_396_773,
            -0.006_631_759,
            -0.058_936_73,
            0.015_103_942,
            -0.017_101_934,
        ];
        for (&actual, &expected) in x.iter().take(8).zip(&expected) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "snapshot actual={:?}",
                &x[..8]
            );
        }
        assert!(x.iter().copied().map(f32::abs).fold(0.0, f32::max) < 0.2);
    }

    #[test]
    fn odd_dimension_determinism_snapshot_and_mixing() {
        let mut first = vec![0.0; 100];
        first[0] = 1.0;
        Rotation::new(100, 42).apply(&mut first);
        let expected = [
            -0.241_009_7_f32,
            0.139_677_18,
            0.124_607_4,
            0.120_030_954,
            0.105_746_36,
            -0.105_746_36,
            0.091_461_78,
            -0.016_017_599,
        ];
        for (&actual, &expected) in first.iter().take(8).zip(&expected) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "snapshot actual={:?}",
                &first[..8]
            );
        }
        let mut second = vec![0.0; 100];
        second[0] = 1.0;
        Rotation::new(100, 42).apply(&mut second);
        assert_eq!(first, second);
        assert!(
            first.iter().filter(|&&value| value != 0.0).count() >= 90,
            "insufficient mixing: {first:?}"
        );
        assert!(
            first.iter().copied().map(f32::abs).fold(0.0, f32::max) < 0.4,
            "insufficient mixing: {first:?}"
        );
    }

    #[test]
    fn odd_dimensions_mix_impulses() {
        for d in [65, 100, 300, 769] {
            let mut values = vec![0.0; d];
            values[d - 1] = 1.0;
            Rotation::new(d, 0x4d49_5849_4e47).apply(&mut values);
            let support = values.iter().filter(|&&value| value != 0.0).count();
            let max = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
            assert!(support >= d * 3 / 4, "d={d}: support={support}");
            assert!(max < 0.5, "d={d}: max={max}");
        }
    }
}

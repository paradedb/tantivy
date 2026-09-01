//! Seeded, format-stable, block-diagonal randomized Hadamard rotation.

#[cfg(debug_assertions)]
use std::cell::Cell;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

const ROUNDS: usize = 3;

#[cfg(debug_assertions)]
thread_local! {
    static APPLY_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug)]
struct Round {
    signs: Vec<u64>,
    perm: Vec<u32>,
    inv_perm: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Rotation {
    d: usize,
    rounds: Vec<Round>,
    blocks: Vec<usize>,
}

impl Rotation {
    pub fn new(d: usize, seed: u64) -> Self {
        assert!(d > 0 && d.is_multiple_of(64));
        assert!(u32::try_from(d).is_ok());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut rounds = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let signs = (0..d.div_ceil(64)).map(|_| rng.next_u64()).collect();
            let mut perm: Vec<u32> = (0..d as u32).collect();
            for i in (1..d).rev() {
                let j = uniform_below(&mut rng, (i + 1) as u32) as usize;
                perm.swap(i, j);
            }
            let mut inv_perm = vec![0; d];
            for (i, &source) in perm.iter().enumerate() {
                inv_perm[source as usize] = i as u32;
            }
            rounds.push(Round {
                signs,
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

    pub fn apply(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.d);
        assert!(x.len().is_multiple_of(64));
        #[cfg(debug_assertions)]
        APPLY_COUNT.with(|count| count.set(count.get() + 1));
        let mut scratch = vec![0.0; self.d];
        for round in &self.rounds {
            apply_signs(x, &round.signs);
            for (target, &source) in round.perm.iter().enumerate() {
                scratch[target] = x[source as usize];
            }
            x.copy_from_slice(&scratch);
            block_hadamard(x, &self.blocks);
        }
    }

    pub fn apply_inverse(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.d);
        assert!(x.len().is_multiple_of(64));
        let mut scratch = vec![0.0; self.d];
        for round in self.rounds.iter().rev() {
            block_hadamard(x, &self.blocks);
            for (target, &source) in round.inv_perm.iter().enumerate() {
                scratch[target] = x[source as usize];
            }
            x.copy_from_slice(&scratch);
            apply_signs(x, &round.signs);
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

fn apply_signs(x: &mut [f32], signs: &[u64]) {
    for (i, value) in x.iter_mut().enumerate() {
        if signs[i / 64] & (1_u64 << (i % 64)) != 0 {
            *value = -*value;
        }
    }
}

fn block_hadamard(x: &mut [f32], blocks: &[usize]) {
    let mut offset = 0;
    for &block_len in blocks {
        hadamard(&mut x[offset..offset + block_len]);
        offset += block_len;
    }
}

fn hadamard(x: &mut [f32]) {
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

    #[test]
    fn preserves_norm_dot_and_round_trips() {
        for d in [128, 768, 1536] {
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
}

//! Numerical oracle for the exact sphere-marginal scalar quantizers.

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

pub mod f16;

/// A symmetric Lloyd-Max grid for one coordinate of a point on a `d`-sphere.
#[derive(Clone, Debug)]
pub struct Grid {
    pub bits: u8,
    pub points: Vec<f32>,
    pub rho_model: f64,
}

const QUADRATURE_POINTS: usize = 1 << 16;

/// Settled Phase-A calibration after the N=1000 three-dimension measurement.
pub const DEFAULT_CAL: f64 = 1.0;

fn sphere_samples(d: usize) -> Vec<(f64, f64)> {
    assert!(d >= 64);
    let limit = (d as f64).sqrt();
    let step = 2.0 * limit / QUADRATURE_POINTS as f64;
    let exponent = (d as f64 - 3.0) * 0.5;
    (0..QUADRATURE_POINTS)
        .map(|i| {
            let z = -limit + (i as f64 + 0.5) * step;
            let base = (1.0 - z * z / d as f64).max(0.0);
            (z, base.powf(exponent))
        })
        .collect()
}

fn rho_from_samples(samples: &[(f64, f64)], points: &[f64]) -> f64 {
    let boundaries: Vec<f64> = points.windows(2).map(|p| (p[0] + p[1]) * 0.5).collect();
    let mut error = 0.0;
    let mut energy = 0.0;
    for &(z, weight) in samples {
        let bucket = boundaries.partition_point(|&boundary| z > boundary);
        error += weight * (z - points[bucket]).powi(2);
        energy += weight * z * z;
    }
    (error / energy).sqrt()
}

/// Evaluate the exact-density normalized RMSE of persisted reconstruction
/// points without rerunning the Lloyd-Max solver.
pub fn rho_model_for_points(d: usize, points: &[f32]) -> f64 {
    assert!(d >= 64);
    assert!(points.len() >= 2 && points.len().is_power_of_two());
    let samples = sphere_samples(d);
    let points: Vec<f64> = points.iter().map(|&point| f64::from(point)).collect();
    rho_from_samples(&samples, &points)
}

/// Exact-density sign grid used only to open legacy metadata that predates
/// persisted model rho. This is one centroid moment plus one error integral,
/// not a Lloyd-Max solve.
pub fn exact_sign_grid(d: usize) -> Grid {
    let samples = sphere_samples(d);
    let (mass, moment) = samples
        .iter()
        .fold((0.0, 0.0), |(mass, moment), &(z, weight)| {
            (mass + weight, moment + weight * z.abs())
        });
    let magnitude = moment / mass;
    let points = vec![-magnitude, magnitude];
    Grid {
        bits: 1,
        points: points.iter().map(|&point| point as f32).collect(),
        rho_model: rho_from_samples(&samples, &points),
    }
}

/// Build an exact-density Lloyd-Max grid for a dimension-normalized sphere marginal.
pub fn build_grid(d: usize, bits: u8) -> Grid {
    assert!(d >= 64);
    assert!((1..=8).contains(&bits));

    let count = 1usize << bits;
    let samples = sphere_samples(d);

    let spread = if count == 2 { 0.8 } else { 3.0 };
    let mut points: Vec<f64> = (0..count)
        .map(|i| -spread + 2.0 * spread * i as f64 / (count - 1) as f64)
        .collect();

    for _ in 0..200 {
        let boundaries: Vec<f64> = points.windows(2).map(|p| (p[0] + p[1]) * 0.5).collect();
        let mut masses = vec![0.0; count];
        let mut moments = vec![0.0; count];
        for &(z, weight) in &samples {
            let bucket = boundaries.partition_point(|&boundary| z > boundary);
            masses[bucket] += weight;
            moments[bucket] += weight * z;
        }

        let mut max_change = 0.0_f64;
        for i in 0..count {
            if masses[i] != 0.0 {
                let next = moments[i] / masses[i];
                max_change = max_change.max((next - points[i]).abs());
                points[i] = next;
            }
        }
        // Preserve exact antisymmetry and keep format generation deterministic.
        for i in 0..count / 2 {
            let magnitude = (points[count - 1 - i] - points[i]) * 0.5;
            points[i] = -magnitude;
            points[count - 1 - i] = magnitude;
        }
        if max_change < 1e-12 {
            break;
        }
    }

    let rho_model = rho_from_samples(&samples, &points);

    Grid {
        bits,
        points: points.into_iter().map(|point| point as f32).collect(),
        rho_model,
    }
}

/// One-sided standard-normal tail probability for an interval miss.
pub fn kappa_miss(kappa: f64) -> f64 {
    assert!(kappa.is_finite() && kappa >= 0.0);
    0.5 * erfc(kappa / 2.0_f64.sqrt())
}

// Numerical Recipes approximation, with maximum absolute error below 1.2e-7.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

pub fn sigma_from_rho(rho: f64, d: usize, cal: f64) -> f64 {
    assert!(rho >= 0.0 && d > 0 && cal >= 0.0);
    cal * rho / (d as f64).sqrt()
}

pub fn empirical_sigma(estimates: &[f32], truths: &[f32]) -> f64 {
    assert_eq!(estimates.len(), truths.len());
    assert!(!estimates.is_empty());
    let mse = estimates
        .iter()
        .zip(truths)
        .map(|(&estimate, &truth)| f64::from(estimate - truth).powi(2))
        .sum::<f64>()
        / estimates.len() as f64;
    mse.sqrt()
}

/// Empirically compare dot-product error to the isotropic product model.
pub fn measure_cal(d: usize, bits: u8, n: usize) -> f64 {
    assert!(d > 0);
    assert!(n > 0);
    let grid = build_grid(d, bits);
    let mut rng = ChaCha8Rng::seed_from_u64(0x4341_4c00_0000_0000 ^ d as u64 ^ bits as u64);
    let query = random_unit(&mut rng, d);
    let mut estimates = Vec::with_capacity(n);
    let mut truths = Vec::with_capacity(n);
    for _ in 0..n {
        let vector = random_unit(&mut rng, d);
        let scale_bits = f16::f32_to_f16(1.0 / (d as f32).sqrt());
        let scale = f16::f16_to_f32(scale_bits);
        let estimate = vector
            .iter()
            .zip(&query)
            .map(|(&value, &q)| {
                let code = nearest(value / scale, &grid.points);
                q * scale * grid.points[code]
            })
            .sum();
        estimates.push(estimate);
        truths.push(dot(&query, &vector));
    }
    empirical_sigma(&estimates, &truths) / sigma_from_rho(grid.rho_model, d, 1.0)
}

fn nearest(value: f32, grid: &[f32]) -> usize {
    grid.windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .position(|boundary| value <= boundary)
        .unwrap_or(grid.len() - 1)
}

fn random_unit(rng: &mut ChaCha8Rng, d: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(d);
    while values.len() < d {
        let u1 = ((rng.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0)).max(f64::MIN_POSITIVE);
        let u2 = (rng.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        values.push((radius * angle.cos()) as f32);
        if values.len() < d {
            values.push((radius * angle.sin()) as f32);
        }
    }
    let norm = values
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_table() {
        let expected = [
            (128, 1, 0.6007, 0.010),
            (128, 2, 0.3406, 0.010),
            (128, 4, 0.0965, 0.004),
            (768, 1, 0.6025, 0.010),
            (768, 2, 0.3424, 0.010),
            (768, 4, 0.0973, 0.004),
            (1536, 1, 0.6026, 0.010),
            (1536, 2, 0.3426, 0.010),
            (1536, 4, 0.0974, 0.004),
        ];
        for (d, bits, target, tolerance) in expected {
            let grid = build_grid(d, bits);
            let actual = measured_reference_rho(d, &grid, 1_000);
            assert!(
                (actual - target).abs() <= tolerance,
                "d={d} b={bits}: {actual}"
            );
        }
    }

    #[test]
    fn sign_convention_anchor() {
        let actual = measured_reference_rho(1536, &build_grid(1536, 1), 1_000);
        let gaussian_limit = (1.0 - 2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (actual - gaussian_limit).abs() <= 0.01,
            "{actual} != {gaussian_limit}"
        );
        assert!(
            actual > 0.5,
            "rho appears to be an MSE ratio rather than normalized RMSE"
        );
    }

    #[test]
    fn kappa_table() {
        for (kappa, expected) in [(2.0, 2.275e-2), (3.0, 1.350e-3), (4.0, 3.167e-5)] {
            assert!((kappa_miss(kappa) - expected).abs() < 2e-6);
        }
    }

    fn measured_reference_rho(d: usize, grid: &Grid, n: usize) -> f64 {
        let mut rng =
            ChaCha8Rng::seed_from_u64(0x5248_4f00_0000_0000 ^ d as u64 ^ grid.bits as u64);
        let mut error = 0.0_f64;
        let mut energy = 0.0_f64;
        for _ in 0..n {
            let vector = random_unit(&mut rng, d);
            let raw_scale = vector
                .iter()
                .map(|&value| value * value)
                .sum::<f32>()
                .sqrt()
                / (d as f32).sqrt();
            let scale = f16::f16_to_f32(f16::f32_to_f16(raw_scale));
            for &value in &vector {
                let code = nearest(value / raw_scale, &grid.points);
                let reconstructed = scale * grid.points[code];
                error += f64::from(value - reconstructed).powi(2);
                energy += f64::from(value).powi(2);
            }
        }
        (error / energy).sqrt()
    }
}

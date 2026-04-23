//! Beta-distribution Lloyd-Max scalar codebooks for TurboQuant.
//!
//! In high-dimensional uniform-on-the-sphere data, after a Haar-random
//! rotation each coordinate's marginal distribution converges to a Beta
//! distribution (and to N(0, 1/d) for d > 50). This module precomputes
//! Lloyd-Max-optimal scalar quantization codebooks for that marginal,
//! which is the data-oblivious core of TurboQuant.
//!
//! The codebook stores `2^bit_width` reconstruction centroids and the
//! decision boundaries between them. Quantization is a binary search on
//! the boundaries; dequantization is a direct lookup.
//!
//! Generation runs in `f64` for numerical stability of the Lloyd-Max
//! iterations (numerical integration via Simpson's rule, log-gamma
//! normalization of the Beta PDF). The final centroids and boundaries
//! are stored as `f32` to match the rest of the vector pipeline.
//!
//! Adapted (and reduced to f32) from
//! https://github.com/abdelstark/turboquant (MIT).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A scalar quantization codebook optimized for the Beta marginal of unit-sphere
/// vectors at a given dimensionality.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// The 2^bit_width reconstruction centroids, sorted ascending.
    pub centroids: Vec<f32>,
    /// The 2^bit_width - 1 decision boundaries between adjacent centroids.
    pub boundaries: Vec<f32>,
    /// Bits per coordinate.
    pub bit_width: u8,
}

impl Codebook {
    pub fn num_levels(&self) -> usize {
        self.centroids.len()
    }

    /// Find the codebook index for a value via binary search on boundaries.
    /// NaN inputs map to index 0.
    #[inline]
    pub fn quantize_scalar(&self, value: f32) -> u8 {
        if value.is_nan() {
            return 0;
        }
        match self
            .boundaries
            .binary_search_by(|b| b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i as u8,
            Err(i) => i.min(self.centroids.len() - 1) as u8,
        }
    }

    #[inline]
    pub fn dequantize_scalar(&self, index: u8) -> f32 {
        self.centroids[index as usize]
    }
}

/// Generate a Lloyd-Max codebook for `dim`-dimensional unit-sphere coordinates
/// at `bit_width` bits, running `iterations` of the Lloyd-Max loop.
///
/// Lloyd-Max alternates:
///   1. Update boundaries as midpoints of adjacent centroids.
///   2. Update centroids as conditional means E[X | b_{k-1} ≤ X < b_k] using numerical integration
///      of the Beta PDF.
pub fn generate_codebook(dim: usize, bit_width: u8, iterations: usize) -> Codebook {
    assert!((1..=8).contains(&bit_width), "bit_width must be 1..=8");
    assert!(dim > 0, "dim must be > 0");

    let k = 1usize << bit_width;

    // Initial centroids: quantile spacing of the marginal.
    let mut centroids: Vec<f64> = (0..k)
        .map(|i| {
            let u = (i as f64 + 0.5) / k as f64;
            sample_beta_marginal(dim, u)
        })
        .collect();
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for _ in 0..iterations {
        let boundaries: Vec<f64> = centroids.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();

        let new_centroids = compute_centroids(&centroids, &boundaries, dim);

        let converged = centroids
            .iter()
            .zip(new_centroids.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12);

        centroids = new_centroids;
        if converged {
            break;
        }
    }

    let boundaries: Vec<f64> = centroids.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();

    Codebook {
        centroids: centroids.into_iter().map(|c| c as f32).collect(),
        boundaries: boundaries.into_iter().map(|b| b as f32).collect(),
        bit_width,
    }
}

/// Process-global cache: codebooks are deterministic functions of (dim, bit_width).
///
/// Generation runs ~1ms for d=768, b=3 (50 iterations), but for typical workloads
/// we want it free after the first encode. Multiple indices on the same machine
/// with the same (dim, bit_width) share the cached codebook.
pub fn get_or_generate_cached(dim: usize, bit_width: u8) -> Codebook {
    static CACHE: OnceLock<Mutex<HashMap<(usize, u8), Codebook>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("turboquant codebook cache poisoned");
    guard
        .entry((dim, bit_width))
        .or_insert_with(|| generate_codebook(dim, bit_width, 100))
        .clone()
}

/// Conditional means E[X | b_{k-1} ≤ X < b_k] over each Voronoi region.
fn compute_centroids(old_centroids: &[f64], boundaries: &[f64], dim: usize) -> Vec<f64> {
    let k = old_centroids.len();
    let n_points = 200usize;
    let mut new_centroids = Vec::with_capacity(k);

    for i in 0..k {
        let lo = if i == 0 { -0.9999 } else { boundaries[i - 1] };
        let hi = if i == k - 1 { 0.9999 } else { boundaries[i] };

        if hi <= lo {
            new_centroids.push(old_centroids[i]);
            continue;
        }

        let (num, den) = simpson_integrate(lo, hi, n_points, |x| {
            let pdf = beta_pdf(x, dim);
            (x * pdf, pdf)
        });

        if den.abs() < 1e-15 {
            new_centroids.push(old_centroids[i]);
        } else {
            new_centroids.push(num / den);
        }
    }

    new_centroids
}

/// Joint Simpson's-rule integration returning (∫f, ∫g) in one pass.
fn simpson_integrate<F>(a: f64, b: f64, n: usize, f: F) -> (f64, f64)
where F: Fn(f64) -> (f64, f64) {
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let (f0, g0) = f(a);
    let (fn_, gn) = f(b);
    let mut sum_f = f0 + fn_;
    let mut sum_g = g0 + gn;
    for i in 1..n {
        let x = a + i as f64 * h;
        let (fi, gi) = f(x);
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum_f += w * fi;
        sum_g += w * gi;
    }
    (sum_f * h / 3.0, sum_g * h / 3.0)
}

/// Beta marginal PDF for one coordinate of a uniform unit-sphere vector
/// in d dimensions. Switches to the N(0, 1/d) Gaussian approximation for
/// d > 50, where it is accurate to <1% near the bulk.
fn beta_pdf(x: f64, dim: usize) -> f64 {
    if dim < 2 {
        return 0.0;
    }
    if dim > 50 {
        let sigma2 = 1.0 / dim as f64;
        return (-x * x / (2.0 * sigma2)).exp()
            / (sigma2.sqrt() * (2.0 * std::f64::consts::PI).sqrt());
    }
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let exponent = (dim as f64 - 3.0) / 2.0;
    let unnorm = (1.0 - x * x).powf(exponent);
    let log_c = lgamma(dim as f64 / 2.0)
        - 0.5 * std::f64::consts::PI.ln()
        - lgamma((dim as f64 - 1.0) / 2.0);
    log_c.exp() * unnorm
}

/// Lanczos g=7 log-gamma. Used only by `beta_pdf` for d ≤ 50.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    #[allow(clippy::excessive_precision)]
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let g = 7.0_f64;
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = c[0];
    let t = x + g + 0.5;
    for (i, &ci) in c[1..].iter().enumerate() {
        a += ci / (x + i as f64 + 1.0);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + a.ln() + (x + 0.5) * t.ln() - t
}

/// Inverse CDF of the marginal: returns x such that P(X ≤ x) = u.
/// Uses N(0, 1/d) ICDF for d > 50, bisection on the numerical CDF for smaller d.
fn sample_beta_marginal(dim: usize, u: f64) -> f64 {
    if dim > 50 {
        return (1.0 / dim as f64).sqrt() * normal_icdf(u);
    }
    let mut lo = -1.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if numerical_cdf(mid, dim) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Standard normal inverse CDF via the Beasley-Springer-Moro rational
/// approximation. Used by `sample_beta_marginal` in the d > 50 branch.
#[allow(clippy::excessive_precision)]
fn normal_icdf(p: f64) -> f64 {
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

fn numerical_cdf(x: f64, dim: usize) -> f64 {
    let n = 200usize;
    let a = -0.9999_f64;
    let b = x.min(0.9999);
    if b <= a {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut sum = beta_pdf(a, dim) + beta_pdf(b, dim);
    for i in 1..n {
        let xi = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * beta_pdf(xi, dim);
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_b1_is_symmetric() {
        let cb = generate_codebook(128, 1, 50);
        assert_eq!(cb.centroids.len(), 2);
        assert!((cb.centroids[0] + cb.centroids[1]).abs() < 1e-5);
        assert!(cb.boundaries[0].abs() < 1e-5);
    }

    #[test]
    fn codebook_b3_centroids_sorted_and_in_range() {
        let cb = generate_codebook(768, 3, 100);
        assert_eq!(cb.centroids.len(), 8);
        for w in cb.centroids.windows(2) {
            assert!(w[0] < w[1]);
        }
        for &c in &cb.centroids {
            assert!(c.abs() < 0.5, "centroid {c} too far from 0 for d=768");
        }
    }

    #[test]
    fn boundaries_lie_between_centroids() {
        let cb = generate_codebook(768, 3, 100);
        for (i, &b) in cb.boundaries.iter().enumerate() {
            assert!(b > cb.centroids[i] && b < cb.centroids[i + 1]);
        }
    }

    #[test]
    fn quantize_extremes() {
        let cb = generate_codebook(768, 3, 100);
        let last = cb.centroids.len() - 1;
        assert_eq!(cb.quantize_scalar(1.0e3) as usize, last);
        assert_eq!(cb.quantize_scalar(-1.0e3), 0);
    }

    #[test]
    fn quantize_dequantize_roundtrip_close() {
        let cb = generate_codebook(768, 3, 100);
        // For d=768 the marginal is N(0, 1/768) so values bigger than ~5σ
        // (≈ 0.18) live near the tails; restrict to the bulk.
        for &v in &[-0.10_f32, -0.03, 0.0, 0.03, 0.10] {
            let idx = cb.quantize_scalar(v);
            let recon = cb.dequantize_scalar(idx);
            assert!((v - recon).abs() < 0.10, "v={v} recon={recon} idx={idx}");
        }
    }

    #[test]
    fn cache_returns_consistent_codebook() {
        let a = get_or_generate_cached(768, 3);
        let b = get_or_generate_cached(768, 3);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.boundaries, b.boundaries);
    }
}

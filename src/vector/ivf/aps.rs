//! Adaptive Partition Scanning geometry: query-ball ∩ Voronoi half-space
//! recall estimates (Quake §5 / `geometry.h`).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use crate::schema::Metric;
use crate::vector::Similarity;

const BETA_TABLE_LEN: usize = 1024;
const BETA_STOP: f64 = 1.0e-8;
const BETA_TINY: f64 = 1.0e-30;

/// Regularized incomplete beta `I_x(a, b)` via Lentz continued fraction.
pub(crate) fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::INFINITY;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    }

    let lbeta_ab = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    let front = ((x.ln() * a + (1.0 - x).ln() * b - lbeta_ab).exp()) / a;

    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;
    for i in 0..=200 {
        let m = i / 2;
        let numerator = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            let m = m as f64;
            (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        } else {
            let m = m as f64;
            -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        };

        d = 1.0 + numerator * d;
        if d.abs() < BETA_TINY {
            d = BETA_TINY;
        }
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if c.abs() < BETA_TINY {
            c = BETA_TINY;
        }

        let cd = c * d;
        f *= cd;
        if (1.0 - cd).abs() < BETA_STOP {
            return front * (f - 1.0);
        }
    }
    f64::INFINITY
}

fn beta_table(dim: usize) -> &'static [f64] {
    static TABLES: OnceLock<Mutex<HashMap<usize, &'static [f64]>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = tables.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(table) = guard.get(&dim) {
        return table;
    }
    let a = (dim as f64 + 1.0) / 2.0;
    let b = 0.5;
    let mut values = Vec::with_capacity(BETA_TABLE_LEN);
    for i in 0..BETA_TABLE_LEN {
        let x = i as f64 / (BETA_TABLE_LEN - 1) as f64;
        values.push(incomplete_beta(a, b, x));
    }
    let leaked: &'static [f64] = Box::leak(values.into_boxed_slice());
    guard.insert(dim, leaked);
    leaked
}

fn incomplete_beta_lookup(x: f64, dim: usize) -> f64 {
    let x = x.clamp(0.0, 1.0);
    let table = beta_table(dim);
    let scaled = x * (BETA_TABLE_LEN - 1) as f64;
    let idx = (scaled as usize).min(BETA_TABLE_LEN - 2);
    let dx = 1.0 / (BETA_TABLE_LEN - 1) as f64;
    let x1 = idx as f64 * dx;
    let y1 = table[idx];
    let y2 = table[idx + 1];
    y1 + (x - x1) * (y2 - y1) / dx
}

/// Distance from `query` to the perpendicular bisector of `c0` and `cj`.
pub(crate) fn boundary_distance(query: &[f32], c0: &[f32], cj: &[f32], euclidean: bool) -> f32 {
    debug_assert_eq!(query.len(), c0.len());
    debug_assert_eq!(c0.len(), cj.len());
    let dim = query.len();
    if euclidean {
        let mut v_norm_sq = 0.0f32;
        let mut cj_norm_sq = 0.0f32;
        let mut c0_norm_sq = 0.0f32;
        let mut dot_qv = 0.0f32;
        for i in 0..dim {
            let v = cj[i] - c0[i];
            v_norm_sq += v * v;
            cj_norm_sq += cj[i] * cj[i];
            c0_norm_sq += c0[i] * c0[i];
            dot_qv += query[i] * v;
        }
        let v_norm = v_norm_sq.sqrt();
        let b = 0.5 * (cj_norm_sq - c0_norm_sq);
        (dot_qv - b).abs() / (v_norm + 1e-12)
    } else {
        let mut v_norm_sq = 0.0f32;
        let mut dot_qv = 0.0f32;
        for i in 0..dim {
            let v = cj[i] - c0[i];
            v_norm_sq += v * v;
            dot_qv += query[i] * v;
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm == 0.0 {
            return 0.0;
        }
        let s = (dot_qv / v_norm).abs().clamp(0.0, 1.0);
        s.asin()
    }
}

/// Distances from `query` to each candidate's bisector with `centroids[0]`.
/// `out[0]` is 0.
pub(crate) fn compute_boundary_distances(
    query: &[f32],
    centroids: &[&[f32]],
    euclidean: bool,
) -> Vec<f32> {
    let mut dist = vec![0.0f32; centroids.len()];
    if centroids.is_empty() {
        return dist;
    }
    let c0 = centroids[0];
    for (j, cj) in centroids.iter().enumerate().skip(1) {
        dist[j] = boundary_distance(query, c0, cj, euclidean);
    }
    dist
}

pub(crate) fn hyperspherical_cap_volume(
    radius: f64,
    boundary_distance: f64,
    dim: usize,
    euclidean: bool,
) -> f64 {
    if euclidean {
        let boundary_distance = boundary_distance.max(0.0);
        if boundary_distance >= radius {
            return 0.0;
        }
        let ratio = boundary_distance / radius;
        let x = (1.0 - ratio * ratio).sqrt().clamp(0.0, 1.0);
        let i = incomplete_beta_lookup(x, dim);
        (0.5 * i).clamp(0.0, 0.5)
    } else {
        let theta_q = radius;
        let delta = boundary_distance;
        if delta >= theta_q {
            return 0.0;
        }
        if theta_q >= PI / 2.0 - delta {
            return 1.0;
        }
        let t = (delta.tan() / theta_q.tan()).clamp(0.0, 1.0);
        let alpha = t.acos();
        let x = (alpha.sin() * alpha.sin()).clamp(0.0, 1.0);
        let a = 0.5 * (dim as f64 - 1.0);
        let b = 0.5;
        0.5 * incomplete_beta(a, b, x)
    }
}

/// Per-list hit probabilities `p_i` for the candidate set (index 0 is `P0`).
pub(crate) fn compute_recall_profile(
    boundary_distances: &[f32],
    query_radius: f32,
    dim: usize,
    euclidean: bool,
) -> Vec<f32> {
    let m = boundary_distances.len();
    const EPS: f32 = 1e-9;

    if m <= 1 {
        return if m == 1 { vec![1.0] } else { Vec::new() };
    }

    let mut radius = query_radius;
    if !euclidean {
        radius = query_radius.clamp(-1.0, 1.0).acos();
    }

    if radius <= EPS {
        let mut p = vec![0.0f32; m];
        p[0] = 1.0;
        return p;
    }

    let mut raw_vols = vec![0.0f32; m];
    for j in 1..m {
        raw_vols[j] =
            hyperspherical_cap_volume(radius as f64, boundary_distances[j] as f64, dim, euclidean)
                as f32;
    }

    let s1: f32 = raw_vols.iter().skip(1).sum();
    let mut norm_vols = raw_vols;
    if s1 > EPS {
        for v in norm_vols.iter_mut().skip(1) {
            *v /= s1;
        }
    } else {
        for v in norm_vols.iter_mut().skip(1) {
            *v = 0.0;
        }
    }

    let mut p0 = 1.0f32;
    for v in norm_vols.iter().skip(1) {
        p0 *= 1.0 - *v;
    }
    p0 = p0.clamp(0.0, 1.0);

    let mut p_prime_sum = 0.0f32;
    for v in norm_vols.iter().skip(1) {
        p_prime_sum += v.max(0.0);
    }

    let mut probs = vec![0.0f32; m];
    probs[0] = p0;
    let target = (1.0 - p0).clamp(0.0, 1.0);
    if target > EPS && p_prime_sum > EPS {
        let scale = target / p_prime_sum;
        for k in 1..m {
            probs[k] = (norm_vols[k] * scale).max(0.0);
        }
        let current: f32 = probs.iter().skip(1).sum();
        if current > EPS {
            let final_scale = target / current;
            if final_scale.is_finite() {
                for p in probs.iter_mut().skip(1) {
                    *p = (*p * final_scale).max(0.0);
                }
            }
        }
    }

    let s: f32 = probs.iter().sum();
    if s > EPS {
        for p in &mut probs {
            *p /= s;
        }
    }
    probs
}

/// Query-ball radius `ρ` from the k-th similarity in the top-k heap.
pub(crate) fn radius_from_kth(kth: Similarity, metric: Metric) -> f32 {
    match metric {
        Metric::L2 => (-kth.score()).max(0.0).sqrt(),
        Metric::Cosine | Metric::Dot => kth.score(),
    }
}

pub(crate) fn is_euclidean(metric: Metric) -> bool {
    matches!(metric, Metric::L2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_beta_endpoints() {
        let a = (2.0 + 1.0) / 2.0;
        let b = 0.5;
        assert!((incomplete_beta(a, b, 0.0) - 0.0).abs() < 1e-9);
        assert!((incomplete_beta(a, b, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cap_volume_plane_through_center_is_half() {
        let vol = hyperspherical_cap_volume(1.0, 0.0, 2, true);
        assert!((vol - 0.5).abs() < 1e-6, "{vol}");
    }

    #[test]
    fn cap_volume_missed_plane_is_zero() {
        let vol = hyperspherical_cap_volume(1.0, 1.5, 8, true);
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn l2_bisector_distance_known_pair() {
        let q = [0.0f32, 0.0];
        let c0 = [0.0f32, 0.0];
        let cj = [2.0f32, 0.0];
        let d = boundary_distance(&q, &c0, &cj, true);
        assert!((d - 1.0).abs() < 1e-5, "{d}");
    }

    #[test]
    fn tiny_radius_puts_all_mass_on_p0() {
        let q = [0.0f32, 0.0];
        let c0 = [0.0f32, 0.0];
        let c1 = [2.0f32, 0.0];
        let bd = compute_boundary_distances(&q, &[&c0[..], &c1[..]], true);
        let p = compute_recall_profile(&bd, 1e-12, 2, true);
        assert!((p[0] - 1.0).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn recall_profile_sums_to_one() {
        let q = [0.1f32, 0.0];
        let c0 = [0.0f32, 0.0];
        let c1 = [2.0f32, 0.0];
        let c2 = [0.0f32, 2.0];
        let bd = compute_boundary_distances(&q, &[&c0[..], &c1[..], &c2[..]], true);
        let p = compute_recall_profile(&bd, 1.5, 2, true);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "{p:?} sum={sum}");
        assert!(p.len() == 3);
    }
}

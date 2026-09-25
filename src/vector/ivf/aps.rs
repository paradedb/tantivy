//! Adaptive Partition Scanning geometry: query-ball ∩ Voronoi half-space
//! recall estimates (Quake §5 / `geometry.h`).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::io;
use std::sync::{Mutex, OnceLock};

use crate::schema::Metric;
use crate::vector::Similarity;

/// The largest dimension that routes with APS. Dimensions above this route
/// with the fixed nprobe fractions even when a recall target is requested.
/// The recall profile rests on `I_x((d+1)/2, 1/2)`; at high dimensions the
/// continued fraction and cap volumes degenerate, and the estimate has not
/// been validated above this dimension.
pub const APS_MAX_DIM: usize = 128;

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
#[cfg(test)]
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

/// Relative change in `ρ` that triggers a recall profile recompute.
pub const APS_RECOMPUTE_THRESHOLD: f32 = 0.10;

/// Centroid rows of a ranked candidate set, fetched on demand.
pub(crate) trait CandidateRows {
    /// Appends candidate `rank`'s centroid row to `out`.
    fn append_row(&self, rank: usize, out: &mut Vec<f32>) -> io::Result<()>;
}

impl CandidateRows for Vec<&[f32]> {
    fn append_row(&self, rank: usize, out: &mut Vec<f32>) -> io::Result<()> {
        out.extend_from_slice(self[rank]);
        Ok(())
    }
}

/// Running APS recall estimate over a ranked candidate set.
///
/// Candidates are covered in rank order, so the covered set is always a
/// prefix. The recall profile depends on `ρ` (the k-th result's radius)
/// and is recomputed only when `ρ` moves by more than
/// [`APS_RECOMPUTE_THRESHOLD`]; otherwise each newly covered candidate adds
/// its cached probability.
///
/// Boundaries are computed lazily. Under L2, candidate `j`'s bisector with
/// the nearest centroid is at least `(δ_j - δ_0) / 2` from the query (`δ`
/// the query-to-centroid distance, by the triangle inequality), so once
/// `δ_j ≥ δ_0 + 2ρ` that candidate and every later one lie outside the
/// query ball and have zero cap volume. Only the candidates before that
/// cutoff are fetched, which keeps the profile identical to one over the
/// whole ranking. Other metrics fetch every candidate at the first estimate.
pub(crate) struct RecallEstimator<'a> {
    query: Vec<f32>,
    metric: Metric,
    rows: Box<dyn CandidateRows + 'a>,
    /// `δ_j` per candidate under L2, ascending; empty for other metrics.
    dists: Vec<f32>,
    len: usize,
    nearest: Vec<f32>,
    row: Vec<f32>,
    /// Boundary distances of the fetched candidate prefix.
    boundary: Vec<f32>,
    rho: Option<f32>,
    profile: Vec<f32>,
    covered: usize,
    estimate: f32,
}

impl<'a> RecallEstimator<'a> {
    /// `sims[i]` is candidate `i`'s similarity to `query`, nearest first,
    /// and `rows` serves its centroid.
    pub(crate) fn new(
        query: &[f32],
        sims: &[Similarity],
        rows: Box<dyn CandidateRows + 'a>,
        metric: Metric,
    ) -> Self {
        let dists = if is_euclidean(metric) {
            sims.iter()
                .map(|sim| (-sim.score()).max(0.0).sqrt())
                .collect()
        } else {
            Vec::new()
        };
        Self {
            query: query.to_vec(),
            metric,
            rows,
            dists,
            len: sims.len(),
            nearest: Vec::new(),
            row: Vec::new(),
            boundary: Vec::new(),
            rho: None,
            profile: Vec::new(),
            covered: 0,
            estimate: 0.0,
        }
    }

    /// An estimator over in-memory `rows`, nearest first, scoring each
    /// against `query` under `metric`.
    #[cfg(test)]
    pub(crate) fn from_rows(query: &[f32], rows: Vec<&'a [f32]>, metric: Metric) -> Self {
        let sims: Vec<Similarity> = rows
            .iter()
            .map(|row| {
                let score = if is_euclidean(metric) {
                    -query
                        .iter()
                        .zip(*row)
                        .map(|(q, c)| (q - c) * (q - c))
                        .sum::<f32>()
                } else {
                    query.iter().zip(*row).map(|(q, c)| q * c).sum()
                };
                Similarity::new(score)
            })
            .collect();
        Self::new(query, &sims, Box::new(rows), metric)
    }

    /// Covers the next candidate and returns the estimated recall of the
    /// covered prefix. `kth` is the current k-th result, `None` until the
    /// result heap holds `k`; the estimate is `None` until then too.
    pub(crate) fn cover_next(&mut self, kth: Option<Similarity>) -> io::Result<Option<f32>> {
        let i = self.covered;
        self.covered += 1;
        let Some(kth) = kth else {
            return Ok(None);
        };
        let rho = radius_from_kth(kth, self.metric);
        let recompute = self.rho.map_or(true, |old| {
            (old - rho).abs() > APS_RECOMPUTE_THRESHOLD * old
        });
        if recompute {
            self.rho = Some(rho);
            self.fetch_boundaries(self.cutoff(rho))?;
            self.profile = compute_recall_profile(
                &self.boundary,
                rho,
                self.query.len(),
                is_euclidean(self.metric),
            );
            self.estimate = self.profile[..self.covered.min(self.profile.len())]
                .iter()
                .sum();
        } else {
            self.estimate += self.profile.get(i).copied().unwrap_or(0.0);
        }
        Ok(Some(self.estimate))
    }

    /// Candidates that can reach a query ball of radius `rho`: a prefix of
    /// the ranking, never empty.
    fn cutoff(&self, rho: f32) -> usize {
        let Some(&nearest) = self.dists.first() else {
            return self.len;
        };
        self.dists
            .partition_point(|&dist| dist < nearest + 2.0 * rho)
            .max(1)
    }

    /// Computes boundary distances through candidate `cutoff - 1`. The
    /// fetched prefix only grows; candidates past a later, smaller cutoff
    /// keep their boundaries and contribute zero volume.
    fn fetch_boundaries(&mut self, cutoff: usize) -> io::Result<()> {
        if self.boundary.is_empty() && self.len > 0 {
            self.rows.append_row(0, &mut self.nearest)?;
            self.boundary.push(0.0);
        }
        let euclidean = is_euclidean(self.metric);
        for rank in self.boundary.len()..cutoff.min(self.len) {
            self.row.clear();
            self.rows.append_row(rank, &mut self.row)?;
            self.boundary.push(boundary_distance(
                &self.query,
                &self.nearest,
                &self.row,
                euclidean,
            ));
        }
        Ok(())
    }
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

    fn three_cells() -> ([f32; 2], [[f32; 2]; 3]) {
        ([0.1, 0.0], [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    }

    /// L2 similarity of a k-th result at distance `r`.
    fn kth_at(r: f32) -> Similarity {
        Similarity::new(-(r * r))
    }

    #[test]
    fn estimator_waits_for_a_full_heap() {
        let (q, c) = three_cells();
        let rows: Vec<&[f32]> = c.iter().map(|r| &r[..]).collect();
        let mut est = RecallEstimator::from_rows(&q, rows, Metric::L2);
        assert_eq!(est.cover_next(None).unwrap(), None);
        // The first estimate covers every candidate seen so far.
        let got = est.cover_next(Some(kth_at(1.5))).unwrap().unwrap();
        let profile = compute_recall_profile(&est.boundary, 1.5, 2, true);
        assert!(
            (got - (profile[0] + profile[1])).abs() < 1e-6,
            "{got} {profile:?}"
        );
    }

    #[test]
    fn estimator_reuses_profile_within_threshold() {
        let (q, c) = three_cells();
        let rows: Vec<&[f32]> = c.iter().map(|r| &r[..]).collect();
        let mut est = RecallEstimator::from_rows(&q, rows, Metric::L2);
        let first = est.cover_next(Some(kth_at(1.5))).unwrap().unwrap();
        let profile = est.profile.clone();
        // A 5% radius change keeps the cached profile.
        let second = est.cover_next(Some(kth_at(1.5 * 0.95))).unwrap().unwrap();
        assert_eq!(est.profile, profile);
        assert!((second - (first + profile[1])).abs() < 1e-6);
        // Covering everything under one profile reaches full recall.
        let third = est.cover_next(Some(kth_at(1.5 * 0.95))).unwrap().unwrap();
        assert!((third - 1.0).abs() < 1e-5, "{third}");
    }

    #[test]
    fn estimator_recomputes_past_threshold() {
        let (q, c) = three_cells();
        let rows: Vec<&[f32]> = c.iter().map(|r| &r[..]).collect();
        let mut est = RecallEstimator::from_rows(&q, rows, Metric::L2);
        est.cover_next(Some(kth_at(1.5))).unwrap();
        let got = est.cover_next(Some(kth_at(0.5))).unwrap().unwrap();
        let profile = compute_recall_profile(&est.boundary, 0.5, 2, true);
        assert_eq!(est.profile, profile);
        assert!((got - (profile[0] + profile[1])).abs() < 1e-6);
    }

    /// Serves in-memory rows and records which ranks were fetched.
    struct CountingRows<'a> {
        rows: Vec<&'a [f32]>,
        fetched: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
    }

    impl CandidateRows for CountingRows<'_> {
        fn append_row(&self, rank: usize, out: &mut Vec<f32>) -> io::Result<()> {
            self.fetched.borrow_mut().push(rank);
            out.extend_from_slice(self.rows[rank]);
            Ok(())
        }
    }

    /// Only candidates with `δ_j < δ_0 + 2ρ` are fetched, and the estimate
    /// matches a profile over the whole ranking.
    #[test]
    fn estimator_fetches_only_candidates_the_ball_can_reach() {
        let q = [0.1f32, 0.0];
        let centroids: Vec<[f32; 2]> = (0..10).map(|j| [j as f32, 0.0]).collect();
        let rows: Vec<&[f32]> = centroids.iter().map(|r| &r[..]).collect();
        let sims: Vec<Similarity> = centroids
            .iter()
            .map(|c| Similarity::new(-((c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2))))
            .collect();
        let fetched = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut est = RecallEstimator::new(
            &q,
            &sims,
            Box::new(CountingRows {
                rows: rows.clone(),
                fetched: fetched.clone(),
            }),
            Metric::L2,
        );

        assert_eq!(est.cover_next(None).unwrap(), None);
        assert!(fetched.borrow().is_empty(), "nothing fetched while filling");

        // δ = 0.1, 0.9, 1.9, ...; ρ = 0.5 reaches δ < 1.1: ranks 0 and 1.
        let got = est.cover_next(Some(kth_at(0.5))).unwrap().unwrap();
        assert_eq!(*fetched.borrow(), vec![0, 1]);
        let all =
            compute_recall_profile(&compute_boundary_distances(&q, &rows, true), 0.5, 2, true);
        assert!((got - (all[0] + all[1])).abs() < 1e-6, "{got} {all:?}");

        // A smaller ball fetches nothing new.
        est.cover_next(Some(kth_at(0.2))).unwrap();
        assert_eq!(*fetched.borrow(), vec![0, 1]);
    }
}

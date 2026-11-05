//! Survival scores binned on a fixed grid for the routed scan. A row's survival score is the
//! highest pruning threshold under which it still counts: `estimate + κσ` for a quantized row, the
//! score itself for an exactly read row.

/// A uniform grid over score space, fixed once at the fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScoreGrid {
    lo: f32,
    inv_width: f32,
}

/// Where a score falls on a grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bucket {
    Below,
    Bin(usize),
    Above,
}

impl ScoreGrid {
    pub(crate) const BINS: usize = 256;

    /// Grid for `bar` and the survival scores seen through the fill.
    ///
    /// `bar` sits on the lower edge of bin 1; the last bin ends 4σ above the larger of `bar` and
    /// the highest score, σ being the scores' standard deviation, or `1e-6 · max(1, |bar|)` when
    /// they have none.
    pub(crate) fn at_fill(bar: f32, scores: &[f32]) -> ScoreGrid {
        let finite = || {
            scores
                .iter()
                .map(|&score| f64::from(score))
                .filter(|s| s.is_finite())
        };
        let count = finite().count();
        let spread = if count == 0 {
            0.0
        } else {
            let mean = finite().sum::<f64>() / count as f64;
            (finite().map(|s| (s - mean) * (s - mean)).sum::<f64>() / count as f64).sqrt()
        };
        let bar64 = f64::from(bar);
        let spread = if spread > 0.0 {
            spread
        } else {
            1e-6 * bar64.abs().max(1.0)
        };
        let top = finite().fold(bar64, f64::max);
        let width = (top + 4.0 * spread - bar64) / (Self::BINS - 1) as f64;
        ScoreGrid {
            lo: (bar64 - width) as f32,
            inv_width: (1.0 / width) as f32,
        }
    }

    /// The bucket `score` falls in.
    #[inline]
    fn bucket(&self, score: f32) -> Bucket {
        let position = (score - self.lo) * self.inv_width;
        if position.is_nan() || position < 0.0 {
            Bucket::Below
        } else if position >= Self::BINS as f32 {
            Bucket::Above
        } else {
            Bucket::Bin(position as usize)
        }
    }
}

/// Row counts by bucket on one grid; `bins + below + above == rows`.
#[derive(Clone, Debug)]
pub(crate) struct ScoreHistogram {
    bins: Box<[f32; ScoreGrid::BINS]>,
    below: f32,
    above: f32,
    rows: f32,
}

impl ScoreHistogram {
    pub(crate) fn zero() -> Self {
        Self {
            bins: Box::new([0.0; ScoreGrid::BINS]),
            below: 0.0,
            above: 0.0,
            rows: 0.0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.bins.fill(0.0);
        self.below = 0.0;
        self.above = 0.0;
        self.rows = 0.0;
    }

    #[cfg(test)]
    pub(crate) fn add(&mut self, grid: ScoreGrid, scores: &[f32]) {
        for &score in scores {
            self.add_score(grid, score);
        }
    }

    /// `add` of `estimate + kappa·sigma` per row, fused.
    pub(crate) fn add_upper_endpoints(
        &mut self,
        grid: ScoreGrid,
        estimates: &[f32],
        sigmas: &[f32],
        kappa: f32,
    ) {
        debug_assert_eq!(estimates.len(), sigmas.len());
        for (&estimate, &sigma) in estimates.iter().zip(sigmas) {
            self.add_score(grid, estimate + kappa * sigma);
        }
    }

    #[inline]
    fn add_score(&mut self, grid: ScoreGrid, score: f32) {
        match grid.bucket(score) {
            Bucket::Below => self.below += 1.0,
            Bucket::Bin(bin) => self.bins[bin] += 1.0,
            Bucket::Above => self.above += 1.0,
        }
        self.rows += 1.0;
    }

    pub(crate) fn add_histogram(&mut self, other: &ScoreHistogram) {
        for (count, &more) in self.bins.iter_mut().zip(other.bins.iter()) {
            *count += more;
        }
        self.below += other.below;
        self.above += other.above;
        self.rows += other.rows;
    }

    pub(crate) fn scale(&mut self, factor: f32) {
        for count in self.bins.iter_mut() {
            *count *= factor;
        }
        self.below *= factor;
        self.above *= factor;
        self.rows *= factor;
    }

    /// Rows at or above `bar`: every bin above the bar's bin, that bin's linear share, and `above`.
    pub(crate) fn at_or_above(&self, grid: ScoreGrid, bar: f32) -> f32 {
        let position = (bar - grid.lo) * grid.inv_width;
        let (first, share) = if position.is_nan() || position < 0.0 {
            (0, 1.0)
        } else if position >= ScoreGrid::BINS as f32 {
            return self.above;
        } else {
            let bin = position as usize;
            (bin, ((bin + 1) as f32 - position).clamp(0.0, 1.0))
        };
        let higher: f32 = self.bins[first + 1..].iter().sum();
        self.above + higher + self.bins[first] * share
    }

    pub(crate) fn rows(&self) -> f32 {
        self.rows
    }
}

/// Where one cluster's survival scores fold: before the fill, into the list that sizes the grid;
/// after it, into the cluster's histogram.
pub(crate) enum Fold<'f> {
    Prefill(&'f mut Vec<f32>),
    Binned(&'f mut ScoreHistogram, ScoreGrid),
}

impl Fold<'_> {
    /// Folds `estimate + kappa·sigma` per row.
    pub(crate) fn upper_endpoints(&mut self, estimates: &[f32], sigmas: &[f32], kappa: f32) {
        match self {
            Fold::Prefill(scores) => scores.extend(
                estimates
                    .iter()
                    .zip(sigmas)
                    .map(|(&estimate, &sigma)| estimate + kappa * sigma),
            ),
            Fold::Binned(histogram, grid) => {
                histogram.add_upper_endpoints(*grid, estimates, sigmas, kappa)
            }
        }
    }

    /// Folds one exact score.
    #[inline]
    pub(crate) fn score(&mut self, score: f32) {
        match self {
            Fold::Prefill(scores) => scores.push(score),
            Fold::Binned(histogram, grid) => histogram.add_score(*grid, score),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::search::interval::Estimate;

    fn width_of(grid: ScoreGrid) -> f64 {
        f64::from(1.0 / grid.inv_width)
    }

    fn total(histogram: &ScoreHistogram) -> f32 {
        histogram.bins.iter().sum::<f32>() + histogram.below + histogram.above
    }

    /// The bar sits one bin above `lo`, the last bin ends 4σ above the top score, and a flat set
    /// of scores falls back to a spread from the bar's magnitude.
    #[test]
    fn grid_at_fill() {
        let scores = [0.2f32, 0.4, 0.6, 0.8];
        let grid = ScoreGrid::at_fill(0.5, &scores);
        let sigma = 0.05f64.sqrt();
        let width = (0.8 + 4.0 * sigma - 0.5) / 255.0;
        assert!((width_of(grid) - width).abs() < 1e-6);
        assert!((f64::from(grid.lo) - (0.5 - width)).abs() < 1e-6);
        assert_eq!(grid.bucket(0.5 + 1e-6), Bucket::Bin(1));
        let hi = f64::from(grid.lo) + ScoreGrid::BINS as f64 * width_of(grid);
        assert!((hi - (0.8 + 4.0 * sigma)).abs() < 1e-5);

        // A bar above every score keeps the grid on the bar.
        let grid = ScoreGrid::at_fill(2.0, &scores);
        assert!((f64::from(grid.lo) - (2.0 - 4.0 * sigma / 255.0)).abs() < 1e-6);

        // No spread: 1e-6 · max(1, |bar|).
        let grid = ScoreGrid::at_fill(-3.0, &[-3.0, -3.0]);
        let width = 4.0 * 3e-6 / 255.0;
        assert!((width_of(grid) - width).abs() < 1e-9);
        let grid = ScoreGrid::at_fill(0.1, &[]);
        assert!((width_of(grid) - 4e-6 / 255.0).abs() < 1e-10);
    }

    /// Scores at `lo`, on an inner edge and at `hi` land in the first bin, the bin above the edge
    /// and the overflow.
    #[test]
    fn buckets_at_the_edges() {
        let grid = ScoreGrid {
            lo: -1.0,
            inv_width: 128.0,
        };
        assert_eq!(grid.bucket(-1.0), Bucket::Bin(0));
        assert_eq!(grid.bucket(-1.0 - 1e-3), Bucket::Below);
        assert_eq!(grid.bucket(-1.0 + 10.0 / 128.0), Bucket::Bin(10));
        assert_eq!(grid.bucket(1.0), Bucket::Above);
        assert_eq!(grid.bucket(1.0 - 1e-3), Bucket::Bin(255));
        assert_eq!(grid.bucket(f32::NAN), Bucket::Below);
    }

    /// Rows at or above a bar against counting the rows: a bar below `lo`, on a bin edge, mid-bin
    /// with the straddled bin's rows spread evenly, and above `hi`.
    #[test]
    fn at_or_above_matches_brute_force() {
        let grid = ScoreGrid {
            lo: 0.0,
            inv_width: 64.0,
        };
        let at = |bin: f32| bin / 64.0;
        let mut rows: Vec<f32> = [0.125, 0.375, 0.625, 0.875]
            .iter()
            .map(|&offset| at(40.0 + offset))
            .collect();
        rows.extend([at(10.5), at(41.5), at(100.5), at(300.0), -0.5]);
        let mut histogram = ScoreHistogram::zero();
        histogram.add(grid, &rows);
        assert_eq!(total(&histogram), histogram.rows());
        let brute = |bar: f32| rows.iter().filter(|&&row| row >= bar).count() as f32;
        // The row under `lo` counts below every bar.
        assert_eq!(histogram.at_or_above(grid, -1.0), brute(-1.0) - 1.0);
        for bar in [at(41.0), at(40.5), at(100.0)] {
            assert!(
                (histogram.at_or_above(grid, bar) - brute(bar)).abs() < 1e-5,
                "{bar}"
            );
        }
        assert_eq!(histogram.at_or_above(grid, at(256.0)), 1.0);
        assert_eq!(histogram.at_or_above(grid, at(400.0)), 1.0);
    }

    /// Decaying then adding, cluster by cluster, is the explicit weighted sum of the clusters, and
    /// every operation keeps the counts adding up to the rows.
    #[test]
    fn scale_and_add_are_a_weighted_sum() {
        let grid = ScoreGrid {
            lo: 0.0,
            inv_width: 100.0,
        };
        let parts: [&[f32]; 3] = [&[0.1, 0.2, 5.0], &[0.15, -1.0], &[0.3, 0.3, 0.31, 2.55]];
        let decays = [0.5f32, 0.8, 0.9];
        let histogram = |part: &[f32]| {
            let mut one = ScoreHistogram::zero();
            one.add(grid, part);
            one
        };
        let mut running = ScoreHistogram::zero();
        for (part, &decay) in parts.iter().zip(&decays) {
            running.scale(decay);
            running.add_histogram(&histogram(part));
            assert!((total(&running) - running.rows()).abs() < 1e-5);
        }
        let mut explicit = ScoreHistogram::zero();
        for (j, part) in parts.iter().enumerate() {
            let mut one = histogram(part);
            one.scale(decays[j + 1..].iter().product());
            explicit.add_histogram(&one);
        }
        assert!((running.rows() - explicit.rows()).abs() < 1e-5);
        assert!((running.rows() - (3.0 * 0.8 * 0.9 + 2.0 * 0.9 + 4.0)).abs() < 1e-5);
        for bar in [0.0f32, 0.15, 0.305, 2.0, 9.0] {
            let (a, b) = (
                running.at_or_above(grid, bar),
                explicit.at_or_above(grid, bar),
            );
            assert!((a - b).abs() < 1e-5, "{bar}: {a} {b}");
        }
        running.clear();
        assert_eq!((running.rows(), total(&running)), (0.0, 0.0));
    }

    /// A fold of upper endpoints records `Estimate::upper` per row, before the fill as scores and
    /// after it as counts.
    #[test]
    fn folds_record_upper_endpoints() {
        let (estimates, sigmas, kappa) = ([0.5f32, -0.25, 1.0], [0.1f32, 0.3, 0.0], 2.5);
        let mut prefill = Vec::new();
        Fold::Prefill(&mut prefill).upper_endpoints(&estimates, &sigmas, kappa);
        Fold::Prefill(&mut prefill).score(0.75);
        let expected: Vec<f32> = estimates
            .iter()
            .zip(&sigmas)
            .map(|(&e, &s)| Estimate(e).upper(s, kappa).0)
            .chain([0.75])
            .collect();
        assert_eq!(prefill, expected);
        let grid = ScoreGrid::at_fill(0.5, &prefill);
        let mut histogram = ScoreHistogram::zero();
        let mut fold = Fold::Binned(&mut histogram, grid);
        fold.upper_endpoints(&estimates, &sigmas, kappa);
        fold.score(0.75);
        let mut direct = ScoreHistogram::zero();
        direct.add(grid, &expected);
        assert_eq!(histogram.bins, direct.bins);
        assert_eq!(histogram.rows(), 4.0);
    }
}

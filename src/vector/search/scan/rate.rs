//! Opening and recent admission rates evaluated at the same current pruning threshold.

use super::histogram::{ScoreGrid, ScoreHistogram};

/// Admission evidence measured in filter-eligible rows.
pub(crate) struct AdmissionHistograms {
    grid: ScoreGrid,
    /// Valid rows through the fill cluster, and valid rows scanned so far.
    w0: usize,
    w: usize,
    /// The cluster being shortlisted; folded into the windows at its end.
    cluster: ScoreHistogram,
    /// The clusters after the fill through the first boundary at or past `2 · w0`.
    opening: ScoreHistogram,
    opening_closed: bool,
    /// The clusters after the fill, each weighted by `(W before / W after)^4`.
    recent: ScoreHistogram,
}

impl AdmissionHistograms {
    /// Starts at the fill: `bar` is the pruning threshold, `scores` every survival score seen
    /// through the fill cluster, `w0` the valid rows through it.
    pub(crate) fn at_fill(bar: f32, scores: &[f32], w0: usize) -> Self {
        Self {
            grid: ScoreGrid::at_fill(bar, scores),
            w0,
            w: w0,
            cluster: ScoreHistogram::zero(),
            opening: ScoreHistogram::zero(),
            opening_closed: false,
            recent: ScoreHistogram::zero(),
        }
    }

    pub(crate) fn grid(&self) -> ScoreGrid {
        self.grid
    }

    pub(crate) fn cluster_mut(&mut self) -> &mut ScoreHistogram {
        &mut self.cluster
    }

    /// Ends a cluster of `valid_rows`: decays the recent window by `((W - rows) / W)^4` before
    /// adding the cluster, and adds it to the opening until a boundary at or past `2 · w0` closes
    /// it, the straddling cluster included.
    pub(crate) fn finish_cluster(&mut self, valid_rows: usize) {
        let before = self.w;
        self.w += valid_rows;
        let ratio = before as f32 / self.w as f32;
        let squared = ratio * ratio;
        self.recent.scale(squared * squared);
        self.recent.add_histogram(&self.cluster);
        if !self.opening_closed {
            self.opening.add_histogram(&self.cluster);
            self.opening_closed = self.w >= 2 * self.w0;
        }
        self.cluster.clear();
    }

    /// The rates under `bar`; `None` until an opening exists.
    pub(crate) fn reading(&self, bar: f32) -> Option<RateReading> {
        if !self.opening_closed {
            return None;
        }
        let opening_rows = self.opening.at_or_above(self.grid, bar);
        let opening = opening_rows / self.opening.rows();
        let recent_rows = self.recent.rows();
        let current = if recent_rows > 0.0 {
            self.recent.at_or_above(self.grid, bar) / recent_rows
        } else {
            0.0
        };
        Some(RateReading {
            opening,
            opening_rows,
            current,
            evidence: opening * recent_rows,
            recent_rows,
            opening_total: self.opening.rows(),
        })
    }
}

/// The admission rate under the bar in force, as the scan reports it after a cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateReading {
    /// Share of the opening window's rows at or above the bar, and that count.
    pub(crate) opening: f32,
    pub(crate) opening_rows: f32,
    /// Share of the recent window's weighted rows at or above the bar; 0 while it is empty.
    pub(crate) current: f32,
    /// `opening × recent rows`: the admissions the recent window would hold had nothing decayed.
    pub(crate) evidence: f32,
    pub(crate) recent_rows: f32,
    pub(crate) opening_total: f32,
}

impl RateReading {
    pub(crate) fn reached(self, fraction: f64) -> bool {
        if self.opening_rows < 1.0 {
            return self.current == 0.0 && self.recent_rows >= 5.0 * self.opening_total;
        }
        self.evidence >= 5.0 && f64::from(self.current) <= fraction * f64::from(self.opening)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_opening_does_not_hide_a_late_admission() {
        let quiet = RateReading {
            opening: 0.0,
            opening_rows: 0.0,
            current: 0.0,
            evidence: 0.0,
            recent_rows: 500.0,
            opening_total: 100.0,
        };
        assert!(quiet.reached(0.01));
        assert!(!RateReading {
            current: 0.001,
            ..quiet
        }
        .reached(0.01));
        assert!(!RateReading {
            recent_rows: 499.0,
            ..quiet
        }
        .reached(0.01));
    }

    #[test]
    fn rate_ratio_is_independent_of_uniform_selectivity_and_score_scale() {
        let mut readings = Vec::new();
        for copies in [1, 10, 100] {
            for (scale, shift) in [(1.0, 0.0), (4.0, -100.0)] {
                let scores: Vec<_> = (0..12)
                    .map(|c| {
                        (0..40 * copies)
                            .map(|i| {
                                let score = if i < (12 - c) * copies { 0.9 } else { 0.1 };
                                score * scale + shift
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let parts: Vec<_> = scores.iter().map(|s| (s.len(), &s[..])).collect();
                readings.push(
                    scan(0.5 * scale + shift, 40 * copies, &parts)
                        .reading(0.5 * scale + shift)
                        .unwrap(),
                );
            }
        }
        let expected = readings[0].current / readings[0].opening;
        for reading in readings {
            assert!((reading.current / reading.opening - expected).abs() < 1e-5);
        }
    }

    /// A segment's scan from the fill: `w0` rows through the fill, then `(rows, scores)` clusters,
    /// every score folded into the cluster before it ends.
    fn scan(bar: f32, w0: usize, clusters: &[(usize, &[f32])]) -> AdmissionHistograms {
        let prefill = vec![bar; w0];
        let mut histograms = AdmissionHistograms::at_fill(bar, &prefill, w0);
        for &(rows, scores) in clusters {
            let grid = histograms.grid();
            histograms.cluster_mut().add(grid, scores);
            histograms.finish_cluster(rows);
        }
        histograms
    }

    /// The opening closes at the first boundary at or past `2 · w0`, the straddling cluster
    /// included, and no reading exists before.
    #[test]
    fn opening_closes_at_twice_w0() {
        let above = [0.9f32; 10];
        let mut histograms = scan(0.5, 100, &[(60, &above[..6])]);
        assert!(!histograms.opening_closed);
        assert_eq!(histograms.reading(0.5), None);
        let grid = histograms.grid();
        histograms.cluster_mut().add(grid, &above[..5]);
        histograms.finish_cluster(50);
        assert!(histograms.opening_closed);
        assert_eq!(histograms.opening.rows(), 11.0);
        // Later clusters stay out of the opening.
        histograms.cluster_mut().add(grid, &above[..3]);
        histograms.finish_cluster(30);
        assert_eq!(histograms.opening.rows(), 11.0);

        // One cluster spanning W0 to 3 · W0 is the whole opening.
        let histograms = scan(0.5, 100, &[(200, &above[..])]);
        assert!(histograms.opening_closed);
        assert_eq!(histograms.opening.rows(), 10.0);
        assert!(histograms.reading(0.5).is_some());
    }

    /// Under a bar that does not move, the reading is the count-based ratio: admissions over rows
    /// in the opening, and `(W before / W after)^4`-weighted admissions over weighted rows since.
    #[test]
    fn a_held_bar_reads_the_counted_ratio() {
        let bar = 0.5f32;
        let cluster = |admitted: usize, rows: usize| {
            let mut scores = vec![0.9f32; admitted];
            scores.extend(vec![0.1f32; rows - admitted]);
            scores
        };
        let sequence = [(4, 50), (3, 60), (2, 80), (1, 120), (0, 200)];
        let clusters: Vec<Vec<f32>> = sequence.iter().map(|&(a, r)| cluster(a, r)).collect();
        let parts: Vec<(usize, &[f32])> = clusters.iter().map(|c| (c.len(), &c[..])).collect();
        let histograms = scan(bar, 100, &parts);
        let reading = histograms.reading(bar).unwrap();
        // The opening is the first two clusters: W reaches 210 >= 200.
        assert!((reading.opening - 7.0 / 110.0).abs() < 1e-6);
        assert_eq!(reading.opening_rows, 7.0);
        let (mut admitted, mut rows, mut w) = (0.0f64, 0.0f64, 100.0f64);
        for &(a, r) in &sequence {
            let before = w;
            w += r as f64;
            let decay = (before / w).powi(4);
            admitted = admitted * decay + a as f64;
            rows = rows * decay + r as f64;
        }
        assert!((f64::from(reading.current) - admitted / rows).abs() < 1e-6);
        assert!((f64::from(reading.evidence) - 7.0 / 110.0 * rows).abs() < 1e-3);
    }

    /// The recent window is empty right after the fill, and under steady clusters the evidence a
    /// reading carries grows with the work scanned.
    #[test]
    fn evidence_grows_with_work() {
        let histograms = scan(0.5, 100, &[]);
        assert_eq!(histograms.recent.rows(), 0.0);
        let mut cluster = vec![0.9f32; 5];
        cluster.extend([0.1f32; 15]);
        let steady: Vec<(usize, &[f32])> = vec![(20, &cluster[..]); 5];
        let mut histograms = scan(0.5, 100, &steady);
        let mut evidence = Vec::new();
        for _ in 0..10 {
            evidence.push(histograms.reading(0.5).unwrap().evidence);
            let grid = histograms.grid();
            histograms.cluster_mut().add(grid, &cluster);
            histograms.finish_cluster(20);
        }
        assert!(
            evidence.windows(2).all(|pair| pair[1] > pair[0]),
            "{evidence:?}"
        );
    }
}

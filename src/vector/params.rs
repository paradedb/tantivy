//! Query-time probe parameters: the work budget and the located path.

use crate::vector::search::source::controller::open_share;

/// Default [`ProbeParams::fixed_probe_cost_rows`]: the cost of opening a cluster, in rows of full
/// work, fitted on the reference fixture.
///
/// Policy constant: the open share `x = cost / (cost + n_avg)` follows the index's granularity.
/// It covers the open only; routing cost is not modeled.
pub const DEFAULT_FIXED_PROBE_COST_ROWS: f64 = 1.64;

/// Query-time probe budget for IVF vector search.
///
/// The budget is in work units: one unit is one average cluster of work, charged per opened
/// cluster and per scored row, so a selective filter probes further down the ranking. A segment
/// scan stops when its budget is spent or its ranking runs out. The numeric defaults are
/// provisional pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct ProbeParams {
    /// Adaptive admission-rate stopping within the routed candidates and work budget.
    pub quantized_recall: Option<QuantizedRecallParams>,
    /// Work ceiling as a fraction of segment capacity. Default 0.01.
    pub max_probe_fraction: f32,
    /// Work-budget floor when adaptive probing is disabled. Default 16.
    pub min_probe_clusters: usize,
    /// Lets a quantized segment score a cluster's rows exactly before the exact phase when that
    /// reads strictly fewer storage blocks (bytes without block geometry) than the cheaper
    /// quantized read. Exact rows join the pruning bound as zero-width intervals. Default `true`.
    pub exact_plan: bool,
    /// Which segments locate their filter's matches through the document location map instead
    /// of routing; unfiltered queries always route. Default [`LocatedPath::Auto`].
    pub located: LocatedPath,
    /// Cost of opening a cluster, in rows of full work; finite and positive. Default
    /// [`DEFAULT_FIXED_PROBE_COST_ROWS`].
    pub fixed_probe_cost_rows: f64,
}

/// Admission-rate decay calibrated to a requested recall target.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedRecallParams {
    pub target: f64,
    pub recall_exponent: f64,
}

impl QuantizedRecallParams {
    pub(crate) fn admission_fraction(self) -> Option<f64> {
        (self.target < 1.0).then(|| (self.recall_exponent * (-self.target).ln_1p()).exp())
    }
}

/// Which segments take the located path: a segment whose filter matches at most a cap of
/// documents, deleted ones included, skips routing and scores every match, located through the
/// document location map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LocatedPath {
    /// Every segment routes.
    #[default]
    Off,
    /// The cap is `floor(selectivity * max_doc)`.
    MaxSelectivity(Selectivity),
    /// The cap is the rows the initial routed work estimate buys with no cluster opened,
    /// `floor(work * n_avg / (1 - x))`. Adaptive scans use their expected work estimate;
    /// nonadaptive scans use their work ceiling.
    Auto,
}

/// A fraction in `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Selectivity(f32);

impl Selectivity {
    /// `value` as a selectivity, or `None` outside `(0, 1]`.
    pub fn new(value: f32) -> Option<Self> {
        (value > 0.0 && value <= 1.0).then_some(Self(value))
    }

    /// The fraction.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for ProbeParams {
    fn default() -> Self {
        Self {
            quantized_recall: None,
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
            exact_plan: true,
            located: LocatedPath::Auto,
            fixed_probe_cost_rows: DEFAULT_FIXED_PROBE_COST_ROWS,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

impl ProbeParams {
    /// Expected initial work for bypass selection and allocation, separate from routing headroom.
    pub(crate) fn initial_work(&self, clusters: usize) -> Option<f64> {
        let recall = self.quantized_recall.filter(|recall| recall.target < 1.0)?;
        let fraction = 0.005 * (-recall.target).ln_1p() / 0.05_f64.ln();
        Some((fraction * clusters as f64).clamp(1.0, clusters.max(1) as f64))
    }

    /// The number of clusters the stacked router is asked to rank for a
    /// segment whose resolved work budget is `budget` units.
    ///
    /// An average cluster costs `open_share + (1 - open_share) ×
    /// match_fraction` units, since only filter matches charge row work, so
    /// the budget buys `budget / that` clusters: `budget` unfiltered, up to
    /// `budget / open_share` as the filter approaches empty. The count is
    /// capped at the segment's cluster count. Adaptive scans scale a base window by the
    /// admission fraction, with a filter adjustment whose relative effect shrinks as the window
    /// grows. Nonadaptive scans apply the `min_probe_clusters` floor.
    ///
    /// * `open_share` (`f64`) — the per-index open share `x`.
    /// * `match_fraction` (`f64`) — the share of the segment's docs the filter matches, in `(0,
    ///   1]`.
    pub(crate) fn router_k(
        &self,
        budget: f64,
        open_share: f64,
        match_fraction: f64,
        clusters_in_segment: usize,
    ) -> usize {
        let match_fraction = match_fraction.clamp(0.0, 1.0);
        let cluster_cost =
            (open_share + (1.0 - open_share) * match_fraction).max(f64::MIN_POSITIVE);
        let clusters = (budget.max(0.0) / cluster_cost).ceil();
        let clusters = if let Some(fraction) = self
            .quantized_recall
            .and_then(QuantizedRecallParams::admission_fraction)
        {
            let base = (0.002 / fraction.powf(0.625)).min(1.0);
            let headroom = (0.05 * (cluster_cost.sqrt().recip() - 1.0)).min(0.4);
            let remaining = 1.0 - base;
            let taper = (remaining / 0.9).min(1.0);
            let adjusted = base + headroom * base.powf(0.125) * remaining * taper * taper;
            let expected = self.initial_work(clusters_in_segment).unwrap_or(0.0) / cluster_cost;
            let window = (adjusted * clusters_in_segment as f64).max(expected).ceil();
            (clusters as usize).min(window as usize)
        } else {
            (clusters as usize).max(self.min_probe_clusters)
        };
        clusters.min(clusters_in_segment).max(1)
    }
}

impl ProbeParams {
    /// The segment's work budget with the pricing it was resolved under, `(budget, n_avg, x)`.
    ///
    /// The budget is `max_probe_fraction` of the segment's capacity in work units, floored at
    /// one for adaptive scans or `min_probe_clusters` otherwise, and capped at capacity.
    /// `n_avg` is the docs per cluster and `x` the
    /// open share. `native_docs_in_segment` counts distinct docs with a vector, as written. A
    /// non-positive `max_probe_fraction` or a non-finite or non-positive `fixed_probe_cost_rows` is
    /// an error.
    pub(crate) fn resolved_work_budget(
        &self,
        clusters_in_segment: usize,
        native_docs_in_segment: usize,
    ) -> crate::Result<(f64, f64, f64)> {
        if self.quantized_recall.is_some_and(|stop| {
            !stop.target.is_finite()
                || !(0.0 < stop.target && stop.target <= 1.0)
                || !stop.recall_exponent.is_finite()
                || stop.recall_exponent <= 0.0
        }) {
            return Err(crate::TantivyError::InvalidArgument(
                "quantized recall requires a target in (0, 1] and a finite positive recall \
                 exponent"
                    .into(),
            ));
        }
        if !(self.max_probe_fraction > 0.0) {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_fraction must be greater than 0".to_string(),
            ));
        }
        if !(self.fixed_probe_cost_rows.is_finite() && self.fixed_probe_cost_rows > 0.0) {
            return Err(crate::TantivyError::InvalidArgument(
                "fixed_probe_cost_rows must be finite and greater than 0".to_string(),
            ));
        }
        let n_avg = native_docs_in_segment as f64 / clusters_in_segment.max(1) as f64;
        // The unit identity (1 unit = x + (1 - x); exhaustive scan = C units) holds for any x.
        let x = open_share(self.fixed_probe_cost_rows, n_avg);
        let segment_units = clusters_in_segment as f64 * x
            + (1.0 - x) * native_docs_in_segment as f64 / n_avg.max(f64::MIN_POSITIVE);
        let budget = (self.max_probe_fraction as f64 * segment_units)
            .max(if self.initial_work(clusters_in_segment).is_some() {
                1.0
            } else {
                self.min_probe_clusters as f64
            })
            .min(segment_units);
        Ok((budget, n_avg, x))
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeParams;

    #[test]
    fn adaptive_window_scales_with_recall_and_segment_size_without_a_probe_floor() {
        let params = |target| ProbeParams {
            quantized_recall: Some(super::QuantizedRecallParams {
                target,
                recall_exponent: 1.25,
            }),
            max_probe_fraction: 0.001,
            ..Default::default()
        };
        assert!(params(0.9).initial_work(1000) < params(0.95).initial_work(1000));
        assert!(params(0.95).initial_work(1000) < params(0.99).initial_work(1000));
        assert!((params(0.95).initial_work(1000).unwrap() - 5.0).abs() < 1e-10);
        assert!((params(0.95).initial_work(4000).unwrap() - 20.0).abs() < 1e-10);
        assert!((params(0.95).resolved_work_budget(1000, 100_000).unwrap().0 - 1.0).abs() < 1e-6);
        assert_eq!(params(0.95).router_k(2.0, 0.1, 1.0, 1000), 2);
        assert_eq!(params(0.9).router_k(1000.0, 0.1, 1.0, 1000), 13);
        assert_eq!(params(0.95).router_k(1000.0, 0.1, 1.0, 1000), 21);
        assert_eq!(params(0.99).router_k(1000.0, 0.1, 1.0, 1000), 74);
        assert_eq!(params(0.95).router_k(2000.0, 0.1, 0.01, 2000), 164);
        let cost = 0.004 + 0.996 * 0.0067;
        let expected = params(0.95).initial_work(4000).unwrap();
        let window = params(0.95).router_k(4000.0, 0.004, 0.0067, 4000);
        assert!(window as f64 * cost >= expected);
        assert_eq!(
            params(0.95).router_k(2.0, 0.004, 0.0067, 4000),
            (2.0_f64 / cost).ceil() as usize
        );
        for open_share in [f64::MIN_POSITIVE, 1e-9, 0.004, 0.1, 1.0] {
            for match_fraction in [0.0, 0.001, 0.01, 0.1, 1.0] {
                let mut previous = 0;
                for step in 1..=1000 {
                    let target = step as f64 / 1000.0;
                    let window = params(target).router_k(4000.0, open_share, match_fraction, 4000);
                    assert!(window >= previous && window <= 4000);
                    previous = window;
                }
            }
        }
        assert_eq!(params(1.0).initial_work(1000), None);
        assert_eq!(
            params(1.0).resolved_work_budget(1000, 100_000).unwrap().0,
            16.0
        );
    }

    /// Capacity is the cluster count, exactly, whatever the granularity:
    /// `units_seg = C*x + (1 - x)*N/n_avg == C` for every x the open
    /// share can resolve to. That identity is what lets `f` keep meaning
    /// "this fraction of the index's work" at any granularity.
    #[test]
    fn capacity_is_the_cluster_count() -> crate::Result<()> {
        let all = ProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            ..Default::default()
        };
        for (clusters, docs) in [(1000usize, 20_000usize), (9, 20), (2, 2), (64, 6400)] {
            let (budget, _n_avg, _x) = all.resolved_work_budget(clusters, docs)?;
            assert!(
                (budget - clusters as f64).abs() <= 1e-9 * clusters as f64,
                "f=1 must buy exactly C units at C={clusters}, N={docs}: {budget}"
            );
        }
        Ok(())
    }

    /// The fraction scales the capacity, the floor lifts small segments,
    /// and the capacity caps both.
    #[test]
    fn budget_resolves_against_capacity() -> crate::Result<()> {
        let params = ProbeParams {
            max_probe_fraction: 0.25,
            ..Default::default()
        };
        // A quarter of the capacity - well above the floor.
        let (budget, _, _) = params.resolved_work_budget(1000, 20_000)?;
        assert!((budget - 250.0).abs() < 1e-6, "{budget}");
        // A fraction resolving below MIN_PROBE_CLUSTERS is lifted to it...
        let (budget, _, _) = params.resolved_work_budget(40, 800)?;
        assert!(
            (budget - super::MIN_PROBE_CLUSTERS as f64).abs() < 1e-6,
            "{budget}"
        );
        // ...but the floor never exceeds the capacity.
        let (budget, _, _) = params.resolved_work_budget(2, 40)?;
        assert!((budget - 2.0).abs() < 1e-6, "{budget}");
        // A fraction above 1.0 clamps to the capacity - small segments
        // scan exhaustively.
        let all = ProbeParams {
            max_probe_fraction: 2.0,
            ..Default::default()
        };
        let (budget, _, _) = all.resolved_work_budget(10, 200)?;
        assert!((budget - 10.0).abs() < 1e-6, "{budget}");
        Ok(())
    }

    /// Unfiltered, the stacked router is asked for the budget; a selective
    /// filter stretches it toward `budget / open_share`. Either way it is
    /// lifted to the probe floor and capped at the segment's clusters.
    #[test]
    fn router_k_tracks_budget_and_filter_selectivity() {
        let params = ProbeParams::default();
        let x = 0.2;
        assert_eq!(params.router_k(20.0, x, 1.0, 1000), 20);
        assert_eq!(params.router_k(20.4, x, 1.0, 1000), 21);
        // Half the docs match: a cluster costs 0.2 + 0.8 * 0.5 = 0.6 units.
        assert_eq!(params.router_k(20.0, x, 0.5, 1000), 34);
        // Nothing matches: only opens charge.
        assert_eq!(params.router_k(20.0, x, 0.0, 1000), 100);
        assert_eq!(
            params.router_k(2.0, x, 1.0, 1000),
            super::MIN_PROBE_CLUSTERS
        );
        assert_eq!(params.router_k(20.0, x, 1.0, 18), 18);
        assert_eq!(params.router_k(20.0, 0.0, 0.0, 1000), 1000);
        assert_eq!(params.router_k(0.0, x, 1.0, 0), 1);
    }

    /// A non-positive probe fraction or a non-finite or non-positive open cost is an error.
    #[test]
    fn invalid_budget_params_error() {
        for fraction in [0.0, -1.0] {
            let params = ProbeParams {
                max_probe_fraction: fraction,
                ..Default::default()
            };
            assert!(params.resolved_work_budget(9, 180).is_err());
        }
        for cost in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let params = ProbeParams {
                fixed_probe_cost_rows: cost,
                ..Default::default()
            };
            assert!(params.resolved_work_budget(9, 180).is_err(), "{cost}");
        }
    }
}

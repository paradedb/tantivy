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
    /// Probe fraction before selectivity normalization. Default 0.01.
    pub max_probe_fraction: f32,
    /// Intercept of the clipped filter-cost exponent. Default 0.
    pub selectivity_exponent: f64,
    /// Non-positive slope of the exponent against log probe fraction. Default 0.
    pub selectivity_slope: f64,
    /// Floor on the resolved budget, in work units, applied before the capacity cap. Default 0.
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
    /// The cap is the rows the segment's routed work budget buys with no cluster opened,
    /// `floor(budget * n_avg / (1 - x))`: a filter matching no more documents than routing may
    /// score is located instead.
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
            max_probe_fraction: 0.01,
            selectivity_exponent: 0.0,
            selectivity_slope: 0.0,
            min_probe_clusters: 0,
            exact_plan: true,
            located: LocatedPath::Auto,
            fixed_probe_cost_rows: DEFAULT_FIXED_PROBE_COST_ROWS,
        }
    }
}

fn filtered_cluster_cost(open_share: f64, match_fraction: f64) -> f64 {
    (open_share + (1.0 - open_share) * match_fraction.clamp(0.0, 1.0)).max(f64::MIN_POSITIVE)
}

impl ProbeParams {
    pub(crate) fn normalized_budget(&self, budget: f64, open_share: f64, matches: f64) -> f64 {
        if self.max_probe_fraction >= 1.0
            || (self.selectivity_exponent == 0.0 && self.selectivity_slope == 0.0)
        {
            return budget;
        }
        let exponent = (self.selectivity_exponent
            + self.selectivity_slope * f64::from(self.max_probe_fraction).ln())
        .clamp(0.0, 2.0);
        (budget * filtered_cluster_cost(open_share, matches).powf(exponent))
            .max(self.min_probe_clusters as f64)
            .min(budget)
    }

    /// The number of clusters the stacked router is asked to rank for a
    /// segment whose resolved work budget is `budget` units.
    ///
    /// An average cluster costs `open_share + (1 - open_share) ×
    /// match_fraction` units, since only filter matches charge row work, so
    /// the budget buys `budget / that` clusters: `budget` unfiltered, up to
    /// `budget / open_share` as the filter approaches empty. The count is
    /// floored at `min_probe_clusters` and capped at the segment's cluster
    /// count. There is no headroom: when ranked clusters run smaller than
    /// average, the ranking can run out
    /// before the budget and the scan ends `Exhausted`.
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
        let cluster_cost = filtered_cluster_cost(open_share, match_fraction);
        let clusters = (budget.max(0.0) / cluster_cost).ceil();
        (clusters as usize)
            .max(self.min_probe_clusters)
            .min(clusters_in_segment)
            .max(1)
    }
}

impl ProbeParams {
    /// The segment's work budget with the pricing it was resolved under, `(budget, n_avg, x)`.
    ///
    /// The budget is `max_probe_fraction` of the segment's capacity in work units, floored at
    /// `min_probe_clusters` and capped at the capacity; `n_avg` is the docs per cluster and `x` the
    /// open share. `native_docs_in_segment` counts distinct docs with a vector, as written. A
    /// non-positive `max_probe_fraction` or a non-finite or non-positive `fixed_probe_cost_rows` is
    /// an error.
    pub(crate) fn resolved_work_budget(
        &self,
        clusters_in_segment: usize,
        native_docs_in_segment: usize,
    ) -> crate::Result<(f64, f64, f64)> {
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
        for (name, value, low, high) in [
            ("selectivity_exponent", self.selectivity_exponent, -2.0, 2.0),
            ("selectivity_slope", self.selectivity_slope, -1.0, 0.0),
        ] {
            if !(low..=high).contains(&value) {
                return Err(crate::TantivyError::InvalidArgument(format!(
                    "{name} must be between {low} and {high}"
                )));
            }
        }
        let n_avg = native_docs_in_segment as f64 / clusters_in_segment.max(1) as f64;
        // The unit identity (1 unit = x + (1 - x); exhaustive scan = C units) holds for any x.
        let x = open_share(self.fixed_probe_cost_rows, n_avg);
        let segment_units = clusters_in_segment as f64 * x
            + (1.0 - x) * native_docs_in_segment as f64 / n_avg.max(f64::MIN_POSITIVE);
        let budget = (self.max_probe_fraction as f64 * segment_units)
            .max(self.min_probe_clusters as f64)
            .min(segment_units);
        Ok((budget, n_avg, x))
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeParams;

    #[test]
    fn normalized_budget_preserves_controls_and_reduces_filtered_budget() {
        let mut params = ProbeParams::default();
        assert_eq!(params.normalized_budget(20.0, 0.2, 0.1), 20.0);
        params.selectivity_exponent = 1.0;
        assert_eq!(params.normalized_budget(20.0, 0.2, 1.0), 20.0);
        assert!((params.normalized_budget(20.0, 0.2, 0.1) - 5.6).abs() < 1e-9);
        params.min_probe_clusters = 8;
        assert_eq!(params.normalized_budget(20.0, 0.2, 0.1), 8.0);
        params.max_probe_fraction = 1.0;
        assert_eq!(params.normalized_budget(20.0, 0.2, 0.1), 20.0);
        params.selectivity_exponent = f64::NAN;
        assert!(params.resolved_work_budget(100, 1000).is_err());
    }

    #[test]
    fn normalized_budget_is_monotone_and_preserves_unfiltered_capacity() {
        for (intercept, slope) in [(-0.86, -0.73), (0.8, 0.0), (-2.0, -1.0), (2.0, -1.0)] {
            let mut params = ProbeParams {
                selectivity_exponent: intercept,
                selectivity_slope: slope,
                ..Default::default()
            };
            for density in [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0] {
                let mut previous = 0.0;
                for step in 1..=1000 {
                    params.max_probe_fraction = step as f32 / 1000.0;
                    let budget = f64::from(params.max_probe_fraction) * 100.0;
                    let normalized = params.normalized_budget(budget, 0.004, density);
                    assert!(normalized >= previous && normalized <= budget);
                    assert_eq!(params.normalized_budget(budget, 0.004, 1.0), budget);
                    previous = normalized;
                }
                assert_eq!(previous, 100.0);
            }
        }
        for slope in [f64::NAN, f64::INFINITY, -1.01, 0.01] {
            let params = ProbeParams {
                selectivity_slope: slope,
                ..Default::default()
            };
            assert!(params.resolved_work_budget(100, 1000).is_err());
        }
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

    /// The fraction scales the capacity without a default floor, including sub-cluster budgets.
    #[test]
    fn budget_resolves_against_capacity() -> crate::Result<()> {
        let params = ProbeParams {
            max_probe_fraction: 0.25,
            ..Default::default()
        };
        let (budget, _, _) = params.resolved_work_budget(1000, 20_000)?;
        assert!((budget - 250.0).abs() < 1e-6, "{budget}");
        let (budget, _, _) = params.resolved_work_budget(40, 800)?;
        assert!((budget - 10.0).abs() < 1e-6, "{budget}");
        let (budget, _, _) = params.resolved_work_budget(2, 40)?;
        assert!((budget - 0.5).abs() < 1e-6, "{budget}");
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
    /// filter stretches it toward `budget / open_share`, capped at the segment's clusters.
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
        assert_eq!(params.router_k(2.0, x, 1.0, 1000), 2);
        assert_eq!(params.router_k(0.01, x, 1.0, 1000), 1);
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

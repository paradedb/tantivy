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
    /// Stop probing when competitive-candidate admission has decayed.
    pub adaptive_probing: Option<AdaptiveProbeParams>,
    /// Global work ceiling in average-cluster units. Default 100.
    pub max_probe: usize,
    /// Approximate members retained per requested cluster by the quantized router.
    /// Fractional values are rounded up to whole candidates and clamped to at
    /// least 1.0. A factor of 1.0 still reranks exactly; it only removes padding.
    /// Does not widen list selection. Default 1.0.
    pub router_rerank_factor: f32,
    /// Minimum fraction of each quantized router level's lists to open.
    /// Clamped to [0, 1]; non-finite values use 0.25. Full-precision routing
    /// keeps its existing beam. Default 0.5.
    pub router_min_beam_fraction: f32,
    /// Whether stacked routing may stop after lists cease improving its heap.
    /// Quantized routing observes its approximate padded heap. Default true.
    pub router_aps: bool,
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

/// Admission-rate stopping within the fixed routing and work budgets.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveProbeParams {
    pub admission_fraction: f64,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            admission_fraction: 0.0001,
        }
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
            adaptive_probing: None,
            max_probe: 100,
            router_rerank_factor: 1.0,
            router_min_beam_fraction: 0.5,
            router_aps: true,
            min_probe_clusters: 0,
            exact_plan: true,
            located: LocatedPath::Auto,
            fixed_probe_cost_rows: DEFAULT_FIXED_PROBE_COST_ROWS,
        }
    }
}

impl ProbeParams {
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
        let match_fraction = match_fraction.clamp(0.0, 1.0);
        let cluster_cost =
            (open_share + (1.0 - open_share) * match_fraction).max(f64::MIN_POSITIVE);
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
    /// The absolute `max_probe` budget is floored at
    /// `min_probe_clusters` and capped at the capacity; `n_avg` is the docs per cluster and `x` the
    /// open share. `native_docs_in_segment` counts distinct docs with a vector, as written. A
    /// zero `max_probe` or a non-finite or non-positive `fixed_probe_cost_rows` is an error.
    pub(crate) fn resolved_work_budget(
        &self,
        clusters_in_segment: usize,
        native_docs_in_segment: usize,
    ) -> crate::Result<(f64, f64, f64)> {
        if self.adaptive_probing.is_some_and(|p| {
            !p.admission_fraction.is_finite() || !(0.0..=1.0).contains(&p.admission_fraction)
        }) {
            return Err(crate::TantivyError::InvalidArgument(
                "adaptive admission fraction must be finite and between zero and one".into(),
            ));
        }
        if self.max_probe == 0 {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe must be greater than 0".to_string(),
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
        let budget = (self.max_probe.max(self.min_probe_clusters) as f64).min(segment_units);
        Ok((budget, n_avg, x))
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeParams;

    #[test]
    fn adaptive_stopping_preserves_budget_and_routing() -> crate::Result<()> {
        for budget in [1, 20, 250, 1000] {
            for min_probe_clusters in [0, 10] {
                let fixed = ProbeParams {
                    max_probe: budget,
                    min_probe_clusters,
                    ..Default::default()
                };
                let adaptive = ProbeParams {
                    adaptive_probing: Some(super::AdaptiveProbeParams::default()),
                    ..fixed.clone()
                };
                for (clusters, rows) in [(2, 40), (1000, 20_000)] {
                    let budget = fixed.resolved_work_budget(clusters, rows)?;
                    assert_eq!(budget, adaptive.resolved_work_budget(clusters, rows)?);
                    for selectivity in [0.0, 0.001, 0.01, 0.1, 1.0] {
                        assert_eq!(
                            fixed.router_k(budget.0, budget.2, selectivity, clusters),
                            adaptive.router_k(budget.0, budget.2, selectivity, clusters),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Exhaustive work is the cluster count at every granularity.
    #[test]
    fn capacity_is_the_cluster_count() -> crate::Result<()> {
        let all = ProbeParams {
            max_probe: usize::MAX,
            min_probe_clusters: 1,
            ..Default::default()
        };
        for (clusters, docs) in [(1000usize, 20_000usize), (9, 20), (2, 2), (64, 6400)] {
            let (budget, _n_avg, _x) = all.resolved_work_budget(clusters, docs)?;
            assert!(
                (budget - clusters as f64).abs() <= 1e-9 * clusters as f64,
                "an exhaustive budget must buy exactly C units at C={clusters}, N={docs}: {budget}"
            );
        }
        Ok(())
    }

    #[test]
    fn budget_is_absolute_and_capped_by_capacity() -> crate::Result<()> {
        let params = ProbeParams {
            max_probe: 10,
            ..Default::default()
        };
        for (clusters, rows, expected) in [(1000, 20_000, 10.0), (40, 800, 10.0), (2, 40, 2.0)] {
            let (budget, _, _) = params.resolved_work_budget(clusters, rows)?;
            assert!((budget - expected).abs() < 1e-9);
        }
        let minimum = ProbeParams {
            max_probe: 1,
            ..params
        };
        assert_eq!(minimum.resolved_work_budget(1000, 20_000)?.0, 1.0);
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

    /// Budgets and open costs must be finite and positive.
    #[test]
    fn invalid_budget_params_error() {
        let params = ProbeParams {
            max_probe: 0,
            ..Default::default()
        };
        assert!(params.resolved_work_budget(9, 180).is_err());
        for cost in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let params = ProbeParams {
                fixed_probe_cost_rows: cost,
                ..Default::default()
            };
            assert!(params.resolved_work_budget(9, 180).is_err(), "{cost}");
        }
    }
}

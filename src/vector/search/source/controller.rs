//! The probe budget and the stop: work units, their pricing, the RateAPS and the probe
//! controller.
//!
//! One unit is one average cluster of work, `n_avg = N / C` across the index's IVF segments. An
//! open charges `x` and a scored row `(1 - x) / n_avg`, so an exhaustive, unfiltered, delete-free
//! scan charges exactly `C`. The budget is checked only between clusters.

use crate::vector::search::scan::rate::RateReading;

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The work budget was spent.
    Ceiling,
    /// The router ran out of clusters before the budget was spent.
    #[default]
    Exhausted,
    /// The RateAPS: the admission rate under the pruning threshold fell to its fraction of the
    /// opening rate.
    RateDecayed,
    /// The RateAPS: no row of the opening window reaches the pruning threshold.
    OpeningEmpty,
}

// --- RateAPS ---

/// The RateAPS: stops a routed scan once the admission rate has fallen to `fraction` of its
/// opening level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateAps {
    fraction: f32,
}

impl RateAps {
    /// Expected admissions below which a zero in the recent window proves nothing.
    ///
    /// Five expected admissions make an empty recent window a 0.7% event under no decay.
    pub(crate) const MIN_EVIDENCE: f32 = 5.0;

    /// The RateAPS for `recall_target`: `f = 1 - target`, clamped to `[1/256, 1/2]`.
    pub(crate) fn from_target(recall_target: f32) -> Self {
        Self {
            fraction: (1.0 - recall_target).clamp(1.0 / 256.0, 0.5),
        }
    }

    /// Fewer than one opening row above the bar ends the segment; too little evidence continues;
    /// otherwise the rate has decayed once the current share is at most `f` of the opening's.
    pub(crate) fn verdict(&self, reading: RateReading) -> RateVerdict {
        if reading.opening_rows < 1.0 {
            RateVerdict::OpeningEmpty
        } else if reading.evidence < Self::MIN_EVIDENCE {
            RateVerdict::Continue
        } else if reading.current <= self.fraction * reading.opening {
            RateVerdict::Decayed
        } else {
            RateVerdict::Continue
        }
    }
}

/// What the RateAPS makes of a reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RateVerdict {
    Continue,
    Decayed,
    OpeningEmpty,
}

// --- Work units ---

/// The open share `x = cost / (cost + n_avg)` for a cluster-open cost of `fixed_cost_rows` rows
/// and `n_avg` native docs per cluster.
///
/// Clamped to at most one half: a larger share means opens outweigh rows, which only clusters of
/// under two rows produce.
pub(crate) fn open_share(fixed_cost_rows: f64, n_avg: f64) -> f64 {
    (fixed_cost_rows / (fixed_cost_rows + n_avg.max(0.0))).min(0.5)
}

/// Probe work in average-cluster units; budgets, prices and spends compose only with each other.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub(crate) struct WorkUnits(f64);

impl WorkUnits {
    /// No work.
    pub(crate) const ZERO: WorkUnits = WorkUnits(0.0);

    /// Wraps an amount in work units.
    #[inline]
    pub(crate) fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// Returns the binary64 amount.
    #[inline]
    pub(crate) fn get(self) -> f64 {
        self.0
    }

    /// Returns the amount narrowed to binary32, for the probe statistics.
    #[inline]
    pub(crate) fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl std::ops::Add for WorkUnits {
    type Output = WorkUnits;
    #[inline]
    fn add(self, rhs: WorkUnits) -> WorkUnits {
        WorkUnits(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for WorkUnits {
    #[inline]
    fn add_assign(&mut self, rhs: WorkUnits) {
        self.0 += rhs.0;
    }
}

impl std::ops::Mul<f64> for WorkUnits {
    type Output = WorkUnits;
    /// Scales work by a count.
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// A segment's work budget and what a cluster open and a scored row cost against it, from
/// [`ProbeParams::resolved_work_budget`](crate::vector::params::ProbeParams::resolved_work_budget).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkBudget {
    /// Work the segment may spend before the ceiling binds.
    pub(crate) total: WorkUnits,
    /// The per-index open share `x`.
    pub(crate) per_cluster: WorkUnits,
    /// `(1 - x) / n_avg`.
    pub(crate) per_row: WorkUnits,
}

/// Charges cluster opens and scored rows against a budget.
///
/// Only scored rows charge row work, and a cluster is never cut short, so the charge overshoots
/// the budget by at most the last cluster's.
pub(crate) struct WorkMeter {
    budget: WorkBudget,
    charged: WorkUnits,
}

impl WorkMeter {
    pub(crate) fn new(budget: WorkBudget) -> Self {
        Self {
            budget,
            charged: WorkUnits::ZERO,
        }
    }

    /// Charges one cluster open.
    pub(crate) fn open(&mut self) {
        self.charged += self.budget.per_cluster;
    }

    /// Charges `rows` scored rows.
    pub(crate) fn rows(&mut self, rows: usize) {
        self.charged += self.budget.per_row * rows as f64;
    }

    /// Whether the charges have reached the budget.
    pub(crate) fn exhausted(&self) -> bool {
        self.charged >= self.budget.total
    }

    /// The work charged so far.
    pub(crate) fn charged(&self) -> WorkUnits {
        self.charged
    }

    pub(crate) fn budget(&self) -> WorkBudget {
        self.budget
    }
}

// --- Controller ---

/// Applies the RateAPS and decides, from the work meter, whether the next ranked cluster opens.
pub(in crate::vector) struct ProbeController {
    termination: ProbeTermination,
    /// Clusters in the segment and clusters pulled from the ranking so far.
    clusters: usize,
    pulled: usize,
    /// The RateAPS; `None` at a target of `1.0`.
    aps: Option<RateAps>,
    /// The last admission-rate reading the RateAPS judged, and its verdict.
    reading: Option<RateReading>,
    verdict: RateVerdict,
}

impl ProbeController {
    pub(in crate::vector) fn new(clusters: usize, aps: Option<RateAps>) -> Self {
        Self {
            termination: ProbeTermination::Exhausted,
            clusters,
            pulled: 0,
            aps,
            reading: None,
            verdict: RateVerdict::Continue,
        }
    }

    /// Whether the cluster just pulled from the ranking may open: not past the RateAPS's stop or
    /// the ceiling, in that order. `rate` is the scan's reading after the clusters before it.
    ///
    /// Checked after the pull, so a stop proves another cluster existed.
    pub(in crate::vector) fn admit(
        &mut self,
        meter: &WorkMeter,
        rate: Option<RateReading>,
    ) -> bool {
        self.pulled += 1;
        if let (Some(aps), Some(reading)) = (self.aps, rate) {
            self.reading = Some(reading);
            self.verdict = aps.verdict(reading);
        }
        let stop = match self.verdict {
            RateVerdict::Continue => None,
            RateVerdict::Decayed => Some(ProbeTermination::RateDecayed),
            RateVerdict::OpeningEmpty => Some(ProbeTermination::OpeningEmpty),
        };
        if let Some(termination) = stop {
            self.termination = termination;
            return false;
        }
        if meter.exhausted() {
            self.termination = ProbeTermination::Ceiling;
            return false;
        }
        true
    }

    /// How the probe loop ended and what `meter` charged.
    pub(crate) fn finish(&self, meter: &WorkMeter) -> ProbeOutcome {
        // A stacked ranking sized to the budget can run out as the budget is spent, short of the
        // segment's clusters; that stop is the ceiling's.
        let termination = match self.termination {
            ProbeTermination::Exhausted if self.pulled < self.clusters && meter.exhausted() => {
                ProbeTermination::Ceiling
            }
            termination => termination,
        };
        ProbeOutcome {
            termination,
            work_charged: meter.charged().to_f32(),
            work_budget: meter.budget().total.to_f32(),
            rate: self.reading,
        }
    }
}

/// How a routed probe loop ended.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeOutcome {
    pub(crate) termination: ProbeTermination,
    /// Work units charged: opens at `x`, scored rows at `(1 - x)/n_avg`.
    pub(crate) work_charged: f32,
    /// The resolved budget the loop ran against, in work units.
    pub(crate) work_budget: f32,
    /// The last admission-rate reading the RateAPS judged; `None` with it off or before the
    /// opening existed.
    pub(crate) rate: Option<RateReading>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::search::scan::rate::OpeningSource;

    fn reading(opening_rows: f32, opening: f32, current: f32, evidence: f32) -> RateReading {
        RateReading {
            opening,
            opening_rows,
            current,
            evidence,
            opening_from: OpeningSource::Own,
        }
    }

    /// Under one opening row the segment ends whatever else holds; under five expected admissions
    /// the RateAPS continues; past both it stops once the current share is at most `f` of the
    /// opening's.
    #[test]
    fn rate_verdicts() {
        use RateVerdict::{Continue, Decayed, OpeningEmpty};
        // f = 1/4.
        let aps = RateAps::from_target(0.75);
        let cases = [
            ((0.0, 0.0, 0.0, 0.0), OpeningEmpty),
            ((0.5, 0.01, 0.0, 100.0), OpeningEmpty),
            ((0.5, 0.01, 1.0, 100.0), OpeningEmpty),
            ((1.0, 0.5, 0.0, 4.99), Continue),
            ((40.0, 0.5, 0.0, 4.99), Continue),
            ((1.0, 0.5, 0.0, 5.0), Decayed),
            ((10.0, 0.5, 0.125, 50.0), Decayed),
            ((10.0, 0.5, 0.126, 50.0), Continue),
            ((10.0, 0.5, 0.5, 50.0), Continue),
            ((10.0, 0.5, 0.75, 50.0), Continue),
        ];
        for ((opening_rows, opening, current, evidence), verdict) in cases {
            let reading = reading(opening_rows, opening, current, evidence);
            assert_eq!(aps.verdict(reading), verdict, "{reading:?}");
        }
    }

    /// `f = 1 - target`, clamped to `[1/256, 1/2]`.
    #[test]
    fn rate_fraction_from_target() {
        for (target, fraction) in [
            (0.0, 0.5),
            (0.4, 0.5),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.9, 1.0 - 0.9f32),
            (0.999, 1.0 / 256.0),
            (1.0, 1.0 / 256.0),
        ] {
            assert_eq!(RateAps::from_target(target).fraction, fraction, "{target}");
        }
    }
}

//! The probe budget: work units, their pricing and the probe controller.
//!
//! One unit is one average cluster of work, `n_avg = N / C` across the index's IVF segments. An
//! open charges `x` and a scored row `(1 - x) / n_avg`, so an exhaustive, unfiltered, delete-free
//! scan charges exactly `C`. The budget is checked only between clusters.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ADAPTIVE_PATIENCE_START_WORK: f64 = 500.0;
const ADAPTIVE_PATIENCE_END_WORK: f64 = 2000.0;
const ADAPTIVE_PATIENCE_MAX_FRACTION: f64 = 0.10;

/// Actual routed work charged by all participants in one query. Located scans are separate.
pub trait SharedVectorWork: Send + Sync {
    /// Adds finite, nonnegative work. Implementations must not lose concurrent charges.
    fn charge(&self, work: f64);
    /// Total work charged by every participant.
    fn charged(&self) -> f64;
}

/// A query work counter that can live in an `Arc` or process-shared memory.
#[derive(Default, Debug)]
#[repr(transparent)]
pub struct AtomicVectorWork(AtomicU64);

impl SharedVectorWork for AtomicVectorWork {
    fn charge(&self, work: f64) {
        assert!(work.is_finite() && work >= 0.0);
        let _ = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some((f64::from_bits(bits) + work).to_bits())
            });
    }

    fn charged(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The adaptive estimate reached the requested recall.
    QuantizedRecall,
    /// The work budget was spent.
    Ceiling,
    /// The router ran out of clusters before the budget was spent.
    #[default]
    Exhausted,
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
#[derive(
    Clone, Copy, PartialEq, PartialOrd, Debug, Default, serde::Serialize, serde::Deserialize,
)]
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
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
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
    last_active_work: WorkUnits,
    shared: Option<Arc<dyn SharedVectorWork>>,
}

impl WorkMeter {
    pub(crate) fn new(budget: WorkBudget, shared: Option<Arc<dyn SharedVectorWork>>) -> Self {
        Self {
            budget,
            charged: WorkUnits::ZERO,
            last_active_work: WorkUnits::ZERO,
            shared,
        }
    }

    /// Charges one cluster open.
    pub(crate) fn open(&mut self) {
        self.charged += self.budget.per_cluster;
        if let Some(shared) = &self.shared {
            shared.charge(self.budget.per_cluster.get());
        }
    }

    /// Charges `rows` scored rows.
    pub(crate) fn rows(&mut self, rows: usize) {
        let work = self.budget.per_row * rows as f64;
        self.charged += work;
        if let Some(shared) = &self.shared {
            shared.charge(work.get());
        }
    }

    /// Whether the charges have reached the budget.
    pub(crate) fn exhausted(&self) -> bool {
        self.shared
            .as_ref()
            .map_or(self.charged.get(), |shared| shared.charged())
            >= self.budget.total.get()
    }

    /// Ramps patience from 0% at 500 work units to 10% at 2,000.
    pub(crate) fn adaptive_stop(&mut self, quiet: bool) -> bool {
        if !quiet {
            self.last_active_work = self.charged;
            return false;
        }
        let budget = self.budget.total.get();
        let ramp = (budget - ADAPTIVE_PATIENCE_START_WORK)
            / (ADAPTIVE_PATIENCE_END_WORK - ADAPTIVE_PATIENCE_START_WORK);
        let fraction = ADAPTIVE_PATIENCE_MAX_FRACTION * ramp.clamp(0.0, 1.0);
        self.charged.get() - self.last_active_work.get() >= fraction * budget
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

/// Decides, from the work meter, whether the next ranked cluster opens.
pub(in crate::vector) struct ProbeController {
    termination: ProbeTermination,
    /// Clusters in the segment and clusters pulled from the ranking so far.
    clusters: usize,
    pulled: usize,
}

impl ProbeController {
    pub(in crate::vector) fn new(clusters: usize) -> Self {
        Self {
            termination: ProbeTermination::Exhausted,
            clusters,
            pulled: 0,
        }
    }

    /// Whether the cluster just pulled from the ranking may open.
    ///
    /// Checked after the pull, so a stop proves another cluster existed.
    pub(in crate::vector) fn admit(&mut self, meter: &WorkMeter) -> bool {
        self.pulled += 1;
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
}

#[cfg(test)]
mod shared_work_tests {
    use super::*;

    #[test]
    fn adaptive_patience_uses_work_and_resets_on_activity() {
        let mut meter = WorkMeter::new(
            WorkBudget {
                total: WorkUnits::new(1000.0),
                per_cluster: WorkUnits::new(0.25),
                per_row: WorkUnits::new(0.75),
            },
            None,
        );
        meter.open();
        meter.rows(1);
        assert!(!meter.adaptive_stop(false));
        for rows in [4, 8] {
            meter.open();
            meter.rows(rows);
            assert!(!meter.adaptive_stop(true));
        }
        for _ in 0..10 {
            assert!(!meter.adaptive_stop(true));
        }
        meter.open();
        meter.rows(32);
        assert!(meter.adaptive_stop(true));
        assert!(!meter.adaptive_stop(false));
        meter.open();
        meter.rows(1);
        assert!(!meter.adaptive_stop(true));
    }

    #[test]
    fn adaptive_patience_scales_with_budget_without_changing_ceiling() {
        for (budget, stop_at) in [
            (20, 0),
            (49, 0),
            (50, 0),
            (499, 0),
            (500, 0),
            (501, 1),
            (1000, 34),
            (1250, 63),
            (1500, 100),
            (1999, 200),
            (2000, 200),
            (2001, 201),
            (4000, 400),
        ] {
            let mut meter = WorkMeter::new(
                WorkBudget {
                    total: WorkUnits::new(budget as f64),
                    per_cluster: WorkUnits::new(0.5),
                    per_row: WorkUnits::new(0.5),
                },
                None,
            );
            assert_eq!(meter.adaptive_stop(true), stop_at == 0);
            for work in 1..=budget {
                meter.open();
                meter.rows(1);
                assert_eq!(meter.adaptive_stop(true), work >= stop_at);
                assert_eq!(meter.exhausted(), work == budget);
            }
        }
    }

    #[test]
    fn concurrent_meters_enforce_one_ceiling() {
        let work: Arc<dyn SharedVectorWork> = Arc::new(AtomicVectorWork::default());
        let barrier = std::sync::Barrier::new(8);
        let charges = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    let work = Arc::clone(&work);
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let mut meter = WorkMeter::new(
                            WorkBudget {
                                total: WorkUnits::new(20.0),
                                per_cluster: WorkUnits::new(0.25),
                                per_row: WorkUnits::new(0.075),
                            },
                            Some(work),
                        );
                        barrier.wait();
                        while !meter.exhausted() {
                            meter.open();
                            meter.rows(10);
                        }
                        meter.charged().get()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .sum::<f64>()
        });
        assert_eq!(work.charged(), charges);
        assert!((20.0..=27.0).contains(&charges));
    }
}

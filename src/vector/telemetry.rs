//! The vector-search stage clock: which stage is running on this thread, and the exclusive time,
//! storage reads and sign-word fallbacks each stage accrues until the next take.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::time::Instant;

use crate::vector::quantization::MAX_QUANTIZATION_LAYERS;
use crate::vector::storage::io::VectorIoStats;

/// Logical stage of a vector search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// Unclassified work.
    #[default]
    Other,
    /// Reader initialization.
    ScanInit,
    /// Predicate evaluation and filter-bitset construction.
    NonVectorSearch,
    /// Query preparation.
    QueryPrep,
    /// Cluster routing.
    Routing,
    /// Quantized layer scanning.
    LayerScan(u8),
    /// Quantized layer boundary selection.
    Boundary(u8),
    /// Full-precision admission and the flat path's scan.
    ExactScan,
    /// Full-precision row and DocIds reads for exact scoring.
    ExactFetch,
    /// Exact scoring into the bound and the result heap.
    ExactScore,
    /// Locating filter matches through the document location map.
    Locate,
    /// Reading centroid rows for clusters whose centroid score is unknown.
    Centroid,
    /// Result assembly.
    ResultAssembly,
}

impl Stage {
    /// Returns the flat telemetry field prefix for this stage.
    pub fn name(self) -> Option<Cow<'static, str>> {
        match self {
            Self::Other => None,
            Self::ScanInit => Some(Cow::Borrowed("scan_init")),
            Self::NonVectorSearch => Some(Cow::Borrowed("non_vector_search")),
            Self::QueryPrep => Some(Cow::Borrowed("query_prep")),
            Self::Routing => Some(Cow::Borrowed("routing")),
            Self::LayerScan(layer) => Some(Cow::Owned(format!("layer{layer}_scan"))),
            Self::Boundary(layer) => Some(Cow::Owned(format!("boundary{layer}"))),
            Self::ExactScan => Some(Cow::Borrowed("exact_scan")),
            Self::ExactFetch => Some(Cow::Borrowed("exact_fetch")),
            Self::ExactScore => Some(Cow::Borrowed("exact_score")),
            Self::Locate => Some(Cow::Borrowed("locate")),
            Self::Centroid => Some(Cow::Borrowed("centroid")),
            Self::ResultAssembly => Some(Cow::Borrowed("result_assembly")),
        }
    }
}

// --- Totals ---

const FIXED_SLOTS: usize = 11;
const STAGE_SLOTS: usize = FIXED_SLOTS + 2 * MAX_QUANTIZATION_LAYERS;
const FALLBACK_LAYERS: usize = 3;

/// Accounting slot of `stage`; layers past the schedule limit fold into `Other`.
fn slot(stage: Stage) -> usize {
    let layer = |base: usize, layer: u8| {
        let layer = usize::from(layer);
        if layer < MAX_QUANTIZATION_LAYERS {
            base + layer
        } else {
            0
        }
    };
    match stage {
        Stage::Other => 0,
        Stage::ScanInit => 1,
        Stage::NonVectorSearch => 2,
        Stage::QueryPrep => 3,
        Stage::Routing => 4,
        Stage::ExactScan => 5,
        Stage::ExactFetch => 6,
        Stage::ExactScore => 7,
        Stage::ResultAssembly => 8,
        Stage::Locate => 9,
        Stage::Centroid => 10,
        Stage::LayerScan(l) => layer(FIXED_SLOTS, l),
        Stage::Boundary(l) => layer(FIXED_SLOTS + MAX_QUANTIZATION_LAYERS, l),
    }
}

/// Exclusive time, storage reads and sign-word fallbacks per stage since the last take.
#[derive(Clone, Debug, Default)]
pub(crate) struct StageTotals {
    time_ns: [u64; STAGE_SLOTS],
    io: [VectorIoStats; STAGE_SLOTS],
    sign_word_fallbacks: [u64; FALLBACK_LAYERS],
}

impl StageTotals {
    /// Nanoseconds spent in `stage` with no inner stage running.
    pub(crate) fn time_ns(&self, stage: Stage) -> u64 {
        self.time_ns[slot(stage)]
    }

    /// Reads charged while `stage` was running.
    pub(crate) fn io(&self, stage: Stage) -> VectorIoStats {
        self.io[slot(stage)]
    }

    /// Owned-word decoding fallbacks in layer `layer`'s sign-plane scoring.
    pub(crate) fn sign_word_fallbacks(&self, layer: usize) -> u64 {
        self.sign_word_fallbacks.get(layer).copied().unwrap_or(0)
    }
}

// --- Clock ---

thread_local! {
    static CURRENT: Cell<Stage> = const { Cell::new(Stage::Other) };
    static CLOCK: RefCell<Clock> = RefCell::new(Clock::new());
}

struct Clock {
    since: Instant,
    totals: StageTotals,
    fallbacks_at_take: [u64; FALLBACK_LAYERS],
}

impl Clock {
    fn new() -> Self {
        Self {
            since: Instant::now(),
            totals: StageTotals::default(),
            fallbacks_at_take: cascade::sign_word_fallback_counts(),
        }
    }

    /// Charges the time since the last transition to `running`.
    fn charge_time(&mut self, running: Stage) {
        let now = Instant::now();
        self.totals.time_ns[slot(running)] += now.duration_since(self.since).as_nanos() as u64;
        self.since = now;
    }
}

/// Moves the clock to `next` and returns the stage that was running.
fn switch(next: Stage) -> Stage {
    let running = CURRENT.replace(next);
    CLOCK.with_borrow_mut(|clock| clock.charge_time(running));
    running
}

/// The logical stage responsible for a vector component read on this thread.
pub fn current_vector_stage() -> Stage {
    CURRENT.get()
}

/// Restores the enclosing stage when dropped.
pub(crate) struct StageGuard(Stage);

/// Enters `stage` until the guard drops. Time accrues only to the innermost stage.
pub(crate) fn enter_vector_stage(stage: Stage) -> StageGuard {
    StageGuard(switch(stage))
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        switch(self.0);
    }
}

/// Charges one storage read of `bytes` touching `storage_blocks` blocks to the running stage.
pub(crate) fn charge_read(bytes: u64, storage_blocks: u64) {
    let stage = CURRENT.get();
    CLOCK.with_borrow_mut(|clock| {
        let io = &mut clock.totals.io[slot(stage)];
        io.reads += 1;
        io.bytes_read += bytes;
        io.storage_blocks += storage_blocks;
    });
}

/// Returns the totals accrued since the last take and resets them.
pub(crate) fn take_stage_totals() -> StageTotals {
    let running = CURRENT.get();
    CLOCK.with_borrow_mut(|clock| {
        clock.charge_time(running);
        let fallbacks = cascade::sign_word_fallback_counts();
        let mut totals = std::mem::take(&mut clock.totals);
        for (layer, count) in totals.sign_word_fallbacks.iter_mut().enumerate() {
            *count = fallbacks[layer] - clock.fallbacks_at_take[layer];
        }
        clock.fallbacks_at_take = fallbacks;
        totals
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_names_are_flat_telemetry_prefixes() {
        assert_eq!(Stage::Other.name(), None);
        assert_eq!(Stage::ScanInit.name().as_deref(), Some("scan_init"));
        assert_eq!(
            Stage::NonVectorSearch.name().as_deref(),
            Some("non_vector_search")
        );
        assert_eq!(Stage::QueryPrep.name().as_deref(), Some("query_prep"));
        assert_eq!(Stage::Routing.name().as_deref(), Some("routing"));
        assert_eq!(Stage::LayerScan(2).name().as_deref(), Some("layer2_scan"));
        assert_eq!(Stage::Boundary(2).name().as_deref(), Some("boundary2"));
        assert_eq!(Stage::ExactScan.name().as_deref(), Some("exact_scan"));
        assert_eq!(Stage::ExactFetch.name().as_deref(), Some("exact_fetch"));
        assert_eq!(Stage::ExactScore.name().as_deref(), Some("exact_score"));
        assert_eq!(Stage::Locate.name().as_deref(), Some("locate"));
        assert_eq!(Stage::Centroid.name().as_deref(), Some("centroid"));
        assert_eq!(
            Stage::ResultAssembly.name().as_deref(),
            Some("result_assembly")
        );
    }

    /// Time and reads go to the innermost stage; leaving a stage resumes the enclosing one.
    #[test]
    fn clock_charges_the_innermost_stage() {
        take_stage_totals();
        {
            let _outer = enter_vector_stage(Stage::ScanInit);
            charge_read(10, 1);
            {
                let _inner = enter_vector_stage(Stage::NonVectorSearch);
                assert_eq!(current_vector_stage(), Stage::NonVectorSearch);
                charge_read(20, 2);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert_eq!(current_vector_stage(), Stage::ScanInit);
        }
        assert_eq!(current_vector_stage(), Stage::Other);
        let totals = take_stage_totals();
        assert_eq!(totals.io(Stage::ScanInit).bytes_read, 10);
        assert_eq!(totals.io(Stage::NonVectorSearch).storage_blocks, 2);
        assert!(totals.time_ns(Stage::NonVectorSearch) >= 2_000_000);
        assert_eq!(take_stage_totals().io(Stage::ScanInit).reads, 0);
    }
}

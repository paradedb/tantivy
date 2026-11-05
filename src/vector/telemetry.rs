//! Vector-search stage telemetry, owned by the operation it measures: [`measure`] puts a meter in
//! scope on this thread, and stage switches and reads charge it and nothing outside one. The scope
//! is per thread because storage reads charge it without a parameter, and embedders read
//! [`current_vector_stage`] beneath the `Directory` API, which carries no context. Another thread
//! charges nothing here: a worker runs its own `measure` and the parent adds the returned totals.
//! Reads outside a `measure` aren't counted; to count an operation, wrap it in `measure`.

use std::borrow::Cow;
use std::cell::RefCell;
use std::time::Instant;

use crate::vector::quantization::MAX_QUANTIZATION_LAYERS;
use crate::vector::storage::io::VectorIoStats;

/// Logical stage of a vector search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VectorStage {
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

impl VectorStage {
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
fn slot(stage: VectorStage) -> usize {
    let layer = |base: usize, layer: u8| {
        let layer = usize::from(layer);
        if layer < MAX_QUANTIZATION_LAYERS {
            base + layer
        } else {
            0
        }
    };
    match stage {
        VectorStage::Other => 0,
        VectorStage::ScanInit => 1,
        VectorStage::NonVectorSearch => 2,
        VectorStage::QueryPrep => 3,
        VectorStage::Routing => 4,
        VectorStage::ExactScan => 5,
        VectorStage::ExactFetch => 6,
        VectorStage::ExactScore => 7,
        VectorStage::ResultAssembly => 8,
        VectorStage::Locate => 9,
        VectorStage::Centroid => 10,
        VectorStage::LayerScan(l) => layer(FIXED_SLOTS, l),
        VectorStage::Boundary(l) => layer(FIXED_SLOTS + MAX_QUANTIZATION_LAYERS, l),
    }
}

/// Exclusive time and storage reads per stage, and sign-word fallbacks per layer, of one measured
/// operation.
#[derive(Clone, Debug, Default)]
pub(crate) struct StageTotals {
    time_ns: [u64; STAGE_SLOTS],
    io: [VectorIoStats; STAGE_SLOTS],
    sign_word_fallbacks: [u64; FALLBACK_LAYERS],
}

impl StageTotals {
    pub(crate) fn append(&mut self, other: &Self) {
        for (total, added) in self.time_ns.iter_mut().zip(other.time_ns) {
            *total += added;
        }
        for (total, added) in self.io.iter_mut().zip(other.io) {
            total.reads += added.reads;
            total.bytes_read += added.bytes_read;
            total.storage_blocks += added.storage_blocks;
        }
        for (total, added) in self
            .sign_word_fallbacks
            .iter_mut()
            .zip(other.sign_word_fallbacks)
        {
            *total += added;
        }
    }

    /// Nanoseconds spent in `stage` with no inner stage running.
    pub(crate) fn time_ns(&self, stage: VectorStage) -> u64 {
        self.time_ns[slot(stage)]
    }

    /// Reads charged while `stage` was running.
    pub(crate) fn io(&self, stage: VectorStage) -> VectorIoStats {
        self.io[slot(stage)]
    }

    /// Owned-word decoding fallbacks in layer `layer`'s sign-plane scoring.
    pub(crate) fn sign_word_fallbacks(&self, layer: usize) -> u64 {
        self.sign_word_fallbacks.get(layer).copied().unwrap_or(0)
    }
}

// --- Meter ---

thread_local! {
    static SCOPE: RefCell<Option<Meter>> = const { RefCell::new(None) };
}

/// One operation's telemetry: the running stage, time and reads per stage, and sign-word fallbacks
/// per layer. It exists only inside [`measure`], which owns it, so there is nothing to reset.
struct Meter {
    running: VectorStage,
    since: Instant,
    totals: StageTotals,
}

impl Meter {
    fn new() -> Self {
        Self {
            running: VectorStage::Other,
            since: Instant::now(),
            totals: StageTotals::default(),
        }
    }

    /// Charges the time since the last transition to the running stage.
    fn charge_time(&mut self) {
        let now = Instant::now();
        self.totals.time_ns[slot(self.running)] += now.duration_since(self.since).as_nanos() as u64;
        self.since = now;
    }

    /// Moves the meter to `next` and returns the stage that was running.
    fn switch(&mut self, next: VectorStage) -> VectorStage {
        self.charge_time();
        std::mem::replace(&mut self.running, next)
    }

    /// Charges the final time and returns the totals.
    fn finish(mut self) -> StageTotals {
        self.charge_time();
        self.totals
    }
}

/// The meter a [`measure`] shadows, put back in scope when dropped, unwinding included.
struct Enclosing(Option<Meter>);

impl Drop for Enclosing {
    fn drop(&mut self) {
        SCOPE.set(self.0.take());
    }
}

/// Runs `op` with a fresh meter in scope on this thread; returns its result and the meter's
/// totals. Stage switches and reads inside `op` charge this meter only. A nested `measure`
/// shadows this one until it returns.
pub(crate) fn measure<R>(op: impl FnOnce() -> R) -> (R, StageTotals) {
    let enclosing = Enclosing(SCOPE.replace(Some(Meter::new())));
    let result = op();
    let totals = SCOPE.take().map(Meter::finish).unwrap_or_default();
    drop(enclosing);
    (result, totals)
}

/// The stage running on this thread, for embedders attributing their own IO: `Other` outside a
/// `measure`.
pub fn current_vector_stage() -> VectorStage {
    SCOPE
        .try_with(|scope| {
            scope
                .borrow()
                .as_ref()
                .map_or(VectorStage::Other, |meter| meter.running)
        })
        .unwrap_or_default()
}

/// Restores the enclosing stage when dropped; holds nothing outside a [`measure`].
pub(crate) struct StageGuard(Option<VectorStage>);

/// Enters `stage` until the guard drops; does nothing outside a [`measure`]. Time accrues only to
/// the innermost stage.
pub(crate) fn enter_vector_stage(stage: VectorStage) -> StageGuard {
    StageGuard(SCOPE.with_borrow_mut(|scope| scope.as_mut().map(|meter| meter.switch(stage))))
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        if let Some(enclosing) = self.0 {
            SCOPE.with_borrow_mut(|scope| {
                if let Some(meter) = scope {
                    meter.switch(enclosing);
                }
            });
        }
    }
}

/// Charges one storage read of `bytes` touching `storage_blocks` blocks to the running stage; does
/// nothing outside a [`measure`].
pub(crate) fn charge_read(bytes: u64, storage_blocks: u64) {
    SCOPE.with_borrow_mut(|scope| {
        if let Some(meter) = scope {
            let io = &mut meter.totals.io[slot(meter.running)];
            io.reads += 1;
            io.bytes_read += bytes;
            io.storage_blocks += storage_blocks;
        }
    });
}

/// Charges `count` sign-word fallbacks in `layer`; does nothing outside a [`measure`].
pub(crate) fn charge_fallbacks(layer: usize, count: u64) {
    // Aligned codes, the usual case, leave the thread-local untouched.
    if count == 0 {
        return;
    }
    SCOPE.with_borrow_mut(|scope| {
        if let Some(fallbacks) = scope
            .as_mut()
            .and_then(|meter| meter.totals.sign_word_fallbacks.get_mut(layer))
        {
            *fallbacks += count;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_names_are_flat_telemetry_prefixes() {
        assert_eq!(VectorStage::Other.name(), None);
        assert_eq!(VectorStage::ScanInit.name().as_deref(), Some("scan_init"));
        assert_eq!(
            VectorStage::NonVectorSearch.name().as_deref(),
            Some("non_vector_search")
        );
        assert_eq!(VectorStage::QueryPrep.name().as_deref(), Some("query_prep"));
        assert_eq!(VectorStage::Routing.name().as_deref(), Some("routing"));
        assert_eq!(
            VectorStage::LayerScan(2).name().as_deref(),
            Some("layer2_scan")
        );
        assert_eq!(
            VectorStage::Boundary(2).name().as_deref(),
            Some("boundary2")
        );
        assert_eq!(VectorStage::ExactScan.name().as_deref(), Some("exact_scan"));
        assert_eq!(
            VectorStage::ExactFetch.name().as_deref(),
            Some("exact_fetch")
        );
        assert_eq!(
            VectorStage::ExactScore.name().as_deref(),
            Some("exact_score")
        );
        assert_eq!(VectorStage::Locate.name().as_deref(), Some("locate"));
        assert_eq!(VectorStage::Centroid.name().as_deref(), Some("centroid"));
        assert_eq!(
            VectorStage::ResultAssembly.name().as_deref(),
            Some("result_assembly")
        );
    }

    /// Time and reads go to the innermost stage; leaving a stage resumes the enclosing one.
    #[test]
    fn meter_charges_the_innermost_stage() {
        let ((), totals) = measure(|| {
            let _outer = enter_vector_stage(VectorStage::ScanInit);
            charge_read(10, 1);
            {
                let _inner = enter_vector_stage(VectorStage::NonVectorSearch);
                assert_eq!(current_vector_stage(), VectorStage::NonVectorSearch);
                charge_read(20, 2);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert_eq!(current_vector_stage(), VectorStage::ScanInit);
        });
        assert_eq!(current_vector_stage(), VectorStage::Other);
        assert_eq!(totals.io(VectorStage::ScanInit).bytes_read, 10);
        assert_eq!(totals.io(VectorStage::NonVectorSearch).storage_blocks, 2);
        assert!(totals.time_ns(VectorStage::NonVectorSearch) >= 2_000_000);
        assert_eq!(measure(|| ()).1.io(VectorStage::ScanInit).reads, 0);
    }

    /// A nested measure starts at `Other`, returns only its own totals, and hands the enclosing
    /// meter back at the stage it left.
    #[test]
    fn nested_measure_returns_only_its_own_totals() {
        let ((inner, resumed), outer) = measure(|| {
            let _routing = enter_vector_stage(VectorStage::Routing);
            charge_read(10, 1);
            let ((), inner) = measure(|| {
                assert_eq!(current_vector_stage(), VectorStage::Other);
                let _locate = enter_vector_stage(VectorStage::Locate);
                charge_read(20, 2);
            });
            let resumed = current_vector_stage();
            charge_read(30, 3);
            (inner, resumed)
        });
        assert_eq!(resumed, VectorStage::Routing);
        assert_eq!(inner.io(VectorStage::Locate).bytes_read, 20);
        assert_eq!(inner.io(VectorStage::Routing).reads, 0);
        assert_eq!(outer.io(VectorStage::Routing).bytes_read, 40);
        assert_eq!(outer.io(VectorStage::Locate).reads, 0);
    }

    /// A panic inside a measure puts the enclosing meter back in scope at the stage it left.
    #[test]
    fn panic_inside_measure_restores_the_enclosing_meter() {
        let ((), totals) = measure(|| {
            let _routing = enter_vector_stage(VectorStage::Routing);
            let unwound = std::panic::catch_unwind(|| {
                measure(|| {
                    let _locate = enter_vector_stage(VectorStage::Locate);
                    charge_read(20, 2);
                    panic!("operation failed inside its measure");
                })
            });
            assert!(unwound.is_err());
            assert_eq!(current_vector_stage(), VectorStage::Routing);
            charge_read(10, 1);
        });
        assert_eq!(current_vector_stage(), VectorStage::Other);
        assert_eq!(totals.io(VectorStage::Routing).bytes_read, 10);
        assert_eq!(totals.io(VectorStage::Locate).reads, 0);
    }

    /// Outside a measure the stage stays `Other`, and neither stage switches nor reads carry into
    /// the next measure.
    #[test]
    fn outside_a_measure_nothing_is_charged() {
        let routing = enter_vector_stage(VectorStage::Routing);
        assert_eq!(current_vector_stage(), VectorStage::Other);
        charge_read(10, 1);
        let ((), totals) = measure(|| {
            assert_eq!(current_vector_stage(), VectorStage::Other);
        });
        drop(routing);
        assert_eq!(current_vector_stage(), VectorStage::Other);
        assert_eq!(totals.io(VectorStage::Other).reads, 0);
        assert_eq!(totals.io(VectorStage::Routing).reads, 0);
        assert_eq!(totals.time_ns(VectorStage::Routing), 0);
    }

    /// Fallbacks charge the meter in scope by layer, and nothing outside one.
    #[test]
    fn fallbacks_charge_the_meter_in_scope() {
        charge_fallbacks(1, 5);
        let ((), totals) = measure(|| {
            charge_fallbacks(1, 2);
            charge_fallbacks(2, 1);
        });
        let fallbacks: Vec<u64> = (0..FALLBACK_LAYERS)
            .map(|layer| totals.sign_word_fallbacks(layer))
            .collect();
        assert_eq!(fallbacks, [0, 2, 1]);
    }

    /// A thread without its own measure charges nothing to the parent's meter.
    #[test]
    fn spawned_thread_charges_nothing_to_the_parent() {
        let ((), totals) = measure(|| {
            let _routing = enter_vector_stage(VectorStage::Routing);
            std::thread::spawn(|| {
                assert_eq!(current_vector_stage(), VectorStage::Other);
                let _locate = enter_vector_stage(VectorStage::Locate);
                charge_read(10, 1);
            })
            .join()
            .unwrap();
        });
        assert_eq!(totals.io(VectorStage::Routing).reads, 0);
        assert_eq!(totals.io(VectorStage::Locate).reads, 0);
    }
}

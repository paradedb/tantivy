/// Query-time configuration for IVF adaptive probing.
///
/// SPANN-style probe loop: visit centroids in descending similarity order
/// and stop when (a) the current centroid score has fallen below a
/// geometric widening of the best (`epsilon`), AND (b) enough survivors
/// have been scored (`min_candidates`), AND (c) enough clusters have
/// been visited (`min_nprobe`). `max_nprobe` is the hard latency
/// ceiling — always wins.
///
/// All four knobs are provisional defaults pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Geometric widening factor used in the per-metric stopping
    /// threshold. Larger ⇒ probe further past the best centroid before
    /// the threshold gate trips. Combined with the other knobs via the
    /// AND-of-three stop condition.
    pub epsilon: f32,
    /// Absolute survivor floor. The call site widens this to
    /// `min_candidates.max(4 * top_n)`, so a 0 default still gives a
    /// sane `4 × top_n` floor.
    pub min_candidates: usize,
    /// Probe-count floor: keep probing until at least this many
    /// clusters have been visited, regardless of threshold. Matters
    /// when the candidate floor is satisfied trivially by one big
    /// cluster.
    pub min_nprobe: usize,
    /// Hard latency ceiling on probes. `usize::MAX` means no hard
    /// ceiling — this is the knob most likely to need a finite value
    /// after benchmarking, and the default should not quietly ship as
    /// the production ceiling.
    pub max_nprobe: usize,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        // PROVISIONAL — placeholders pending real-data benchmarking.
        // Conservative direction: epsilon wide enough to admit a
        // reasonable probe radius, candidate floor resolved at the
        // call site, no nprobe ceiling.
        Self {
            epsilon: 0.3,
            min_candidates: 0,
            min_nprobe: 1,
            max_nprobe: usize::MAX,
        }
    }
}

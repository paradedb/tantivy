//! Built-in [`VectorSearchStats`] recorders.
//!
//! [`VectorSearchCounters`] is the cheap default: lock-free atomic
//! totals suitable for production telemetry (Prometheus, tracing
//! fields). For per-window distributions, callers should implement
//! the [`VectorSearchStats`] trait directly against an HDR-histogram
//! or similar.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::collector::turboquant_collector::{
    VectorSearchStats, WindowOutcome, WindowStats,
};

#[derive(Default)]
pub struct VectorSearchCounters {
    windows_searched: AtomicU64,
    windows_filter_empty: AtomicU64,
    windows_centroids_too_far: AtomicU64,
    clusters_probed: AtomicU64,
    candidates_visited: AtomicU64,
    candidates_deduped: AtomicU64,
    batches_stage1_only: AtomicU64,
    batches_stage2: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VectorSearchSnapshot {
    pub windows_searched: u64,
    pub windows_filter_empty: u64,
    pub windows_centroids_too_far: u64,
    pub clusters_probed: u64,
    pub candidates_visited: u64,
    pub candidates_deduped: u64,
    pub batches_stage1_only: u64,
    pub batches_stage2: u64,
}

impl VectorSearchCounters {
    pub fn snapshot(&self) -> VectorSearchSnapshot {
        VectorSearchSnapshot {
            windows_searched: self.windows_searched.load(Relaxed),
            windows_filter_empty: self.windows_filter_empty.load(Relaxed),
            windows_centroids_too_far: self.windows_centroids_too_far.load(Relaxed),
            clusters_probed: self.clusters_probed.load(Relaxed),
            candidates_visited: self.candidates_visited.load(Relaxed),
            candidates_deduped: self.candidates_deduped.load(Relaxed),
            batches_stage1_only: self.batches_stage1_only.load(Relaxed),
            batches_stage2: self.batches_stage2.load(Relaxed),
        }
    }
}

impl VectorSearchStats for VectorSearchCounters {
    fn record_window(&self, w: WindowStats) {
        match w.outcome {
            WindowOutcome::FilterEmpty => {
                self.windows_filter_empty.fetch_add(1, Relaxed);
            }
            WindowOutcome::CentroidsTooFar { .. } => {
                self.windows_centroids_too_far.fetch_add(1, Relaxed);
            }
            WindowOutcome::Searched {
                clusters_probed,
                candidates_visited,
                candidates_deduped,
                batches_stage1_only,
                batches_stage2,
            } => {
                self.windows_searched.fetch_add(1, Relaxed);
                self.clusters_probed
                    .fetch_add(clusters_probed as u64, Relaxed);
                self.candidates_visited
                    .fetch_add(candidates_visited as u64, Relaxed);
                self.candidates_deduped
                    .fetch_add(candidates_deduped as u64, Relaxed);
                self.batches_stage1_only
                    .fetch_add(batches_stage1_only as u64, Relaxed);
                self.batches_stage2
                    .fetch_add(batches_stage2 as u64, Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn counters_aggregate_across_outcomes() {
        let c = Arc::new(VectorSearchCounters::default());
        let s: Arc<dyn VectorSearchStats> = c.clone();

        s.record_window(WindowStats {
            segment_ord: 0,
            window_ord: 0,
            outcome: WindowOutcome::FilterEmpty,
        });
        s.record_window(WindowStats {
            segment_ord: 0,
            window_ord: 1,
            outcome: WindowOutcome::CentroidsTooFar {
                nearest_centroid_dist: 12.5,
            },
        });
        s.record_window(WindowStats {
            segment_ord: 0,
            window_ord: 2,
            outcome: WindowOutcome::Searched {
                clusters_probed: 4,
                candidates_visited: 200,
                candidates_deduped: 7,
                batches_stage1_only: 8,
                batches_stage2: 5,
            },
        });
        s.record_window(WindowStats {
            segment_ord: 1,
            window_ord: 0,
            outcome: WindowOutcome::Searched {
                clusters_probed: 6,
                candidates_visited: 320,
                candidates_deduped: 11,
                batches_stage1_only: 14,
                batches_stage2: 6,
            },
        });

        let snap = c.snapshot();
        assert_eq!(snap.windows_filter_empty, 1);
        assert_eq!(snap.windows_centroids_too_far, 1);
        assert_eq!(snap.windows_searched, 2);
        assert_eq!(snap.clusters_probed, 10);
        assert_eq!(snap.candidates_visited, 520);
        assert_eq!(snap.candidates_deduped, 18);
        assert_eq!(snap.batches_stage1_only, 22);
        assert_eq!(snap.batches_stage2, 11);
    }
}

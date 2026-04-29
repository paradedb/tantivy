//! Public wrapper collector for TurboQuant vector search.
//!
//! Returned by [`TopDocs::order_by_turboquant_distance`]. Carries
//! optional knobs (`with_probe`, `with_stats`) without exposing the
//! crate-internal `TopBySortKeyCollector` or `SortByTurboQuantDistance`
//! types directly.

use std::ops::Range;
use std::sync::Arc;

use crate::collector::sort_key::{NaturalComparator, SortByTurboQuantDistance};
use crate::collector::sort_key_top_collector::{
    TopBySortKeyCollector, TopBySortKeySegmentCollector,
};
use crate::collector::Collector;
use crate::query::Weight;
use crate::schema::Schema;
use crate::vector::cluster::plugin::ProbeConfig;
use crate::{DocAddress, Result, Score, SegmentReader};

pub struct TurboQuantCollector {
    inner: TopBySortKeyCollector<SortByTurboQuantDistance>,
}

impl TurboQuantCollector {
    pub(crate) fn new(sort_key: SortByTurboQuantDistance, doc_range: Range<usize>) -> Self {
        Self {
            inner: TopBySortKeyCollector::new(sort_key, doc_range),
        }
    }

    /// Override the [`ProbeConfig`] used to drive multi-stage probing.
    /// Defaults to `ProbeConfig::default()`.
    pub fn with_probe(self, probe: ProbeConfig) -> Self {
        Self {
            inner: self.inner.map_sort_key(|sk| sk.with_probe(probe)),
        }
    }

    /// Attach a [`VectorSearchStats`] sink. The sink receives one
    /// [`WindowStats`] event per processed window per segment. Default
    /// path (no sink) is zero-cost: a single `Option::is_some` branch
    /// per window that the optimizer collapses.
    pub fn with_stats(self, stats: Arc<dyn VectorSearchStats>) -> Self {
        Self {
            inner: self.inner.map_sort_key(|sk| sk.with_stats(stats)),
        }
    }
}

impl Collector for TurboQuantCollector {
    type Fruit = Vec<(Score, DocAddress)>;
    type Child = TopBySortKeySegmentCollector<
        crate::collector::sort_key::TurboQuantSegmentComputer,
        NaturalComparator,
    >;

    fn check_schema(&self, schema: &Schema) -> Result<()> {
        self.inner.check_schema(schema)
    }

    fn for_segment(&self, ord: u32, reader: &SegmentReader) -> Result<Self::Child> {
        self.inner.for_segment(ord, reader)
    }

    fn requires_scoring(&self) -> bool {
        self.inner.requires_scoring()
    }

    fn merge_fruits(&self, fruits: Vec<Self::Fruit>) -> Result<Self::Fruit> {
        self.inner.merge_fruits(fruits)
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        ord: u32,
        reader: &SegmentReader,
    ) -> Result<Self::Fruit> {
        self.inner.collect_segment(weight, ord, reader)
    }
}

/// Sink for vector-search debug telemetry. Implementations are called
/// once per processed window per segment from the search loop in
/// [`SortByTurboQuantDistance::collect_segment_top_k`]. Calls happen
/// on the segment thread, so implementations must be `Send + Sync`
/// and tolerate concurrent invocations.
pub trait VectorSearchStats: Send + Sync {
    fn record_window(&self, window: WindowStats);
}

#[derive(Debug, Clone, Copy)]
pub struct WindowStats {
    pub segment_ord: u32,
    pub window_ord: u32,
    pub outcome: WindowOutcome,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowOutcome {
    /// Filter scorer matched no docs in this window; centroid probe
    /// was skipped entirely.
    FilterEmpty,
    /// Heap was full and this window's nearest centroid sat past the
    /// `distance_ratio` cutoff vs. the global best, so no clusters
    /// were probed.
    CentroidsTooFar { nearest_centroid_dist: f32 },
    /// Window was probed and at least one cluster was scored.
    Searched {
        clusters_probed: u16,
        candidates_visited: u32,
        candidates_deduped: u32,
        batches_stage1_only: u16,
        batches_stage2: u16,
    },
}

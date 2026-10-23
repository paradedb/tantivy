//! Per-segment probe statistics: assembled once from the stage reports and the stage clock, and
//! serialized flat for EXPLAIN.

use crate::vector::router::RouterMetrics;
use crate::vector::search::scan::ScanReport;
#[cfg(test)]
use crate::vector::search::scan::ScanTrace;
use crate::vector::search::segment::plan::PlanReport;
use crate::vector::search::source::controller::ProbeTermination;
use crate::vector::search::source::matches::MatchReport;
use crate::vector::search::source::router::RouterReport;
use crate::vector::search::source::SourceReport;
use crate::vector::telemetry::StageTotals;
use crate::vector::{VectorIoStats, VectorStage};

// --- Layers ---

/// One quantized layer's instrumentation.
#[derive(Debug, Default)]
pub(crate) struct LayerProbeStats {
    /// Owned-word decoding fallbacks in sign-plane scoring.
    pub(crate) sign_word_fallbacks: u64,
    /// Reads charged to the layer's scan.
    pub(crate) io: VectorIoStats,
    scan_ns: u64,
    boundary_ns: u64,
    scored: usize,
    admission_pruned: usize,
    survivors: usize,
    sparse_clusters: usize,
    full_clusters: usize,
    exact_clusters: usize,
    exact_rows: usize,
}

#[cfg(test)]
impl LayerProbeStats {
    /// Nanoseconds scanning the layer.
    pub(crate) fn scan_ns(&self) -> u64 {
        self.scan_ns
    }

    /// Nanoseconds in the layer's boundary.
    pub(crate) fn boundary_ns(&self) -> u64 {
        self.boundary_ns
    }

    /// Candidates the layer scored.
    pub(crate) fn scored(&self) -> usize {
        self.scored
    }

    /// Rows admission pruning dropped before they became candidates.
    pub(crate) fn admission_pruned(&self) -> usize {
        self.admission_pruned
    }

    /// Candidates left after the layer's boundary.
    pub(crate) fn survivors(&self) -> usize {
        self.survivors
    }

    /// Clusters read sparsely at this layer.
    pub(crate) fn sparse_clusters(&self) -> usize {
        self.sparse_clusters
    }

    /// Clusters read as a whole band at this layer.
    pub(crate) fn full_clusters(&self) -> usize {
        self.full_clusters
    }

    /// Clusters scored exactly, and finished, at this layer.
    pub(crate) fn exact_clusters(&self) -> usize {
        self.exact_clusters
    }

    /// Rows scored exactly, and finished, at this layer.
    pub(crate) fn exact_rows(&self) -> usize {
        self.exact_rows
    }
}

/// Per-layer instrumentation, flattened on the wire as `layer{i}_*` and `boundary{i}_ns`.
#[derive(Debug, Default)]
pub(crate) struct LayerProbeStatsSet(Vec<LayerProbeStats>);

#[cfg(test)]
impl LayerProbeStatsSet {
    /// Layer `layer`'s stats, when the scan reached it.
    pub(crate) fn get(&self, layer: usize) -> Option<&LayerProbeStats> {
        self.0.get(layer)
    }
}

impl serde::Serialize for LayerProbeStatsSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        for (index, layer) in self.0.iter().enumerate() {
            map.serialize_entry(
                &format!("layer{index}_sign_word_fallbacks"),
                &layer.sign_word_fallbacks,
            )?;
            serialize_prefixed(&layer.io, &format!("layer{index}"), &mut map)?;
            map.serialize_entry(&format!("layer{index}_scan_ns"), &layer.scan_ns)?;
            map.serialize_entry(&format!("layer{index}_scored"), &layer.scored)?;
            if index == 0 {
                map.serialize_entry("layer0_admission_pruned", &layer.admission_pruned)?;
            }
            map.serialize_entry(&format!("layer{index}_survivors"), &layer.survivors)?;
            map.serialize_entry(
                &format!("layer{index}_sparse_clusters"),
                &layer.sparse_clusters,
            )?;
            map.serialize_entry(&format!("layer{index}_full_clusters"), &layer.full_clusters)?;
            map.serialize_entry(
                &format!("layer{index}_exact_clusters"),
                &layer.exact_clusters,
            )?;
            map.serialize_entry(&format!("layer{index}_exact_rows"), &layer.exact_rows)?;
            map.serialize_entry(&format!("boundary{index}_ns"), &layer.boundary_ns)?;
        }
        map.end()
    }
}

// --- Serialization ---

fn serialize_prefixed<M>(io: &VectorIoStats, prefix: &str, map: &mut M) -> Result<(), M::Error>
where M: serde::ser::SerializeMap {
    map.serialize_entry(&format!("{prefix}_reads"), &io.reads)?;
    map.serialize_entry(&format!("{prefix}_bytes_read"), &io.bytes_read)?;
    map.serialize_entry(&format!("{prefix}_storage_blocks"), &io.storage_blocks)
}

fn serialize_io<S>(io: &VectorIoStats, prefix: &str, serializer: S) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    use serde::ser::SerializeMap;

    let mut map = serializer.serialize_map(Some(3))?;
    serialize_prefixed(io, prefix, &mut map)?;
    map.end()
}

fn serialize_exact_io<S>(io: &VectorIoStats, serializer: S) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    serialize_io(io, "exact", serializer)
}

fn serialize_locate_io<S>(io: &VectorIoStats, serializer: S) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    serialize_io(io, "locate", serializer)
}

fn serialize_centroid_io<S>(io: &VectorIoStats, serializer: S) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    serialize_io(io, "centroid", serializer)
}

/// Where a segment's layer-0 clusters came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, serde::Serialize)]
pub(crate) enum AccessPath {
    /// The router's ranking, gated by the filter bitset.
    #[default]
    Routed,
    /// The filter's own matches, located through the document location map.
    Located,
}

// --- Probe stats ---

/// One segment's vector search statistics. Counters sum across segments; `termination` and the
/// `segment_*` fields are per segment.
#[derive(Debug, Default, serde::Serialize)]
pub struct ProbeStats {
    /// Rows admitted and scored at layer 0 or exactly.
    pub candidates_scored: usize,
    /// Rows the routed source admitted and charged.
    pub layer0_eligible: usize,
    /// Routed clusters opened with no eligible row.
    pub clusters_skipped_empty: usize,
    /// Reads for exact scoring: full-precision rows and the DocIds they need.
    #[serde(flatten, serialize_with = "serialize_exact_io")]
    pub exact_io: VectorIoStats,
    /// The segment's located-path cap in matches; 0 with the located path off.
    pub located_cap: usize,
    /// Where this segment's layer-0 clusters came from.
    pub(crate) access_path: AccessPath,
    /// Live filter matches with a vector, on the located path.
    pub located_matches: usize,
    /// Live filter matches without a vector, on the located path.
    pub located_absent: usize,
    /// Clusters holding a located match.
    pub located_clusters: usize,
    /// Nanoseconds locating filter matches through the location map.
    pub locate_ns: u64,
    /// Reads of the location map.
    #[serde(flatten, serialize_with = "serialize_locate_io")]
    pub locate_io: VectorIoStats,
    /// Nanoseconds reading centroid rows for clusters with an unknown centroid score.
    pub centroid_ns: u64,
    /// Reads of centroid rows, one per row.
    #[serde(flatten, serialize_with = "serialize_centroid_io")]
    pub centroid_io: VectorIoStats,
    /// Per-layer counters of a quantized scan.
    #[serde(flatten)]
    pub(crate) layers: LayerProbeStatsSet,
    #[cfg(test)]
    #[serde(skip)]
    pub(crate) quantized_trace: ScanTrace,
    /// Rows scored at full precision, on every path.
    pub exact_rows: usize,
    /// Nanoseconds initializing the scan.
    pub scan_init_ns: u64,
    /// Nanoseconds evaluating the filter.
    pub non_vector_search_ns: u64,
    /// Nanoseconds preparing the query.
    pub query_prep_ns: u64,
    /// Nanoseconds ranking clusters and gating rows.
    pub routing_ns: u64,
    /// Nanoseconds in full-precision admission or the flat scan, outside exact reads and scoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_scan_ns: Option<u64>,
    /// Nanoseconds assembling results, including unattributed time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_assembly_ns: Option<u64>,
    /// Nanoseconds reading rows and DocIds for exact scoring.
    pub exact_fetch_ns: u64,
    /// Nanoseconds scoring rows exactly.
    pub exact_score_ns: u64,
    /// Rows the routed gate looked at.
    pub vectors_visited: usize,
    /// Rows the gate rejected by the filter.
    pub pruned_filter: usize,
    /// Rows the gate rejected as deleted.
    pub pruned_dead: usize,
    /// Clusters admitted with at least one row.
    pub postings_row: usize,
    /// Clusters opened that contributed no row.
    pub postings_skipped: usize,
    /// The router's counters.
    #[serde(skip)]
    pub routing: Option<RouterMetrics>,
    /// Centroids scored while ranking.
    pub routing_visited_count: usize,
    /// Clusters the bounds gate passed over.
    pub bounds_skips: u32,
    /// Segment scans in which the query bound armed: 0 or 1.
    pub bound_armed_count: u32,
    /// The zero-based probe at which the query bound armed; 0 when it never armed.
    pub bound_armed_probe_sum: u64,
    /// How the routed probe loop stopped.
    pub termination: ProbeTermination,
    /// Work units the probe loop charged.
    pub work_charged: f32,
    /// The work budget the probe loop ran against, in work units; 0 when no loop ran.
    pub work_budget: f32,
    /// Vector rows in the segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_rows: Option<usize>,
    /// IVF clusters in the segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_clusters: Option<usize>,
}

// --- Segment report ---

/// What one segment search reports; its statistics are built from it once its measure ends.
pub(crate) enum SegmentReport {
    /// A segment without an IVF index: rows scored by the flat scan.
    Flat { rows_read: usize },
    /// Nothing could match; the located cap still reports.
    Empty { located_cap: usize },
    /// An IVF segment search: its plan, cluster source and scan.
    Ivf {
        plan: PlanReport,
        source: SourceReport,
        scan: ScanReport,
    },
}

impl ProbeStats {
    /// Statistics of one segment search from its report and the totals of its measure.
    pub(crate) fn new(report: SegmentReport, totals: StageTotals) -> Self {
        let mut stats = match report {
            SegmentReport::Flat { rows_read } => {
                let mut stats = Self {
                    exact_rows: rows_read,
                    exact_scan_ns: Some(0),
                    ..Self::default()
                };
                stats.record_io(&totals);
                stats
            }
            SegmentReport::Empty { located_cap } => Self {
                located_cap,
                ..Self::default()
            },
            SegmentReport::Ivf { plan, source, scan } => Self::ivf(plan, source, scan, &totals),
        };
        // Every segment search reports its result assembly, an empty one included.
        stats.result_assembly_ns = Some(0);
        stats.apply_timings(&totals);
        stats
    }

    /// Statistics of an IVF segment search from each stage's report, timings aside.
    fn ivf(plan: PlanReport, source: SourceReport, scan: ScanReport, totals: &StageTotals) -> Self {
        let full_precision = scan.layers.is_empty();
        let mut stats = Self {
            candidates_scored: scan.candidates_scored,
            exact_rows: scan.exact_rows,
            layers: LayerProbeStatsSet(
                scan.layers
                    .iter()
                    .enumerate()
                    .map(|(layer, report)| LayerProbeStats {
                        sign_word_fallbacks: totals.sign_word_fallbacks(layer),
                        io: totals.io(VectorStage::LayerScan(layer as u8)),
                        scored: report.scored,
                        admission_pruned: report.admission_pruned,
                        survivors: report.survivors,
                        sparse_clusters: report.sparse_clusters,
                        full_clusters: report.full_clusters,
                        exact_clusters: report.exact_clusters,
                        exact_rows: report.exact_rows,
                        ..LayerProbeStats::default()
                    })
                    .collect(),
            ),
            #[cfg(test)]
            quantized_trace: *scan.trace,
            exact_scan_ns: full_precision.then_some(0),
            segment_rows: Some(plan.segment_rows),
            segment_clusters: Some(plan.segment_clusters),
            located_cap: plan.located_cap,
            ..Self::default()
        };
        match source {
            SourceReport::Router(routed) => stats.record_routed(routed),
            SourceReport::Matches(matches) => stats.record_matches(matches),
        }
        stats.record_io(totals);
        debug_assert_eq!(
            stats.centroid_io.reads, scan.centroid_reads as u64,
            "one read per centroid row"
        );
        stats
    }

    fn record_routed(&mut self, report: RouterReport) {
        let RouterReport {
            counters,
            routing,
            probe,
        } = report;
        self.layer0_eligible = counters.eligible;
        self.clusters_skipped_empty = counters.clusters_skipped_empty;
        self.vectors_visited = counters.vectors_visited;
        self.pruned_filter = counters.pruned_filter;
        self.pruned_dead = counters.pruned_dead;
        self.postings_row = counters.postings_row;
        self.postings_skipped = counters.postings_skipped;
        self.bounds_skips = counters.bounds_skips;
        if let Some(probe) = counters.bound_armed_at {
            self.bound_armed_count = 1;
            self.bound_armed_probe_sum = u64::from(probe);
        }
        self.routing = Some(routing);
        self.routing_visited_count = match routing {
            RouterMetrics::Stacked {
                candidate_count, ..
            } => candidate_count,
            RouterMetrics::Exact { visited_count } => visited_count,
        };
        self.termination = probe.termination;
        self.work_charged = probe.work_charged;
        self.work_budget = probe.work_budget;
    }

    fn record_matches(&mut self, report: MatchReport) {
        self.access_path = AccessPath::Located;
        self.located_matches = report.matches;
        self.located_absent = report.absent;
        self.located_clusters = report.clusters;
        self.vectors_visited = report.matches + report.dead;
        self.pruned_dead = report.dead;
        self.postings_row = report.clusters;
        self.work_charged = report.work_charged;
        self.work_budget = report.work_budget;
    }

    fn record_io(&mut self, totals: &StageTotals) {
        let (fetch, score) = (
            totals.io(VectorStage::ExactFetch),
            totals.io(VectorStage::ExactScore),
        );
        self.exact_io = VectorIoStats {
            reads: fetch.reads + score.reads,
            bytes_read: fetch.bytes_read + score.bytes_read,
            storage_blocks: fetch.storage_blocks + score.storage_blocks,
        };
        self.locate_io = totals.io(VectorStage::Locate);
        self.centroid_io = totals.io(VectorStage::Centroid);
    }

    /// Sets every timing field from `totals`; optional timings stay absent when unset.
    fn apply_timings(&mut self, totals: &StageTotals) {
        self.scan_init_ns = totals.time_ns(VectorStage::ScanInit);
        self.non_vector_search_ns = totals.time_ns(VectorStage::NonVectorSearch);
        self.query_prep_ns = totals.time_ns(VectorStage::QueryPrep);
        self.routing_ns = totals.time_ns(VectorStage::Routing);
        self.locate_ns = totals.time_ns(VectorStage::Locate);
        self.centroid_ns = totals.time_ns(VectorStage::Centroid);
        self.exact_fetch_ns = totals.time_ns(VectorStage::ExactFetch);
        self.exact_score_ns = totals.time_ns(VectorStage::ExactScore);
        if let Some(exact_scan_ns) = self.exact_scan_ns.as_mut() {
            *exact_scan_ns = totals.time_ns(VectorStage::ExactScan);
        }
        if let Some(result_assembly_ns) = self.result_assembly_ns.as_mut() {
            *result_assembly_ns =
                totals.time_ns(VectorStage::ResultAssembly) + totals.time_ns(VectorStage::Other);
        }
        for (layer, stats) in self.layers.0.iter_mut().enumerate() {
            stats.scan_ns = totals.time_ns(VectorStage::LayerScan(layer as u8));
            stats.boundary_ns = totals.time_ns(VectorStage::Boundary(layer as u8));
        }
    }

    /// Adds the result-assembly time of the merge, measured after the segment's statistics were
    /// built.
    pub(crate) fn add_assembly(&mut self, totals: &StageTotals) {
        let extra =
            totals.time_ns(VectorStage::ResultAssembly) + totals.time_ns(VectorStage::Other);
        self.result_assembly_ns = Some(self.result_assembly_ns.unwrap_or(0) + extra);
    }

    /// Zeroes every timing field, keeping which optional timings are present.
    #[cfg(test)]
    pub(crate) fn clear_stage_timings(&mut self) {
        self.apply_timings(&StageTotals::default());
    }

    /// Clusters the probe loop opened: `postings_row + postings_skipped`.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::vector::search::scan::LayerReport;
    use crate::vector::search::source::controller::ProbeOutcome;
    use crate::vector::search::source::router::RouterCounters;

    fn keys(stats: &ProbeStats) -> BTreeSet<String> {
        serde_json::to_value(stats)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    fn routed() -> SourceReport {
        SourceReport::Router(RouterReport {
            counters: RouterCounters::default(),
            routing: RouterMetrics::Exact { visited_count: 7 },
            probe: ProbeOutcome::default(),
        })
    }

    const COMMON: &[&str] = &[
        "candidates_scored",
        "layer0_eligible",
        "clusters_skipped_empty",
        "exact_reads",
        "exact_bytes_read",
        "exact_storage_blocks",
        "located_cap",
        "access_path",
        "located_matches",
        "located_absent",
        "located_clusters",
        "locate_ns",
        "locate_reads",
        "locate_bytes_read",
        "locate_storage_blocks",
        "centroid_ns",
        "centroid_reads",
        "centroid_bytes_read",
        "centroid_storage_blocks",
        "exact_rows",
        "scan_init_ns",
        "non_vector_search_ns",
        "query_prep_ns",
        "routing_ns",
        "result_assembly_ns",
        "exact_fetch_ns",
        "exact_score_ns",
        "vectors_visited",
        "pruned_filter",
        "pruned_dead",
        "postings_row",
        "postings_skipped",
        "routing_visited_count",
        "bounds_skips",
        "bound_armed_count",
        "bound_armed_probe_sum",
        "termination",
        "work_charged",
        "work_budget",
    ];

    fn expected(extra: &[&str], layers: usize) -> BTreeSet<String> {
        let mut keys: BTreeSet<String> = COMMON
            .iter()
            .chain(extra)
            .map(|key| key.to_string())
            .collect();
        for layer in 0..layers {
            for suffix in [
                "sign_word_fallbacks",
                "reads",
                "bytes_read",
                "storage_blocks",
                "scan_ns",
                "scored",
                "survivors",
                "sparse_clusters",
                "full_clusters",
                "exact_clusters",
                "exact_rows",
            ] {
                keys.insert(format!("layer{layer}_{suffix}"));
            }
            keys.insert(format!("boundary{layer}_ns"));
        }
        if layers > 0 {
            keys.insert("layer0_admission_pruned".to_string());
        }
        keys
    }

    /// The serialized keys of each kind of segment search: IO flattened per stage, layer keys only
    /// for a quantized scan, and optional keys only when set.
    #[test]
    fn probe_stats_keys() {
        let plan = PlanReport {
            segment_rows: 100,
            segment_clusters: 5,
            located_cap: 0,
        };
        let quantized = ScanReport {
            layers: vec![LayerReport::default(); 2],
            ..ScanReport::default()
        };
        let segment = ["segment_rows", "segment_clusters"];
        let ivf = |scan| SegmentReport::Ivf {
            plan,
            source: routed(),
            scan,
        };
        let stats = ProbeStats::new(ivf(quantized), StageTotals::default());
        assert_eq!(keys(&stats), expected(&segment, 2));
        let layer = stats.layers.get(1).unwrap();
        assert_eq!((layer.scan_ns(), layer.boundary_ns()), (0, 0));
        let full_precision = [&segment[..], &["exact_scan_ns"]].concat();
        assert_eq!(
            keys(&ProbeStats::new(
                ivf(ScanReport::default()),
                StageTotals::default()
            )),
            expected(&full_precision, 0)
        );
        let flat = ProbeStats::new(SegmentReport::Flat { rows_read: 3 }, StageTotals::default());
        assert_eq!(keys(&flat), expected(&["exact_scan_ns"], 0));
        let empty = ProbeStats::new(
            SegmentReport::Empty { located_cap: 0 },
            StageTotals::default(),
        );
        assert_eq!(keys(&empty), expected(&[], 0));
    }
}

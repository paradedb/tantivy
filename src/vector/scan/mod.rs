//! The IVF segment scan: shortlists each cluster a source supplies, narrows the quantized
//! shortlist layer by layer under one pruning bound, and scores what is left exactly.

pub(crate) mod bound;
pub(crate) mod columns;
pub(crate) mod estimate;
mod exact;
pub(crate) mod heap;
pub(crate) mod layer_read;
pub(crate) mod plan;

use std::ops::Range;

use self::bound::KthBound;
use self::columns::{Columns, QuantizedCandidates};
use self::estimate::{combine_initial, combine_refine, CombineCtx};
use self::exact::ExactScorer;
use self::heap::TieBreakHits;
use self::layer_read::{decode_layer, Decoded, LayerRead, LayerSpec};
#[cfg(test)]
use self::plan::{forced_plan, PlanOverride};
use self::plan::{price, Read, ReadCosts};
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::index_reader::{QuantizedFieldReader, ReadScratch, VectorIndexReader};
use crate::vector::ivf::IvfIndex;
use crate::vector::prepared::{QuantizedQueryCtx, ScoringQuery};
use crate::vector::{enter_vector_stage, Similarity, Stage, VectorElement};
use crate::SegmentOrdinal;

// --- Contracts ---

/// Sizes for one segment scan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) top_n: usize,
    /// Upper bound on rows admitted, for sizing the candidate columns.
    pub(crate) capacity: usize,
    #[cfg(test)]
    pub(crate) plan_override: Option<PlanOverride>,
}

/// Per-layer counts of a quantized scan.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LayerReport {
    /// Candidates the layer scored.
    pub(crate) scored: usize,
    /// Candidates left after the layer's boundary.
    pub(crate) survivors: usize,
    /// Clusters read sparsely at this layer.
    pub(crate) sparse_clusters: usize,
    /// Clusters read as a whole band at this layer.
    pub(crate) full_clusters: usize,
}

impl LayerReport {
    fn count(&mut self, read: LayerRead) {
        match read {
            LayerRead::Sparse => self.sparse_clusters += 1,
            LayerRead::Band => self.full_clusters += 1,
        }
    }
}

/// What a finished scan reports.
#[derive(Debug, Default)]
pub(crate) struct ScanReport {
    /// Rows shortlisted from the source.
    pub(crate) candidates_scored: usize,
    /// One per active layer on a quantized segment; empty at full precision.
    pub(crate) layers: Vec<LayerReport>,
    /// Rows scored at full precision.
    pub(crate) exact_rows: usize,
    #[cfg(test)]
    pub(crate) trace: ScanTrace,
}

/// Rows each stage of a scan scored and kept, by global row.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ScanTrace {
    /// Rows shortlisted at layer 0.
    pub(crate) scored_rows: Vec<usize>,
    pub(crate) boundary_rows: Vec<Vec<usize>>,
    /// `(row, estimate bits)` after the shortlist and after each refining layer.
    pub(crate) estimate_rows: Vec<Vec<(usize, u32)>>,
    /// `(row, [raw prefix, base, estimate, sigma] bits)` at the same points.
    pub(crate) layer_columns: Vec<Vec<(usize, [u32; 4])>>,
    /// Rows scored in the exact phase.
    pub(crate) exact_phase_rows: Vec<usize>,
}

/// What the scan remembers about a cluster it admitted.
#[derive(Clone, Copy, Debug)]
struct AdmittedCluster {
    centroid: Similarity,
    /// Whether the candidates carry this cluster's doc ids.
    docs_in_candidates: bool,
}

// --- Quantized state ---

/// The quantized cascade's state: candidate columns plus each admitted cluster's record.
struct QuantizedState<'a> {
    /// The active layers, in scan order.
    layers: Vec<LayerSpec<'a>>,
    metric: Metric,
    dimension: usize,
    candidates: QuantizedCandidates,
    /// By cluster id; `None` for clusters never admitted.
    admitted: Vec<Option<AdmittedCluster>>,
    decoded: Decoded,
    /// A shortlisted cluster's layer-0 columns, before they join the candidates.
    cluster_columns: Columns,
    #[cfg(test)]
    plan_override: Option<PlanOverride>,
}

impl<'a> QuantizedState<'a> {
    fn new(
        query: &'a QuantizedQueryCtx,
        field: &'a QuantizedFieldReader,
        config: &ScanConfig,
    ) -> Self {
        let meta = query.index.meta.field();
        Self {
            layers: (0..query.active_layers())
                .map(|index| LayerSpec {
                    query,
                    reader: &field.layers()[index],
                    index,
                    query_error_squared: query.query_error_squared(index) as f32,
                })
                .collect(),
            metric: meta.metric,
            dimension: meta.dim as usize,
            candidates: QuantizedCandidates::with_capacity(config.capacity),
            admitted: Vec::new(),
            decoded: Decoded::default(),
            cluster_columns: Columns::default(),
            #[cfg(test)]
            plan_override: config.plan_override,
        }
    }

    /// The plan for `costs`, or a test's override.
    fn choose(&self, costs: &ReadCosts) -> Read {
        #[cfg(test)]
        return forced_plan(costs, self.plan_override);
        #[cfg(not(test))]
        plan::plan(costs)
    }

    /// Prices the rows of `selected` at `layer` and plans their read. A sparse plan leaves its rows
    /// and code runs in `scratch` for the read.
    fn read_plan(
        &self,
        selected: &ClusterRows<'_>,
        layer: usize,
        scratch: &mut ReadScratch,
    ) -> crate::Result<Read> {
        let costs = price(&self.layers[layer], selected, scratch)?;
        Ok(self.choose(&costs))
    }

    fn record(&mut self, cluster: usize, admitted: AdmittedCluster) {
        if self.admitted.len() <= cluster {
            self.admitted.resize(cluster + 1, None);
        }
        self.admitted[cluster] = Some(admitted);
    }

    /// The admitted clusters in ascending id, which is ascending row order.
    fn admitted_clusters(&self) -> impl Iterator<Item = (usize, AdmittedCluster)> + '_ {
        self.admitted
            .iter()
            .enumerate()
            .filter_map(|(cluster, admitted)| admitted.map(|admitted| (cluster, admitted)))
    }

    /// Reads `layer` for the rows of `selected` as `read` plans and decodes them; returns how to
    /// combine the decoded columns into estimates.
    fn decode(
        &mut self,
        layer: usize,
        selected: &ClusterRows<'_>,
        read: LayerRead,
        scratch: &mut ReadScratch,
    ) -> crate::Result<CombineCtx> {
        let spec = self.layers[layer];
        decode_layer(&spec, selected, read, &mut self.decoded, scratch)?;
        let cluster_score = selected.centroid.score();
        let query_norm = spec.query.score_query_norm(cluster_score);
        Ok(CombineCtx {
            metric: self.metric,
            dimension: self.dimension,
            query_norm_squared: query_norm * query_norm,
            query_error_squared: spec.query_error_squared,
            cluster_score,
        })
    }

    /// Scores a cluster's rows on layer 0, read as `read`, and appends them as candidates.
    fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        read: LayerRead,
        bound: &mut KthBound,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let ctx = self.decode(0, rows, read, scratch)?;
        combine_initial(&ctx, &self.decoded, &mut self.cluster_columns);
        self.record(
            rows.cluster,
            AdmittedCluster {
                centroid: rows.centroid,
                docs_in_candidates: rows.docs.is_some(),
            },
        );
        let cluster_start = self.candidates.len();
        self.candidates.append(
            rows.rows.clone(),
            rows.offsets,
            rows.docs,
            &self.cluster_columns,
        );
        self.candidates.finish_cluster(cluster_start, bound);
        Ok(())
    }

    /// Refines the candidates at `at`, the survivors `group` of one cluster, with `layer` read as
    /// `read`.
    fn refine(
        &mut self,
        layer: usize,
        group: &ClusterRows<'_>,
        read: LayerRead,
        at: Range<usize>,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let ctx = self.decode(layer, group, read, scratch)?;
        combine_refine(&ctx, &self.decoded, &mut self.candidates.columns, at);
        Ok(())
    }

    /// Records the candidates `layer` scored and prunes them at its boundary.
    fn prune(&mut self, layer: usize, bound: &mut KthBound, report: &mut ScanReport) {
        report.layers[layer].scored = self.candidates.len();
        #[cfg(test)]
        {
            report
                .trace
                .estimate_rows
                .push(self.candidates.estimate_trace());
            report
                .trace
                .layer_columns
                .push(self.candidates.column_trace());
        }
        let _boundary = enter_vector_stage(Stage::Boundary(layer as u8));
        self.candidates.band(bound);
        report.layers[layer].survivors = self.candidates.len();
        #[cfg(test)]
        report
            .trace
            .boundary_rows
            .push(self.candidates.rows.clone());
    }
}
/// Splits row-sorted survivors into one candidate range per admitted cluster.
fn survivor_groups(
    rows: &[usize],
    admitted: impl Iterator<Item = (usize, AdmittedCluster)>,
    index: &IvfIndex,
) -> Vec<(usize, AdmittedCluster, Range<usize>)> {
    debug_assert!(
        rows.windows(2).all(|pair| pair[0] < pair[1]),
        "survivors are row-sorted"
    );
    let mut groups = Vec::new();
    let mut first = 0;
    for (cluster, record) in admitted {
        if first == rows.len() {
            break;
        }
        let cluster_rows = index.cluster_range(cluster);
        let end = first + rows[first..].partition_point(|&row| row < cluster_rows.end);
        if end > first {
            debug_assert!(rows[first] >= cluster_rows.start);
            groups.push((cluster, record, first..end));
        }
        first = end;
    }
    debug_assert_eq!(
        first,
        rows.len(),
        "every survivor belongs to an admitted cluster"
    );
    groups
}

// --- Segment scan ---

/// One segment's IVF scan. A cluster source supplies each cluster's rows; the scan shortlists them
/// under the segment's quantization, keeps the pruning bound, and scores what is left exactly.
pub(crate) struct SegmentScan<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    index: &'a IvfIndex,
    exact: ExactScorer<'a, T, K, C>,
    /// The quantized cascade; `None` scores every shortlisted row exactly.
    quantized: Option<QuantizedState<'a>>,
    scratch: ReadScratch,
    /// A survivor group's cluster-local offsets.
    group_offsets: Vec<usize>,
    report: ScanReport,
}

impl<'a, T, K, C> SegmentScan<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    /// Fails with `DataCorruption` when a quantized query meets a segment without quantized slots.
    pub(crate) fn new(
        reader: &'a VectorIndexReader,
        index: &'a IvfIndex,
        query: ScoringQuery<'a, T>,
        config: ScanConfig,
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> crate::Result<Self> {
        let quantized = match (query.quantized, reader.quantization()) {
            (Some(quantized), Some(field)) => Some(QuantizedState::new(quantized, field, &config)),
            (Some(_), None) => {
                return Err(DataCorruption::comment_only(
                    "quantized query requires quantized slots",
                )
                .into())
            }
            (None, _) => None,
        };
        let layers = quantized.as_ref().map_or(0, |state| state.layers.len());
        Ok(Self {
            index,
            exact: ExactScorer::new(reader, query.query, config.top_n, tie_break, tie_comparator),
            quantized,
            scratch: ReadScratch::default(),
            group_offsets: Vec::new(),
            report: ScanReport {
                layers: vec![LayerReport::default(); layers],
                ..ScanReport::default()
            },
        })
    }

    /// Scores one segment: shortlists each cluster `source` supplies, narrows the shortlist one
    /// quantized layer at a time, then scores what is left exactly.
    pub(crate) fn run(&mut self, source: &mut impl ClusterSource) -> crate::Result<()> {
        let stage = enter_vector_stage(self.shortlist_stage());
        while let Some(rows) = source.next(self.feedback())? {
            let scored = self.shortlist(&rows)?;
            source.admitted(scored, self.feedback())?;
        }
        drop(stage);
        self.close_shortlist();
        for layer in self.refining_layers() {
            self.narrow(layer)?;
        }
        self.score_shortlist()
    }

    /// The stage the shortlist is built in: layer 0, or the exact scan at full precision.
    fn shortlist_stage(&self) -> Stage {
        match self.quantized {
            Some(_) => Stage::LayerScan(0),
            None => Stage::ExactScan,
        }
    }

    /// The pruning thresholds as of the last shortlisted cluster.
    fn feedback(&self) -> Feedback {
        Feedback {
            kth_lower: self.exact.bound.running_lower(),
            kth_estimate: self.exact.bound.running_estimate(),
        }
    }

    /// Prunes the shortlist at layer 0's boundary.
    fn close_shortlist(&mut self) {
        if let Some(state) = &mut self.quantized {
            state.prune(0, &mut self.exact.bound, &mut self.report);
        }
    }

    /// The quantized layers after layer 0; none at full precision.
    fn refining_layers(&self) -> Range<usize> {
        1..self
            .quantized
            .as_ref()
            .map_or(1, |state| state.layers.len())
    }

    /// Adds a cluster's rows to the shortlist: scored exactly at full precision, otherwise as
    /// quantized candidates. Returns the rows scored.
    fn shortlist(&mut self, rows: &ClusterRows<'_>) -> crate::Result<usize> {
        let len = rows.len();
        debug_assert!(len > 0, "empty clusters never reach the scan");
        #[cfg(test)]
        self.report.trace.scored_rows.extend(rows.global_rows());
        match &mut self.quantized {
            Some(state) => {
                let Read::Layer(read) = state.read_plan(rows, 0, &mut self.scratch)?;
                self.report.layers[0].count(read);
                state.admit(rows, read, &mut self.exact.bound, &mut self.scratch)?;
            }
            None => {
                self.report.exact_rows += self.exact.score(rows)?;
            }
        }
        self.report.candidates_scored += len;
        Ok(len)
    }

    /// Refines each cluster's surviving candidates with `layer`, then prunes them at the layer's
    /// boundary.
    fn narrow(&mut self, layer: usize) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        let groups = survivor_groups(
            &state.candidates.rows,
            state.admitted_clusters(),
            self.index,
        );
        let layer_scan = enter_vector_stage(Stage::LayerScan(layer as u8));
        for (cluster, admitted, at) in groups {
            let rows = self.index.cluster_range(cluster);
            local_offsets(
                &mut self.group_offsets,
                &state.candidates.rows[at.clone()],
                rows.start,
            );
            let group = ClusterRows {
                cluster,
                rows,
                offsets: Some(&self.group_offsets),
                docs: None,
                centroid: admitted.centroid,
            };
            let Read::Layer(read) = state.read_plan(&group, layer, &mut self.scratch)?;
            self.report.layers[layer].count(read);
            state.refine(layer, &group, read, at, &mut self.scratch)?;
        }
        drop(layer_scan);
        state.prune(layer, &mut self.exact.bound, &mut self.report);
        Ok(())
    }

    /// Scores the last boundary's survivors exactly, one cluster group at a time.
    fn score_shortlist(&mut self) -> crate::Result<()> {
        let Some(state) = &self.quantized else {
            return Ok(());
        };
        let groups = survivor_groups(
            &state.candidates.rows,
            state.admitted_clusters(),
            self.index,
        );
        for (cluster, admitted, at) in groups {
            let rows = self.index.cluster_range(cluster);
            local_offsets(
                &mut self.group_offsets,
                &state.candidates.rows[at.clone()],
                rows.start,
            );
            let group = ClusterRows {
                cluster,
                rows,
                offsets: Some(&self.group_offsets),
                docs: admitted
                    .docs_in_candidates
                    .then(|| &state.candidates.docs[at]),
                centroid: admitted.centroid,
            };
            self.report.exact_rows += self.exact.score_survivors(&group)?;
        }
        Ok(())
    }

    /// Returns the hits and the scan's report.
    pub(crate) fn finish(
        self,
        segment_ord: SegmentOrdinal,
    ) -> crate::Result<(TieBreakHits<K>, ScanReport)> {
        let _assembly = enter_vector_stage(Stage::ResultAssembly);
        let Self { exact, report, .. } = self;
        #[cfg(test)]
        let report = {
            let mut report = report;
            report.trace.exact_phase_rows = exact.exact_phase_rows.clone();
            report
        };
        Ok((exact.into_hits(segment_ord), report))
    }
}

/// Writes the cluster-local offsets of `rows`, in a cluster starting at `start`, into `offsets`.
fn local_offsets(offsets: &mut Vec<usize>, rows: &[usize], start: usize) {
    offsets.clear();
    offsets.extend(rows.iter().map(|&row| row - start));
}

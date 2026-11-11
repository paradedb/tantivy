//! The router cluster source: clusters in the router's order, admitted under the work budget,
//! and the filter.

use std::sync::Arc;

use common::BitSet;

use crate::fastfield::AliveBitSet;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::{RouterIter, RouterMetrics};
use crate::vector::search::cluster_source::{ClusterRows, ClusterSource};
use crate::vector::search::source::controller::{
    ProbeController, ProbeOutcome, SharedVectorWork, WorkBudget, WorkMeter,
};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::VectorStage;
use crate::{DocId, SegmentReader};

// --- Filter and gate ---

/// The filter's matches as the router source gates rows with them.
#[derive(Clone)]
pub(crate) enum DocFilter {
    /// Every doc id below `max_doc` matches, deleted docs included; no bitset is built.
    All,
    /// The matching doc ids.
    Docs(Arc<BitSet>),
}

impl DocFilter {
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Docs(docs) if docs.len() == 0)
    }

    /// The share of doc ids below `max_doc` that match.
    pub(crate) fn match_fraction(&self, max_doc: DocId) -> f64 {
        match self {
            Self::All => 1.0,
            Self::Docs(docs) => docs.len() as f64 / f64::from(max_doc.max(1)),
        }
    }
}

/// How a cluster row is tested before scoring: against the filter's matches and the alive docs.
pub(crate) enum RowGate<'a> {
    Open,
    AliveOnly(&'a AliveBitSet),
    FilterOnly(Arc<BitSet>),
    FilterAndAlive {
        filter: Arc<BitSet>,
        alive: &'a AliveBitSet,
    },
}

/// What a gate kept of one cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatedRows {
    /// Every row is kept; the offsets are left empty.
    pub(crate) all: bool,
    pub(crate) pruned_filter: usize,
    pub(crate) pruned_dead: usize,
}

impl<'a> RowGate<'a> {
    pub(crate) fn new(filter: DocFilter, alive: Option<&'a AliveBitSet>) -> Self {
        match (filter, alive) {
            (DocFilter::All, None) => RowGate::Open,
            (DocFilter::All, Some(alive)) => RowGate::AliveOnly(alive),
            (DocFilter::Docs(filter), None) => RowGate::FilterOnly(filter),
            (DocFilter::Docs(filter), Some(alive)) => RowGate::FilterAndAlive { filter, alive },
        }
    }

    /// Whether every row passes without reading its doc id.
    pub(crate) fn is_open(&self) -> bool {
        matches!(self, RowGate::Open)
    }

    /// Gates a cluster whose DocIds column is `doc_ids`, writing the kept cluster-local offsets.
    pub(crate) fn select(&self, doc_ids: &[DocId], offsets: &mut Vec<u32>) -> GatedRows {
        offsets.clear();
        let verdicts = match self {
            RowGate::Open => {
                return GatedRows {
                    all: true,
                    pruned_filter: 0,
                    pruned_dead: 0,
                }
            }
            RowGate::AliveOnly(alive) => gate(doc_ids, offsets, |doc| {
                let alive = alive.is_alive(doc);
                RowVerdict {
                    matches: true,
                    keep: alive,
                }
            }),
            RowGate::FilterOnly(filter) => gate(doc_ids, offsets, |doc| {
                let matches = filter.contains(doc);
                RowVerdict {
                    matches,
                    keep: matches,
                }
            }),
            RowGate::FilterAndAlive { filter, alive } => gate(doc_ids, offsets, |doc| {
                let matches = filter.contains(doc);
                RowVerdict {
                    matches,
                    keep: matches & alive.is_alive(doc),
                }
            }),
        };
        GatedRows {
            all: false,
            pruned_filter: doc_ids.len() - verdicts.matched,
            pruned_dead: verdicts.matched - verdicts.kept,
        }
    }
}

/// One row's verdict: whether it matches the filter, and whether it is kept, which also needs it
/// alive.
struct RowVerdict {
    matches: bool,
    keep: bool,
}

/// Rows a gate counted: those matching the filter, and those kept.
struct GateCounts {
    matched: usize,
    kept: usize,
}

/// Keeps the offsets `verdict` accepts. Each row's offset is written at the next free slot and the
/// slot advances by the verdict, so the loop has no branch on it.
#[inline(always)]
fn gate(
    doc_ids: &[DocId],
    offsets: &mut Vec<u32>,
    verdict: impl Fn(DocId) -> RowVerdict,
) -> GateCounts {
    offsets.resize(doc_ids.len(), 0);
    let mut kept = 0usize;
    let mut matched = 0usize;
    for (offset, &doc) in (0u32..).zip(doc_ids) {
        let RowVerdict { matches, keep } = verdict(doc);
        offsets[kept] = offset;
        kept += usize::from(keep);
        matched += usize::from(matches);
    }
    offsets.truncate(kept);
    GateCounts { matched, kept }
}

// --- Source ---

/// What a router source counted, kept in place while it runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RouterCounters {
    /// Clusters admitted with at least one row.
    pub(crate) postings_row: usize,
    /// Clusters opened that contributed no row: empty, or every row gated out.
    pub(crate) postings_skipped: usize,
    /// Of `postings_skipped`, clusters with no eligible row.
    pub(crate) clusters_skipped_empty: usize,
    /// Rows the gate looked at.
    pub(crate) vectors_visited: usize,
    pub(crate) pruned_filter: usize,
    pub(crate) pruned_dead: usize,
    /// Rows admitted and charged.
    pub(crate) eligible: usize,
}

/// A finished router source's report.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RouterReport {
    pub(crate) counters: RouterCounters,
    pub(crate) routing: RouterMetrics,
    pub(crate) probe: ProbeOutcome,
}

/// Clusters in the router's order, admitted under the work budget and the filter.
pub(crate) struct RouterClusters<'a> {
    index: &'a ClusterIndex,
    reader: &'a VectorIndexReader,
    ranked: RouterIter,
    meter: WorkMeter,
    controller: ProbeController,
    gate: RowGate<'a>,
    /// Whether a document may sit in two clusters: the scan then needs every opened cluster's
    /// doc ids to score each document once.
    replicated: bool,
    /// The current cluster's DocIds column; read when the gate is closed or the segment is
    /// replicated.
    docs: Vec<DocId>,
    offsets: Vec<u32>,
    selected_docs: Vec<DocId>,
    counters: RouterCounters,
}

impl<'a> RouterClusters<'a> {
    pub(crate) fn from_ranked(
        reader: &'a VectorIndexReader,
        ranked: RouterIter,
        budget: WorkBudget,
        filter: DocFilter,
        segment: &'a SegmentReader,
        shared_work: Option<Arc<dyn SharedVectorWork>>,
    ) -> Self {
        let index = reader.index().expect("IVF segment");
        Self {
            index,
            reader,
            ranked,
            meter: WorkMeter::new(budget, shared_work),
            controller: ProbeController::new(index.num_clusters()),
            gate: RowGate::new(filter, segment.alive_bitset()),
            replicated: reader.replicated(),
            docs: Vec::new(),
            offsets: Vec::new(),
            selected_docs: Vec::new(),
            counters: RouterCounters::default(),
        }
    }

    /// Prepares this segment for a global search across multiple segments.
    /// The global search supplies clusters in ranked order and enforces the shared budget.
    pub(crate) fn for_global(
        reader: &'a VectorIndexReader,
        budget: WorkBudget,
        filter: DocFilter,
        segment: &'a SegmentReader,
        metrics: RouterMetrics,
    ) -> Self {
        Self::from_ranked(
            reader,
            RouterIter::new(Vec::new(), metrics),
            budget,
            filter,
            segment,
            None,
        )
    }

    /// Prepares a cluster chosen by the global router.
    pub(crate) fn probe_global(
        &mut self,
        candidate: Candidate,
    ) -> crate::Result<Option<ClusterRows<'_>>> {
        let Candidate { sim, node } = candidate;
        let cluster = node as usize;
        self.meter.open();
        let rows = 0..self.open_cluster(cluster)?;
        if rows.is_empty() {
            self.skip_empty();
            return Ok(None);
        }
        let gated = self.gate_cluster(rows.len());
        if !gated.all && self.offsets.is_empty() {
            self.skip_empty();
            return Ok(None);
        }
        let (offsets, docs) = self.selection(gated);
        Ok(Some(ClusterRows {
            cluster,
            rows,
            offsets,
            docs,
            centroid: Some(sim),
        }))
    }

    /// The gated selection as the scan takes it: offsets and doc ids of the kept rows, or, with
    /// every row kept, no offsets and the whole DocIds column when the segment is replicated.
    fn selection(&mut self, gated: GatedRows) -> (Option<&[u32]>, Option<&[DocId]>) {
        if gated.all {
            return (None, self.replicated.then_some(&self.docs[..]));
        }
        self.selected_docs.clear();
        self.selected_docs.extend(
            self.offsets
                .iter()
                .map(|&offset| self.docs[offset as usize]),
        );
        (Some(&self.offsets[..]), Some(&self.selected_docs[..]))
    }

    /// The row count of `cluster`, with its DocIds column in `docs` when the gate or replication
    /// needs it: the count then comes from the block lookup that reads the column.
    fn open_cluster(&mut self, cluster: usize) -> crate::Result<usize> {
        // DocIds alone, then the gate, before any payload read; a replicated segment reads them for
        // every cluster so the scan can drop a document's second copy.
        if !self.gate.is_open() || self.replicated {
            self.reader.read_doc_ids(cluster, &mut self.docs)?;
            Ok(self.docs.len())
        } else {
            self.index.cluster_num_rows(cluster)
        }
    }

    /// Selects the opened cluster's `rows` into `offsets` and counts what the gate pruned.
    fn gate_cluster(&mut self, rows: usize) -> GatedRows {
        debug_assert!((self.gate.is_open() && !self.replicated) || self.docs.len() == rows);
        let gated = {
            let _routing = enter_vector_stage(VectorStage::Routing);
            self.gate.select(&self.docs, &mut self.offsets)
        };
        self.counters.vectors_visited += rows;
        self.counters.pruned_filter += gated.pruned_filter;
        self.counters.pruned_dead += gated.pruned_dead;
        gated
    }

    /// Counts an opened cluster that contributes no rows.
    fn skip_empty(&mut self) {
        self.counters.postings_skipped += 1;
        self.counters.clusters_skipped_empty += 1;
    }

    /// Ends the walk and reports what it counted.
    pub(crate) fn finish(self) -> RouterReport {
        RouterReport {
            counters: self.counters,
            routing: self.ranked.metrics(),
            probe: self.controller.finish(&self.meter),
        }
    }
}

impl ClusterSource for RouterClusters<'_> {
    fn next(&mut self) -> crate::Result<Option<ClusterRows<'_>>> {
        loop {
            let next = {
                let _routing = enter_vector_stage(VectorStage::Routing);
                self.ranked.next()
            };
            let Some(Candidate { sim, node }) = next else {
                return Ok(None);
            };
            if !self.controller.admit(&self.meter) {
                return Ok(None);
            }
            let cluster = node as usize;

            self.meter.open();
            let rows = 0..self.open_cluster(cluster)?;
            if rows.is_empty() {
                self.skip_empty();
                continue;
            }

            let gated = self.gate_cluster(rows.len());
            if !gated.all && self.offsets.is_empty() {
                self.skip_empty();
                continue;
            }
            let (offsets, docs) = self.selection(gated);
            return Ok(Some(ClusterRows {
                cluster,
                rows,
                offsets,
                docs,
                centroid: Some(sim),
            }));
        }
    }

    fn admitted(&mut self, rows: usize) -> crate::Result<()> {
        self.meter.rows(rows);
        self.counters.eligible += rows;
        self.counters.postings_row += 1;
        Ok(())
    }
}

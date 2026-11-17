//! The router cluster source: clusters in the router's order, admitted under the work budget,
//! and the filter.

use std::sync::Arc;

use common::BitSet;

use crate::vector::filter::{DocFilter, GatedRows, RowGate};
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
    visited: BitSet,
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
            visited: BitSet::with_max_value(index.num_clusters() as u32),
        }
    }

    pub(crate) fn replace_ranking(&mut self, ranked: RouterIter) {
        self.ranked = ranked;
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.meter.exhausted()
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

    pub(crate) fn set_filter(&mut self, filter: DocFilter, segment: &'a SegmentReader) {
        self.gate = RowGate::new(filter, segment.alive_bitset());
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
    fn work_meter(&mut self) -> &mut WorkMeter {
        &mut self.meter
    }

    fn next(&mut self) -> crate::Result<Option<ClusterRows<'_>>> {
        loop {
            let next = {
                let _routing = enter_vector_stage(VectorStage::Routing);
                self.ranked.next()
            };
            let Some(Candidate { sim, node }) = next else {
                return Ok(None);
            };
            if !self.visited.insert(node) {
                continue;
            }
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

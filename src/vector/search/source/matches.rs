//! The filter-match source: a sparse filter's own matches, found through the document location
//! map and served one cluster at a time in ascending id. No router or bounds gate; the work meter
//! charges every cluster and row, but every match is scored.

use std::ops::Range;

use crate::vector::search::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::search::source::controller::{WorkBudget, WorkMeter};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::VectorStage;
use crate::{DocId, SegmentReader};

/// What the filter-match source counted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MatchReport {
    /// Live matches with a vector.
    pub(crate) matches: usize,
    /// Live matches without a vector.
    pub(crate) absent: usize,
    /// Clusters holding a match.
    pub(crate) clusters: usize,
    /// Matches pruned as deleted.
    pub(crate) dead: usize,
    /// Work units charged: opens at `x`, scored rows at `(1 - x) / n_avg`.
    pub(crate) work_charged: f32,
    /// The resolved budget the meter charged against, in work units.
    pub(crate) work_budget: f32,
}

/// A filter's live matches with a vector, grouped by cluster in ascending id.
pub(crate) struct MatchClusters<'a> {
    index: &'a ClusterIndex,
    /// Cluster-local row offsets, ascending within each cluster.
    offsets: Vec<u32>,
    /// The doc id of each offset.
    docs: Vec<DocId>,
    /// `(cluster, positions in offsets and docs)`, ascending by cluster.
    clusters: Vec<(usize, Range<usize>)>,
    next: usize,
    meter: WorkMeter,
    report: MatchReport,
}

impl<'a> MatchClusters<'a> {
    /// Prunes the dead among `segment`'s ascending `docs`, locates the rest through the location
    /// map and groups them by cluster; the work they cost is charged against `budget`. `reader` is
    /// the segment's IVF reader.
    pub(crate) fn locate(
        reader: &'a VectorIndexReader,
        mut docs: Vec<DocId>,
        segment: &SegmentReader,
        budget: WorkBudget,
    ) -> crate::Result<Self> {
        let index = reader
            .index()
            .expect("the filter source runs on an IVF segment");
        let _locate = enter_vector_stage(VectorStage::Locate);
        let matched = docs.len();
        if let Some(alive) = segment.alive_bitset() {
            docs.retain(|&doc| alive.is_alive(doc));
        }
        let mut located = Vec::with_capacity(docs.len());
        reader.locate_docs(&docs, &mut located)?;
        // Cluster ids ascend with their row ranges, so this is ascending row order.
        located.sort_unstable_by_key(|&(_, location)| (location.cluster, location.local));
        let report = MatchReport {
            matches: located.len(),
            absent: docs.len() - located.len(),
            clusters: 0,
            dead: matched - docs.len(),
            work_charged: 0.0,
            work_budget: budget.total.to_f32(),
        };
        // The located doc ids, in location order, overwrite the handed ones: never more of them.
        docs.truncate(located.len());
        let mut source = Self {
            index,
            offsets: Vec::with_capacity(located.len()),
            docs,
            clusters: Vec::new(),
            next: 0,
            meter: WorkMeter::new(budget),
            report,
        };
        for (position, &(doc, location)) in located.iter().enumerate() {
            source.offsets.push(location.local);
            source.docs[position] = doc;
            let cluster = location.cluster as usize;
            match source.clusters.last_mut() {
                Some((last, positions)) if *last == cluster => positions.end = position + 1,
                _ => source.clusters.push((cluster, position..position + 1)),
            }
        }
        source.report.clusters = source.clusters.len();
        Ok(source)
    }

    /// Ends the walk and reports what it counted and charged.
    pub(crate) fn finish(self) -> MatchReport {
        debug_assert_eq!(
            self.next,
            self.clusters.len(),
            "every located cluster is served"
        );
        MatchReport {
            work_charged: self.meter.charged().to_f32(),
            ..self.report
        }
    }
}

impl ClusterSource for MatchClusters<'_> {
    fn next(&mut self) -> crate::Result<Option<ClusterRows<'_>>> {
        let Some((cluster, positions)) = self.clusters.get(self.next).cloned() else {
            return Ok(None);
        };
        self.next += 1;
        self.meter.open();
        Ok(Some(ClusterRows {
            cluster,
            rows: self.index.cluster_range(cluster),
            offsets: Some(&self.offsets[positions.clone()]),
            docs: Some(&self.docs[positions]),
            centroid: None,
        }))
    }

    fn admitted(&mut self, rows: usize, _feedback: Feedback) -> crate::Result<()> {
        self.meter.rows(rows);
        Ok(())
    }
}

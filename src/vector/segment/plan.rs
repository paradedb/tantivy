//! Planning one segment's search: the work budget, the filter, and where clusters come from.

use common::BitSet;

use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::query::{for_each_docset_buffered, AllScorer, Weight};
use crate::vector::ivf::{AdaptiveProbeParams, IvfIndex};
use crate::vector::source::controller::{WorkBudget, WorkUnits};
use crate::vector::source::router::DocFilter;
use crate::vector::{enter_vector_stage, Stage};
use crate::{DocId, SegmentReader};

// --- Plan ---

/// Where a segment's clusters come from.
pub(crate) enum Source {
    /// The router's ranking, gated by the filter.
    Router(DocFilter),
}

/// What planning reports for the segment's statistics.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanReport {
    pub(crate) segment_rows: usize,
    pub(crate) segment_clusters: usize,
}

/// One segment's search plan.
pub(crate) struct SegmentPlan {
    pub(crate) budget: WorkBudget,
    pub(crate) source: Source,
    pub(crate) report: PlanReport,
}

impl SegmentPlan {
    /// Resolves the budget once, drains the filter, and picks the source.
    ///
    /// Returns `None` when nothing can match: `top_n` is zero, the segment holds no clusters or
    /// docs, or the filter matches nothing.
    pub(crate) fn build(
        params: &AdaptiveProbeParams,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Option<Self>> {
        let max_doc = segment.max_doc();
        if top_n == 0 || max_doc == 0 || index.num_clusters() == 0 {
            return Ok(None);
        }
        let filter = {
            let _filter = enter_vector_stage(Stage::NonVectorSearch);
            collect_segment_filter(weight, segment, max_doc)?
        };
        if filter.is_empty() {
            return Ok(None);
        }
        let clusters = index.num_clusters();
        // Capacity counts native docs as written: deleted rows still charge on first touch.
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        Ok(Some(Self {
            budget: WorkBudget {
                total: WorkUnits::new(budget),
                per_cluster: WorkUnits::new(x),
                per_row: WorkUnits::new((1.0 - x) / n_avg),
            },
            source: Source::Router(filter),
            report: PlanReport {
                segment_rows: index.num_rows(),
                segment_clusters: clusters,
            },
        }))
    }

    /// Upper bound on rows the scan will admit, for sizing its columns.
    pub(crate) fn capacity_hint(&self) -> usize {
        let rows = (self.budget.total.get() / self.budget.per_row.get()).ceil() as usize;
        rows.min(self.report.segment_rows)
    }
}

// --- Filter ---

/// Drains the filter `DocSet` into a bitset for random membership tests on cluster doc ids.
///
/// A match-all scorer, or a drain that matches every doc, is [`DocFilter::All`]. The bitset
/// allocates `max_doc / 8` bytes whatever the selectivity.
#[inline(never)]
pub(crate) fn collect_segment_filter(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
) -> crate::Result<DocFilter> {
    let mut scorer = weight.scorer(segment_reader, 1.0)?;
    if scorer.is::<AllScorer>() {
        return Ok(DocFilter::All);
    }
    let mut filter = BitSet::with_max_value(max_doc);
    let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
    for_each_docset_buffered(scorer.as_mut(), &mut buffer, |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    });
    if filter.len() == max_doc as usize {
        return Ok(DocFilter::All);
    }
    Ok(DocFilter::Docs(filter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{AllQuery, BooleanQuery, EnableScoring, Occur, Query};
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;
    use crate::vector::tests::exhaustive_params;

    /// The plan resolves the budget once, builds no bitset for match-all filters, sees an empty
    /// filter or `top_n == 0` as nothing to search, and sizes the scan from the budget.
    #[test]
    fn segment_plan() -> crate::Result<()> {
        let fixture = FixtureSpec::ivf(
            Metric::L2,
            (0..9)
                .map(|c| vec![(c % 3) as f32 * 5.0, (c / 3) as f32 * 5.0])
                .collect(),
            (0..90)
                .map(|i| {
                    vec![
                        (i % 3) as f32 * 5.0 + 0.01 * i as f32,
                        ((i / 3) % 3) as f32 * 5.0,
                    ]
                })
                .collect(),
        )
        .label("tenth", |i| i % 10 == 0)
        .label("rest", |i| i % 10 != 0)
        .label("every", |_| true)
        .build()?;
        let label = |text: &str| -> Box<dyn Query> { Box::new(fixture.filter(text)) };
        let all_and_all = BooleanQuery::new(vec![
            (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
            (Occur::Should, Box::new(AllQuery)),
        ]);
        let all_or_label = BooleanQuery::new(vec![
            (Occur::Should, Box::new(AllQuery) as Box<dyn Query>),
            (Occur::Should, label("tenth")),
        ]);
        let labels_covering_all = BooleanQuery::new(vec![
            (Occur::Should, label("tenth")),
            (Occur::Should, label("rest")),
        ]);
        let params = exhaustive_params(9);
        let plan_for = |filter: &dyn Query, top_n: usize| {
            fixture.with_segment(filter, &[0.0, 0.0], top_n, &params, |seg| {
                Ok(seg.plan()?.map(|plan| {
                    let Source::Router(filter) = &plan.source;
                    let matches = match filter {
                        DocFilter::All => None,
                        DocFilter::Docs(docs) => Some(docs.len()),
                    };
                    (plan.budget, plan.capacity_hint(), plan.report, matches)
                }))
            })
        };
        let every = label("every");
        for filter in [
            &AllQuery as &dyn Query,
            &all_and_all,
            &all_or_label,
            &labels_covering_all,
            every.as_ref(),
        ] {
            let (_, _, _, matches) = plan_for(filter, 10)?.expect("a plan");
            assert_eq!(matches, None, "match-all builds no bitset");
        }
        assert!(plan_for(label("no_such_label").as_ref(), 10)?.is_none());
        assert!(
            plan_for(&AllQuery, 0)?.is_none(),
            "top_n == 0 searches nothing"
        );

        let searcher = fixture.index.reader()?.searcher();
        let segment = &searcher.segment_readers()[0];
        let selective = label("tenth");
        let weight = selective.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let count = weight.count(segment)? as usize;
        let (work, capacity, report, matches) = plan_for(selective.as_ref(), 10)?.expect("a plan");
        assert_eq!(matches, Some(count));
        let vectors = segment.vector_index(fixture.field)?;
        let ivf = vectors.index().unwrap();
        let (budget, n_avg, x) = params.resolved_work_budget(ivf.num_clusters(), ivf.num_docs())?;
        assert_eq!(work.total.get(), budget);
        assert_eq!(work.per_cluster.get(), x);
        assert_eq!(work.per_row.get(), (1.0 - x) / n_avg);
        assert_eq!(
            capacity,
            ((budget / work.per_row.get()).ceil() as usize).min(ivf.num_rows())
        );
        assert_eq!(report.segment_rows, ivf.num_rows());
        assert_eq!(report.segment_clusters, ivf.num_clusters());

        let (hits, stats) =
            fixture.run_segment(0, fixture.all()?.as_ref(), &[0.0, 0.0], 0, params)?;
        assert!(hits.is_empty());
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }
}

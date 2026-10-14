//! Planning one segment's search: the work budget, the filter, and where clusters come from.

use common::BitSet;

use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::query::{for_each_docset_buffered, AllScorer, Weight};
use crate::vector::ivf::{AdaptiveProbeParams, IvfIndex, LocatedPath};
use crate::vector::source::controller::{WorkBudget, WorkUnits};
use crate::vector::source::router::DocFilter;
use crate::vector::{enter_vector_stage, Stage};
use crate::{DocId, SegmentReader};

// --- Plan ---

/// Where a segment's clusters come from.
pub(crate) enum Source {
    /// The router's ranking, gated by the filter.
    Router(DocFilter),
    /// The filter's matches, few enough to look up directly: ascending, deleted ones included.
    FilterMatches(Vec<DocId>),
}

impl Source {
    /// Whether the filter matches nothing.
    fn is_empty(&self) -> bool {
        match self {
            Self::Router(filter) => filter.is_empty(),
            Self::FilterMatches(docs) => docs.is_empty(),
        }
    }
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
    /// Resolves the budget once, drains the filter under the located cap, and picks the source.
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
        let clusters = index.num_clusters();
        // Capacity counts native docs as written: deleted rows still charge on first touch.
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let cap = located_cap(params.located, max_doc);
        let source = {
            let _filter = enter_vector_stage(Stage::NonVectorSearch);
            collect_segment_filter(weight, segment, max_doc, cap)?
        };
        if source.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            budget: WorkBudget {
                total: WorkUnits::new(budget),
                per_cluster: WorkUnits::new(x),
                per_row: WorkUnits::new((1.0 - x) / n_avg),
            },
            source,
            report: PlanReport {
                segment_rows: index.num_rows(),
                segment_clusters: clusters,
            },
        }))
    }

    /// Upper bound on rows the scan will admit, for sizing its columns.
    pub(crate) fn capacity_hint(&self) -> usize {
        let rows = match &self.source {
            Source::Router(_) => {
                (self.budget.total.get() / self.budget.per_row.get()).ceil() as usize
            }
            Source::FilterMatches(docs) => docs.len(),
        };
        rows.min(self.report.segment_rows)
    }
}

/// The most matches, deleted ones included, a filter may have and take the located path.
fn located_cap(located: LocatedPath, max_doc: DocId) -> usize {
    match located {
        LocatedPath::Off => 0,
        LocatedPath::MaxSelectivity(selectivity) => {
            (f64::from(selectivity.get()) * f64::from(max_doc)).floor() as usize
        }
    }
}

// --- Filter ---

/// Drains the filter `DocSet` once, in ascending doc order, keeping up to `cap` matches as a list.
///
/// The first match past `cap` moves the list into a bitset and the drain continues there. A
/// match-all scorer, or a drain that matches every doc, routes unfiltered. The bitset allocates
/// `max_doc / 8` bytes whatever the selectivity.
#[inline(never)]
pub(crate) fn collect_segment_filter(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
    cap: usize,
) -> crate::Result<Source> {
    let mut scorer = weight.scorer(segment_reader, 1.0)?;
    if scorer.is::<AllScorer>() {
        return Ok(Source::Router(DocFilter::All));
    }
    let mut sparse = Vec::new();
    let mut dense = (cap == 0).then(|| BitSet::with_max_value(max_doc));
    let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
    for_each_docset_buffered(scorer.as_mut(), &mut buffer, |docs| {
        for &doc in docs {
            match &mut dense {
                Some(filter) => {
                    filter.insert(doc);
                }
                None if sparse.len() < cap => sparse.push(doc),
                None => {
                    let mut filter = BitSet::with_max_value(max_doc);
                    for &doc in &sparse {
                        filter.insert(doc);
                    }
                    filter.insert(doc);
                    sparse = Vec::new();
                    dense = Some(filter);
                }
            }
        }
    });
    let matched = dense.as_ref().map_or(sparse.len(), BitSet::len);
    if matched == max_doc as usize {
        return Ok(Source::Router(DocFilter::All));
    }
    Ok(match dense {
        Some(filter) => Source::Router(DocFilter::Docs(filter)),
        None => Source::FilterMatches(sparse),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{AllQuery, BooleanQuery, Occur, Query};
    use crate::schema::Metric;
    use crate::vector::ivf::{AdaptiveProbeParams, Selectivity};
    use crate::vector::test_support::{DocsWeight, FixtureSpec};
    use crate::vector::tests::exhaustive_params;

    /// The plan resolves the budget once, builds no bitset for match-all filters, sees an empty
    /// filter or `top_n == 0` as nothing to search, and sizes the scan from the budget. A filter
    /// with at most the located cap of matches, deleted ones included, is located; one more
    /// spills into the routed bitset, and match-all always routes.
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
        .deletes(&[0, 10])
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
        // The source as `(located, matches)`; `None` matches for a match-all routed filter.
        let plan_with = |filter: &dyn Query, top_n: usize, params: &AdaptiveProbeParams| {
            fixture.with_segment(filter, &[0.0, 0.0], top_n, params, |seg| {
                Ok(seg.plan()?.map(|plan| {
                    let source = match &plan.source {
                        Source::Router(filter) => match filter {
                            DocFilter::All => (false, None),
                            DocFilter::Docs(docs) => (false, Some(docs.len())),
                        },
                        Source::FilterMatches(docs) => (true, Some(docs.len())),
                    };
                    (plan.budget, plan.capacity_hint(), plan.report, source)
                }))
            })
        };
        let plan_for = |filter: &dyn Query, top_n: usize| plan_with(filter, top_n, &params);
        let every = label("every");
        for filter in [
            &AllQuery as &dyn Query,
            &all_and_all,
            &all_or_label,
            &labels_covering_all,
            every.as_ref(),
        ] {
            let (_, _, _, source) = plan_for(filter, 10)?.expect("a plan");
            assert_eq!(source, (false, None), "match-all builds no bitset");
        }
        assert!(plan_for(label("no_such_label").as_ref(), 10)?.is_none());
        assert!(
            plan_for(&AllQuery, 0)?.is_none(),
            "top_n == 0 searches nothing"
        );

        let searcher = fixture.index.reader()?.searcher();
        let segment = &searcher.segment_readers()[0];
        let selective = label("tenth");
        // The drain keeps deleted matches: docs 0 and 10 count.
        let count = 9;
        let (work, capacity, report, source) = plan_for(selective.as_ref(), 10)?.expect("a plan");
        assert_eq!(source, (false, Some(count)));
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

        // The located cap is `floor(selectivity * max_doc)`.
        let max_doc = segment.max_doc() as f32;
        let capped = |cap: usize| AdaptiveProbeParams {
            located: LocatedPath::MaxSelectivity(
                Selectivity::new(((cap as f32 + 0.5) / max_doc).min(1.0)).unwrap(),
            ),
            ..params.clone()
        };
        for (name, filter, located, expected) in [
            (
                "matches == cap",
                selective.as_ref(),
                capped(count),
                (true, Some(count)),
            ),
            (
                "cap + 1 spills",
                selective.as_ref(),
                capped(count - 1),
                (false, Some(count)),
            ),
            (
                "match-all routes",
                &AllQuery as &dyn Query,
                capped(90),
                (false, None),
            ),
        ] {
            let (_, capacity, _, source) = plan_with(filter, 10, &located)?.expect("a plan");
            assert_eq!(source, expected, "{name}");
            if source.0 {
                assert_eq!(capacity, count, "{name}: located columns hold the matches");
            }
        }
        // A drain matching every doc routes unfiltered, whatever the cap.
        let every_doc = DocsWeight {
            max_doc: segment.max_doc(),
            docs: (0..segment.max_doc()).collect(),
        };
        for cap in [0, 90] {
            let source = collect_segment_filter(&every_doc, segment, segment.max_doc(), cap)?;
            assert!(
                matches!(source, Source::Router(DocFilter::All)),
                "cap {cap}"
            );
        }
        assert_eq!(Selectivity::new(0.0), None);
        assert_eq!(Selectivity::new(1.5), None);
        assert_eq!(Selectivity::new(f32::NAN), None);

        let (hits, stats) =
            fixture.run_segment(0, fixture.all()?.as_ref(), &[0.0, 0.0], 0, params)?;
        assert!(hits.is_empty());
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }
}

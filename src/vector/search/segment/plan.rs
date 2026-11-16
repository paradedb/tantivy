//! Planning one segment's search: the work budget, the filter, and where clusters come from.

use crate::vector::filter::DocFilter;
use crate::vector::params::LocatedPath;
use crate::vector::search::source::controller::WorkBudget;
use crate::DocId;

// --- Plan ---

/// What planning reports for the segment's statistics.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanReport {
    pub(crate) segment_rows: usize,
    pub(crate) segment_clusters: usize,
    pub(crate) planned_clusters: usize,
    /// The located-path cap in matches; 0 with the located path off.
    pub(crate) located_cap: usize,
}

#[derive(Clone)]
pub(crate) enum ScanInput {
    Flat {
        filter: DocFilter,
    },
    Routed {
        filter: DocFilter,
        budget: WorkBudget,
        report: PlanReport,
    },
    Located {
        docs: Vec<DocId>,
        budget: WorkBudget,
        report: PlanReport,
    },
}

impl ScanInput {
    pub(crate) fn capacity_hint(&self) -> usize {
        match self {
            Self::Routed { budget, report, .. } => ((budget.total.get() / budget.per_row.get())
                .ceil() as usize)
                .min(report.segment_rows),
            Self::Located { docs, report, .. } => docs.len().min(report.segment_rows),
            Self::Flat { .. } => 0,
        }
    }
}

/// The most matches, deleted ones included, a filter may have and take the located path; `Auto`
/// prices the segment's size share of the global work budget `(budget, n_avg, x)`.
pub(crate) fn located_cap(
    located: LocatedPath,
    max_doc: DocId,
    budget: f64,
    n_avg: f64,
    x: f64,
) -> usize {
    match located {
        LocatedPath::Off => 0,
        LocatedPath::MaxSelectivity(selectivity) => {
            (f64::from(selectivity.get()) * f64::from(max_doc)).floor() as usize
        }
        LocatedPath::Auto => (budget * n_avg / (1.0 - x)).floor() as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{AllQuery, BooleanQuery, Occur, Query};
    use crate::schema::Metric;
    use crate::vector::filter::{FilterInput, FilterMatches};
    use crate::vector::params::{ProbeParams, Selectivity};
    use crate::vector::stats::AccessPath;
    use crate::vector::test_support::{first_segment, DocsWeight, FixtureSpec};
    use crate::vector::tests::exhaustive_params;
    use crate::vector::VectorFilterStrategy;

    /// The plan resolves the budget once, builds no bitset for match-all filters, sees an empty
    /// filter or `top_n == 0` as nothing to search, and sizes the scan from the budget. A filter
    /// with at most the located cap of matches, deleted ones included, is located; one more
    /// routes through the bitset, match-all always routes, and every mode reports its cap.
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
        let params = ProbeParams {
            exact_plan: false,
            located: LocatedPath::Off,
            ..exhaustive_params(9)
        };
        // The source as `(located, matches)`; `None` matches for a match-all routed filter.
        let plan_with = |filter: &dyn Query, top_n: usize, params: &ProbeParams| {
            fixture.with_segment(filter, &[0.0, 0.0], top_n, params, |seg| {
                Ok(seg.input()?.map(|input| {
                    let capacity = input.capacity_hint();
                    let (budget, report, source) = match input {
                        ScanInput::Routed {
                            filter,
                            budget,
                            report,
                        } => {
                            let matches = match filter {
                                DocFilter::All => None,
                                DocFilter::Late => unreachable!(),
                                DocFilter::Docs(docs) => Some(docs.len()),
                                DocFilter::Pending => unreachable!(),
                            };
                            (budget, report, (false, matches))
                        }
                        ScanInput::Located {
                            docs,
                            budget,
                            report,
                        } => (budget, report, (true, Some(docs.len()))),
                        ScanInput::Flat { .. } => unreachable!(),
                    };
                    (budget, capacity, report, source)
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
        let segment = first_segment(&searcher).1;
        let selective = label("tenth");
        // The drain keeps deleted matches: docs 0 and 10 count.
        let count = 9;
        let (work, capacity, report, source) = plan_for(selective.as_ref(), 10)?.expect("a plan");
        assert_eq!(source, (false, Some(count)));
        let vectors = segment.vector_index(fixture.field)?;
        let ivf = vectors.index().unwrap();
        let (budget, n_avg, x) = params.resolved_work_budget(ivf.num_clusters(), ivf.num_rows())?;
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
        let capped = |cap: usize| ProbeParams {
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
                "cap + 1 routes",
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
            let source = VectorFilterStrategy::Pre {
                uses_fast_fields: false,
            }
            .collect_matches(
                &every_doc,
                segment,
                FilterInput::Segment {
                    vectors: &vectors,
                    located_cap: cap,
                },
            )?;
            assert!(
                matches!(source, FilterMatches::Routed(DocFilter::All)),
                "cap {cap}"
            );
        }
        assert_eq!(Selectivity::new(0.0), None);
        assert_eq!(Selectivity::new(1.5), None);
        assert_eq!(Selectivity::new(f32::NAN), None);

        // `Auto` caps at the rows the routed budget buys: matches at the cap are located, one more
        // routes, and a filter matching nothing still reports the cap.
        let ivf_rows = ivf.num_rows();
        let auto = ProbeParams {
            max_probe: 2,
            min_probe_clusters: 1,
            located: LocatedPath::Auto,
            ..ProbeParams::prefilter()
        };
        let (budget, n_avg, x) = auto.resolved_work_budget(ivf.num_clusters(), ivf_rows)?;
        let cap = (budget * n_avg / (1.0 - x)).floor() as usize;
        assert!(cap > 0 && cap + 1 < segment.max_doc() as usize, "cap {cap}");
        for (matches, path) in [
            (cap, AccessPath::Located),
            (cap + 1, AccessPath::Routed),
            (0, AccessPath::Routed),
        ] {
            let weight = DocsWeight {
                max_doc: segment.max_doc(),
                docs: (0..matches as DocId).collect(),
            };
            let (_, stats) = fixture.run_segment(0, &weight, &[0.0, 0.0], 10, auto.clone())?;
            assert_eq!(
                (stats.located_cap, stats.access_path),
                (cap, path),
                "matches={matches}"
            );
        }
        // Every mode reports its cap, and an unfiltered query routes.
        for (located, cap) in [(LocatedPath::Off, 0), (capped(9).located, 9)] {
            let params = ProbeParams {
                located,
                ..params.clone()
            };
            let (_, stats) =
                fixture.run_segment(0, fixture.all()?.as_ref(), &[0.0, 0.0], 10, params)?;
            assert_eq!(
                (stats.located_cap, stats.access_path),
                (cap, AccessPath::Routed)
            );
        }

        let (hits, stats) =
            fixture.run_segment(0, fixture.all()?.as_ref(), &[0.0, 0.0], 0, params)?;
        assert!(hits.is_empty());
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }
}

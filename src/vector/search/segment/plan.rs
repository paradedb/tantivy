//! Planning one segment's search: the work budget, the filter, and where clusters come from.

use common::{BitSet, HasLen};

use crate::directory::FileSlice;
use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::query::{for_each_docset_buffered, AllScorer, Weight};
use crate::vector::params::{LocatedPath, ProbeParams, QuantizedRecallParams};
use crate::vector::search::source::controller::{WorkBudget, WorkUnits};
use crate::vector::search::source::router::DocFilter;
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::VectorStage;
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
    /// The located-path cap in matches; 0 with the located path off.
    pub(crate) located_cap: usize,
}

/// One segment's search plan.
pub(crate) struct SegmentPlan {
    /// Zero when nothing can match before the budget resolves.
    pub(crate) budget: WorkBudget,
    /// `None` when nothing can match.
    pub(crate) source: Option<Source>,
    pub(crate) report: PlanReport,
}

impl SegmentPlan {
    /// Resolves the budget once, drains the filter under the located cap, and picks the source.
    ///
    /// Plans no source when nothing can match: `top_n` is zero, the segment holds no clusters or
    /// docs, or the filter matches nothing.
    pub(crate) fn build(
        params: &ProbeParams,
        index: &ClusterIndex,
        weight: &dyn Weight,
        segment: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Self> {
        let max_doc = segment.max_doc();
        let clusters = index.num_clusters();
        let report = |located_cap| PlanReport {
            segment_rows: index.num_rows(),
            segment_clusters: clusters,
            located_cap,
        };
        if top_n == 0 || max_doc == 0 || clusters == 0 {
            return Ok(Self {
                budget: WorkBudget::default(),
                source: None,
                report: report(0),
            });
        }
        // Capacity counts native docs as written: deleted rows still charge on first touch.
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let cap =
            if let (Some(recall), LocatedPath::Auto) = (params.quantized_recall, params.located) {
                adaptive_located_cap(recall, clusters, max_doc, &index.centroid_row_slice(0)?)
            } else {
                located_cap(params.located, max_doc, budget, n_avg, x)
            };
        let source = {
            let _filter = enter_vector_stage(VectorStage::NonVectorSearch);
            collect_segment_filter(weight, segment, max_doc, cap)?
        };
        Ok(Self {
            budget: WorkBudget {
                total: WorkUnits::new(budget),
                per_cluster: WorkUnits::new(x),
                per_row: WorkUnits::new((1.0 - x) / n_avg),
            },
            source: (!source.is_empty()).then_some(source),
            report: report(cap),
        })
    }

    /// The rows the scan admits under the budget, for reserving its columns once. The last
    /// cluster can overshoot the budget, growing the columns once more.
    pub(crate) fn capacity_hint(&self) -> usize {
        let rows = match &self.source {
            Some(Source::Router(_)) => {
                (self.budget.total.get() / self.budget.per_row.get()).ceil() as usize
            }
            Some(Source::FilterMatches(docs)) => docs.len(),
            None => 0,
        };
        rows.min(self.report.segment_rows)
    }
}

fn adaptive_located_cap(
    recall: QuantizedRecallParams,
    clusters: usize,
    max_doc: DocId,
    centroid: &FileSlice,
) -> usize {
    let baseline = -1.5 * (-0.95_f64).ln_1p();
    let pressure = recall.gap_threshold(1).unwrap_or(f64::INFINITY) / baseline;
    let per_cluster = centroid
        .storage_block_len()
        .map_or(0.5, |block| (centroid.len() as f64 / block as f64).min(0.5));
    (clusters as f64 * per_cluster * pressure).min(f64::from(max_doc)) as usize
}

/// The most matches, deleted ones included, a filter may have and take the located path; `Auto`
/// prices the work budget `(budget, n_avg, x)` the routed path runs against.
fn located_cap(located: LocatedPath, max_doc: DocId, budget: f64, n_avg: f64, x: f64) -> usize {
    match located {
        LocatedPath::Off => 0,
        LocatedPath::MaxSelectivity(selectivity) => {
            (f64::from(selectivity.get()) * f64::from(max_doc)).floor() as usize
        }
        LocatedPath::Auto => (budget * n_avg / (1.0 - x)).floor() as usize,
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
    use crate::vector::params::{ProbeParams, Selectivity};
    use crate::vector::stats::AccessPath;
    use crate::vector::test_support::{DocsWeight, FixtureSpec};
    use crate::vector::tests::exhaustive_params;

    #[test]
    fn adaptive_bypass_prices_centroid_pages_without_reading_them() {
        use std::sync::Arc;

        use crate::vector::storage::io::test_support::BlockTrackedBytes;

        let recall = QuantizedRecallParams {
            target: 0.95,
            recall_exponent: 1.5,
        };
        for (dims, block, cap) in [
            (128, 8192, 1000),
            (128, 16384, 500),
            (1024, 8192, 8000),
            (2048, 8192, 8000),
        ] {
            let reads = Default::default();
            let centroid = FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: vec![0; dims * 4],
                reads: std::sync::Arc::clone(&reads),
                block_len: block,
            }));
            assert_eq!(
                adaptive_located_cap(recall, 16_000, 1_000_000, &centroid),
                cap
            );
            assert_eq!(
                adaptive_located_cap(
                    QuantizedRecallParams {
                        recall_exponent: 3.0,
                        ..recall
                    },
                    16_000,
                    1_000_000,
                    &centroid
                ),
                2 * cap
            );
            assert_eq!(
                adaptive_located_cap(
                    QuantizedRecallParams {
                        target: 1.0,
                        ..recall
                    },
                    16_000,
                    1_000_000,
                    &centroid
                ),
                1_000_000
            );
            assert!(reads.lock().unwrap().is_empty());
        }
    }

    /// The plan resolves the budget once, builds no bitset for match-all filters, sees an empty
    /// filter or `top_n == 0` as nothing to search, and sizes the scan from the budget. A filter
    /// with at most the located cap of matches, deleted ones included, is located; one more
    /// spills into the routed bitset, match-all always routes, and every mode reports its cap.
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
                let plan = seg.plan()?;
                Ok(plan.source.as_ref().map(|source| {
                    let source = match source {
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

        // `Auto` caps at the rows the routed budget buys: matches at the cap are located, one more
        // routes, and a filter matching nothing still reports the cap.
        let ivf_docs = ivf.num_docs();
        let auto = ProbeParams {
            max_probe_fraction: 0.2,
            min_probe_clusters: 1,
            located: LocatedPath::Auto,
            ..ProbeParams::default()
        };
        let (budget, n_avg, x) = auto.resolved_work_budget(ivf.num_clusters(), ivf_docs)?;
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
        assert_eq!(ivf.num_clusters(), 9);
        for (target, recall_exponent, adaptive_cap) in [
            (0.90, 1.5, 3),
            (0.95, 1.5, 4),
            (0.99, 1.5, 6),
            (0.95, 3.0, 9),
            (1.0, 1.5, 90),
        ] {
            for (max_probe_fraction, min_probe_clusters) in [(0.000001, 0), (1.0, usize::MAX)] {
                let adaptive = ProbeParams {
                    quantized_recall: Some(crate::vector::QuantizedRecallParams {
                        target,
                        recall_exponent,
                    }),
                    max_probe_fraction,
                    min_probe_clusters,
                    ..auto.clone()
                };
                for (located, matches, cap, path) in [
                    (
                        LocatedPath::Auto,
                        adaptive_cap,
                        adaptive_cap,
                        if adaptive_cap == segment.max_doc() as usize {
                            AccessPath::Routed
                        } else {
                            AccessPath::Located
                        },
                    ),
                    (
                        LocatedPath::Auto,
                        adaptive_cap + 1,
                        adaptive_cap,
                        AccessPath::Routed,
                    ),
                    (LocatedPath::Auto, 0, adaptive_cap, AccessPath::Routed),
                    (
                        LocatedPath::Auto,
                        segment.max_doc() as usize,
                        adaptive_cap,
                        AccessPath::Routed,
                    ),
                    (LocatedPath::Off, adaptive_cap, 0, AccessPath::Routed),
                    (capped(9).located, 9, 9, AccessPath::Located),
                ]
                .into_iter()
                .filter(|(_, matches, _, _)| *matches <= segment.max_doc() as usize)
                {
                    let weight = DocsWeight {
                        max_doc: segment.max_doc(),
                        docs: (0..matches as DocId).collect(),
                    };
                    let (_, stats) = fixture.run_segment(
                        0,
                        &weight,
                        &[0.0, 0.0],
                        10,
                        ProbeParams {
                            located,
                            ..adaptive.clone()
                        },
                    )?;
                    assert_eq!(
                        (stats.located_cap, stats.access_path),
                        (cap, path),
                        "matches={matches}"
                    );
                }
            }
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

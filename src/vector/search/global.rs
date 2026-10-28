use super::cluster_source::ClusterSource;
use super::scan::heap::TieBreakHits;
use super::scan::{ScanConfig, SegmentScan};
use super::segment::plan::{SegmentPlan, Source};
use super::source::controller::{ProbeTermination, WorkBudget, WorkMeter, WorkUnits};
use super::source::matches::MatchClusters;
use super::source::router::RouterClusters;
use super::source::SourceReport;
use super::SegmentSearcher;
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::query::Weight;
use crate::vector::router::global;
use crate::vector::stats::SegmentReport;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::{VectorElement, VectorStage};
use crate::SegmentReader;

#[allow(clippy::large_enum_variant)]
enum GlobalSource<'a, T: VectorElement> {
    Router(RouterClusters<'a, T>),
    RouterBypass(MatchClusters<'a>),
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search<T, K, C>(
    backends: &[SegmentSearcher<T>],
    segments: &[&SegmentReader],
    weight: &dyn Weight,
    top_n: usize,
    tie_breaks: &mut [K],
    comparator: impl Fn() -> C,
    totals: &mut [StageTotals],
    on_segment: &mut impl FnMut(usize),
) -> crate::Result<Vec<(TieBreakHits<K>, SegmentReport)>>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    let clusters = backends
        .iter()
        .filter_map(|b| b.reader.index())
        .map(|i| i.num_clusters())
        .sum();
    let docs = backends
        .iter()
        .filter_map(|b| b.reader.index())
        .map(|i| i.num_docs())
        .sum();
    let Some(first) = backends.first() else {
        return Ok(Vec::new());
    };
    let (total, n_avg, x) = first.probe_params.resolved_work_budget(clusters, docs)?;
    let global_budget = WorkBudget {
        total: WorkUnits::new(total),
        per_cluster: WorkUnits::new(x),
        per_row: WorkUnits::new((1.0 - x) / n_avg),
    };
    let mut exact = Vec::with_capacity(top_n);
    let mut plans = Vec::with_capacity(backends.len());
    let mut finished = Vec::with_capacity(backends.len());
    for (i, ((backend, segment), tie_break)) in backends
        .iter()
        .zip(segments)
        .zip(tie_breaks.iter_mut())
        .enumerate()
    {
        let (result, measured) = measure(|| -> crate::Result<_> {
            let _init = enter_vector_stage(VectorStage::ScanInit);
            match backend.reader.index() {
                Some(index) => {
                    let budget = WorkBudget {
                        total: global_budget.total * (index.num_docs() as f64 / docs.max(1) as f64),
                        ..global_budget
                    };
                    let plan = SegmentPlan::with_budget(
                        &backend.probe_params,
                        index,
                        weight,
                        segment,
                        top_n,
                        budget,
                    )?;
                    Ok((Some(plan), None))
                }
                None => {
                    let (hits, report) =
                        backend.top_n_by(weight, segment, top_n, tie_break, comparator())?;
                    exact.extend(hits.iter().map(|((score, _), _)| *score));
                    Ok((None, Some((hits, report))))
                }
            }
        });
        totals[i].append(&measured);
        on_segment(i);
        let (plan, result) = result?;
        plans.push(plan);
        finished.push(result);
    }
    exact.sort_unstable_by(|a, b| b.total_cmp(a));
    exact.truncate(top_n);
    let routers: Vec<_> = backends
        .iter()
        .zip(&plans)
        .map(|(backend, plan)| {
            matches!(
                plan.as_ref().and_then(|p| p.source.as_ref()),
                Some(Source::Router(_))
            )
            .then(|| backend.reader.index().unwrap().opened_router())
        })
        .collect();
    let (matched, routed_docs) =
        plans
            .iter()
            .zip(segments)
            .fold((0.0, 0u64), |(matched, docs), (plan, segment)| {
                match plan.as_ref().and_then(|plan| plan.source.as_ref()) {
                    Some(Source::Router(filter)) => (
                        matched
                            + filter.match_fraction(segment.max_doc())
                                * f64::from(segment.max_doc()),
                        docs + u64::from(segment.max_doc()),
                    ),
                    _ => (matched, docs),
                }
            });
    let k = first
        .probe_params
        .router_k(total, x, matched / routed_docs.max(1) as f64, clusters);
    let ranking = global::rank(
        &routers,
        first.query.widened(),
        first.query.metric(),
        k,
        totals,
        on_segment,
    );
    let mut reserved = WorkUnits::ZERO;
    let mut scans = Vec::with_capacity(backends.len());
    let mut sources = Vec::with_capacity(backends.len());
    let mut reports = Vec::with_capacity(backends.len());
    for (i, (((backend, segment), plan), tie_break)) in backends
        .iter()
        .zip(segments)
        .zip(plans)
        .zip(tie_breaks.iter_mut())
        .enumerate()
    {
        let (result, measured) = measure(|| -> crate::Result<_> {
            let _init = enter_vector_stage(VectorStage::ScanInit);
            let Some(plan) = plan else {
                return Ok((None, None, None));
            };
            let capacity = plan.capacity_hint().min(4096);
            let SegmentPlan {
                budget,
                source,
                report,
            } = plan;
            let Some(source) = source else {
                finished[i] = Some((
                    Vec::new(),
                    SegmentReport::Empty {
                        located_cap: report.located_cap,
                    },
                ));
                return Ok((None, None, None));
            };
            let config = ScanConfig {
                top_n,
                capacity,
                exact_plan: backend.probe_params.exact_plan,
                #[cfg(test)]
                plan_override: backend.plan_override,
                #[cfg(test)]
                admission_pruning_disabled: backend.admission_pruning_disabled,
            };
            let scan = SegmentScan::new(
                &backend.reader,
                backend.reader.index().unwrap(),
                backend.scoring_query(),
                config,
                tie_break,
                comparator(),
            )?;
            let source = match source {
                Source::Router(filter) => GlobalSource::Router(RouterClusters::for_global(
                    &backend.reader,
                    &backend.query,
                    budget,
                    filter,
                    segment,
                    ranking.metrics[i],
                )),
                Source::FilterMatches(docs) => {
                    reserved += budget.total;
                    GlobalSource::RouterBypass(MatchClusters::locate(
                        &backend.reader,
                        docs,
                        segment,
                        budget,
                    )?)
                }
            };
            Ok((Some(scan), Some(source), Some(report)))
        });
        totals[i].append(&measured);
        on_segment(i);
        let (scan, source, report) = result?;
        scans.push(scan);
        sources.push(source);
        reports.push(report);
    }
    let mut meter = WorkMeter::new(WorkBudget {
        total: WorkUnits::new((global_budget.total.get() - reserved.get()).max(0.0)),
        ..global_budget
    });
    for (i, (source, scan)) in sources.iter_mut().zip(&mut scans).enumerate() {
        let (Some(GlobalSource::RouterBypass(source)), Some(scan)) = (source, scan) else {
            continue;
        };
        scan.swap_exact(&mut exact);
        let (result, measured) = measure(|| {
            while let Some(rows) = source.next(scan.feedback())? {
                let scored = scan.run_cluster(&rows)?;
                source.admitted(scored, scan.feedback())?;
            }
            Ok::<_, crate::TantivyError>(())
        });
        scan.swap_exact(&mut exact);
        totals[i].append(&measured);
        on_segment(i);
        result?;
    }
    for (i, candidate) in ranking.clusters {
        if meter.exhausted() {
            break;
        }
        let Some(GlobalSource::Router(source)) = &mut sources[i] else {
            unreachable!()
        };
        let scan = scans[i].as_mut().unwrap();
        scan.swap_exact(&mut exact);
        let (result, measured) = measure(|| {
            meter.open();
            if let Some(rows) = source.probe_global(candidate, scan.feedback())? {
                let scored = scan.run_cluster(&rows)?;
                source.admitted(scored, scan.feedback())?;
                meter.rows(scored);
            }
            Ok::<_, crate::TantivyError>(())
        });
        scan.swap_exact(&mut exact);
        totals[i].append(&measured);
        on_segment(i);
        result?;
    }
    for (i, ((scan, source), report)) in scans.into_iter().zip(sources).zip(reports).enumerate() {
        let Some(scan) = scan else {
            continue;
        };
        let (result, measured) = measure(|| {
            let source = match source.unwrap() {
                GlobalSource::Router(source) => {
                    let mut report = source.finish();
                    report.probe.termination = if meter.exhausted() {
                        ProbeTermination::Ceiling
                    } else {
                        ProbeTermination::Exhausted
                    };
                    SourceReport::Router(report)
                }
                GlobalSource::RouterBypass(source) => SourceReport::Matches(source.finish()),
            };
            let (hits, scan) = scan.finish(backends[i].segment_ord)?;
            Ok::<_, crate::TantivyError>((
                hits,
                SegmentReport::Ivf {
                    plan: report.unwrap(),
                    source,
                    scan,
                },
            ))
        });
        totals[i].append(&measured);
        on_segment(i);
        finished[i] = Some(result?);
    }
    Ok(finished.into_iter().map(Option::unwrap).collect())
}

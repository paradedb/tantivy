use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use common::BitSet;

use super::backend::{ProbeStats, ProbeTermination};
use super::prepared::PreparedQuery;
use super::{
    build_filter_bitset, collect_cluster_survivors, FilterState, GlobalHeap, SegmentSearch,
    Survivor,
};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::query::Weight;
use crate::schema::{Field, FieldType, Metric};
use crate::vector::distance::norm_squared_wide;
use crate::vector::ivf::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, QueryBound, Verdict,
};
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::router::RouterWorkspace;
use crate::vector::{RouterMetrics, VectorElement};
use crate::{DocAddress, Score, Searcher, SegmentOrdinal, TantivyError};

pub const PROBE_WAVE_SIZE: usize = 256;
const PROBE_BATCH_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RankedCluster {
    pub id: u32,
    pub similarity: Score,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ProbeBudget {
    pub limit: f64,
    pub open: f64,
    pub row: f64,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ClusterWork {
    pub opens: u64,
    pub rows: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeWave {
    pub end: usize,
    pub spent: f64,
}

impl ProbeBudget {
    pub fn charge(self, cost: ClusterWork) -> f64 {
        cost.opens as f64 * self.open + cost.rows as f64 * self.row
    }

    pub fn select(self, start: usize, costs: &[ClusterWork], mut spent: f64) -> ProbeWave {
        let mut end = start;
        for &cost in costs {
            if spent >= self.limit {
                break;
            }
            spent += self.charge(cost);
            end += 1;
        }
        ProbeWave { end, spent }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedVectorSearch {
    pub clusters: Vec<RankedCluster>,
    pub num_centroids: usize,
    pub shareable: bool,
    pub budget: ProbeBudget,
    pub initial_wave: Option<ProbeWave>,
    pub routing: Option<RouterMetrics>,
    pub routing_time_ns: u64,
}

pub trait VectorSearchControl {
    fn work_sharing(&self) -> bool {
        false
    }
    fn claim_work(&mut self, _segment: SegmentOrdinal) -> usize {
        unreachable!()
    }
    fn publish_initial_wave(&mut self, _wave: ProbeWave) {}
    fn begin(&mut self) {}
    fn end(&mut self) {}
    fn select_wave(
        &mut self,
        _segments: usize,
        start: usize,
        costs: &[ClusterWork],
        budget: ProbeBudget,
        spent: f64,
    ) -> ProbeWave {
        budget.select(start, costs, spent)
    }
    fn synchronize(&mut self, _segments: usize, local: Option<Score>) -> Option<Score> {
        local
    }
    fn accept(&mut self, _doc: DocAddress) -> bool {
        true
    }
    fn publish(&mut self, _candidates: &[(Score, DocAddress)]) {}
    fn check_interrupt(&mut self) {}
    fn finish(&mut self, _stats: &ProbeStats) {}
}

impl VectorSearchControl for () {}

impl PreparedVectorSearch {
    pub fn cluster_capacity(searcher: &Searcher, field: Field) -> crate::Result<usize> {
        if searcher.index().load_metas()?.centroid_index.is_none() {
            return Ok(0);
        }
        Ok(searcher
            .index()
            .cached_centroid_index()?
            .field_router(field)
            .map_or(0, |router| router.num_centroids()))
    }

    pub fn new<T: VectorElement>(
        searcher: &Searcher,
        field: Field,
        query: &[T],
        adaptive: &AdaptiveProbeParams,
    ) -> crate::Result<Self> {
        super::collector::check_query_schema(searcher.schema(), field, query)?;
        let started = Instant::now();
        let mut plan = Self {
            clusters: Vec::new(),
            num_centroids: 0,
            shareable: true,
            budget: ProbeBudget::default(),
            initial_wave: None,
            routing: None,
            routing_time_ns: 0,
        };
        let mut options = None;
        let mut docs = 0;
        let mut nonempty = 0;
        for reader in searcher.segment_readers() {
            let vector = reader.vector_index(field)?;
            plan.shareable &= reader.alive_bitset().is_none()
                && vector
                    .clusters()
                    .is_some_and(|ivf| ivf.num_rows() == ivf.num_docs());
            if let Some(ivf) = vector.clusters() {
                options = Some(vector.options().clone());
                docs += ivf.num_docs();
                nonempty += ivf.num_non_empty_clusters();
            }
        }
        let Some(options) = options else {
            return Ok(plan);
        };
        let set = searcher.index().cached_centroid_index()?;
        let router = set.field_router(field).ok_or_else(|| {
            TantivyError::InternalError(format!("missing centroid router for {field:?}"))
        })?;
        plan.num_centroids = router.num_centroids();
        let (limit, open, row) =
            super::resolve_budget_counts(adaptive, plan.num_centroids, docs, nonempty)?;
        plan.budget = ProbeBudget {
            limit: limit.get(),
            open: open.get(),
            row: row.get(),
        };
        let mut values: Vec<f32> = query.iter().map(|value| value.to_f32()).collect();
        if options.metric() == Metric::Cosine {
            let norm = norm_squared_wide(&values).sqrt();
            if norm.is_finite() && norm > 0.0 {
                for value in &mut values {
                    *value = (f64::from(*value) / norm) as f32;
                }
            }
        }
        let mut workspace = RouterWorkspace::default();
        let mut ranked = router.rank_clusters(&mut workspace, &values);
        plan.clusters = ranked
            .by_ref()
            .map(|candidate| RankedCluster {
                id: candidate.node,
                similarity: candidate.sim.score(),
            })
            .collect();
        if plan.shareable && !plan.clusters.is_empty() {
            let mut costs = vec![ClusterWork::default(); plan.clusters.len().min(PROBE_WAVE_SIZE)];
            for reader in searcher.segment_readers() {
                let vector = reader.vector_index(field)?;
                let clusters = vector.clusters().unwrap();
                for (cost, cluster) in costs.iter_mut().zip(&plan.clusters) {
                    if let Some(range) = clusters.non_empty_cluster_range(cluster.id as usize) {
                        cost.opens += 1;
                        cost.rows += range.len() as u64;
                    }
                }
            }
            let wave = plan.budget.select(0, &costs, 0.0);
            plan.initial_wave = (wave.end > 0).then_some(wave);
        }
        plan.routing = Some(ranked.metrics());
        plan.routing_time_ns = started.elapsed().as_nanos() as u64;
        Ok(plan)
    }
}

#[derive(Default)]
struct ClusterProbe {
    survivors: Range<usize>,
    direct: Option<Range<usize>>,
    work: ClusterWork,
    visited: usize,
    filtered: usize,
    dead: usize,
    duplicates: usize,
    skipped: bool,
}

impl ClusterProbe {
    fn prepare<TChild>(
        segment: &mut SegmentSearch<'_, TChild>,
        cluster: &RankedCluster,
        threshold: Option<Score>,
        metric: Metric,
        q_norm: f32,
        scratch: &mut Vec<Survivor>,
        rows: &mut Vec<Survivor>,
    ) -> Self {
        let mut probe = Self::default();
        if let Some(range) = segment.ivf().non_empty_cluster_range(cluster.id as usize) {
            probe.work.opens = 1;
            let bound = threshold
                .filter(|t| t.is_finite())
                .map_or(QueryBound::Filling, |t| QueryBound::Armed {
                    t: to_bound_space(metric, t),
                });
            probe.skipped = !segment.needs_dedup
                && bounds_verdict(bound, || {
                    let QueryBound::Armed { t } = bound else {
                        return f32::INFINITY;
                    };
                    let radius = segment.ivf().bounds().ball_r(cluster.id as usize);
                    match metric {
                        Metric::L2 | Metric::Cosine => {
                            margin_ball_ball(t, radius, to_bound_space(metric, cluster.similarity))
                        }
                        Metric::Dot => margin_ball_halfspace(cluster.similarity, q_norm, radius, t),
                    }
                }) == Verdict::Skip;
            if !probe.skipped {
                let filter = match &segment.filter {
                    FilterState::Built(filter) => Some(filter),
                    _ => None,
                };
                if filter.is_none() && segment.alive.is_none() && !segment.needs_dedup {
                    probe.visited = range.len();
                    probe.work.rows = range.len() as u64;
                    probe.direct = Some(range);
                } else {
                    let (visited, filtered, dead, duplicates, scored) = collect_cluster_survivors(
                        &segment.vec,
                        range,
                        filter,
                        segment.alive,
                        segment.seen.as_mut(),
                        scratch,
                    );
                    probe.visited = visited;
                    probe.filtered = filtered;
                    probe.dead = dead;
                    probe.duplicates = duplicates;
                    probe.work.rows = scored as u64;
                    probe.survivors = rows.len()..rows.len() + scratch.len();
                    rows.extend_from_slice(&scratch);
                }
            }
        }
        probe
    }
}

struct ProbeCollector<'a, T: VectorElement, S: SortKeyComputer> {
    prepared: PreparedQuery<T>,
    heap: GlobalHeap<S, S::Comparator>,
    pending: TopNComputer<Score, DocAddress, NaturalComparator>,
    limit: usize,
    control: &'a mut dyn VectorSearchControl,
    stats: ProbeStats,
    work: ClusterWork,
}

impl<T: VectorElement, S: SortKeyComputer> ProbeCollector<'_, T, S> {
    fn score(
        &mut self,
        segment: &mut SegmentSearch<'_, S::Child>,
        survivors: &[Survivor],
        threshold: Option<Score>,
    ) -> crate::Result<()> {
        let stride = segment.vec.options().bytes_per_vector();
        let mut begin = 0;
        while begin < survivors.len() {
            self.control.check_interrupt();
            let mut end = begin + 1;
            while end < survivors.len()
                && end - begin < 64
                && survivors[end].row == survivors[end - 1].row + 1
            {
                end += 1;
            }
            let bytes = segment
                .vec
                .vector_bytes_for_rows(survivors[begin].row..survivors[end - 1].row + 1)?;
            for (i, survivor) in survivors[begin..end].iter().enumerate() {
                let score = self
                    .prepared
                    .score_doc_bytes(&bytes[i * stride..(i + 1) * stride]);
                if self
                    .heap
                    .threshold
                    .as_ref()
                    .is_some_and(|((t, _), _)| score < *t)
                    || threshold.is_some_and(|t| score < t)
                {
                    continue;
                }
                let address = DocAddress::new(segment.ord, survivor.doc);
                if !self.control.accept(address) {
                    self.stats.pruned_invisible += 1;
                    continue;
                }
                let key = segment.tie.segment_sort_key(survivor.doc, score);
                let key = segment.tie.convert_segment_sort_key(key);
                self.heap.push_unordered((score, key), address);
                self.pending.push_unordered(score, address);
            }
            begin = end;
        }
        Ok(())
    }

    fn probe(
        &mut self,
        segment: &mut SegmentSearch<'_, S::Child>,
        rank: usize,
        probe: &ClusterProbe,
        rows: &[Survivor],
        scratch: &mut Vec<Survivor>,
        threshold: Option<Score>,
    ) -> crate::Result<()> {
        self.work.opens += probe.work.opens;
        self.work.rows += probe.work.rows;
        if probe.skipped {
            self.stats.bounds_skips += 1;
            return Ok(());
        }
        if probe.work.opens == 0 {
            return Ok(());
        }
        self.stats.segment_opens += 1;
        self.stats.vectors_visited += probe.visited;
        self.stats.pruned_filter += probe.filtered;
        self.stats.pruned_dead += probe.dead;
        self.stats.pruned_seen += probe.duplicates;
        self.stats.candidates_scored += probe.work.rows as usize;
        self.stats.cluster_flags[rank] |= if probe.work.rows == 0 { 1 } else { 3 };
        if let Some(range) = &probe.direct {
            collect_cluster_survivors(&segment.vec, range.clone(), None, None, None, scratch);
            self.score(segment, scratch, threshold)?;
        } else {
            self.score(segment, &rows[probe.survivors.clone()], threshold)?;
        }
        Ok(())
    }

    fn synchronize(&mut self, segments: usize) -> Option<Score> {
        let batch = std::mem::replace(
            &mut self.pending,
            TopNComputer::new_with_comparator(self.limit, NaturalComparator),
        );
        let candidates: Vec<_> = batch
            .into_vec()
            .into_iter()
            .map(|doc| (doc.sort_key, doc.doc))
            .collect();
        self.control.publish(&candidates);
        let local = (self.limit <= 1024)
            .then(|| self.heap.kth_best().map(|(score, _)| score))
            .flatten();
        self.control.synchronize(segments, local)
    }
}

fn open_segment<'a, S: SortKeyComputer>(
    searcher: &'a Searcher,
    field: Field,
    ord: SegmentOrdinal,
    tie_break: &S,
) -> crate::Result<SegmentSearch<'a, S::Child>> {
    let reader = searcher.segment_reader(ord);
    let vec = reader.vector_index(field)?;
    let needs_dedup = vec
        .clusters()
        .is_some_and(|ivf| ivf.num_rows() > ivf.num_docs());
    Ok(SegmentSearch {
        ord,
        reader,
        tie: tie_break.segment_sort_key_computer(reader)?,
        alive: reader.alive_bitset(),
        filter: FilterState::All,
        seen: None,
        needs_dedup,
        dead: false,
        vec,
    })
}

pub(super) fn search<T, S>(
    searcher: &Searcher,
    weight: &dyn Weight,
    field: Field,
    query: &Arc<Vec<T>>,
    top_n: usize,
    tie_break: &S,
    plan: &PreparedVectorSearch,
    segment_ordinals: impl Iterator<Item = SegmentOrdinal>,
    control: &mut dyn VectorSearchControl,
) -> crate::Result<(Vec<((Score, S::SortKey), DocAddress)>, ProbeStats)>
where
    T: VectorElement,
    S: SortKeyComputer,
{
    let started = Instant::now();
    let mut owned = 0;
    if top_n == 0 {
        control.end();
        return Ok((Vec::new(), ProbeStats::default()));
    }
    let FieldType::Vector(options) = searcher.schema().get_field_entry(field).field_type() else {
        unreachable!("vector schema checked");
    };
    let metric = options.metric();
    let mut collector = ProbeCollector::<T, S> {
        prepared: PreparedQuery::new(metric, Arc::clone(query)),
        heap: TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_break.comparator())),
        pending: TopNComputer::new_with_comparator(top_n, NaturalComparator),
        limit: top_n,
        work: ClusterWork::default(),
        control,
        stats: ProbeStats {
            cluster_flags: vec![0; plan.clusters.len()],
            ..Default::default()
        },
    };
    let all = weight.matches_all_docs();
    let mut segments = Vec::new();
    let mut scratch = Vec::new();
    for ord in segment_ordinals {
        if owned == 0 {
            collector.control.begin();
        }
        owned += 1;
        collector.control.check_interrupt();
        let setup_started = Instant::now();
        let mut segment = open_segment(searcher, field, ord, tie_break)?;
        if segment.vec.num_vectors() == 0 {
            continue;
        }
        collector.stats.segments_searched += 1;
        collector.stats.segment_setup_time_ns += setup_started.elapsed().as_nanos() as u64;
        if !all {
            let start = Instant::now();
            let filter = build_filter_bitset(weight, segment.reader)?;
            collector.stats.filters_built += 1;
            collector.stats.filter_time_ns += start.elapsed().as_nanos() as u64;
            segment.dead = filter.len() == 0;
            segment.filter = FilterState::Built(filter);
        }
        if segment.dead {
            continue;
        }
        if segment.vec.clusters().is_none() {
            for begin in (0..segment.vec.num_vectors()).step_by(64) {
                let filter = match &segment.filter {
                    FilterState::Built(filter) => Some(filter),
                    _ => None,
                };
                collect_cluster_survivors(
                    &segment.vec,
                    begin..(begin + 64).min(segment.vec.num_vectors()),
                    filter,
                    segment.alive,
                    None,
                    &mut scratch,
                );
                collector.stats.exact_rows_read += scratch.len();
                let survivors = scratch.as_slice();
                collector.score(&mut segment, survivors, None)?;
            }
        } else {
            if segment.ivf().num_clusters() != plan.num_centroids {
                return Err(TantivyError::InternalError(
                    "centroid and segment cluster counts differ".into(),
                ));
            }
            if segment.needs_dedup {
                segment.seen = Some(BitSet::with_max_value(segment.reader.max_doc()));
            }
            segments.push(segment);
        }
    }
    if owned == 0 {
        collector.control.end();
        return Ok((Vec::new(), collector.stats));
    }
    let sharing = all && plan.shareable && collector.control.work_sharing();
    let initial_wave = sharing.then_some(plan.initial_wave).flatten();
    let mut threshold = if initial_wave.is_some() {
        None
    } else {
        collector.synchronize(owned)
    };
    let q_norm = norm_squared_wide(collector.prepared.query()).sqrt() as f32;
    let mut start = 0;
    let mut spent = 0.0;
    let assigned = segments.len();
    while start < plan.clusters.len() && spent < plan.budget.limit {
        let end = (start + PROBE_WAVE_SIZE).min(plan.clusters.len());
        let initial_wave = (start == 0).then_some(initial_wave).flatten();
        let mut costs = vec![ClusterWork::default(); end - start];
        let mut prepared_segments = Vec::with_capacity(segments.len());
        for segment in segments[..assigned]
            .iter_mut()
            .filter(|_| initial_wave.is_none())
        {
            collector.control.check_interrupt();
            let mut probes = Vec::with_capacity(end - start);
            let mut rows = Vec::new();
            for (offset, cluster) in plan.clusters[start..end].iter().enumerate() {
                let probe = ClusterProbe::prepare(
                    segment,
                    cluster,
                    threshold,
                    metric,
                    q_norm,
                    &mut scratch,
                    &mut rows,
                );
                costs[offset].opens += probe.work.opens;
                costs[offset].rows += probe.work.rows;
                probes.push(probe);
            }
            prepared_segments.push((probes, rows));
        }
        let wave = if let Some(wave) = initial_wave {
            collector.control.publish_initial_wave(wave);
            wave
        } else {
            collector
                .control
                .select_wave(owned, start, &costs, plan.budget, spent)
        };
        assert!(wave.end >= start && wave.end <= end);
        let mut order: Vec<_> = (start..wave.end).collect();
        let keep = 16usize.saturating_sub(start).min(order.len());
        order[keep..].sort_unstable_by_key(|&rank| plan.clusters[rank].id);
        if sharing {
            let segment_count = searcher.segment_readers().len();
            let assigned_ordinals: Vec<_> = segments[..assigned]
                .iter()
                .map(|segment| segment.ord)
                .collect();
            let first_helper =
                (assigned_ordinals.first().copied().unwrap_or(0) as usize + 1) % segment_count;
            let order_segments = assigned_ordinals
                .into_iter()
                .chain((0..segment_count).map(|i| ((first_helper + i) % segment_count) as u32));
            for ord in order_segments {
                loop {
                    collector.control.check_interrupt();
                    let batch = collector.control.claim_work(ord);
                    if batch >= order.len().div_ceil(PROBE_BATCH_SIZE) {
                        break;
                    }
                    let position = if let Some(position) =
                        segments.iter().position(|segment| segment.ord == ord)
                    {
                        position
                    } else {
                        let setup_started = Instant::now();
                        segments.push(open_segment(searcher, field, ord, tie_break)?);
                        collector.stats.segment_setup_time_ns +=
                            setup_started.elapsed().as_nanos() as u64;
                        segments.len() - 1
                    };
                    let segment = &mut segments[position];
                    let begin = batch * PROBE_BATCH_SIZE;
                    for &rank in &order[begin..(begin + PROBE_BATCH_SIZE).min(order.len())] {
                        let probe = ClusterProbe::prepare(
                            segment,
                            &plan.clusters[rank],
                            threshold,
                            metric,
                            q_norm,
                            &mut scratch,
                            &mut Vec::new(),
                        );
                        collector.probe(segment, rank, &probe, &[], &mut scratch, threshold)?;
                    }
                }
            }
        } else {
            for (segment, (probes, rows)) in segments.iter_mut().zip(&prepared_segments) {
                for &rank in &order {
                    let probe = &probes[rank - start];
                    collector.probe(segment, rank, probe, rows, &mut scratch, threshold)?;
                }
            }
        }
        threshold = collector.synchronize(owned);
        spent = wave.spent;
        start = wave.end;
    }
    collector.stats.work_charged = plan.budget.charge(collector.work) as f32;
    collector.stats.termination = if start < plan.clusters.len() {
        ProbeTermination::Ceiling
    } else {
        ProbeTermination::Exhausted
    };
    collector.stats.postings_row = collector
        .stats
        .cluster_flags
        .iter()
        .filter(|&&f| f & 2 != 0)
        .count();
    collector.stats.postings_skipped = collector
        .stats
        .cluster_flags
        .iter()
        .filter(|&&f| f == 1)
        .count();
    collector.stats.probe_time_ns = started.elapsed().as_nanos() as u64;
    collector.control.finish(&collector.stats);
    collector.control.end();
    let hits = collector
        .heap
        .into_sorted_vec()
        .into_iter()
        .map(|doc| (doc.sort_key, doc.doc))
        .collect();
    Ok((hits, collector.stats))
}

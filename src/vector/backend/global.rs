use super::*;
use crate::collector::SortKeyComputer;
use crate::vector::ivf::RouterIndex;

pub(crate) type GlobalHits<K> = Vec<((Score, K), DocAddress)>;
type GlobalHeap<S> = TopNComputer<
    (Score, <S as SortKeyComputer>::SortKey),
    DocAddress,
    (NaturalComparator, <S as SortKeyComputer>::Comparator),
>;
type ScoreHeap = TopNComputer<Score, DocAddress, NaturalComparator>;

#[derive(Clone)]
struct AdmissionScore {
    lower: LowerEndpoint,
    estimate: Estimate,
}

#[derive(Clone, Debug, Default)]
struct LowerComparator;

impl Comparator<AdmissionScore> for LowerComparator {
    fn compare(&self, left: &AdmissionScore, right: &AdmissionScore) -> std::cmp::Ordering {
        left.lower.0.total_cmp(&right.lower.0)
    }
}

type AdmissionHeap = TopNComputer<AdmissionScore, DocAddress, LowerComparator>;

struct GlobalTop<S: SortKeyComputer> {
    hits: GlobalHeap<S>,
    exact: ScoreHeap,
    lower: AdmissionHeap,
    limit: usize,
}

impl<S: SortKeyComputer> GlobalTop<S> {
    fn new(limit: usize, sort: &S) -> Self {
        Self {
            hits: TopNComputer::new_with_comparator(limit, (NaturalComparator, sort.comparator())),
            exact: ScoreHeap::new_with_comparator(limit, NaturalComparator),
            lower: AdmissionHeap::new_with_comparator(limit, LowerComparator),
            limit,
        }
    }

    fn push_exact(&mut self, sort: &mut S::Child, address: DocAddress, score: Score) {
        self.exact.push_unordered(score, address);
        self.lower.push_unordered(
            AdmissionScore {
                lower: LowerEndpoint(score),
                estimate: Estimate(score),
            },
            address,
        );
        if self
            .hits
            .threshold
            .as_ref()
            .is_some_and(|((threshold, _), _)| score < *threshold)
        {
            return;
        }
        let key = sort.segment_sort_key(address.doc_id, score);
        self.hits
            .push_unordered((score, sort.convert_segment_sort_key(key)), address);
    }

    fn push_quantized(
        &mut self,
        ordinal: SegmentOrdinal,
        candidates: &QuantizedCandidates,
        start: usize,
    ) {
        for i in start..candidates.len() {
            let address = DocAddress::new(ordinal, candidates.rows[i] as DocId);
            let estimate = candidates.estimate(i);
            self.lower.push_unordered(
                AdmissionScore {
                    lower: estimate.lower(candidates.sigmas[i], QUANTIZED_BOUNDARY_KAPPA),
                    estimate,
                },
                address,
            );
        }
    }

    fn threshold(&mut self) -> Option<Threshold> {
        self.lower.kth_best().map(|score| Threshold(score.lower))
    }

    // APS uses the lowest point estimate among the lower-endpoint top-k.
    fn estimate_kth(&mut self) -> Option<Score> {
        self.lower.kth_best()?;
        self.lower
            .clone()
            .into_vec()
            .iter()
            .map(|hit| hit.sort_key.estimate.0)
            .min_by(f32::total_cmp)
    }

    fn rebuild<T: VectorElement>(&mut self, segments: &[SegmentScan<'_, T, S>]) {
        self.lower = AdmissionHeap::new_with_comparator(self.limit, LowerComparator);
        for hit in self.exact.clone().into_vec() {
            self.lower.push_unordered(
                AdmissionScore {
                    lower: LowerEndpoint(hit.sort_key),
                    estimate: Estimate(hit.sort_key),
                },
                hit.doc,
            );
        }
        for segment in segments {
            if let Some(scorer) = &segment.quantized {
                self.push_quantized(segment.backend.segment_ord, &scorer.scan.candidates, 0);
            }
        }
    }
}

struct SegmentScan<'a, T: VectorElement, S: SortKeyComputer> {
    backend: &'a VectorBackend<T>,
    reader: &'a SegmentReader,
    gate: Option<RowGate<'a>>,
    sort: S::Child,
    quantized: Option<QuantizedScorer>,
    stats: ProbeStats,
    docs: Vec<DocId>,
    offsets: Vec<usize>,
    rows: Vec<usize>,
    ranges: Vec<Range<usize>>,
    blocks: Vec<(usize, usize)>,
}

impl<T: VectorElement, S: SortKeyComputer> SegmentScan<'_, T, S> {
    fn with_io<R>(
        &mut self,
        action: impl FnOnce(&mut Self) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let before = crate::vector::storage_io::snapshot();
        let fallbacks = cascade::sign_word_fallback_counts();
        let result = action(self);
        let after = crate::vector::storage_io::snapshot();
        let after_fallbacks = cascade::sign_word_fallback_counts();
        for (layer, stats) in self.stats.layers.0.iter_mut().enumerate() {
            let delta = after[layer].since(before[layer]);
            stats.io.reads += delta.reads;
            stats.io.bytes_read += delta.bytes_read;
            stats.io.storage_blocks += delta.storage_blocks;
            stats.sign_word_fallbacks += after_fallbacks[layer] - fallbacks[layer];
        }
        let delta = after[3].since(before[3]);
        self.stats.rerank_io.reads += delta.reads;
        self.stats.rerank_io.bytes_read += delta.bytes_read;
        self.stats.rerank_io.storage_blocks += delta.storage_blocks;
        result
    }

    fn probe(
        &mut self,
        candidate: Candidate,
        bound: QueryBound,
        q_norm: f32,
        weight: &dyn Weight,
        top: &mut GlobalTop<S>,
    ) -> crate::Result<usize> {
        let reader = &self.backend.reader;
        let index = reader.index().expect("IVF segment");
        let metric = reader.options().metric();
        let cluster = candidate.node as usize;
        let rows = index.cluster_range(cluster);
        if rows.is_empty() {
            self.stats.postings_skipped += 1;
            self.stats.clusters_skipped_empty += 1;
            return Ok(0);
        }
        let verdict = bounds_verdict(bound, || {
            let QueryBound::Armed { t } = bound else {
                return f32::INFINITY;
            };
            let r = index.bounds().ball_r(cluster);
            match metric {
                Metric::L2 | Metric::Cosine => {
                    margin_ball_ball(t, r, to_bound_space(metric, candidate.sim.score()))
                }
                Metric::Dot => margin_ball_halfspace(candidate.sim.score(), q_norm, r, t),
            }
        });
        if verdict == Verdict::Skip {
            self.stats.bounds_skips += 1;
            return Ok(0);
        }
        self.backend
            .prepare_row_gate(weight, self.reader, &mut self.gate, &mut self.stats)?;
        if matches!(self.gate, Some(RowGate::Empty)) {
            return Ok(0);
        }
        if self.quantized.is_none() && self.backend.quantized_query.is_some() {
            let start = Instant::now();
            self.quantized = Some(QuantizedScorer::new(self.reader.max_doc(), 0));
            self.stats.start_layer(0);
            self.stats.scan_init_ns += start.elapsed().as_nanos() as u64;
        }
        let start = Instant::now();
        let quantized = self.quantized.is_some();
        let _stage = enter_vector_stage(if quantized {
            Stage::LayerScan(0)
        } else {
            Stage::ExactScan
        });
        let count = self.with_io(|segment| segment.probe_rows(candidate, rows, top))?;
        let elapsed = start.elapsed().as_nanos() as u64;
        if quantized {
            self.stats.record_layer_scan(0, count, elapsed);
        } else {
            *self.stats.exact_scan_ns.get_or_insert(0) += elapsed;
        }
        Ok(count)
    }

    fn probe_rows(
        &mut self,
        candidate: Candidate,
        rows: Range<usize>,
        top: &mut GlobalTop<S>,
    ) -> crate::Result<usize> {
        let reader = &self.backend.reader;
        let cluster = candidate.node as usize;
        let gate = self.gate.as_ref().expect("filter prepared before scanning");
        let open = matches!(gate, RowGate::Open);
        if !open {
            reader.cache_doc_ids()?;
            reader.read_doc_ids(cluster, &mut self.docs)?;
        }
        let (selection, visited, pruned_filter, pruned_dead) =
            select_cluster_rows(&self.docs, rows.clone(), gate, &mut self.offsets);
        self.stats.vectors_visited += visited;
        self.stats.pruned_filter += pruned_filter;
        self.stats.pruned_dead += pruned_dead;
        let count = selection.len(&rows);
        if count == 0 {
            self.stats.postings_skipped += 1;
            self.stats.clusters_skipped_empty += 1;
            return Ok(0);
        }
        self.stats.postings_row += 1;
        self.stats.candidates_scored += count;
        self.stats.eligible_charged += count;
        if let Some(scorer) = &mut self.quantized {
            let first = scorer.scan.candidates.len();
            scorer.score_cluster(
                reader,
                self.backend.quantized_query.as_ref().unwrap(),
                cluster,
                candidate.sim,
                rows,
                &selection,
                (!open).then_some(self.docs.as_slice()),
            )?;
            top.push_quantized(self.backend.segment_ord, &scorer.scan.candidates, first);
            self.stats.layer0_eligible += count;
        } else {
            if open {
                let bytes = reader.read_cluster_rows(cluster)?;
                let mut resolved = false;
                for (offset, bytes) in bytes
                    .chunks_exact(reader.options().bytes_per_vector())
                    .enumerate()
                {
                    #[cfg(test)]
                    self.stats
                        .quantized_trace
                        .scored_rows
                        .push(rows.start + offset);
                    let score = self.backend.query.score_doc_bytes(bytes);
                    if top
                        .hits
                        .threshold
                        .as_ref()
                        .is_some_and(|((threshold, _), _)| score < *threshold)
                    {
                        continue;
                    }
                    if !resolved {
                        reader.read_doc_ids(cluster, &mut self.docs)?;
                        resolved = true;
                    }
                    top.push_exact(
                        &mut self.sort,
                        DocAddress::new(self.backend.segment_ord, self.docs[offset]),
                        score,
                    );
                }
            } else {
                self.rows.clear();
                match selection {
                    Selection::All => self.rows.extend(rows.clone()),
                    Selection::Rows(offsets) => self
                        .rows
                        .extend(offsets.iter().map(|offset| rows.start + offset)),
                    Selection::None => unreachable!("nonempty selection"),
                }
                let batch = reader.read_vector_rows_planned(
                    &self.rows,
                    &mut self.ranges,
                    &mut self.blocks,
                )?;
                for (row, bytes) in batch.iter() {
                    #[cfg(test)]
                    self.stats.quantized_trace.scored_rows.push(row);
                    top.push_exact(
                        &mut self.sort,
                        DocAddress::new(self.backend.segment_ord, self.docs[row - rows.start]),
                        self.backend.query.score_doc_bytes(bytes),
                    );
                }
            }
        }
        Ok(count)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search<T: VectorElement, S: SortKeyComputer>(
    router: &RouterIndex,
    query: &[f32],
    backends: &[VectorBackend<T>],
    readers: &[SegmentReader],
    weight: &dyn Weight,
    adaptive: &AdaptiveProbeParams,
    top_n: usize,
    sort: &S,
) -> crate::Result<(GlobalHits<S::SortKey>, Vec<ProbeStats>)> {
    if top_n == 0 {
        return Ok((
            Vec::new(),
            readers.iter().map(|_| ProbeStats::default()).collect(),
        ));
    }
    let mut top = GlobalTop::new(top_n, sort);
    let mut segments = Vec::with_capacity(backends.len());
    for (backend, reader) in backends.iter().zip(readers) {
        let start = Instant::now();
        let filter = backend.filter.as_ref().map(|(filter, _)| filter);
        let mut stats = ProbeStats {
            scan_init_ns: backend.scan_init_ns,
            query_prep_ns: backend.query_prep_ns,
            non_vector_search_ns: backend.filter.as_ref().map_or(0, |(_, elapsed)| *elapsed),
            ..Default::default()
        };
        let mut child = sort.segment_sort_key_computer(reader)?;
        if let Some(index) = backend.reader.index() {
            stats.segment_rows = Some(index.num_rows());
            stats.segment_clusters = Some(index.num_clusters());
            stats.scan_init_ns += start.elapsed().as_nanos() as u64;
        } else {
            let (hits, flat_stats) =
                backend.top_n_by(weight, reader, top_n, &mut child, sort.comparator())?;
            stats = flat_stats;
            for ((score, key), address) in hits {
                top.exact.push_unordered(score, address);
                top.lower.push_unordered(
                    AdmissionScore {
                        lower: LowerEndpoint(score),
                        estimate: Estimate(score),
                    },
                    address,
                );
                top.hits
                    .push_unordered((score, child.convert_segment_sort_key(key)), address);
            }
        }
        segments.push(SegmentScan {
            backend,
            reader,
            gate: filter.map(|filter| RowGate::new(Cow::Borrowed(filter), reader.alive_bitset())),
            sort: child,
            quantized: None,
            stats,
            docs: Vec::new(),
            offsets: Vec::new(),
            rows: Vec::new(),
            ranges: Vec::new(),
            blocks: Vec::new(),
        });
    }
    let active = |segment: &SegmentScan<'_, T, S>| {
        segment
            .backend
            .reader
            .index()
            .is_some_and(|index| index.num_docs() > 0)
            && !matches!(segment.gate, Some(RowGate::Empty))
    };
    if segments.iter().any(active) {
        let num_docs: usize = backends
            .iter()
            .filter_map(|backend| backend.reader.index())
            .map(IvfIndex::num_docs)
            .sum();
        let clusters = router.num_clusters();
        let (budget, n_avg, open) = adaptive.resolved_work_budget(clusters, num_docs)?;
        let pricing = UnitPricing {
            budget: WorkUnits::new(budget),
            open: WorkUnits::new(open),
            row: WorkUnits::new((1.0 - open) / n_avg),
        };
        let (matched, docs) = segments
            .iter()
            .filter(|segment| segment.backend.reader.index().is_some())
            .fold((0.0, 0u64), |(matched, docs), segment| {
                let max_doc = segment.reader.max_doc();
                let fraction = segment
                    .gate
                    .as_ref()
                    .map_or(1.0, |gate| gate.match_fraction(max_doc));
                (
                    matched + fraction * f64::from(max_doc),
                    docs + u64::from(max_doc),
                )
            });
        let params = RoutingParams {
            k: adaptive.router_k(budget, open, matched / docs.max(1) as f64, clusters),
            recall: adaptive.router_recall_target,
        };
        let mut workspace = RouterWorkspace::default();
        let start = Instant::now();
        let mut ranked = {
            let _stage = enter_vector_stage(Stage::Routing);
            router.rank_clusters(&mut workspace, query, params)
        };
        let estimator = router.recall_estimator(&ranked, query, adaptive.recall_target);
        let mut controller =
            ProbeController::new(pricing, clusters, estimator, adaptive.recall_target);
        let mut routing_ns = start.elapsed().as_nanos() as u64;
        let metric = backends[0].query.metric();
        let q_norm = norm_squared_wide(query).sqrt() as f32;
        let mut armed_at = None;
        let mut probe = 0;
        loop {
            let start = Instant::now();
            let next = {
                let _stage = enter_vector_stage(Stage::Routing);
                ranked.next()
            };
            routing_ns += start.elapsed().as_nanos() as u64;
            let Some(candidate) = next else { break };
            if !controller.admit() {
                break;
            }
            let bound =
                top.threshold()
                    .map_or(QueryBound::Filling, |threshold| QueryBound::Armed {
                        t: to_bound_space(metric, threshold.0 .0),
                    });
            controller.charge_open();
            let mut scored = 0;
            for segment in &mut segments {
                if active(segment) {
                    let count = segment.probe(candidate, bound, q_norm, weight, &mut top)?;
                    scored += count;
                }
            }
            controller.charge_rows(scored);
            if armed_at.is_none() && top.threshold().is_some() {
                armed_at = Some(probe)
            }
            if controller.estimator.is_some() {
                controller.cover(top.estimate_kth())?;
            }
            probe += 1;
            if !segments.iter().any(active) {
                break;
            }
        }
        let stats = &mut segments[0].stats;
        controller.finish(stats);
        stats.record_routing(ranked.metrics());
        stats.record_bound_armed(armed_at);
        stats.routing_ns += routing_ns;
    }
    #[cfg(test)]
    for segment in &mut segments {
        if let Some(scorer) = &segment.quantized {
            segment.stats.quantized_trace.scored_rows = scorer.scan.candidates.rows.clone();
            segment
                .stats
                .quantized_trace
                .estimate_rows
                .push(scorer.scan.candidates.estimate_trace());
        }
    }
    let levels = backends
        .iter()
        .filter_map(|backend| backend.quantized_query.as_ref())
        .map(|query| query.active_layers())
        .max()
        .unwrap_or(0);
    for layer in 0..levels {
        top.rebuild(&segments);
        let threshold = top.threshold();
        for segment in &mut segments {
            let Some(scorer) = &mut segment.quantized else {
                continue;
            };
            let start = Instant::now();
            {
                let _stage = enter_vector_stage(Stage::Boundary(layer as u8));
                scorer
                    .scan
                    .band_at_threshold(threshold, QUANTIZED_BOUNDARY_KAPPA);
            }
            #[cfg(test)]
            segment
                .stats
                .quantized_trace
                .boundary_rows
                .push(scorer.scan.candidates.rows.clone());
            segment.stats.record_boundary(
                layer,
                scorer.scan.candidates.len(),
                start.elapsed().as_nanos() as u64,
            );
        }
        for segment in &mut segments {
            if segment.quantized.is_none() {
                continue;
            }
            segment.with_io(|segment| {
                let query = segment.backend.quantized_query.as_ref().unwrap();
                if layer + 1 == query.active_layers() {
                    let mut scorer = segment.quantized.take().unwrap();
                    scorer.rerank(
                        &segment.backend.reader,
                        &segment.backend.query,
                        &mut segment.stats,
                        |doc, score| {
                            top.push_exact(
                                &mut segment.sort,
                                DocAddress::new(segment.backend.segment_ord, doc),
                                score,
                            )
                        },
                    )?;
                } else {
                    segment.quantized.as_mut().unwrap().refine(
                        &segment.backend.reader,
                        query,
                        layer + 1,
                        &mut segment.stats,
                    )?;
                }
                Ok(())
            })?;
        }
    }
    #[cfg(test)]
    for segment in &mut segments {
        let mapping = crate::vector::storage_io::test_support::with_unarmed_log(|| {
            let mut mapping = Vec::new();
            if let Some(index) = segment.backend.reader.index() {
                for cluster in 0..index.num_clusters() {
                    segment
                        .backend
                        .reader
                        .read_doc_ids(cluster, &mut segment.docs)?;
                    mapping.extend_from_slice(&segment.docs);
                }
            }
            Ok::<_, TantivyError>(mapping)
        })?;
        segment.stats.quantized_trace.translate(&mapping);
    }
    let start = Instant::now();
    let _stage = enter_vector_stage(Stage::ResultAssembly);
    let hits = top
        .hits
        .into_sorted_vec()
        .into_iter()
        .map(|hit| (hit.sort_key, hit.doc))
        .collect();
    if let Some(first) = segments.first_mut() {
        *first.stats.result_assembly_ns.get_or_insert(0) += start.elapsed().as_nanos() as u64;
    }
    Ok((
        hits,
        segments.into_iter().map(|segment| segment.stats).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_boundary_uses_lower_endpoints_and_keeps_ties() {
        let mut top = GlobalTop::new(2, &NoTieBreak);
        top.push_exact(&mut NoTieBreak, DocAddress::new(0, 0), 8.0);
        let mut scan = QuantizedScanCtx::new(4, 4);
        for (row, (estimate, sigma)) in [(100.0, 100.0), (7.5, 0.0), (7.5, 0.0), (6.0, 0.0)]
            .into_iter()
            .enumerate()
        {
            scan.push(row, row as u32, 0.0, 0.0, estimate, sigma, 0.0, 1.0, 0.0);
        }
        top.push_quantized(1, &scan.candidates, 0);
        assert_eq!(top.estimate_kth(), Some(7.5));
        let threshold = top.threshold();
        assert_eq!(threshold, Some(Threshold(LowerEndpoint(7.5))));
        scan.band_at_threshold(threshold, QUANTIZED_BOUNDARY_KAPPA);
        assert_eq!(scan.candidates.rows, [0, 1, 2]);
    }
}

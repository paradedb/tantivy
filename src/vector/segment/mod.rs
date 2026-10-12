//! The IVF segment search: routing, admission and the cascade.

pub(in crate::vector) mod plan;

use std::time::Instant;

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::error::DataCorruption;
use crate::query::Weight;
use crate::schema::Metric;
use crate::vector::backend::VectorBackend;
use crate::vector::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, HeapPeek, QueryBound,
    QueryBoundTracker, Verdict,
};
use crate::vector::distance::norm_squared_wide;
use crate::vector::ivf::{Candidate, IvfIndex};
use crate::vector::prepared::{ArithmeticError, QuantizedQueryCtx};
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::vector::router::{RouterWorkspace, RoutingParams};
use crate::vector::scan::estimate::{
    combine_initial_decoded, combine_refinement_decoded, cosine_refinement_batches,
};
use crate::vector::scan::heap::{tie_break_key, TieBreakHeap, TieBreakHits};
use crate::vector::scan::layer_read::{candidate_selection, score_layer, Selection};
use crate::vector::scan::QuantizedScanCtx;
use crate::vector::segment::plan::build_segment_filter;
use crate::vector::source::controller::{ProbeController, UnitPricing, WorkUnits};
use crate::vector::source::routed::{select_cluster_rows, RowGate};
use crate::vector::stats::ProbeStats;
use crate::vector::{enter_vector_stage, Stage, VectorElement};
use crate::{DocAddress, DocId, Score, SegmentReader, TantivyError};

impl<T: VectorElement> VectorBackend<T> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quantized_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        query: &QuantizedQueryCtx,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 || segment_reader.max_doc() == 0 || index.num_clusters() == 0 {
            return Ok(Vec::new());
        }
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        let max_doc = segment_reader.max_doc();
        let non_vector_start = Instant::now();
        let non_vector_stage = enter_vector_stage(Stage::NonVectorSearch);
        let filter = build_segment_filter(weight, segment_reader, max_doc)?;
        let alive = segment_reader.alive_bitset();
        drop(non_vector_stage);
        let non_vector_search_ns = non_vector_start.elapsed().as_nanos() as u64;
        stats.non_vector_search_ns = stats
            .non_vector_search_ns
            .saturating_add(non_vector_search_ns);
        if filter.is_empty() {
            drop(init_stage);
            stats.scan_init_ns = stats.scan_init_ns.saturating_add(
                (init_start.elapsed().as_nanos() as u64).saturating_sub(non_vector_search_ns),
            );
            return Ok(Vec::new());
        }
        let row_gate = RowGate::new(&filter, alive);
        let mut cluster_docs = Vec::new();
        let scan_levels = query.active_layers();
        let quantized = self
            .reader
            .quantization()
            .expect("quantized query requires quantized slots");
        stats.segment_rows = Some(index.num_rows());
        stats.segment_clusters = Some(index.num_clusters());
        stats.start_layer(0);
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(index.num_clusters(), index.num_docs())?;
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };
        let mut routing_ws = RouterWorkspace::default();
        let candidate_capacity =
            ((pricing.budget.get() / pricing.row.get()).ceil() as usize).min(index.num_rows());
        let mut scan = QuantizedScanCtx::new(max_doc, candidate_capacity);
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let bounds = index.bounds();
        let metric = query.index.meta.field().metric;
        let q_norm = norm_squared_wide(query.query()).sqrt() as f32;
        let mut bounds_skips = 0u32;
        let mut armed_probe = None;
        let mut selection_offsets = Vec::new();
        let mut kernel_scores = Vec::new();
        let mut decoded_scales = Vec::new();
        let mut decoded_gammas = Vec::new();
        let mut decoded_error_ratios = Vec::new();
        let mut decoded_constants = Vec::new();
        let mut decoded_residual_norms = Vec::new();
        let mut base_scores = Vec::new();
        let mut estimate_scores = Vec::new();
        let mut sigma_scores = Vec::new();
        let mut residual_norm_squared_scores = Vec::new();
        let mut sign_query_error_terms = Vec::new();
        let mut arithmetic_variances = Vec::new();
        let mut selected_rows = Vec::new();
        let mut indexed_row_offsets = Vec::new();
        let mut survivor_read_ranges = Vec::new();
        let mut survivor_block_scratch = Vec::new();
        drop(init_stage);
        stats.scan_init_ns = stats.scan_init_ns.saturating_add(
            (init_start.elapsed().as_nanos() as u64).saturating_sub(non_vector_search_ns),
        );

        let routing = RoutingParams {
            k: self.adaptive.router_k(
                work_budget,
                x,
                filter.match_fraction(max_doc),
                index.num_clusters(),
            ),
            recall: self.adaptive.router_recall_target,
        };
        let routing_start = Instant::now();
        let (mut ranked, mut controller) = {
            let _routing_stage = enter_vector_stage(Stage::Routing);
            let ranked = index.rank_clusters(&mut routing_ws, query.query(), routing);
            let estimator =
                index.recall_estimator(&ranked, query.query(), self.adaptive.recall_target);
            let controller = ProbeController::new(
                pricing,
                index.num_clusters(),
                estimator,
                self.adaptive.recall_target,
            );
            (ranked, controller)
        };
        let mut routing_ns = routing_start.elapsed().as_nanos() as u64;
        let routing_before_scan = routing_ns;
        let scan_start = Instant::now();
        let layer0_stage = enter_vector_stage(Stage::LayerScan(0));

        loop {
            let routing_start = Instant::now();
            let next = {
                let _routing_stage = enter_vector_stage(Stage::Routing);
                ranked.next()
            };
            routing_ns += routing_start.elapsed().as_nanos() as u64;
            let Some(Candidate { sim, node }) = next else {
                break;
            };
            if !controller.admit() {
                break;
            }
            let cluster = node as usize;
            // The bounds gate needs the pessimistic kth so a skip never
            // drops a true top-k row; APS takes the point estimate, since
            // the pessimistic radius inflates the query ball and
            // underestimates recall.
            let kth = scan
                .running_pessimistic_kth(top_n, QUANTIZED_BOUNDARY_KAPPA)
                .map(|score| score.0 .0);
            let aps_kth = scan.running_estimate_kth(top_n);
            let query_bound = kth.map_or(QueryBound::Filling, |score| QueryBound::Armed {
                t: to_bound_space(metric, score),
            });
            let verdict = bounds_verdict(query_bound, || {
                let QueryBound::Armed { t } = query_bound else {
                    return f32::INFINITY;
                };
                #[cfg(debug_assertions)]
                {
                    let stride = self.reader.options().bytes_per_vector();
                    let centroid_bytes = index.centroid_bytes().expect("readable centroid rows");
                    let exact = metric.similarity_bytes::<f32>(
                        query.query(),
                        &centroid_bytes[cluster * stride..(cluster + 1) * stride],
                    );
                    debug_assert_eq!(
                        sim, exact,
                        "routing stream key must be the exact centroid similarity; the quantized \
                         L2 base and query-residual norm are derived from it"
                    );
                }
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if verdict == Verdict::Skip {
                controller.charge_open();
                controller.cover(aps_kth)?;
                bounds_skips += 1;
                continue;
            }
            controller.charge_open();
            let rows = index.cluster_range(cluster);
            // Empty clusters have no band to read and use the probe's empty-cluster counter.
            if rows.is_empty() {
                postings_skipped += 1;
                stats.clusters_skipped_empty += 1;
                controller.cover(aps_kth)?;
                continue;
            }
            let layer = &quantized.layers()[0];
            if !matches!(row_gate, RowGate::Open) {
                self.reader.read_doc_ids(cluster, &mut cluster_docs)?;
            }
            let selection_start = Instant::now();
            let (selection, visited, pruned_filter, pruned_dead) = {
                let _routing_stage = enter_vector_stage(Stage::Routing);
                select_cluster_rows(
                    &cluster_docs,
                    rows.clone(),
                    &row_gate,
                    &mut selection_offsets,
                )
            };
            routing_ns += selection_start.elapsed().as_nanos() as u64;
            stats.vectors_visited += visited;
            stats.pruned_filter += pruned_filter;
            stats.pruned_dead += pruned_dead;
            let selected_count = selection.len(&rows);
            if selected_count == 0 {
                postings_skipped += 1;
                stats.clusters_skipped_empty += 1;
                controller.cover(aps_kth)?;
                continue;
            }

            let batch = layer.read_batch_in_block(cluster, rows.clone())?;
            let score_query_norm = query.score_query_norm(sim.score());
            score_layer(
                query,
                0,
                layer,
                Some((&batch, &mut decoded_residual_norms)),
                Some(cluster),
                rows.clone(),
                &selection,
                &mut kernel_scores,
                &mut decoded_scales,
                &mut decoded_gammas,
                &mut decoded_error_ratios,
                &mut decoded_constants,
                &mut survivor_read_ranges,
                &mut selected_rows,
                &mut indexed_row_offsets,
            )?;
            base_scores.resize(selected_count, 0.0);
            estimate_scores.resize(selected_count, 0.0);
            sigma_scores.resize(selected_count, 0.0);
            residual_norm_squared_scores.resize(selected_count, 0.0);
            sign_query_error_terms.resize(selected_count, 0.0);
            arithmetic_variances.resize(selected_count, ArithmeticError::default());
            let cluster_score = sim.score();
            combine_initial_decoded(
                metric,
                query.index.meta.field().dim as usize,
                &mut kernel_scores,
                &mut base_scores,
                &mut estimate_scores,
                &mut sigma_scores,
                &mut residual_norm_squared_scores,
                &mut sign_query_error_terms,
                &mut arithmetic_variances,
                &decoded_scales,
                &decoded_gammas,
                &decoded_error_ratios,
                &decoded_constants,
                &decoded_residual_norms,
                cluster_score,
                score_query_norm * score_query_norm,
                query.query_error_squared(0) as f32,
            );

            scan.set_cluster_query_norm(cluster, score_query_norm);
            scan.begin_cluster(top_n);
            scan.candidates.append_selected(
                rows.clone(),
                &selection,
                (!matches!(row_gate, RowGate::Open)).then_some(cluster_docs.as_slice()),
                &base_scores[..selected_count],
                &kernel_scores[..selected_count],
                &estimate_scores[..selected_count],
                &sigma_scores[..selected_count],
                &residual_norm_squared_scores[..selected_count],
                &decoded_gammas[..selected_count],
                &sign_query_error_terms[..selected_count],
                &arithmetic_variances[..selected_count],
            );
            scan.finish_cluster_bound();
            controller.charge_rows(selected_count);
            stats.layer0_eligible += selected_count;
            stats.eligible_charged += selected_count;
            postings_row += 1;
            let kth = scan
                .running_pessimistic_kth(top_n, QUANTIZED_BOUNDARY_KAPPA)
                .map(|score| score.0 .0);
            if armed_probe.is_none() && kth.is_some() {
                armed_probe = Some((postings_row + postings_skipped - 1) as u32);
            }
            controller.cover(scan.running_estimate_kth(top_n))?;
        }
        stats.record_routing(ranked.metrics());
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += scan.candidates.len();
        let layer0_scored = scan.candidates.len();
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(armed_probe);
        controller.finish(stats);
        drop(layer0_stage);
        let scan_ns = scan_start.elapsed().as_nanos() as u64;
        stats.routing_ns += routing_ns;
        stats.record_layer_scan(
            0,
            layer0_scored,
            scan_ns.saturating_sub(routing_ns.saturating_sub(routing_before_scan)),
        );

        #[cfg(test)]
        {
            stats.quantized_trace.scored_rows = scan.candidates.rows.clone();
            stats
                .quantized_trace
                .estimate_rows
                .push(scan.candidates.estimate_trace());
        }

        let boundary_start = Instant::now();
        let boundary_stage = enter_vector_stage(Stage::Boundary(0));
        scan.band(top_n, QUANTIZED_BOUNDARY_KAPPA);
        #[cfg(test)]
        stats
            .quantized_trace
            .boundary_rows
            .push(scan.candidates.rows.clone());
        drop(boundary_stage);
        stats.record_boundary(
            0,
            scan.candidates.len(),
            boundary_start.elapsed().as_nanos() as u64,
        );
        for layer_idx in 1..scan_levels {
            stats.start_layer(layer_idx);
            let layer_start = Instant::now();
            let layer_stage = enter_vector_stage(Stage::LayerScan(layer_idx as u8));
            let layer = &quantized.layers()[layer_idx];
            let layer_scored = scan.candidates.len();
            if metric == Metric::Cosine {
                let query_norm = query.score_query_norm(0.0);
                for candidate_range in cosine_refinement_batches(scan.candidates.len()) {
                    let candidate_start = candidate_range.start;
                    let candidate_end = candidate_range.end;
                    let first_row = scan.candidates.rows[candidate_start];
                    let last_row = scan.candidates.rows[candidate_end - 1];
                    let available_rows = first_row..last_row + 1;
                    let selection = candidate_selection(
                        &scan.candidates.rows[candidate_range.clone()],
                        &available_rows,
                        &mut selection_offsets,
                    );
                    let rows = score_layer(
                        query,
                        layer_idx,
                        layer,
                        None,
                        None,
                        available_rows,
                        &selection,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_gammas,
                        &mut decoded_error_ratios,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut selected_rows,
                        &mut indexed_row_offsets,
                    )?;
                    let decoded_constants = if metric == Metric::L2 {
                        &decoded_constants[..rows]
                    } else {
                        &[]
                    };
                    let sign_query_error_squared =
                        if matches!(query.index.specs[layer_idx].kind, cascade::LayerKind::Sign) {
                            query.query_error_squared(layer_idx) as f32
                        } else {
                            0.0
                        };
                    combine_refinement_decoded(
                        metric,
                        query.index.meta.field().dim as usize,
                        &mut scan.candidates,
                        candidate_range,
                        &kernel_scores[..rows],
                        &decoded_scales[..rows],
                        &decoded_gammas[..rows],
                        &decoded_error_ratios[..rows],
                        decoded_constants,
                        query_norm * query_norm,
                        sign_query_error_squared,
                    );
                }
            } else {
                let mut candidate_start = 0;
                let mut cluster = 0;
                while candidate_start < scan.candidates.len() {
                    let first_row = scan.candidates.rows[candidate_start];
                    while cluster < index.num_clusters()
                        && index.cluster_range(cluster).end <= first_row
                    {
                        cluster += 1;
                    }
                    if cluster == index.num_clusters() {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} is outside IVF cluster ranges"
                            ),
                        )));
                    }
                    let cluster_rows = index.cluster_range(cluster);
                    if first_row < cluster_rows.start {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} precedes cluster {cluster} \
                                 range {cluster_rows:?}"
                            ),
                        )));
                    }
                    let mut candidate_end = candidate_start + 1;
                    while candidate_end < scan.candidates.len()
                        && scan.candidates.rows[candidate_end] < cluster_rows.end
                    {
                        candidate_end += 1;
                    }
                    let query_norm = scan.cluster_query_norm(cluster);
                    let candidate_range = candidate_start..candidate_end;
                    let selection = candidate_selection(
                        &scan.candidates.rows[candidate_range.clone()],
                        &cluster_rows,
                        &mut selection_offsets,
                    );
                    let rows = score_layer(
                        query,
                        layer_idx,
                        layer,
                        None,
                        Some(cluster),
                        cluster_rows,
                        &selection,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_gammas,
                        &mut decoded_error_ratios,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut selected_rows,
                        &mut indexed_row_offsets,
                    )?;
                    let decoded_constants = if metric == Metric::L2 {
                        &decoded_constants[..rows]
                    } else {
                        &[]
                    };
                    let sign_query_error_squared =
                        if matches!(query.index.specs[layer_idx].kind, cascade::LayerKind::Sign) {
                            query.query_error_squared(layer_idx) as f32
                        } else {
                            0.0
                        };
                    combine_refinement_decoded(
                        metric,
                        query.index.meta.field().dim as usize,
                        &mut scan.candidates,
                        candidate_range,
                        &kernel_scores[..rows],
                        &decoded_scales[..rows],
                        &decoded_gammas[..rows],
                        &decoded_error_ratios[..rows],
                        decoded_constants,
                        query_norm * query_norm,
                        sign_query_error_squared,
                    );
                    candidate_start = candidate_end;
                }
            }
            drop(layer_stage);
            stats.record_layer_scan(
                layer_idx,
                layer_scored,
                layer_start.elapsed().as_nanos() as u64,
            );
            #[cfg(test)]
            stats
                .quantized_trace
                .estimate_rows
                .push(scan.candidates.estimate_trace());
            let boundary_start = Instant::now();
            let boundary_stage = enter_vector_stage(Stage::Boundary(layer_idx as u8));
            scan.band(top_n, QUANTIZED_BOUNDARY_KAPPA);
            #[cfg(test)]
            stats
                .quantized_trace
                .boundary_rows
                .push(scan.candidates.rows.clone());
            drop(boundary_stage);
            stats.record_boundary(
                layer_idx,
                scan.candidates.len(),
                boundary_start.elapsed().as_nanos() as u64,
            );
        }

        let rerank_fetch_start = Instant::now();
        let rerank_fetch_stage = enter_vector_stage(Stage::RerankFetch);
        debug_assert!(
            scan.candidates
                .rows
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "boundary survivors are row-sorted"
        );
        let mut first = 0;
        let mut cluster = 0;
        while first < scan.candidates.len() {
            while index.cluster_range(cluster).end <= scan.candidates.rows[first] {
                cluster += 1;
            }
            let cluster_rows = index.cluster_range(cluster);
            let end = first
                + scan.candidates.rows[first..].partition_point(|&row| row < cluster_rows.end);
            if scan.candidates.docs[first..end].contains(&DocId::MAX) {
                self.reader.read_doc_ids(cluster, &mut cluster_docs)?;
                for candidate in first..end {
                    if scan.candidates.docs[candidate] == DocId::MAX {
                        scan.candidates.docs[candidate] =
                            cluster_docs[scan.candidates.rows[candidate] - cluster_rows.start];
                    }
                }
            }
            first = end;
        }
        #[cfg(test)]
        {
            stats.quantized_trace.rerank_docs = scan.candidates.docs.clone();
            stats.quantized_trace.rerank_docs.sort_unstable();
        }
        let rerank_batch = self.reader.read_vector_rows_planned(
            &scan.candidates.rows,
            &mut survivor_read_ranges,
            &mut survivor_block_scratch,
        )?;
        drop(rerank_fetch_stage);
        stats.rerank_fetch_ns += rerank_fetch_start.elapsed().as_nanos() as u64;
        stats.rerank_rows += scan.candidates.len();

        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        // Row-sorted survivors own their document ids; the batch ordinal addresses both.
        for (candidate, (batch_row, bytes)) in rerank_batch.iter().enumerate() {
            debug_assert_eq!(scan.candidates.rows[candidate], batch_row);
            let doc = scan.candidates.docs[candidate];
            let score_start = Instant::now();
            {
                let _rerank_score_stage = enter_vector_stage(Stage::RerankScore);
                let score = self.query.score_doc_bytes(bytes);
                if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                    topn.push_unordered(key, doc);
                }
            }
            stats.rerank_score_ns += score_start.elapsed().as_nanos() as u64;
            stats.exact_rows_read += 1;
        }
        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| {
                (
                    candidate.sort_key,
                    DocAddress::new(segment_ord, candidate.doc),
                )
            })
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }

    /// Top-N by IVF probe. Fills `stats` with this segment's probe-loop
    /// counters.
    /// Returns the top matches from an IVF probe.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn approximate_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        let max_doc = segment_reader.max_doc();
        if max_doc == 0 {
            return Ok(Vec::new());
        }

        let non_vector_start = Instant::now();
        let non_vector_stage = enter_vector_stage(Stage::NonVectorSearch);
        let filter = build_segment_filter(weight, segment_reader, max_doc)?;
        drop(non_vector_stage);
        let non_vector_search_ns = non_vector_start.elapsed().as_nanos() as u64;
        stats.non_vector_search_ns = stats
            .non_vector_search_ns
            .saturating_add(non_vector_search_ns);
        if filter.is_empty() {
            drop(init_stage);
            stats.scan_init_ns = stats.scan_init_ns.saturating_add(
                (init_start.elapsed().as_nanos() as u64).saturating_sub(non_vector_search_ns),
            );
            return Ok(Vec::new());
        }
        let alive = segment_reader.alive_bitset();

        let num_centroids = index.num_clusters();
        if num_centroids == 0 {
            drop(init_stage);
            stats.scan_init_ns = stats.scan_init_ns.saturating_add(
                (init_start.elapsed().as_nanos() as u64).saturating_sub(non_vector_search_ns),
            );
            return Ok(Vec::new());
        }
        // Capacity counts native docs as WRITTEN (deleted rows still
        // charge on first touch - see `WorkModel::for_searcher`), and the
        // open share x is derived from the index's own n_avg at query
        // init - see `open_share`.
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(num_centroids, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };

        // Phase 1: rank the clusters to probe, lazily — the scan below pulls
        // ranked clusters on demand, so routing cost is paid only as far as
        // probing actually reaches. The filter-effective budget can pull far
        // past its nominal cluster count on a selective filter (each passed-
        // over cluster streams few unseen rows), and lazy routing keeps that
        // cheap.
        // Routing operates in `f32` (centroid rows are `f32` today), so the
        // query is widened losslessly per element.
        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        let mut routing_ws = RouterWorkspace::default();
        stats.segment_rows = Some(index.num_rows());
        stats.segment_clusters = Some(index.num_clusters());
        drop(init_stage);
        stats.scan_init_ns = stats.scan_init_ns.saturating_add(
            (init_start.elapsed().as_nanos() as u64).saturating_sub(non_vector_search_ns),
        );
        // The stacked router is told how many clusters this budget buys
        // under the filter and the recall target; it drops to the fixed
        // nprobe path itself when the dimension is past `APS_MAX_DIM`.
        let routing = RoutingParams {
            k: self.adaptive.router_k(
                work_budget,
                x,
                filter.match_fraction(max_doc),
                num_centroids,
            ),
            recall: self.adaptive.router_recall_target,
        };
        let routing_start = Instant::now();
        let (mut ranked, controller) = {
            let _routing_stage = enter_vector_stage(Stage::Routing);
            let ranked = index.rank_clusters(&mut routing_ws, &query_f32, routing);
            let estimator =
                index.recall_estimator(&ranked, &query_f32, self.adaptive.recall_target);
            let controller = ProbeController::new(
                pricing,
                num_centroids,
                estimator,
                self.adaptive.recall_target,
            );
            (ranked, controller)
        };
        let mut routing_ns = routing_start.elapsed().as_nanos() as u64;
        let routing_before_scan = routing_ns;

        let scan_start = Instant::now();
        let exact_scan_stage = enter_vector_stage(Stage::ExactScan);
        let topn = self.scan_clusters(
            index,
            &mut ranked,
            controller,
            &RowGate::new(&filter, alive),
            top_n,
            tie_break,
            tie_comparator,
            &query_f32,
            stats,
            &mut routing_ns,
        )?;
        drop(exact_scan_stage);
        stats.routing_ns += routing_ns;
        stats.exact_scan_ns = Some(
            scan_start
                .elapsed()
                .as_nanos()
                .saturating_sub(u128::from(routing_ns.saturating_sub(routing_before_scan)))
                as u64,
        );

        stats.record_routing(ranked.metrics());

        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }

    /// Probes ranked clusters under bounds and work-budget gates, filtering before payload reads.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters<K, CTail>(
        &self,
        index: &IvfIndex,
        ranked: &mut impl Iterator<Item = Candidate>,
        mut controller: ProbeController<'_>,
        row_gate: &RowGate<'_>,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        routing_query: &[f32],
        stats: &mut ProbeStats,
        routing_ns: &mut u64,
    ) -> crate::Result<TieBreakHeap<K, CTail>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        // `candidates` is the cumulative scored count that drives the gate; the
        // prune counters accumulate into locals and fold into `ProbeStats` once
        // after the loop, keeping the hot per-doc path free of indirection.
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut bounds_skips = 0u32;
        // P2: the query bound, maintained at cluster boundaries. The
        // bound-space conversion runs on kth improvement only, inside the
        // tracker.
        let metric = self.query.metric();
        let mut bound_tracker = QueryBoundTracker::new();
        // P4: `||q||` for the dot margin's Cauchy-Schwarz term; once per
        // segment-query.
        let q_norm = norm_squared_wide(self.query.query()).sqrt() as f32;
        let bounds = index.bounds();
        // The probed cluster's gate survivors; allocated once, reused
        // across clusters.
        let mut cluster_docs = Vec::new();
        let mut selection_offsets = Vec::new();
        let mut selected_rows = Vec::new();
        let mut read_ranges = Vec::new();
        let mut block_scratch = Vec::new();
        // The heap's kth after the last covered cluster.
        let mut kth: Option<Score> = None;

        loop {
            let routing_start = Instant::now();
            let next = {
                let _routing_stage = enter_vector_stage(Stage::Routing);
                ranked.next()
            };
            *routing_ns += routing_start.elapsed().as_nanos() as u64;
            let Some(Candidate { sim, node: cluster }) = next else {
                break;
            };
            if !controller.admit() {
                break;
            }
            let cluster = cluster as usize;

            // P5: the bounds verdict. The bound is consumed only through
            // `Armed` (the heap holds k results) — enforced by the enum;
            // Filling probes, and SATURATED probes arithmetically (+inf
            // margin). The margin closure runs on armed clusters only.
            let qb = bound_tracker.bound();
            let verdict = bounds_verdict(qb, || {
                let QueryBound::Armed { t } = qb else {
                    // `bounds_verdict` never calls the margin while
                    // Filling; +inf keeps even that impossibility
                    // fail-open.
                    return f32::INFINITY;
                };
                // The separation IS the routing key the ranked stream
                // already computed: `to_bound_space` maps the similarity
                // key into the metric's distance space for L2/cosine
                // (the heap-key and routing-key spaces coincide), and
                // dot consumes the raw `q . c` key directly.
                #[cfg(debug_assertions)]
                {
                    // Precondition of every margin: the stream key is the
                    // EXACT centroid similarity — an approximate key
                    // makes a skip unsound.
                    let stride = self.reader.options().bytes_per_vector();
                    let centroid_bytes = index.centroid_bytes().expect("readable centroid rows");
                    let exact = metric.similarity_bytes::<f32>(
                        routing_query,
                        &centroid_bytes[cluster * stride..(cluster + 1) * stride],
                    );
                    debug_assert_eq!(
                        sim, exact,
                        "routing stream key must be the exact centroid similarity"
                    );
                }
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if let Verdict::Skip = verdict {
                // A skip charges the open share: skips are search work,
                // and free skips break the work identity (validated to
                // +-0.03% in benchmarks). No row work is spent.
                controller.charge_open();
                controller.cover(kth)?;
                bounds_skips += 1;
                continue;
            }

            // Event-wise charging, part 1: the open.
            controller.charge_open();

            let rows = index.cluster_range(cluster);

            if rows.is_empty() {
                postings_skipped += 1;
                controller.cover(kth)?;
                continue;
            }
            let open = matches!(row_gate, RowGate::Open);
            if !open {
                self.reader.read_doc_ids(cluster, &mut cluster_docs)?;
            }
            let (selection, v, pf, pd) = select_cluster_rows(
                &cluster_docs,
                rows.clone(),
                row_gate,
                &mut selection_offsets,
            );
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;
            let scored_rows = selection.len(&rows);
            controller.charge_rows(scored_rows);
            candidates += scored_rows;
            if scored_rows == 0 {
                postings_skipped += 1;
            } else {
                postings_row += 1;
                if open {
                    let bytes = self.reader.read_cluster_rows(cluster)?;
                    let mut docs_resolved = false;
                    for (local, vbytes) in bytes
                        .chunks_exact(self.reader.options().bytes_per_vector())
                        .enumerate()
                    {
                        #[cfg(test)]
                        stats.quantized_trace.scored_rows.push(rows.start + local);
                        let score = self.query.score_doc_bytes(vbytes);
                        if topn
                            .threshold
                            .as_ref()
                            .is_some_and(|((threshold, _), _)| score < *threshold)
                        {
                            continue;
                        }
                        if !docs_resolved {
                            self.reader.read_doc_ids(cluster, &mut cluster_docs)?;
                            docs_resolved = true;
                        }
                        let doc = cluster_docs[local];
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push_unordered(key, doc);
                        }
                    }
                } else {
                    selected_rows.clear();
                    match selection {
                        Selection::All => selected_rows.extend(rows.clone()),
                        Selection::Rows(offsets) => {
                            selected_rows.extend(offsets.iter().map(|&offset| rows.start + offset))
                        }
                        Selection::None => unreachable!("nonempty selection"),
                    }
                    let batch = self.reader.read_vector_rows_planned(
                        &selected_rows,
                        &mut read_ranges,
                        &mut block_scratch,
                    )?;
                    for (row, bytes) in batch.iter() {
                        #[cfg(test)]
                        stats.quantized_trace.scored_rows.push(row);
                        let doc = cluster_docs[row - rows.start];
                        let score = self.query.score_doc_bytes(bytes);
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push_unordered(key, doc);
                        }
                    }
                }
            }

            // P2: fold the exact kth into the bound at the cluster
            // boundary. `kth_best` is O(buffer) and force-truncates —
            // results and every counter above are unaffected (truncation
            // only drops already-lost entries and tightens the push
            // threshold, which prunes pushes, not scoring).
            let probe_idx = (postings_row + postings_skipped - 1) as u32;
            kth = topn.kth_best().map(|(score, _tie)| score);
            bound_tracker.observe(metric, HeapPeek::from_kth(kth), probe_idx);
            controller.cover(kth)?;
        }
        // The armed index exists exactly when the bound armed.
        debug_assert!(
            bound_tracker.armed_at_probe().is_some()
                == matches!(bound_tracker.bound(), QueryBound::Armed { .. })
        );

        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += candidates;
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(bound_tracker.armed_at_probe());
        controller.finish(stats);
        #[cfg(test)]
        {
            stats.quantized_trace.scored_docs.sort_unstable();
        }

        Ok(topn)
    }
}

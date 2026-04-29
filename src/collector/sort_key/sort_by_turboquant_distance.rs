//! `SortByTurboQuantDistance` — top-k by TurboQuant-estimated distance.
//!
//! Single-path collector: every segment is expected to have at least
//! one cluster (the writer emits a trivial single-cluster window for
//! sub-threshold doc counts) so query execution is uniformly:
//!
//!   1. Probe centroids per window via the cluster plugin's HNSW.
//!   2. For each probed cluster, fetch its raw bytes (coalesced across clusters) and slice records
//!      out by 16-doc batch.
//!   3. Score each batch with the NEON kernel; threshold-skip whole batches whose max score can't
//!      beat the heap min.
//!
//! Higher score = more similar (we push raw IP into the TopK; the
//! `NaturalComparator` keeps the largest values).

use std::sync::{Arc, OnceLock};

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::docset::{DocSet, TERMINATED};
use crate::schema::Field;
use crate::vector::cluster::plugin::{ClusterPluginReader, ProbeConfig, WINDOW_SIZE};
use crate::vector::turboquant::transposed::{self as t9d, BatchedQueryLut, BATCH_DOCS};
use crate::vector::turboquant::{TurboQuantQuery, TurboQuantizer};
use crate::{DocAddress, DocId, Score};

/// Per-query state computed once per search, shared across segments.
/// `batched_lut` is the i8-quantized per-coord codebook table used
/// by the SIMD batched scorer (~12 KB at d = 768).
struct QueryState {
    batched_lut: BatchedQueryLut,
}

#[derive(Clone)]
pub struct SortByTurboQuantDistance {
    query_vector: Vec<f32>,
    field: Field,
    /// Quantizer used at index time. The collector needs it to build
    /// the per-query state from `query_vector`. All segments of a
    /// single index must share the same quantizer (same rotator seeds,
    /// codebook, etc.).
    quantizer: TurboQuantizer,
    probe: ProbeConfig,
    query_state: Arc<OnceLock<QueryState>>,
}

impl SortByTurboQuantDistance {
    pub fn new(query_vector: Vec<f32>, field: Field, quantizer: TurboQuantizer) -> Self {
        Self {
            query_vector,
            field,
            quantizer,
            probe: ProbeConfig::default(),
            query_state: Arc::new(OnceLock::new()),
        }
    }

    pub fn with_probe(mut self, probe: ProbeConfig) -> Self {
        self.probe = probe;
        self
    }
}

impl SortKeyComputer for SortByTurboQuantDistance {
    type SortKey = Score;
    type Child = TurboQuantSegmentComputer;
    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        false
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(TurboQuantSegmentComputer)
    }

    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        let field = self.field;

        // Cluster file is the only per-doc record store; the writer
        // always emits at least one cluster per non-empty window
        // (single trivial cluster below clustering_threshold, real
        // k-means above), so there is no separate "unclustered"
        // path. A missing or empty cluster reader = no docs to score.
        let cluster_reader: Option<Arc<ClusterPluginReader>> =
            reader.plugin_reader::<ClusterPluginReader>("cluster")?;
        let cluster_field = cluster_reader.as_ref().and_then(|c| c.field_reader(field));
        let Some(cluster_field) = cluster_field else {
            return Ok(Vec::new());
        };
        if !cluster_field.is_clustered() {
            return Ok(Vec::new());
        }

        // Build per-query state once and reuse for every segment.
        let qs = self.query_state.get_or_init(|| {
            let tq_query = TurboQuantQuery::new(&self.quantizer, &self.query_vector);
            let batched_lut = BatchedQueryLut::new(&tq_query);
            QueryState { batched_lut }
        });
        let batched_lut = &qs.batched_lut;

        let mut top_n: TopNComputer<Score, DocId, NaturalComparator> =
            TopNComputer::new_with_comparator(k, NaturalComparator);

        let mut filter_scorer = weight.scorer(reader, 1.0)?;
        let has_filter = filter_scorer.size_hint() < reader.max_doc();

        {
            let bitset_words = WINDOW_SIZE.div_ceil(64);
            const COALESCE_GAP_TOLERANCE: usize = 16 * 1024;
            let batch_b = t9d::batch_bytes(self.quantizer.padded_dim);

            // Best (smallest) nearest-centroid distance observed in
            // any *probed* window of this segment. Carried across the
            // window loop to power the cross-window skip: once the
            // top-K heap is full, a window whose own nearest centroid
            // is much farther than this minimum can't plausibly
            // contain a top-K candidate and is dropped wholesale.
            let mut global_nearest_centroid_dist = f32::INFINITY;
            let early_stop_enabled = self.probe.distance_ratio.is_finite();

            for win_idx in 0..cluster_field.num_windows() {
                let win = cluster_field.window_reader(win_idx);
                if !win.is_clustered() {
                    continue;
                }
                let win_offset = win_idx * WINDOW_SIZE;
                let win_num_docs = win.num_docs as usize;
                let win_end = (win_offset + win_num_docs) as DocId;

                // Per-window seen bitset for replicated assignment:
                // when a doc is replicated across multiple clusters,
                // probing more than one of them yields the same
                // (local_doc_id, score) tuple. The TurboQuant record
                // bytes are identical across replicas (same encoding,
                // just placed in multiple cluster batches), so the
                // first-seen score equals every later score. Skip
                // already-pushed docs without re-scoring them.
                //
                // For non-replicated indexes every bit naturally
                // stays unset on the second visit, so this is
                // effectively a no-op and costs one bit test per
                // scored doc.
                let mut seen_bits = vec![0u64; bitset_words];

                // Build the filter bitset first so we can drop windows
                // that contain zero matching docs without paying for
                // their HNSW centroid probe. Walking the filter scorer
                // through a window is cheap: for selective range
                // filters whose matches lie in other windows the
                // scorer just seeks past the window in O(1).
                let mut filter_bits = vec![0u64; bitset_words];
                let mut filter_matched_any = !has_filter;
                if has_filter {
                    let mut doc = filter_scorer.doc();
                    if doc < win_offset as DocId {
                        doc = filter_scorer.seek(win_offset as DocId);
                    }
                    while doc != TERMINATED && doc < win_end {
                        let local = (doc as usize) - win_offset;
                        filter_bits[local / 64] |= 1u64 << (local % 64);
                        filter_matched_any = true;
                        doc = filter_scorer.advance();
                    }
                }
                if !filter_matched_any {
                    continue;
                }

                // Probe centroids using the raw query vector. Cluster
                // plugin's centroid index is in unrotated input space,
                // matching how it was built.
                let centroid_results =
                    win.search_centroids(&self.query_vector, self.probe.max_probe);
                let win_nearest = centroid_results
                    .first()
                    .map(|r| r.1)
                    .unwrap_or(f32::INFINITY);

                // Cross-window skip: if the heap already has K valid
                // candidates and this window's nearest centroid is
                // too far compared to the best probed elsewhere, no
                // doc in this window can beat the heap's worst.
                let cross_window_skip = early_stop_enabled
                    && top_n.len() >= k
                    && global_nearest_centroid_dist.is_finite()
                    && win_nearest > global_nearest_centroid_dist * self.probe.distance_ratio;
                if cross_window_skip {
                    if has_filter && filter_scorer.doc() < win_end {
                        filter_scorer.seek(win_end);
                    }
                    continue;
                }

                // Adaptive intra-window probe iteration. The first
                // pass touches `initial_probe` clusters; each
                // follow-up pass touches `probe_step` more, capped at
                // `max_probe`. The loop terminates early when the
                // heap holds K valid candidates and the next
                // un-probed centroid is too far to plausibly beat
                // the heap (per `distance_ratio`).
                //
                // Each pass is two-phase:
                //   1. Coalesced doc_ids fetch for the batch of cluster ids; intersect with
                //      `filter_bits` to drop clusters whose docs are all filtered out (saves the
                //      much larger records read).
                //   2. Coalesced records fetch for the survivors; score every 16-doc batch with
                //      NEON; push valid lanes into top_n with the per-batch threshold
                //      short-circuit.
                let max_to_probe = self.probe.max_probe.min(centroid_results.len());
                let mut processed = 0;
                while processed < max_to_probe {
                    let take = if processed == 0 {
                        self.probe.initial_probe
                    } else {
                        self.probe.probe_step
                    }
                    .max(1);
                    let end = (processed + take).min(max_to_probe);

                    let probe_ids: Vec<u32> = centroid_results[processed..end]
                        .iter()
                        .map(|(c, _)| *c)
                        .collect();
                    let doc_ids_per_cluster = win
                        .cluster_doc_ids_many(&probe_ids, COALESCE_GAP_TOLERANCE)
                        .unwrap_or_else(|_| vec![None; probe_ids.len()]);

                    let mut surviving_ids: Vec<u32> = Vec::with_capacity(probe_ids.len());
                    let mut surviving_doc_ids: Vec<Vec<DocId>> =
                        Vec::with_capacity(probe_ids.len());
                    for (i, doc_ids_opt) in doc_ids_per_cluster.into_iter().enumerate() {
                        let Some(doc_ids) = doc_ids_opt else { continue };
                        if has_filter {
                            let any_survives = doc_ids.iter().any(|&d| {
                                let local = d as usize;
                                (filter_bits[local / 64] >> (local % 64)) & 1 == 1
                            });
                            if !any_survives {
                                continue;
                            }
                        }
                        surviving_ids.push(probe_ids[i]);
                        surviving_doc_ids.push(doc_ids);
                    }

                    let records_per_cluster = win
                        .cluster_records_raw_many(&surviving_ids, COALESCE_GAP_TOLERANCE)
                        .unwrap_or_else(|_| vec![None; surviving_ids.len()]);

                    let mut scores = [0.0f32; BATCH_DOCS];
                    for (entry, local_doc_ids) in records_per_cluster
                        .into_iter()
                        .zip(surviving_doc_ids.into_iter())
                    {
                        let Some((batch_meta, raw)) = entry else {
                            continue;
                        };
                        let num_docs = batch_meta.num_docs as usize;
                        let num_batches = num_docs.div_ceil(BATCH_DOCS);

                        for batch_idx in 0..num_batches {
                            let lo_slot = batch_idx * BATCH_DOCS;
                            let hi_slot = (lo_slot + BATCH_DOCS).min(num_docs);
                            let off = batch_idx * batch_b;
                            let batch = &raw[off..off + batch_b];

                            // Two-phase scoring: compute Stage 1 only first
                            // (cheap — i8-LUT pass over the stage1_t slab),
                            // derive a tight upper bound on the eventual
                            // full score, and skip Stage 2 + the heap push
                            // entirely if no doc in this 16-doc batch can
                            // beat the current TopK threshold. Once the
                            // heap is full this fires on the majority of
                            // batches and saves the per-coord Stage-2
                            // sign-XOR + f32 accumulate work plus the
                            // s2_t slab read.
                            let stage1 = t9d::score_batch_stage1(batched_lut, batch);
                            if let Some(threshold) = top_n.threshold {
                                if stage1.max_score_upper_bound(batched_lut) <= threshold {
                                    continue;
                                }
                            }
                            t9d::score_batch_finish(batched_lut, batch, &stage1, &mut scores);

                            for slot in 0..(hi_slot - lo_slot) {
                                let local_did = local_doc_ids[lo_slot + slot];
                                let local = local_did as usize;
                                if has_filter && (filter_bits[local / 64] >> (local % 64)) & 1 == 0
                                {
                                    continue;
                                }
                                // Replication dedup: bail before the
                                // heap push if we've already scored
                                // this doc via another cluster.
                                let mask = 1u64 << (local % 64);
                                if seen_bits[local / 64] & mask != 0 {
                                    continue;
                                }
                                seen_bits[local / 64] |= mask;
                                let segment_did = (win_offset + local) as DocId;
                                top_n.push(scores[slot], segment_did);
                            }
                        }
                    }

                    processed = end;

                    // Intra-window early stop: heap has K survivors
                    // AND the next un-probed centroid in this window
                    // is too far to beat the heap.
                    if processed >= max_to_probe {
                        break;
                    }
                    if !early_stop_enabled {
                        continue;
                    }
                    if top_n.len() < k {
                        continue;
                    }
                    if processed < self.probe.min_probe {
                        continue;
                    }
                    let next_dist = centroid_results[processed].1;
                    if win_nearest > 0.0 && next_dist > win_nearest * self.probe.distance_ratio {
                        break;
                    }
                }

                // Window contributed; update the cross-window
                // baseline so subsequent windows are gated against it.
                if win_nearest.is_finite() {
                    global_nearest_centroid_dist = global_nearest_centroid_dist.min(win_nearest);
                }
            }
        }

        Ok(top_n
            .into_vec()
            .into_iter()
            .map(|cid| (cid.sort_key, DocAddress::new(segment_ord, cid.doc)))
            .collect())
    }
}

pub struct TurboQuantSegmentComputer;

impl SegmentSortKeyComputer for TurboQuantSegmentComputer {
    type SortKey = Score;
    type SegmentSortKey = Score;
    type SegmentComparator = NaturalComparator;

    fn segment_sort_key(&mut self, _doc: DocId, score: Score) -> Score {
        score
    }

    fn convert_segment_sort_key(&self, score: Score) -> Score {
        score
    }
}

use std::sync::Arc;

use crate::collector::sort_key::{ReverseComparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::docset::{DocSet, TERMINATED};
use crate::schema::Field;
use crate::vector::bqvec::BqVecPluginReader;
use crate::vector::cluster::plugin::{ClusterPluginReader, ProbeConfig, WINDOW_SIZE};
use crate::vector::rabitq::fastscan::{self, BATCH_SIZE};
use crate::vector::rabitq::rotation::DynamicRotator;
use crate::vector::rabitq::{self, Metric, RaBitQQuery};
use crate::{DocAddress, DocId, Score};

#[derive(Clone)]
pub struct SortByVectorDistance {
    query_vector: Vec<f32>,
    field: Field,
    probe: ProbeConfig,
}

impl SortByVectorDistance {
    pub fn new(query_vector: Vec<f32>, field: Field) -> Self {
        Self {
            query_vector,
            field,
            probe: ProbeConfig::default(),
        }
    }

    pub fn with_probe(mut self, probe: ProbeConfig) -> Self {
        self.probe = probe;
        self
    }
}

impl SortKeyComputer for SortByVectorDistance {
    type SortKey = Score;
    type Child = VectorDistanceSegmentComputer;
    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        false
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(VectorDistanceSegmentComputer)
    }

    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        eprintln!(
            "[VEC_TANTIVY] collect_segment_top_k seg={segment_ord} k={k} max_doc={} field={:?}",
            reader.max_doc(),
            self.field
        );
        let field = self.field;

        let bq_plugin: Arc<BqVecPluginReader> = reader
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .ok_or_else(|| {
                crate::TantivyError::InternalError("bqvec plugin reader not found".into())
            })?;
        let bq_reader = bq_plugin.field_reader(field).ok_or_else(|| {
            crate::TantivyError::InternalError("bqvec field reader not found".into())
        })?;

        let cluster_plugin: Arc<ClusterPluginReader> = reader
            .plugin_reader::<ClusterPluginReader>("cluster")?
            .ok_or_else(|| {
                crate::TantivyError::InternalError("cluster plugin reader not found".into())
            })?;
        let cluster_field = cluster_plugin.field_reader(field).ok_or_else(|| {
            crate::TantivyError::InternalError("cluster field reader not found".into())
        })?;

        let meta = cluster_field.field_meta().ok_or_else(|| {
            crate::TantivyError::InternalError("vector field meta not found".into())
        })?;

        let rotator = Arc::new(DynamicRotator::new(
            meta.dims,
            meta.rotator_type,
            meta.rotator_seed,
        ));
        let rabitq_query = RaBitQQuery::new(
            &self.query_vector,
            &rotator,
            meta.ex_bits,
            meta.metric,
        );

        let padded_dims = meta.padded_dims;
        let ex_bits = meta.ex_bits;
        let dim_bytes = padded_dims / 8;
        let ex_b = rabitq::record::ex_bytes(padded_dims, ex_bits);
        let binary_bytes = padded_dims / 8;
        let rec_scalar_off = binary_bytes + ex_b;

        let mut top_n: TopNComputer<Score, DocId, NaturalComparator> =
            TopNComputer::new_with_comparator(k, NaturalComparator);

        // Build filter scorer from the query's weight
        let mut filter_scorer = weight.scorer(reader, 1.0)?;
        // AllQuery has size_hint == max_doc — treat as no filter
        let has_filter = filter_scorer.size_hint() < reader.max_doc();

        if cluster_field.is_clustered() {
            let bitset_words = (WINDOW_SIZE + 63) / 64;

            for win_idx in 0..cluster_field.num_windows() {
                let win = cluster_field.window_reader(win_idx);
                if !win.is_clustered() {
                    continue;
                }
                let win_offset = win_idx * WINDOW_SIZE;
                let win_num_docs = win.num_docs as usize;

                // Build bounded filter bitset for this window
                let mut filter_bits = vec![0u64; bitset_words];
                let has_window_filter = has_filter;
                if has_window_filter {
                    let win_end = (win_offset + win_num_docs) as DocId;
                    let mut doc = filter_scorer.doc();
                    if doc < win_offset as DocId {
                        doc = filter_scorer.seek(win_offset as DocId);
                    }
                    while doc != TERMINATED && doc < win_end {
                        let local = (doc as usize) - win_offset;
                        filter_bits[local / 64] |= 1u64 << (local % 64);
                        doc = filter_scorer.advance();
                    }
                }

                let centroid_results =
                    win.search_centroids(&self.query_vector, self.probe.max_probe);
                let nearest_dist = centroid_results.first().map(|r| r.1).unwrap_or(0.0);

                for (i, &(cluster_id, dist)) in centroid_results.iter().enumerate() {
                    if i >= self.probe.min_probe
                        && nearest_dist > 0.0
                        && dist / nearest_dist > self.probe.distance_ratio
                    {
                        break;
                    }
                    let g_add = dist;

                    if let Ok(Some((local_doc_ids, batch_meta, raw))) =
                        win.cluster_batch_raw(cluster_id as usize)
                    {
                        let num_docs = batch_meta.num_docs as usize;
                        let num_batches = batch_meta.num_batches as usize;
                        let db = dim_bytes;

                        let doc_id_bytes = num_docs * 4;
                        let codes_bytes = num_batches * db * BATCH_SIZE;
                        let scalars_per_batch = BATCH_SIZE;
                        let total_scalars = num_batches * scalars_per_batch;

                        let codes_start = doc_id_bytes;
                        let f_add_start = codes_start + codes_bytes;
                        let f_rescale_start = f_add_start + total_scalars * 4;
                        let f_error_start = f_rescale_start + total_scalars * 4;

                        for batch_idx in 0..num_batches {
                            let batch_start = batch_idx * BATCH_SIZE;
                            let batch_end = (batch_start + BATCH_SIZE).min(num_docs);
                            let batch_local_ids = &local_doc_ids[batch_start..batch_end];

                            // Filter check via bitset
                            let mut matched = [true; BATCH_SIZE];
                            let mut any_match = !batch_local_ids.is_empty();
                            if has_window_filter {
                                any_match = false;
                                matched = [false; BATCH_SIZE];
                                for (j, &local_did) in batch_local_ids.iter().enumerate() {
                                    let local = local_did as usize;
                                    if (filter_bits[local / 64] >> (local % 64)) & 1 != 0 {
                                        matched[j] = true;
                                        any_match = true;
                                    }
                                }
                            }
                            if !any_match {
                                continue;
                            }

                            // SIMD batch accumulate
                            let tc_off = codes_start + batch_idx * db * BATCH_SIZE;
                            let transposed = &raw[tc_off..tc_off + db * BATCH_SIZE];

                            let mut accu = [0u32; BATCH_SIZE];
                            fastscan::accumulate_batch(
                                transposed,
                                rabitq_query.lut().lut_u8(),
                                db,
                                &mut accu,
                            );

                            let mut binary_dots = [0.0f32; BATCH_SIZE];
                            fastscan::denormalize_batch(
                                &accu,
                                rabitq_query.lut().delta(),
                                rabitq_query.lut().sum_vl(),
                                &mut binary_dots,
                            );

                            let scalar_off_base = batch_idx * scalars_per_batch;
                            let read_f32_arr = |start: usize, idx: usize| -> f32 {
                                let off = start + (scalar_off_base + idx) * 4;
                                f32::from_le_bytes([
                                    raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                                ])
                            };

                            let mut distances = [0.0f32; BATCH_SIZE];
                            let mut lower_bounds = [0.0f32; BATCH_SIZE];
                            for j in 0..BATCH_SIZE {
                                let fa = read_f32_arr(f_add_start, j);
                                let fr = read_f32_arr(f_rescale_start, j);
                                let fe = read_f32_arr(f_error_start, j);
                                let binary_term = binary_dots[j] + rabitq_query.k1x_sum_q();
                                distances[j] = fa + g_add + fr * binary_term;
                                lower_bounds[j] = distances[j] - fe.abs();
                            }

                            for (j, &local_did) in batch_local_ids.iter().enumerate() {
                                if !matched[j] { continue; }
                                let raw_threshold = -(top_n.threshold.unwrap_or(f32::MIN));
                                if lower_bounds[j] >= raw_threshold { continue; }

                                let segment_did = (win_offset + local_did as usize) as DocId;
                                let distance = if ex_bits > 0 {
                                    if let Ok(record) = bq_reader.record(segment_did) {
                                        let ex_code_packed =
                                            &record[binary_bytes..binary_bytes + ex_b];
                                        let ex_dot = fastscan::ip_packed_ex_f32(
                                            rabitq_query.rotated_query(),
                                            ex_code_packed,
                                            padded_dims,
                                            ex_bits,
                                        );
                                        let f_add_ex = f32::from_le_bytes([
                                            record[rec_scalar_off + 24],
                                            record[rec_scalar_off + 25],
                                            record[rec_scalar_off + 26],
                                            record[rec_scalar_off + 27],
                                        ]);
                                        let f_rescale_ex = f32::from_le_bytes([
                                            record[rec_scalar_off + 28],
                                            record[rec_scalar_off + 29],
                                            record[rec_scalar_off + 30],
                                            record[rec_scalar_off + 31],
                                        ]);
                                        let total_term = rabitq_query.binary_scale()
                                            * binary_dots[j]
                                            + ex_dot
                                            + rabitq_query.kbx_sum_q();
                                        let dist =
                                            f_add_ex + g_add + f_rescale_ex * total_term;
                                        match meta.metric {
                                            Metric::L2 => dist,
                                            Metric::InnerProduct => -dist,
                                        }
                                    } else { continue; }
                                } else {
                                    distances[j]
                                };

                                let score = -distance;
                                top_n.push(score, segment_did);
                            }
                        }
                    }
                }
            }
        } else {
            // Unclustered fallback
            let mut doc = filter_scorer.doc();
            while doc != TERMINATED {
                if let Ok(record) = bq_reader.record(doc) {
                    let dist = rabitq_query.estimate_distance_from_record(
                        &record, padded_dims, 0.0,
                    );
                    let score = -dist;
                    top_n.push(score, doc);
                }
                doc = filter_scorer.advance();
            }
        }

        Ok(top_n
            .into_vec()
            .into_iter()
            .map(|cid| (cid.sort_key, DocAddress::new(segment_ord, cid.doc)))
            .collect())
    }
}

pub struct VectorDistanceSegmentComputer;

impl SegmentSortKeyComputer for VectorDistanceSegmentComputer {
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

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::vector::bqvec::{BqVecFieldReader, BqVecPluginReader};
use crate::vector::cluster::plugin::{ClusterFieldReader, ClusterPluginReader, ProbeConfig};
use crate::docset::{DocSet, TERMINATED};
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::score_combiner::DoNothingCombiner;
use crate::query::{
    AllScorer, BufferedUnionScorer, ConstScorer, EnableScoring, Explanation, Intersection, Query,
    Scorer, Weight,
};
use crate::vector::rabitq::fastscan::{self, QueryLut, BATCH_SIZE};
use crate::vector::rabitq::rotation::DynamicRotator;
use crate::vector::rabitq::{self, Metric, RaBitQQuery};
use crate::schema::Field;
use crate::{DocId, Score};

pub struct VectorQueryConfig {
    pub field: Field,
    pub padded_dims: usize,
    pub ex_bits: usize,
    pub metric: Metric,
    pub rotator: Arc<DynamicRotator>,
    pub probe: ProbeConfig,
}

pub struct VectorQuery {
    query_vector: Vec<f32>,
    config: Arc<VectorQueryConfig>,
    filter: Option<Box<dyn Query>>,
}

impl VectorQuery {
    pub fn new(query_vector: Vec<f32>, config: VectorQueryConfig) -> Self {
        Self {
            query_vector,
            config: Arc::new(config),
            filter: None,
        }
    }

    pub fn with_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.filter = Some(filter);
        self
    }
}

impl Clone for VectorQuery {
    fn clone(&self) -> Self {
        Self {
            query_vector: self.query_vector.clone(),
            config: self.config.clone(),
            filter: self.filter.as_ref().map(|f| f.box_clone()),
        }
    }
}

impl fmt::Debug for VectorQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorQuery")
            .field("field", &self.config.field)
            .field("dims", &self.query_vector.len())
            .finish()
    }
}

impl Query for VectorQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let filter_weight = self
            .filter
            .as_ref()
            .map(|f| f.weight(enable_scoring))
            .transpose()?;
        Ok(Box::new(VectorWeight {
            query_vector: self.query_vector.clone(),
            config: self.config.clone(),
            filter_weight,
        }))
    }
}

struct VectorWeight {
    query_vector: Vec<f32>,
    config: Arc<VectorQueryConfig>,
    filter_weight: Option<Box<dyn Weight>>,
}

struct CandidateResult {
    scorer: Box<dyn Scorer>,
    centroid_dists: HashMap<u16, f32>,
}

impl VectorWeight {
    fn build_candidates(
        &self,
        cluster_reader: &ClusterFieldReader,
        max_doc: u32,
    ) -> crate::Result<CandidateResult> {
        if !cluster_reader.is_clustered() {
            return Ok(CandidateResult {
                scorer: Box::new(AllScorer::new(max_doc)),
                centroid_dists: HashMap::new(),
            });
        }

        let centroid_results =
            cluster_reader.search_centroids(&self.query_vector, self.config.probe.max_probe);
        if centroid_results.is_empty() {
            return Ok(CandidateResult {
                scorer: Box::new(AllScorer::new(max_doc)),
                centroid_dists: HashMap::new(),
            });
        }

        let mut centroid_dists = HashMap::new();
        for &(cluster_id, dist) in &centroid_results {
            centroid_dists.insert(cluster_id as u16, dist);
        }

        let probed = cluster_reader.probe_clusters(&self.query_vector, &self.config.probe)?;
        if probed.is_empty() {
            return Ok(CandidateResult {
                scorer: Box::new(AllScorer::new(max_doc)),
                centroid_dists,
            });
        }

        let scorer: Box<dyn Scorer> = if probed.len() == 1 {
            let (_, postings) = probed.into_iter().next().unwrap();
            Box::new(ConstScorer::new(postings, 1.0))
        } else {
            let scorers: Vec<ConstScorer<SegmentPostings>> = probed
                .into_iter()
                .map(|(_, postings)| ConstScorer::new(postings, 1.0))
                .collect();
            Box::new(BufferedUnionScorer::build(
                scorers,
                DoNothingCombiner::default,
                max_doc,
            ))
        };

        Ok(CandidateResult {
            scorer,
            centroid_dists,
        })
    }
}

impl Weight for VectorWeight {
    fn scorer(&self, reader: &SegmentReader, _boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let field = self.config.field;

        let bq_plugin: Arc<BqVecPluginReader> = reader
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .ok_or_else(|| {
                crate::TantivyError::InternalError("bqvec plugin reader not found".into())
            })?;
        if bq_plugin.field_reader(field).is_none() {
            return Err(crate::TantivyError::InternalError(
                "bqvec field reader not found".into(),
            ));
        }

        let cluster_plugin: Arc<ClusterPluginReader> = reader
            .plugin_reader::<ClusterPluginReader>("cluster")?
            .ok_or_else(|| {
                crate::TantivyError::InternalError("cluster plugin reader not found".into())
            })?;
        let cluster_field = cluster_plugin.field_reader(field).ok_or_else(|| {
            crate::TantivyError::InternalError("cluster field reader not found".into())
        })?;

        let result = self.build_candidates(cluster_field, reader.max_doc())?;

        let doc_set: Box<dyn DocSet> = match &self.filter_weight {
            Some(filter_weight) => {
                let filter_scorer = filter_weight.scorer(reader, 1.0)?;
                Box::new(Intersection::with_two_sets(
                    result.scorer,
                    filter_scorer,
                    reader.max_doc(),
                ))
            }
            None => result.scorer,
        };

        let rabitq_query = RaBitQQuery::new(
            &self.query_vector,
            &self.config.rotator,
            self.config.ex_bits,
            self.config.metric,
        );

        Ok(Box::new(VectorScorer {
            doc_set,
            bq_plugin: bq_plugin.clone(),
            cluster_plugin: cluster_plugin.clone(),
            field,
            rabitq_query,
            padded_dims: self.config.padded_dims,
            centroid_dists: result.centroid_dists,
        }))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
        Ok(Explanation::new("VectorQuery", 0.0))
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        let field = self.config.field;

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

        let rabitq_query = RaBitQQuery::new(
            &self.query_vector,
            &self.config.rotator,
            self.config.ex_bits,
            self.config.metric,
        );

        let padded_dims = self.config.padded_dims;
        let dim_bytes = padded_dims / 8;
        let ex_bits = self.config.ex_bits;
        let ex_b = rabitq::record::ex_bytes(padded_dims, ex_bits);
        let scalar_off = dim_bytes + ex_b;

        if cluster_field.is_clustered() {
            let mut neg_threshold = threshold;
            let db = cluster_field.dim_bytes();
            let binary_bytes = padded_dims / 8;
            let rec_scalar_off = binary_bytes + ex_b;

            // Build filter scorer once — advances forward across windows
            let mut filter_scorer: Option<Box<dyn Scorer>> = self
                .filter_weight
                .as_ref()
                .map(|fw| fw.scorer(reader, 1.0))
                .transpose()?;

            // Iterate windows in doc ID order (contiguous ranges)
            use crate::vector::cluster::plugin::WINDOW_SIZE;
            let bitset_words = (WINDOW_SIZE + 63) / 64;

            for win_idx in 0..cluster_field.num_windows() {
                let win = cluster_field.window_reader(win_idx);
                if !win.is_clustered() {
                    continue;
                }
                let win_offset = win_idx * WINDOW_SIZE;
                let win_num_docs = win.num_docs as usize;

                // Build bounded filter bitset for this window (15KB max)
                let mut filter_bits = vec![0u64; bitset_words];
                let has_filter = filter_scorer.is_some();
                if let Some(ref mut fs) = filter_scorer {
                    // Advance filter scorer through this window's doc range
                    let win_end = (win_offset + win_num_docs) as DocId;
                    let mut doc = fs.doc();
                    if doc < win_offset as DocId {
                        doc = fs.seek(win_offset as DocId);
                    }
                    while doc != TERMINATED && doc < win_end {
                        let local = (doc as usize) - win_offset;
                        filter_bits[local / 64] |= 1u64 << (local % 64);
                        doc = fs.advance();
                    }
                }

                // Centroid search within this window
                let centroid_results =
                    win.search_centroids(&self.query_vector, self.config.probe.max_probe);

                let nearest_dist = centroid_results.first().map(|r| r.1).unwrap_or(0.0);

                for (i, &(cluster_id, dist)) in centroid_results.iter().enumerate() {
                    if i >= self.config.probe.min_probe
                        && nearest_dist > 0.0
                        && dist / nearest_dist > self.config.probe.distance_ratio
                    {
                        break;
                    }
                    let g_add = dist;

                    if let Ok(Some((local_doc_ids, meta, raw))) =
                        win.cluster_batch_raw(cluster_id as usize)
                    {
                        let num_docs = meta.num_docs as usize;
                        let num_batches = meta.num_batches as usize;

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

                            // Filter check via bitset (O(1), order-independent)
                            let mut matched = [true; BATCH_SIZE];
                            let mut any_match = !batch_local_ids.is_empty();
                            if has_filter {
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
                                let binary_term =
                                    binary_dots[j] + rabitq_query.k1x_sum_q();
                                distances[j] = fa + g_add + fr * binary_term;
                                lower_bounds[j] = distances[j] - fe.abs();
                            }

                            let raw_threshold = -neg_threshold;
                            for (j, &local_did) in batch_local_ids.iter().enumerate() {
                                if !matched[j] {
                                    continue;
                                }
                                if lower_bounds[j] >= raw_threshold {
                                    continue;
                                }
                                // Map window-local doc ID to segment doc ID
                                let segment_did =
                                    (win_offset + local_did as usize) as DocId;
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
                                        match self.config.metric {
                                            Metric::L2 => dist,
                                            Metric::InnerProduct => -dist,
                                        }
                                    } else {
                                        continue;
                                    }
                                } else {
                                    distances[j]
                                };
                                let score = -distance;
                                if score > neg_threshold {
                                    neg_threshold = callback(segment_did, score);
                                }
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        // Unclustered fallback: per-doc scalar path
        let mut neg_threshold = threshold;
        let mut doc_set: Box<dyn DocSet> = Box::new(AllScorer::new(reader.max_doc()));
        let mut doc = doc_set.doc();
        while doc != TERMINATED {
            if let Ok(record) = bq_reader.record(doc) {
                let raw_threshold = -neg_threshold;
                if let Some(distance) = rabitq_query.estimate_distance_pruned(
                    &record,
                    padded_dims,
                    0.0,
                    raw_threshold,
                ) {
                    let score = -distance;
                    if score > neg_threshold {
                        neg_threshold = callback(doc, score);
                    }
                }
            }
            doc = doc_set.advance();
        }
        Ok(())
    }
}

struct VectorScorer {
    doc_set: Box<dyn DocSet>,
    bq_plugin: Arc<BqVecPluginReader>,
    cluster_plugin: Arc<ClusterPluginReader>,
    field: Field,
    rabitq_query: RaBitQQuery,
    padded_dims: usize,
    centroid_dists: HashMap<u16, f32>,
}

impl DocSet for VectorScorer {
    fn advance(&mut self) -> DocId {
        self.doc_set.advance()
    }

    fn doc(&self) -> DocId {
        self.doc_set.doc()
    }

    fn size_hint(&self) -> u32 {
        self.doc_set.size_hint()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.doc_set.seek(target)
    }
}

impl Scorer for VectorScorer {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        if doc == TERMINATED {
            return f32::MIN;
        }
        let bq_reader = match self.bq_plugin.field_reader(self.field) {
            Some(r) => r,
            None => return f32::MIN,
        };
        let g_add = self
            .cluster_plugin
            .field_reader(self.field)
            .filter(|cr| cr.is_clustered())
            .map(|cr| {
                let cluster_id = cr.doc_cluster(doc);
                self.centroid_dists.get(&cluster_id).copied().unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        match bq_reader.record(doc) {
            Ok(record) => {
                // Negate so that TopDocs (descending sort) ranks closer vectors higher.
                -self
                    .rabitq_query
                    .estimate_distance_from_record(&record, self.padded_dims, g_add)
            }
            Err(_) => f32::MIN,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::vector::bqvec::BqVecPlugin;
    use crate::vector::cluster::kmeans::KMeansConfig;
    use crate::vector::cluster::plugin::{ClusterConfig, ClusterFieldConfig, ClusterPlugin, ProbeConfig};
    use crate::vector::cluster::sampler::test_utils::InMemorySamplerFactory;
    use crate::collector::TopDocs;
    use crate::docset::{DocSet, TERMINATED};
    use crate::plugin::SegmentPlugin;
    use crate::query::{EnableScoring, Query, TermQuery};
    use crate::vector::rabitq::rotation::DynamicRotator;
    use crate::vector::rabitq::{self, Metric, RabitqConfig, RotatorType};
    use crate::schema::{Field, IndexRecordOption, Schema, Term, STORED, STRING, TEXT};
    use crate::vector::search::{VectorQuery, VectorQueryConfig};
    use crate::{Index, IndexWriter};

    const DIMS: usize = 32;
    const CLUSTER_THRESHOLD: u32 = 3;

    fn make_rotator() -> Arc<DynamicRotator> {
        Arc::new(DynamicRotator::new(DIMS, RotatorType::MatrixRotator, 42))
    }

    struct TestIndex {
        index: Index,
        #[allow(dead_code)]
        text_field: Field,
        category_field: Field,
        vec_field: Field,
        rotator: Arc<DynamicRotator>,
        vectors: Vec<Vec<f32>>,
    }

    fn build_test_index(num_docs: usize, cluster: bool) -> crate::Result<TestIndex> {
        use rand::prelude::*;

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let category_field = schema_builder.add_text_field("category", STRING);
        let vec_field = schema_builder.add_vector_field("embedding", DIMS);
        let schema = schema_builder.build();

        let rotator = make_rotator();
        let padded_dims = rotator.padded_dim();

        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let vectors: Vec<Vec<f32>> = (0..num_docs)
            .map(|_| (0..DIMS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let shared_vecs = Arc::new(Mutex::new(vectors.clone()));

        let config = RabitqConfig::new(1);
        let rotator_enc = rotator.clone();
        let bqvec = Arc::new(
            BqVecPlugin::builder()
                .vector_field(
                    vec_field,
                    rabitq::bytes_per_record(padded_dims, 6),
                    Arc::new(move |v: &[f32]| {
                        let zero = vec![0.0f32; v.len()];
                        rabitq::encode(&rotator_enc, &config, Metric::L2, v, &zero)
                    }),
                )
                .build(),
        );

        let threshold = if cluster { CLUSTER_THRESHOLD } else { u32::MAX };
        let cluster_plugin = Arc::new(ClusterPlugin::new(ClusterConfig {
            clustering_threshold: threshold,
            sample_ratio: 1.0,
            sample_cap: 100_000,
            kmeans: KMeansConfig {
                niter: 20,
                nredo: 1,
                seed: 42,
                ..Default::default()
            },
            num_clusters_fn: Arc::new(|n| (n / 4).max(2)),
            fields: vec![ClusterFieldConfig {
                field: vec_field,
                dims: DIMS,
                padded_dims,
                ex_bits: 6,
                metric: Metric::L2,
                rotator: rotator.clone(),
            }],
            sampler_factory: Arc::new(InMemorySamplerFactory {
                vectors: shared_vecs,
            }),
        }));

        let index = Index::builder()
            .schema(schema)
            .plugin(bqvec as Arc<dyn SegmentPlugin>)
            .plugin(cluster_plugin as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
        for (i, v) in vectors.iter().enumerate() {
            let category = if i % 2 == 0 { "even" } else { "odd" };
            writer.add_document(crate::doc!(
                text_field => format!("doc{i}"),
                category_field => category,
                vec_field => v.clone()
            ))?;
        }
        writer.commit()?;

        Ok(TestIndex {
            index,
            text_field,
            category_field,
            vec_field,
            rotator,
            vectors,
        })
    }

    fn make_query_config(
        t: &TestIndex,
        max_probe: usize,
        distance_ratio: f32,
    ) -> VectorQueryConfig {
        VectorQueryConfig {
            field: t.vec_field,
            padded_dims: t.rotator.padded_dim(),
            ex_bits: 6,
            metric: Metric::L2,
            rotator: t.rotator.clone(),
            probe: ProbeConfig::new(max_probe, distance_ratio),
        }
    }

    #[test]
    fn test_vector_search_no_filter() -> crate::Result<()> {
        let t = build_test_index(100, true)?;
        let reader = t.index.reader()?;
        let searcher = reader.searcher();

        let query = VectorQuery::new(t.vectors[0].clone(), make_query_config(&t, 10, 100.0));
        let top_docs = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;

        assert_eq!(top_docs.len(), 5);
        assert_eq!(top_docs[0].1.doc_id, 0);
        for &(score, _) in &top_docs {
            assert!(score.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_vector_search_with_filter() -> crate::Result<()> {
        let t = build_test_index(100, true)?;
        let reader = t.index.reader()?;
        let searcher = reader.searcher();

        let term = Term::from_field_text(t.category_field, "even");
        let filter = TermQuery::new(term, IndexRecordOption::Basic);

        let query = VectorQuery::new(t.vectors[0].clone(), make_query_config(&t, 10, 100.0))
            .with_filter(Box::new(filter));
        let top_docs = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;

        assert!(!top_docs.is_empty());
        for &(_, doc_addr) in &top_docs {
            assert_eq!(doc_addr.doc_id % 2, 0, "expected only even docs");
        }

        Ok(())
    }

    #[test]
    fn test_vector_search_unclustered() -> crate::Result<()> {
        let t = build_test_index(20, false)?;
        let reader = t.index.reader()?;
        let searcher = reader.searcher();

        let query = VectorQuery::new(t.vectors[0].clone(), make_query_config(&t, 10, 100.0));
        let top_docs = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;

        assert_eq!(top_docs.len(), 5);
        assert_eq!(top_docs[0].1.doc_id, 0);

        Ok(())
    }

    #[test]
    fn test_vector_search_top1_accuracy() -> crate::Result<()> {
        let t = build_test_index(200, true)?;
        let reader = t.index.reader()?;
        let searcher = reader.searcher();

        for doc_id in 0..10u32 {
            let query = VectorQuery::new(
                t.vectors[doc_id as usize].clone(),
                make_query_config(&t, 20, 100.0),
            );
            let top_docs = searcher.search(&query, &TopDocs::with_limit(1).order_by_score())?;
            assert_eq!(
                top_docs[0].1.doc_id, doc_id,
                "top-1 for doc {doc_id} should be itself"
            );
        }

        Ok(())
    }

    #[test]
    fn test_vector_scorer_iterates_all_candidates() -> crate::Result<()> {
        let t = build_test_index(50, true)?;
        let reader = t.index.reader()?;
        let searcher = reader.searcher();

        let query = VectorQuery::new(t.vectors[0].clone(), make_query_config(&t, 50, 100.0));
        let weight = query.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;

        let seg_reader = &searcher.segment_readers()[0];
        let mut scorer = weight.scorer(seg_reader, 1.0)?;

        let mut count = 0;
        while scorer.doc() != TERMINATED {
            assert!(scorer.score().is_finite());
            count += 1;
            scorer.advance();
        }
        assert!(count > 0);

        Ok(())
    }
}

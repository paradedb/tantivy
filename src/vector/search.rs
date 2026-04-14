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

        let result = self.build_candidates(cluster_field, reader.max_doc())?;

        let mut doc_set: Box<dyn DocSet> = match &self.filter_weight {
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

        let padded_dims = self.config.padded_dims;
        let dim_bytes = padded_dims / 8;
        let ex_bits = self.config.ex_bits;
        let ex_b = rabitq::record::ex_bytes(padded_dims, ex_bits);
        let scalar_off = dim_bytes + ex_b;

        if cluster_field.is_clustered() {
            let mut neg_threshold = threshold;
            let probed =
                cluster_field.probe_clusters(&self.query_vector, &self.config.probe)?;

            // Collect (cluster_id, doc_ids) per cluster, then sort clusters
            // by min doc ID so a single filter scorer can seek forward
            let mut cluster_docs: Vec<(u16, f32, Vec<DocId>)> = Vec::new();
            for (cluster_id, mut postings) in probed {
                let cid = cluster_id as u16;
                let g_add = result.centroid_dists.get(&cid).copied().unwrap_or(0.0);
                let mut doc_ids = Vec::new();
                let mut doc = postings.doc();
                while doc != TERMINATED {
                    doc_ids.push(doc);
                    doc = postings.advance();
                }
                if !doc_ids.is_empty() {
                    cluster_docs.push((cid, g_add, doc_ids));
                }
            }
            cluster_docs.sort_unstable_by_key(|(_, _, ids)| ids[0]);

            let mut filter_scorer: Option<Box<dyn Scorer>> = self
                .filter_weight
                .as_ref()
                .map(|fw| fw.scorer(reader, 1.0))
                .transpose()?;

            for (_cid, g_add, ref doc_ids) in &cluster_docs {
                let g_add = *g_add;

                // Process in batches of 32
                for chunk in doc_ids.chunks(BATCH_SIZE) {
                    // Filter check: seek forward for each doc in batch.
                    // Doc IDs within a cluster are sorted. Across sorted clusters,
                    // some IDs may be behind the filter's current position — skip those.
                    let mut matched = [false; BATCH_SIZE];
                    let mut any_match = false;
                    for (i, &did) in chunk.iter().enumerate() {
                        if let Some(ref mut fs) = filter_scorer {
                            let cur = fs.doc();
                            if cur == TERMINATED {
                                break;
                            }
                            if cur <= did {
                                if fs.seek(did) == did {
                                    matched[i] = true;
                                    any_match = true;
                                }
                            }
                            // else: filter already past this doc, not a match
                        } else {
                            matched[i] = true;
                            any_match = true;
                        }
                    }
                    if !any_match {
                        continue;
                    }

                // Read binary codes + scalars for this batch
                let mut codes: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
                let mut f_add_batch = [0.0f32; BATCH_SIZE];
                let mut f_rescale_batch = [0.0f32; BATCH_SIZE];
                let mut f_error_batch = [0.0f32; BATCH_SIZE];

                for (i, &did) in chunk.iter().enumerate() {
                    if let Ok(record) = bq_reader.record(did) {
                        codes.push(record[..dim_bytes].to_vec());
                        let read_f32 = |off: usize| -> f32 {
                            f32::from_le_bytes([
                                record[scalar_off + off],
                                record[scalar_off + off + 1],
                                record[scalar_off + off + 2],
                                record[scalar_off + off + 3],
                            ])
                        };
                        f_add_batch[i] = read_f32(8);
                        f_rescale_batch[i] = read_f32(12);
                        f_error_batch[i] = read_f32(16);
                    } else {
                        codes.push(vec![0u8; dim_bytes]);
                    }
                }
                while codes.len() < BATCH_SIZE {
                    codes.push(vec![0u8; dim_bytes]);
                }

                // Transpose + SIMD batch accumulate
                let code_refs: Vec<&[u8]> =
                    codes.iter().map(|c| c.as_slice()).collect();
                let mut transposed = vec![0u8; dim_bytes * BATCH_SIZE];
                fastscan::pack_batch_simple(&code_refs, dim_bytes, &mut transposed);

                let mut accu = [0u32; BATCH_SIZE];
                fastscan::accumulate_batch(
                    &transposed,
                    rabitq_query.lut().lut_u8(),
                    dim_bytes,
                    &mut accu,
                );

                let mut binary_dots = [0.0f32; BATCH_SIZE];
                fastscan::denormalize_batch(
                    &accu,
                    rabitq_query.lut().delta(),
                    rabitq_query.lut().sum_vl(),
                    &mut binary_dots,
                );

                let mut distances = [0.0f32; BATCH_SIZE];
                let mut lower_bounds = [0.0f32; BATCH_SIZE];
                fastscan::compute_batch_distances(
                    &binary_dots,
                    &f_add_batch,
                    &f_rescale_batch,
                    &f_error_batch,
                    g_add,
                    rabitq_query.k1x_sum_q(),
                    &mut distances,
                    &mut lower_bounds,
                );

                let raw_threshold = -neg_threshold;
                for (i, &did) in chunk.iter().enumerate() {
                    if !matched[i] {
                        continue;
                    }
                    if lower_bounds[i] >= raw_threshold {
                        continue;
                    }
                    let distance = if ex_bits > 0 {
                        if let Ok(record) = bq_reader.record(did) {
                            rabitq_query.estimate_distance_from_record(
                                &record, padded_dims, g_add,
                            )
                        } else {
                            continue;
                        }
                    } else {
                        distances[i]
                    };
                    let score = -distance;
                    if score > neg_threshold {
                        neg_threshold = callback(did, score);
                    }
                }
                }
            }
            return Ok(());
        }

        // Unclustered fallback: per-doc scalar path
        let mut neg_threshold = threshold;
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

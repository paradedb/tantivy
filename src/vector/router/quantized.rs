//! Version-1 compressed stacked storage. Runtime routing and scoring live in
//! the shared stacked algorithm; legacy files retain their original policy.

use std::io::{self, Write};
use std::sync::Arc;

use common::{BinarySerializable, HasLen};

use super::source::{EncodingContext, VectorSet, VectorSource};
use super::stacked::levels::{read_level_topology, InMemoryStore, LazyStore, StackedIvf};
use super::stacked::StackedRouter;
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::storage::quantized::sign::{invalid, SignVectors};

const VERSION: u32 = 1;

pub(crate) fn serialize(
    router: &StackedRouter<InMemoryStore>,
    out: &mut (impl Write + ?Sized),
) -> io::Result<()> {
    VERSION.serialize(out)?;
    (router.dim as u32).serialize(out)?;
    metric_code(router.metric).serialize(out)?;
    let (_, context) = router
        .level
        .vectors
        .compressed()
        .ok_or_else(|| invalid("missing compressed member bank"))?;
    let seed = context.seed;
    seed.serialize(out)?;
    router
        .level
        .serialize_topology(out, |store| store.num_vectors(router.dim))?;
    router
        .level
        .vectors
        .compressed()
        .unwrap()
        .0
        .serialize(out)?;
    for level in router.level.levels() {
        let (codes, context) = level
            .centroids
            .compressed()
            .ok_or_else(|| invalid("version-1 router requires encoded internal banks"))?;
        if context.seed != seed || level.centroids.raw().is_some() {
            return Err(invalid(
                "version-1 router cannot persist independent contexts or raw internal banks",
            ));
        }
        codes.serialize(out)?;
    }
    Ok(())
}
fn metric_code(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::Dot => 1,
        Metric::Cosine => 2,
    }
}

pub(crate) fn open(
    payload: FileSlice,
    raw: FileSlice,
    options: &VectorOptions,
) -> crate::Result<StackedRouter<LazyStore>> {
    let mut payload = payload;
    let read_metadata = |input: &mut FileSlice, len: usize| -> io::Result<common::OwnedBytes> {
        if len > input.len() {
            return Err(invalid("truncated quantized router metadata"));
        }
        let bytes = input.slice_to(len).read_bytes()?;
        *input = input.slice_from(len);
        Ok(bytes)
    };
    // Fixed header only. The remaining cursor skips vector column ranges
    // without fetching them, including the small root centroid codes.
    let header = read_metadata(&mut payload, 21)?;
    let mut input = header.as_slice();
    if u32::deserialize(&mut input)? != VERSION {
        return Err(
            invalid("unsupported quantized router payload version; rebuild required").into(),
        );
    }
    let dim = u32::deserialize(&mut input)? as usize;
    let metric = u8::deserialize(&mut input)?;
    if dim == 0 || dim != options.dim() || metric != metric_code(options.metric()) {
        return Err(invalid("quantized router dimension or metric mismatch").into());
    }
    let seed = u64::deserialize(&mut input)?;
    let count = u32::deserialize(&mut input)? as usize;
    if count == 0 || count > payload.len() / 4 {
        return Err(invalid("invalid quantized router level count").into());
    }
    let topology = read_level_topology(count, |len| read_metadata(&mut payload, len))?;
    let context = Arc::new(EncodingContext::new(dim, seed));
    let mut sets = Vec::with_capacity(count + 1);
    for _ in 0..=count {
        sets.push(Arc::new(SignVectors::open(&mut payload, dim)?));
    }
    if !payload.is_empty() {
        return Err(invalid("trailing quantized router bytes").into());
    }
    let raw_len = sets[0]
        .len()
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("quantized router raw size overflow"))?;
    if raw.len() != raw_len {
        return Err(invalid("quantized router member count differs from raw centroid rows").into());
    }
    let mut raw = Some(Arc::new(LazyStore::new(raw, dim)));
    let sets = sets
        .into_iter()
        .enumerate()
        .map(|(bank, codes)| VectorSet::Encoded {
            codes,
            context: context.clone(),
            raw: if bank == 0 { raw.take() } else { None },
        })
        .collect();
    let level = StackedIvf::from_vector_sets(topology, sets, Default::default(), |source| {
        source.num_vectors(dim)
    })?;
    StackedRouter::new(level, options)
}
#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Mutex;

    use super::*;
    use crate::directory::FileHandle;
    use crate::vector::clustering::{IvfCentroids, IvfMatrix};
    use crate::vector::router::candidate::Candidate;
    use crate::vector::router::scoring::score_initial;
    use crate::vector::router::search::{
        self, rerank_exact, OutputPrecision, ScoreBuffer, SearchPolicy, TopCandidates,
    };
    use crate::vector::router::source::ROUTER_SEED;
    use crate::vector::router::stacked::levels::ClusterId;
    use crate::vector::router::{RouterMetrics, RoutingParams};
    use crate::vector::search::prepared::VectorQuery;

    #[derive(Debug)]
    struct TrackedRaw {
        bytes: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
    }
    impl HasLen for TrackedRaw {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }
    impl FileHandle for TrackedRaw {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<common::OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(common::OwnedBytes::new(self.bytes[range].to_vec()))
        }
    }

    fn encoded_codes<S>(source: &VectorSet<S>) -> &Arc<SignVectors> {
        match source {
            VectorSet::Encoded { codes, .. } => codes,
            _ => panic!("expected codes"),
        }
    }

    fn vectors(dim: usize, count: usize) -> IvfCentroids {
        IvfCentroids::F32(IvfMatrix {
            values: (0..dim * count)
                .map(|i| ((i * 31 % 257) as f32 * 0.17).sin())
                .collect(),
            rows: count,
            dims: dim,
        })
    }

    #[test]
    fn open_reads_metadata_only_and_probing_reads_only_selected_columns() -> crate::Result<()> {
        let (dim, rows) = (65, 96);
        let options = VectorOptions::new(dim, Metric::L2);
        let mut centroids = vectors(dim, rows);
        let built = StackedRouter::build(&options, &mut centroids, true)?;
        assert!(built.level.parent.is_none());
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        let topology_end = 21 + 4 + built.level.offsets.len() * 16;
        let codes_start = topology_end + 4;
        let words = dim.div_ceil(64);
        let coefficients_start = codes_start + rows * words * 8;
        let norms_start = coefficients_start + rows * 4;
        let root_header = norms_start + rows * 4;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let tracked = FileSlice::new(Arc::new(TrackedRaw {
            bytes: payload,
            reads: reads.clone(),
        }));
        let raw = FileSlice::from(vec![0u8; rows * dim * 4]);
        let opened = open(tracked, raw, &options)?;
        assert_eq!(
            *reads.lock().unwrap(),
            vec![
                0..21,
                21..25,
                25..topology_end,
                topology_end..codes_start,
                root_header..root_header + 4,
            ]
        );
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.15; dim]));
        let mut actual = ScoreBuffer::default();
        let mut expected = ScoreBuffer::default();
        reads.lock().unwrap().clear();
        let (start, end) = opened.level.offsets[0];
        score_initial(
            &opened.level.vectors,
            &query,
            start as usize..end as usize,
            &mut actual,
        );
        score_initial(
            &built.level.vectors,
            &query,
            start as usize..end as usize,
            &mut expected,
        );
        assert_eq!(actual.candidates, expected.candidates);
        let (start, end) = opened.level.offsets[0];
        let (start, end) = (start as usize, end as usize);
        assert_eq!(
            *reads.lock().unwrap(),
            vec![
                codes_start + start * words * 8..codes_start + end * words * 8,
                coefficients_start + start * 4..coefficients_start + end * 4,
                norms_start + start * 4..norms_start + end * 4,
            ]
        );
        reads.lock().unwrap().clear();
        let empty = opened
            .level
            .vectors
            .compressed()
            .unwrap()
            .0
            .read(rows..rows)?;
        assert!(empty.codes().is_empty());
        assert!(reads.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn flat_batch_matches_exhaustive_beam_and_reads_leaf_columns_once() -> crate::Result<()> {
        let (dim, rows) = (65, 96);
        let options = VectorOptions::new(dim, Metric::Cosine);
        let mut centroids = vectors(dim, rows);
        let built = StackedRouter::build(&options, &mut centroids, true)?;
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        let topology_end = 21 + 4 + built.level.offsets.len() * 16;
        let codes_start = topology_end + 4;
        let coefficients_start = codes_start + rows * dim.div_ceil(64) * 8;
        let norms_start = coefficients_start + rows * 4;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let raw = FileSlice::from(
            built
                .level
                .vectors
                .raw()
                .unwrap()
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let opened = open(
            FileSlice::new(Arc::new(TrackedRaw {
                bytes: payload,
                reads: reads.clone(),
            })),
            raw,
            &options,
        )?;
        let query = VectorQuery::new(Metric::Cosine, Arc::new(vec![0.15; dim]));
        let params = RoutingParams {
            k: 3,
            min_beam_fraction: 1.0,
            aps: false,
            ..Default::default()
        };
        let hierarchical = opened.rank(&query, params).collect::<Vec<_>>();
        reads.lock().unwrap().clear();
        // Exercise flat composition internally without a production search flag.
        let (flat, stats) = search::rank_vectors(
            &opened.level.vectors,
            &query,
            params.k,
            SearchPolicy {
                aps: false,
                min_fraction: params.min_beam_fraction,
                rerank_factor: params.rerank_factor,
            },
            OutputPrecision::Exact,
        );
        assert_eq!(stats.lists_scanned, 0);
        assert_eq!(stats.approximate_scored, 96);
        assert_eq!(stats.leaf_exact_scored, params.k);
        let flat: Vec<_> = flat
            .into_iter()
            .map(|c| Candidate {
                node: c.node.0,
                sim: c.sim,
            })
            .collect();
        assert_eq!(flat, hierarchical);
        assert_eq!(
            *reads.lock().unwrap(),
            vec![
                codes_start..coefficients_start,
                coefficients_start..norms_start,
                norms_start..norms_start + rows * 4,
            ]
        );
        Ok(())
    }

    #[test]
    fn persisted_router_reads_only_rerank_rows_and_returns_exact_scores() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let dim = 65;
            let options = VectorOptions::new(dim, metric);
            let mut centroids = vectors(dim, 96);
            let built = StackedRouter::build(&options, &mut centroids, true)?;
            let mut payload = Vec::new();
            built.serialize(&mut payload)?;
            let IvfCentroids::F32(matrix) = centroids;
            let reads = Arc::new(Mutex::new(Vec::new()));
            let raw = FileSlice::new(Arc::new(TrackedRaw {
                bytes: matrix.values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                reads: reads.clone(),
            }));
            let opened = open(FileSlice::from(payload.clone()), raw, &options)?;
            assert!(
                reads.lock().unwrap().is_empty(),
                "opening must not fetch raw rows"
            );
            let query = VectorQuery::new(metric, Arc::new(vec![0.15; dim]));
            for (factor, retained) in [
                (-1.0, 3),
                (0.0, 3),
                (0.5, 3),
                (f32::NAN, 3),
                (1.0, 3),
                (1.25, 4),
                (1.5, 5),
                (2.0, 6),
                (4.0, 12),
                (f32::MAX, 96),
                (f32::INFINITY, 96),
            ] {
                reads.lock().unwrap().clear();
                let params = RoutingParams {
                    k: 3,
                    rerank_factor: factor,
                    aps: false,
                    ..RoutingParams::default()
                };
                let mut ranking = opened.rank(&query, params);
                let metrics = ranking.metrics();
                let results: Vec<_> = ranking.by_ref().collect();
                assert!(matches!(
                    ranking.metrics(),
                    RouterMetrics::QuantizedStacked {
                        candidate_count: 3,
                        ..
                    }
                ));
                assert_eq!(results.len(), 3);
                assert!(results.windows(2).all(|w| w[0] >= w[1]));
                for c in &results {
                    let row = &matrix.values[c.node as usize * dim..(c.node as usize + 1) * dim];
                    assert_eq!(
                        c.sim,
                        metric.similarity_with_query_norm(
                            query.widened(),
                            row,
                            query.routing_norm()
                        )
                    );
                }
                let RouterMetrics::QuantizedStacked {
                    exact_scored,
                    candidate_count,
                    approximate_scored,
                    approximate_members_scored,
                    ..
                } = metrics
                else {
                    unreachable!()
                };
                assert_eq!(candidate_count, 3);
                assert!(approximate_scored > approximate_members_scored);
                assert!(approximate_members_scored >= exact_scored);
                assert_eq!(exact_scored, retained.min(approximate_members_scored));
                let read_rows: usize = reads
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|r| r.len() / (dim * 4))
                    .sum();
                assert_eq!(
                    read_rows, exact_scored,
                    "only retained rows get exact reads"
                );
            }
            // An exhaustive request must recover the independent flat centroid oracle.
            let mut oracle: Vec<_> = matrix
                .values
                .chunks_exact(dim)
                .enumerate()
                .map(|(node, row)| Candidate {
                    node: node as u32,
                    sim: metric.similarity_with_query_norm(
                        query.widened(),
                        row,
                        query.routing_norm(),
                    ),
                })
                .collect();
            oracle.sort_unstable_by(|a, b| b.cmp(a));
            assert_eq!(
                opened
                    .rank(
                        &query,
                        RoutingParams {
                            aps: false,
                            ..Default::default()
                        }
                    )
                    .collect::<Vec<_>>(),
                oracle
            );

            let raw_bytes: Vec<_> = matrix.values.iter().flat_map(|x| x.to_le_bytes()).collect();
            for end in [0, 4, 8, 9, 17, payload.len() - 1] {
                assert!(open(
                    FileSlice::from(payload[..end].to_vec()),
                    FileSlice::from(raw_bytes.clone()),
                    &options
                )
                .is_err());
            }
            let mut trailing = payload;
            trailing.push(0);
            assert!(open(
                FileSlice::from(trailing.clone()),
                FileSlice::from(raw_bytes.clone()),
                &options
            )
            .is_err());
            let mut wrong_version = trailing[..trailing.len() - 1].to_vec();
            wrong_version[..4].copy_from_slice(&2u32.to_le_bytes());
            assert!(open(
                FileSlice::from(wrong_version),
                FileSlice::from(raw_bytes.clone()),
                &options
            )
            .is_err());
            let mut bad_range = trailing[..trailing.len() - 1].to_vec();
            // Header (21), list count (4), first range's start (8), then end.
            bad_range[33..41].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(open(
                FileSlice::from(bad_range),
                FileSlice::from(raw_bytes),
                &options
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn exact_rerank_ignores_estimates_and_never_adds_excluded_members() {
        let raw = InMemoryStore::new(vec![20.0, 1.0, 0.0, 3.0], 1);
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.0]));
        let candidates = vec![
            Candidate {
                node: ClusterId(0),
                sim: crate::vector::distance::Similarity::new(100.0),
            },
            Candidate {
                node: ClusterId(3),
                sim: crate::vector::distance::Similarity::new(50.0),
            },
            Candidate {
                node: ClusterId(1),
                sim: crate::vector::distance::Similarity::new(-100.0),
            },
        ];
        let result = rerank_exact(&raw, &query, &candidates, 2);
        assert_eq!(result.iter().map(|c| c.node.0).collect::<Vec<_>>(), [1, 3]);
        assert_eq!(
            result.iter().map(|c| c.sim.score()).collect::<Vec<_>>(),
            [-1.0, -9.0]
        );
    }

    #[test]
    fn deeper_hierarchy_shares_vector_sets_and_reranks_only_final_members() -> crate::Result<()> {
        let dim = 8;
        let context = Arc::new(EncodingContext::new(dim, ROUTER_SEED));
        let rotation = &context.rotation;
        let raw: Vec<_> = (0..16).flat_map(|row| vec![row as f32; dim]).collect();
        let centroids: Vec<_> = [1.5, 5.5, 9.5, 13.5]
            .into_iter()
            .flat_map(|x| vec![x; dim])
            .collect();
        let roots: Vec<_> = [3.5, 11.5].into_iter().flat_map(|x| vec![x; dim]).collect();
        let members = VectorSet::Encoded {
            codes: Arc::new(SignVectors::encode(&raw, dim, rotation)?),
            context: context.clone(),
            raw: Some(Arc::new(InMemoryStore::new(raw.clone(), dim))),
        };
        let centroids = VectorSet::Encoded {
            codes: Arc::new(SignVectors::encode(&centroids, dim, rotation)?),
            context: context.clone(),
            raw: None,
        };
        let parent = StackedIvf {
            config: Default::default(),
            parent: None,
            offsets: vec![(0, 2), (2, 4)],
            vectors: centroids.clone(),
            centroids: VectorSet::Encoded {
                codes: Arc::new(SignVectors::encode(&roots, dim, rotation)?),
                context: context.clone(),
                raw: None,
            },
        };
        let built = StackedRouter {
            level: StackedIvf {
                config: Default::default(),
                parent: Some(Box::new(parent)),
                offsets: vec![(0, 4), (4, 8), (8, 12), (12, 16)],
                centroids,
                vectors: members,
            },
            dim,
            metric: Metric::L2,
        };
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        // Three matrices (16, 4, 2 rows), with no duplicated parent members or raw rows.
        assert_eq!(
            payload.len(),
            21 + (4 + 4 * 16) + (4 + 2 * 16) + (4 + 16 * 16) + (4 + 4 * 16) + (4 + 2 * 16)
        );
        let options = VectorOptions::new(dim, Metric::L2);
        let opened = open(
            FileSlice::from(payload),
            FileSlice::from(raw.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()),
            &options,
        )?;
        assert_eq!(opened.level_sizes(), [4, 2]);
        assert!(Arc::ptr_eq(
            encoded_codes(&opened.level.centroids),
            encoded_codes(&opened.level.parent.as_ref().unwrap().vectors)
        ));
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![2.1; dim]));
        let ranking = opened.rank(
            &query,
            RoutingParams {
                k: 16,
                aps: false,
                ..Default::default()
            },
        );
        assert!(matches!(
            ranking.metrics(),
            RouterMetrics::QuantizedStacked {
                exact_scored: 16,
                approximate_scored: 22,
                approximate_members_scored: 16,
                ..
            }
        ));
        let mut oracle: Vec<_> = raw
            .chunks_exact(dim)
            .enumerate()
            .map(|(node, row)| Candidate {
                node: node as u32,
                sim: Metric::L2.similarity(query.widened(), row),
            })
            .collect();
        oracle.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(ranking.collect::<Vec<_>>(), oracle);
        Ok(())
    }

    #[test]
    fn collector_preserves_filtered_deleted_and_quantized_document_results() -> crate::Result<()> {
        use crate::query::AllQuery;
        use crate::vector::test_support::FixtureSpec;
        use crate::vector::{LocatedPath, ProbeParams, RouterKind};

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for schedule in [&[][..], &[1, 1][..]] {
                let fixture = FixtureSpec::clustered(metric, 64, 8, 160)
                    .router(RouterKind::QuantizedStacked)
                    .schedule(schedule)
                    .label("third", |i| i % 3 == 0)
                    .deletes(&[0, 7, 40])
                    .paged()
                    .build()?;
                let query: Vec<_> = (0..64).map(|i| (i as f32 * 0.31).cos()).collect();
                let params = ProbeParams {
                    max_probe: usize::MAX,
                    router_aps: false,
                    located: LocatedPath::Off,
                    ..Default::default()
                };
                let filter = fixture.filter("third");
                for filter in [&AllQuery as &dyn crate::query::Query, &filter] {
                    let expected = fixture.oracle(filter, &query, 7)?;
                    let actual = fixture.search(filter, &query, 7, params.clone())?;
                    assert_eq!(
                        actual.results, expected,
                        "{metric:?}, schedule {schedule:?}"
                    );
                    assert!(actual.stats.iter().any(|stats| matches!(
                        stats.routing,
                        Some(RouterMetrics::QuantizedStacked { .. })
                    )));
                    for stats in actual.stats.iter().filter(|stats| stats.routing.is_some()) {
                        let json = serde_json::to_value(stats)?;
                        assert_eq!(json["routing_kind"], "quantized_stacked");
                        assert_eq!(json["routing_mode"], "hierarchical");
                        let count = |key: &str| json[key].as_u64().unwrap();
                        assert!(count("routing_approximate_scored") > 0);
                        assert_eq!(
                            count("routing_approximate_members_scored"),
                            count("routing_approximate_pruned") + count("routing_exact_scored")
                        );
                        assert_eq!(
                            count("routing_exact_scored"),
                            count("routing_exact_pruned") + count("routing_candidate_count")
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn empty_router_and_zero_request_return_no_candidates() -> crate::Result<()> {
        let options = VectorOptions::new(64, Metric::L2);
        let mut rows = vectors(64, 0);
        let built = StackedRouter::build(&options, &mut rows, true)?;
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        let opened = open(
            FileSlice::from(payload),
            FileSlice::from(Vec::<u8>::new()),
            &options,
        )?;
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.0; 64]));
        assert_eq!(opened.rank(&query, RoutingParams::default()).count(), 0);
        assert_eq!(
            built
                .rank(
                    &query,
                    RoutingParams {
                        k: 0,
                        ..Default::default()
                    }
                )
                .count(),
            0
        );
        Ok(())
    }

    #[test]
    fn centroid_quantization_setting_is_backward_compatible_and_restored_on_open(
    ) -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::{Index, IndexSettings};

        let legacy: IndexSettings = serde_json::from_str("{}")?;
        assert!(!legacy.vector_centroid_quantization);
        assert!(serde_json::to_value(&legacy)?
            .get("vector_centroid_quantization")
            .is_none());

        let directory = RamDirectory::create();
        let schema = crate::schema::Schema::builder().build();
        let settings = IndexSettings {
            vector_centroid_quantization: true,
            ..Default::default()
        };
        let index = Index::create(directory.clone(), schema, settings)?;
        assert_eq!(
            index.ivf_router(),
            super::super::RouterKind::QuantizedStacked
        );
        drop(index);
        let reopened = Index::open(directory)?;
        assert!(reopened.settings().vector_centroid_quantization);
        assert_eq!(
            reopened.ivf_router(),
            super::super::RouterKind::QuantizedStacked
        );
        Ok(())
    }

    /// Small reproducible routing diagnostic; run in release mode with
    /// `cargo test --release --lib quantized_router_measurement -- --ignored --nocapture`.
    /// Reports synthetic centroid recall, not document recall or production tuning.
    #[test]
    #[ignore]
    fn quantized_router_measurement() -> crate::Result<()> {
        use std::time::Instant;

        let (dim, count, queries, k) = (128, 8192, 64, 32);
        let mut state = 42u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
        };
        let mut rows = IvfCentroids::F32(IvfMatrix {
            values: (0..count * dim).map(|_| random()).collect(),
            rows: count,
            dims: dim,
        });
        let queries: Vec<_> = (0..queries)
            .map(|_| VectorQuery::new(Metric::L2, Arc::new((0..dim).map(|_| random()).collect())))
            .collect();
        let options = VectorOptions::new(dim, Metric::L2);
        let index = super::super::stacked::build(&options, &mut rows)?;
        let truth: Vec<_> = queries
            .iter()
            .map(|q| {
                let mut best = TopCandidates::new(k);
                for (row, vector) in index.vectors.chunks_exact(dim).enumerate() {
                    best.offer(Candidate {
                        node: ClusterId(row as u32),
                        sim: Metric::L2.similarity(q.widened(), vector),
                    });
                }
                best.into_sorted()
                    .into_iter()
                    .map(|c| c.node.0)
                    .collect::<Vec<_>>()
            })
            .collect();
        for aps in [false, true] {
            let start = Instant::now();
            let mut found = 0;
            let mut scored = 0;
            for (query, truth) in queries.iter().zip(&truth) {
                let (hits, stats) = search::search(
                    &index,
                    query,
                    k,
                    SearchPolicy {
                        aps,
                        min_fraction: 0.25,
                        rerank_factor: 1.0,
                    },
                    OutputPrecision::Exact,
                );
                let hits = hits;
                found += hits.iter().filter(|c| truth.contains(&c.node.0)).count();
                scored += stats.members_scored;
            }
            eprintln!(
                "full-precision aps={aps}: recall={:.3}, us/query={:.1}, evaluations/query={:.0}",
                found as f64 / (queries.len() * k) as f64,
                start.elapsed().as_secs_f64() * 1e6 / queries.len() as f64,
                scored as f64 / queries.len() as f64
            );
        }
        let quantized = StackedRouter::from_index(index, &options, true)?;
        for aps in [false, true] {
            for factor in [1.0, 2.0, 4.0, 8.0] {
                let start = Instant::now();
                let (mut found, mut approx, mut exact) = (0, 0, 0);
                for (query, truth) in queries.iter().zip(&truth) {
                    let hits = quantized.rank(
                        query,
                        RoutingParams {
                            k,
                            rerank_factor: factor,
                            aps,
                            ..RoutingParams::default()
                        },
                    );
                    if let RouterMetrics::QuantizedStacked {
                        approximate_scored,
                        exact_scored,
                        ..
                    } = hits.metrics()
                    {
                        approx += approximate_scored;
                        exact += exact_scored;
                    }
                    found += hits.filter(|c| truth.contains(&c.node)).count();
                }
                eprintln!(
                    "1-bit aps={aps}, C={factor}: recall={:.3}, us/query={:.1}, \
                     approximate/query={:.0}, exact/query={:.0}",
                    found as f64 / (queries.len() * k) as f64,
                    start.elapsed().as_secs_f64() * 1e6 / queries.len() as f64,
                    approx as f64 / queries.len() as f64,
                    exact as f64 / queries.len() as f64
                );
            }
        }
        Ok(())
    }
    /// Fresh-reader CPU diagnostic, including initialization rather than only a
    /// warmed router. FileSlice memory does not model PostgreSQL buffer costs.
    #[test]
    #[ignore]
    fn quantized_router_open_measurement() -> crate::Result<()> {
        use std::hint::black_box;
        use std::time::Instant;
        let (dim, count, repeats) = (1024, 4096, 128);
        let options = VectorOptions::new(dim, Metric::Cosine);
        let mut centroids = vectors(dim, count);
        let index = super::super::stacked::build(&options, &mut centroids)?;
        let raw = FileSlice::from(
            index
                .vectors
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let mut full_payload = Vec::new();
        index.serialize_router_payload(&mut full_payload)?;
        let full_payload = FileSlice::from(full_payload);
        let quantized = StackedRouter::from_index(index, &options, true)?;
        let mut payload = Vec::new();
        quantized.serialize(&mut payload)?;
        let payload = FileSlice::from(payload);
        let query = VectorQuery::new(
            Metric::Cosine,
            Arc::new((0..dim).map(|i| (i as f32 * 0.19).cos()).collect()),
        );
        for quantized in [false, true] {
            let start = Instant::now();
            for _ in 0..repeats {
                if quantized {
                    black_box(open(payload.clone(), raw.clone(), &options)?);
                } else {
                    black_box(super::super::stacked::open(
                        full_payload.clone(),
                        raw.clone(),
                        &options,
                    )?);
                }
            }
            let open_us = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
            for k in [16, 128, 1024] {
                let params = RoutingParams {
                    k,
                    rerank_factor: 2.0,
                    aps: false,
                    ..RoutingParams::default()
                };
                let start = Instant::now();
                let mut checksum = 0u64;
                for _ in 0..repeats {
                    let hits: Vec<_> = if quantized {
                        open(payload.clone(), raw.clone(), &options)?
                            .rank(&query, params)
                            .collect()
                    } else {
                        let opened = super::super::stacked::open(
                            full_payload.clone(),
                            raw.clone(),
                            &options,
                        )?;
                        super::super::stacked::rank(&opened, &query, params).collect()
                    };
                    checksum += hits.iter().map(|c| c.node as u64).sum::<u64>();
                    black_box(hits);
                }
                eprintln!(
                    "quantized={quantized}, k={k}: open_us={open_us:.1}, open_rank_us={:.1}, \
                     checksum={checksum}",
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                );
            }
        }
        Ok(())
    }
}

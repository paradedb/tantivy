use super::candidate::Candidate;
use super::search::{OutputPrecision, SearchPolicy};
use super::stacked::search::{AllScored, StackedRouting};
use super::{OpenedRouter, Router, RouterMetrics, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::{VectorElement, VectorStage};

/// Clusters to probe across all segments, ordered by centroid similarity to the query.
/// Each cluster carries its segment index, and metrics track the routing work for each segment.
/// With [`RoutingParams::flat`], every segment is ranked as an exact router ranks it and partition
/// centroids are unused; the consumer bounds the merged ranking as it bounds a stored ranking.
pub(crate) struct GlobalRanking {
    pub(crate) clusters: Vec<(usize, Candidate)>,
    pub(crate) metrics: Vec<RouterMetrics>,
}

impl GlobalRanking {
    pub(crate) fn from<T: VectorElement>(
        routers: &[Option<&OpenedRouter>],
        partitions: &[Option<&[f32]>],
        query: &VectorQuery<T>,
        params: RoutingParams,
        totals: &mut [StageTotals],
        on_segment: &mut impl FnMut(usize),
    ) -> Self {
        let mut indexes = Vec::with_capacity(routers.len());
        let mut leaves = Vec::new();
        let mut metrics = vec![
            RouterMetrics::Stacked {
                candidate_count: 0,
                lists_scanned: 0,
                members_scored: 0,
            };
            routers.len()
        ];
        for (segment, router) in routers.iter().enumerate() {
            let index = match router {
                Some(Router::Stacked(index)) if !params.flat => Some(&index.level),
                Some(router) => {
                    let (ranked, measured) = measure(|| {
                        let _stage = enter_vector_stage(VectorStage::Routing);
                        router.rank_exhaustive(query).collect::<Vec<_>>()
                    });
                    totals[segment].append(&measured);
                    on_segment(segment);
                    metrics[segment] = RouterMetrics::Exact {
                        visited_count: ranked.len(),
                    };
                    leaves.extend(ranked.into_iter().map(|c| (segment, c)));
                    None
                }
                None => None,
            };
            indexes.push(index);
        }
        let compressed = routers
            .iter()
            .flatten()
            .any(|router| router.kind() == super::RouterKind::QuantizedStacked);
        let ranking = StackedRouting::Global {
            indexes,
            partitions,
            totals,
            on_segment,
        }
        .search(
            query,
            params.k.max(1),
            SearchPolicy {
                aps: params.aps,
                min_fraction: if compressed {
                    params.min_beam_fraction
                } else {
                    super::stacked::levels::PARENT_NPROBE_FRACTION
                },
                rerank_factor: params.rerank_factor,
            },
            OutputPrecision::Exact,
            AllScored::default(),
        );
        leaves.extend(ranking.candidates.into_iter().map(|candidate| {
            (
                candidate.node.0,
                Candidate {
                    sim: candidate.sim,
                    node: candidate.node.1,
                },
            )
        }));
        leaves.sort_unstable_by(|a, b| {
            b.1.sim
                .cmp(&a.1.sim)
                .then(a.0.cmp(&b.0))
                .then(a.1.node.cmp(&b.1.node))
        });
        let mut counts = vec![0; routers.len()];
        for (segment, _) in &leaves {
            counts[*segment] += 1;
        }
        for (segment, router) in routers.iter().enumerate() {
            if let (Some(Router::Stacked(index)), false) = (router, params.flat) {
                metrics[segment] = super::stacked::routing_metrics(
                    index.kind() == super::RouterKind::QuantizedStacked,
                    counts[segment],
                    ranking.stats[segment],
                );
            }
        }
        Self {
            clusters: leaves,
            metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::directory::FileSlice;
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::clustering::{IvfCentroids, IvfMatrix};
    use crate::vector::router::stacked::levels::{
        ClusterId, InMemoryStore, StackedIvf, StackedIvfConfig,
    };
    use crate::vector::router::stacked::search::TopK;
    use crate::vector::router::RouterKind;

    fn open_line_router(base: f32, count: u64, nested: bool) -> crate::Result<OpenedRouter> {
        let options = VectorOptions::new(1, Metric::L2);
        let rows: Vec<_> = (0..count).map(|i| base + i as f32).collect();
        let parent = nested.then(|| {
            Box::new(StackedIvf {
                config: StackedIvfConfig::default(),
                parent: None,
                offsets: vec![(0, 2), (2, 4)],
                centroids: InMemoryStore::new(vec![rows[0], rows[2]], 1),
                vectors: InMemoryStore::new(rows.clone(), 1),
            })
        });
        let index = StackedIvf {
            config: StackedIvfConfig::default(),
            parent,
            offsets: (0..count).map(|i| (i, i + 1)).collect(),
            centroids: InMemoryStore::new(rows.clone(), 1),
            vectors: InMemoryStore::new(rows.clone(), 1),
        };
        let router = Router::Stacked(super::super::stacked::StackedRouter::from_index(
            index, &options, false,
        )?);
        let mut payload = Vec::new();
        router.serialize(&mut payload)?;
        let bytes: Vec<_> = rows.into_iter().flat_map(f32::to_le_bytes).collect();
        RouterKind::open(FileSlice::from(payload), FileSlice::from(bytes), &options)
    }

    #[test]
    fn partition_children_join_at_their_actual_depth() -> crate::Result<()> {
        for nested in [[true, false, true], [false, true, false]] {
            let indexes = nested
                .into_iter()
                .enumerate()
                .map(|(i, nested)| open_line_router(i as f32 * 100.0, 4, nested))
                .collect::<crate::Result<Vec<_>>>()?;
            let routers: Vec<_> = indexes.iter().map(Some).collect();
            for partitions in [
                vec![Some(&[0.0][..]), Some(&[100.0][..]), None],
                vec![None, Some(&[100.0][..]), Some(&[200.0][..])],
            ] {
                for query in [[0.0], [100.0], [200.0]] {
                    let mut totals = vec![StageTotals::default(); 3];
                    let ranked = GlobalRanking::from(
                        &routers,
                        &partitions,
                        &VectorQuery::new(Metric::L2, Arc::new(query.to_vec())).unwrap(),
                        RoutingParams {
                            k: 1,
                            ..Default::default()
                        },
                        &mut totals,
                        &mut |_| {},
                    );
                    assert_eq!(ranked.clusters[0].0, (query[0] / 100.0) as usize);
                    let full = GlobalRanking::from(
                        &routers,
                        &partitions,
                        &VectorQuery::new(Metric::L2, Arc::new(query.to_vec())).unwrap(),
                        RoutingParams::default(),
                        &mut totals,
                        &mut |_| {},
                    );
                    let plain = GlobalRanking::from(
                        &routers,
                        &[],
                        &VectorQuery::new(Metric::L2, Arc::new(query.to_vec())).unwrap(),
                        RoutingParams::default(),
                        &mut totals,
                        &mut |_| {},
                    );
                    assert_eq!(full.clusters, plain.clusters);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn layers_compete_across_segments_and_depths() -> crate::Result<()> {
        let options = VectorOptions::new(1, Metric::L2);
        let mut centroids = IvfCentroids::F32(IvfMatrix {
            values: vec![0.25],
            rows: 1,
            dims: 1,
        });
        let exact = RouterKind::Exact.build(&options, &mut centroids)?;
        let mut payload = Vec::new();
        exact.serialize(&mut payload)?;
        let exact = RouterKind::open(
            FileSlice::from(payload),
            FileSlice::from(0.25f32.to_le_bytes().to_vec()),
            &options,
        )?;
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.0_f32])).unwrap();
        for (far_nested, near_nested) in
            [(false, false), (true, true), (true, false), (false, true)]
        {
            let far = open_line_router(100.0, 4, far_nested)?;
            let near = open_line_router(0.0, 4, near_nested)?;
            let routers = [Some(&far), None, Some(&near), Some(&exact)];
            let mut totals = vec![StageTotals::default(); routers.len()];
            let ranked = GlobalRanking::from(
                &routers,
                &[],
                &query,
                RoutingParams {
                    k: 0,
                    ..Default::default()
                },
                &mut totals,
                &mut |_| {},
            );
            let exhaustive = GlobalRanking::from(
                &routers,
                &[],
                &query,
                RoutingParams::default(),
                &mut totals,
                &mut |_| {},
            );
            assert_eq!(
                exhaustive.clusters.len(),
                9,
                "depths {far_nested}/{near_nested}"
            );
            let identities: Vec<_> = ranked
                .clusters
                .iter()
                .map(|(segment, c)| (*segment, c.node))
                .collect();
            assert_eq!(
                identities,
                vec![(2, 0), (3, 0), (2, 1)],
                "depths {far_nested}/{near_nested}"
            );
        }
        Ok(())
    }
    #[test]
    fn one_stack_uses_the_same_walk_in_both_constructions() -> crate::Result<()> {
        for nested in [false, true] {
            let Router::Stacked(index) = open_line_router(0.0, 4, nested)? else {
                unreachable!()
            };
            for query in [[0.0], [1.5], [200.0]] {
                let prepared = VectorQuery::new(Metric::L2, Arc::new(query.to_vec())).unwrap();
                for k in [0, 1, 2, 4, 16] {
                    let (segment, stats) = super::super::search::search(
                        &index.level,
                        &prepared,
                        k,
                        SearchPolicy {
                            aps: true,
                            min_fraction: 0.25,
                            rerank_factor: 1.0,
                        },
                        OutputPrecision::Exact,
                    );
                    let mut totals = vec![StageTotals::default(); 2];
                    let mut visited = Vec::new();
                    let global = StackedRouting::Global {
                        indexes: vec![None, Some(&index.level)],
                        partitions: &[],
                        totals: &mut totals,
                        on_segment: &mut |segment| visited.push(segment),
                    }
                    .search(
                        &prepared,
                        k,
                        SearchPolicy {
                            aps: true,
                            min_fraction: 0.25,
                            rerank_factor: 1.0,
                        },
                        OutputPrecision::Exact,
                        TopK,
                    );
                    let best: Vec<_> = global
                        .candidates
                        .into_iter()
                        .map(|candidate| Candidate {
                            sim: candidate.sim,
                            node: ClusterId(candidate.node.1),
                        })
                        .collect();
                    assert_eq!(best, segment);
                    assert_eq!(global.stats[1].lists_scanned, stats.lists_scanned);
                    assert_eq!(global.stats[1].members_scored, stats.members_scored);
                    assert!(visited.iter().all(|&segment| segment == 1));
                    let all = StackedRouting::Global {
                        indexes: vec![None, Some(&index.level)],
                        partitions: &[],
                        totals: &mut totals,
                        on_segment: &mut |_| {},
                    }
                    .search(
                        &prepared,
                        k,
                        SearchPolicy {
                            aps: true,
                            min_fraction: 0.25,
                            rerank_factor: 1.0,
                        },
                        OutputPrecision::Exact,
                        AllScored::default(),
                    );
                    assert_eq!(all.stats[1].lists_scanned, stats.lists_scanned);
                    assert_eq!(all.stats[1].members_scored, stats.members_scored);
                    let mut best_from_all = all.candidates;
                    best_from_all.sort_unstable_by(|a, b| b.cmp(a));
                    best_from_all.truncate(k);
                    assert_eq!(
                        best_from_all
                            .iter()
                            .map(|c| (c.sim, ClusterId(c.node.1)))
                            .collect::<Vec<_>>(),
                        segment.iter().map(|c| (c.sim, c.node)).collect::<Vec<_>>()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn patience_follows_global_order_and_counts_work_per_segment() -> crate::Result<()> {
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.0_f32])).unwrap();
        for far_base in [0.5, 100.0] {
            let far = open_line_router(far_base, 80, false)?;
            let near = open_line_router(0.0, 80, false)?;
            let routers = [Some(&far), Some(&near)];
            let mut totals = vec![StageTotals::default(); routers.len()];
            let ranked = GlobalRanking::from(
                &routers,
                &[],
                &query,
                RoutingParams {
                    k: 1,
                    ..Default::default()
                },
                &mut totals,
                &mut |_| {},
            );
            assert_eq!(ranked.clusters.len(), 6);
            assert_eq!(ranked.clusters[0].0, 1);
            assert_eq!(ranked.clusters[0].1.node, 0);
            let expected_lists = if far_base < 1.0 { [3, 3] } else { [0, 6] };
            for (metrics, expected) in ranked.metrics.iter().zip(expected_lists) {
                assert!(matches!(metrics, RouterMetrics::Stacked {
                    candidate_count, lists_scanned, members_scored,
                } if *candidate_count == expected && *lists_scanned == expected && *members_scored == 80 + expected));
            }
            let exhaustive = GlobalRanking::from(
                &routers,
                &[],
                &query,
                RoutingParams::default(),
                &mut totals,
                &mut |_| {},
            );
            assert_eq!(exhaustive.clusters.len(), 160);
        }
        Ok(())
    }

    type Reads = Arc<std::sync::Mutex<Vec<std::ops::Range<usize>>>>;

    #[derive(Debug)]
    struct TrackedCentroids {
        bytes: common::OwnedBytes,
        reads: Reads,
    }

    impl common::HasLen for TrackedCentroids {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl crate::directory::FileHandle for TrackedCentroids {
        fn read_bytes(&self, range: std::ops::Range<usize>) -> std::io::Result<common::OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(self.bytes.slice(range))
        }
    }

    fn open_represented_router(
        metric: Metric,
        nested: bool,
        seed: Option<u64>,
    ) -> crate::Result<(OpenedRouter, Vec<f32>, Reads)> {
        use super::super::source::{EncodingContext, VectorSet, VectorSource};
        use super::super::stacked::StackedRouter;
        use crate::vector::storage::quantized::sign::SignVectors;

        let dim = 8;
        let rows: Vec<f32> = (0..12 * dim)
            .map(|i| ((i * 31 % 257) as f32 * 0.17).sin())
            .collect();
        let centers: Vec<f32> = (0..4)
            .flat_map(|i| {
                let rows = &rows;
                (0..dim)
                    .map(move |d| (0..3).map(|r| rows[(i * 3 + r) * dim + d]).sum::<f32>() / 3.0)
            })
            .collect();
        let parent = nested.then(|| {
            Box::new(StackedIvf {
                config: Default::default(),
                parent: None,
                offsets: vec![(0, 2), (2, 4)],
                centroids: InMemoryStore::new(centers[..2 * dim].to_vec(), dim),
                vectors: InMemoryStore::new(centers.clone(), dim),
            })
        });
        let index = StackedIvf {
            config: Default::default(),
            parent,
            offsets: vec![(0, 3), (3, 6), (6, 9), (9, 12)],
            centroids: InMemoryStore::new(centers, dim),
            vectors: InMemoryStore::new(rows.clone(), dim),
        };
        let options = VectorOptions::new(dim, metric);
        let router = if let Some(seed) = seed {
            let context = Arc::new(EncodingContext::new(dim, seed));
            let (config, topology, banks) = index.into_vector_sets();
            let banks = banks
                .into_iter()
                .enumerate()
                .map(|(bank, raw)| {
                    Ok(VectorSet::Encoded {
                        codes: Arc::new(SignVectors::encode(&raw, dim, &context.rotation)?),
                        context: context.clone(),
                        raw: (bank == 0).then(|| Arc::new(raw)),
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            StackedRouter::new(
                StackedIvf::from_vector_sets(topology, banks, config, |s| s.num_vectors(dim))?,
                &options,
            )?
        } else {
            StackedRouter::from_index(index, &options, false)?
        };
        let mut payload = Vec::new();
        Router::Stacked(router).serialize(&mut payload)?;
        let reads = Reads::default();
        let raw = FileSlice::new(Arc::new(TrackedCentroids {
            bytes: common::OwnedBytes::new(
                rows.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            reads: reads.clone(),
        }));
        let router = RouterKind::open(FileSlice::from(payload), raw, &options)?;
        assert!(
            reads.lock().unwrap().is_empty(),
            "opening must leave raw centroid rows lazy"
        );
        Ok((router, rows, reads))
    }

    #[test]
    fn quantized_segment_and_global_routing_share_selection_and_work() -> crate::Result<()> {
        use super::super::source::ROUTER_SEED;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for nested in [false, true] {
                let (router, _, _) = open_represented_router(metric, nested, Some(ROUTER_SEED))?;
                let query = VectorQuery::new(metric, Arc::new(vec![0.3; 8])).unwrap();
                #[cfg(debug_assertions)]
                quant::rotation::debug_reset_apply_count();
                for k in [1, 3, usize::MAX] {
                    for factor in [1.0, 1.25, 2.0] {
                        let params = RoutingParams {
                            k,
                            rerank_factor: factor,
                            ..Default::default()
                        };
                        let segment = router.rank(&query, params);
                        let metrics = serde_json::to_value(segment.metrics()).unwrap();
                        let mut expected: Vec<_> = segment.collect();
                        expected
                            .sort_unstable_by(|a, b| b.sim.cmp(&a.sim).then(a.node.cmp(&b.node)));
                        let mut totals = vec![StageTotals::default(); 2];
                        let global = GlobalRanking::from(
                            &[None, Some(&router)],
                            &[],
                            &query,
                            params,
                            &mut totals,
                            &mut |_| {},
                        );
                        assert_eq!(
                            global.clusters,
                            expected.into_iter().map(|c| (1, c)).collect::<Vec<_>>()
                        );
                        assert_eq!(serde_json::to_value(global.metrics[1]).unwrap(), metrics);
                    }
                }
                #[cfg(debug_assertions)]
                assert_eq!(
                    quant::rotation::debug_apply_count(),
                    1,
                    "all segment and global rankings must reuse the shared query preparation"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn mixed_representations_partitions_and_seeds_return_exact_centroid_scores() -> crate::Result<()>
    {
        use super::super::source::ROUTER_SEED;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let a = open_represented_router(metric, true, Some(ROUTER_SEED))?;
            let b = open_represented_router(metric, false, None)?;
            let c = open_represented_router(metric, false, Some(ROUTER_SEED + 1))?;
            let query = VectorQuery::new(metric, Arc::new(vec![0.3; 8])).unwrap();
            let mut centroids = IvfCentroids::F32(IvfMatrix {
                values: vec![0.5; 8],
                rows: 1,
                dims: 8,
            });
            let options = VectorOptions::new(8, metric);
            let exact = RouterKind::Exact.build(&options, &mut centroids)?;
            let mut payload = Vec::new();
            exact.serialize(&mut payload)?;
            let exact = RouterKind::open(
                FileSlice::from(payload),
                FileSlice::from(
                    [0.5f32; 8]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                &options,
            )?;
            let routers = [Some(&a.0), Some(&b.0), Some(&c.0), Some(&exact), None];
            let partitions = [
                Some(&[0.1f32; 8][..]),
                None,
                Some(&[0.4f32; 8][..]),
                None,
                None,
            ];
            let mut totals = vec![StageTotals::default(); routers.len()];
            let params = RoutingParams {
                k: usize::MAX,
                aps: false,
                min_beam_fraction: 1.0,
                ..Default::default()
            };
            #[cfg(debug_assertions)]
            quant::rotation::debug_reset_apply_count();
            let ranking = GlobalRanking::from(
                &routers,
                &partitions,
                &query,
                params,
                &mut totals,
                &mut |_| {},
            );
            #[cfg(debug_assertions)]
            assert_eq!(
                quant::rotation::debug_apply_count(),
                2,
                "prepare once for each persisted rotation seed"
            );
            assert_eq!(ranking.clusters.len(), 37);
            for (segment, candidate) in ranking.clusters {
                let values = match segment {
                    0 => &a.1,
                    1 => &b.1,
                    2 => &c.1,
                    3 => {
                        assert_eq!(candidate.sim, metric.similarity(query.widened(), &[0.5; 8]));
                        continue;
                    }
                    _ => unreachable!(),
                };
                let row = candidate.node as usize;
                assert_eq!(
                    candidate.sim,
                    metric.similarity(query.widened(), &values[row * 8..][..8])
                );
            }
            for (segment, reads) in [(0, &a.2), (2, &c.2)] {
                let bytes: usize = reads.lock().unwrap().iter().map(|r| r.len()).sum();
                assert_eq!(bytes, 12 * 8 * 4);
                assert!(matches!(
                    ranking.metrics[segment],
                    RouterMetrics::QuantizedStacked {
                        exact_scored: 12,
                        approximate_members_scored: 12,
                        candidate_count: 12,
                        ..
                    }
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn global_quantized_shortlist_is_shared_and_only_retained_rows_are_read() -> crate::Result<()> {
        use super::super::source::ROUTER_SEED;
        let a = open_represented_router(Metric::L2, true, Some(ROUTER_SEED))?;
        let b = open_represented_router(Metric::L2, false, Some(ROUTER_SEED))?;
        let query = VectorQuery::new(Metric::L2, Arc::new(vec![0.3; 8])).unwrap();
        let mut totals = vec![StageTotals::default(); 2];
        let ranking = GlobalRanking::from(
            &[Some(&a.0), Some(&b.0)],
            &[],
            &query,
            RoutingParams {
                k: 3,
                rerank_factor: 1.5,
                aps: false,
                min_beam_fraction: 1.0,
                flat: false,
            },
            &mut totals,
            &mut |_| {},
        );
        assert_eq!(ranking.clusters.len(), 3);
        let exact: usize = ranking
            .metrics
            .iter()
            .map(|m| match m {
                RouterMetrics::QuantizedStacked { exact_scored, .. } => *exact_scored,
                _ => unreachable!(),
            })
            .sum();
        assert_eq!(exact, 5, "ceil(1.5 * 3) is shared across segments");
        let bytes: usize = [&a.2, &b.2]
            .iter()
            .map(|reads| reads.lock().unwrap().iter().map(|r| r.len()).sum::<usize>())
            .sum();
        assert_eq!(bytes, exact * 8 * 4);
        for (segment, candidate) in ranking.clusters {
            let values = if segment == 0 { &a.1 } else { &b.1 };
            assert_eq!(
                candidate.sim,
                Metric::L2.similarity(query.widened(), &values[candidate.node as usize * 8..][..8])
            );
        }
        Ok(())
    }

    #[test]
    fn non_finite_routing_norms_bypass_sign_scoring() -> crate::Result<()> {
        use super::super::source::ROUTER_SEED;
        let (router, _, reads) = open_represented_router(Metric::Cosine, true, Some(ROUTER_SEED))?;
        // Cosine admits any finite query; this one's squared norm overflows f32.
        let query = VectorQuery::new(Metric::Cosine, Arc::new(vec![f32::MAX; 8])).unwrap();
        assert!(!query.routing_norm().is_finite());
        let mut totals = vec![StageTotals::default()];
        let ranking = GlobalRanking::from(
            &[Some(&router)],
            &[],
            &query,
            RoutingParams {
                k: 3,
                ..Default::default()
            },
            &mut totals,
            &mut |_| {},
        );
        assert_eq!(ranking.clusters.len(), 3);
        assert!(matches!(
            ranking.metrics[0],
            RouterMetrics::QuantizedStacked {
                approximate_scored: 0,
                exact_scored: 12,
                ..
            }
        ));
        assert_eq!(*reads.lock().unwrap(), vec![0..12 * 8 * 4]);
        Ok(())
    }

    fn open_exact_router(metric: Metric, rows: &[f32], dim: usize) -> crate::Result<OpenedRouter> {
        let options = VectorOptions::new(dim, metric);
        let mut centroids = IvfCentroids::F32(IvfMatrix {
            values: rows.to_vec(),
            rows: rows.len() / dim,
            dims: dim,
        });
        let mut payload = Vec::new();
        RouterKind::Exact
            .build(&options, &mut centroids)?
            .serialize(&mut payload)?;
        let bytes: Vec<_> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        RouterKind::open(FileSlice::from(payload), FileSlice::from(bytes), &options)
    }

    /// Flat routing of a stacked and a quantized stacked segment reads each bottom bank once,
    /// ignores partition centroids, and returns the ranking exact routers over the same centroid
    /// rows return, whatever `k` the caller passes.
    #[test]
    fn flat_ranking_matches_exact_routers_over_the_same_centroids() -> crate::Result<()> {
        use super::super::source::ROUTER_SEED;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for nested in [false, true] {
                let stacked = open_represented_router(metric, nested, None)?;
                let quantized = open_represented_router(metric, nested, Some(ROUTER_SEED))?;
                assert_eq!(stacked.0.kind(), RouterKind::Stacked);
                assert_eq!(quantized.0.kind(), RouterKind::QuantizedStacked);
                let exact = [
                    open_exact_router(metric, &stacked.1, 8)?,
                    open_exact_router(metric, &quantized.1, 8)?,
                ];
                let query = VectorQuery::new(
                    metric,
                    Arc::new((0..8).map(|d| 0.1 * d as f32 - 0.3).collect()),
                );
                let mut totals = vec![StageTotals::default(); 2];
                let expected = GlobalRanking::from(
                    &[Some(&exact[0]), Some(&exact[1])],
                    &[],
                    &query,
                    RoutingParams::default(),
                    &mut totals,
                    &mut |_| {},
                );
                assert_eq!(expected.clusters.len(), 24);
                for k in [1, 5, usize::MAX] {
                    stacked.2.lock().unwrap().clear();
                    quantized.2.lock().unwrap().clear();
                    let mut totals = vec![StageTotals::default(); 2];
                    let mut routed = Vec::new();
                    let flat = GlobalRanking::from(
                        &[Some(&stacked.0), Some(&quantized.0)],
                        &[Some(&[0.9f32; 8][..]), Some(&[-0.9f32; 8][..])],
                        &query,
                        RoutingParams {
                            k,
                            flat: true,
                            ..Default::default()
                        },
                        &mut totals,
                        &mut |segment| routed.push(segment),
                    );
                    assert_eq!(flat.clusters, expected.clusters);
                    assert!(flat
                        .metrics
                        .iter()
                        .all(|m| matches!(m, RouterMetrics::Exact { visited_count: 12 })));
                    assert_eq!(routed, [0, 1]);
                    for reads in [&stacked.2, &quantized.2] {
                        assert_eq!(*reads.lock().unwrap(), vec![0..12 * 8 * 4]);
                    }
                }
            }
        }
        Ok(())
    }
}

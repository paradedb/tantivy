use super::candidate::Candidate;
use super::stacked::search::{AllScored, StackedRouting, TopK};
use super::{OpenedRouter, Router, RouterMetrics};
#[cfg(test)]
use crate::schema::Metric;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::{VectorElement, VectorStage};

/// Clusters to probe across all segments, ordered by centroid similarity to the query.
/// Each cluster carries its segment index, and metrics track the routing work for each segment.
pub(crate) struct GlobalRanking {
    pub(crate) clusters: Vec<(usize, Candidate)>,
    pub(crate) metrics: Vec<RouterMetrics>,
}

impl GlobalRanking {
    #[cfg(test)]
    pub(crate) fn from(
        routers: &[Option<&OpenedRouter>],
        partitions: &[Option<&[f32]>],
        query: &[f32],
        metric: Metric,
        k: usize,
        totals: &mut [StageTotals],
        on_segment: &mut impl FnMut(usize),
    ) -> Self {
        Self::from_prepared(
            routers,
            partitions,
            &VectorQuery::new(metric, std::sync::Arc::new(query.to_vec())),
            k,
            false,
            totals,
            on_segment,
        )
    }

    pub(crate) fn from_prepared<T: VectorElement>(
        routers: &[Option<&OpenedRouter>],
        partitions: &[Option<&[f32]>],
        query: &VectorQuery<T>,
        k: usize,
        bounded: bool,
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
                Some(Router::Stacked(index)) => Some(index),
                Some(Router::Exact(index)) => {
                    let (ranked, measured) = measure(|| {
                        let _stage = enter_vector_stage(VectorStage::Routing);
                        index.rank(query).collect::<Vec<_>>()
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
        let routing = StackedRouting::Global {
            indexes,
            partitions,
            totals,
            on_segment,
        };
        let ranking = if bounded {
            routing.search_prepared(query, k.max(1), TopK)
        } else {
            routing.search_prepared(query, k.max(1), AllScored::default())
        };
        for (stats, metrics) in ranking.stats.iter().zip(&mut metrics) {
            if let RouterMetrics::Stacked {
                lists_scanned,
                members_scored,
                ..
            } = metrics
            {
                *lists_scanned = stats.lists_scanned;
                *members_scored = stats.members_scored;
            }
        }
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
        if bounded {
            leaves.truncate(k.max(1));
        }
        for (segment, _) in &leaves {
            if let RouterMetrics::Stacked {
                candidate_count, ..
            } = &mut metrics[*segment]
            {
                *candidate_count += 1;
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
    use super::*;
    use crate::directory::FileSlice;
    use crate::schema::VectorOptions;
    use crate::vector::clustering::{IvfCentroids, IvfMatrix};
    use crate::vector::router::stacked::levels::{
        ClusterId, InMemoryStore, StackedIvf, StackedIvfConfig,
    };
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
        let router = Router::Stacked(StackedIvf {
            config: StackedIvfConfig::default(),
            parent,
            offsets: (0..count).map(|i| (i, i + 1)).collect(),
            centroids: InMemoryStore::new(rows.clone(), 1),
            vectors: InMemoryStore::new(rows.clone(), 1),
        });
        let mut payload = Vec::new();
        router.serialize(&mut payload)?;
        let bytes: Vec<_> = rows.into_iter().flat_map(f32::to_le_bytes).collect();
        RouterKind::open(FileSlice::from(payload), FileSlice::from(bytes), &options)
    }

    #[test]
    fn partition_layer_skips_distant_stacks_but_admits_unassigned_segments() -> crate::Result<()> {
        let indexes = (0..8)
            .map(|i| open_line_router(i as f32 * 1000.0, 80, false))
            .collect::<crate::Result<Vec<_>>>()?;
        let routers: Vec<_> = indexes.iter().map(Some).collect();
        let centroids: Vec<_> = (0..8).map(|i| [i as f32 * 1000.0]).collect();
        let mut partitions: Vec<_> = centroids.iter().map(|c| Some(c.as_slice())).collect();
        let mut totals = vec![StageTotals::default(); routers.len()];
        let plain = GlobalRanking::from(
            &routers,
            &[],
            &[0.0],
            Metric::L2,
            1,
            &mut totals,
            &mut |_| {},
        );
        let partitioned = GlobalRanking::from(
            &routers,
            &partitions,
            &[0.0],
            Metric::L2,
            1,
            &mut totals,
            &mut |_| {},
        );
        assert_eq!(partitioned.clusters[0].0, 0);
        let scored = |ranking: &GlobalRanking| {
            ranking
                .metrics
                .iter()
                .map(|m| match m {
                    RouterMetrics::Stacked { members_scored, .. } => *members_scored,
                    _ => 0,
                })
                .sum::<usize>()
        };
        assert!(scored(&partitioned) < scored(&plain) * 3 / 4);
        assert!(partitioned.metrics.iter().skip(4).all(|m| matches!(
            m,
            RouterMetrics::Stacked {
                members_scored: 1,
                lists_scanned: 0,
                ..
            }
        )));
        partitions[7] = None;
        let mixed = GlobalRanking::from(
            &routers,
            &partitions,
            &[7000.0],
            Metric::L2,
            1,
            &mut totals,
            &mut |_| {},
        );
        assert_eq!((mixed.clusters[0].0, mixed.clusters[0].1.node), (7, 0));
        let exhaustive = GlobalRanking::from(
            &routers,
            &partitions,
            &[0.0],
            Metric::L2,
            usize::MAX,
            &mut totals,
            &mut |_| {},
        );
        assert_eq!(exhaustive.clusters.len(), 640);
        Ok(())
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
                        &query,
                        Metric::L2,
                        1,
                        &mut totals,
                        &mut |_| {},
                    );
                    assert_eq!(ranked.clusters[0].0, (query[0] / 100.0) as usize);
                    let full = GlobalRanking::from(
                        &routers,
                        &partitions,
                        &query,
                        Metric::L2,
                        usize::MAX,
                        &mut totals,
                        &mut |_| {},
                    );
                    let plain = GlobalRanking::from(
                        &routers,
                        &[],
                        &query,
                        Metric::L2,
                        usize::MAX,
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
                &[0.0],
                Metric::L2,
                0,
                &mut totals,
                &mut |_| {},
            );
            let exhaustive = GlobalRanking::from(
                &routers,
                &[],
                &[0.0],
                Metric::L2,
                usize::MAX,
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
            let query = VectorQuery::new(Metric::L2, std::sync::Arc::new(vec![0.0]));
            for k in [1, 2, 4, 16] {
                let all = GlobalRanking::from_prepared(
                    &routers,
                    &[],
                    &query,
                    k,
                    false,
                    &mut totals,
                    &mut |_| {},
                );
                let bounded = GlobalRanking::from_prepared(
                    &routers,
                    &[],
                    &query,
                    k,
                    true,
                    &mut totals,
                    &mut |_| {},
                );
                assert_eq!(bounded.clusters, all.clusters[..k.min(all.clusters.len())]);
                for (limited, all) in bounded.metrics.iter().zip(&all.metrics) {
                    match (limited, all) {
                        (
                            RouterMetrics::Stacked {
                                lists_scanned: limited_lists,
                                members_scored: limited_members,
                                ..
                            },
                            RouterMetrics::Stacked {
                                lists_scanned,
                                members_scored,
                                ..
                            },
                        ) => {
                            assert_eq!(limited_lists, lists_scanned);
                            assert_eq!(limited_members, members_scored);
                        }
                        (
                            RouterMetrics::Exact { visited_count: a },
                            RouterMetrics::Exact { visited_count: b },
                        ) => assert_eq!(a, b),
                        _ => unreachable!(),
                    }
                }
            }
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
                for k in [0, 1, 2, 4, 16] {
                    let (segment, stats) = index.search(&query, k, Metric::L2);
                    let mut totals = vec![StageTotals::default(); 2];
                    let mut visited = Vec::new();
                    let global = StackedRouting::Global {
                        indexes: vec![None, Some(&index)],
                        partitions: &[],
                        totals: &mut totals,
                        on_segment: &mut |segment| visited.push(segment),
                    }
                    .search(&query, k, Metric::L2, TopK);
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
                        indexes: vec![None, Some(&index)],
                        partitions: &[],
                        totals: &mut totals,
                        on_segment: &mut |_| {},
                    }
                    .search(&query, k, Metric::L2, AllScored::default());
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
        for far_base in [0.5, 100.0] {
            let far = open_line_router(far_base, 80, false)?;
            let near = open_line_router(0.0, 80, false)?;
            let routers = [Some(&far), Some(&near)];
            let mut totals = vec![StageTotals::default(); routers.len()];
            let ranked = GlobalRanking::from(
                &routers,
                &[],
                &[0.0],
                Metric::L2,
                1,
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
                &[0.0],
                Metric::L2,
                usize::MAX,
                &mut totals,
                &mut |_| {},
            );
            assert_eq!(exhaustive.clusters.len(), 160);
        }
        Ok(())
    }
}

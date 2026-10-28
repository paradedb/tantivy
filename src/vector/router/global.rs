use super::adaptive::AdaptiveProbe;
use super::arena::VectorArena;
use super::candidate::Candidate;
use super::stacked::levels::{lists_to_open, LazyStore, StackedIvf};
use super::{OpenedRouter, Router, RouterMetrics};
use crate::schema::Metric;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::VectorStage;

/// Clusters to probe across all segments, ordered by centroid similarity to the query.
/// Each cluster carries its segment index, and metrics track the routing work for each segment.
pub(crate) struct GlobalRanking {
    pub(crate) clusters: Vec<(usize, Candidate)>,
    pub(crate) metrics: Vec<RouterMetrics>,
}

impl GlobalRanking {
    pub(crate) fn from(
        routers: &[Option<&OpenedRouter>],
        query: &[f32],
        metric: Metric,
        k: usize,
        totals: &mut [StageTotals],
        on_segment: &mut impl FnMut(usize),
    ) -> Self {
        let mut levels: Vec<Vec<&StackedIvf<LazyStore>>> = Vec::with_capacity(routers.len());
        let mut metrics = Vec::with_capacity(routers.len());
        let mut leaves = Vec::new();
        for (segment, router) in routers.iter().enumerate() {
            let mut stack = Vec::new();
            let mut stats = RouterMetrics::Stacked {
                candidate_count: 0,
                lists_scanned: 0,
                members_scored: 0,
            };
            match router {
                Some(Router::Stacked(router)) => {
                    let mut current = Some(router);
                    while let Some(level) = current {
                        stack.push(level);
                        current = level.parent.as_deref();
                    }
                }
                Some(Router::Exact(router)) => {
                    let (ranked, measured) = measure(|| {
                        let _stage = enter_vector_stage(VectorStage::Routing);
                        router.rank(query).collect::<Vec<_>>()
                    });
                    totals[segment].append(&measured);
                    on_segment(segment);
                    stats = RouterMetrics::Exact {
                        visited_count: ranked.len(),
                    };
                    leaves.extend(ranked.into_iter().map(|c| (segment, c)));
                }
                None => {}
            }
            levels.push(stack);
            metrics.push(stats);
        }
        let depth = levels.iter().map(Vec::len).max().unwrap_or(0);
        let mut requested = k;
        let open: Vec<_> = (0..depth)
            .map(|level| {
                let indexes: Vec<_> = levels.iter().filter_map(|stack| stack.get(level)).collect();
                let lists = indexes.iter().map(|index| index.nlist()).sum();
                let members = indexes
                    .iter()
                    .flat_map(|index| &index.offsets)
                    .map(|&(start, end)| (end - start) as usize)
                    .sum();
                requested = lists_to_open(lists, members, requested);
                requested
            })
            .collect();
        let mut candidates: Vec<(usize, Candidate)> = Vec::new();
        for level in (0..depth).rev() {
            for (segment, stack) in levels.iter().enumerate() {
                if stack.len() != level + 1 {
                    continue;
                }
                let index = stack[level];
                let (_, measured) = measure(|| {
                    let _stage = enter_vector_stage(VectorStage::Routing);
                    candidates.extend((0..index.nlist()).map(|node| {
                        (
                            segment,
                            Candidate {
                                sim: index.centroids.similarity(
                                    metric,
                                    query.len(),
                                    node as u32,
                                    query,
                                ),
                                node: node as u32,
                            },
                        )
                    }));
                });
                totals[segment].append(&measured);
                on_segment(segment);
                if let RouterMetrics::Stacked { members_scored, .. } = &mut metrics[segment] {
                    *members_scored += index.nlist();
                }
            }
            candidates.sort_unstable_by(|a, b| {
                b.1.sim
                    .cmp(&a.1.sim)
                    .then(a.0.cmp(&b.0))
                    .then(a.1.node.cmp(&b.1.node))
            });
            candidates.truncate(open[level]);
            let mut next = Vec::new();
            let requested = if level == 0 { k } else { open[level - 1] };
            let mut probe = AdaptiveProbe::new(requested);
            for (segment, candidate) in candidates {
                let index = levels[segment][level];
                let (scanned, measured) = measure(|| {
                    let _stage = enter_vector_stage(VectorStage::Routing);
                    let (start, end) = index.offsets[candidate.node as usize];
                    for node in start..end {
                        let candidate = Candidate {
                            sim: index
                                .vectors
                                .similarity(metric, query.len(), node as u32, query),
                            node: node as u32,
                        };
                        probe.push(Candidate {
                            sim: candidate.sim,
                            node: (segment, candidate.node),
                        });
                        next.push((segment, candidate));
                    }
                    (end - start) as usize
                });
                totals[segment].append(&measured);
                on_segment(segment);
                if let RouterMetrics::Stacked {
                    lists_scanned,
                    members_scored,
                    ..
                } = &mut metrics[segment]
                {
                    *lists_scanned += 1;
                    *members_scored += scanned;
                }
                if probe.should_stop(scanned) {
                    break;
                }
            }
            candidates = next;
        }
        leaves.extend(candidates);
        leaves.sort_unstable_by(|a, b| {
            b.1.sim
                .cmp(&a.1.sim)
                .then(a.0.cmp(&b.0))
                .then(a.1.node.cmp(&b.1.node))
        });
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
    use crate::vector::router::stacked::levels::{InMemoryStore, StackedIvfConfig};
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
            let ranked =
                GlobalRanking::from(&routers, &[0.0], Metric::L2, 0, &mut totals, &mut |_| {});
            let exhaustive = GlobalRanking::from(
                &routers,
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
            let ranked =
                GlobalRanking::from(&routers, &[0.0], Metric::L2, 1, &mut totals, &mut |_| {});
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

//! One-bit routing sources. The level walk is shared with full-precision IVF;
//! only its source scores estimates. Exact scoring is composed once at output.

use std::io::{self, Write};
use std::sync::Arc;

use common::{BinarySerializable, HasLen};
use quant::rotation::Rotation;

use super::arena::VectorArena;
use super::candidate::Candidate;
use super::search::{self, Hierarchy, LevelSource, ScoreBuffer, TopCandidates};
use super::sign::{invalid, SignQuery, SignVectors};
use super::stacked::levels::{ClusterId, InMemoryStore, LazyStore, StackedIvf, StackedSearchStats};
use super::{RouterMetrics, RoutingParams};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::IvfCentroids;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

const VERSION: u32 = 1;
const ROTATION_SEED: u64 = 0x726f_7574_6572_0001;

struct QuantizedLevel {
    parent: Option<Box<Self>>,
    offsets: Vec<(u64, u64)>,
    centroids: Arc<SignVectors>,
    members: Arc<SignVectors>,
}

impl LevelSource<SignQuery> for QuantizedLevel {
    type ClusterId = ClusterId;
    type MemberId = ClusterId;
    fn list_count(&self) -> usize {
        self.offsets.len()
    }
    fn member_count(&self) -> usize {
        self.members.len()
    }
    fn score_centroids(&self, query: &SignQuery, out: &mut ScoreBuffer<ClusterId>) {
        self.centroids
            .score(0..self.centroids.len(), query, out, ClusterId::from);
    }
    fn score_members(&self, list: ClusterId, query: &SignQuery, out: &mut ScoreBuffer<ClusterId>) {
        let (start, end) = self.offsets[usize::from(list)];
        self.members
            .score(start as usize..end as usize, query, out, ClusterId::from);
    }
}

impl Hierarchy<SignQuery> for QuantizedLevel {
    type Parent = Self;
    fn parent(&self) -> Option<&Self> {
        self.parent.as_deref()
    }
}

pub(crate) struct QuantizedRouter<S> {
    level: QuantizedLevel,
    raw: S,
    dim: usize,
    metric: Metric,
    seed: u64,
    rotation: Rotation,
}

impl QuantizedRouter<InMemoryStore> {
    pub(crate) fn build(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Self> {
        if options.dim() == 0 || options.dim() > u32::MAX as usize {
            return Err(invalid("quantized routing requires a positive dimension").into());
        }
        let index = super::stacked::build(options, centroids)?;
        Self::from_index(index, options)
    }

    fn from_index(
        mut index: StackedIvf<InMemoryStore>,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        let rotation = Rotation::new(options.dim(), ROTATION_SEED);
        let members = Arc::new(SignVectors::encode(
            &index.vectors,
            options.dim(),
            &rotation,
        )?);
        let raw = std::mem::replace(
            &mut index.vectors,
            InMemoryStore::new(Vec::new(), options.dim()),
        );
        fn encode(
            index: StackedIvf<InMemoryStore>,
            members: Arc<SignVectors>,
            dim: usize,
            rotation: &Rotation,
        ) -> io::Result<QuantizedLevel> {
            let centroids = Arc::new(SignVectors::encode(&index.centroids, dim, rotation)?);
            let parent = index
                .parent
                .map(|parent| encode(*parent, centroids.clone(), dim, rotation).map(Box::new))
                .transpose()?;
            Ok(QuantizedLevel {
                parent,
                offsets: index.offsets,
                centroids,
                members,
            })
        }
        Ok(Self {
            level: encode(index, members, options.dim(), &rotation)?,
            raw,
            dim: options.dim(),
            metric: options.metric(),
            seed: ROTATION_SEED,
            rotation,
        })
    }

    pub(crate) fn serialize(&self, out: &mut (impl Write + ?Sized)) -> io::Result<()> {
        VERSION.serialize(out)?;
        (self.dim as u32).serialize(out)?;
        metric_code(self.metric).serialize(out)?;
        self.seed.serialize(out)?;
        let mut levels = Vec::new();
        let mut next = Some(&self.level);
        while let Some(level) = next {
            levels.push(level);
            next = level.parent.as_deref();
        }
        (levels.len() as u32).serialize(out)?;
        for level in &levels {
            (level.offsets.len() as u32).serialize(out)?;
            for &(start, end) in &level.offsets {
                start.serialize(out)?;
                end.serialize(out)?;
            }
        }
        self.level.members.serialize(out)?;
        for level in levels {
            level.centroids.serialize(out)?;
        }
        Ok(())
    }
}

fn metric_code(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::Dot => 1,
        Metric::Cosine => 2,
    }
}

impl QuantizedRouter<LazyStore> {
    pub(crate) fn open(
        payload: FileSlice,
        raw: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        let bytes = payload.read_bytes()?;
        let mut input = bytes.as_slice();
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
        if count == 0 || count > input.len() / 4 {
            return Err(invalid("invalid quantized router level count").into());
        }
        let mut topology = Vec::with_capacity(count);
        for _ in 0..count {
            let lists = u32::deserialize(&mut input)? as usize;
            if lists > input.len() / 16 {
                return Err(invalid("truncated quantized router topology").into());
            }
            let offsets = (0..lists)
                .map(|_| Ok((u64::deserialize(&mut input)?, u64::deserialize(&mut input)?)))
                .collect::<io::Result<Vec<_>>>()?;
            topology.push(offsets);
        }
        let mut sets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            sets.push(Arc::new(SignVectors::deserialize(&mut input, dim)?));
        }
        if !input.is_empty() {
            return Err(invalid("trailing quantized router bytes").into());
        }
        let raw_len = sets[0]
            .len()
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("quantized router raw size overflow"))?;
        if raw.len() != raw_len {
            return Err(
                invalid("quantized router member count differs from raw centroid rows").into(),
            );
        }
        let mut parent = None;
        for (level, offsets) in topology.into_iter().enumerate().rev() {
            if offsets.len() != sets[level + 1].len()
                || !super::stacked::levels::ranges_cover_members(&offsets, sets[level].len())
            {
                return Err(
                    invalid("quantized router membership does not cover its vector set").into(),
                );
            }
            parent = Some(Box::new(QuantizedLevel {
                parent,
                offsets,
                centroids: sets[level + 1].clone(),
                members: sets[level].clone(),
            }));
        }
        Ok(Self {
            level: *parent.expect("validated nonempty hierarchy"),
            raw: LazyStore::new(raw, dim),
            dim,
            metric: options.metric(),
            seed,
            rotation: Rotation::new(dim, seed),
        })
    }
}

impl<S: VectorArena<Elem = f32>> QuantizedRouter<S> {
    pub(crate) fn level_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut level = Some(&self.level);
        while let Some(current) = level {
            sizes.push(current.offsets.len());
            level = current.parent.as_deref();
        }
        sizes
    }

    pub(crate) fn rank<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        params: RoutingParams,
    ) -> Ranking {
        assert_eq!(query.widened().len(), self.dim);
        debug_assert_eq!(query.metric(), self.metric);
        let k = params.k.min(self.level.members.len());
        let capacity = k
            .saturating_mul(params.rerank_factor.max(1))
            .min(self.level.members.len());
        let (candidates, stats) = if k == 0 {
            (Vec::new(), StackedSearchStats::default())
        } else if !query.routing_norm().is_finite()
            || query.widened().iter().any(|x| !x.is_finite())
        {
            // The sign-query encoder requires finite inputs. Preserve the raw
            // scorer's total-order behavior for exceptional queries instead of
            // passing infinities/NaNs to bitplane preparation.
            (
                (0..self.level.members.len())
                    .map(|row| Candidate {
                        node: ClusterId(row as u32),
                        sim: crate::vector::distance::Similarity::new(0.0),
                    })
                    .collect(),
                StackedSearchStats::default(),
            )
        } else {
            let prepared = SignQuery::new(query, &self.rotation);
            search::search(&self.level, &prepared, k, capacity, params.aps)
        };
        let exact_scored = candidates.len();
        let ranked = rerank_exact(&self.raw, query, &candidates, k);
        let candidate_count = ranked.len();
        Ranking {
            ranked: ranked.into_iter(),
            stats,
            exact_scored,
            candidate_count,
        }
    }
}

/// Exact reads are grouped into contiguous selected runs. The heap stores
/// canonical identities, so changing read order does not change tie breaking.
pub(crate) fn rerank_exact<S: VectorArena<Elem = f32>, T: VectorElement>(
    source: &S,
    query: &VectorQuery<T>,
    candidates: &[Candidate<ClusterId>],
    k: usize,
) -> Vec<Candidate<ClusterId>> {
    let mut rows: Vec<_> = candidates.iter().map(|c| c.node).collect();
    rows.sort_unstable();
    let mut result = TopCandidates::new(k);
    let mut scores = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && rows[end].0 == rows[end - 1].0 + 1 {
            end += 1;
        }
        let first = rows[start].0 as usize;
        source.batch_similarity(
            query.widened(),
            query.metric(),
            query.routing_norm(),
            first..first + end - start,
            &mut scores,
        );
        for (&node, &sim) in rows[start..end].iter().zip(&scores) {
            result.offer(Candidate { node, sim });
        }
        start = end;
    }
    result.into_sorted()
}

pub(crate) struct Ranking {
    ranked: std::vec::IntoIter<Candidate<ClusterId>>,
    stats: StackedSearchStats,
    exact_scored: usize,
    candidate_count: usize,
}

impl Ranking {
    pub(crate) fn metrics(&self) -> RouterMetrics {
        RouterMetrics::QuantizedStacked {
            candidate_count: self.candidate_count,
            lists_scanned: self.stats.lists_scanned,
            approximate_scored: self.stats.members_scored,
            exact_scored: self.exact_scored,
        }
    }
}

impl Iterator for Ranking {
    type Item = Candidate;
    fn next(&mut self) -> Option<Self::Item> {
        self.ranked.next().map(|c| Candidate {
            sim: c.sim,
            node: c.node.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Mutex;

    use super::*;
    use crate::directory::FileHandle;
    use crate::vector::clustering::IvfMatrix;

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
    fn persisted_router_reads_only_rerank_rows_and_returns_exact_scores() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let dim = 65;
            let options = VectorOptions::new(dim, metric);
            let mut centroids = vectors(dim, 96);
            let built = QuantizedRouter::build(&options, &mut centroids)?;
            let mut payload = Vec::new();
            built.serialize(&mut payload)?;
            let IvfCentroids::F32(matrix) = centroids;
            let reads = Arc::new(Mutex::new(Vec::new()));
            let raw = FileSlice::new(Arc::new(TrackedRaw {
                bytes: matrix.values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                reads: reads.clone(),
            }));
            let opened = QuantizedRouter::open(FileSlice::from(payload.clone()), raw, &options)?;
            assert!(
                reads.lock().unwrap().is_empty(),
                "opening must not fetch raw rows"
            );
            let query = VectorQuery::new(metric, Arc::new(vec![0.15; dim]));
            for factor in [1, 2, 4, usize::MAX] {
                reads.lock().unwrap().clear();
                let params = RoutingParams {
                    k: 3,
                    rerank_factor: factor,
                    aps: false,
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
                    ..
                } = metrics
                else {
                    unreachable!()
                };
                assert_eq!(candidate_count, 3);
                assert!(
                    exact_scored >= 3
                        && exact_scored <= 3usize.saturating_mul(factor).min(matrix.rows)
                );
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
                assert!(QuantizedRouter::open(
                    FileSlice::from(payload[..end].to_vec()),
                    FileSlice::from(raw_bytes.clone()),
                    &options
                )
                .is_err());
            }
            let mut trailing = payload;
            trailing.push(0);
            assert!(QuantizedRouter::open(
                FileSlice::from(trailing.clone()),
                FileSlice::from(raw_bytes.clone()),
                &options
            )
            .is_err());
            let mut wrong_version = trailing[..trailing.len() - 1].to_vec();
            wrong_version[..4].copy_from_slice(&2u32.to_le_bytes());
            assert!(QuantizedRouter::open(
                FileSlice::from(wrong_version),
                FileSlice::from(raw_bytes.clone()),
                &options
            )
            .is_err());
            let mut bad_range = trailing[..trailing.len() - 1].to_vec();
            // Header (21), list count (4), first range's start (8), then end.
            bad_range[33..41].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(QuantizedRouter::open(
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
        let rotation = Rotation::new(dim, ROTATION_SEED);
        let raw: Vec<_> = (0..16).flat_map(|row| vec![row as f32; dim]).collect();
        let centroids: Vec<_> = [1.5, 5.5, 9.5, 13.5]
            .into_iter()
            .flat_map(|x| vec![x; dim])
            .collect();
        let roots: Vec<_> = [3.5, 11.5].into_iter().flat_map(|x| vec![x; dim]).collect();
        let members = Arc::new(SignVectors::encode(&raw, dim, &rotation)?);
        let centroids = Arc::new(SignVectors::encode(&centroids, dim, &rotation)?);
        let parent = QuantizedLevel {
            parent: None,
            offsets: vec![(0, 2), (2, 4)],
            members: centroids.clone(),
            centroids: Arc::new(SignVectors::encode(&roots, dim, &rotation)?),
        };
        let built = QuantizedRouter {
            level: QuantizedLevel {
                parent: Some(Box::new(parent)),
                offsets: vec![(0, 4), (4, 8), (8, 12), (12, 16)],
                centroids,
                members,
            },
            raw: InMemoryStore::new(raw.clone(), dim),
            dim,
            metric: Metric::L2,
            seed: ROTATION_SEED,
            rotation,
        };
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        // Three matrices (16, 4, 2 rows), with no duplicated parent members or raw rows.
        assert_eq!(
            payload.len(),
            21 + (4 + 4 * 16) + (4 + 2 * 16) + (4 + 16 * 16) + (4 + 4 * 16) + (4 + 2 * 16)
        );
        let options = VectorOptions::new(dim, Metric::L2);
        let opened = QuantizedRouter::open(
            FileSlice::from(payload),
            FileSlice::from(raw.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()),
            &options,
        )?;
        assert_eq!(opened.level_sizes(), [4, 2]);
        assert!(Arc::ptr_eq(
            &opened.level.centroids,
            &opened.level.parent.as_ref().unwrap().members
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
                    max_probe_fraction: 1.0,
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
                }
            }
        }
        Ok(())
    }

    #[test]
    fn empty_router_and_zero_request_return_no_candidates() -> crate::Result<()> {
        let options = VectorOptions::new(64, Metric::L2);
        let mut rows = vectors(64, 0);
        let built = QuantizedRouter::build(&options, &mut rows)?;
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        let opened = QuantizedRouter::open(
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
                let (hits, stats) = search::search(&index, query, k, k, aps);
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
        let quantized = QuantizedRouter::from_index(index, &options)?;
        for aps in [false, true] {
            for factor in [1, 2, 4, 8] {
                let start = Instant::now();
                let (mut found, mut approx, mut exact) = (0, 0, 0);
                for (query, truth) in queries.iter().zip(&truth) {
                    let hits = quantized.rank(
                        query,
                        RoutingParams {
                            k,
                            rerank_factor: factor,
                            aps,
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
}

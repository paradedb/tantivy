//! Vector filter selection and match collection at each search stage.

use std::sync::Arc;

use common::BitSet;
use itertools::Either;

use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::fastfield::AliveBitSet;
use crate::query::{for_each_docset_buffered, AllScorer, ExcludedDocSet, ExclusionSet, Weight};
use crate::schema::Field;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::{BlockAddress, Candidate, VectorSegment, VectorStage};
use crate::{DocId, Result, SegmentReader, TantivyError};

/// Minimum estimated matches for deferring a fast-field filter. Keep the strategy independent
/// of the probe ceiling so increasing recall does not switch access paths mid-sweep.
pub const MIN_POST_FILTER_MATCHES: usize = 50_000;

/// Minimum estimated matches for shortlist filtering.
pub const MIN_LATE_FILTER_MATCHES: usize = 250_000;

const LATE_FILTER_BATCH_HEADROOM: usize = 2;
pub(crate) const MAX_LATE_FILTER_BATCH_SIZE: usize = 64;

/// Query-wide filter evaluation, selected before worker assignment.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VectorFilterStrategy {
    /// Materialize the matches before vector search.
    Pre { uses_fast_fields: bool },
    /// Filter routed clusters before scoring their vectors.
    Post { estimated_selectivity: f64 },
    /// Filter the approximate shortlist before full-precision scoring.
    Late { estimated_selectivity: f64 },
}

impl VectorFilterStrategy {
    /// Chooses when to evaluate the filter, using its estimated match count and segment storage.
    /// Uses Pre for small match sets or filters without fast fields, and Post for larger match
    /// sets when clustered storage is available.
    /// Prefers Late for large match sets when refinement is enabled and storage supports it.
    pub fn plan(
        field: Field,
        weight: &dyn Weight,
        estimated_matches: usize,
        max_scan_levels: usize,
        segments: &[VectorSegment],
    ) -> Result<Self> {
        let uses_fast_fields = weight.uses_fast_fields();
        let pre = Self::Pre { uses_fast_fields };
        if !uses_fast_fields || estimated_matches < MIN_POST_FILTER_MATCHES || segments.is_empty() {
            return Ok(pre);
        }
        let mut supports_late = max_scan_levels > 0;
        let mut has_clusters = false;
        for segment in segments {
            let vectors = segment.reader.vector_index(field)?;
            has_clusters |= vectors.index().is_some();
            supports_late &= vectors.index().is_none() || vectors.has_exact_rows();
        }
        let docs: u64 = segments
            .iter()
            .map(|s| u64::from(s.reader.num_docs()))
            .sum();
        let estimated_selectivity =
            (estimated_matches as f64 / docs.max(1) as f64).clamp(f64::EPSILON, 1.0);
        Ok(
            if estimated_matches >= MIN_LATE_FILTER_MATCHES && supports_late {
                Self::Late {
                    estimated_selectivity,
                }
            } else if has_clusters {
                Self::Post {
                    estimated_selectivity,
                }
            } else {
                pre
            },
        )
    }
}

pub(crate) enum FilterInput<'a> {
    Shortlist {
        candidates: &'a mut Vec<(DocId, BlockAddress)>,
    },
    Segment {
        vectors: &'a VectorIndexReader,
        located_cap: usize,
    },
    Clusters {
        vectors: &'a VectorIndexReader,
        candidates: &'a mut dyn Iterator<Item = Candidate>,
    },
}

/// Matches collected for the current search stage.
#[derive(Clone)]
pub(crate) enum FilterMatches {
    /// The router's ranking, gated by the filter.
    Routed(DocFilter),
    /// The filter's matches, few enough to look up directly: ascending, deleted ones included.
    Located(Vec<DocId>),
    Shortlist {
        checked: usize,
        matched: usize,
    },
}

impl VectorFilterStrategy {
    /// Whether to check the filter on routed clusters before scoring their vectors (Post only).
    pub(crate) fn filters_routed_clusters(self) -> bool {
        matches!(self, Self::Post { .. })
    }

    /// Returns Post's estimated matching fraction for deciding how many clusters to route.
    /// Otherwise, the router uses the prepared filter's matching fraction.
    pub(crate) fn routing_selectivity(self) -> Option<f64> {
        match self {
            Self::Pre { .. } | Self::Late { .. } => None,
            Self::Post {
                estimated_selectivity,
            } => Some(estimated_selectivity),
        }
    }

    /// Collects matches for this strategy at the supplied search stage.
    /// Pre checks the whole segment; Post waits for routed clusters and checks only their docs.
    /// Post also checks the whole segment when it is flat or its estimated matches are few
    /// enough to look up directly.
    /// Late keeps only the matching docs in the supplied approximate shortlist.
    pub(crate) fn collect_matches(
        self,
        weight: &dyn Weight,
        segment: &SegmentReader,
        input: FilterInput<'_>,
    ) -> Result<FilterMatches> {
        match input {
            FilterInput::Shortlist { candidates } => match self {
                Self::Late { .. } => Self::filter_late_shortlist(weight, segment, candidates),
                Self::Pre { .. } | Self::Post { .. } => {
                    unreachable!("only Late filters the shortlist")
                }
            },
            FilterInput::Segment {
                vectors,
                located_cap,
            } => {
                let cap = if vectors.index().is_some() {
                    located_cap
                } else {
                    0
                };
                match self {
                    Self::Late { .. } => {
                        if vectors.index().is_some() && vectors.quantization()?.is_none() {
                            return Err(TantivyError::InvalidArgument(
                                "late filtering requires quantized IVF segments".into(),
                            ));
                        }
                        Ok(FilterMatches::Routed(DocFilter::Late))
                    }
                    Self::Post {
                        estimated_selectivity,
                    } if vectors.index().is_some()
                        && estimated_selectivity * f64::from(segment.max_doc()) > cap as f64 =>
                    {
                        Ok(FilterMatches::Routed(DocFilter::Pending))
                    }
                    Self::Pre { .. } | Self::Post { .. } => {
                        let _filter = enter_vector_stage(VectorStage::NonVectorSearch);
                        Self::collect_pre_matches(weight, segment, cap)
                    }
                }
            }
            FilterInput::Clusters {
                vectors,
                candidates,
            } => {
                let filter = match self {
                    Self::Pre { .. } | Self::Late { .. } => {
                        unreachable!("only Post filters routed clusters")
                    }
                    Self::Post { .. } => {
                        let _filter = enter_vector_stage(VectorStage::NonVectorSearch);
                        Self::filter_post_clusters(vectors, weight, segment, candidates)?
                    }
                };
                Ok(FilterMatches::Routed(filter))
            }
        }
    }

    /// Sizes a Late shortlist batch to leave enough expected matches for `top_k`, with headroom.
    pub(crate) fn shortlist_batch_size(self, top_k: usize) -> usize {
        let Self::Late {
            estimated_selectivity,
        } = self
        else {
            unreachable!("only Late batches shortlist filters")
        };
        (top_k.saturating_mul(LATE_FILTER_BATCH_HEADROOM) as f64
            / estimated_selectivity.clamp(f64::EPSILON, 1.0))
        .ceil() as usize
    }

    fn filter_late_shortlist(
        weight: &dyn Weight,
        segment: &SegmentReader,
        candidates: &mut Vec<(DocId, BlockAddress)>,
    ) -> Result<FilterMatches> {
        let checked = candidates.len();
        {
            let _sort = enter_vector_stage(VectorStage::LateFilterSort);
            candidates.sort_unstable_by_key(|&(doc, _)| doc);
        }
        let mut scorer = {
            let _setup = enter_vector_stage(VectorStage::LateFilterSetup);
            ExcludedDocSet::new(weight.scorer_for_candidates_unpositioned(
                segment,
                1.0,
                checked as u32,
            )?)
        };
        {
            let _predicate = enter_vector_stage(VectorStage::LateFilterPredicate);
            candidates.retain(|&(doc, _)| scorer.contains(doc));
        }
        Ok(FilterMatches::Shortlist {
            checked,
            matched: candidates.len(),
        })
    }

    /// Iterates over matching doc ids in a flat segment, skipping deletions.
    /// Pre and Post use prepared matches; Late checks the filter as each doc is visited.
    pub(crate) fn flat_matches<'a>(
        self,
        weight: &dyn Weight,
        segment: &'a SegmentReader,
        filter: &'a DocFilter,
    ) -> Result<impl Iterator<Item = DocId> + 'a> {
        let docs = match filter {
            DocFilter::All | DocFilter::Late => Either::Left(0..segment.max_doc()),
            DocFilter::Docs(docs) => {
                Either::Right((0..docs.max_value().div_ceil(64)).flat_map(move |bucket| {
                    docs.tinyset(bucket)
                        .into_iter()
                        .map(move |bit| bucket * 64 + bit)
                }))
            }
            DocFilter::Pending => {
                unreachable!("flat filters are prepared before routing")
            }
        };
        let mut scorer = if matches!(self, Self::Late { .. }) && matches!(filter, DocFilter::Late) {
            let _setup = enter_vector_stage(VectorStage::LateFilterSetup);
            Some(ExcludedDocSet::new(
                weight.scorer_for_candidates_unpositioned(segment, 1.0, segment.max_doc())?,
            ))
        } else {
            None
        };
        let alive = segment.alive_bitset();
        Ok(docs.filter(move |&doc| {
            if alive.is_some_and(|alive| !alive.is_alive(doc)) {
                return false;
            }
            if let Some(scorer) = &mut scorer {
                let _predicate = enter_vector_stage(VectorStage::LateFilterPredicate);
                scorer.contains(doc)
            } else {
                true
            }
        }))
    }

    /// Drains the filter `DocSet` once, in ascending doc order, keeping up to `cap` matches as a
    /// list.
    ///
    /// The first match past `cap` moves the list into a bitset and the drain continues there. A
    /// match-all scorer, or a drain that matches every doc, routes unfiltered. The bitset allocates
    /// `max_doc / 8` bytes whatever the selectivity.
    #[inline(never)]
    fn collect_pre_matches(
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        cap: usize,
    ) -> crate::Result<FilterMatches> {
        let max_doc = segment_reader.max_doc();
        let mut scorer = weight.scorer(segment_reader, 1.0)?;
        if scorer.is::<AllScorer>() {
            return Ok(FilterMatches::Routed(DocFilter::All));
        }
        let mut sparse = Vec::new();
        let mut dense = (cap == 0).then(|| BitSet::with_max_value(max_doc));
        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        for_each_docset_buffered(scorer.as_mut(), &mut buffer, |docs| {
            for &doc in docs {
                match &mut dense {
                    Some(filter) => {
                        filter.insert(doc);
                    }
                    None if sparse.len() < cap => sparse.push(doc),
                    None => {
                        let mut filter = BitSet::with_max_value(max_doc);
                        for &doc in &sparse {
                            filter.insert(doc);
                        }
                        filter.insert(doc);
                        sparse = Vec::new();
                        dense = Some(filter);
                    }
                }
            }
        });
        let matched = dense.as_ref().map_or(sparse.len(), BitSet::len);
        if matched == max_doc as usize {
            return Ok(FilterMatches::Routed(DocFilter::All));
        }
        Ok(match dense {
            Some(filter) => FilterMatches::Routed(DocFilter::Docs(filter.into())),
            None => FilterMatches::Located(sparse),
        })
    }

    fn filter_post_clusters(
        vectors: &VectorIndexReader,
        weight: &dyn Weight,
        segment: &SegmentReader,
        candidates: impl Iterator<Item = Candidate>,
    ) -> Result<DocFilter> {
        let mut candidates = candidates.peekable();
        let mut docs = BitSet::with_max_value(0);
        if candidates.peek().is_some() {
            docs = BitSet::with_max_value(segment.max_doc());
            let mut scratch = Vec::new();
            for candidate in candidates {
                vectors.read_doc_ids(candidate.node as usize, &mut scratch)?;
                for &doc in &scratch {
                    docs.insert(doc);
                }
            }
            drop(scratch);
            let candidate_count = docs.len();
            let mut scorer = ExcludedDocSet::new(weight.scorer_for_candidates_unpositioned(
                segment,
                1.0,
                candidate_count as u32,
            )?);
            let mut bucket = docs.first_non_empty_bucket(0);
            while let Some(current_bucket) = bucket {
                for bit in docs.tinyset(current_bucket) {
                    let doc = current_bucket * u64::BITS + bit;
                    if !scorer.contains(doc) {
                        docs.remove(doc);
                    }
                }
                bucket = docs.first_non_empty_bucket(current_bucket + 1);
            }
        }
        Ok(DocFilter::Docs(Arc::new(docs)))
    }
}

/// The filter's matches as the router source gates rows with them.
#[derive(Clone)]
pub(crate) enum DocFilter {
    /// Every doc id in the segment matches; deleted docs are excluded separately.
    All,
    /// Matching document ids, stored as a bitset.
    Docs(Arc<BitSet>),
    /// Post has not checked the routed clusters yet; resolve this before scoring their vectors.
    Pending,
    /// Defer filter checks to the approximate shortlist, or to each doc during a flat scan.
    Late,
}

impl DocFilter {
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Docs(docs) => docs.len() == 0,
            Self::All | Self::Pending | Self::Late => false,
        }
    }

    /// The share of doc ids below `max_doc` that match.
    pub(crate) fn match_fraction(&self, max_doc: DocId) -> f64 {
        match self {
            Self::All | Self::Late => 1.0,
            Self::Docs(docs) => docs.len() as f64 / f64::from(max_doc.max(1)),
            Self::Pending => unreachable!("pending filter has no observed selectivity"),
        }
    }
}

/// How a cluster row is tested before scoring: against the filter's matches and the alive docs.
pub(crate) enum RowGate<'a> {
    Open,
    Pending,
    AliveOnly(&'a AliveBitSet),
    FilterOnly(Arc<BitSet>),
    FilterAndAlive {
        filter: Arc<BitSet>,
        alive: &'a AliveBitSet,
    },
}

/// What a gate kept of one cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatedRows {
    /// Every row is kept; the offsets are left empty.
    pub(crate) all: bool,
    pub(crate) pruned_filter: usize,
    pub(crate) pruned_dead: usize,
}

impl<'a> RowGate<'a> {
    pub(crate) fn new(filter: DocFilter, alive: Option<&'a AliveBitSet>) -> Self {
        match (filter, alive) {
            (DocFilter::Pending, _) => RowGate::Pending,
            (DocFilter::All | DocFilter::Late, None) => RowGate::Open,
            (DocFilter::All | DocFilter::Late, Some(alive)) => RowGate::AliveOnly(alive),
            (DocFilter::Docs(filter), None) => RowGate::FilterOnly(filter),
            (DocFilter::Docs(filter), Some(alive)) => RowGate::FilterAndAlive { filter, alive },
        }
    }

    /// Whether every row passes without reading its doc id.
    pub(crate) fn is_open(&self) -> bool {
        matches!(self, RowGate::Open)
    }

    /// Gates a cluster whose DocIds column is `doc_ids`, writing the kept cluster-local offsets.
    pub(crate) fn select(&self, doc_ids: &[DocId], offsets: &mut Vec<u32>) -> GatedRows {
        offsets.clear();
        let verdicts = match self {
            RowGate::Pending => unreachable!("routed filter must be collected before scoring"),
            RowGate::Open => {
                return GatedRows {
                    all: true,
                    pruned_filter: 0,
                    pruned_dead: 0,
                }
            }
            RowGate::AliveOnly(alive) => gate(doc_ids, offsets, |doc| {
                let alive = alive.is_alive(doc);
                RowVerdict {
                    matches: true,
                    keep: alive,
                }
            }),
            RowGate::FilterOnly(filter) => gate(doc_ids, offsets, |doc| {
                let matches = filter.contains(doc);
                RowVerdict {
                    matches,
                    keep: matches,
                }
            }),
            RowGate::FilterAndAlive { filter, alive } => gate(doc_ids, offsets, |doc| {
                let matches = filter.contains(doc);
                RowVerdict {
                    matches,
                    keep: matches & alive.is_alive(doc),
                }
            }),
        };
        GatedRows {
            all: false,
            pruned_filter: doc_ids.len() - verdicts.matched,
            pruned_dead: verdicts.matched - verdicts.kept,
        }
    }
}

/// One row's verdict: whether it matches the filter, and whether it is kept, which also needs it
/// alive.
struct RowVerdict {
    matches: bool,
    keep: bool,
}

/// Rows a gate counted: those matching the filter, and those kept.
struct GateCounts {
    matched: usize,
    kept: usize,
}

/// Keeps the offsets `verdict` accepts. Each row's offset is written at the next free slot and the
/// slot advances by the verdict, so the loop has no branch on it.
#[inline(always)]
fn gate(
    doc_ids: &[DocId],
    offsets: &mut Vec<u32>,
    verdict: impl Fn(DocId) -> RowVerdict,
) -> GateCounts {
    offsets.resize(doc_ids.len(), 0);
    let mut kept = 0usize;
    let mut matched = 0usize;
    for (offset, &doc) in (0u32..).zip(doc_ids) {
        let RowVerdict { matches, keep } = verdict(doc);
        offsets[kept] = offset;
        kept += usize::from(keep);
        matched += usize::from(matches);
    }
    offsets.truncate(kept);
    GateCounts { matched, kept }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::collector::{Collector, TopDocs};
    use crate::docset::SeekDangerResult;
    use crate::query::score_combiner::DoNothingCombiner;
    use crate::query::{
        AllQuery, BooleanWeight, ConstScorer, EnableScoring, Explanation, Occur, Query, Scorer,
    };
    use crate::schema::Metric;
    use crate::vector::test_support::{DocsWeight, FixtureSpec};
    use crate::vector::tests::exhaustive_params;
    use crate::vector::{
        AtomicVectorWork, LocatedPath, ProbeParams, SharedVectorWork, VectorSearchBudget,
    };
    use crate::{DocSet, Score};

    struct ProbeWeight {
        matches: Arc<Vec<DocId>>,
        probes: Arc<Mutex<Vec<DocId>>>,
        sizes: Arc<Mutex<Vec<u32>>>,
    }

    impl Weight for ProbeWeight {
        fn uses_fast_fields(&self) -> bool {
            true
        }

        fn scorer(&self, _: &SegmentReader, _: Score) -> Result<Box<dyn Scorer>> {
            panic!("Post must not enumerate the segment's matches")
        }

        fn scorer_for_candidates_unpositioned(
            &self,
            _: &SegmentReader,
            boost: Score,
            candidate_size: u32,
        ) -> Result<Box<dyn Scorer>> {
            self.sizes.lock().unwrap().push(candidate_size);
            Ok(Box::new(ConstScorer::new(
                ProbeDocs {
                    matches: Arc::clone(&self.matches),
                    probes: Arc::clone(&self.probes),
                    doc: 0,
                },
                boost,
            )))
        }

        fn explain(&self, _: &SegmentReader, _: DocId) -> Result<Explanation> {
            unreachable!()
        }
    }

    struct ProbeDocs {
        matches: Arc<Vec<DocId>>,
        probes: Arc<Mutex<Vec<DocId>>>,
        doc: DocId,
    }

    impl DocSet for ProbeDocs {
        fn advance(&mut self) -> DocId {
            panic!("Post must only probe candidate document IDs")
        }

        fn doc(&self) -> DocId {
            self.doc
        }

        fn size_hint(&self) -> u32 {
            self.matches.len() as u32
        }

        fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
            self.probes.lock().unwrap().push(target);
            self.doc = target;
            if self.matches.binary_search(&target).is_ok() {
                SeekDangerResult::Found
            } else {
                SeekDangerResult::SeekLowerBound(target + 1)
            }
        }
    }

    #[test]
    fn post_only_checks_routed_documents_in_every_executor() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const CLUSTERS: usize = 8;
        const DOCUMENTS: usize = 129;
        const TOP_K: usize = 5;
        for replicated in [false, true] {
            let mut spec = FixtureSpec::clustered(Metric::L2, DIMENSIONS, CLUSTERS, DOCUMENTS)
                .deletes(&[0, 3]);
            if replicated {
                spec = spec.replicated();
            }
            let fixture = spec.build()?;
            let searcher = fixture.index.reader()?.searcher();
            let segment = searcher.segment_reader(0);
            let segments = [VectorSegment::from((0, segment))];
            let vectors = segment.vector_index(fixture.field)?;
            let matches: Vec<_> = (0..segment.max_doc()).filter(|doc| doc % 3 == 0).collect();
            let pre_weight = DocsWeight {
                max_doc: segment.max_doc(),
                docs: matches.clone(),
            };
            let params = ProbeParams {
                located: LocatedPath::Off,
                ..exhaustive_params(CLUSTERS)
            };
            let budget = VectorSearchBudget::for_segments(fixture.field, &params, &segments)?;
            let builder = TopDocs::with_limit(TOP_K)
                .order_by_similarity(fixture.field, vec![0.0f32; DIMENSIONS])
                .with_probe_params(params.clone());
            let all_clusters: Vec<_> = (0..CLUSTERS).collect();
            for chosen in [&[][..], &[0, 2][..], all_clusters.as_slice()] {
                let candidates = fixture.candidates(chosen, &[0.0; DIMENSIONS])?;
                let mut expected_probes = Vec::new();
                let mut scratch = Vec::new();
                for candidate in &candidates {
                    vectors.read_doc_ids(candidate.node as usize, &mut scratch)?;
                    expected_probes.extend_from_slice(&scratch);
                }
                expected_probes.sort_unstable();
                expected_probes.dedup();
                if chosen.len() < CLUSTERS {
                    assert!(expected_probes.len() < segment.max_doc() as usize);
                } else {
                    assert_eq!(expected_probes.len(), segment.max_doc() as usize);
                }
                let routed: Vec<_> = candidates
                    .iter()
                    .map(|c| Candidate {
                        sim: c.sim,
                        node: (0, c.node),
                    })
                    .collect();
                let prepared = builder.prepare(&pre_weight, &segments, budget, |_| {})?;
                let pre = builder
                    .clone()
                    .build(prepared.with_candidates(routed.clone(), None)?);
                let expected =
                    pre.merge_fruits(pre.collect_segments(&pre_weight, &segments, |_| {})?)?;
                for (shared, independent, wrapped) in [
                    (false, false, false),
                    (true, false, false),
                    (false, true, false),
                    (false, false, true),
                    (true, false, true),
                    (false, true, true),
                ] {
                    let probes = Arc::default();
                    let sizes = Arc::default();
                    let weight = Box::new(ProbeWeight {
                        matches: Arc::new(matches.clone()),
                        probes: Arc::clone(&probes),
                        sizes: Arc::clone(&sizes),
                    });
                    let weight: Box<dyn Weight> = if wrapped {
                        Box::new(BooleanWeight::with_minimum_number_should_match(
                            vec![
                                (
                                    Occur::Must,
                                    AllQuery
                                        .weight(EnableScoring::disabled_from_searcher(&searcher))?,
                                ),
                                (Occur::Must, weight),
                            ],
                            0,
                            false,
                            Box::new(DoNothingCombiner::default),
                        ))
                    } else {
                        weight
                    };
                    let post_builder = builder.clone().with_probe_params(ProbeParams {
                        filter_strategy: Some(VectorFilterStrategy::Post {
                            estimated_selectivity: 0.1,
                        }),
                        ..params.clone()
                    });
                    let prepared =
                        post_builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
                    let work = shared.then(|| {
                        Arc::new(AtomicVectorWork::default()) as Arc<dyn SharedVectorWork>
                    });
                    let post = post_builder.build(prepared.with_candidates(routed.clone(), work)?);
                    let fruits = if independent {
                        vec![post.collect_segment(weight.as_ref(), 0, segment)?]
                    } else {
                        post.collect_segments(weight.as_ref(), &segments, |_| {})?
                    };
                    let actual = post.merge_fruits(fruits)?;
                    assert_eq!(
                        actual.results, expected.results,
                        "shared={shared}, independent={independent}, wrapped={wrapped}"
                    );
                    assert_eq!(*probes.lock().unwrap(), expected_probes);
                    let sizes = sizes.lock().unwrap();
                    if expected_probes.is_empty() {
                        assert!(sizes.is_empty());
                    } else {
                        assert_eq!(*sizes, [expected_probes.len() as u32]);
                    }
                }
            }
        }
        Ok(())
    }
}

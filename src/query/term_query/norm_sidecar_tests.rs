use std::cell::{Cell, RefCell};
use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::HasLen;

use super::{TermQuery, TermScorer};
use crate::collector::{Count, TopDocs};
use crate::directory::{FileHandle, FileSlice, OwnedBytes};
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentId;
use crate::merge_policy::NoMergePolicy;
use crate::postings::{set_norm_sidecar_provider, SegmentPostings};
use crate::query::{Bm25Weight, EnableScoring, QueryParser, Scorer, Weight};
use crate::schema::{IndexRecordOption, Schema, TEXT};
use crate::{Bm25Params, DocSet, Executor, Index, Term, TERMINATED};

#[derive(Debug)]
struct CountedBytes {
    bytes: Vec<u8>,
    reads: AtomicUsize,
}

impl CountedBytes {
    fn new(bytes: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            reads: AtomicUsize::new(0),
        })
    }
}

impl HasLen for CountedBytes {
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl FileHandle for CountedBytes {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        self.reads.fetch_add(range.len(), Ordering::Relaxed);
        Ok(OwnedBytes::new(self.bytes[range].to_vec()))
    }
}

#[test]
fn test_norm_sidecar_counts_exact_scores_after_advance_seek_shallow_seek_and_clone() {
    let mut empty = TermScorer::new(
        SegmentPostings::empty(),
        FieldNormReader::constant(0, 1),
        Bm25Weight::for_one_term(0, 1, 1.0, Bm25Params::default()),
    );
    empty.set_norm_sidecar(FileSlice::from(Vec::<u8>::new()));
    assert_eq!(empty.doc(), TERMINATED);
    assert_eq!(empty.block_max_score(), 0.0);
    for count in [1, 127, 128, 129, 255, 256, 301] {
        let norms: Vec<u32> = (0..count * 3 + 2)
            .map(|doc| 1 + doc as u32 * 31 % 5000)
            .collect();
        let docs: Vec<_> = (0..count)
            .map(|ord| ((ord * 3 + 2) as u32, 1 + ord as u32 % 17))
            .collect();
        let original = CountedBytes::new(
            norms
                .iter()
                .copied()
                .map(FieldNormReader::fieldnorm_to_id)
                .collect(),
        );
        let lane = CountedBytes::new(
            docs.iter()
                .map(|(doc, _)| original.bytes[*doc as usize])
                .collect(),
        );
        let weight = Bm25Weight::for_one_term(
            count as u64,
            norms.len() as u64,
            100.0,
            Bm25Params::default(),
        );
        let mut scorer = TermScorer::new(
            SegmentPostings::create_from_docs_and_tfs(&docs, Some(&norms)),
            FieldNormReader::open(FileSlice::new(original.clone())),
            weight.clone(),
        );
        scorer.set_norm_sidecar(FileSlice::new(lane.clone()));
        let initial = scorer.clone();
        for (ordinal, &(doc, tf)) in docs.iter().enumerate() {
            assert_eq!(scorer.doc(), doc);
            assert_eq!(scorer.fieldnorm_id(), lane.bytes[ordinal]);
            assert_eq!(scorer.score(), weight.score(lane.bytes[ordinal], tf));
            scorer.advance();
        }
        assert_eq!(scorer.doc(), TERMINATED);
        for shallow in [false, true] {
            let mut seeker = initial.clone();
            for target in [
                2, 3, 380, 383, 384, 385, 386, 767, 768, 770, 899, TERMINATED,
            ] {
                if target < seeker.doc() {
                    continue;
                }
                if shallow {
                    seeker.seek_block(target);
                }
                let ordinal = docs.partition_point(|(doc, _)| *doc < target);
                let expected_doc = docs.get(ordinal).map_or(TERMINATED, |(doc, _)| *doc);
                assert_eq!(seeker.seek(target), expected_doc);
                if expected_doc != TERMINATED {
                    assert_eq!(seeker.fieldnorm_id(), lane.bytes[ordinal]);
                    assert_eq!(
                        seeker.score(),
                        weight.score(lane.bytes[ordinal], docs[ordinal].1)
                    );
                    let mut cloned = seeker.clone();
                    assert_eq!(cloned.score(), seeker.score());
                    cloned.advance();
                    assert_eq!(
                        cloned.doc(),
                        docs.get(ordinal + 1).map_or(TERMINATED, |(doc, _)| *doc)
                    );
                }
            }
        }
        assert_eq!(original.reads.load(Ordering::Relaxed), 0);
        assert!(lane.reads.load(Ordering::Relaxed) >= count * 2);
        let tail_doc = docs.last().unwrap().0;
        let mut tail = initial.clone();
        tail.seek(tail_doc);
        let prior_reads = lane.reads.load(Ordering::Relaxed);
        let maximum = tail.block_max_score();
        assert_eq!(tail.block_max_score(), maximum);
        assert_eq!(
            lane.reads.load(Ordering::Relaxed) - prior_reads,
            count % 128
        );
        assert_eq!(original.reads.load(Ordering::Relaxed), 0);
        let mut fallback = TermScorer::new(
            SegmentPostings::create_from_docs_and_tfs(&docs, Some(&norms)),
            FieldNormReader::open(FileSlice::new(original.clone())),
            weight,
        );
        fallback.seek(tail_doc);
        assert_eq!(fallback.block_max_score(), maximum);
        assert_eq!(original.reads.swap(0, Ordering::Relaxed), count % 128);
        for mode in [1, 2] {
            crate::postings::set_max_score_bound_mode(mode);
            let mut bounds = initial.clone();
            bounds.prepare_max_score_bounds();
            assert_eq!(original.reads.load(Ordering::Relaxed), 0);
        }
        crate::postings::set_max_score_bound_mode(0);
    }
}

thread_local! {
    static LANES: RefCell<Vec<(SegmentId, Term, FileSlice)>> = const { RefCell::new(Vec::new()) };
    static PROVIDER_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn provider(segment: SegmentId, term: &Term) -> Option<FileSlice> {
    PROVIDER_CALLS.set(PROVIDER_CALLS.get() + 1);
    LANES.with_borrow(|lanes| {
        lanes
            .iter()
            .find(|(id, key, _)| *id == segment && key == term)
            .map(|(_, _, lane)| lane.clone())
    })
}

struct ResetProvider;

impl Drop for ResetProvider {
    fn drop(&mut self) {
        set_norm_sidecar_provider(None);
        crate::postings::set_union_deferred_seeks(false);
        crate::postings::set_max_score_bound_mode(0);
        crate::postings::set_postings_read_buffer_size(0);
        LANES.with_borrow_mut(Vec::clear);
        PROVIDER_CALLS.set(0);
    }
}

#[test]
fn test_norm_sidecar_provider_single_term_or_fallback_and_scoring_disabled() -> crate::Result<()> {
    let _reset = ResetProvider;
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", TEXT);
    let index = Index::create_in_ram(schema.build());
    let mut writer = index.writer_for_tests()?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for doc in 0..779 {
        let mut body = "padding ".repeat(1 + doc % 41);
        for (term, divisor) in [("alpha", 2), ("beta", 3), ("gamma", 5)] {
            if doc % divisor == 0 {
                body.push_str(&format!("{term} ").repeat(1 + doc % 7));
            }
        }
        writer.add_document(crate::doc!(text => body))?;
    }
    writer.commit()?;
    let reader = index.reader()?;
    let original_searcher = reader.searcher();
    let mut lane_counters = Vec::new();
    for segment in original_searcher.segment_readers() {
        let inverted = segment.inverted_index(text)?;
        let norms = segment.fieldnorms_readers().get_field(text)?.unwrap();
        for value in ["alpha", "beta", "gamma"] {
            let term = Term::from_field_text(text, value);
            let mut postings = inverted
                .read_postings(&term, IndexRecordOption::WithFreqs)?
                .unwrap();
            let mut bytes = Vec::new();
            while postings.doc() != TERMINATED {
                bytes.push(norms.fieldnorm_id(postings.doc()));
                postings.advance();
            }
            if value == "gamma" {
                bytes = vec![0; bytes.len() - 1];
            }
            let counter = CountedBytes::new(bytes);
            LANES.with_borrow_mut(|lanes| {
                lanes.push((segment.segment_id(), term, FileSlice::new(counter.clone())))
            });
            lane_counters.push((value, counter));
        }
    }
    writer.add_document(crate::doc!(text => "alpha beta gamma alpha"))?;
    writer.commit()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert!(searcher.segment_readers().len() > original_searcher.segment_readers().len());
    let parser = QueryParser::for_index(&index, vec![text]);
    let executor = Executor::single_thread();
    for query_text in ["alpha", "gamma", "alpha OR beta OR gamma", "alpha AND beta"] {
        let query = parser.parse_query(query_text)?;
        for (limit, (defer, bounds, buffer)) in [1, 10, 1000].into_iter().flat_map(|limit| {
            [(false, 0, 0), (true, 0, 16), (true, 2, 8192)].map(|mode| (limit, mode))
        }) {
            crate::postings::set_union_deferred_seeks(defer);
            crate::postings::set_max_score_bound_mode(bounds);
            crate::postings::set_postings_read_buffer_size(buffer);
            let collector = TopDocs::with_limit(limit).order_by_score();
            set_norm_sidecar_provider(None);
            let expected = searcher.search_with_executor(
                query.as_ref(),
                &collector,
                &executor,
                EnableScoring::enabled_from_searcher(&searcher),
            )?;
            set_norm_sidecar_provider(Some(provider));
            let actual = searcher.search_with_executor(
                query.as_ref(),
                &collector,
                &executor,
                EnableScoring::enabled_from_searcher(&searcher),
            )?;
            assert_eq!(
                actual, expected,
                "{query_text}, limit={limit}, defer={defer}, bounds={bounds}, buffer={buffer}"
            );
        }
        let calls = PROVIDER_CALLS.get();
        searcher.search_with_executor(
            query.as_ref(),
            &Count,
            &executor,
            EnableScoring::disabled_from_searcher(&searcher),
        )?;
        assert_eq!(PROVIDER_CALLS.get(), calls);
    }
    assert!(PROVIDER_CALLS.get() > 0);
    for (term, counter) in lane_counters {
        if term == "gamma" {
            assert_eq!(counter.reads.load(Ordering::Relaxed), 0);
        } else {
            assert!(counter.reads.load(Ordering::Relaxed) > 0);
        }
    }
    let disabled = TermQuery::new(
        Term::from_field_text(text, "alpha"),
        IndexRecordOption::WithFreqs,
    )
    .specialized_weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let calls = PROVIDER_CALLS.get();
    disabled.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(PROVIDER_CALLS.get(), calls);
    Ok(())
}

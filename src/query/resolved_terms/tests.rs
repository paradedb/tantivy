use std::io;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use super::*;
use crate::collector::TopDocs;
use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
use crate::directory::{
    Directory, FileHandle, InnerWritePtr, RamDirectory, WatchCallback, WatchHandle,
};
use crate::indexer::NoMergePolicy;
use crate::query::{
    BooleanQuery, BoostQuery, EnableScoring, Occur, PhraseQuery, Query, TermQuery, Weight,
};
use crate::schema::{IndexRecordOption, Schema, TEXT};
use crate::{Index, IndexWriter};

#[derive(Debug, Default)]
struct Reads {
    count: AtomicUsize,
    fail: AtomicBool,
}

#[derive(Debug)]
struct CountedFile {
    inner: Arc<dyn FileHandle>,
    reads: Arc<Reads>,
}

impl HasLen for CountedFile {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl FileHandle for CountedFile {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        self.reads.count.fetch_add(1, Ordering::Relaxed);
        if self.reads.fail.load(Ordering::Relaxed) {
            return Err(io::Error::other("unexpected term dictionary read"));
        }
        self.inner.read_bytes(range)
    }
}

#[derive(Clone, Debug, Default)]
struct CountedDirectory {
    inner: RamDirectory,
    reads: Arc<Reads>,
}

impl Directory for CountedDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let inner = self.inner.get_file_handle(path)?;
        if path.extension().is_some_and(|ext| ext == "term") {
            Ok(Arc::new(CountedFile {
                inner,
                reads: Arc::clone(&self.reads),
            }))
        } else {
            Ok(inner)
        }
    }
    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)
    }
    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }
    fn open_write_inner(&self, path: &Path) -> Result<InnerWritePtr, OpenWriteError> {
        self.inner.open_write_inner(path)
    }
    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.inner.atomic_read(path)
    }
    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.inner.atomic_write(path, data)
    }
    fn sync_directory(&self) -> io::Result<()> {
        self.inner.sync_directory()
    }
    fn watch(&self, callback: WatchCallback) -> crate::Result<WatchHandle> {
        self.inner.watch(callback)
    }
}

fn fixture(num_segments: usize) -> crate::Result<(Index, Searcher, Arc<Reads>, Field, Field)> {
    let directory = CountedDirectory::default();
    let reads = Arc::clone(&directory.reads);
    let mut schema = Schema::builder();
    let first = schema.add_text_field("first", TEXT);
    let second = schema.add_text_field("second", TEXT);
    let index = Index::create(directory, schema.build(), Default::default())?;
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let words = (0..2048)
        .map(|i| format!("word{i:04}"))
        .collect::<Vec<_>>()
        .join(" ");
    for segment in 0..num_segments {
        for i in 0..16 {
            let text = if i == 0 {
                words.clone()
            } else if i % 2 == 0 {
                "run runner running rust".into()
            } else {
                "rust rust memory safety".into()
            };
            let other = if (i + segment) % 3 == 0 {
                "run rust"
            } else {
                "memory"
            };
            writer.add_document(doc!(first => text, second => other))?;
        }
        writer.commit()?;
    }
    writer.wait_merging_threads()?;
    let searcher = index.reader()?.searcher();
    for segment in searcher.segment_readers() {
        segment.inverted_index(first)?;
        segment.inverted_index(second)?;
    }
    reads.count.store(0, Ordering::Relaxed);
    Ok((index, searcher, reads, first, second))
}

struct Unresolved<'a>(&'a Searcher);

impl Bm25StatisticsProvider for Unresolved<'_> {
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
        Bm25StatisticsProvider::total_num_tokens(self.0, field)
    }
    fn total_num_docs(&self) -> crate::Result<u64> {
        Bm25StatisticsProvider::total_num_docs(self.0)
    }
    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        self.0.doc_freq(term)
    }
    fn bm25_params(&self, field: Field) -> crate::index::Bm25Params {
        Bm25StatisticsProvider::bm25_params(self.0, field)
    }
}

fn term_query(field: Field, value: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(
        Term::from_field_text(field, value),
        IndexRecordOption::WithFreqs,
    ))
}

#[test]
fn resolved_terms_preserve_scores_and_remove_scorer_dictionary_reads() -> crate::Result<()> {
    let (_, searcher, reads, field, _) = fixture(2)?;
    for size in [1, 10, 129, 256, 1024] {
        let query = BooleanQuery::new(
            (0..size)
                .rev()
                .map(|i| (Occur::Should, term_query(field, &format!("word{i:04}"))))
                .collect(),
        );
        let collector = TopDocs::with_limit(32).order_by_score();
        assert_eq!(
            searcher.search(&query, &collector)?,
            searcher.search_with_statistics_provider(&query, &collector, &Unresolved(&searcher))?
        );
        reads.count.store(0, Ordering::Relaxed);
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let resolution_reads = reads.count.load(Ordering::Relaxed);
        assert!(resolution_reads > 0);
        reads.fail.store(true, Ordering::Relaxed);
        for segment in searcher.segment_readers() {
            weight.scorer(segment, 1.0)?;
        }
        reads.fail.store(false, Ordering::Relaxed);
        assert_eq!(reads.count.load(Ordering::Relaxed), resolution_reads);
    }
    Ok(())
}

#[test]
fn mixed_fields_duplicates_boosts_and_phrases_preserve_semantics() -> crate::Result<()> {
    let (_, searcher, reads, first, second) = fixture(2)?;
    let phrase = PhraseQuery::new(vec![
        Term::from_field_text(first, "rust"),
        Term::from_field_text(first, "rust"),
    ]);
    let queries: Vec<Box<dyn Query>> = vec![
        term_query(first, "rust"),
        term_query(first, "runner_absent"),
        Box::new(phrase),
        Box::new(BooleanQuery::new(vec![
            (Occur::Must, term_query(first, "rust")),
            (Occur::Should, term_query(first, "rust")),
            (Occur::Should, term_query(second, "run")),
            (Occur::MustNot, term_query(second, "missing")),
            (
                Occur::Should,
                Box::new(BoostQuery::new(term_query(first, "run"), 2.5)),
            ),
        ])),
    ];
    for query in queries {
        let collector = TopDocs::with_limit(32).order_by_score();
        assert_eq!(
            searcher.search(query.as_ref(), &collector)?,
            searcher.search_with_statistics_provider(
                query.as_ref(),
                &collector,
                &Unresolved(&searcher)
            )?
        );
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        reads.fail.store(true, Ordering::Relaxed);
        for segment in searcher.segment_readers() {
            weight.scorer(segment, 1.0)?;
        }
        reads.fail.store(false, Ordering::Relaxed);
    }
    Ok(())
}

#[test]
fn failed_resolution_is_retryable_and_reopened_segments_reuse_metadata() -> crate::Result<()> {
    let (index, searcher, reads, field, _) = fixture(2)?;
    let terms = [
        Term::from_field_text(field, "run"),
        Term::from_field_text(field, "rust"),
    ];
    reads.fail.store(true, Ordering::Relaxed);
    assert!(ResolvedTerms::new(&searcher, &terms).is_err());
    reads.fail.store(false, Ordering::Relaxed);
    let resolved = ResolvedTerms::new(&searcher, &terms)?;
    let other = index.reader()?.searcher();
    for term in &terms {
        let info = resolved.term_infos.get(term).unwrap();
        assert_eq!(resolved.doc_freqs[term], searcher.doc_freq(term)?);
        for segment in searcher.segment_readers() {
            assert!(info.get(&segment.segment_id()).is_some());
        }
        for segment in other.segment_readers() {
            assert!(info.get(&segment.segment_id()).is_some());
        }
    }
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    writer.delete_term(Term::from_field_text(field, "memory"));
    writer.commit()?;
    writer.wait_merging_threads()?;
    let deleted = index.reader()?.searcher();
    assert!(deleted
        .segment_readers()
        .iter()
        .all(|segment| segment.num_deleted_docs() > 0));
    let queries: Vec<Box<dyn Query>> = vec![
        term_query(field, "rust"),
        Box::new(PhraseQuery::new(terms.to_vec())),
    ];
    let (_, unrelated, unrelated_reads, _, _) = fixture(2)?;
    for query in queries {
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        for (target, target_reads, known) in [
            (&other, &reads, true),
            (&deleted, &reads, true),
            (&unrelated, &unrelated_reads, false),
        ] {
            let fallback = query.weight(EnableScoring::enabled_from_statistics_provider(
                &Unresolved(target),
                target,
            ))?;
            for segment in target.segment_readers() {
                target_reads.count.store(0, Ordering::Relaxed);
                let mut actual = weight.scorer(segment, 1.0)?;
                assert_eq!(target_reads.count.load(Ordering::Relaxed) == 0, known);
                let mut expected = fallback.scorer(segment, 1.0)?;
                while actual.doc() != crate::TERMINATED {
                    assert_eq!(actual.doc(), expected.doc());
                    assert_eq!(actual.score(), expected.score());
                    actual.advance();
                    expected.advance();
                }
                assert_eq!(expected.doc(), crate::TERMINATED);
            }
        }
    }
    Ok(())
}

#[test]
fn many_segments_preserve_metadata_and_cached_absence() -> crate::Result<()> {
    let (_, searcher, reads, first, second) = fixture(128)?;
    assert_eq!(searcher.segment_readers().len(), 128);
    let terms = [
        Term::from_field_text(first, "rust"),
        Term::from_field_text(first, "missing"),
        Term::from_field_text(second, "run"),
        Term::from_field_text(second, "missing"),
    ];
    let resolved = ResolvedTerms::new(&searcher, &terms)?;
    let unfamiliar = crate::index::SegmentId::generate_random();
    for term in &terms {
        let info = resolved.term_infos.get(term).unwrap();
        assert_eq!(resolved.doc_freqs[term], searcher.doc_freq(term)?);
        assert_eq!(info.get(&unfamiliar), None);
        for segment in searcher.segment_readers().iter().rev() {
            let reader = segment.inverted_index(term.field())?;
            let expected = reader.get_term_info(term)?;
            reads.fail.store(true, Ordering::Relaxed);
            assert_eq!(info.get(&segment.segment_id()), Some(&expected));
            reads.fail.store(false, Ordering::Relaxed);
        }
    }
    Ok(())
}

#[test]
fn concurrent_scorers_share_resolved_metadata_without_dictionary_io() -> crate::Result<()> {
    let (_, searcher, reads, field, _) = fixture(2)?;
    let query = BooleanQuery::new(
        (0..256)
            .map(|i| (Occur::Should, term_query(field, &format!("word{i:04}"))))
            .collect(),
    );
    let weight: Arc<dyn Weight> =
        Arc::from(query.weight(EnableScoring::enabled_from_searcher(&searcher))?);
    reads.fail.store(true, Ordering::Relaxed);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let weight = &weight;
            let searcher = &searcher;
            scope.spawn(move || {
                for _ in 0..4 {
                    for segment in searcher.segment_readers() {
                        assert_eq!(weight.scorer(segment, 1.0).unwrap().doc(), 0);
                    }
                }
            });
        }
    });
    reads.fail.store(false, Ordering::Relaxed);
    Ok(())
}

#[test]
fn unscored_weights_do_not_resolve_terms() -> crate::Result<()> {
    let (_, searcher, reads, field, _) = fixture(1)?;
    let queries: Vec<Box<dyn Query>> = vec![
        term_query(field, "rust"),
        Box::new(PhraseQuery::new(vec![
            Term::from_field_text(field, "run"),
            Term::from_field_text(field, "rust"),
        ])),
        Box::new(BooleanQuery::new(vec![
            (Occur::Should, term_query(field, "run")),
            (Occur::Should, term_query(field, "rust")),
        ])),
    ];
    reads.fail.store(true, Ordering::Relaxed);
    for query in queries {
        query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    }
    assert_eq!(reads.count.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn custom_statistics_are_not_replaced_by_local_frequencies() -> crate::Result<()> {
    struct CustomStatistics {
        calls: AtomicUsize,
    }

    impl Bm25StatisticsProvider for CustomStatistics {
        fn total_num_tokens(&self, _field: Field) -> crate::Result<u64> {
            Ok(10_000)
        }

        fn total_num_docs(&self) -> crate::Result<u64> {
            Ok(100)
        }

        fn doc_freq(&self, _term: &Term) -> crate::Result<u64> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        }

        fn bm25_params(&self, _field: Field) -> crate::index::Bm25Params {
            crate::index::Bm25Params::new(2.0, 0.3)
        }
    }

    let (_, searcher, reads, field, _) = fixture(1)?;
    let queries: Vec<Box<dyn Query>> = vec![
        term_query(field, "rust"),
        Box::new(PhraseQuery::new(vec![
            Term::from_field_text(field, "memory"),
            Term::from_field_text(field, "safety"),
        ])),
        Box::new(BooleanQuery::new(vec![
            (Occur::Should, term_query(field, "run")),
            (Occur::Should, term_query(field, "rust")),
        ])),
    ];
    let statistics = CustomStatistics {
        calls: AtomicUsize::new(0),
    };
    for query in queries {
        reads.fail.store(true, Ordering::Relaxed);
        statistics.calls.store(0, Ordering::Relaxed);
        query.weight(EnableScoring::enabled_from_statistics_provider(
            &statistics,
            &searcher,
        ))?;
        assert!(statistics.calls.load(Ordering::Relaxed) > 0);
        reads.fail.store(false, Ordering::Relaxed);
        let collector = TopDocs::with_limit(32).order_by_score();
        let local = searcher.search(query.as_ref(), &collector)?;
        let custom =
            searcher.search_with_statistics_provider(query.as_ref(), &collector, &statistics)?;
        assert_eq!(local.len(), custom.len());
        assert!(!local.is_empty());
        assert_ne!(local[0].0, custom[0].0);
    }
    Ok(())
}

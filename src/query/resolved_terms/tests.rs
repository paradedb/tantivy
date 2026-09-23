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

fn fixture() -> crate::Result<(Index, Searcher, Arc<Reads>, Field, Field)> {
    let directory = CountedDirectory::default();
    let reads = Arc::clone(&directory.reads);
    let mut schema = Schema::builder();
    let first = schema.add_text_field("first", TEXT);
    let second = schema.add_text_field("second", TEXT);
    let index = Index::create(directory, schema.build(), Default::default())?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let words = (0..2048)
        .map(|i| format!("word{i:04}"))
        .collect::<Vec<_>>()
        .join(" ");
    for segment in 0..2 {
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

struct Unbatched<'a>(&'a Searcher);

impl Bm25StatisticsProvider for Unbatched<'_> {
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
fn large_batches_preserve_scores_and_remove_scorer_dictionary_reads() -> crate::Result<()> {
    let (_, searcher, reads, field, _) = fixture()?;
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
            searcher.search_with_statistics_provider(&query, &collector, &Unbatched(&searcher))?
        );
        reads.count.store(0, Ordering::Relaxed);
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let batch_reads = reads.count.load(Ordering::Relaxed);
        reads.fail.store(true, Ordering::Relaxed);
        for segment in searcher.segment_readers() {
            weight.scorer(segment, 1.0)?;
        }
        reads.fail.store(false, Ordering::Relaxed);
        assert_eq!(reads.count.load(Ordering::Relaxed), batch_reads);
        reads.count.store(0, Ordering::Relaxed);
        query.weight(EnableScoring::enabled_from_statistics_provider(
            &Unbatched(&searcher),
            &searcher,
        ))?;
        let scalar_reads = reads.count.load(Ordering::Relaxed);
        assert!(batch_reads > 0);
        if size > 1 {
            assert!(
                batch_reads < scalar_reads,
                "{size}: {batch_reads} >= {scalar_reads}"
            );
        }
    }
    Ok(())
}

#[test]
fn mixed_fields_duplicates_boosts_and_phrases_preserve_semantics() -> crate::Result<()> {
    let (_, searcher, reads, first, second) = fixture()?;
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
                &Unbatched(&searcher)
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
fn failed_resolution_is_retryable_and_new_readers_do_not_reuse_addresses() -> crate::Result<()> {
    let (index, searcher, reads, field, _) = fixture()?;
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
        let info = resolved.get(term).unwrap();
        assert_eq!(info.doc_freq, searcher.doc_freq(term)?);
        for segment in searcher.segment_readers() {
            assert!(info.get(&segment.inverted_index(field)?).is_some());
        }
        for segment in other.segment_readers() {
            assert!(info.get(&segment.inverted_index(field)?).is_none());
        }
    }
    let weight =
        term_query(field, "rust").weight(EnableScoring::enabled_from_searcher(&searcher))?;
    reads.count.store(0, Ordering::Relaxed);
    for segment in other.segment_readers() {
        weight.scorer(segment, 1.0)?;
    }
    assert!(reads.count.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn concurrent_scorers_share_resolved_metadata_without_dictionary_io() -> crate::Result<()> {
    let (_, searcher, reads, field, _) = fixture()?;
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

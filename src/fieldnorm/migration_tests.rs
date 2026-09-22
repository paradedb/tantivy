use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::collector::{Count, TopDocs};
use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
use crate::directory::{
    Directory, FileHandle, InnerWritePtr, RamDirectory, WatchCallback, WatchHandle,
};
use crate::index::SegmentComponent;
use crate::indexer::operation::AddOperation;
use crate::indexer::SegmentWriter;
use crate::merge_policy::NoMergePolicy;
use crate::postings::Postings;
use crate::query::{
    BooleanQuery, EnableScoring, PhrasePrefixQuery, PhraseQuery, Query, RegexPhraseQuery, TermQuery,
};
use crate::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, TEXT,
};
use crate::{
    DocSet, Index, IndexSettings, IndexSortByField, IndexWriter, Order, Score, Searcher, Term,
    TERMINATED,
};

#[derive(Clone, Debug)]
struct NormDirectory {
    inner: RamDirectory,
    legacy: Arc<Mutex<HashSet<PathBuf>>>,
}

impl NormDirectory {
    fn check(&self, path: &Path) {
        assert!(
            path.extension().is_none_or(|ext| ext != "fieldnorm")
                || self.legacy.lock().unwrap().contains(path),
            "unexpected global fieldnorm access: {path:?}"
        );
    }
}

impl Directory for NormDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.check(path);
        self.inner.get_file_handle(path)
    }
    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)
    }
    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }
    fn open_write_inner(&self, path: &Path) -> Result<InnerWritePtr, OpenWriteError> {
        self.check(path);
        self.inner.open_write_inner(path)
    }
    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.check(path);
        self.inner.atomic_read(path)
    }
    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.check(path);
        self.inner.atomic_write(path, data)
    }
    fn sync_directory(&self) -> io::Result<()> {
        self.inner.sync_directory()
    }
    fn watch(&self, callback: WatchCallback) -> crate::Result<WatchHandle> {
        self.inner.watch(callback)
    }
}

fn fixture(sorted: bool, legacy: &[bool]) -> crate::Result<(Index, IndexWriter, NormDirectory)> {
    let mut schema = Schema::builder();
    let id = schema.add_u64_field("id", INDEXED | FAST);
    let text = schema.add_text_field("text", TEXT);
    let basic = schema.add_text_field(
        "basic",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default().set_index_option(IndexRecordOption::Basic),
        ),
    );
    let disabled = schema.add_text_field(
        "disabled",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_fieldnorms(false)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    );
    schema.add_text_field("empty", TEXT);
    let directory = NormDirectory {
        inner: RamDirectory::create(),
        legacy: Default::default(),
    };
    let settings = IndexSettings {
        sort_by_field: sorted.then(|| IndexSortByField {
            field: "id".into(),
            order: Order::Desc,
        }),
        ..Default::default()
    };
    let index = Index::builder()
        .schema(schema.build())
        .settings(settings)
        .create(directory.clone())?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for (ordinal, &legacy) in legacy.iter().enumerate() {
        let segment = index.new_segment();
        let mut segment_writer = SegmentWriter::for_segment(15_000_000, segment.clone(), false)?;
        if legacy {
            directory
                .legacy
                .lock()
                .unwrap()
                .insert(segment.relative_path(SegmentComponent::FieldNorms));
            segment_writer.inverted_index.use_legacy_norms(&segment)?;
        }
        for i in 0..201u64 {
            let value = i * 3 + ordinal as u64;
            let body = if value % 11 == 0 {
                String::new()
            } else {
                format!(
                    "{} {} {}",
                    if value % 7 == 0 {
                        "postman databases"
                    } else {
                        "postgres database"
                    },
                    "database ".repeat((value % 4) as usize),
                    "padding ".repeat((value % 53) as usize)
                )
            };
            segment_writer.add_document(AddOperation {
                opstamp: 0,
                document: doc!(id=>value, text=>body.clone(), basic=>body.clone(), disabled=>body),
            })?;
        }
        let max_doc = segment_writer.max_doc();
        segment_writer.finalize()?;
        writer.add_segment(segment.with_max_doc(max_doc).meta().clone())?;
    }
    writer.commit()?;
    Ok((index, writer, directory))
}

fn queries(schema: &Schema) -> Vec<Box<dyn Query>> {
    let text = schema.get_field("text").unwrap();
    let term = |field: Field, word: &str| {
        Box::new(TermQuery::new(
            Term::from_field_text(field, word),
            IndexRecordOption::WithFreqs,
        )) as Box<dyn Query>
    };
    let mut slop = PhraseQuery::new(vec![
        Term::from_field_text(text, "postgres"),
        Term::from_field_text(text, "padding"),
    ]);
    slop.set_slop(4);
    vec![
        term(text, "database"),
        term(text, "postman"),
        term(schema.get_field("basic").unwrap(), "database"),
        term(schema.get_field("disabled").unwrap(), "database"),
        term(schema.get_field("empty").unwrap(), "database"),
        Box::new(TermQuery::new(
            Term::from_field_u64(schema.get_field("id").unwrap(), 100),
            IndexRecordOption::Basic,
        )),
        Box::new(BooleanQuery::intersection(vec![
            term(text, "database"),
            term(text, "postgres"),
        ])),
        Box::new(BooleanQuery::union(vec![
            term(text, "database"),
            term(text, "postgres"),
        ])),
        Box::new(PhraseQuery::new(vec![
            Term::from_field_text(text, "postgres"),
            Term::from_field_text(text, "database"),
        ])),
        Box::new(slop),
        Box::new(PhrasePrefixQuery::new(vec![
            Term::from_field_text(text, "postgres"),
            Term::from_field_text(text, "data"),
        ])),
        Box::new(PhrasePrefixQuery::new(vec![
            Term::from_field_text(text, "postgres"),
            Term::from_field_text(text, "database"),
            Term::from_field_text(text, "pad"),
        ])),
        Box::new(RegexPhraseQuery::new(
            text,
            vec!["post.*".into(), "data.*".into()],
        )),
        Box::new(RegexPhraseQuery::new(
            text,
            vec!["postman".into(), "databases".into()],
        )),
    ]
}

fn scores(searcher: &Searcher, query: &dyn Query) -> crate::Result<Vec<(u64, Score)>> {
    let hits = searcher.search(query, &TopDocs::with_limit(1000).order_by_score())?;
    assert_eq!(hits.len(), searcher.search(query, &Count)?);
    let mut scores = Vec::new();
    for (score, address) in hits {
        let id = searcher
            .segment_reader(address.segment_ord)
            .fast_fields()
            .u64("id")?
            .first(address.doc_id)
            .unwrap();
        assert_eq!(query.explain(searcher, address)?.value(), score);
        scores.push((id, score));
    }
    scores.sort_unstable_by_key(|&(id, _)| id);
    Ok(scores)
}

#[test]
fn mixed_formats_scoring_deletions_and_sorted_merges() -> crate::Result<()> {
    for sorted in [false, true] {
        for modes in [&[false, false, false][..], &[true, false, true][..]] {
            let (reference, mut reference_writer, _) = fixture(sorted, &[true, true, true])?;
            let (index, mut writer, _) = fixture(sorted, modes)?;
            let reference_reader = reference.reader()?;
            let reader = index.reader()?;
            for deleted in [false, true] {
                if deleted {
                    for id in [100, 203, 411] {
                        let term = Term::from_field_u64(index.schema().get_field("id")?, id);
                        writer.delete_term(term.clone());
                        reference_writer.delete_term(term);
                    }
                    writer.commit()?;
                    reference_writer.commit()?;
                    reader.reload()?;
                    reference_reader.reload()?;
                }
                for query in queries(&index.schema()) {
                    assert_eq!(
                        scores(&reader.searcher(), &*query)?,
                        scores(&reference_reader.searcher(), &*query)?,
                        "{query:?}, sorted={sorted}, deleted={deleted}"
                    );
                }
            }
            writer.merge(&index.searchable_segment_ids()?).wait()?;
            reader.reload()?;
            let searcher = reader.searcher();
            let segment = searcher.segment_reader(0);
            assert_eq!(segment.num_docs(), 600);
            for name in ["text", "basic", "id"] {
                let field = index.schema().get_field(name)?;
                let inverted = segment.inverted_index(field)?;
                assert_eq!(inverted.norm_storage(), super::NormStorage::Posting);
                let norms = segment.get_fieldnorms_reader(field)?;
                let mut terms = inverted.terms().stream()?;
                while terms.advance() {
                    let mut postings = inverted
                        .read_postings_from_terminfo(terms.value(), IndexRecordOption::Basic)?;
                    while postings.doc() != TERMINATED {
                        assert_eq!(
                            postings.fieldnorm_id(),
                            Some(norms.fieldnorm_id(postings.doc()))
                        );
                        postings.advance();
                    }
                }
                let expected_tokens: u64 = reference_reader
                    .searcher()
                    .segment_readers()
                    .iter()
                    .map(|source| {
                        let norms = source.get_fieldnorms_reader(field).unwrap();
                        source
                            .doc_ids_alive()
                            .map(|doc| norms.fieldnorm(doc) as u64)
                            .sum::<u64>()
                    })
                    .sum();
                assert_eq!(inverted.total_num_tokens(), expected_tokens);
            }
            searcher.space_usage()?;
            for query in queries(&index.schema()) {
                scores(&searcher, &*query)?;
            }
            // Merge the reference once as well so its post-deletion BM25 statistics match.
            reference_writer
                .merge(&reference.searchable_segment_ids()?)
                .wait()?;
            reference_reader.reload()?;
            for query in queries(&index.schema()) {
                assert_eq!(
                    scores(&searcher, &*query)?,
                    scores(&reference_reader.searcher(), &*query)?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn required_posting_norms_cannot_be_disabled() -> crate::Result<()> {
    let (index, _writer, directory) = fixture(false, &[false])?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    crate::postings::set_posting_norms_enabled(false);
    for query in queries(&index.schema()) {
        scores(&searcher, &*query)?;
    }
    crate::postings::set_posting_norms_enabled(true);
    let field = index.schema().get_field("text")?;
    let query = TermQuery::new(
        Term::from_field_text(field, "database"),
        IndexRecordOption::WithFreqs,
    );
    let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
    let segment = &index.searchable_segments()?[0];
    index
        .directory()
        .delete(&segment.relative_path(SegmentComponent::Custom("pnorm".into())))
        .unwrap();
    let reopened = Index::open(directory)?;
    let searcher = reopened.reader()?.searcher();
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scorer.score())).is_err());
    Ok(())
}

#[test]
fn inserts_into_legacy_index_and_merge_without_deletions() -> crate::Result<()> {
    let (index, mut writer, directory) = fixture(false, &[true])?;
    let schema = index.schema();
    let text = schema.get_field("text")?;
    writer.add_document(doc!(schema.get_field("id")?=>999u64, text=>"postgres database"))?;
    writer.commit()?;
    let reader = index.reader()?;
    let before = reader.searcher();
    let modes: HashSet<_> = before
        .segment_readers()
        .iter()
        .map(|segment| segment.inverted_index(text).unwrap().norm_storage() as usize)
        .collect();
    assert_eq!(
        modes,
        HashSet::from([
            super::NormStorage::Legacy as usize,
            super::NormStorage::Posting as usize
        ])
    );
    let query = TermQuery::new(
        Term::from_field_text(text, "database"),
        IndexRecordOption::WithFreqs,
    );
    let expected = scores(&before, &query)?;
    writer.merge(&index.searchable_segment_ids()?).wait()?;
    reader.reload()?;
    assert_eq!(scores(&reader.searcher(), &query)?, expected);
    assert_eq!(
        reader
            .searcher()
            .segment_reader(0)
            .inverted_index(text)?
            .norm_storage(),
        super::NormStorage::Posting
    );
    let reopened = Index::open(directory)?;
    assert_eq!(scores(&reopened.reader()?.searcher(), &query)?, expected);
    Ok(())
}

#[test]
fn missing_required_legacy_norms_and_posting_headers_are_errors() -> crate::Result<()> {
    let (index, _writer, directory) = fixture(false, &[true])?;
    let field = index.schema().get_field("text")?;
    let term = Term::from_field_text(field, "database");
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let inverted = searcher.segment_reader(0).inverted_index(field)?;
    let mut block = inverted
        .read_block_postings(&term, IndexRecordOption::WithFreqs)?
        .unwrap();
    assert!(block
        .set_term_norm_source(
            Arc::new(common::file_slice::DeferredFileSlice::new(|| Ok(
                crate::directory::FileSlice::empty()
            ))),
            super::NormStorage::Posting
        )
        .is_err());
    let segment = &index.searchable_segments()?[0];
    index
        .directory()
        .delete(&segment.relative_path(SegmentComponent::FieldNorms))
        .unwrap();
    let reopened = Index::open(directory)?;
    let searcher = reopened.reader()?.searcher();
    let query = TermQuery::new(term, IndexRecordOption::WithFreqs);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scores(&searcher, &query)))
            .is_err()
    );
    Ok(())
}

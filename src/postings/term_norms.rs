use std::cell::{Cell, RefCell};
use std::io;
use std::sync::Arc;

use common::file_slice::DeferredFileSlice;
use common::{BinarySerializable, HasLen};

use crate::directory::{BufferedFileSlice, OwnedBytes};

pub(crate) const MAGIC: [u8; 10] = [127, 127, 127, 127, 127, 127, 127, 127, 127, 130];
const BUFFER_SIZE: usize = 8192;

thread_local! {
    static READS: Cell<u64> = const { Cell::new(0) };
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Selects term-local norm reads for newly opened scorers in this thread.
pub fn set_posting_norms_enabled(enabled: bool) -> bool {
    READS.set(0);
    ENABLED.replace(enabled)
}

/// Returns the number of term-local norm lookups since the last mode change.
pub fn posting_norm_reads() -> u64 {
    READS.get()
}

pub(crate) fn read_header(mut bytes: OwnedBytes) -> io::Result<(Option<u64>, OwnedBytes)> {
    if !bytes.starts_with(&MAGIC) {
        return Ok((None, bytes));
    }
    bytes.advance(MAGIC.len());
    let offset = u64::deserialize(&mut bytes)?;
    Ok((Some(offset), bytes))
}

#[derive(Clone)]
pub(crate) struct TermNormReader {
    source: Arc<DeferredFileSlice>,
    offset: usize,
    len: usize,
    buffer: RefCell<Option<BufferedFileSlice>>,
}

impl TermNormReader {
    pub(crate) fn new(
        source: Arc<DeferredFileSlice>,
        offset: u64,
        len: u32,
        required: bool,
    ) -> Option<Self> {
        (required || ENABLED.get()).then(|| Self {
            source,
            offset: offset as usize,
            len: len as usize,
            buffer: RefCell::new(None),
        })
    }

    pub(crate) fn read(&self, ordinal: usize) -> io::Result<u8> {
        if ordinal >= self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "posting norm ordinal out of bounds",
            ));
        }
        READS.set(READS.get() + 1);
        let mut buffer = self.buffer.borrow_mut();
        if buffer.is_none() {
            let source = self.source.open()?;
            let end = self
                .offset
                .checked_add(self.len)
                .filter(|&end| end <= source.len())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms")
                })?;
            *buffer = Some(BufferedFileSlice::new(
                source.slice(self.offset..end),
                BUFFER_SIZE,
            ));
        }
        buffer.as_ref().unwrap().read_byte(ordinal as u64)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::directory::{FileHandle, FileSlice};

    #[derive(Debug)]
    struct TrackedFile {
        reads: Arc<Mutex<Vec<Range<usize>>>>,
    }

    impl HasLen for TrackedFile {
        fn len(&self) -> usize {
            30000
        }
    }

    impl FileHandle for TrackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(
                range.map(|i| (i % 251) as u8).collect::<Vec<_>>(),
            ))
        }
    }

    #[test]
    fn lazy_reads_and_retained_bytes() {
        set_posting_norms_enabled(true);
        let reads = Arc::new(Mutex::new(Vec::new()));
        let opens = Arc::new(AtomicUsize::new(0));
        let file = FileSlice::new(Arc::new(TrackedFile {
            reads: reads.clone(),
        }));
        let open_count = opens.clone();
        let source = Arc::new(DeferredFileSlice::new(move || {
            open_count.fetch_add(1, Ordering::Relaxed);
            Ok(file.clone())
        }));
        let reader = TermNormReader::new(source, 10, 29000, false).unwrap();
        assert_eq!(opens.load(Ordering::Relaxed), 0);
        assert!(reads.lock().unwrap().is_empty());
        for ordinal in [0, 127, 8000, 8191] {
            assert_eq!(reader.read(ordinal).unwrap(), ((ordinal + 10) % 251) as u8);
        }
        assert_eq!(*reads.lock().unwrap(), vec![10..8202]);
        let clone = reader.clone();
        assert_eq!(clone.read(8001).unwrap(), (8011 % 251) as u8);
        assert_eq!(reads.lock().unwrap().len(), 1);
        assert_eq!(reader.read(20000).unwrap(), (20010 % 251) as u8);
        assert_eq!(*reads.lock().unwrap(), vec![10..8202, 20010..28202]);
        assert_eq!(reader.read(28192).unwrap(), (28202 % 251) as u8);
        assert_eq!(reader.read(28999).unwrap(), (29009 % 251) as u8);
        assert_eq!(reads.lock().unwrap().last().unwrap(), &(28202..29010));
        assert_eq!(opens.load(Ordering::Relaxed), 1);
        assert!(reader.read(29000).is_err());
        let empty = BufferedFileSlice::empty();
        assert!(empty.read_byte(0).is_err());
        assert!(empty.read_byte(u64::MAX).is_err());
    }

    #[test]
    fn malformed_header_and_stream() {
        assert!(read_header(OwnedBytes::new(MAGIC.to_vec())).is_err());
        set_posting_norms_enabled(true);
        let reader = TermNormReader::new(
            Arc::new(DeferredFileSlice::new(|| Ok(FileSlice::from(vec![1])))),
            0,
            2,
            false,
        )
        .unwrap();
        assert!(reader.read(0).is_err());
    }

    #[cfg(feature = "posting-norms")]
    #[test]
    fn scores_seeks_deletes_and_merges() -> crate::Result<()> {
        use crate::collector::TopDocs;
        use crate::merge_policy::NoMergePolicy;
        use crate::query::{BooleanQuery, Occur, Query, TermQuery};
        use crate::schema::{IndexRecordOption, Schema, INDEXED, TEXT};
        use crate::{DocSet, Index, IndexWriter, Term, TERMINATED};

        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", TEXT);
        let id = schema.add_u64_field("id", INDEXED);
        let index = Index::create_in_ram(schema.build());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..26000u64 {
            let text = if i % 3 == 0 {
                "empty".to_owned()
            } else {
                format!(
                    "{} {} {}",
                    "database ".repeat((i % 4 + 1) as usize),
                    "padding ".repeat((i % 73) as usize),
                    if i % 7 == 0 { "postgres" } else { "" }
                )
            };
            writer.add_document(doc!(id=>i, title=>text))?;
            if i == 12999 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        writer.delete_term(Term::from_field_u64(id, 301));
        writer.commit()?;
        let reader = index.reader()?;
        for merged in [false, true] {
            if merged {
                let ids = index.searchable_segment_ids()?;
                writer.merge(&ids).wait()?;
                reader.reload()?;
            }
            let searcher = reader.searcher();
            let make_term = |word: &str| {
                Box::new(TermQuery::new(
                    Term::from_field_text(title, word),
                    IndexRecordOption::WithFreqs,
                )) as Box<dyn Query>
            };
            let queries = [
                make_term("database"),
                Box::new(BooleanQuery::new(vec![
                    (Occur::Must, make_term("database")),
                    (Occur::Must, make_term("postgres")),
                ])) as Box<dyn Query>,
                Box::new(BooleanQuery::new(vec![
                    (Occur::Should, make_term("database")),
                    (Occur::Should, make_term("postgres")),
                ])) as Box<dyn Query>,
            ];
            for query in queries {
                set_posting_norms_enabled(true);
                let actual = searcher.search(&*query, &TopDocs::with_limit(25).order_by_score())?;
                assert!(!actual.is_empty());
                assert!(posting_norm_reads() > 0);
            }
            for segment in searcher.segment_readers() {
                let inv = segment.inverted_index(title)?;
                let norms = segment.get_fieldnorms_reader(title)?;
                let term = Term::from_field_text(title, "database");
                let info = inv.get_term_info(&term)?.unwrap();
                let query = TermQuery::new(term.clone(), IndexRecordOption::WithFreqs);
                let weight = query.weight(crate::query::EnableScoring::disabled_from_searcher(
                    &searcher,
                ))?;
                set_posting_norms_enabled(true);
                let mut scorer = weight.scorer(segment, 1.0)?;
                while scorer.doc() != TERMINATED {
                    scorer.score();
                    scorer.advance();
                }
                assert_eq!(posting_norm_reads(), 0);
                let mut postings = inv
                    .read_postings(&term, IndexRecordOption::WithFreqs)?
                    .unwrap();
                set_posting_norms_enabled(true);
                postings.seek(150);
                assert_eq!(posting_norm_reads(), 0);
                while postings.doc() != TERMINATED {
                    assert_eq!(
                        postings
                            .block_cursor
                            .fieldnorm_id_at(postings.block_offset(), &norms),
                        norms.fieldnorm_id(postings.doc())
                    );
                    postings.advance();
                }
                let mut block =
                    inv.read_block_postings_from_terminfo(&info, IndexRecordOption::WithFreqs)?;
                block.seek(200);
                inv.reset_block_postings_from_terminfo(&info, &mut block)?;
                assert_eq!(
                    block.fieldnorm_id_at(0, &norms),
                    norms.fieldnorm_id(block.doc(0))
                );
            }
        }
        set_posting_norms_enabled(true);
        Ok(())
    }
}

use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::Arc;

use common::{BinarySerializable, CountingWriter, HasLen};
use once_cell::sync::OnceCell;

use crate::directory::{BufferedFileSlice, FileSlice};

const BUFFER_SIZE: usize = 8192;

// Each field contains norm bytes, sorted (postings offset, norm offset) pairs, and the index
// length.
pub(crate) struct TermNormsWriter<'a, W: Write> {
    write: &'a mut CountingWriter<W>,
    start_offset: u64,
    offsets: Vec<(u64, u64)>,
}

impl<'a, W: Write> TermNormsWriter<'a, W> {
    pub(crate) fn new(write: &'a mut CountingWriter<W>) -> io::Result<Self> {
        Ok(Self {
            start_offset: write.written_bytes(),
            write,
            offsets: Vec::new(),
        })
    }

    pub(crate) fn write(&mut self, postings_offset: usize, norms: &[u8]) -> io::Result<()> {
        let postings_offset = postings_offset as u64;
        if self
            .offsets
            .last()
            .is_some_and(|&(previous, _)| previous >= postings_offset)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "posting offsets must increase",
            ));
        }
        self.offsets.push((
            postings_offset,
            self.write.written_bytes() - self.start_offset,
        ));
        self.write.write_all(norms)
    }

    pub(crate) fn close(self) -> io::Result<()> {
        let index_len = self.offsets.len() as u64 * 16;
        for (postings_offset, norm_offset) in self.offsets {
            postings_offset.serialize(self.write)?;
            norm_offset.serialize(self.write)?;
        }
        index_len.serialize(self.write)
    }
}

pub(crate) struct PostingNormsReader {
    source: FileSlice,
    index: OnceCell<(FileSlice, FileSlice)>,
}

impl PostingNormsReader {
    pub(crate) fn new(source: FileSlice) -> Self {
        Self {
            source,
            index: OnceCell::new(),
        }
    }

    fn term_slice(&self, postings_offset: usize, len: usize) -> io::Result<FileSlice> {
        let (norms, offsets) = self.index.get_or_try_init(|| {
            let source = &self.source;
            if source.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated posting norms",
                ));
            }
            let (body, footer) = source.clone().split_from_end(8);
            let index_len = u64::deserialize(&mut footer.read_bytes()?)?;
            if index_len > body.len() as u64 || index_len % 16 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid posting norm index length",
                ));
            }
            let (norms, index) = body.split_from_end(index_len as usize);
            Ok((norms, index))
        })?;
        let mut range = 0..offsets.len() / 16;
        let mut norm_offset = None;
        while range.start < range.end {
            let mid = range.start + (range.end - range.start) / 2;
            let mut entry = offsets.read_bytes_slice(mid * 16..(mid + 1) * 16)?;
            let key = u64::deserialize(&mut entry)?;
            match key.cmp(&(postings_offset as u64)) {
                std::cmp::Ordering::Less => range.start = mid + 1,
                std::cmp::Ordering::Greater => range.end = mid,
                std::cmp::Ordering::Equal => {
                    norm_offset = Some(u64::deserialize(&mut entry)?);
                    break;
                }
            }
        }
        let offset = norm_offset.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing posting norm term")
        })?;
        let end = offset
            .checked_add(len as u64)
            .filter(|&end| end <= norms.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms"))?;
        Ok(norms.slice(offset as usize..end as usize))
    }
}

#[derive(Clone)]
pub(crate) struct TermNormReader {
    source: Arc<PostingNormsReader>,
    postings_offset: usize,
    len: usize,
    buffer: RefCell<Option<BufferedFileSlice>>,
}

impl TermNormReader {
    pub(crate) fn new(source: Arc<PostingNormsReader>, postings_offset: usize, len: u32) -> Self {
        Self {
            source,
            postings_offset,
            len: len as usize,
            buffer: RefCell::new(None),
        }
    }

    #[inline]
    pub(crate) fn read(&self, ordinal: usize) -> io::Result<u8> {
        if ordinal >= self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "posting norm ordinal out of bounds",
            ));
        }
        let mut buffer = self.buffer.borrow_mut();
        if buffer.is_none() {
            *buffer = Some(BufferedFileSlice::new(
                self.source.term_slice(self.postings_offset, self.len)?,
                BUFFER_SIZE,
            ));
        }
        buffer.as_ref().unwrap().read_byte(ordinal as u64)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Mutex;

    use super::*;
    use crate::directory::{FileHandle, FileSlice, OwnedBytes};

    #[derive(Debug)]
    struct TrackedFile {
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        data: Vec<u8>,
    }

    impl HasLen for TrackedFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for TrackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.data[range].to_vec()))
        }
    }

    #[test]
    fn lazy_reads_and_retained_bytes() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let data: Vec<u8> = (0..30000).map(|i| (i % 251) as u8).collect();
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut writer = TermNormsWriter::new(&mut write).unwrap();
        writer.write(0, &data[..10]).unwrap();
        writer.write(42, &data[10..29010]).unwrap();
        writer.write(100, &data[29010..]).unwrap();
        writer.close().unwrap();
        let file = FileSlice::new(Arc::new(TrackedFile {
            reads: reads.clone(),
            data: bytes,
        }));
        let source = Arc::new(PostingNormsReader::new(file));
        let reader = TermNormReader::new(source.clone(), 42, 29000);
        assert!(reads.lock().unwrap().is_empty());
        for ordinal in [0, 127, 8000, 8191] {
            assert_eq!(reader.read(ordinal).unwrap(), ((ordinal + 10) % 251) as u8);
        }
        assert_eq!(reads.lock().unwrap().last().unwrap(), &(10..8202));
        let num_reads = reads.lock().unwrap().len();
        let clone = reader.clone();
        assert_eq!(clone.read(8001).unwrap(), (8011 % 251) as u8);
        assert_eq!(reads.lock().unwrap().len(), num_reads);
        assert_eq!(reader.read(20000).unwrap(), (20010 % 251) as u8);
        assert_eq!(reads.lock().unwrap().last().unwrap(), &(20010..28202));
        assert_eq!(reads.lock().unwrap().len(), num_reads + 1);
        assert_eq!(reader.read(28192).unwrap(), (28202 % 251) as u8);
        assert_eq!(reader.read(28999).unwrap(), (29009 % 251) as u8);
        assert_eq!(reads.lock().unwrap().last().unwrap(), &(28202..29010));
        assert!(reader.read(29000).is_err());
        assert!(TermNormReader::new(source.clone(), 43, 1).read(0).is_err());
        assert!(TermNormReader::new(source, 100, 1000).read(0).is_err());
        let empty = BufferedFileSlice::empty();
        assert!(empty.read_byte(0).is_err());
        assert!(empty.read_byte(u64::MAX).is_err());
    }

    #[test]
    fn malformed_index_and_stream() {
        for data in [vec![1], vec![255; 8], vec![0; 16]] {
            let source = Arc::new(PostingNormsReader::new(FileSlice::from(data)));
            assert!(TermNormReader::new(source, 0, 2).read(0).is_err());
        }
    }

    #[test]
    fn million_term_lookup_reads_bounded_ranges() {
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut writer = TermNormsWriter::new(&mut write).unwrap();
        for term in 0..1_000_000 {
            writer.write(term * 37, &[(term % 251) as u8]).unwrap();
        }
        writer.close().unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(TrackedFile {
            reads: reads.clone(),
            data: bytes,
        }));
        for term in [0, 123_456, 999_999] {
            reads.lock().unwrap().clear();
            let source = Arc::new(PostingNormsReader::new(file.clone()));
            let reader = TermNormReader::new(source, term * 37, 1);
            assert!(reads.lock().unwrap().is_empty());
            assert_eq!(reader.read(0).unwrap(), (term % 251) as u8);
            let ranges = reads.lock().unwrap().clone();
            assert!(ranges.iter().all(|range| range.len() <= 16));
            assert!(ranges.iter().map(|range| range.len()).sum::<usize>() <= 8 + 20 * 16 + 1);
            for _ in 0..100 {
                assert_eq!(reader.read(0).unwrap(), (term % 251) as u8);
            }
            assert_eq!(*reads.lock().unwrap(), ranges);
        }
    }

    #[test]
    fn pnorms_are_per_field_after_reopen_and_merge() -> crate::Result<()> {
        use crate::collector::TopDocs;
        use crate::directory::{CompositeFile, Directory, RamDirectory};
        use crate::index::SegmentComponent;
        use crate::indexer::NoMergePolicy;
        use crate::postings::Postings;
        use crate::query::TermQuery;
        use crate::schema::{IndexRecordOption, Schema, INDEXED, TEXT};
        use crate::{DateTime, DocSet, Index, Term, TERMINATED};

        let mut schema = Schema::builder();
        let indexing = TEXT
            .get_indexing_options()
            .unwrap()
            .clone()
            .set_pnorms(true);
        let text = schema.add_text_field("text", TEXT.set_indexing_options(indexing.clone()));
        let fallback = schema.add_text_field("fallback", TEXT);
        let unnormed = schema.add_text_field(
            "unnormed",
            TEXT.set_indexing_options(indexing.set_fieldnorms(false).set_pnorms(false)),
        );
        let number = schema.add_u64_field("number", INDEXED);
        let date = schema.add_date_field("date", INDEXED);
        let bytes = schema.add_bytes_field("bytes", INDEXED);
        let ip = schema.add_ip_addr_field("ip", INDEXED);
        let schema = schema.build();
        let directory = RamDirectory::create();
        let index = Index::create(directory.clone(), schema.clone(), Default::default())?;
        let mut writer = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let timestamp = DateTime::from_timestamp_secs(100);
        let address = std::net::Ipv6Addr::LOCALHOST;
        for body in ["one two", "one two three"] {
            for _ in 0..150 {
                let mut document =
                    doc!(text => body, fallback => body, unnormed => body, number => 1u64);
                document.add_date(date, timestamp);
                document.add_bytes(bytes, b"abc");
                document.add_ip_addr(ip, address);
                writer.add_document(document)?;
            }
            writer.commit()?;
        }
        drop(writer);
        let index = Index::open(directory.clone())?;
        assert_eq!(index.schema(), schema);
        let mut writer: crate::IndexWriter = index.writer_for_tests()?;
        let reader = index.reader()?;
        for merged in [false, true] {
            if merged {
                writer.delete_term(Term::from_field_text(text, "three"));
                writer.commit()?;
                writer.merge(&index.searchable_segment_ids()?).wait()?;
                reader.reload()?;
            }
            let searcher = reader.searcher();
            let query = |field| {
                TermQuery::new(
                    Term::from_field_text(field, "one"),
                    IndexRecordOption::WithFreqs,
                )
            };
            assert_eq!(
                searcher.search(&query(text), &TopDocs::with_limit(10).order_by_score())?,
                searcher.search(&query(fallback), &TopDocs::with_limit(10).order_by_score())?,
            );
            for segment in searcher.segment_readers() {
                let composite =
                    CompositeFile::open(&segment.open_read(SegmentComponent::PostingNorms)?)?;
                for field in [fallback, unnormed] {
                    assert!(composite.open_read(field).is_none());
                    assert!(!segment.inverted_index(field)?.has_pnorms());
                }
                for term in [
                    Term::from_field_text(text, "one"),
                    Term::from_field_u64(number, 1),
                    Term::from_field_date_for_search(date, timestamp),
                    Term::from_field_bytes(bytes, b"abc"),
                    Term::from_field_ip_addr(ip, address),
                ] {
                    let field = term.field();
                    let enabled = field == text;
                    assert_eq!(composite.open_read(field).is_some(), enabled);
                    let inverted = segment.inverted_index(field)?;
                    assert_eq!(inverted.has_pnorms(), enabled);
                    let norms = segment.get_fieldnorms_reader(field)?;
                    let mut postings = inverted
                        .read_postings(&term, IndexRecordOption::Basic)?
                        .unwrap();
                    while postings.doc() != TERMINATED {
                        assert_eq!(
                            postings.fieldnorm_id(),
                            enabled.then(|| norms.fieldnorm_id(postings.doc()))
                        );
                        postings.advance();
                    }
                }
            }
        }
        writer.garbage_collect_files().wait()?;
        for segment in index.searchable_segments()? {
            assert!(directory.exists(&segment.relative_path(SegmentComponent::PostingNorms))?);
        }
        Ok(())
    }

    #[test]
    fn pnorms_leave_builtin_files_unchanged() -> crate::Result<()> {
        use crate::index::SegmentComponent;
        use crate::schema::{Schema, TEXT};
        use crate::Index;

        let mut segments = Vec::new();
        for pnorms in [false, true] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(
                    TEXT.get_indexing_options()
                        .unwrap()
                        .clone()
                        .set_pnorms(pnorms),
                ),
            );
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            for id in 0..300 {
                writer.add_document(
                    doc!(text => format!("common anchor rare{id} {}", "padding ".repeat(id % 40))),
                )?;
            }
            writer.commit()?;
            let segment = index.searchable_segments()?.pop().unwrap();
            assert_eq!(
                segment.open_read(SegmentComponent::PostingNorms).is_ok(),
                pnorms
            );
            segments.push(segment);
        }
        for component in [
            SegmentComponent::Postings,
            SegmentComponent::Terms,
            SegmentComponent::Positions,
            SegmentComponent::FieldNorms,
        ] {
            assert_eq!(
                segments[0]
                    .open_read(component.clone())?
                    .read_bytes()?
                    .as_slice(),
                segments[1].open_read(component)?.read_bytes()?.as_slice()
            );
        }
        Ok(())
    }

    #[test]
    fn pnorms_follow_document_remapping() -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::indexer::DocIdMapping;
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{DocSet, Index, IndexSettings, TantivyDocument, Term, TERMINATED};

        let mapping = DocIdMapping::new_permutation(vec![1, 2, 0])?;
        for pnorms in [false, true] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(
                    TEXT.get_indexing_options()
                        .unwrap()
                        .clone()
                        .set_pnorms(pnorms),
                ),
            );
            let mut writer = Index::builder()
                .schema(schema.build())
                .settings(IndexSettings {
                    manual_doc_id_mapping: true,
                    ..Default::default()
                })
                .single_segment_index_writer(RamDirectory::default(), 15_000_000)?;
            writer.add_document(doc!(text => "common padding padding"))?;
            writer.add_document(TantivyDocument::default())?;
            writer.add_document(doc!(text => "common"))?;
            let index = writer.finalize_with_doc_id_mapping(&mapping)?;
            let searcher = index.reader()?.searcher();
            let segment = searcher.segment_reader(0);
            let norms = segment.get_fieldnorms_reader(text)?;
            assert_eq!(
                (0..3).map(|doc| norms.fieldnorm(doc)).collect::<Vec<_>>(),
                [0, 1, 3]
            );
            let inverted = segment.inverted_index(text)?;
            let mut postings = inverted
                .read_postings(
                    &Term::from_field_text(text, "common"),
                    IndexRecordOption::WithFreqs,
                )?
                .unwrap();
            for doc in [1, 2] {
                assert_eq!(postings.doc(), doc);
                assert_eq!(
                    postings
                        .block_cursor
                        .posting_fieldnorm_id_at(postings.block_offset()),
                    pnorms.then(|| norms.fieldnorm_id(doc))
                );
                postings.advance();
            }
            assert_eq!(postings.doc(), TERMINATED);
        }
        Ok(())
    }

    #[test]
    fn scoring_selects_norms_by_file_presence() -> crate::Result<()> {
        use crate::collector::TopDocs;
        use crate::directory::{Directory, RamDirectory};
        use crate::index::SegmentComponent;
        use crate::query::{
            BooleanQuery, Occur, PhrasePrefixQuery, PhraseQuery, Query, RegexPhraseQuery, TermQuery,
        };
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{Index, Term};

        for missing in [SegmentComponent::PostingNorms, SegmentComponent::FieldNorms] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(
                    TEXT.get_indexing_options()
                        .unwrap()
                        .clone()
                        .set_pnorms(true),
                ),
            );
            let directory = RamDirectory::create();
            let mut index = Index::builder()
                .schema(schema.build())
                .create(directory.clone())?;
            let mut writer = index.writer_for_tests()?;
            for body in ["red apple", "red apple pie", "green apple pie"]
                .into_iter()
                .cycle()
                .take(300)
            {
                writer.add_document(doc!(text => body))?;
            }
            writer.commit()?;
            drop(writer);
            if missing == SegmentComponent::FieldNorms {
                let mut metas = index.load_metas()?;
                let mut legacy_schema = Schema::builder();
                legacy_schema.add_text_field("text", TEXT);
                metas.schema = legacy_schema.build();
                index.directory_mut().atomic_write(
                    std::path::Path::new("meta.json"),
                    &serde_json::to_vec(&metas)?,
                )?;
                index = Index::open(directory.clone())?;
                assert!(!index.schema().get_field_entry(text).has_pnorms());
            }
            let red = Term::from_field_text(text, "red");
            let apple = Term::from_field_text(text, "apple");
            let queries: Vec<Box<dyn Query>> = vec![
                Box::new(TermQuery::new(red.clone(), IndexRecordOption::WithFreqs)),
                Box::new(BooleanQuery::new(vec![
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(red.clone(), IndexRecordOption::WithFreqs)),
                    ),
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(apple.clone(), IndexRecordOption::WithFreqs)),
                    ),
                ])),
                Box::new(PhraseQuery::new(vec![red.clone(), apple.clone()])),
                Box::new(PhrasePrefixQuery::new(vec![red, apple])),
                Box::new(RegexPhraseQuery::new(
                    text,
                    vec!["r.*".into(), "apple".into()],
                )),
            ];
            let searcher = index.reader()?.searcher();
            let expected = queries
                .iter()
                .map(|query| searcher.search(&**query, &TopDocs::with_limit(3).order_by_score()))
                .collect::<crate::Result<Vec<_>>>()?;
            for segment in index.searchable_segments()? {
                index
                    .directory()
                    .delete(&segment.relative_path(missing.clone()))
                    .unwrap();
            }
            let searcher = index.reader()?.searcher();
            for (query, expected) in queries.iter().zip(expected) {
                assert_eq!(
                    searcher.search(&**query, &TopDocs::with_limit(3).order_by_score())?,
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn scores_seeks_deletes_and_merges() -> crate::Result<()> {
        use crate::collector::TopDocs;
        use crate::merge_policy::NoMergePolicy;
        use crate::query::{BooleanQuery, Occur, Query, TermQuery};
        use crate::schema::{IndexRecordOption, Schema, INDEXED, TEXT};
        use crate::{DocSet, Index, IndexWriter, Term, TERMINATED};

        let mut schema = Schema::builder();
        let title = schema.add_text_field(
            "title",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_pnorms(true),
            ),
        );
        let id = schema.add_u64_field("id", INDEXED);
        let index = Index::builder().schema(schema.build()).create_in_ram()?;
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
                let actual = searcher.search(&*query, &TopDocs::with_limit(25).order_by_score())?;
                assert!(!actual.is_empty());
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
                let mut scorer = weight.scorer(segment, 1.0)?;
                while scorer.doc() != TERMINATED {
                    scorer.score();
                    scorer.advance();
                }
                let mut postings = inv
                    .read_postings(&term, IndexRecordOption::WithFreqs)?
                    .unwrap();
                postings.seek(150);
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
        Ok(())
    }
}

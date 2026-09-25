use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::Arc;

use common::file_slice::DeferredFileSlice;
use common::{BinarySerializable, CountingWriter, HasLen};
use once_cell::sync::OnceCell;

use crate::directory::{BufferedFileSlice, FileSlice, OwnedBytes};

const BUFFER_SIZE: usize = 8192;

// Each field contains norm bytes, an FST (postings offset -> norm offset), and the FST length.
pub(crate) struct TermNormsWriter<'a, W: Write> {
    write: &'a mut CountingWriter<W>,
    start_offset: u64,
    offsets: tantivy_fst::MapBuilder<Vec<u8>>,
}

impl<'a, W: Write> TermNormsWriter<'a, W> {
    pub(crate) fn new(write: &'a mut CountingWriter<W>) -> io::Result<Self> {
        Ok(Self {
            start_offset: write.written_bytes(),
            write,
            offsets: tantivy_fst::MapBuilder::new(Vec::new()).map_err(io::Error::other)?,
        })
    }

    pub(crate) fn write(&mut self, postings_offset: usize, norms: &[u8]) -> io::Result<()> {
        self.offsets
            .insert(
                (postings_offset as u64).to_be_bytes(),
                self.write.written_bytes() - self.start_offset,
            )
            .map_err(io::Error::other)?;
        self.write.write_all(norms)
    }

    pub(crate) fn close(self) -> io::Result<()> {
        let offsets = self.offsets.into_inner().map_err(io::Error::other)?;
        self.write.write_all(&offsets)?;
        (offsets.len() as u64).serialize(self.write)
    }
}

pub(crate) struct PostingNormsReader {
    source: DeferredFileSlice,
    index: OnceCell<(FileSlice, tantivy_fst::Map<OwnedBytes>)>,
}

impl PostingNormsReader {
    pub(crate) fn new(source: DeferredFileSlice) -> Self {
        Self {
            source,
            index: OnceCell::new(),
        }
    }

    fn term_slice(&self, postings_offset: usize, len: usize) -> io::Result<FileSlice> {
        let (norms, offsets) = self.index.get_or_try_init(|| {
            let source = self.source.open()?;
            if source.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated posting norms",
                ));
            }
            let (body, footer) = source.clone().split_from_end(8);
            let index_len = u64::deserialize(&mut footer.read_bytes()?)?;
            if index_len > body.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid posting norm index length",
                ));
            }
            let (norms, index) = body.split_from_end(index_len as usize);
            let fst = tantivy_fst::raw::Fst::new(index.read_bytes()?)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            Ok((norms, tantivy_fst::Map::from(fst)))
        })?;
        let offset = offsets
            .get((postings_offset as u64).to_be_bytes())
            .ok_or_else(|| {
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
        let offset = ordinal as u64;
        Ok(buffer.as_ref().unwrap().get_bytes(offset..offset + 1)?[0])
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
        let opens = Arc::new(AtomicUsize::new(0));
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
        let open_count = opens.clone();
        let source = Arc::new(PostingNormsReader::new(DeferredFileSlice::new(move || {
            open_count.fetch_add(1, Ordering::Relaxed);
            Ok(file.clone())
        })));
        let reader = TermNormReader::new(source.clone(), 42, 29000);
        assert_eq!(opens.load(Ordering::Relaxed), 0);
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
        assert_eq!(opens.load(Ordering::Relaxed), 1);
        assert!(reader.read(29000).is_err());
        assert!(TermNormReader::new(source.clone(), 43, 1).read(0).is_err());
        assert!(TermNormReader::new(source, 100, 1000).read(0).is_err());
        let empty = BufferedFileSlice::empty();
        assert!(empty.get_bytes(0..1).is_err());
    }

    #[test]
    fn malformed_index_and_stream() {
        for data in [vec![1], vec![255; 8], vec![0; 16]] {
            let source = Arc::new(PostingNormsReader::new(DeferredFileSlice::new(move || {
                Ok(FileSlice::from(data.clone()))
            })));
            assert!(TermNormReader::new(source, 0, 2).read(0).is_err());
        }
    }

    #[test]
    fn posting_norms_leave_builtin_files_unchanged() -> crate::Result<()> {
        use crate::index::SegmentComponent;
        use crate::schema::{Schema, TEXT};
        use crate::{Index, IndexSettings};

        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let schema = schema.build();
        let mut segments = Vec::new();
        for posting_norms in [false, true] {
            let index = Index::builder()
                .schema(schema.clone())
                .settings(IndexSettings {
                    posting_norms,
                    ..Default::default()
                })
                .create_in_ram()?;
            let mut writer = index.writer_for_tests()?;
            for id in 0..300 {
                writer.add_document(
                    doc!(text => format!("common anchor rare{id} {}", "padding ".repeat(id % 40))),
                )?;
            }
            writer.commit()?;
            segments.push(index.searchable_segments()?.pop().unwrap());
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
    fn posting_norms_follow_document_remapping() -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::indexer::DocIdMapping;
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{DocSet, Index, IndexSettings, TantivyDocument, Term, TERMINATED};

        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let schema = schema.build();
        let mapping = DocIdMapping::new_permutation(vec![1, 2, 0])?;
        for posting_norms in [false, true] {
            let mut writer = Index::builder()
                .schema(schema.clone())
                .settings(IndexSettings {
                    posting_norms,
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
                    posting_norms.then(|| norms.fieldnorm_id(doc))
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
        use crate::directory::Directory;
        use crate::index::SegmentComponent;
        use crate::query::{
            BooleanQuery, Occur, PhrasePrefixQuery, PhraseQuery, Query, RegexPhraseQuery, TermQuery,
        };
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{Index, Term};

        for missing in [SegmentComponent::PostingNorms, SegmentComponent::FieldNorms] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field("text", TEXT);
            let mut index = Index::builder()
                .schema(schema.build())
                .settings(crate::IndexSettings {
                    posting_norms: true,
                    ..Default::default()
                })
                .create_in_ram()?;
            let mut writer = index.writer_for_tests()?;
            for body in ["red apple", "red apple pie", "green apple pie"]
                .into_iter()
                .cycle()
                .take(300)
            {
                writer.add_document(doc!(text => body))?;
            }
            writer.commit()?;
            if missing == SegmentComponent::FieldNorms {
                index.settings_mut().posting_norms = false;
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
        let title = schema.add_text_field("title", TEXT);
        let id = schema.add_u64_field("id", INDEXED);
        let index = Index::builder()
            .schema(schema.build())
            .settings(crate::IndexSettings {
                posting_norms: true,
                ..Default::default()
            })
            .create_in_ram()?;
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

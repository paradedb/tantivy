use std::io::{self, Write};

use common::{CountingWriter, HasLen};

use crate::directory::{BufferedFileSlice, FileSlice};
use crate::postings::compression::{compressed_block_size, BlockDecoder, VIntDecoder};

const BUFFER_SIZE: usize = 8192;

pub(crate) struct TermNormsWriter<'a, W: Write> {
    write: &'a mut CountingWriter<W>,
    start_offset: u64,
}

impl<'a, W: Write> TermNormsWriter<'a, W> {
    pub(crate) fn new(write: &'a mut CountingWriter<W>) -> Self {
        let start_offset = write.written_bytes();
        Self {
            write,
            start_offset,
        }
    }

    pub(crate) fn write_term(&mut self, packed_blocks: &[u8], tail_vint: &[u8]) -> io::Result<u64> {
        let offset = self.write.written_bytes() - self.start_offset;
        self.write.write_all(packed_blocks)?;
        self.write.write_all(tail_vint)?;
        Ok(offset)
    }

    #[cfg(test)]
    pub(crate) fn write_slice(
        &mut self,
        lengths: &[u32],
    ) -> io::Result<((Vec<(u8, usize)>, usize), u64)> {
        use crate::postings::compression::{BlockEncoder, VIntEncoder, COMPRESSION_BLOCK_SIZE};

        let mut encoder = BlockEncoder::new();
        let mut packed_blocks = Vec::new();
        let mut tail_vint = Vec::new();
        let mut block_specs = Vec::new();
        let mut offset = 0;
        for chunk in lengths.chunks(COMPRESSION_BLOCK_SIZE) {
            if chunk.len() == COMPRESSION_BLOCK_SIZE {
                let (num_bits, block_bytes) = encoder.compress_block_unsorted(chunk, false);
                block_specs.push((num_bits, offset));
                offset += block_bytes.len();
                packed_blocks.extend_from_slice(block_bytes);
            } else {
                let tail_bytes = encoder.compress_vint_unsorted(chunk);
                tail_vint.extend_from_slice(tail_bytes);
            }
        }
        let tail_offset = offset;
        let norm_offset = self.write_term(&packed_blocks, &tail_vint)?;
        Ok(((block_specs, tail_offset), norm_offset))
    }
}

#[derive(Clone)]
pub(crate) struct TermNormReader {
    buffer: Option<BufferedFileSlice>,
    slice_len: usize,
}

impl TermNormReader {
    pub(crate) fn new(source: FileSlice, offset: u64) -> Self {
        // Unscored queries must not fail on a norm stream they never read.
        if offset <= source.len() as u64 {
            let slice = source.slice_from(offset as usize);
            let slice_len = slice.len();
            Self {
                buffer: Some(BufferedFileSlice::new(slice, BUFFER_SIZE)),
                slice_len,
            }
        } else {
            Self {
                buffer: None,
                slice_len: 0,
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            buffer: None,
            slice_len: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn decode_packed_block(
        &self,
        offset: usize,
        num_bits: u8,
        decoder: &mut BlockDecoder,
    ) -> io::Result<()> {
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms"))?;
        let block_size = compressed_block_size(num_bits);
        let end = offset.checked_add(block_size).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "posting norm slice shorter than compressed block",
            )
        })?;
        if end > self.slice_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting norm slice shorter than compressed block",
            ));
        }
        let block_bytes = buffer.get_bytes(offset as u64..end as u64)?;
        decoder.uncompress_block_unsorted(&block_bytes, num_bits, false);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn decode_vint_block(
        &self,
        offset: usize,
        num_docs: usize,
        decoder: &mut BlockDecoder,
    ) -> io::Result<()> {
        if num_docs == 0 {
            return Ok(());
        }
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms"))?;
        if offset >= self.slice_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting norm slice offset out of bounds",
            ));
        }
        let max_vint_len = (num_docs * 5).min(self.slice_len - offset);
        let tail_bytes = buffer.get_bytes(offset as u64..(offset + max_vint_len) as u64)?;
        decoder.uncompress_vint_unsorted(&tail_bytes, num_docs, 0);
        Ok(())
    }

    pub(crate) fn decode_scoring_packed_block(
        &self,
        offset: usize,
        bitwidths: BlockBitwidths,
        freq_decoder: &mut BlockDecoder,
        fieldnorm_decoder: &mut BlockDecoder,
    ) -> io::Result<()> {
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms"))?;
        let tf_size = compressed_block_size(bitwidths.tf);
        let norm_size = compressed_block_size(bitwidths.pnorm);
        let total_size = tf_size.checked_add(norm_size).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "overflow in scoring block size")
        })?;
        let end = offset.checked_add(total_size).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "scoring slice shorter than compressed block",
            )
        })?;
        if end > self.slice_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "scoring slice shorter than compressed block",
            ));
        }
        let bytes = buffer.get_bytes(offset as u64..end as u64)?;
        if bitwidths.tf > 0 {
            freq_decoder.uncompress_block_unsorted(&bytes[..tf_size], bitwidths.tf, true);
        } else {
            freq_decoder.fill_val(1, crate::postings::compression::COMPRESSION_BLOCK_SIZE);
        }
        if bitwidths.pnorm > 0 {
            fieldnorm_decoder.uncompress_block_unsorted(
                &bytes[tf_size..total_size],
                bitwidths.pnorm,
                false,
            );
        } else {
            fieldnorm_decoder.fill_val(0, crate::postings::compression::COMPRESSION_BLOCK_SIZE);
        }
        Ok(())
    }

    pub(crate) fn decode_scoring_vint_block(
        &self,
        offset: usize,
        num_docs: usize,
        has_freq: bool,
        freq_decoder: &mut BlockDecoder,
        fieldnorm_decoder: &mut BlockDecoder,
    ) -> io::Result<()> {
        if num_docs == 0 {
            return Ok(());
        }
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated posting norms"))?;
        if offset >= self.slice_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "scoring slice offset out of bounds",
            ));
        }
        let max_bytes_per_doc = if has_freq { 10 } else { 5 };
        let max_vint_len = (num_docs * max_bytes_per_doc).min(self.slice_len - offset);
        let bytes = buffer.get_bytes(offset as u64..(offset + max_vint_len) as u64)?;
        let tf_consumed = if has_freq {
            freq_decoder.uncompress_vint_unsorted(&bytes, num_docs, 1)
        } else {
            freq_decoder.fill_val(1, num_docs);
            0
        };
        if tf_consumed > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "scoring slice offset out of bounds for fieldnorm vint",
            ));
        }
        let norm_bytes = &bytes[tf_consumed..];
        fieldnorm_decoder.uncompress_vint_unsorted(norm_bytes, num_docs, 0);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockBitwidths {
    pub(crate) tf: u8,
    pub(crate) pnorm: u8,
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

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
        let data: Vec<u32> = (0..30000).map(|i| (i % 251) as u32).collect();
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut writer = TermNormsWriter::new(&mut write);
        writer.write_slice(&data[..10]).unwrap();
        let ((block_specs, tail_offset), norm_offset) =
            writer.write_slice(&data[10..29010]).unwrap();
        writer.write_slice(&data[29010..]).unwrap();
        let file = FileSlice::new(Arc::new(TrackedFile {
            reads: reads.clone(),
            data: bytes,
        }));
        let reader = TermNormReader::new(file.clone(), norm_offset);
        assert!(reads.lock().unwrap().is_empty());
        let mut decoder = BlockDecoder::default();
        // Block 0 (ordinals 0..128)
        let (num_bits, offset) = block_specs[0];
        reader
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(0), 10);
        assert_eq!(decoder.output(127), ((127 + 10) % 251) as u32);
        // Block 62 (ordinals 7936..8064, covers 8000 and 8001)
        let (num_bits, offset) = block_specs[8000 / 128];
        reader
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(8000 % 128), ((8000 + 10) % 251) as u32);
        let clone = reader.clone();
        clone
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(8001 % 128), ((8001 + 10) % 251) as u32);
        // Block 63 (ordinals 8064..8192, covers 8191)
        let (num_bits, offset) = block_specs[8191 / 128];
        reader
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(8191 % 128), ((8191 + 10) % 251) as u32);
        // Block 156 (ordinals 19968..20096, covers 20000)
        let (num_bits, offset) = block_specs[20000 / 128];
        reader
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(20000 % 128), ((20000 + 10) % 251) as u32);
        // Block 220 (ordinals 28160..28288, covers 28192)
        let (num_bits, offset) = block_specs[28192 / 128];
        reader
            .decode_packed_block(offset, num_bits, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(28192 % 128), ((28192 + 10) % 251) as u32);
        // Tail vint (covers ordinals 28928..29000, covers 28999)
        reader
            .decode_vint_block(tail_offset, 72, &mut decoder)
            .unwrap();
        assert_eq!(decoder.output(28999 - 28928), ((28999 + 10) % 251) as u32);
        assert!(reader
            .decode_packed_block(usize::MAX, 1, &mut decoder)
            .is_err());
        assert!(TermNormReader::new(file.clone(), u64::MAX)
            .decode_packed_block(0, 1, &mut decoder)
            .is_err());
        assert!(TermNormReader::new(file, 29999)
            .decode_packed_block(usize::MAX, 1, &mut decoder)
            .is_err());
        let empty = TermNormReader::empty();
        assert!(empty.decode_packed_block(0, 1, &mut decoder).is_err());
        assert!(empty.decode_vint_block(0, 1, &mut decoder).is_err());
    }

    #[test]
    fn direct_norm_read_needs_no_metadata_io() {
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut writer = TermNormsWriter::new(&mut write);
        let mut targets = Vec::new();
        for term in 0..1_000_000 {
            let (_, offset) = writer.write_slice(&[(term % 251) as u32]).unwrap();
            if term == 0 || term == 123_456 || term == 999_999 {
                targets.push((term, offset));
            }
        }
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(TrackedFile {
            reads: reads.clone(),
            data: bytes,
        }));
        let mut decoder = BlockDecoder::default();
        for (term, offset) in targets {
            reads.lock().unwrap().clear();
            let reader = TermNormReader::new(file.clone(), offset);
            assert!(reads.lock().unwrap().is_empty());
            for _ in 0..100 {
                reader.decode_vint_block(0, 1, &mut decoder).unwrap();
                assert_eq!(decoder.output(0), (term % 251) as u32);
            }
            let ranges = reads.lock().unwrap().clone();
            assert_eq!(ranges.len(), 1);
            let range = &ranges[0];
            assert_eq!(range.start, offset as usize);
            assert!(range.len() <= 8192);
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
    fn pnorms_leave_postings_positions_and_fieldnorms_unchanged() -> crate::Result<()> {
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
        for component in [SegmentComponent::Positions, SegmentComponent::FieldNorms] {
            assert_eq!(
                segments[0]
                    .open_read(component.clone())?
                    .read_bytes()?
                    .as_slice(),
                segments[1]
                    .open_read(component.clone())?
                    .read_bytes()?
                    .as_slice(),
                "Component {component:?} should be identical",
            );
        }
        // Postings without pnorms contain [postings][termfreqs].
        // When pnorms are enabled, term frequencies are moved out of Postings (.idx) into
        // PostingNorms (.pnorm), making Postings strictly smaller.
        let postings_without_pnorms = segments[0]
            .open_read(SegmentComponent::Postings)?
            .read_bytes()?;
        let postings_with_pnorms = segments[1]
            .open_read(SegmentComponent::Postings)?
            .read_bytes()?;
        assert!(postings_with_pnorms.len() < postings_without_pnorms.len());
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

    #[test]
    fn pnorms_full_fidelity_preserved_across_merge() -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::indexer::NoMergePolicy;
        use crate::postings::Postings;
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{DocSet, Index, Term, TERMINATED};

        let mut schema = Schema::builder();
        let indexing = TEXT
            .get_indexing_options()
            .unwrap()
            .clone()
            .set_pnorms(true);
        let text = schema.add_text_field("text", TEXT.set_indexing_options(indexing));
        let schema = schema.build();
        let directory = RamDirectory::create();
        let index = Index::create(directory, schema, Default::default())?;
        let mut writer = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // 41 tokens: quantizes to 40 in .fieldnorm table, but .pnorm has full fidelity 41.
        let words_41 = std::iter::repeat("word")
            .take(40)
            .chain(std::iter::once("target"))
            .collect::<Vec<_>>()
            .join(" ");
        // 75 tokens: quantizes to 74 in .fieldnorm table, but .pnorm has full fidelity 75.
        let words_75 = std::iter::repeat("word")
            .take(74)
            .chain(std::iter::once("target"))
            .collect::<Vec<_>>()
            .join(" ");

        let mut doc1 = crate::schema::TantivyDocument::default();
        doc1.add_text(text, &words_41);
        writer.add_document(doc1)?;
        let mut doc2 = crate::schema::TantivyDocument::default();
        doc2.add_text(text, &words_75);
        writer.add_document(doc2)?;
        writer.commit()?;

        let mut doc3 = crate::schema::TantivyDocument::default();
        doc3.add_text(text, &words_41);
        writer.add_document(doc3)?;
        writer.commit()?;

        let reader = index.reader()?;
        let term = Term::from_field_text(text, "target");

        let verify_readers = |reader: &crate::IndexReader| -> crate::Result<()> {
            let searcher = reader.searcher();
            for segment in searcher.segment_readers() {
                let inv = segment.inverted_index(text)?;
                assert!(inv.has_pnorms());
                let norms = segment.get_fieldnorms_reader(text)?;
                let mut postings = inv
                    .read_postings(&term, IndexRecordOption::WithFreqs)?
                    .unwrap();
                while postings.doc() != TERMINATED {
                    let full_norm = postings.fieldnorm().expect("pnorms enabled");
                    let quantized_norm = norms.fieldnorm(postings.doc());
                    assert!(full_norm == 41 || full_norm == 75);
                    if full_norm == 41 {
                        assert_eq!(quantized_norm, 40);
                    } else if full_norm == 75 {
                        assert_eq!(quantized_norm, 72);
                    }
                    postings.advance();
                }
            }
            Ok(())
        };

        verify_readers(&reader)?;

        // Now merge all segments.
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        reader.reload()?;

        assert_eq!(reader.searcher().segment_readers().len(), 1);
        verify_readers(&reader)?;

        Ok(())
    }

    #[test]
    fn scoring_data_lazily_loaded() -> crate::Result<()> {
        use crate::postings::Postings;
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{DocSet, Index, Term, TERMINATED};

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
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        // Index 200 documents so we span across block 0 (128 docs) and tail vint (72 docs)
        for i in 0..200 {
            let freq = (i % 5) + 1;
            let doc_text = format!("{} other", "target ".repeat(freq));
            writer.add_document(doc!(text => doc_text))?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let inv = segment.inverted_index(text)?;
        assert!(inv.has_pnorms());

        let term = Term::from_field_text(text, "target");
        let info = inv.get_term_info(&term)?.unwrap();
        let mut block =
            inv.read_block_postings_from_terminfo(&info, IndexRecordOption::WithFreqs)?;

        // Freshly loaded block has not loaded scoring data yet
        assert!(!block.is_scoring_loaded());

        // Inspecting docs does not load scoring data
        assert_eq!(block.doc(0), 0);
        assert_eq!(block.doc(10), 10);
        assert!(!block.is_scoring_loaded());

        // Accessing freq on doc 0 triggers lazy loading of scoring data
        let freq_0 = block.freq(0);
        assert_eq!(freq_0, 1);
        assert!(block.is_scoring_loaded());

        // Verify freqs across the first block (128 docs)
        for i in 0..128 {
            let expected_freq = ((i % 5) + 1) as u32;
            assert_eq!(block.freq(i), expected_freq);
        }

        // Advance to next block (tail vint block)
        block.advance();
        // In the new block, scoring data is again not loaded yet
        assert!(!block.is_scoring_loaded());

        // Checking doc ID in the tail block does not load scoring data
        assert_eq!(block.doc(0), 128);
        assert!(!block.is_scoring_loaded());

        // Accessing freq loads scoring data for tail block
        assert_eq!(block.freq(0), ((128 % 5) + 1) as u32);
        assert!(block.is_scoring_loaded());

        // Verify all docs in tail block
        for i in 0..block.block_len() {
            let doc_id = block.doc(i);
            let expected_freq = ((doc_id % 5) + 1) as u32;
            assert_eq!(block.freq(i), expected_freq);
        }

        // Also verify SegmentPostings API
        let mut postings = inv
            .read_postings(&term, IndexRecordOption::WithFreqs)?
            .unwrap();
        let mut doc_count = 0;
        while postings.doc() != TERMINATED {
            let doc = postings.doc();
            let expected_freq = ((doc % 5) + 1) as u32;
            assert_eq!(postings.term_freq(), expected_freq);
            doc_count += 1;
            postings.advance();
        }
        assert_eq!(doc_count, 200);

        Ok(())
    }
}

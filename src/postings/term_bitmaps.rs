use std::io::{self, Write};

use common::CountingWriter;

use crate::index::BitmapPostingsConfig;
use crate::{DocId, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW};

pub(crate) const HIERARCHICAL_BITMAP_MAGIC: [u8; 8] = *b"TVBMAP02";
pub(crate) const HIERARCHICAL_BITMAP_HEADER_LEN: usize = 16;
pub(crate) const HIERARCHICAL_BITMAP_ENTRY_LEN: usize = 8;
pub(crate) const SPARSE_BLOCK_MAX_DOCS: usize = BLOCK_NUM_TINYBITSETS * 64 / 16;

pub(crate) fn flat_bitmap_num_bytes(max_doc: DocId) -> u64 {
    u64::from(max_doc).div_ceil(64) * 8
}

pub(crate) fn bitmap_block_count(max_doc: DocId) -> usize {
    max_doc.div_ceil(BLOCK_WINDOW) as usize
}

pub(crate) fn bitmap_summary_num_bytes(max_doc: DocId) -> usize {
    bitmap_block_count(max_doc).div_ceil(64) * 8
}

pub(crate) fn encode_hierarchical_bitmap(
    max_doc: DocId,
    sparse: &[DocId],
    words: &[u64],
) -> Vec<u8> {
    let block_count = bitmap_block_count(max_doc);
    let summary_len = bitmap_summary_num_bytes(max_doc);
    let mut summary = vec![0u64; block_count.div_ceil(64)];
    let mut entries = Vec::new();
    let mut payload = Vec::new();

    let mut encode_block = |block: usize, docs: &[u16], dense_words: Option<&[u64]>| {
        summary[block / 64] |= 1u64 << (block % 64);
        let count = dense_words.map_or(docs.len(), |words| {
            words.iter().map(|word| word.count_ones() as usize).sum()
        });
        let payload_offset = payload.len();
        if count <= SPARSE_BLOCK_MAX_DOCS {
            if let Some(words) = dense_words {
                for (word_ord, mut word) in words.iter().copied().enumerate() {
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        payload.extend_from_slice(&((word_ord * 64 + bit) as u16).to_le_bytes());
                        word &= word - 1;
                    }
                }
            } else {
                for doc in docs {
                    payload.extend_from_slice(&doc.to_le_bytes());
                }
            }
        } else if let Some(words) = dense_words {
            for word in words {
                payload.extend_from_slice(&word.to_le_bytes());
            }
            for _ in words.len()..BLOCK_NUM_TINYBITSETS {
                payload.extend_from_slice(&0u64.to_le_bytes());
            }
        } else {
            let mut block_words = [0u64; BLOCK_NUM_TINYBITSETS];
            for &doc in docs {
                let doc = usize::from(doc);
                block_words[doc / 64] |= 1u64 << (doc % 64);
            }
            for word in block_words {
                payload.extend_from_slice(&word.to_le_bytes());
            }
        }
        entries.push((count as u16, payload_offset as u32));
    };

    if words.is_empty() {
        let mut start = 0;
        while start < sparse.len() {
            let block = sparse[start] / BLOCK_WINDOW;
            let mut end = start + 1;
            while end < sparse.len() && sparse[end] / BLOCK_WINDOW == block {
                end += 1;
            }
            let docs: Vec<u16> = sparse[start..end]
                .iter()
                .map(|doc| (doc % BLOCK_WINDOW) as u16)
                .collect();
            encode_block(block as usize, &docs, None);
            start = end;
        }
    } else {
        for block in 0..block_count {
            let word_start = block * BLOCK_NUM_TINYBITSETS;
            let word_end = (word_start + BLOCK_NUM_TINYBITSETS).min(words.len());
            let block_words = &words[word_start..word_end];
            if block_words.iter().any(|word| *word != 0) {
                encode_block(block, &[], Some(block_words));
            }
        }
    }

    let mut encoded = Vec::with_capacity(
        HIERARCHICAL_BITMAP_HEADER_LEN
            + summary_len
            + entries.len() * HIERARCHICAL_BITMAP_ENTRY_LEN
            + payload.len(),
    );
    encoded.extend_from_slice(&HIERARCHICAL_BITMAP_MAGIC);
    encoded.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    for word in summary {
        encoded.extend_from_slice(&word.to_le_bytes());
    }
    for (count, offset) in entries {
        encoded.extend_from_slice(&count.to_le_bytes());
        encoded.extend_from_slice(&0u16.to_le_bytes());
        encoded.extend_from_slice(&offset.to_le_bytes());
    }
    encoded.extend_from_slice(&payload);
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::CompositeFile;
    use crate::index::SegmentComponent;
    use crate::schema::{IndexRecordOption, Schema, TextFieldIndexing, TEXT};
    use crate::{DocSet, Index, TantivyDocument, Term};

    #[test]
    fn density_and_unknown_frequency() -> io::Result<()> {
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut writer = TermBitmapWriter::new(&mut write, 1024, BitmapPostingsConfig::default());
        for (hint, len, offset) in [
            (127, 127, None),
            (128, 128, Some(0)),
            (0, 200, Some(288)),
            (400, 400, Some(720)),
            (0, 127, None),
            (1024, 1024, None),
            (800, 800, Some(1552)),
        ] {
            writer.new_term(hint);
            for doc in 0..len {
                writer.write_doc(doc);
            }
            assert_eq!(writer.close_term(len)?, offset);
        }
        drop(writer);
        assert_eq!(bytes.len(), 2608);
        assert_eq!(&bytes[..8], &HIERARCHICAL_BITMAP_MAGIC);
        assert_eq!(&bytes[288..296], &HIERARCHICAL_BITMAP_MAGIC);
        assert_eq!(&bytes[720..728], &HIERARCHICAL_BITMAP_MAGIC);
        assert_eq!(&bytes[1552..1560], &HIERARCHICAL_BITMAP_MAGIC);
        Ok(())
    }

    #[test]
    fn bitmap_component_is_optional_and_follows_remapping() -> crate::Result<()> {
        use crate::directory::RamDirectory;
        use crate::indexer::DocIdMapping;
        use crate::IndexSettings;
        for enabled in [false, true] {
            let mut schema = Schema::builder();
            let field = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(
                    TextFieldIndexing::default().set_bitmap_postings(enabled),
                ),
            );
            let directory = RamDirectory::default();
            let mut writer = Index::builder()
                .schema(schema.build())
                .settings(IndexSettings {
                    manual_doc_id_mapping: true,
                    bitmap_postings: BitmapPostingsConfig {
                        min_docs: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .single_segment_index_writer(directory.clone(), 15_000_000)?;
            writer.add_document(doc!(field => "common padding"))?;
            writer.add_document(TantivyDocument::default())?;
            writer.add_document(doc!(field => "common"))?;
            let index = writer
                .finalize_with_doc_id_mapping(&DocIdMapping::new_permutation(vec![1, 2, 0])?)?;
            drop(index);
            let reopened = Index::open(directory)?;
            let searcher = reopened.reader()?.searcher();
            let segment = searcher.segment_reader(0);
            assert_eq!(
                segment.open_read(SegmentComponent::PostingBitmaps).is_ok(),
                enabled
            );
            let info = segment
                .inverted_index(field)?
                .get_term_info(&Term::from_field_text(field, "common"))?
                .unwrap();
            assert_eq!(info.bitmap_offset.is_some(), enabled);
            if let Some(offset) = info.bitmap_offset {
                let file =
                    CompositeFile::open(&segment.open_read(SegmentComponent::PostingBitmaps)?)?
                        .open_read(field)
                        .unwrap();
                let mut bitmap = crate::query::BitmapDocSet::open(
                    file,
                    offset,
                    segment.max_doc(),
                    info.doc_freq,
                    || panic!("hierarchical bitmap navigation should not open postings"),
                )?;
                assert_eq!(bitmap.doc(), 1);
                assert_eq!(bitmap.advance(), 2);
                assert_eq!(bitmap.advance(), crate::TERMINATED);
                assert!(
                    searcher.space_usage()?.segments()[0]
                        .component(SegmentComponent::PostingBitmaps)
                        .total()
                        > common::ByteCount::default()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn bitmap_component_roundtrip_and_merge() -> crate::Result<()> {
        for record in [
            IndexRecordOption::Basic,
            IndexRecordOption::WithFreqsAndPositions,
        ] {
            let mut schema = Schema::builder();
            let field = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(TextFieldIndexing::default().set_index_option(record)),
            );
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests::<TantivyDocument>()?;
            writer.set_merge_policy(Box::new(crate::merge_policy::NoMergePolicy));
            for _ in 0..2 {
                for doc_id in 0..1000 {
                    let text = if doc_id < 200 {
                        "common all"
                    } else if doc_id == 900 {
                        "rare all"
                    } else {
                        "all"
                    };
                    writer.add_document(doc!(field => text))?;
                }
                writer.commit()?;
            }
            for merge in [false, true] {
                if merge {
                    writer.delete_term(Term::from_field_text(field, "rare"));
                    writer.commit()?;
                    writer.merge(&index.searchable_segment_ids()?).wait()?;
                }
                let reader = index.reader()?;
                for segment in reader.searcher().segment_readers() {
                    let inverted = segment.inverted_index(field)?;
                    let common = inverted
                        .get_term_info(&Term::from_field_text(field, "common"))?
                        .unwrap();
                    let all = inverted
                        .get_term_info(&Term::from_field_text(field, "all"))?
                        .unwrap();
                    assert!(common.bitmap_offset.is_some());
                    assert!(all.bitmap_offset.is_none());
                    if let Some(rare) =
                        inverted.get_term_info(&Term::from_field_text(field, "rare"))?
                    {
                        assert!(rare.bitmap_offset.is_none());
                    }
                    let file =
                        CompositeFile::open(&segment.open_read(SegmentComponent::PostingBitmaps)?)?
                            .open_read(field)
                            .unwrap();
                    let mut bitmap = crate::query::BitmapDocSet::open(
                        file,
                        common.bitmap_offset.unwrap(),
                        segment.max_doc(),
                        common.doc_freq,
                        || panic!("hierarchical bitmap navigation should not open postings"),
                    )?;
                    let mut postings =
                        inverted.read_postings_from_terminfo(&common, IndexRecordOption::Basic)?;
                    for doc in 0..segment.max_doc() {
                        let bitmap_matches = bitmap.doc() == doc;
                        assert_eq!(bitmap_matches, postings.doc() == doc);
                        if bitmap_matches {
                            bitmap.advance();
                            postings.advance();
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct TermBitmapWriter<'a, W: Write> {
    write: &'a mut CountingWriter<W>,
    start_offset: u64,
    max_doc: DocId,
    config: BitmapPostingsConfig,
    sparse: Vec<DocId>,
    words: Vec<u64>,
    collecting: bool,
}

impl<'a, W: Write> TermBitmapWriter<'a, W> {
    pub(crate) fn new(
        write: &'a mut CountingWriter<W>,
        max_doc: DocId,
        config: BitmapPostingsConfig,
    ) -> Self {
        Self {
            start_offset: write.written_bytes(),
            write,
            max_doc,
            config,
            sparse: Vec::new(),
            words: Vec::new(),
            collecting: false,
        }
    }

    pub(crate) fn new_term(&mut self, doc_freq: u32) {
        self.sparse.clear();
        self.words.clear();
        self.collecting =
            self.max_doc > 0 && (doc_freq == 0 || self.config.eligible(doc_freq, self.max_doc));
    }

    pub(crate) fn write_doc(&mut self, doc: DocId) {
        if !self.collecting {
            return;
        }
        assert!(doc < self.max_doc);
        if self.words.is_empty() {
            self.sparse.push(doc);
            if (self.sparse.len() as u64) * 4 < flat_bitmap_num_bytes(self.max_doc) {
                return;
            }
            self.words.resize(self.max_doc.div_ceil(64) as usize, 0);
            for doc in self.sparse.drain(..) {
                self.words[(doc / 64) as usize] |= 1u64 << (doc % 64);
            }
        } else {
            self.words[(doc / 64) as usize] |= 1u64 << (doc % 64);
        }
    }

    pub(crate) fn close_term(&mut self, doc_freq: u32) -> io::Result<Option<u64>> {
        if !self.collecting || !self.config.eligible(doc_freq, self.max_doc) {
            return Ok(None);
        }
        let encoded = encode_hierarchical_bitmap(self.max_doc, &self.sparse, &self.words);
        let offset = self.write.written_bytes() - self.start_offset;
        self.write.write_all(&encoded)?;
        Ok(Some(offset))
    }
}

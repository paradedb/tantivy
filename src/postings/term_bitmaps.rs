use std::io::{self, Write};

use common::CountingWriter;

use crate::index::BitmapPostingsConfig;
use crate::DocId;

pub(crate) fn bitmap_num_bytes(max_doc: DocId) -> u64 {
    u64::from(max_doc).div_ceil(64) * 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::CompositeFile;
    use crate::index::SegmentComponent;
    use crate::schema::{IndexRecordOption, Schema, TextFieldIndexing, TEXT};
    use crate::{Index, TantivyDocument, Term};

    #[test]
    fn density_budget_and_unknown_frequency() -> io::Result<()> {
        let mut bytes = Vec::new();
        let mut write = CountingWriter::wrap(&mut bytes);
        let mut budget = 256;
        let mut writer = TermBitmapWriter::new(
            &mut write,
            1024,
            BitmapPostingsConfig::default(),
            &mut budget,
        );
        for (hint, len, offset) in [
            (127, 127, None),
            (128, 128, Some(0)),
            (0, 200, Some(128)),
            (400, 400, None),
        ] {
            writer.new_term(hint);
            for doc in 0..len {
                writer.write_doc(doc);
            }
            assert_eq!(writer.close_term(len)?, offset);
        }
        drop(writer);
        assert_eq!(budget, 0);
        assert_eq!(bytes.len(), 256);
        assert!(bytes[..16].iter().all(|byte| *byte == 255));
        assert!(bytes[16..128].iter().all(|byte| *byte == 0));
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
                let data =
                    CompositeFile::open(&segment.open_read(SegmentComponent::PostingBitmaps)?)?
                        .open_read(field)
                        .unwrap()
                        .read_bytes()?;
                assert_eq!(
                    u64::from_le_bytes(
                        data[offset as usize..offset as usize + 8]
                            .try_into()
                            .unwrap()
                    ),
                    0b110
                );
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
                TEXT.set_indexing_options(
                    TextFieldIndexing::default()
                        .set_index_option(record)
                        .set_bitmap_postings(true),
                ),
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
                    let bytes = file.read_bytes()?;
                    assert_eq!(bytes.len() as u64, bitmap_num_bytes(segment.max_doc()));
                    let mut postings =
                        inverted.read_postings_from_terminfo(&common, IndexRecordOption::Basic)?;
                    use crate::DocSet;
                    for doc in 0..segment.max_doc() {
                        let bit = bytes[doc as usize / 8] & (1 << (doc % 8)) != 0;
                        assert_eq!(bit, postings.doc() == doc);
                        if bit {
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
    remaining: &'a mut u64,
    sparse: Vec<DocId>,
    words: Vec<u64>,
    collecting: bool,
}

impl<'a, W: Write> TermBitmapWriter<'a, W> {
    pub(crate) fn new(
        write: &'a mut CountingWriter<W>,
        max_doc: DocId,
        config: BitmapPostingsConfig,
        remaining: &'a mut u64,
    ) -> Self {
        Self {
            start_offset: write.written_bytes(),
            write,
            max_doc,
            config,
            remaining,
            sparse: Vec::new(),
            words: Vec::new(),
            collecting: false,
        }
    }

    pub(crate) fn new_term(&mut self, doc_freq: u32) {
        self.sparse.clear();
        self.words.clear();
        self.collecting = self.max_doc > 0
            && bitmap_num_bytes(self.max_doc) <= *self.remaining
            && (doc_freq == 0 || self.config.eligible(doc_freq, self.max_doc));
    }

    pub(crate) fn write_doc(&mut self, doc: DocId) {
        if !self.collecting {
            return;
        }
        assert!(doc < self.max_doc);
        if self.words.is_empty() {
            self.sparse.push(doc);
            if (self.sparse.len() as u64) * 4 < bitmap_num_bytes(self.max_doc) {
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
        if self.words.is_empty() {
            self.words.resize(self.max_doc.div_ceil(64) as usize, 0);
            for doc in self.sparse.drain(..) {
                self.words[(doc / 64) as usize] |= 1u64 << (doc % 64);
            }
        }
        let offset = self.write.written_bytes() - self.start_offset;
        for word in &self.words {
            self.write.write_all(&word.to_le_bytes())?;
        }
        *self.remaining -= bitmap_num_bytes(self.max_doc);
        Ok(Some(offset))
    }
}

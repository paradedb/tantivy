use std::cmp::Ordering;
use std::io::{self, Write};

use common::{BinarySerializable, CountingWriter, VInt};

use super::TermInfo;
use crate::directory::{CompositeWrite, WritePtr};
use crate::fieldnorm::FieldNormReader;
use crate::index::{Bm25Params, Segment};
use crate::positions::PositionSerializer;
use crate::postings::compression::{BlockEncoder, VIntEncoder, COMPRESSION_BLOCK_SIZE};
use crate::postings::skip::SkipSerializer;
use crate::query::Bm25Weight;
use crate::schema::{Field, FieldEntry, IndexRecordOption, Schema};
use crate::termdict::TermDictionaryBuilder;
use crate::{DocId, Score};

/// `InvertedIndexSerializer` is in charge of serializing
/// postings on disk, in the
/// * `.idx` (document IDs and skip data)
/// * `.freqs` (term frequencies, when enabled)
/// * `.pnorm` (posting norms, when enabled)
/// * `.pos` (positions file)
/// * `.term` (term dictionary)
///
/// `PostingsWriter` are in charge of pushing the data to the
/// serializer.
///
/// The serializer expects to receive the following calls
/// in this order :
/// * `set_field(...)`
/// * `new_term(...)`
/// * `write_doc(...)`
/// * `write_doc(...)`
/// * `write_doc(...)`
/// * ...
/// * `close_term()`
/// * `new_term(...)`
/// * `write_doc(...)`
/// * ...
/// * `close_term()`
/// * `set_field(...)`
/// * ...
/// * `close()`
///
/// Terms have to be pushed in a lexicographically-sorted order.
/// Within a term, documents have to be pushed in increasing order.
///
/// A description of the serialization format is
/// [available here](https://fulmicoton.gitbooks.io/tantivy-doc/content/inverted-index.html).
pub struct InvertedIndexSerializer {
    terms_write: CompositeWrite<WritePtr>,
    postings_write: CompositeWrite<WritePtr>,
    positions_write: CompositeWrite<WritePtr>,
    schema: Schema,
    pnorms_write: Option<CompositeWrite<WritePtr>>,
    freqs_write: Option<CompositeWrite<WritePtr>>,
}

impl InvertedIndexSerializer {
    /// Open a new `InvertedIndexSerializer` for the given segment
    pub fn open(segment: &Segment) -> crate::Result<InvertedIndexSerializer> {
        use crate::index::SegmentComponent::{Positions, Postings, Terms};
        let inv_index_serializer = InvertedIndexSerializer {
            terms_write: CompositeWrite::wrap(segment.open_write(Terms)?),
            postings_write: CompositeWrite::wrap(segment.open_write(Postings)?),
            positions_write: CompositeWrite::wrap(segment.open_write(Positions)?),
            schema: segment.schema(),
            freqs_write: if segment.schema().fields().any(|(_, entry)| {
                entry
                    .field_type()
                    .index_record_option()
                    .is_some_and(IndexRecordOption::has_freq)
            }) {
                Some(CompositeWrite::wrap(segment.open_write(
                    crate::index::SegmentComponent::TermFrequencies,
                )?))
            } else {
                None
            },
            pnorms_write: if segment
                .schema()
                .fields()
                .any(|(_, entry)| entry.has_pnorms())
            {
                Some(CompositeWrite::wrap(
                    segment.open_write(crate::index::SegmentComponent::PostingNorms)?,
                ))
            } else {
                None
            },
        };
        Ok(inv_index_serializer)
    }

    /// Must be called before starting pushing terms of
    /// a given field.
    ///
    /// Loads the indexing options for the given field.
    pub fn new_field(
        &mut self,
        field: Field,
        total_num_tokens: u64,
        fieldnorm_reader: Option<FieldNormReader>,
    ) -> io::Result<FieldSerializer<'_>> {
        let field_entry: &FieldEntry = self.schema.get_field_entry(field);
        let term_dictionary_write = self.terms_write.for_field(field);
        let postings_write = self.postings_write.for_field(field);
        let positions_write = self.positions_write.for_field(field);
        let index_record_option = field_entry
            .field_type()
            .index_record_option()
            .unwrap_or(IndexRecordOption::Basic);
        let bm25_params = field_entry.field_type().bm25_params().unwrap_or_default();
        let mut serializer = FieldSerializer::create(
            index_record_option,
            total_num_tokens,
            term_dictionary_write,
            postings_write,
            positions_write,
            fieldnorm_reader,
            bm25_params,
        )?;
        if index_record_option.has_freq() {
            let freqs_write = self.freqs_write.as_mut().unwrap().for_field(field);
            serializer.freqs_start_offset = freqs_write.written_bytes();
            serializer.freqs_write = Some(freqs_write);
            serializer.postings_serializer.freqs = Some(Vec::new());
        }
        if let Some(pnorms_write) = self.pnorms_write.as_mut() {
            if field_entry.has_pnorms() {
                serializer.pnorms_writer = Some(super::term_norms::TermNormsWriter::new(
                    pnorms_write.for_field(field),
                ));
                serializer.postings_serializer.pnorms = Some(Vec::new());
            }
        }
        Ok(serializer)
    }

    /// Closes the serializer.
    pub fn close(self) -> io::Result<()> {
        self.terms_write.close()?;
        self.postings_write.close()?;
        self.positions_write.close()?;
        if let Some(freqs_write) = self.freqs_write {
            freqs_write.close()?;
        }
        if let Some(pnorms_write) = self.pnorms_write {
            pnorms_write.close()?;
        }
        Ok(())
    }
}

/// The field serializer is in charge of
/// the serialization of a specific field.
pub struct FieldSerializer<'a, W: Write = WritePtr> {
    term_dictionary_builder: TermDictionaryBuilder<&'a mut CountingWriter<W>>,
    postings_serializer: PostingsSerializer,
    positions_serializer_opt: Option<PositionSerializer<&'a mut CountingWriter<W>>>,
    current_term_info: TermInfo,
    term_open: bool,
    postings_write: &'a mut CountingWriter<W>,
    postings_start_offset: u64,
    pnorms_writer: Option<super::term_norms::TermNormsWriter<'a, W>>,
    freqs_write: Option<&'a mut CountingWriter<W>>,
    freqs_start_offset: u64,
}

impl<'a, W: Write> FieldSerializer<'a, W> {
    /// Creates a new `FieldSerializer` for the given field type.
    pub fn create(
        index_record_option: IndexRecordOption,
        total_num_tokens: u64,
        term_dictionary_write: &'a mut CountingWriter<W>,
        postings_write: &'a mut CountingWriter<W>,
        positions_write: &'a mut CountingWriter<W>,
        fieldnorm_reader: Option<FieldNormReader>,
        bm25_params: Bm25Params,
    ) -> io::Result<FieldSerializer<'a, W>> {
        total_num_tokens.serialize(postings_write)?;
        let term_dictionary_builder = TermDictionaryBuilder::create(term_dictionary_write)?;
        let average_fieldnorm = fieldnorm_reader
            .as_ref()
            .map(|ff_reader| total_num_tokens as Score / ff_reader.num_docs() as Score)
            .unwrap_or(0.0);
        let postings_serializer = PostingsSerializer::new(
            average_fieldnorm,
            index_record_option,
            fieldnorm_reader,
            bm25_params,
        );
        let positions_serializer_opt = if index_record_option.has_positions() {
            Some(PositionSerializer::new(positions_write))
        } else {
            None
        };

        let postings_start_offset = postings_write.written_bytes();
        Ok(FieldSerializer {
            term_dictionary_builder,
            postings_serializer,
            positions_serializer_opt,
            current_term_info: TermInfo::default(),
            term_open: false,
            postings_write,
            postings_start_offset,
            pnorms_writer: None,
            freqs_write: None,
            freqs_start_offset: 0,
        })
    }

    fn postings_offset(&self) -> usize {
        (self.postings_write.written_bytes() - self.postings_start_offset) as usize
    }

    fn current_term_info(&self) -> TermInfo {
        let positions_start =
            if let Some(positions_serializer) = self.positions_serializer_opt.as_ref() {
                positions_serializer.written_bytes()
            } else {
                0u64
            } as usize;
        let addr = self.postings_offset();
        TermInfo {
            doc_freq: 0,
            postings_range: addr..addr,
            positions_range: positions_start..positions_start,
            pnorms_offset: None,
            freqs_range: self.freqs_write.as_ref().map(|writer| {
                let start = (writer.written_bytes() - self.freqs_start_offset) as usize;
                start..start
            }),
        }
    }

    /// Starts the postings for a new term.
    /// * term - the term. It needs to come after the previous term according to the lexicographical
    ///   order.
    /// * term_doc_freq - return the number of document containing the term.
    pub fn new_term(
        &mut self,
        term: &[u8],
        term_doc_freq: u32,
        record_term_freq: bool,
    ) -> io::Result<()> {
        assert!(
            !self.term_open,
            "Called new_term, while the previous term was not closed."
        );
        self.term_open = true;
        self.postings_serializer.clear();
        self.current_term_info = self.current_term_info();
        self.term_dictionary_builder.insert_key(term)?;
        self.postings_serializer
            .new_term(term_doc_freq, record_term_freq);
        Ok(())
    }

    /// Starts the postings for a new term without recording term frequencies.
    pub fn new_term_without_freq(&mut self, term: &[u8]) -> io::Result<()> {
        self.new_term(term, 0, false)
    }

    /// Serialize the information that a document contains for the current term:
    /// its term frequency, and the position deltas.
    ///
    /// At this point, the positions are already `delta-encoded`.
    /// For instance, if the positions are `2, 3, 17`,
    /// `position_deltas` is `2, 1, 14`
    ///
    /// Term frequencies and positions may be ignored by the serializer depending
    /// on the configuration of the field in the `Schema`.
    pub fn write_doc(&mut self, doc_id: DocId, term_freq: u32, position_deltas: &[u32]) {
        self.current_term_info.doc_freq += 1;
        self.postings_serializer.write_doc(doc_id, term_freq);
        if let Some(ref mut positions_serializer) = self.positions_serializer_opt.as_mut() {
            assert_eq!(term_freq as usize, position_deltas.len());
            positions_serializer.write_positions_delta(position_deltas);
        }
    }

    /// Finish the serialization for this term postings.
    ///
    /// If the current block is incomplete, it needs to be encoded
    /// using `VInt` encoding.
    pub fn close_term(&mut self) -> io::Result<()> {
        crate::fail_point!("FieldSerializer::close_term", |msg: Option<String>| {
            Err(io::Error::other(format!("{msg:?}")))
        });

        if !self.term_open {
            return Ok(());
        };

        self.postings_serializer
            .close_term(self.current_term_info.doc_freq, self.postings_write)?;
        if let Some(freqs) = self.postings_serializer.freqs.as_ref() {
            self.freqs_write.as_mut().unwrap().write_all(freqs)?;
            self.current_term_info.freqs_range.as_mut().unwrap().end += freqs.len();
        }
        if let Some(norms) = self.postings_serializer.pnorms.as_ref() {
            assert_eq!(norms.len(), self.current_term_info.doc_freq as usize);
            self.current_term_info.pnorms_offset =
                Some(self.pnorms_writer.as_mut().unwrap().write(norms)?);
        }
        self.current_term_info.postings_range.end = self.postings_offset();
        if let Some(positions_serializer) = self.positions_serializer_opt.as_mut() {
            positions_serializer.close_term()?;
            self.current_term_info.positions_range.end =
                positions_serializer.written_bytes() as usize;
        }
        self.term_dictionary_builder
            .insert_value(&self.current_term_info)?;
        self.term_open = false;
        Ok(())
    }

    /// Closes the current field.
    pub fn close(mut self) -> io::Result<()> {
        self.close_term()?;
        if let Some(positions_serializer) = self.positions_serializer_opt {
            positions_serializer.close()?;
        }
        self.postings_write.flush()?;
        self.term_dictionary_builder.finish()?;
        Ok(())
    }
}

struct Block {
    doc_ids: [DocId; COMPRESSION_BLOCK_SIZE],
    term_freqs: [u32; COMPRESSION_BLOCK_SIZE],
    len: usize,
}

impl Block {
    fn new() -> Self {
        Block {
            doc_ids: [0u32; COMPRESSION_BLOCK_SIZE],
            term_freqs: [0u32; COMPRESSION_BLOCK_SIZE],
            len: 0,
        }
    }

    fn doc_ids(&self) -> &[DocId] {
        &self.doc_ids[..self.len]
    }

    fn term_freqs(&self) -> &[u32] {
        &self.term_freqs[..self.len]
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn append_doc(&mut self, doc: DocId, term_freq: u32) {
        let len = self.len;
        self.doc_ids[len] = doc;
        self.term_freqs[len] = term_freq;
        self.len = len + 1;
    }

    fn is_full(&self) -> bool {
        self.len == COMPRESSION_BLOCK_SIZE
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn last_doc(&self) -> DocId {
        assert_eq!(self.len, COMPRESSION_BLOCK_SIZE);
        self.doc_ids[COMPRESSION_BLOCK_SIZE - 1]
    }
}

/// Serializer for postings lists.
pub struct PostingsSerializer {
    last_doc_id_encoded: u32,

    block_encoder: BlockEncoder,
    block: Box<Block>,

    postings_write: Vec<u8>,
    skip_write: SkipSerializer,

    mode: IndexRecordOption,
    fieldnorm_reader: Option<FieldNormReader>,

    bm25_weight: Option<Bm25Weight>,
    avg_fieldnorm: Score,
    bm25_params: Bm25Params,
    term_has_freq: bool,
    pnorms: Option<Vec<u8>>,
    freqs: Option<Vec<u8>>,
}

impl PostingsSerializer {
    /// Creates a new `PostingsSerializer`.
    /// * avg_fieldnorm - average field norm for the field being serialized.
    /// * mode - indexing options for the field being serialized.
    pub fn new(
        avg_fieldnorm: Score,
        mode: IndexRecordOption,
        fieldnorm_reader: Option<FieldNormReader>,
        bm25_params: Bm25Params,
    ) -> PostingsSerializer {
        PostingsSerializer {
            block_encoder: BlockEncoder::new(),
            block: Box::new(Block::new()),

            postings_write: Vec::new(),
            skip_write: SkipSerializer::new(),

            last_doc_id_encoded: 0u32,
            mode,

            fieldnorm_reader,
            bm25_weight: None,
            avg_fieldnorm,
            bm25_params,
            term_has_freq: false,
            pnorms: None,
            freqs: None,
        }
    }

    /// Starts the serialization for a new term.
    /// * term_doc_freq - the number of documents containing the term.
    pub fn new_term(&mut self, term_doc_freq: u32, record_term_freq: bool) {
        if let Some(freqs) = self.freqs.as_mut() {
            freqs.clear();
        }
        if let Some(norms) = self.pnorms.as_mut() {
            norms.clear();
        }
        self.bm25_weight = None;

        self.term_has_freq = self.mode.has_freq() && record_term_freq;
        if !self.term_has_freq {
            return;
        }

        let num_docs_in_segment: u64 =
            if let Some(fieldnorm_reader) = self.fieldnorm_reader.as_ref() {
                fieldnorm_reader.num_docs() as u64
            } else {
                return;
            };

        if num_docs_in_segment == 0 {
            return;
        }

        self.bm25_weight = Some(Bm25Weight::for_one_term_without_explain(
            term_doc_freq as u64,
            num_docs_in_segment,
            self.avg_fieldnorm,
            self.bm25_params,
        ));
    }

    fn write_block(&mut self) {
        {
            // encode the doc ids
            let (num_bits, block_encoded): (u8, &[u8]) = self
                .block_encoder
                .compress_block_sorted(self.block.doc_ids(), self.last_doc_id_encoded);
            self.last_doc_id_encoded = self.block.last_doc();
            self.skip_write
                .write_doc(self.last_doc_id_encoded, num_bits);
            // last el block 0, offset block 1,
            self.postings_write.extend(block_encoded);
        }
        if self.term_has_freq {
            // encode the term frequencies
            let (num_bits, block_encoded): (u8, &[u8]) = self
                .block_encoder
                .compress_block_unsorted(self.block.term_freqs(), true);
            self.freqs
                .as_mut()
                .unwrap_or(&mut self.postings_write)
                .extend(block_encoded);
            self.skip_write.write_term_freq(num_bits);
            if self.mode.has_positions() {
                // We serialize the sum of term freqs within the skip information
                // in order to navigate through positions.
                let sum_freq = self.block.term_freqs().iter().cloned().sum();
                self.skip_write.write_total_term_freq(sum_freq);
            }
            let mut blockwand_params = (0u8, 0u32);
            if let Some(bm25_weight) = self.bm25_weight.as_ref() {
                if let Some(fieldnorm_reader) = self.fieldnorm_reader.as_ref() {
                    let term_freqs = self.block.term_freqs().iter().cloned();
                    let fieldnorms =
                        self.block
                            .doc_ids()
                            .iter()
                            .enumerate()
                            .map(|(offset, &doc)| match self.pnorms.as_ref() {
                                Some(norms) => norms[norms.len() - self.block.len + offset],
                                None => fieldnorm_reader.fieldnorm_id(doc),
                            });
                    blockwand_params = fieldnorms
                        .zip(term_freqs)
                        .max_by(
                            |(left_fieldnorm_id, left_term_freq),
                             (right_fieldnorm_id, right_term_freq)| {
                                let left_score =
                                    bm25_weight.tf_factor(*left_fieldnorm_id, *left_term_freq);
                                let right_score =
                                    bm25_weight.tf_factor(*right_fieldnorm_id, *right_term_freq);
                                left_score
                                    .partial_cmp(&right_score)
                                    .unwrap_or(Ordering::Equal)
                            },
                        )
                        .unwrap();
                }
            }
            let (fieldnorm_id, term_freq) = blockwand_params;
            self.skip_write.write_blockwand_max(fieldnorm_id, term_freq);
        }
        self.block.clear();
    }

    /// Register that the given document contains the current term.
    /// * doc_id - the document id.
    /// * term_freq - the term frequency within the document.
    pub fn write_doc(&mut self, doc_id: DocId, term_freq: u32) {
        if let Some(norms) = self.pnorms.as_mut() {
            norms.push(self.fieldnorm_reader.as_ref().unwrap().fieldnorm_id(doc_id));
        }
        self.block.append_doc(doc_id, term_freq);
        if self.block.is_full() {
            self.write_block();
        }
    }

    /// Finish the serialization for this term.
    pub fn close_term(
        &mut self,
        doc_freq: u32,
        output_write: &mut impl std::io::Write,
    ) -> io::Result<()> {
        if !self.block.is_empty() {
            // we have doc ids waiting to be written
            // this happens when the number of doc ids is
            // not a perfect multiple of our block size.
            //
            // In that case, the remaining part is encoded
            // using variable int encoding.
            {
                let block_encoded = self
                    .block_encoder
                    .compress_vint_sorted(self.block.doc_ids(), self.last_doc_id_encoded);
                self.postings_write.write_all(block_encoded)?;
            }
            // ... Idem for term frequencies
            if self.term_has_freq {
                let block_encoded = self
                    .block_encoder
                    .compress_vint_unsorted(self.block.term_freqs());
                self.freqs
                    .as_mut()
                    .unwrap_or(&mut self.postings_write)
                    .write_all(block_encoded)?;
            }
            self.block.clear();
        }
        if doc_freq >= COMPRESSION_BLOCK_SIZE as u32 {
            let skip_data = self.skip_write.data();
            VInt(skip_data.len() as u64).serialize(output_write)?;
            output_write.write_all(skip_data)?;
        }
        output_write.write_all(&self.postings_write[..])?;
        self.skip_write.clear();
        self.postings_write.clear();
        self.bm25_weight = None;
        Ok(())
    }

    fn clear(&mut self) {
        self.block.clear();
        self.last_doc_id_encoded = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{CompositeFile, Directory, FileSlice, OwnedBytes, RamDirectory};
    use crate::index::SegmentComponent;
    use crate::indexer::NoMergePolicy;
    use crate::postings::{BlockSegmentPostings, Postings};
    use crate::schema::TEXT;
    use crate::{DocSet, Index, Term, TERMINATED};

    #[test]
    fn separated_freqs_preserve_codecs_and_skip_data() -> crate::Result<()> {
        for count in [1, 127, 128, 129, 255, 256, 257, 1024] {
            for mode in [
                IndexRecordOption::WithFreqs,
                IndexRecordOption::WithFreqsAndPositions,
            ] {
                for max_freq in [1, 17, 257] {
                    let norms = FieldNormReader::constant(count * 3, 400);
                    let mut encoded = Vec::new();
                    for separated in [false, true] {
                        let mut serializer = PostingsSerializer::new(
                            400.0,
                            mode,
                            Some(norms.clone()),
                            Bm25Params::default(),
                        );
                        serializer.pnorms = Some(Vec::new());
                        serializer.freqs = separated.then(Vec::new);
                        serializer.new_term(count, true);
                        for doc in 0..count {
                            serializer.write_doc(doc * 3, doc % max_freq + 1);
                        }
                        let mut bytes = Vec::new();
                        serializer.close_term(count, &mut bytes)?;
                        assert_eq!(
                            serializer.pnorms.unwrap(),
                            vec![norms.fieldnorm_id(0); count as usize]
                        );
                        encoded.push((
                            OwnedBytes::new(bytes),
                            serializer.freqs.map(OwnedBytes::new),
                        ));
                    }
                    let (legacy, _) = &encoded[0];
                    let (docs, freqs) = &encoded[1];
                    assert_eq!(legacy.len(), docs.len() + freqs.as_ref().unwrap().len());
                    if count >= COMPRESSION_BLOCK_SIZE as u32 {
                        let mut legacy = legacy.clone();
                        let mut docs = docs.clone();
                        let skip_len = VInt::deserialize_u64(&mut legacy)? as usize;
                        assert_eq!(VInt::deserialize_u64(&mut docs)? as usize, skip_len);
                        assert_eq!(&legacy[..skip_len], &docs[..skip_len]);
                    }
                    let mut old =
                        BlockSegmentPostings::open(count, legacy.clone(), None, mode, mode)?;
                    let mut split =
                        BlockSegmentPostings::open(count, docs.clone(), freqs.clone(), mode, mode)?;
                    loop {
                        assert_eq!(old.docs(), split.docs());
                        assert_eq!(old.freqs(), split.freqs());
                        if old.docs().is_empty() {
                            break;
                        }
                        old.advance();
                        split.advance();
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn legacy_segments_merge_into_separate_freqs() -> crate::Result<()> {
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
            let directory = RamDirectory::create();
            let index = Index::create(directory.clone(), schema.build(), Default::default())?;
            {
                let mut writer = index.writer_for_tests()?;
                for _ in 0..257 {
                    writer.add_document(doc!(text => "x x y"))?;
                }
                writer.commit()?;
            }
            let segment = index.searchable_segments()?.pop().unwrap();
            for component in [
                SegmentComponent::Terms,
                SegmentComponent::Postings,
                SegmentComponent::Positions,
            ] {
                directory.delete(&segment.relative_path(component)).unwrap();
            }
            if pnorms {
                directory
                    .delete(&segment.relative_path(SegmentComponent::PostingNorms))
                    .unwrap();
            }
            let norms = FieldNormReader::constant(257, 3);
            let mut terms = CompositeWrite::wrap(segment.open_write(SegmentComponent::Terms)?);
            let mut postings =
                CompositeWrite::wrap(segment.open_write(SegmentComponent::Postings)?);
            let mut positions =
                CompositeWrite::wrap(segment.open_write(SegmentComponent::Positions)?);
            let mut norm_file = if pnorms {
                Some(CompositeWrite::wrap(
                    segment.open_write(SegmentComponent::PostingNorms)?,
                ))
            } else {
                None
            };
            let mut serializer = FieldSerializer::create(
                IndexRecordOption::WithFreqsAndPositions,
                257 * 3,
                terms.for_field(text),
                postings.for_field(text),
                positions.for_field(text),
                Some(norms),
                Bm25Params::default(),
            )?;
            if let Some(norm_file) = &mut norm_file {
                serializer.pnorms_writer = Some(super::super::term_norms::TermNormsWriter::new(
                    norm_file.for_field(text),
                ));
                serializer.postings_serializer.pnorms = Some(Vec::new());
            }
            for (term, deltas) in [("x", &[0, 1][..]), ("y", &[2][..])] {
                serializer.new_term(
                    Term::from_field_text(text, term).serialized_value_bytes(),
                    257,
                    true,
                )?;
                for doc in 0..257 {
                    serializer.write_doc(doc, deltas.len() as u32, deltas);
                }
                serializer.close_term()?;
            }
            serializer.close()?;
            terms.close()?;
            postings.close()?;
            positions.close()?;
            if let Some(norm_file) = norm_file {
                norm_file.close()?;
            }
            directory
                .delete(&segment.relative_path(SegmentComponent::TermFrequencies))
                .unwrap();
            let index = Index::open(directory.clone())?;
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for _ in 0..129 {
                writer.add_document(doc!(text => "x x y"))?;
            }
            writer.commit()?;
            let reader = index.reader()?;
            for merged in [false, true] {
                if merged {
                    writer.merge(&index.searchable_segment_ids()?).wait()?;
                    reader.reload()?;
                }
                let searcher = reader.searcher();
                let mut total = 0;
                for segment in searcher.segment_readers() {
                    let inverted = segment.inverted_index(text)?;
                    let term = Term::from_field_text(text, "x");
                    let info = inverted.get_term_info(&term)?.unwrap();
                    #[cfg(feature = "quickwit")]
                    {
                        futures::executor::block_on(inverted.warm_postings(&term, true))?;
                        futures::executor::block_on(inverted.warm_postings_full(true))?;
                    }
                    if merged {
                        assert!(info.freqs_range.is_some());
                    }
                    let mut postings = inverted
                        .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)?
                        .unwrap();
                    let mut positions = Vec::new();
                    while postings.doc() != TERMINATED {
                        assert_eq!(postings.term_freq(), 2);
                        postings.positions(&mut positions);
                        assert_eq!(positions, [0, 1]);
                        total += 1;
                        postings.advance();
                    }
                }
                assert_eq!(total, 386);
            }
            writer.garbage_collect_files().wait()?;
            assert!(index.validate_checksum()?.is_empty());
            let segment = index.searchable_segments()?.pop().unwrap();
            assert!(directory.exists(&segment.relative_path(SegmentComponent::TermFrequencies))?);
            assert!(index.load_metas()?.persisted_custom_extensions.is_empty());
        }
        Ok(())
    }

    #[test]
    fn freqs_follow_field_options_and_are_not_opened_for_basic_reads() -> crate::Result<()> {
        use crate::schema::{TextFieldIndexing, TextOptions};
        for enabled in [false, true] {
            let mut schema = Schema::builder();
            let basic = schema.add_text_field(
                "basic",
                TEXT.set_indexing_options(
                    TextFieldIndexing::default().set_index_option(IndexRecordOption::Basic),
                ),
            );
            let freqs = enabled.then(|| {
                schema.add_text_field(
                    "freqs",
                    TextOptions::default().set_indexing_options(
                        TextFieldIndexing::default()
                            .set_fieldnorms(false)
                            .set_index_option(IndexRecordOption::WithFreqs),
                    ),
                )
            });
            let directory = RamDirectory::create();
            let index = Index::create(directory.clone(), schema.build(), Default::default())?;
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for _ in 0..2 {
                for _ in 0..129 {
                    let mut doc = doc!(basic => "x x");
                    if let Some(freqs) = freqs {
                        doc.add_text(freqs, "x x");
                    }
                    writer.add_document(doc)?;
                }
                writer.commit()?;
            }
            writer.merge(&index.searchable_segment_ids()?).wait()?;
            writer.garbage_collect_files().wait()?;
            let index = Index::open(directory.clone())?;
            let reader = index.reader()?;
            let searcher = reader.searcher();
            let segment = searcher.segment_reader(0);
            assert_eq!(
                segment.open_read(SegmentComponent::TermFrequencies).is_ok(),
                enabled
            );
            if let Some(freqs) = freqs {
                let component =
                    CompositeFile::open(&segment.open_read(SegmentComponent::TermFrequencies)?)?;
                assert!(component.open_read(basic).is_none());
                assert!(component.open_read(freqs).is_some());
                let mut inverted = crate::index::InvertedIndexReader::new(
                    crate::termdict::TermDictionary::open(
                        CompositeFile::open(&segment.open_read(SegmentComponent::Terms)?)?
                            .open_read(freqs)
                            .unwrap(),
                    )?,
                    CompositeFile::open(&segment.open_read(SegmentComponent::Postings)?)?
                        .open_read(freqs)
                        .unwrap(),
                    common::file_slice::DeferredFileSlice::new(|| Ok(FileSlice::empty())),
                    IndexRecordOption::WithFreqs,
                )?;
                let term = Term::from_field_text(freqs, "x");
                let error_opener = || Err(io::Error::other("frequency component was opened"));
                inverted.set_freqs_file(common::file_slice::DeferredFileSlice::new(error_opener));
                let mut docs = inverted
                    .read_postings(&term, IndexRecordOption::Basic)?
                    .unwrap();
                for doc in 0..258 {
                    assert_eq!(docs.doc(), doc);
                    docs.advance();
                }
                assert_eq!(docs.doc(), TERMINATED);
                assert!(inverted
                    .read_postings(&term, IndexRecordOption::WithFreqs)
                    .is_err());
                let inverted = segment.inverted_index(freqs)?;
                let usage = searcher.space_usage()?;
                assert!(
                    usage.segments()[0]
                        .component(SegmentComponent::TermFrequencies)
                        .total()
                        .get_bytes()
                        > 0
                );
                directory
                    .delete(
                        &index
                            .searchable_segments()?
                            .pop()
                            .unwrap()
                            .relative_path(SegmentComponent::TermFrequencies),
                    )
                    .unwrap();
                let postings = inverted
                    .read_postings(&term, IndexRecordOption::WithFreqs)?
                    .unwrap();
                assert_eq!(postings.term_freq(), 2);
                let reopened = Index::open(directory.clone())?.reader()?.searcher();
                let inverted = reopened.segment_reader(0).inverted_index(freqs)?;
                assert!(inverted
                    .read_postings(&term, IndexRecordOption::Basic)?
                    .is_some());
                assert!(inverted
                    .read_postings(&term, IndexRecordOption::WithFreqs)
                    .is_err());
            }
        }
        Ok(())
    }
}

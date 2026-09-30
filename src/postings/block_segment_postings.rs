use std::io;

use common::{HasLen, VInt};
use once_cell::unsync::OnceCell;

use crate::directory::{BufferedFileSlice, FileSlice, OwnedBytes};
use crate::fieldnorm::FieldNormReader;
use crate::postings::compression::{
    compressed_block_size, BlockDecoder, VIntDecoder, COMPRESSION_BLOCK_SIZE,
};
use crate::postings::{BlockInfo, FreqReadingOption, SkipReader};
use crate::query::Bm25Weight;
use crate::schema::IndexRecordOption;
use crate::{DocId, Score, TERMINATED};

pub(crate) fn max_score<I: Iterator<Item = Score>>(mut it: I) -> Option<Score> {
    it.next().map(|first| it.fold(first, Score::max))
}

/// `BlockSegmentPostings` is a cursor iterating over blocks
/// of documents.
///
/// # Warning
///
/// While it is useful for some very specific high-performance
/// use cases, you should prefer using `SegmentPostings` for most usage.
#[derive(Clone)]
pub struct BlockSegmentPostings {
    pub(crate) doc_decoder: BlockDecoder,
    block_loaded: bool,
    freq_decoder: OnceCell<BlockDecoder>,
    freq_reading_option: FreqReadingOption,
    record_option: IndexRecordOption,
    requested_option: IndexRecordOption,
    block_max_score_cache: Option<Score>,
    doc_freq: u32,
    data: PostingData,
    freqs_data: Option<PostingData>,
    skip_reader: SkipReader,
    term_norms: Option<super::term_norms::TermNormReader>,
}

const POSTINGS_BUFFER_SIZE: usize = 1024;

#[derive(Clone)]
enum PostingData {
    Eager(OwnedBytes),
    Buffered(BufferedFileSlice, usize),
}

impl PostingData {
    fn buffered(file: FileSlice) -> Self {
        let len = file.len();
        Self::Buffered(BufferedFileSlice::new(file, POSTINGS_BUFFER_SIZE), len)
    }

    fn with_bytes<T>(
        &self,
        range: std::ops::Range<usize>,
        consume: impl FnOnce(&[u8]) -> T,
    ) -> io::Result<T> {
        if range.is_empty() {
            return Ok(consume(&[]));
        }
        match self {
            Self::Eager(bytes) => Ok(consume(&bytes[range])),
            Self::Buffered(buffer, _) => {
                let bytes = buffer.get_bytes(range.start as u64..range.end as u64)?;
                Ok(consume(&bytes))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Eager(bytes) => bytes.len(),
            Self::Buffered(_, len) => *len,
        }
    }
}

pub(crate) fn decode_bitpacked_block(
    doc_decoder: &mut BlockDecoder,
    freq_decoder_opt: Option<&mut BlockDecoder>,
    data: &[u8],
    doc_offset: DocId,
    doc_num_bits: u8,
    tf_num_bits: u8,
    strict_delta: bool,
) {
    let num_consumed_bytes =
        doc_decoder.uncompress_block_sorted(data, doc_offset, doc_num_bits, strict_delta);
    if let Some(freq_decoder) = freq_decoder_opt {
        freq_decoder.uncompress_block_unsorted(
            &data[num_consumed_bytes..],
            tf_num_bits,
            strict_delta,
        );
    }
}

pub(crate) fn decode_vint_block(
    doc_decoder: &mut BlockDecoder,
    freq_decoder_opt: Option<&mut BlockDecoder>,
    data: &[u8],
    doc_offset: DocId,
    num_vint_docs: usize,
) {
    let num_consumed_bytes =
        doc_decoder.uncompress_vint_sorted(data, doc_offset, num_vint_docs, TERMINATED);
    if let Some(freq_decoder) = freq_decoder_opt {
        // if it's a json term with freq, containing less than 256 docs, we can reach here thinking
        // we have a freq, despite not really having one.
        if data.len() > num_consumed_bytes {
            freq_decoder.uncompress_vint_unsorted(
                &data[num_consumed_bytes..],
                num_vint_docs,
                TERMINATED,
            );
        }
    }
}

fn split_into_skips_and_postings(
    doc_freq: u32,
    mut bytes: OwnedBytes,
) -> io::Result<(Option<OwnedBytes>, OwnedBytes)> {
    if doc_freq < COMPRESSION_BLOCK_SIZE as u32 {
        return Ok((None, bytes));
    }
    let skip_len = VInt::deserialize_u64(&mut bytes)? as usize;
    let (skip_data, postings_data) = bytes.split(skip_len);
    Ok((Some(skip_data), postings_data))
}

impl BlockSegmentPostings {
    /// Opens a `BlockSegmentPostings`.
    /// `doc_freq` is the number of documents in the posting list.
    /// `record_option` represents the amount of data available according to the schema.
    /// `requested_option` is the amount of data requested by the user.
    /// If for instance, we do not request for term frequencies, this function will not decompress
    /// term frequency blocks.
    pub(crate) fn open(
        doc_freq: u32,
        bytes: OwnedBytes,
        freqs: Option<OwnedBytes>,
        record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> io::Result<BlockSegmentPostings> {
        let (skip_data_opt, postings_data) = split_into_skips_and_postings(doc_freq, bytes)?;
        Self::from_parts(
            doc_freq,
            skip_data_opt,
            PostingData::Eager(postings_data),
            freqs.map(PostingData::Eager),
            record_option,
            requested_option,
        )
    }

    pub(crate) fn open_file_slice(
        doc_freq: u32,
        file: FileSlice,
        freqs: Option<FileSlice>,
        record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> io::Result<Self> {
        if file.storage_block_len().is_none()
            || file.len() <= POSTINGS_BUFFER_SIZE
            || doc_freq < COMPRESSION_BLOCK_SIZE as u32
        {
            let (skips, postings) = split_into_skips_and_postings(doc_freq, file.read_bytes()?)?;
            return Self::from_parts(
                doc_freq,
                skips,
                PostingData::Eager(postings),
                freqs.map(PostingData::buffered),
                record_option,
                requested_option,
            );
        }
        let header = file.read_bytes_slice(0..file.len().min(10))?;
        let (skip_len, header_len) = VInt::deserialize_with_size(&mut header.as_slice())?;
        let skip_len = usize::try_from(skip_len.0)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "postings skips too large"))?;
        let postings_start = header_len
            .checked_add(skip_len)
            .filter(|&end| end <= file.len())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "truncated postings skips")
            })?;
        let skips = file.read_bytes_slice(header_len..postings_start)?;
        let len = file.len() - postings_start;
        let buffer = BufferedFileSlice::new(file.slice_from(postings_start), POSTINGS_BUFFER_SIZE);
        Self::from_parts(
            doc_freq,
            Some(skips),
            PostingData::Buffered(buffer, len),
            freqs.map(PostingData::buffered),
            record_option,
            requested_option,
        )
    }

    fn from_parts(
        doc_freq: u32,
        skip_data_opt: Option<OwnedBytes>,
        data: PostingData,
        freqs_data: Option<PostingData>,
        mut record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> io::Result<Self> {
        let schema_record_option = record_option;
        let skip_reader = match skip_data_opt {
            Some(skip_data) => {
                let block_count = doc_freq as usize / COMPRESSION_BLOCK_SIZE;
                // 8 is the minimum size of a block with frequency (can be more if pos are stored
                // too)
                if skip_data.len() < 8 * block_count {
                    // the field might be encoded with frequency, but this term in particular isn't.
                    // This can happen for JSON field with term frequencies:
                    // - text terms are encoded with term freqs.
                    // - numerical terms are encoded without term freqs.
                    record_option = IndexRecordOption::Basic;
                }
                SkipReader::new(skip_data, doc_freq, record_option, freqs_data.is_some())
            }
            None => SkipReader::new(
                OwnedBytes::empty(),
                doc_freq,
                record_option,
                freqs_data.is_some(),
            ),
        };

        let freq_reading_option = match (record_option, requested_option) {
            (IndexRecordOption::Basic, _) => FreqReadingOption::NoFreq,
            (_, IndexRecordOption::Basic) => FreqReadingOption::SkipFreq,
            (_, _) => FreqReadingOption::ReadFreq,
        };

        let mut block_segment_postings = BlockSegmentPostings {
            doc_decoder: BlockDecoder::with_val(TERMINATED),
            block_loaded: false,
            freq_decoder: OnceCell::from(BlockDecoder::with_val(1)),
            freq_reading_option,
            record_option: schema_record_option,
            requested_option,
            block_max_score_cache: None,
            doc_freq,
            data,
            freqs_data,
            skip_reader,
            term_norms: None,
        };
        block_segment_postings.try_load_block()?;
        Ok(block_segment_postings)
    }

    /// Returns the block_max_score for the current block.
    /// It does not require the block to be loaded. For instance, it is ok to call this method
    /// after having called `.shallow_advance(..)`.
    ///
    /// See `TermScorer::block_max_score(..)` for more information.
    pub fn block_max_score(
        &mut self,
        fieldnorm_reader: &FieldNormReader,
        bm25_weight: &Bm25Weight,
    ) -> Score {
        if let Some(score) = self.block_max_score_cache {
            return score;
        }
        if let Some(skip_reader_max_score) = self.skip_reader.block_max_score(bm25_weight) {
            // if we are on a full block, the skip reader should have the block max information
            // for us
            self.block_max_score_cache = Some(skip_reader_max_score);
            return skip_reader_max_score;
        }
        // this is the last block of the segment posting list.
        // If it is actually loaded, we can compute block max manually.
        if self.block_is_loaded() {
            let docs = self.doc_decoder.output_array().iter().cloned();
            let freqs = self.freq_decoder().output_array().iter().cloned();
            let bm25_scores = docs.zip(freqs).enumerate().map(|(offset, (_, term_freq))| {
                let fieldnorm_id = self.fieldnorm_id_at(offset, fieldnorm_reader);
                bm25_weight.score(fieldnorm_id, term_freq)
            });
            let block_max_score = max_score(bm25_scores).unwrap_or(0.0);
            self.block_max_score_cache = Some(block_max_score);
            return block_max_score;
        }
        // We do not have access to any good block max value. We return bm25_weight.max_score()
        // as it is a valid upperbound.
        //
        // We do not cache it however, so that it gets computed when once block is loaded.
        bm25_weight.max_score()
    }

    pub(crate) fn freq_reading_option(&self) -> FreqReadingOption {
        self.freq_reading_option
    }

    pub(crate) fn requested_option(&self) -> IndexRecordOption {
        self.requested_option
    }

    pub(crate) fn set_term_norm_source(
        &mut self,
        source: Option<FileSlice>,
        norm_offset: Option<u64>,
    ) {
        self.term_norms = source.zip(norm_offset).map(|(source, offset)| {
            super::term_norms::TermNormReader::new(source, offset, self.doc_freq)
        });
    }

    pub(crate) fn disable_term_norms(&mut self) {
        self.term_norms = None;
    }

    #[inline]
    pub(crate) fn fieldnorm_id_at(&self, offset: usize, fallback: &FieldNormReader) -> u8 {
        self.posting_fieldnorm_id_at(offset)
            .unwrap_or_else(|| fallback.fieldnorm_id(self.doc(offset)))
    }

    #[inline]
    pub(crate) fn posting_fieldnorm_id_at(&self, offset: usize) -> Option<u8> {
        self.term_norms.as_ref().map(|norms| {
            let ordinal = (self.doc_freq - self.skip_reader.remaining_docs()) as usize + offset;
            norms
                .read(ordinal)
                .expect("failed to read posting fieldnorm")
        })
    }

    // Resets the block segment postings on another position
    // in the postings file.
    //
    // This is useful for enumerating through a list of terms,
    // and consuming the associated posting lists while avoiding
    // reallocating a `BlockSegmentPostings`.
    //
    // # Warning
    //
    // This does not reset the positions list.
    pub(crate) fn reset(
        &mut self,
        doc_freq: u32,
        postings_data: OwnedBytes,
        freqs: Option<FileSlice>,
    ) -> io::Result<()> {
        self.term_norms = None;
        let (skip_data_opt, postings_data) =
            split_into_skips_and_postings(doc_freq, postings_data)?;
        self.data = PostingData::Eager(postings_data);
        self.freqs_data = freqs.map(PostingData::buffered);
        self.block_max_score_cache = None;
        self.block_loaded = false;
        let skip_data = skip_data_opt.unwrap_or_else(OwnedBytes::empty);
        let record_option = if doc_freq >= COMPRESSION_BLOCK_SIZE as u32
            && skip_data.len() < 8 * (doc_freq as usize / COMPRESSION_BLOCK_SIZE)
        {
            IndexRecordOption::Basic
        } else {
            self.record_option
        };
        self.skip_reader.reset(
            skip_data,
            doc_freq,
            record_option,
            self.freqs_data.is_some(),
        );
        self.freq_reading_option = match (record_option, self.requested_option) {
            (IndexRecordOption::Basic, _) => FreqReadingOption::NoFreq,
            (_, IndexRecordOption::Basic) => FreqReadingOption::SkipFreq,
            _ => FreqReadingOption::ReadFreq,
        };
        self.freq_decoder = OnceCell::from(BlockDecoder::with_val(1));
        self.doc_freq = doc_freq;
        self.try_load_block()
    }

    /// Returns the overall number of documents in the block postings.
    /// It does not take in account whether documents are deleted or not.
    ///
    /// This `doc_freq` is simply the sum of the length of all of the blocks
    /// length, and it does not take in account deleted documents.
    pub fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Returns the array of docs in the current block.
    ///
    /// Before the first call to `.advance()`, the block
    /// returned by `.docs()` is empty.
    #[inline]
    pub fn docs(&self) -> &[DocId] {
        debug_assert!(self.block_is_loaded());
        self.doc_decoder.output_array()
    }

    /// Return the document at index `idx` of the block.
    #[inline]
    pub fn doc(&self, idx: usize) -> u32 {
        self.doc_decoder.output(idx)
    }

    /// Return the array of `term freq` in the block.
    #[inline]
    pub fn freqs(&self) -> &[u32] {
        debug_assert!(self.block_is_loaded());
        self.freq_decoder().output_array()
    }

    /// Return the frequency at index `idx` of the block.
    #[inline]
    pub fn freq(&self, idx: usize) -> u32 {
        debug_assert!(self.block_is_loaded());
        self.freq_decoder().output(idx)
    }

    #[inline]
    fn freq_decoder(&self) -> &BlockDecoder {
        self.freq_decoder
            .get_or_try_init(|| {
                let mut decoder = BlockDecoder::with_val(1);
                if self.freq_reading_option == FreqReadingOption::ReadFreq {
                    if let Some(freqs_data) = &self.freqs_data {
                        let start = self.skip_reader.freq_byte_offset();
                        match self.skip_reader.block_info() {
                            BlockInfo::BitPacked {
                                tf_num_bits,
                                strict_delta_encoded,
                                ..
                            } => {
                                freqs_data.with_bytes(
                                    start..start + compressed_block_size(tf_num_bits),
                                    |bytes| {
                                        decoder.uncompress_block_unsorted(
                                            bytes,
                                            tf_num_bits,
                                            strict_delta_encoded,
                                        );
                                    },
                                )?;
                            }
                            BlockInfo::VInt { num_docs }
                                if num_docs > 0 && start < freqs_data.len() =>
                            {
                                freqs_data.with_bytes(start..freqs_data.len(), |bytes| {
                                    decoder.uncompress_vint_unsorted(
                                        bytes,
                                        num_docs as usize,
                                        TERMINATED,
                                    );
                                })?;
                            }
                            BlockInfo::VInt { .. } => {}
                        }
                    }
                }
                Ok::<_, io::Error>(decoder)
            })
            .expect("term frequencies became unreadable after the reader was opened")
    }

    /// Returns the length of the current block.
    ///
    /// Returns the decoded term-frequency buffer for the current block.
    #[inline]
    pub(crate) fn freq_output_array(&self) -> &[u32] {
        self.freq_decoder().output_array()
    }

    /// All blocks have a length of `NUM_DOCS_PER_BLOCK`,
    /// except the last block that may have a length
    /// of any number between 1 and `NUM_DOCS_PER_BLOCK - 1`
    #[inline]
    pub fn block_len(&self) -> usize {
        debug_assert!(self.block_is_loaded());
        self.doc_decoder.output_len
    }

    /// Position on a block that may contains `target_doc`.
    ///
    /// If all docs are smaller than target, the block loaded may be empty,
    /// or be the last an incomplete VInt block.
    pub fn seek(&mut self, target_doc: DocId) -> usize {
        // Move to the block that might contain our document.
        self.seek_block(target_doc);
        self.load_block();

        // At this point we are on the block that might contain our document.
        let doc = self.doc_decoder.seek_within_block(target_doc);

        // The last block is not full and padded with TERMINATED,
        // so we are guaranteed to have at least one value (real or padding)
        // that is >= target_doc.
        debug_assert!(doc < COMPRESSION_BLOCK_SIZE);

        // `doc` is now the first element >= `target_doc`.
        // If all docs are smaller than target, the current block is incomplete and padded
        // with TERMINATED. After the search, the cursor points to the first TERMINATED.
        doc
    }

    /// Returns the number of documents with a doc id strictly smaller than `target`
    /// (i.e. the *rank* of `target` in this posting list).
    ///
    /// This jumps to the block that may contain `target` through the skip list, so no
    /// skipped block is decoded; a single block is then decoded to locate `target`
    /// within it. The cost is therefore `O(number_of_skip_list_entries)` plus one block
    /// decode, rather than `O(doc_freq)`.
    ///
    /// Like [`Self::seek`], the underlying cursor only ever moves forward. This method
    /// must be called with **non-decreasing** `target` values (galloping); calling it
    /// with a `target` smaller than a previous one yields an incorrect result. `target`
    /// must be a valid doc id (i.e. `target <= TERMINATED`), exactly as for `seek`.
    ///
    /// Edge cases: returns `0` when `target` is smaller than every doc id, and
    /// `doc_freq()` when `target` is larger than every doc id.
    pub fn rank(&mut self, target: DocId) -> u32 {
        if self.doc_freq == 0 {
            return 0;
        }
        // `within` = number of docs in the landed block with a doc id < target.
        let within = self.seek(target);
        // `remaining_docs` counts the landed block and everything after it, so the
        // difference is the number of docs in all blocks strictly before it.
        let docs_before_block = self.doc_freq - self.skip_reader.remaining_docs();
        docs_before_block + within as u32
    }

    pub(crate) fn position_offset(&self) -> u64 {
        self.skip_reader.position_offset()
    }

    /// Dangerous API! This calls seeks the next block on the skip list,
    /// but does not `.load_block()` afterwards.
    ///
    /// `.load_block()` needs to be called manually afterwards.
    /// If all docs are smaller than target, the block loaded may be empty,
    /// or be the last an incomplete VInt block.
    pub(crate) fn seek_block(&mut self, target_doc: DocId) {
        if self.skip_reader.seek(target_doc) {
            self.block_max_score_cache = None;
            self.block_loaded = false;
        }
    }

    #[inline]
    pub(crate) fn has_remaining_docs(&self) -> bool {
        self.skip_reader.has_remaining_docs()
    }

    pub(crate) fn block_is_loaded(&self) -> bool {
        self.block_loaded
    }

    pub(crate) fn block_max_score_up_to(
        &mut self,
        target: DocId,
        fieldnorms: &FieldNormReader,
        weight: &Bm25Weight,
    ) -> (Score, DocId) {
        let mut bound = self.block_max_score(fieldnorms, weight);
        if self.skip_reader.last_doc_in_block() >= target {
            return (bound, self.skip_reader.last_doc_in_block());
        }
        let mut impacts = self.skip_reader.clone();
        while impacts.last_doc_in_block() < target {
            impacts.advance();
            bound = bound.max(
                impacts
                    .block_max_score(weight)
                    .unwrap_or_else(|| weight.max_score()),
            );
        }
        (bound, impacts.last_doc_in_block())
    }

    pub(crate) fn load_block(&mut self) {
        self.try_load_block()
            .expect("posting data became unreadable after the reader was opened");
    }

    fn block_data_range(&self) -> std::ops::Range<usize> {
        let start = self.skip_reader.byte_offset();
        let end = match self.skip_reader.block_info() {
            BlockInfo::BitPacked {
                doc_num_bits,
                tf_num_bits,
                ..
            } => {
                start
                    + compressed_block_size(
                        doc_num_bits
                            + if self.freqs_data.is_some() {
                                0
                            } else {
                                tf_num_bits
                            },
                    )
            }
            BlockInfo::VInt { num_docs: 0 } => start,
            BlockInfo::VInt { .. } => self.data.len(),
        };
        start..end
    }

    fn try_load_block(&mut self) -> io::Result<()> {
        if self.block_is_loaded() {
            return Ok(());
        }
        if self.freqs_data.is_some()
            && !matches!(
                self.skip_reader.block_info(),
                BlockInfo::VInt { num_docs: 0 }
            )
        {
            self.freq_decoder.take();
        }
        let range = self.block_data_range();
        match self.skip_reader.block_info() {
            BlockInfo::BitPacked {
                doc_num_bits,
                strict_delta_encoded,
                tf_num_bits,
                ..
            } => {
                self.data.with_bytes(range, |data| {
                    decode_bitpacked_block(
                        &mut self.doc_decoder,
                        if self.freqs_data.is_none()
                            && self.freq_reading_option == FreqReadingOption::ReadFreq
                        {
                            self.freq_decoder.get_mut()
                        } else {
                            None
                        },
                        data,
                        self.skip_reader.last_doc_in_previous_block,
                        doc_num_bits,
                        tf_num_bits,
                        strict_delta_encoded,
                    );
                })?;
            }
            BlockInfo::VInt { num_docs } => {
                self.data.with_bytes(range, |data| {
                    decode_vint_block(
                        &mut self.doc_decoder,
                        if self.freqs_data.is_none()
                            && self.freq_reading_option == FreqReadingOption::ReadFreq
                        {
                            self.freq_decoder.get_mut()
                        } else {
                            None
                        },
                        data,
                        self.skip_reader.last_doc_in_previous_block,
                        num_docs as usize,
                    );
                })?;
            }
        }
        self.block_loaded = true;
        Ok(())
    }

    /// Advance to the next block.
    pub fn advance(&mut self) {
        self.skip_reader.advance();
        self.block_loaded = false;
        self.block_max_score_cache = None;
        self.load_block();
    }

    /// Returns an empty segment postings object
    pub fn empty() -> BlockSegmentPostings {
        BlockSegmentPostings {
            doc_decoder: BlockDecoder::with_val(TERMINATED),
            block_loaded: true,
            freq_decoder: OnceCell::from(BlockDecoder::with_val(1)),
            freq_reading_option: FreqReadingOption::NoFreq,
            record_option: IndexRecordOption::Basic,
            requested_option: IndexRecordOption::Basic,
            block_max_score_cache: None,
            doc_freq: 0,
            data: PostingData::Eager(OwnedBytes::empty()),
            freqs_data: None,
            skip_reader: SkipReader::new(OwnedBytes::empty(), 0, IndexRecordOption::Basic, false),
            term_norms: None,
        }
    }

    pub(crate) fn skip_reader(&self) -> &SkipReader {
        &self.skip_reader
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use common::HasLen;

    use super::BlockSegmentPostings;
    use crate::directory::{CompositeFile, FileHandle, FileSlice, OwnedBytes};
    use crate::docset::{DocSet, TERMINATED};
    use crate::index::{Index, SegmentComponent};
    use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
    use crate::postings::postings::Postings;
    use crate::postings::SegmentPostings;
    use crate::schema::{IndexRecordOption, Schema, Term, INDEXED, TEXT};
    use crate::DocId;

    #[derive(Debug)]
    struct BlockBackedFile {
        data: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        fail_after: Option<usize>,
    }

    impl HasLen for BlockBackedFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for BlockBackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            if self.fail_after.is_some_and(|end| range.end > end) {
                return Err(io::Error::other("injected read failure"));
            }
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.data[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(4096)
        }
    }

    #[test]
    fn test_lazy_posting_blocks_match_eager() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let number = schema.add_u64_field("number", INDEXED);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_with_num_threads(1, 100_000_000)?;
        for doc_id in 0..150_000u64 {
            writer.add_document(doc!(
                text => "x ".repeat((doc_id % 17 + 1) as usize),
                number => doc_id % 3
            ))?;
        }
        writer.commit()?;
        drop(writer);
        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        let postings = CompositeFile::open(&reader.open_read(SegmentComponent::Postings)?)?;
        for (term, record_option) in [
            (
                Term::from_field_text(text, "x"),
                IndexRecordOption::WithFreqsAndPositions,
            ),
            (Term::from_field_u64(number, 1), IndexRecordOption::Basic),
            (
                Term::from_field_u64(number, 1),
                IndexRecordOption::WithFreqsAndPositions,
            ),
        ] {
            let inverted = reader.inverted_index(term.field())?;
            let info = inverted.get_term_info(&term)?.unwrap();
            let bytes = postings
                .open_read(term.field())
                .unwrap()
                .slice_from(8)
                .slice(info.postings_range.clone())
                .read_bytes()?;
            let freqs = info.freqs_range.as_ref().map(|range| {
                CompositeFile::open(&reader.open_read(SegmentComponent::TermFrequencies).unwrap())
                    .unwrap()
                    .open_read(term.field())
                    .unwrap()
                    .slice(range.clone())
                    .read_bytes()
                    .unwrap()
            });
            assert!(bytes.len() > super::POSTINGS_BUFFER_SIZE);
            for option in [
                IndexRecordOption::Basic,
                IndexRecordOption::WithFreqs,
                IndexRecordOption::WithFreqsAndPositions,
            ] {
                let reads = Arc::new(Mutex::new(Vec::new()));
                let file = FileSlice::new(Arc::new(BlockBackedFile {
                    data: bytes.to_vec(),
                    reads: reads.clone(),
                    fail_after: None,
                }));
                let mut lazy = BlockSegmentPostings::open_file_slice(
                    info.doc_freq,
                    file,
                    freqs.clone().map(|bytes| FileSlice::new(Arc::new(bytes))),
                    record_option,
                    option,
                )?;
                let opening_reads = reads.lock().unwrap();
                assert!(opening_reads.iter().map(|range| range.len()).sum::<usize>() < bytes.len());
                drop(opening_reads);
                let mut eager = BlockSegmentPostings::open(
                    info.doc_freq,
                    bytes.clone(),
                    freqs.clone(),
                    record_option,
                    option,
                )?;
                let mut lazy_seek = lazy.clone();
                let mut eager_seek = eager.clone();
                let reads_before = reads.lock().unwrap().len();
                let mut lazy_clone = lazy.clone();
                let mut eager_clone = eager.clone();
                lazy_clone.advance();
                eager_clone.advance();
                assert_eq!(lazy_clone.docs(), eager_clone.docs());
                assert_eq!(lazy_clone.freqs(), eager_clone.freqs());
                assert_eq!(reads.lock().unwrap().len(), reads_before);

                loop {
                    assert_eq!(lazy.docs(), eager.docs());
                    assert_eq!(lazy.freqs(), eager.freqs());
                    if eager.docs().is_empty() {
                        break;
                    }
                    lazy.advance();
                    eager.advance();
                }
                for target in [
                    0, 1, 127, 128, 129, 8191, 32000, 49999, 50000, 149999, 150000, TERMINATED,
                ] {
                    assert_eq!(lazy_seek.seek(target), eager_seek.seek(target));
                    assert_eq!(lazy_seek.docs(), eager_seek.docs());
                    assert_eq!(lazy_seek.freqs(), eager_seek.freqs());
                }
                lazy_seek.reset(
                    info.doc_freq,
                    bytes.clone(),
                    freqs.clone().map(|bytes| FileSlice::new(Arc::new(bytes))),
                )?;
                eager_seek.reset(
                    info.doc_freq,
                    bytes.clone(),
                    freqs.clone().map(|bytes| FileSlice::new(Arc::new(bytes))),
                )?;
                assert_eq!(lazy_seek.docs(), eager_seek.docs());
                assert_eq!(lazy_seek.freqs(), eager_seek.freqs());
            }
            let header = bytes.slice(0..10);
            let (skip_len, header_len) =
                common::VInt::deserialize_with_size(&mut header.as_slice())?;
            let file = FileSlice::new(Arc::new(BlockBackedFile {
                data: bytes.to_vec(),
                reads: Arc::default(),
                fail_after: Some(header_len + skip_len.0 as usize),
            }));
            let failing_freqs = freqs.as_ref().map(|bytes| {
                FileSlice::new(Arc::new(BlockBackedFile {
                    data: bytes.to_vec(),
                    reads: Arc::default(),
                    fail_after: Some(0),
                }))
            });
            if failing_freqs.is_some() {
                for option in [
                    IndexRecordOption::Basic,
                    IndexRecordOption::WithFreqs,
                    IndexRecordOption::WithFreqsAndPositions,
                ] {
                    let mut docs = BlockSegmentPostings::open_file_slice(
                        info.doc_freq,
                        FileSlice::new(Arc::new(bytes.clone())),
                        failing_freqs.clone(),
                        record_option,
                        option,
                    )?;
                    docs.seek(149999);
                    if option.has_freq() {
                        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || docs.freq(0)
                        ))
                        .is_err());
                    }
                }
            }
            assert!(BlockSegmentPostings::open_file_slice(
                info.doc_freq,
                file,
                failing_freqs,
                record_option,
                IndexRecordOption::WithFreqs,
            )
            .and_then(|mut postings| {
                postings.seek_block(149999);
                postings.try_load_block()
            })
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn separated_frequencies_are_read_only_on_access() -> crate::Result<()> {
        for count in [1, 127, 128, 129, 4097] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            for doc in 0..count {
                writer.add_document(doc!(text => "x ".repeat((doc % 17 + 1) as usize)))?;
            }
            writer.commit()?;
            let searcher = index.reader()?.searcher();
            let segment = searcher.segment_reader(0);
            let inverted = segment.inverted_index(text)?;
            let info = inverted
                .get_term_info(&Term::from_field_text(text, "x"))?
                .unwrap();
            let docs = CompositeFile::open(&segment.open_read(SegmentComponent::Postings)?)?
                .open_read(text)
                .unwrap()
                .slice_from(8)
                .slice(info.postings_range)
                .read_bytes()?;
            let freqs =
                CompositeFile::open(&segment.open_read(SegmentComponent::TermFrequencies)?)?
                    .open_read(text)
                    .unwrap()
                    .slice(info.freqs_range.unwrap())
                    .read_bytes()?;
            for option in [
                IndexRecordOption::Basic,
                IndexRecordOption::WithFreqs,
                IndexRecordOption::WithFreqsAndPositions,
            ] {
                let reads = Arc::new(Mutex::new(Vec::new()));
                let file = FileSlice::new(Arc::new(BlockBackedFile {
                    data: freqs.to_vec(),
                    reads: reads.clone(),
                    fail_after: None,
                }));
                let mut blocks = BlockSegmentPostings::open_file_slice(
                    count,
                    FileSlice::new(Arc::new(docs.clone())),
                    Some(file.clone()),
                    IndexRecordOption::WithFreqsAndPositions,
                    option,
                )?;
                while !blocks.docs().is_empty() {
                    blocks.advance();
                }
                blocks.reset(count, docs.clone(), Some(file.clone()))?;
                let offset = blocks.seek(count / 2);
                assert!(reads.lock().unwrap().is_empty());
                let start = blocks.skip_reader().freq_byte_offset();
                let expected = if option.has_freq() {
                    (count / 2) % 17 + 1
                } else {
                    1
                };
                assert_eq!(blocks.freq(offset), expected);
                if option.has_freq() {
                    assert_eq!(reads.lock().unwrap()[0].start, start);
                    for (doc, freq) in blocks.docs().iter().zip(blocks.freqs()) {
                        assert_eq!(*freq, doc % 17 + 1);
                    }
                    assert_eq!(blocks.freq_output_array(), blocks.freqs());
                } else {
                    assert!(reads.lock().unwrap().is_empty());
                }
                let loaded = blocks.clone();
                let accesses = reads.lock().unwrap().len();
                assert_eq!(loaded.freq(offset), expected);
                assert_eq!(blocks.freq(offset), expected);
                let tail = blocks.seek(count - 1);
                assert_eq!(reads.lock().unwrap().len(), accesses);
                assert_eq!(
                    blocks.freq(tail),
                    if option.has_freq() {
                        (count - 1) % 17 + 1
                    } else {
                        1
                    }
                );
                assert_eq!(loaded.freq(offset), expected);
                let accesses = reads.lock().unwrap().len();
                blocks.reset(count, docs.clone(), Some(file))?;
                while !blocks.docs().is_empty() {
                    blocks.advance();
                }
                assert_eq!(reads.lock().unwrap().len(), accesses);
            }
        }
        Ok(())
    }

    #[test]
    fn test_lazy_postings_reject_truncated_skip_data() -> io::Result<()> {
        use common::BinarySerializable;

        for skip_len in [100_000, u64::MAX] {
            let mut data = Vec::new();
            common::VInt(skip_len).serialize(&mut data)?;
            data.resize(super::POSTINGS_BUFFER_SIZE + 1, 0);
            let file = FileSlice::new(Arc::new(BlockBackedFile {
                data,
                reads: Arc::default(),
                fail_after: None,
            }));
            assert!(BlockSegmentPostings::open_file_slice(
                128,
                file,
                None,
                IndexRecordOption::WithFreqs,
                IndexRecordOption::WithFreqs,
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn reset_between_json_numeric_and_text_terms() -> crate::Result<()> {
        for count in [1, 127, 128, 129, 257] {
            let mut schema = Schema::builder();
            let json = schema.add_json_field("json", TEXT);
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            for _ in 0..count {
                writer
                    .add_document(doc!(json => serde_json::json!({"n": 1u64, "text": "a a a"})))?;
            }
            writer.commit()?;
            let searcher = index.reader()?.searcher();
            let inverted = searcher.segment_reader(0).inverted_index(json)?;
            let mut number = Term::from_field_json_path(json, "n", false);
            number.append_type_and_fast_value(1i64);
            let mut text = Term::from_field_json_path(json, "text", false);
            text.append_type_and_str("a");
            for option in [IndexRecordOption::Basic, IndexRecordOption::WithFreqs] {
                let mut block = inverted.read_block_postings(&number, option)?.unwrap();
                for (term, freq) in [(&text, 3), (&number, 1), (&text, 3)] {
                    let info = inverted.get_term_info(term)?.unwrap();
                    inverted.reset_block_postings_from_terminfo(&info, &mut block)?;
                    let mut docs = 0;
                    while !block.docs().is_empty() {
                        for offset in 0..block.docs().len() {
                            assert_eq!(
                                block.freq(offset),
                                if option.has_freq() { freq } else { 1 }
                            );
                            docs += 1;
                        }
                        block.advance();
                    }
                    assert_eq!(docs, count);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn term_norms_require_source_and_offset() {
        let source = FileSlice::from(vec![7u8]);
        let mut postings = BlockSegmentPostings::empty();
        for (file, offset) in [
            (Some(source.clone()), Some(0)),
            (Some(source.clone()), None),
            (None, Some(0)),
            (None, None),
        ] {
            let present = file.is_some() && offset.is_some();
            postings.set_term_norm_source(file, offset);
            assert_eq!(postings.term_norms.is_some(), present);
        }
    }

    #[test]
    fn test_empty_segment_postings() {
        let mut postings = SegmentPostings::empty();
        assert_eq!(postings.doc(), TERMINATED);
        assert_eq!(postings.advance(), TERMINATED);
        assert_eq!(postings.advance(), TERMINATED);
        assert_eq!(postings.doc_freq(), 0);
        assert_eq!(postings.len(), 0);
    }

    #[test]
    fn test_empty_postings_doc_returns_terminated() {
        let mut postings = SegmentPostings::empty();
        assert_eq!(postings.doc(), TERMINATED);
        assert_eq!(postings.advance(), TERMINATED);
    }

    #[test]
    fn test_empty_postings_doc_term_freq_returns_0() {
        let postings = SegmentPostings::empty();
        assert_eq!(postings.term_freq(), 1);
    }

    #[test]
    fn test_empty_block_segment_postings() {
        let mut postings = BlockSegmentPostings::empty();
        assert!(postings.docs().is_empty());
        assert_eq!(postings.doc_freq(), 0);
        postings.advance();
        assert!(postings.docs().is_empty());
        assert_eq!(postings.doc_freq(), 0);
    }

    #[test]
    fn test_block_segment_postings() -> crate::Result<()> {
        let mut block_segments = build_block_postings(&(0..100_000).collect::<Vec<u32>>())?;
        let mut offset: u32 = 0u32;
        // checking that the `doc_freq` is correct
        assert_eq!(block_segments.doc_freq(), 100_000);
        loop {
            let block = block_segments.docs();
            if block.is_empty() {
                break;
            }
            for (i, doc) in block.iter().cloned().enumerate() {
                assert_eq!(offset + (i as u32), doc);
            }
            offset += block.len() as u32;
            block_segments.advance();
        }
        Ok(())
    }

    #[test]
    fn test_skip_right_at_new_block() -> crate::Result<()> {
        let mut doc_ids = (0..128).collect::<Vec<u32>>();
        // 128 is missing
        doc_ids.push(129);
        doc_ids.push(130);
        {
            let block_segments = build_block_postings(&doc_ids)?;
            let mut docset = SegmentPostings::from_block_postings(block_segments, None);
            assert_eq!(docset.seek(128), 129);
            assert_eq!(docset.doc(), 129);
            assert_eq!(docset.advance(), 130);
            assert_eq!(docset.doc(), 130);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let block_segments = build_block_postings(&doc_ids).unwrap();
            let mut docset = SegmentPostings::from_block_postings(block_segments, None);
            assert_eq!(docset.seek(129), 129);
            assert_eq!(docset.doc(), 129);
            assert_eq!(docset.advance(), 130);
            assert_eq!(docset.doc(), 130);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let block_segments = build_block_postings(&doc_ids)?;
            let mut docset = SegmentPostings::from_block_postings(block_segments, None);
            assert_eq!(docset.doc(), 0);
            assert_eq!(docset.seek(131), TERMINATED);
            assert_eq!(docset.doc(), TERMINATED);
        }
        Ok(())
    }

    fn build_block_postings(docs: &[DocId]) -> crate::Result<BlockSegmentPostings> {
        let mut schema_builder = Schema::builder();
        let int_field = schema_builder.add_u64_field("id", INDEXED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;
        let mut last_doc = 0u32;
        for &doc in docs {
            for _ in last_doc..doc {
                index_writer.add_document(doc!(int_field=>1u64))?;
            }
            index_writer.add_document(doc!(int_field=>0u64))?;
            last_doc = doc + 1;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let segment_reader = searcher.segment_reader(0);
        let inverted_index = segment_reader.inverted_index(int_field).unwrap();
        let term = Term::from_field_u64(int_field, 0u64);
        let term_info = inverted_index.get_term_info(&term)?.unwrap();
        let block_postings = inverted_index
            .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
        Ok(block_postings)
    }

    #[test]
    fn test_block_segment_postings_seek() -> crate::Result<()> {
        let mut docs = vec![0];
        for i in 0..1300 {
            docs.push((i * i / 100) + i);
        }
        let mut block_postings = build_block_postings(&docs[..])?;
        for i in &[0, 424, 10000] {
            block_postings.seek(*i);
            let docs = block_postings.docs();
            assert!(docs[0] <= *i);
            assert!(docs.last().cloned().unwrap_or(0u32) >= *i);
        }
        block_postings.seek(100_000);
        assert_eq!(block_postings.doc(COMPRESSION_BLOCK_SIZE - 1), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_reset_block_segment_postings() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let int_field = schema_builder.add_u64_field("id", INDEXED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;
        // create two postings list, one containing even number,
        // the other containing odd numbers.
        for i in 0..6 {
            let doc = doc!(int_field=> (i % 2) as u64);
            index_writer.add_document(doc)?;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let segment_reader = searcher.segment_reader(0);

        let mut block_segments;
        {
            let term = Term::from_field_u64(int_field, 0u64);
            let inverted_index = segment_reader.inverted_index(int_field)?;
            let term_info = inverted_index.get_term_info(&term)?.unwrap();
            block_segments = inverted_index
                .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
        }
        assert_eq!(block_segments.docs(), &[0, 2, 4]);
        {
            let term = Term::from_field_u64(int_field, 1u64);
            let inverted_index = segment_reader.inverted_index(int_field)?;
            let term_info = inverted_index.get_term_info(&term)?.unwrap();
            inverted_index.reset_block_postings_from_terminfo(&term_info, &mut block_segments)?;
        }
        assert_eq!(block_segments.docs(), &[1, 3, 5]);
        Ok(())
    }

    #[test]
    fn test_block_segment_postings_rank() -> crate::Result<()> {
        // ~8 blocks worth of docs so the skip list is actually exercised.
        let docs: Vec<DocId> = (0..1000u32).map(|i| i * 3).collect();
        let mut block_postings = build_block_postings(&docs[..])?;
        let doc_freq = block_postings.doc_freq();

        // rank(target) must equal the number of docs strictly below target.
        // Targets are queried in non-decreasing order, as the API requires.
        // `target` values must be a valid doc id (<= TERMINATED) and non-decreasing.
        let targets = [
            0u32, 1, 2, 3, 4, 299, 300, 301, 1500, 2996, 2997, 3000, 10_000,
        ];
        for &target in &targets {
            let expected = docs.iter().filter(|&&d| d < target).count() as u32;
            assert_eq!(
                block_postings.rank(target),
                expected,
                "rank({target}) mismatch"
            );
        }

        // Edge cases: below the first doc -> 0, above the last doc -> doc_freq.
        let mut fresh = build_block_postings(&docs[..])?;
        assert_eq!(fresh.rank(0), 0);
        let mut fresh = build_block_postings(&docs[..])?;
        assert_eq!(fresh.rank(1_000_000), doc_freq);

        // Empty postings: rank is always 0.
        let mut empty = BlockSegmentPostings::empty();
        assert_eq!(empty.rank(42), 0);
        Ok(())
    }
}

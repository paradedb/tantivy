use std::io;

use common::{HasLen, VInt};

use crate::directory::{FileSlice, OwnedBytes};
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
    freq_decoder: BlockDecoder,
    freq_reading_option: FreqReadingOption,
    block_max_score_cache: Option<Score>,
    doc_freq: u32,
    data: OwnedBytes,
    lazy_data: Option<(FileSlice, usize)>,
    data_offset: usize,
    skip_reader: SkipReader,
    term_norm_offset: Option<u64>,
    term_norms: Option<super::term_norms::TermNormReader>,
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
        record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> io::Result<BlockSegmentPostings> {
        let (term_norm_offset, bytes) = super::term_norms::read_header(bytes)?;
        let (skip_data_opt, postings_data) = split_into_skips_and_postings(doc_freq, bytes)?;
        Self::open_with_data(
            doc_freq,
            skip_data_opt,
            postings_data,
            None,
            term_norm_offset,
            record_option,
            requested_option,
        )
    }

    pub(crate) fn open_from_file(
        doc_freq: u32,
        file: FileSlice,
        record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
        buffer_size: usize,
    ) -> io::Result<Self> {
        if buffer_size == 0 || file.len() <= buffer_size || doc_freq < COMPRESSION_BLOCK_SIZE as u32
        {
            return Self::open(
                doc_freq,
                file.read_bytes()?,
                record_option,
                requested_option,
            );
        }
        let prefix = file.read_bytes_slice(0..file.len().min(28))?;
        let prefix_len = prefix.len();
        let (term_norm_offset, mut header) = super::term_norms::read_header(prefix)?;
        let skip_len = VInt::deserialize_u64(&mut header)? as usize;
        let header_len = prefix_len - header.len();
        let postings_start = header_len
            .checked_add(skip_len)
            .filter(|&end| end <= file.len())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "invalid postings skip length")
            })?;
        let skips = file.read_bytes_slice(header_len..postings_start)?;
        let payload = file.slice(postings_start..);
        Self::open_with_data(
            doc_freq,
            Some(skips),
            OwnedBytes::empty(),
            Some((payload, buffer_size)),
            term_norm_offset,
            record_option,
            requested_option,
        )
    }

    fn open_with_data(
        doc_freq: u32,
        skip_data_opt: Option<OwnedBytes>,
        postings_data: OwnedBytes,
        lazy_data: Option<(FileSlice, usize)>,
        term_norm_offset: Option<u64>,
        mut record_option: IndexRecordOption,
        requested_option: IndexRecordOption,
    ) -> io::Result<Self> {
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
                SkipReader::new(skip_data, doc_freq, record_option)
            }
            None => SkipReader::new(OwnedBytes::empty(), doc_freq, record_option),
        };

        let freq_reading_option = match (record_option, requested_option) {
            (IndexRecordOption::Basic, _) => FreqReadingOption::NoFreq,
            (_, IndexRecordOption::Basic) => FreqReadingOption::SkipFreq,
            (_, _) => FreqReadingOption::ReadFreq,
        };

        let mut block_segment_postings = BlockSegmentPostings {
            doc_decoder: BlockDecoder::with_val(TERMINATED),
            block_loaded: false,
            freq_decoder: BlockDecoder::with_val(1),
            freq_reading_option,
            block_max_score_cache: None,
            doc_freq,
            data: postings_data,
            lazy_data,
            data_offset: 0,
            skip_reader,
            term_norm_offset,
            term_norms: None,
        };
        block_segment_postings.load_block();
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
            let freqs = self.freq_decoder.output_array().iter().cloned();
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

    pub(crate) fn set_term_norm_source(
        &mut self,
        source: std::sync::Arc<common::file_slice::DeferredFileSlice>,
        storage: crate::fieldnorm::NormStorage,
    ) -> io::Result<()> {
        let required = storage == crate::fieldnorm::NormStorage::Posting;
        if required && self.term_norm_offset.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing required posting norm header",
            ));
        }
        self.term_norms = self.term_norm_offset.and_then(|offset| {
            super::term_norms::TermNormReader::new(source, offset, self.doc_freq, required)
        });
        Ok(())
    }

    pub(crate) fn disable_term_norms(&mut self) {
        self.term_norms = None;
    }

    #[cfg(test)]
    pub(crate) fn set_posting_norms_for_test(&mut self, norms: Vec<u8>) {
        let file = crate::directory::FileSlice::from(norms);
        let source = std::sync::Arc::new(common::file_slice::DeferredFileSlice::new(move || {
            Ok(file.clone())
        }));
        self.term_norms = super::term_norms::TermNormReader::new(source, 0, self.doc_freq, true);
    }

    pub(crate) fn fieldnorm_id_at(&self, offset: usize, fallback: &FieldNormReader) -> u8 {
        self.posting_fieldnorm_id_at(offset)
            .unwrap_or_else(|| fallback.fieldnorm_id(self.doc(offset)))
    }

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
    pub(crate) fn reset(&mut self, doc_freq: u32, postings_data: OwnedBytes) -> io::Result<()> {
        let (term_norm_offset, postings_data) = super::term_norms::read_header(postings_data)?;
        self.term_norm_offset = term_norm_offset;
        self.term_norms = None;
        let (skip_data_opt, postings_data) =
            split_into_skips_and_postings(doc_freq, postings_data)?;
        self.data = postings_data;
        self.lazy_data = None;
        self.data_offset = 0;
        self.block_max_score_cache = None;
        self.block_loaded = false;
        if let Some(skip_data) = skip_data_opt {
            self.skip_reader.reset(skip_data, doc_freq);
        } else {
            self.skip_reader.reset(OwnedBytes::empty(), doc_freq);
        }
        self.doc_freq = doc_freq;
        self.load_block();
        Ok(())
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
        self.freq_decoder.output_array()
    }

    /// Return the frequency at index `idx` of the block.
    #[inline]
    pub fn freq(&self, idx: usize) -> u32 {
        debug_assert!(self.block_is_loaded());
        self.freq_decoder.output(idx)
    }

    /// Returns the length of the current block.
    ///
    /// Returns the decoded term-frequency buffer for the current block.
    #[inline]
    pub(crate) fn freq_output_array(&self) -> &[u32] {
        self.freq_decoder.output_array()
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
        if self.block_is_loaded() {
            return;
        }
        let offset = self.skip_reader.byte_offset();
        if let Some((file, buffer_size)) = &self.lazy_data {
            let end = match self.skip_reader.block_info() {
                BlockInfo::BitPacked {
                    doc_num_bits,
                    tf_num_bits,
                    ..
                } => {
                    offset
                        + compressed_block_size(doc_num_bits)
                        + if self.freq_reading_option == FreqReadingOption::ReadFreq {
                            compressed_block_size(tf_num_bits)
                        } else {
                            0
                        }
                }
                BlockInfo::VInt { num_docs: 0 } => offset,
                BlockInfo::VInt { .. } => file.len(),
            };
            if end > offset
                && (offset < self.data_offset || end > self.data_offset + self.data.len())
            {
                let read_end = end.max(offset.saturating_add(*buffer_size)).min(file.len());
                self.data = file
                    .read_bytes_slice(offset..read_end)
                    .expect("Failed to read postings block");
                self.data_offset = offset;
            }
        }
        let data = if self.skip_reader.block_info() == (BlockInfo::VInt { num_docs: 0 }) {
            &[]
        } else {
            &self.data.as_slice()[offset - self.data_offset..]
        };
        match self.skip_reader.block_info() {
            BlockInfo::BitPacked {
                doc_num_bits,
                strict_delta_encoded,
                tf_num_bits,
                ..
            } => {
                decode_bitpacked_block(
                    &mut self.doc_decoder,
                    if let FreqReadingOption::ReadFreq = self.freq_reading_option {
                        Some(&mut self.freq_decoder)
                    } else {
                        None
                    },
                    data,
                    self.skip_reader.last_doc_in_previous_block,
                    doc_num_bits,
                    tf_num_bits,
                    strict_delta_encoded,
                );
            }
            BlockInfo::VInt { num_docs } => {
                decode_vint_block(
                    &mut self.doc_decoder,
                    if let FreqReadingOption::ReadFreq = self.freq_reading_option {
                        Some(&mut self.freq_decoder)
                    } else {
                        None
                    },
                    data,
                    self.skip_reader.last_doc_in_previous_block,
                    num_docs as usize,
                );
            }
        }
        self.block_loaded = true;
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
            freq_decoder: BlockDecoder::with_val(1),
            freq_reading_option: FreqReadingOption::NoFreq,
            block_max_score_cache: None,
            doc_freq: 0,
            data: OwnedBytes::empty(),
            lazy_data: None,
            data_offset: 0,
            skip_reader: SkipReader::new(OwnedBytes::empty(), 0, IndexRecordOption::Basic),
            term_norm_offset: None,
            term_norms: None,
        }
    }

    pub(crate) fn skip_reader(&self) -> &SkipReader {
        &self.skip_reader
    }
}

#[cfg(test)]
mod tests {
    include!("lazy_postings_bench.rs");

    use common::HasLen;

    use super::BlockSegmentPostings;
    use crate::docset::{DocSet, TERMINATED};
    use crate::index::Index;
    use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
    use crate::postings::postings::Postings;
    use crate::postings::SegmentPostings;
    use crate::schema::{IndexRecordOption, Schema, Term, INDEXED};
    use crate::DocId;

    #[test]
    fn test_lazy_postings_reads_and_seeks() -> crate::Result<()> {
        use std::ops::Range;
        use std::sync::{Arc, Mutex};

        use crate::directory::{FileHandle, FileSlice, OwnedBytes};
        use crate::index::Bm25Params;
        use crate::postings::serializer::PostingsSerializer;

        #[derive(Debug)]
        struct TrackedFile {
            bytes: OwnedBytes,
            reads: Arc<Mutex<Vec<Range<usize>>>>,
        }
        impl HasLen for TrackedFile {
            fn len(&self) -> usize {
                self.bytes.len()
            }
        }
        impl FileHandle for TrackedFile {
            fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
                self.reads.lock().unwrap().push(range.clone());
                Ok(self.bytes.slice(range))
            }
        }

        for count in [0, 1, 127, 128, 129, 256, 257, 100_003] {
            for record in [
                IndexRecordOption::Basic,
                IndexRecordOption::WithFreqs,
                IndexRecordOption::WithFreqsAndPositions,
            ] {
                for dense in [false, true] {
                    let mut serializer =
                        PostingsSerializer::new(20.0, record, None, Bm25Params::default());
                    serializer.new_term(count, record.has_freq());
                    let docs: Vec<_> = (0..count)
                        .map(|i| if dense { i } else { i * 13 + i % 7 })
                        .collect();
                    for (i, &doc) in docs.iter().enumerate() {
                        serializer.write_doc(doc, if dense { 1 } else { (i % 17 + 1) as u32 });
                    }
                    let mut bytes = Vec::new();
                    serializer.close_term(count, &mut bytes)?;
                    let bytes = OwnedBytes::new(bytes);
                    for requested in [IndexRecordOption::Basic, IndexRecordOption::WithFreqs] {
                        for size in [1, 4096, 8192, 65536] {
                            let reads = Arc::new(Mutex::new(Vec::new()));
                            let file = FileSlice::new(Arc::new(TrackedFile {
                                bytes: bytes.clone(),
                                reads: reads.clone(),
                            }));
                            let lazy = BlockSegmentPostings::open_from_file(
                                count, file, record, requested, size,
                            )?;
                            let eager = BlockSegmentPostings::open(
                                count,
                                bytes.clone(),
                                record,
                                requested,
                            )?;
                            if count > 100_000 && !dense && size == 1 {
                                let read_bytes: usize =
                                    reads.lock().unwrap().iter().map(|r| r.len()).sum();
                                assert!(read_bytes < bytes.len() / 2);
                            }
                            let mut reset = lazy.clone();
                            reset.seek(1_000_000);
                            reset.reset(count, bytes.clone())?;
                            assert_eq!(reset.docs(), eager.docs());
                            assert_eq!(reset.freqs(), eager.freqs());
                            let mut scanned = lazy.clone();
                            let mut expected = eager.clone();
                            loop {
                                assert_eq!(scanned.docs(), expected.docs());
                                assert_eq!(scanned.freqs(), expected.freqs());
                                if expected.docs().is_empty() {
                                    break;
                                }
                                scanned.advance();
                                expected.advance();
                            }
                            let mut sought = SegmentPostings::from_block_postings(lazy, None);
                            let mut expected = SegmentPostings::from_block_postings(eager, None);
                            for target in
                                [0, 126, 127, 128, 129, 255, 256, 1000, 100_000, 2_000_000]
                            {
                                if target >= sought.doc() {
                                    assert_eq!(sought.seek(target), expected.seek(target));
                                } else {
                                    assert_eq!(sought.doc(), expected.doc());
                                }
                                assert_eq!(sought.term_freq(), expected.term_freq());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
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

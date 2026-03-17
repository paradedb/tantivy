//! Postings (inverted index) as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `InvertedIndexSerializer`, `PerFieldPostingsWriter`, and
//! `IndexingContext` types behind the plugin interface so that the inverted index
//! participates in the unified plugin lifecycle.
//!
//! Postings have a write phase of 1 because they must read back field norms
//! (phase 0) from disk.

use std::any::Any;
use std::sync::Arc;

use measure_time::debug_time;

use crate::directory::{CompositeWrite, Directory, WritePtr};
use crate::docset::{DocSet, TERMINATED};
use crate::error::DataCorruption;
use crate::fieldnorm::{FieldNormReader, FieldNormReaders};
use crate::index::merge_optimized_inverted_index_reader::MergeOptimizedInvertedIndexReader;
use crate::index::{Segment, SegmentComponent, SegmentReader};
use crate::indexer::doc_id_mapping::{DocIdMapping, SegmentDocIdMapping};
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::postings::{
    serialize_postings, IndexingContext, InvertedIndexSerializer,
    PerFieldPostingsWriter, Postings, SegmentPostings,
};
use crate::schema::{Field, FieldType, Schema};
use crate::termdict::{TermMerger, TermOrdinal};
use crate::DocId;

/// Built-in plugin for the inverted index (postings).
///
/// The inverted index maps terms to posting lists — sorted sequences of document
/// ids that contain each term, optionally with term frequencies and positions.
/// This is the core data structure that enables full-text search.
///
/// During indexing, `PostingsPluginWriter` wraps `PerFieldPostingsWriter` and
/// `IndexingContext` which accumulate postings in memory. At serialize time,
/// they are consumed and written to the `.idx`, `.pos`, and `.term` files.
///
/// During merge, the plugin reads inverted index data from source segments and
/// writes merged postings into the target segment.
pub struct PostingsPlugin;

impl SegmentPlugin for PostingsPlugin {
    fn name(&self) -> &str {
        "postings"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["idx", "pos", "term"]
    }

    fn write_phase(&self) -> u32 {
        1 // Must execute after FieldNorms (phase 0), before FastFields/Store (phase 2).
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        // During merge, the merge() method handles file creation directly.
        // Only open the serializer during normal indexing.
        let serializer = if !ctx.is_in_merge {
            // Replicate InvertedIndexSerializer::open but using the directory directly
            // to avoid needing &mut Segment.
            let segment = ctx.segment;
            let directory = ctx.directory;
            let terms_path = segment.relative_path(SegmentComponent::Terms);
            let postings_path = segment.relative_path(SegmentComponent::Postings);
            let positions_path = segment.relative_path(SegmentComponent::Positions);
            let terms_write = CompositeWrite::wrap(directory.open_write(&terms_path)?);
            let postings_write = CompositeWrite::wrap(directory.open_write(&postings_path)?);
            let positions_write = CompositeWrite::wrap(directory.open_write(&positions_path)?);
            let schema = ctx.schema.clone();
            Some(InvertedIndexSerializer::from_parts(
                terms_write,
                postings_write,
                positions_write,
                schema,
            ))
        } else {
            None
        };

        Ok(Box::new(PostingsPluginWriter {
            per_field_postings_writers: None,
            ctx: None,
            serializer,
            schema: ctx.schema.clone(),
        }))
    }

    fn open_reader(&self, _ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        // The inverted index reader is opened on-demand via SegmentReader::inverted_index()
        // because it requires a field parameter. We provide a no-op reader here.
        Ok(Arc::new(PostingsPluginReader))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        debug_time!("write-postings");
        debug!("write-postings");

        // Open the target inverted index serializer
        let target_segment = &*ctx.target_segment;
        let directory = target_segment.index().directory();
        let terms_path = target_segment.relative_path(SegmentComponent::Terms);
        let postings_path = target_segment.relative_path(SegmentComponent::Postings);
        let positions_path = target_segment.relative_path(SegmentComponent::Positions);
        let terms_write: WritePtr = directory.open_write(&terms_path)?;
        let postings_write: WritePtr = directory.open_write(&postings_path)?;
        let positions_write: WritePtr = directory.open_write(&positions_path)?;
        let mut serializer = InvertedIndexSerializer::from_parts(
            CompositeWrite::wrap(terms_write),
            CompositeWrite::wrap(postings_write),
            CompositeWrite::wrap(positions_write),
            ctx.schema.clone(),
        );

        // Read back fieldnorms written by FieldNormsPlugin (phase 0)
        let fieldnorm_data = target_segment.open_read(SegmentComponent::FieldNorms)?;
        let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;

        // Write postings for all indexed fields
        write_postings_merge(
            ctx.readers,
            ctx.schema,
            &mut serializer,
            fieldnorm_readers,
            ctx.doc_id_mapping,
            ctx.cancel,
        )?;

        serializer.close()?;
        Ok(())
    }
}

// --- Merge helper functions (moved from IndexMerger) ---

struct DeltaComputer {
    buffer: Vec<u32>,
}

impl DeltaComputer {
    fn new() -> DeltaComputer {
        DeltaComputer {
            buffer: vec![0u32; 512],
        }
    }

    fn compute_delta(&mut self, positions: &[u32]) -> &[u32] {
        if positions.len() > self.buffer.len() {
            self.buffer.resize(positions.len(), 0u32);
        }
        let mut last_pos = 0u32;
        for (cur_pos, dest) in positions.iter().cloned().zip(self.buffer.iter_mut()) {
            *dest = cur_pos - last_pos;
            last_pos = cur_pos;
        }
        &self.buffer[..positions.len()]
    }
}

fn estimate_total_num_tokens_in_single_segment(
    reader: &SegmentReader,
    field: Field,
) -> crate::Result<u64> {
    if !reader.has_deletes() {
        return Ok(reader.inverted_index(field)?.total_num_tokens());
    }
    if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
        let mut count: [usize; 256] = [0; 256];
        for doc in reader.doc_ids_alive() {
            let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
            count[fieldnorm_id as usize] += 1;
        }
        let total_num_tokens = count
            .iter()
            .cloned()
            .enumerate()
            .map(|(fieldnorm_ord, count)| {
                count as u64 * u64::from(FieldNormReader::id_to_fieldnorm(fieldnorm_ord as u8))
            })
            .sum::<u64>();
        return Ok(total_num_tokens);
    }
    let segment_num_tokens = reader.inverted_index(field)?.total_num_tokens();
    if reader.max_doc() == 0 {
        return Ok(0u64);
    }
    let ratio = reader.num_docs() as f64 / reader.max_doc() as f64;
    Ok((segment_num_tokens as f64 * ratio) as u64)
}

fn estimate_total_num_tokens(readers: &[SegmentReader], field: Field) -> crate::Result<u64> {
    let mut total_num_tokens: u64 = 0;
    for reader in readers {
        total_num_tokens += estimate_total_num_tokens_in_single_segment(reader, field)?;
    }
    Ok(total_num_tokens)
}

fn write_postings_for_field(
    readers: &[SegmentReader],
    schema: &Schema,
    indexed_field: Field,
    _field_type: &FieldType,
    serializer: &mut InvertedIndexSerializer,
    fieldnorm_reader: Option<FieldNormReader>,
    doc_id_mapping: &SegmentDocIdMapping,
    cancel: &dyn CancelSentinel,
) -> crate::Result<()> {
    debug_time!("write-postings-for-field");
    let mut positions_buffer: Vec<u32> = Vec::with_capacity(1_000);
    let mut delta_computer = DeltaComputer::new();

    let mut max_term_ords: Vec<TermOrdinal> = Vec::new();

    let field_readers: Vec<Arc<MergeOptimizedInvertedIndexReader>> = readers
        .iter()
        .map(|reader| reader.merge_optimized_inverted_index(indexed_field))
        .collect::<crate::Result<Vec<_>>>()?;

    let mut field_term_streams = Vec::new();
    for field_reader in &field_readers {
        let terms = field_reader.terms();
        field_term_streams.push(terms.stream()?);
        max_term_ords.push(terms.num_terms() as u64);
    }

    let mut merged_terms = TermMerger::new(field_term_streams);

    let mut merged_doc_id_map: Vec<Vec<Option<DocId>>> = readers
        .iter()
        .map(|reader| {
            let mut segment_local_map = vec![];
            segment_local_map.resize(reader.max_doc() as usize, None);
            segment_local_map
        })
        .collect();
    for (new_doc_id, old_doc_addr) in doc_id_mapping.iter_old_doc_addrs().enumerate() {
        let segment_map = &mut merged_doc_id_map[old_doc_addr.segment_ord as usize];
        segment_map[old_doc_addr.doc_id as usize] = Some(new_doc_id as DocId);
    }

    let total_num_tokens: u64 = estimate_total_num_tokens(readers, indexed_field)?;

    let mut field_serializer =
        serializer.new_field(indexed_field, total_num_tokens, fieldnorm_reader)?;

    let field_entry = schema.get_field_entry(indexed_field);

    let segment_postings_option = field_entry.field_type().get_index_record_option().expect(
        "Encountered a field that is not supposed to be
                     indexed. Have you modified the schema?",
    );

    let mut segment_postings_containing_the_term: Vec<(usize, SegmentPostings)> = vec![];
    let mut doc_id_and_positions = vec![];

    let mut cnt = 0;
    while merged_terms.advance() {
        if cnt % 1000 == 0 {
            if cancel.wants_cancel() {
                return Err(crate::TantivyError::Cancelled);
            }
        }
        cnt += 1;

        segment_postings_containing_the_term.clear();
        let term_bytes: &[u8] = merged_terms.key();

        let mut total_doc_freq = 0;

        for (segment_ord, term_info) in merged_terms.current_segment_ords_and_term_infos() {
            let segment_reader = &readers[segment_ord];
            let inverted_index: &MergeOptimizedInvertedIndexReader = &field_readers[segment_ord];
            let segment_postings = inverted_index
                .read_postings_from_terminfo(&term_info, segment_postings_option)?;
            let alive_bitset_opt = segment_reader.alive_bitset();
            let doc_freq = if let Some(alive_bitset) = alive_bitset_opt {
                segment_postings.doc_freq_given_deletes(alive_bitset)
            } else {
                segment_postings.doc_freq()
            };
            if doc_freq > 0u32 {
                total_doc_freq += doc_freq;
                segment_postings_containing_the_term.push((segment_ord, segment_postings));
            }
        }

        if total_doc_freq == 0u32 {
            continue;
        }

        assert!(!segment_postings_containing_the_term.is_empty());

        let has_term_freq = {
            let has_term_freq = !segment_postings_containing_the_term[0]
                .1
                .block_cursor
                .freqs()
                .is_empty();
            for (_, postings) in &segment_postings_containing_the_term[1..] {
                if has_term_freq == postings.block_cursor.freqs().is_empty() {
                    return Err(DataCorruption::comment_only(
                        "Term freqs are inconsistent across segments",
                    )
                    .into());
                }
            }
            has_term_freq
        };

        field_serializer.new_term(term_bytes, total_doc_freq, has_term_freq)?;

        for (segment_ord, mut segment_postings) in
            segment_postings_containing_the_term.drain(..)
        {
            let old_to_new_doc_id = &merged_doc_id_map[segment_ord];

            let mut doc = segment_postings.doc();
            while doc != TERMINATED {
                if doc % 1000 == 0 {
                    if cancel.wants_cancel() {
                        return Err(crate::TantivyError::Cancelled);
                    }
                }
                if let Some(remapped_doc_id) = old_to_new_doc_id[doc as usize] {
                    let term_freq = if has_term_freq {
                        segment_postings.positions(&mut positions_buffer);
                        segment_postings.term_freq()
                    } else {
                        positions_buffer.clear();
                        0u32
                    };

                    if !doc_id_mapping.is_trivial() {
                        doc_id_and_positions.push((
                            remapped_doc_id,
                            term_freq,
                            positions_buffer.to_vec(),
                        ));
                    } else {
                        let delta_positions = delta_computer.compute_delta(&positions_buffer);
                        field_serializer.write_doc(remapped_doc_id, term_freq, delta_positions);
                    }
                }

                doc = segment_postings.advance();
            }
        }
        if !doc_id_mapping.is_trivial() {
            doc_id_and_positions.sort_unstable_by_key(|&(doc_id, _, _)| doc_id);

            for (doc_id, term_freq, positions) in &doc_id_and_positions {
                let delta_positions = delta_computer.compute_delta(positions);
                field_serializer.write_doc(*doc_id, *term_freq, delta_positions);
            }
            doc_id_and_positions.clear();
        }
        field_serializer.close_term()?;
    }
    field_serializer.close()?;
    Ok(())
}

fn write_postings_merge(
    readers: &[SegmentReader],
    schema: &Schema,
    serializer: &mut InvertedIndexSerializer,
    fieldnorm_readers: FieldNormReaders,
    doc_id_mapping: &SegmentDocIdMapping,
    cancel: &dyn CancelSentinel,
) -> crate::Result<()> {
    for (field, field_entry) in schema.fields() {
        if cancel.wants_cancel() {
            return Err(crate::TantivyError::Cancelled);
        }
        let fieldnorm_reader = fieldnorm_readers.get_field(field)?;
        if field_entry.is_indexed() {
            write_postings_for_field(
                readers,
                schema,
                field,
                field_entry.field_type(),
                serializer,
                fieldnorm_reader,
                doc_id_mapping,
                cancel,
            )?;
        }
    }
    Ok(())
}

// --- Plugin writer ---

/// Plugin writer wrapping [`PerFieldPostingsWriter`], [`IndexingContext`], and
/// [`InvertedIndexSerializer`].
///
/// During normal indexing, the `SegmentWriter` populates `per_field_postings_writers`
/// and `ctx` via downcast, then at serialize time they are consumed along with
/// the serializer.
pub struct PostingsPluginWriter {
    /// Per-field postings writers. Populated by `SegmentWriter` before serialization.
    pub(crate) per_field_postings_writers: Option<PerFieldPostingsWriter>,
    /// Indexing context (arena, term index). Populated by `SegmentWriter` before serialization.
    pub(crate) ctx: Option<IndexingContext>,
    /// The inverted index serializer. `None` during merge (merge() handles it directly).
    serializer: Option<InvertedIndexSerializer>,
    /// Schema needed for serialization.
    schema: Schema,
}

#[allow(dead_code)]
impl PostingsPluginWriter {
    /// Access the per-field postings writers mutably. Used by `SegmentWriter` hot path.
    pub(crate) fn per_field_postings_writers_mut(&mut self) -> &mut PerFieldPostingsWriter {
        self.per_field_postings_writers
            .as_mut()
            .expect("PostingsPluginWriter: per_field_postings_writers not set")
    }

    /// Access the per-field postings writers immutably.
    pub(crate) fn per_field_postings_writers_ref(&self) -> &PerFieldPostingsWriter {
        self.per_field_postings_writers
            .as_ref()
            .expect("PostingsPluginWriter: per_field_postings_writers not set")
    }

    /// Access the indexing context mutably. Used by `SegmentWriter` hot path.
    pub(crate) fn ctx_mut(&mut self) -> &mut IndexingContext {
        self.ctx
            .as_mut()
            .expect("PostingsPluginWriter: ctx not set")
    }

    /// Access the indexing context immutably.
    pub(crate) fn ctx_ref(&self) -> &IndexingContext {
        self.ctx
            .as_ref()
            .expect("PostingsPluginWriter: ctx not set")
    }
}

impl PluginWriter for PostingsPluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        if let Some(mut serializer) = self.serializer.take() {
            // Read back fieldnorms from disk (written by FieldNormsPlugin in phase 0)
            let fieldnorm_data = segment.open_read(SegmentComponent::FieldNorms)?;
            let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;

            let ctx = self
                .ctx
                .take()
                .expect("PostingsPluginWriter: ctx not set at serialize time");
            let per_field_postings_writers = self
                .per_field_postings_writers
                .as_ref()
                .expect("PostingsPluginWriter: per_field_postings_writers not set at serialize time");

            serialize_postings(
                ctx,
                self.schema.clone(),
                per_field_postings_writers,
                fieldnorm_readers,
                doc_id_map,
                &mut serializer,
            )?;

            serializer.close()?;
        }
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        // If serializer wasn't consumed by serialize(), close it now.
        if let Some(serializer) = self.serializer {
            serializer.close()?;
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.ctx.as_ref().map_or(0, |c| c.mem_usage())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Plugin reader for the inverted index.
///
/// The actual inverted index reading is done via `SegmentReader::inverted_index()`
/// which requires a field parameter, so this reader is a placeholder.
pub struct PostingsPluginReader;

impl PluginReader for PostingsPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

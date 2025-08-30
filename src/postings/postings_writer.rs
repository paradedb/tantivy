use std::io;
use std::marker::PhantomData;
use std::ops::Range;

use stacker::Addr;

use crate::fieldnorm::FieldNormReaders;
use crate::indexer::path_to_unordered_id::OrderedPathId;
use crate::postings::recorder::{BufferLender, Recorder};
use crate::postings::{
    FieldSerializer, IndexingContext, InvertedIndexSerializer, PerFieldPostingsWriter,
};
use crate::schema::{Field, Schema, Term, Type};
use crate::tokenizer::{Token, TokenStream, MAX_TOKEN_LEN};
use crate::DocId;
use rayon::prelude::*;

const POSITION_GAP: u32 = 1;

#[inline]
fn json_unordered_id_from_key(key: &[u8]) -> u32 {
    // Bytes 5..9 contain the unordered path id for JSON terms
    debug_assert!(key.len() >= 9);
    u32::from_be_bytes([key[5], key[6], key[7], key[8]])
}

#[inline]
fn nonjson_term_suffix(key: &[u8]) -> &[u8] {
    // After the 5-byte header
    &key[5..]
}

#[inline]
fn json_term_suffix(key: &[u8]) -> &[u8] {
    // Skip the 4-byte unordered path id after the 5-byte header
    &key[9..]
}

fn make_field_partition(
    term_offsets: &[(Field, OrderedPathId, &[u8], Addr)],
) -> Vec<(Field, Range<usize>)> {
    let len = term_offsets.len();
    if len == 0 {
        return Vec::new();
    }

    // term_offsets is already sorted by Field, so we can scan once.
    let mut field_offsets: Vec<(Field, Range<usize>)> = Vec::new();
    field_offsets.reserve(16);

    let mut start_idx = 0usize;
    let mut current_field = term_offsets[0].0;

    for i in 1..len {
        let f = term_offsets[i].0;
        if f != current_field {
            field_offsets.push((current_field, start_idx..i));
            current_field = f;
            start_idx = i;
        }
    }
    field_offsets.push((current_field, start_idx..len));
    field_offsets
}

/// Serialize the inverted index.
/// It pushes all term, one field at a time, towards the
/// postings serializer.
pub(crate) fn serialize_postings(
    ctx: IndexingContext,
    schema: Schema,
    per_field_postings_writers: &PerFieldPostingsWriter,
    fieldnorm_readers: FieldNormReaders,
    serializer: &mut InvertedIndexSerializer,
) -> crate::Result<()> {
    // 1) Precompute path-id remap and per-field JSON mask (one schema lookup per field)
    let unordered_id_to_ordered_id: Vec<OrderedPathId> =
        ctx.path_to_unordered_id.unordered_id_to_ordered_id();

    let num_fields = schema.num_fields();
    let mut is_json_field = vec![false; num_fields];
    for i in 0..num_fields {
        let f = Field::from_field_id(i as u32);
        is_json_field[i] =
            schema.get_field_entry(f).field_type().value_type() == Type::Json;
    }

    // 2) Collect term offsets without extra schema work per term
    //    (We still use Term::wrap to get the Field exactly as Tantivy expects.)
    let mut term_offsets: Vec<(Field, OrderedPathId, &[u8], Addr)> =
        Vec::with_capacity(ctx.term_index.len());

    for (key, addr) in ctx.term_index.iter() {
        let field = Term::wrap(key).field();
        let field_idx = field.field_id() as usize;

        if is_json_field.get(field_idx).copied().unwrap_or(false) {
            let unordered_id = json_unordered_id_from_key(key) as usize;
            // SAFETY: In correct data this mapping must exist. Using direct index to match original behavior.
            let path_id = unordered_id_to_ordered_id[unordered_id];
            term_offsets.push((field, path_id, json_term_suffix(key), addr));
        } else {
            term_offsets.push((field, OrderedPathId::from(0u32), nonjson_term_suffix(key), addr));
        }
    }

    // 3) Sort by (field, path, term) — parallelized
    term_offsets.par_sort_unstable_by(|(f1, p1, b1, _), (f2, p2, b2, _)| {
        f1.cmp(f2).then_with(|| p1.cmp(p2)).then_with(|| b1.cmp(b2))
    });

    // 4) Partition by field (single linear pass)
    let ordered_id_to_path = ctx.path_to_unordered_id.ordered_id_to_path();
    let field_offsets = make_field_partition(&term_offsets);

    // 5) Serialize per field
    for (field, byte_offsets) in field_offsets {
        let postings_writer = per_field_postings_writers.get_for_field(field);
        let fieldnorm_reader = fieldnorm_readers.get_field(field)?;
        let mut field_serializer =
            serializer.new_field(field, postings_writer.total_num_tokens(), fieldnorm_reader)?;
        postings_writer.serialize(
            &term_offsets[byte_offsets],
            &ordered_id_to_path,
            &ctx,
            &mut field_serializer,
        )?;
        field_serializer.close()?;
    }

    IndexingContext::checkin(ctx);
    Ok(())
}

#[derive(Default, Debug)]
pub(crate) struct IndexingPosition {
    pub num_tokens: u32,
    pub end_position: u32,
}

/// The `PostingsWriter` is in charge of receiving documenting
/// and building a `Segment` in anonymous memory.
///
/// `PostingsWriter` writes in a `MemoryArena`.
pub(crate) trait PostingsWriter: Send + Sync {
    /// Record that a document contains a term at a given position.
    ///
    /// * doc  - the document id
    /// * pos  - the term position (expressed in tokens)
    /// * term - the term
    /// * ctx - Contains a term hashmap and a memory arena to store all necessary posting list
    ///   information.
    fn subscribe(&mut self, doc: DocId, pos: u32, term: &Term, ctx: &mut IndexingContext);

    /// Serializes the postings on disk.
    /// The actual serialization format is handled by the `PostingsSerializer`.
    fn serialize(
        &self,
        term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        ordered_id_to_path: &[&str],
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()>;

    /// Tokenize a text and subscribe all of its token.
    fn index_text(
        &mut self,
        doc_id: DocId,
        token_stream: &mut dyn TokenStream,
        term_buffer: &mut Term,
        ctx: &mut IndexingContext,
        indexing_position: &mut IndexingPosition,
    ) {
        let end_of_path_idx = term_buffer.len_bytes();
        let mut num_tokens = 0;
        let mut end_position = indexing_position.end_position;
        token_stream.process(&mut |token: &Token| {
            // We skip all tokens with a len greater than u16.
            if token.text.len() > MAX_TOKEN_LEN {
                warn!(
                    "A token exceeding MAX_TOKEN_LEN ({}>{}) was dropped. Search for \
                     MAX_TOKEN_LEN in the documentation for more information.",
                    token.text.len(),
                    MAX_TOKEN_LEN
                );
                return;
            }
            term_buffer.truncate_value_bytes(end_of_path_idx);
            term_buffer.append_bytes(token.text.as_bytes());
            let start_position = indexing_position.end_position + token.position as u32;
            end_position = end_position.max(start_position + token.position_length as u32);
            self.subscribe(doc_id, start_position, term_buffer, ctx);
            num_tokens += 1;
        });

        indexing_position.end_position = end_position + POSITION_GAP;
        indexing_position.num_tokens += num_tokens;
        term_buffer.truncate_value_bytes(end_of_path_idx);
    }

    fn total_num_tokens(&self) -> u64;
}

/// The `SpecializedPostingsWriter` is just here to remove dynamic
/// dispatch to the recorder information.
#[derive(Default)]
pub(crate) struct SpecializedPostingsWriter<Rec: Recorder> {
    total_num_tokens: u64,
    _recorder_type: PhantomData<Rec>,
}

impl<Rec: Recorder> From<SpecializedPostingsWriter<Rec>> for Box<dyn PostingsWriter> {
    fn from(
        specialized_postings_writer: SpecializedPostingsWriter<Rec>,
    ) -> Box<dyn PostingsWriter> {
        Box::new(specialized_postings_writer)
    }
}

impl<Rec: Recorder> SpecializedPostingsWriter<Rec> {
    #[inline]
    pub(crate) fn serialize_one_term(
        term: &[u8],
        addr: Addr,
        buffer_lender: &mut BufferLender,
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        let recorder: Rec = ctx.term_index.read(addr);
        let term_doc_freq = recorder.term_doc_freq().unwrap_or(0u32);
        serializer.new_term(term, term_doc_freq, recorder.has_term_freq())?;
        recorder.serialize(&ctx.arena, serializer, buffer_lender);
        serializer.close_term()?;
        Ok(())
    }
}

impl<Rec: Recorder> PostingsWriter for SpecializedPostingsWriter<Rec> {
    #[inline]
    fn subscribe(&mut self, doc: DocId, position: u32, term: &Term, ctx: &mut IndexingContext) {
        debug_assert!(term.serialized_term().len() >= 4);
        self.total_num_tokens += 1;
        let (term_index, arena) = (&mut ctx.term_index, &mut ctx.arena);
        term_index.mutate_or_create(term.serialized_term(), |opt_recorder: Option<Rec>| {
            if let Some(mut recorder) = opt_recorder {
                let current_doc = recorder.current_doc();
                if current_doc != doc {
                    recorder.close_doc(arena);
                    recorder.new_doc(doc, arena);
                }
                recorder.record_position(position, arena);
                recorder
            } else {
                let mut recorder = Rec::default();
                recorder.new_doc(doc, arena);
                recorder.record_position(position, arena);
                recorder
            }
        });
    }

    fn serialize(
        &self,
        term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        _ordered_id_to_path: &[&str],
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        let mut buffer_lender = BufferLender::default();
        for (_field, _path_id, term, addr) in term_addrs {
            Self::serialize_one_term(term, *addr, &mut buffer_lender, ctx, serializer)?;
        }
        Ok(())
    }

    fn total_num_tokens(&self) -> u64 {
        self.total_num_tokens
    }
}

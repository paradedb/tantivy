use std::io;
use std::marker::PhantomData;
use std::ops::Range;

use downcast_rs::Downcast;
use itertools::Itertools;
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

const POSITION_GAP: u32 = 1;

fn make_field_partition(
    term_offsets: &[(Field, OrderedPathId, &[u8], Addr)],
) -> Vec<(Field, Range<usize>)> {
    let term_offsets_it = term_offsets
        .iter()
        .map(|(field, _, _, _)| *field)
        .enumerate();
    let mut prev_field_opt = None;
    let mut fields = vec![];
    let mut offsets = vec![];
    for (offset, field) in term_offsets_it {
        if Some(field) != prev_field_opt {
            prev_field_opt = Some(field);
            fields.push(field);
            offsets.push(offset);
        }
    }
    offsets.push(term_offsets.len());
    let mut field_offsets = vec![];
    for i in 0..fields.len() {
        field_offsets.push((fields[i], offsets[i]..offsets[i + 1]));
    }
    field_offsets
}

pub fn serialize_postings(
    ctx: IndexingContext,
    schema: Schema,
    per_field_postings_writers: &PerFieldPostingsWriter,
    fieldnorm_readers: FieldNormReaders,
    serializer: &mut InvertedIndexSerializer,
) -> crate::Result<()> {
    // 1) Path-id mapping

    let unordered_id_to_ordered_id = ctx.path_to_unordered_id.unordered_id_to_ordered_id();

    // 2) Gather all into `term_offsets`, but keep the type code for sorting

    let mut term_offsets = Vec::with_capacity(ctx.term_index.len());
    for (raw_key, postings_addr) in ctx.term_index.iter() {
        let field = Term::wrap(raw_key).field();
        let type_code = raw_key[4];
        let leftover = &raw_key[5..];

        // If it's a JSON field => parse the 4-byte path ID
        let path_id = if type_code == b'j' {
            let unordered_id = u32::from_be_bytes(leftover[0..4].try_into().unwrap());

            unordered_id_to_ordered_id[unordered_id as usize]
        } else {
            0.into()
        };

        // For JSON fields, skip the 4 path bytes to get the actual token text
        let text_bytes = if type_code == b'j' {
            &leftover[4..]
        } else {
            leftover
        };

        term_offsets.push((field, type_code, path_id, text_bytes, postings_addr));
    }

    // 3) Sort by (field, type_code, path_id, text_bytes).

    term_offsets.sort_unstable_by(|(f1, tc1, p1, text1, _), (f2, tc2, p2, text2, _)| {
        (f1, tc1, p1, *text1).cmp(&(f2, tc2, p2, *text2))
    });

    // 4) Partition by field

    let field_offsets = term_offsets
        .iter()
        .enumerate()
        .group_by(|(_i, (field, ..))| *field)
        .into_iter()
        .map(|(field, group)| {
            let indices: Vec<_> = group.map(|(i, _)| i).collect();
            let start = *indices.iter().min().unwrap();
            let end = *indices.iter().max().unwrap() + 1;

            (field, start..end)
        })
        .collect::<Vec<_>>();

    // 5) For each field, call `.serialize(...)`

    let ordered_id_to_path = ctx.path_to_unordered_id.ordered_id_to_path();
    for (field, range) in &field_offsets {
        // No calls to `as_any()`:
        let postings_writer = per_field_postings_writers.get_for_field(*field);

        let fieldnorm_reader = fieldnorm_readers.get_field(*field)?;

        let total_num_tokens = postings_writer.total_num_tokens();

        let mut field_serializer =
            serializer.new_field(*field, total_num_tokens, fieldnorm_reader)?;

        // transform back to (field, path_id, bytes, addr)
        let subset_for_field: Vec<_> = term_offsets[range.clone()]
            .iter()
            .map(|(f, _tc, p, txt, addr)| (*f, *p, *txt, *addr))
            .collect();

        postings_writer.serialize(
            &subset_for_field,
            &ordered_id_to_path,
            &ctx,
            &mut field_serializer,
        )?;

        field_serializer.close()?;
    }

    // We do *not* call serializer.close() here (it would consume `serializer`).

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

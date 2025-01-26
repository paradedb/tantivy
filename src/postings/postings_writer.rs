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
    println!(
        "make_field_partition: Started with {} term_offsets",
        term_offsets.len()
    );
    let term_offsets_it = term_offsets
        .iter()
        .map(|(field, _, _, _)| *field)
        .enumerate();
    let mut prev_field_opt = None;
    let mut fields = vec![];
    let mut offsets = vec![];
    for (offset, field) in term_offsets_it {
        println!(
            "make_field_partition: Processing field {:?} at offset {}",
            field, offset
        );
        if Some(field) != prev_field_opt {
            println!(
                "make_field_partition: Field {:?} differs from previous. Updating partition.",
                field
            );
            prev_field_opt = Some(field);
            fields.push(field);
            offsets.push(offset);
        } else {
            println!(
                "make_field_partition: Field {:?} is same as previous. Continuing.",
                field
            );
        }
    }
    offsets.push(term_offsets.len());
    println!(
        "make_field_partition: Finalizing partitions with {} fields",
        fields.len()
    );
    let mut field_offsets = vec![];
    for i in 0..fields.len() {
        println!(
            "make_field_partition: Adding partition for field {:?} from {} to {}",
            fields[i],
            offsets[i],
            offsets[i + 1]
        );
        field_offsets.push((fields[i], offsets[i]..offsets[i + 1]));
    }
    println!(
        "make_field_partition: Completed with {} partitions",
        field_offsets.len()
    );
    field_offsets
}

pub fn serialize_postings(
    ctx: IndexingContext,
    schema: Schema,
    per_field_postings_writers: &PerFieldPostingsWriter,
    fieldnorm_readers: FieldNormReaders,
    serializer: &mut InvertedIndexSerializer,
) -> crate::Result<()> {
    println!("serialize_postings: Started serialization process");

    // 1) Path-id mapping
    println!("serialize_postings: Creating unordered_id_to_ordered_id mapping");
    let unordered_id_to_ordered_id = ctx.path_to_unordered_id.unordered_id_to_ordered_id();
    println!(
        "serialize_postings: Mapped {} unordered IDs to ordered IDs",
        unordered_id_to_ordered_id.len()
    );

    // 2) Gather all into `term_offsets`, but keep the type code for sorting
    println!("serialize_postings: Gathering term_offsets");
    let mut term_offsets = Vec::with_capacity(ctx.term_index.len());
    for (raw_key, postings_addr) in ctx.term_index.iter() {
        println!(
            "serialize_postings: Processing raw_key: {:?}, postings_addr: {:?}",
            raw_key, postings_addr
        );
        let field = Term::wrap(raw_key).field();
        let type_code = raw_key[4];
        let leftover = &raw_key[5..];
        println!(
            "serialize_postings: Extracted field {:?}, type_code {}, leftover bytes {:?}",
            field, type_code, leftover
        );

        // If it's a JSON field => parse the 4-byte path ID
        let path_id = if type_code == b'j' {
            println!("serialize_postings: Type code is 'j' (JSON). Parsing path_id");
            let unordered_id = u32::from_be_bytes(leftover[0..4].try_into().unwrap());
            println!(
                "serialize_postings: Parsed unordered_id {}. Mapping to ordered_id",
                unordered_id
            );
            unordered_id_to_ordered_id[unordered_id as usize]
        } else {
            println!("serialize_postings: Type code is not 'j'. Setting path_id to 0");
            0.into()
        };

        // For JSON fields, skip the 4 path bytes to get the actual token text
        let text_bytes = if type_code == b'j' {
            println!("serialize_postings: Skipping first 4 bytes for JSON field");
            &leftover[4..]
        } else {
            leftover
        };
        println!(
            "serialize_postings: Final text_bytes for term: {:?}",
            std::str::from_utf8(text_bytes).unwrap_or("<invalid utf8>")
        );

        term_offsets.push((field, type_code, path_id, text_bytes, postings_addr));
    }
    println!(
        "serialize_postings: Collected {} term_offsets",
        term_offsets.len()
    );

    // 3) Sort by (field, type_code, path_id, text_bytes).
    println!("serialize_postings: Sorting term_offsets");
    term_offsets.sort_unstable_by(|(f1, tc1, p1, text1, _), (f2, tc2, p2, text2, _)| {
        (f1, tc1, p1, *text1).cmp(&(f2, tc2, p2, *text2))
    });
    println!("serialize_postings: Sorting completed");

    // 4) Partition by field
    println!("serialize_postings: Partitioning term_offsets by field");
    let field_offsets = term_offsets
        .iter()
        .enumerate()
        .group_by(|(_i, (field, ..))| *field)
        .into_iter()
        .map(|(field, group)| {
            let indices: Vec<_> = group.map(|(i, _)| i).collect();
            let start = *indices.iter().min().unwrap();
            let end = *indices.iter().max().unwrap() + 1;
            println!(
                "serialize_postings: Field {:?} has term_offsets from {} to {}",
                field, start, end
            );
            (field, start..end)
        })
        .collect::<Vec<_>>();
    println!(
        "serialize_postings: Partitioned into {} fields",
        field_offsets.len()
    );

    // 5) For each field, call `.serialize(...)`
    println!("serialize_postings: Starting serialization for each field");
    let ordered_id_to_path = ctx.path_to_unordered_id.ordered_id_to_path();
    for (field, range) in &field_offsets {
        println!(
            "serialize_postings: Serializing field {:?} with range {:?}",
            field, range
        );
        // No calls to `as_any()`:
        let postings_writer = per_field_postings_writers.get_for_field(*field);
        println!(
            "serialize_postings: Retrieved postings_writer for field {:?}",
            field
        );

        let fieldnorm_reader = match fieldnorm_readers.get_field(*field) {
            Ok(reader) => {
                println!(
                    "serialize_postings: Retrieved fieldnorm_reader for field {:?}",
                    field
                );
                reader
            }
            Err(e) => {
                println!(
                    "serialize_postings: Failed to retrieve fieldnorm_reader for field {:?}: {}",
                    field, e
                );
                return Err(e.into());
            }
        };

        let total_num_tokens = postings_writer.total_num_tokens();
        println!(
            "serialize_postings: Total number of tokens for field {:?}: {}",
            field, total_num_tokens
        );

        let mut field_serializer =
            match serializer.new_field(*field, total_num_tokens, fieldnorm_reader) {
                Ok(fs) => {
                    println!(
                        "serialize_postings: Created new field_serializer for field {:?}",
                        field
                    );
                    fs
                }
                Err(e) => {
                    println!(
                        "serialize_postings: Failed to create field_serializer for field {:?}: {}",
                        field, e
                    );
                    return Err(e.into());
                }
            };

        // transform back to (field, path_id, bytes, addr)
        let subset_for_field: Vec<_> = term_offsets[range.clone()]
            .iter()
            .map(|(f, _tc, p, txt, addr)| (*f, *p, *txt, *addr))
            .collect();
        println!(
            "serialize_postings: Preparing to serialize {} terms for field {:?}",
            subset_for_field.len(),
            field
        );

        if let Err(e) = postings_writer.serialize(
            &subset_for_field,
            &ordered_id_to_path,
            &ctx,
            &mut field_serializer,
        ) {
            println!(
                "serialize_postings: Serialization failed for field {:?}: {}",
                field, e
            );
            return Err(e.into());
        }
        println!(
            "serialize_postings: Serialized {} terms for field {:?}",
            subset_for_field.len(),
            field
        );

        if let Err(e) = field_serializer.close() {
            println!(
                "serialize_postings: Failed to close field_serializer for field {:?}: {}",
                field, e
            );
            return Err(e.into());
        }
        println!(
            "serialize_postings: Closed field_serializer for field {:?}",
            field
        );
    }
    println!("serialize_postings: Completed serialization for all fields");

    // We do *not* call serializer.close() here (it would consume `serializer`).
    println!("serialize_postings: Serialization process completed successfully");
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
        println!(
            "PostingsWriter::index_text: Indexing text for doc_id {:?}",
            doc_id
        );
        let end_of_path_idx = term_buffer.len_bytes();
        let mut num_tokens = 0;
        let mut end_position = indexing_position.end_position;
        token_stream.process(&mut |token: &Token| {
            println!(
                "PostingsWriter::index_text: Processing token '{}' at position {}",
                token.text, token.position
            );
            // We skip all tokens with a len greater than u16.
            if token.text.len() > MAX_TOKEN_LEN {
                println!(
                    "PostingsWriter::index_text: Token length {} exceeds MAX_TOKEN_LEN {}. Dropping token.",
                    token.text.len(),
                    MAX_TOKEN_LEN
                );
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
            println!(
                "PostingsWriter::index_text: Subscribing at position {}",
                start_position
            );
            self.subscribe(doc_id, start_position, term_buffer, ctx);
            num_tokens += 1;
        });

        indexing_position.end_position = end_position + POSITION_GAP;
        indexing_position.num_tokens += num_tokens;
        println!(
            "PostingsWriter::index_text: Updated indexing_position to end_position {}, num_tokens {}",
            indexing_position.end_position, indexing_position.num_tokens
        );
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
        println!(
            "SpecializedPostingsWriter: Converting SpecializedPostingsWriter<{}> into Box<dyn PostingsWriter>",
            std::any::type_name::<Rec>()
        );
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
        println!(
            "SpecializedPostingsWriter::serialize_one_term: Serializing term {:?} at addr {:?}",
            std::str::from_utf8(term).unwrap_or("<invalid utf8>"),
            addr
        );
        let recorder: Rec = ctx.term_index.read(addr);
        let term_doc_freq = recorder.term_doc_freq().unwrap_or(0u32);
        println!(
            "SpecializedPostingsWriter::serialize_one_term: Term doc freq {}",
            term_doc_freq
        );
        serializer.new_term(term, term_doc_freq, recorder.has_term_freq())?;
        recorder.serialize(&ctx.arena, serializer, buffer_lender);
        serializer.close_term()?;
        println!(
            "SpecializedPostingsWriter::serialize_one_term: Serialized term {:?}",
            std::str::from_utf8(term).unwrap_or("<invalid utf8>")
        );
        Ok(())
    }
}

impl<Rec: Recorder> PostingsWriter for SpecializedPostingsWriter<Rec> {
    #[inline]
    fn subscribe(&mut self, doc: DocId, position: u32, term: &Term, ctx: &mut IndexingContext) {
        println!(
            "SpecializedPostingsWriter::subscribe: Subscribing doc_id {:?}, position {}",
            doc, position
        );
        debug_assert!(term.serialized_term().len() >= 4);
        self.total_num_tokens += 1;
        let (term_index, arena) = (&mut ctx.term_index, &mut ctx.arena);
        term_index.mutate_or_create(term.serialized_term(), |opt_recorder: Option<Rec>| {
            if let Some(mut recorder) = opt_recorder {
                let current_doc = recorder.current_doc();
                println!(
                    "SpecializedPostingsWriter::subscribe: Current recorder doc_id {:?}, new doc_id {:?}",
                    current_doc, doc
                );
                if current_doc != doc {
                    println!(
                        "SpecializedPostingsWriter::subscribe: Document changed from {:?} to {:?}. Closing and starting new doc.",
                        current_doc, doc
                    );
                    recorder.close_doc(arena);
                    recorder.new_doc(doc, arena);
                }
                recorder.record_position(position, arena);
                println!(
                    "SpecializedPostingsWriter::subscribe: Recorded position {} for doc_id {:?}",
                    position, doc
                );
                recorder
            } else {
                println!(
                    "SpecializedPostingsWriter::subscribe: No recorder found. Creating new recorder for doc_id {:?}",
                    doc
                );
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
        println!(
            "SpecializedPostingsWriter::serialize: Starting serialization of {} terms",
            term_addrs.len()
        );
        let mut buffer_lender = BufferLender::default();
        for (_field, _path_id, term, addr) in term_addrs {
            println!(
                "SpecializedPostingsWriter::serialize: Serializing term {:?} at addr {:?}",
                std::str::from_utf8(term).unwrap_or("<invalid utf8>"),
                addr
            );
            if let Err(e) =
                Self::serialize_one_term(term, *addr, &mut buffer_lender, ctx, serializer)
            {
                println!(
                    "SpecializedPostingsWriter::serialize: Failed to serialize term {:?}: {}",
                    std::str::from_utf8(term).unwrap_or("<invalid utf8>"),
                    e
                );
                return Err(e);
            }
        }
        println!("SpecializedPostingsWriter::serialize: Completed serialization of all terms");
        Ok(())
    }

    fn total_num_tokens(&self) -> u64 {
        println!(
            "SpecializedPostingsWriter::total_num_tokens: Total tokens = {}",
            self.total_num_tokens
        );
        self.total_num_tokens
    }
}

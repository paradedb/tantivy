use std::io;

use common::json_path_writer::JSON_END_OF_PATH;
use stacker::Addr;

use crate::indexer::path_to_unordered_id::OrderedPathId;
use crate::postings::postings_writer::SpecializedPostingsWriter;
use crate::postings::recorder::{BufferLender, DocIdRecorder, Recorder};
use crate::postings::{FieldSerializer, IndexingContext, IndexingPosition, PostingsWriter};
use crate::schema::{Field, Type};
use crate::tokenizer::TokenStream;
use crate::{DocId, Term};

/// The `JsonPostingsWriter` is odd in that it relies on a hidden contract:
///
/// `subscribe` is called directly to index non-text tokens, while
/// `index_text` is used to index text.
#[derive(Default)]
pub(crate) struct JsonPostingsWriter<Rec: Recorder> {
    str_posting_writer: SpecializedPostingsWriter<Rec>,
    non_str_posting_writer: SpecializedPostingsWriter<DocIdRecorder>,
}

impl<Rec: Recorder> From<JsonPostingsWriter<Rec>> for Box<dyn PostingsWriter> {
    fn from(json_postings_writer: JsonPostingsWriter<Rec>) -> Box<dyn PostingsWriter> {
        println!(
            "JsonPostingsWriter::from: Converting JsonPostingsWriter<{}> into Box<dyn PostingsWriter>",
            std::any::type_name::<Rec>()
        );
        Box::new(json_postings_writer)
    }
}

impl<Rec: Recorder> PostingsWriter for JsonPostingsWriter<Rec> {
    #[inline]
    fn subscribe(
        &mut self,
        doc: crate::DocId,
        pos: u32,
        term: &crate::Term,
        ctx: &mut IndexingContext,
    ) {
        println!(
            "JsonPostingsWriter::subscribe: Subscribing doc_id {:?}, pos {} ",
            doc, pos
        );
        self.non_str_posting_writer.subscribe(doc, pos, term, ctx);
        println!(
            "JsonPostingsWriter::subscribe: Subscribed doc_id {:?}, pos {}",
            doc, pos
        );
    }

    fn index_text(
        &mut self,
        doc_id: DocId,
        token_stream: &mut dyn TokenStream,
        term_buffer: &mut Term,
        ctx: &mut IndexingContext,
        indexing_position: &mut IndexingPosition,
    ) {
        println!(
            "JsonPostingsWriter::index_text: Indexing text for doc_id {:?}",
            doc_id
        );
        self.str_posting_writer.index_text(
            doc_id,
            token_stream,
            term_buffer,
            ctx,
            indexing_position,
        );
        println!(
            "JsonPostingsWriter::index_text: Completed indexing text for doc_id {:?}",
            doc_id
        );
    }

    /// The actual serialization format is handled by the `PostingsSerializer`.
    fn serialize(
        &self,
        ordered_term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        ordered_id_to_path: &[&str],
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        println!(
            "JsonPostingsWriter::serialize: Starting serialization for {} term_addrs",
            ordered_term_addrs.len()
        );
        let mut term_buffer = Term::with_capacity(48);
        let mut buffer_lender = BufferLender::default();
        term_buffer.clear_with_field_and_type(Type::Json, Field::from_field_id(0));
        let mut prev_term_id = u32::MAX;
        let mut term_path_len = 0; // this will be set in the first iteration

        for (_field, path_id, term, addr) in ordered_term_addrs {
            println!(
                "JsonPostingsWriter::serialize: Processing term {:?}, path_id {:?}, addr {:?}",
                std::str::from_utf8(term).unwrap_or("<invalid utf8>"),
                path_id,
                addr
            );

            if prev_term_id != path_id.path_id() {
                println!(
                    "JsonPostingsWriter::serialize: New path_id {:?} detected. Updating term_buffer.",
                    path_id.path_id()
                );
                term_buffer.truncate_value_bytes(0);
                term_buffer.append_path(ordered_id_to_path[path_id.path_id() as usize].as_bytes());
                term_buffer.append_bytes(&[JSON_END_OF_PATH]);
                term_path_len = term_buffer.len_bytes();
                prev_term_id = path_id.path_id();
                println!(
                    "JsonPostingsWriter::serialize: Updated term_buffer to {:?} with term_path_len {}",
                    term_buffer.value().raw_value_bytes_payload(),
                    term_path_len
                );
            }

            term_buffer.truncate_value_bytes(term_path_len);
            term_buffer.append_bytes(term);
            println!(
                "JsonPostingsWriter::serialize: Appended bytes to term_buffer. Current term_buffer: {:?}",
                    term_buffer.value().raw_value_bytes_payload(),
            );

            if let Some(json_value) = term_buffer.value().as_json_value_bytes() {
                let typ = json_value.typ();
                println!(
                    "JsonPostingsWriter::serialize: Term type {:?} for term {:?}",
                    typ,
                    std::str::from_utf8(term).unwrap_or("<invalid utf8>")
                );
                if typ == Type::Str {
                    println!(
                        "JsonPostingsWriter::serialize: Term type is Str. Serializing as string term."
                    );
                    SpecializedPostingsWriter::<Rec>::serialize_one_term(
                        term_buffer.serialized_value_bytes(),
                        *addr,
                        &mut buffer_lender,
                        ctx,
                        serializer,
                    )?;
                } else {
                    println!(
                        "JsonPostingsWriter::serialize: Term type is {:?}. Serializing as non-string term.",
                        typ
                    );
                    SpecializedPostingsWriter::<DocIdRecorder>::serialize_one_term(
                        term_buffer.serialized_value_bytes(),
                        *addr,
                        &mut buffer_lender,
                        ctx,
                        serializer,
                    )?;
                }
            } else {
                println!(
                    "JsonPostingsWriter::serialize: Term_buffer does not contain valid JSON value bytes."
                );
            }
        }
        println!("JsonPostingsWriter::serialize: Completed serialization");
        Ok(())
    }

    fn total_num_tokens(&self) -> u64 {
        println!(
            "JsonPostingsWriter::total_num_tokens: Total tokens = {}",
            self.str_posting_writer.total_num_tokens()
                + self.non_str_posting_writer.total_num_tokens()
        );
        self.str_posting_writer.total_num_tokens() + self.non_str_posting_writer.total_num_tokens()
    }
}

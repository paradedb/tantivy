use columnar::MonotonicallyMappableToU64;
use common::JsonPathWriter;
use itertools::Itertools;
use tokenizer_api::BoxTokenStream;

use super::operation::AddOperation;
use crate::fastfield::FastFieldsWriter;
use crate::fieldnorm::{FieldNormReaders, FieldNormsWriter};
use crate::index::{Segment, SegmentComponent};
use crate::indexer::segment_serializer::SegmentSerializer;
use crate::json_utils::IndexingPositionsPerPath;
use crate::postings::{
    compute_table_memory_size, serialize_postings, IndexingContext, IndexingPosition,
    PerFieldPostingsWriter, PostingsWriter,
};
use crate::schema::document::{Document, Value};
use crate::schema::{FieldEntry, FieldType, Schema, Term, DATE_TIME_PRECISION_INDEXED};
use crate::tokenizer::{FacetTokenizer, PreTokenizedStream, TextAnalyzer, Tokenizer};
use crate::{DocId, Opstamp, TantivyError};

/// Computes the initial size of the hash table.
///
/// Returns the recommended initial table size as a power of 2.
///
/// Note this is a very dumb way to compute log2, but it is easier to proofread that way.
fn compute_initial_table_size(per_thread_memory_budget: usize) -> crate::Result<usize> {
    let table_memory_upper_bound = per_thread_memory_budget / 3;
    (10..20) // We cap it at 2^19 = 512K capacity.
        // TODO: There are cases where this limit causes a
        // reallocation in the hashmap. Check if this affects performance.
        .map(|power| 1 << power)
        .take_while(|capacity| compute_table_memory_size(*capacity) < table_memory_upper_bound)
        .last()
        .ok_or_else(|| {
            crate::TantivyError::InvalidArgument(format!(
                "per thread memory budget (={per_thread_memory_budget}) is too small. Raise the \
                 memory budget or lower the number of threads."
            ))
        })
}

/// A `SegmentWriter` is in charge of creating segment index from a
/// set of documents.
///
/// They creates the postings list in anonymous memory.
/// The segment is laid on disk when the segment gets `finalized`.
pub struct SegmentWriter {
    pub(crate) max_doc: DocId,
    pub(crate) ctx: IndexingContext,
    pub(crate) per_field_postings_writers: PerFieldPostingsWriter,
    pub(crate) segment_serializer: SegmentSerializer,
    pub(crate) fast_field_writers: FastFieldsWriter,
    pub(crate) fieldnorms_writer: FieldNormsWriter,
    pub(crate) json_path_writer: JsonPathWriter,
    pub(crate) json_positions_per_path: IndexingPositionsPerPath,
    pub(crate) doc_opstamps: Vec<Opstamp>,
    per_field_text_analyzers: Vec<TextAnalyzer>,
    term_buffer: Term,
    schema: Schema,
}

impl SegmentWriter {
    /// Creates a new `SegmentWriter`
    ///
    /// The arguments are defined as follows
    ///
    /// - memory_budget: most of the segment writer data (terms, and postings lists recorders) is
    ///   stored in a memory arena. This makes it possible for the user to define the flushing
    ///   behavior as a memory limit.
    /// - segment: The segment being written
    /// - schema
    pub fn for_segment(memory_budget_in_bytes: usize, segment: Segment) -> crate::Result<Self> {
        let schema = segment.schema();
        let tokenizer_manager = segment.index().tokenizers().clone();
        let tokenizer_manager_fast_field = segment.index().fast_field_tokenizer().clone();
        let table_size = compute_initial_table_size(memory_budget_in_bytes)?;
        let segment_serializer = SegmentSerializer::for_segment(segment)?;
        let per_field_postings_writers = PerFieldPostingsWriter::for_schema(&schema);
        let per_field_text_analyzers = schema
            .fields()
            .map(|(_, field_entry): (_, &FieldEntry)| {
                let text_options = match field_entry.field_type() {
                    FieldType::Str(ref text_options) => text_options.get_indexing_options(),
                    FieldType::JsonObject(ref json_object_options) => {
                        json_object_options.get_text_indexing_options()
                    }
                    _ => None,
                };
                let tokenizer_name = text_options
                    .map(|text_index_option| text_index_option.tokenizer())
                    .unwrap_or("default");

                tokenizer_manager.get(tokenizer_name).ok_or_else(|| {
                    TantivyError::SchemaError(format!(
                        "Error getting tokenizer for field: {}",
                        field_entry.name()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            max_doc: 0,
            ctx: IndexingContext::new(table_size),
            per_field_postings_writers,
            fieldnorms_writer: FieldNormsWriter::for_schema(&schema),
            json_path_writer: JsonPathWriter::default(),
            json_positions_per_path: IndexingPositionsPerPath::default(),
            segment_serializer,
            fast_field_writers: FastFieldsWriter::from_schema_and_tokenizer_manager(
                &schema,
                tokenizer_manager_fast_field,
            )?,
            doc_opstamps: Vec::with_capacity(1_000),
            per_field_text_analyzers,
            term_buffer: Term::with_capacity(16),
            schema,
        })
    }

    /// Lay on disk the current content of the `SegmentWriter`
    ///
    /// Finalize consumes the `SegmentWriter`, so that it cannot
    /// be used afterwards.
    pub fn finalize(mut self) -> crate::Result<Vec<u64>> {
        self.fieldnorms_writer.fill_up_to_max_doc(self.max_doc);
        remap_and_write(
            self.schema,
            &self.per_field_postings_writers,
            self.ctx,
            self.fast_field_writers,
            &self.fieldnorms_writer,
            self.segment_serializer,
        )?;
        Ok(self.doc_opstamps)
    }

    /// Returns an estimation of the current memory usage of the segment writer.
    /// If the mem usage exceeds the `memory_budget`, the segment be serialized.
    pub fn mem_usage(&self) -> usize {
        self.ctx.mem_usage()
            + self.fieldnorms_writer.mem_usage()
            + self.fast_field_writers.mem_usage()
            + self.segment_serializer.mem_usage()
    }

    fn index_document<D: Document>(&mut self, doc: &D) -> crate::Result<()> {
        let doc_id = self.max_doc;
        println!("index_document: Assigning doc_id: {}", doc_id);

        // TODO: Can this be optimised a bit?
        println!("index_document: Iterating and grouping fields and values.");
        let vals_grouped_by_field = doc
            .iter_fields_and_values()
            .sorted_by_key(|(field, _)| *field)
            .chunk_by(|(field, _)| *field);

        // It's useful to know how many field groups we're processing

        for (field, field_values) in &vals_grouped_by_field {
            println!(
                "index_document: Processing field: {:?} (Field ID: {})",
                field,
                field.field_id()
            );

            // Collect the iterator into an owned Vec to break the borrowing chain
            let values: Vec<_> = field_values.map(|el| el.1.clone()).collect();
            println!(
                "index_document: Collected {} values for field {:?}.",
                values.len(),
                self.schema.get_field_name(field)
            );

            let field_entry = self.schema.get_field_entry(field);
            println!(
                "index_document: Retrieved FieldEntry for field {:?}: {:?}",
                field, field_entry
            );

            let make_schema_error = || {
                TantivyError::SchemaError(format!(
                    "Expected a {:?} for field {:?}",
                    field_entry.field_type().value_type(),
                    field_entry.name()
                ))
            };

            if !field_entry.is_indexed() {
                println!(
                    "index_document: Field {:?} is not indexed. Skipping.",
                    field_entry.name()
                );
                continue;
            }
            println!(
                "index_document: Field {:?} is indexed. Proceeding with indexing.",
                field_entry.name()
            );

            let (term_buffer, ctx) = (&mut self.term_buffer, &mut self.ctx);
            let postings_writer: &mut dyn PostingsWriter =
                self.per_field_postings_writers.get_for_field_mut(field);
            println!(
                "index_document: Retrieved PostingsWriter for field {:?}.",
                field_entry.name()
            );

            term_buffer.clear_with_field_and_type(field_entry.field_type().value_type(), field);
            println!(
                "index_document: Cleared term_buffer for field {:?} with type {:?}.",
                field_entry.name(),
                field_entry.field_type().value_type()
            );

            match field_entry.field_type() {
                FieldType::Facet(_) => {
                    println!(
                        "index_document: Handling FieldType::Facet for field {:?}.",
                        field_entry.name()
                    );
                    let mut facet_tokenizer = FacetTokenizer::default(); // this can be global
                    for value in values {
                        println!(
                            "index_document: Processing Facet value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();

                        let facet_str = match value.as_facet() {
                            Some(facet) => facet,
                            None => {
                                println!(
                                "index_document: Error converting value to Facet for field {:?}.",
                                field_entry.name()
                            );
                                return Err(make_schema_error());
                            }
                        };
                        println!("index_document: Facet string extracted: '{}'", facet_str);

                        let mut facet_tokenizer = facet_tokenizer.token_stream(facet_str);
                        let mut indexing_position = IndexingPosition::default();
                        println!(
                            "index_document: Indexing Facet text for field {:?}.",
                            field_entry.name()
                        );
                        postings_writer.index_text(
                            doc_id,
                            &mut facet_tokenizer,
                            term_buffer,
                            ctx,
                            &mut indexing_position,
                        );
                        println!(
                            "index_document: Indexed Facet text with indexing_position: {:?}",
                            indexing_position
                        );
                    }
                }
                FieldType::Str(_) => {
                    println!(
                        "index_document: Handling FieldType::Str for field {:?}.",
                        field_entry.name()
                    );
                    let mut indexing_position = IndexingPosition::default();
                    for value in values {
                        println!(
                            "index_document: Processing Str value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();

                        let mut token_stream = if let Some(text) = value.as_str() {
                            println!(
                            "index_document: Value is a string. Creating token_stream using text analyzer."
                        );
                            let text_analyzer =
                                &mut self.per_field_text_analyzers[field.field_id() as usize];
                            text_analyzer.token_stream(text)
                        } else if let Some(tok_str) = value.into_pre_tokenized_text() {
                            println!(
                            "index_document: Value is pre-tokenized text. Creating PreTokenizedStream."
                        );
                            BoxTokenStream::new(PreTokenizedStream::from(*tok_str.clone()))
                        } else {
                            println!(
                            "index_document: Value does not contain a valid string or pre-tokenized text. Skipping."
                        );
                            continue;
                        };

                        assert!(term_buffer.is_empty());
                        println!(
                        "index_document: Indexing text for field {:?} with term_buffer cleared.",
                        field_entry.name()
                    );
                        postings_writer.index_text(
                            doc_id,
                            &mut *token_stream,
                            term_buffer,
                            ctx,
                            &mut indexing_position,
                        );
                        println!(
                            "index_document: Indexed text with indexing_position: {:?}",
                            indexing_position
                        );
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_tokens: {}.",
                        field_entry.name(),
                        indexing_position.num_tokens
                    );
                        self.fieldnorms_writer
                            .record(doc_id, field, indexing_position.num_tokens);
                    }
                }
                FieldType::U64(_) => {
                    println!(
                        "index_document: Handling FieldType::U64 for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing U64 value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();

                        num_vals += 1;
                        let u64_val = match value.as_u64() {
                            Some(val) => val,
                            None => {
                                println!(
                                    "index_document: Error converting value to u64 for field {:?}.",
                                    field_entry.name()
                                );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: U64 value extracted: {}. Setting term_buffer.",
                            u64_val
                        );
                        term_buffer.set_u64(u64_val);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Date(_) => {
                    println!(
                        "index_document: Handling FieldType::Date for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing Date value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();

                        num_vals += 1;
                        let date_val = match value.as_datetime() {
                            Some(val) => val,
                            None => {
                                println!(
                                "index_document: Error converting value to DateTime for field {:?}.",
                                field_entry.name()
                            );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                        "index_document: DateTime value extracted: {:?}. Truncating and setting term_buffer.",
                        date_val
                    );
                        term_buffer
                            .set_u64(date_val.truncate(DATE_TIME_PRECISION_INDEXED).to_u64());
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::I64(_) => {
                    println!(
                        "index_document: Handling FieldType::I64 for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing I64 value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();

                        num_vals += 1;
                        let i64_val = match value.as_i64() {
                            Some(val) => val,
                            None => {
                                println!(
                                    "index_document: Error converting value to i64 for field {:?}.",
                                    field_entry.name()
                                );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: i64 value extracted: {}. Setting term_buffer.",
                            i64_val
                        );
                        term_buffer.set_i64(i64_val);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::F64(_) => {
                    println!(
                        "index_document: Handling FieldType::F64 for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing F64 value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();
                        num_vals += 1;
                        let f64_val = match value.as_f64() {
                            Some(val) => val,
                            None => {
                                println!(
                                    "index_document: Error converting value to f64 for field {:?}.",
                                    field_entry.name()
                                );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: f64 value extracted: {}. Setting term_buffer.",
                            f64_val
                        );
                        term_buffer.set_f64(f64_val);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Bool(_) => {
                    println!(
                        "index_document: Handling FieldType::Bool for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing Bool value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();
                        num_vals += 1;
                        let bool_val = match value.as_bool() {
                            Some(val) => val,
                            None => {
                                println!(
                                "index_document: Error converting value to bool for field {:?}.",
                                field_entry.name()
                            );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: Bool value extracted: {}. Setting term_buffer.",
                            bool_val
                        );
                        term_buffer.set_bool(bool_val);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Bytes(_) => {
                    println!(
                        "index_document: Handling FieldType::Bytes for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing Bytes value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();
                        num_vals += 1;
                        let bytes = match value.as_bytes() {
                            Some(val) => val,
                            None => {
                                println!(
                                "index_document: Error converting value to bytes for field {:?}.",
                                field_entry.name()
                            );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: Bytes value extracted: {:?}. Setting term_buffer.",
                            bytes
                        );
                        term_buffer.set_bytes(bytes);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::JsonObject(json_options) => {
                    println!(
                    "index_document: Handling FieldType::JsonObject for field {:?} with options {:?}.",
                    field_entry.name(),
                    json_options
                );
                    let text_analyzer =
                        &mut self.per_field_text_analyzers[field.field_id() as usize];
                    println!(
                        "index_document: Retrieved TextAnalyzer for field {:?}.",
                        field_entry.name()
                    );

                    self.json_positions_per_path.clear();
                    println!("index_document: Cleared json_positions_per_path.");

                    self.json_path_writer
                        .set_expand_dots(json_options.is_expand_dots_enabled());
                    println!(
                        "index_document: Set expand_dots to {} in json_path_writer.",
                        json_options.is_expand_dots_enabled()
                    );

                    for json_value in values {
                        println!(
                            "index_document: Processing JSON value for field {:?}: {:?}",
                            field_entry.name(),
                            json_value
                        );
                        self.json_path_writer.clear();
                        println!("index_document: Cleared json_path_writer for new JSON value.");

                        crate::json_utils::index_json_value_nested(
                            doc_id,
                            json_value,
                            text_analyzer,
                            term_buffer,
                            &mut self.json_path_writer,
                            postings_writer,
                            ctx,
                            &mut self.json_positions_per_path,
                            true,
                        );
                        // #[allow(clippy::too_many_arguments)]
                        // index_json_value(
                        //     doc_id,
                        //     json_value,
                        //     text_analyzer,
                        //     term_buffer,
                        //     &mut self.json_path_writer,
                        //     postings_writer,
                        //     ctx,
                        //     &mut self.json_positions_per_path,
                        // );
                        println!(
                            "index_document: Completed indexing JSON value for field {:?}.",
                            field_entry.name()
                        );
                    }
                }
                FieldType::IpAddr(_) => {
                    println!(
                        "index_document: Handling FieldType::IpAddr for field {:?}.",
                        field_entry.name()
                    );
                    let mut num_vals = 0;
                    for value in values {
                        println!(
                            "index_document: Processing IpAddr value for field {:?}: {:?}",
                            field_entry.name(),
                            value
                        );
                        let value = value.as_value();
                        num_vals += 1;
                        let ip_addr = match value.as_ip_addr() {
                            Some(val) => val,
                            None => {
                                println!(
                                "index_document: Error converting value to IpAddr for field {:?}.",
                                field_entry.name()
                            );
                                return Err(make_schema_error());
                            }
                        };
                        println!(
                            "index_document: IpAddr value extracted: {}. Setting term_buffer.",
                            ip_addr
                        );
                        term_buffer.set_ip_addr(ip_addr);
                        println!(
                        "index_document: Subscribing to postings_writer for doc_id: {}, field: {:?}.",
                        doc_id,
                        field_entry.name()
                    );
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        println!(
                        "index_document: Recording fieldnorms for field {:?} with num_vals: {}.",
                        field_entry.name(),
                        num_vals
                    );
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
            }
        }

        println!(
            "index_document: Successfully indexed document with doc_id: {}",
            doc_id
        );
        Ok(())
    }

    /// Indexes a new document
    ///
    /// As a user, you should rather use `IndexWriter`'s add_document.
    pub fn add_document<D: Document>(
        &mut self,
        add_operation: AddOperation<D>,
    ) -> crate::Result<()> {
        let AddOperation { document, opstamp } = add_operation;
        self.doc_opstamps.push(opstamp);
        self.fast_field_writers.add_document(&document)?;
        self.index_document(&document)?;
        let doc_writer = self.segment_serializer.get_store_writer();
        doc_writer.store(&document, &self.schema)?;
        self.max_doc += 1;
        Ok(())
    }

    /// Max doc is
    /// - the number of documents in the segment assuming there is no deletes
    /// - the maximum document id (including deleted documents) + 1
    ///
    /// Currently, **tantivy** does not handle deletes anyway,
    /// so `max_doc == num_docs`
    pub fn max_doc(&self) -> u32 {
        self.max_doc
    }

    /// Number of documents in the index.
    /// Deleted documents are not counted.
    ///
    /// Currently, **tantivy** does not handle deletes anyway,
    /// so `max_doc == num_docs`
    #[allow(dead_code)]
    pub fn num_docs(&self) -> u32 {
        self.max_doc
    }
}

/// This method is used as a trick to workaround the borrow checker
/// Writes a view of a segment by pushing information
/// to the `SegmentSerializer`.
///
/// `doc_id_map` is used to map to the new doc_id order.
fn remap_and_write(
    schema: Schema,
    per_field_postings_writers: &PerFieldPostingsWriter,
    ctx: IndexingContext,
    fast_field_writers: FastFieldsWriter,
    fieldnorms_writer: &FieldNormsWriter,
    mut serializer: SegmentSerializer,
) -> crate::Result<()> {
    debug!("remap-and-write");
    if let Some(fieldnorms_serializer) = serializer.extract_fieldnorms_serializer() {
        fieldnorms_writer.serialize(fieldnorms_serializer)?;
    }
    let fieldnorm_data = serializer
        .segment()
        .open_read(SegmentComponent::FieldNorms)?;
    let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;
    serialize_postings(
        ctx,
        schema,
        per_field_postings_writers,
        fieldnorm_readers,
        serializer.get_postings_serializer(),
    )?;
    debug!("fastfield-serialize");
    fast_field_writers.serialize(serializer.get_fast_field_write())?;

    debug!("serializer-close");
    serializer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use columnar::ColumnType;
    use tempfile::TempDir;

    use crate::collector::{Count, TopDocs};
    use crate::directory::RamDirectory;
    use crate::fastfield::FastValue;
    use crate::postings::{Postings, TermInfo};
    use crate::query::{PhraseQuery, QueryParser};
    use crate::schema::{
        Document, IndexRecordOption, OwnedValue, Schema, TextFieldIndexing, TextOptions, Value,
        DATE_TIME_PRECISION_INDEXED, FAST, STORED, STRING, TEXT,
    };
    use crate::store::{Compressor, StoreReader, StoreWriter};
    use crate::time::format_description::well_known::Rfc3339;
    use crate::time::OffsetDateTime;
    use crate::tokenizer::{PreTokenizedString, Token};
    use crate::{
        DateTime, Directory, DocAddress, DocSet, Index, IndexWriter, SegmentReader,
        TantivyDocument, Term, TERMINATED,
    };

    #[test]
    #[cfg(not(feature = "compare_hash_only"))]
    fn test_hashmap_size() {
        use super::compute_initial_table_size;
        assert_eq!(compute_initial_table_size(100_000).unwrap(), 1 << 12);
        assert_eq!(compute_initial_table_size(1_000_000).unwrap(), 1 << 15);
        assert_eq!(compute_initial_table_size(15_000_000).unwrap(), 1 << 19);
        assert_eq!(compute_initial_table_size(1_000_000_000).unwrap(), 1 << 19);
        assert_eq!(compute_initial_table_size(4_000_000_000).unwrap(), 1 << 19);
    }

    #[test]
    fn test_prepare_for_store() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("title", TEXT | STORED);
        let schema = schema_builder.build();
        let mut doc = TantivyDocument::default();
        let pre_tokenized_text = PreTokenizedString {
            text: String::from("A"),
            tokens: vec![Token {
                offset_from: 0,
                offset_to: 1,
                position: 0,
                text: String::from("A"),
                position_length: 1,
            }],
        };

        doc.add_pre_tokenized_text(text_field, pre_tokenized_text);
        doc.add_text(text_field, "title");

        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path).unwrap();

        let mut store_writer = StoreWriter::new(store_wrt, Compressor::None, 0, false).unwrap();
        store_writer.store(&doc, &schema).unwrap();
        store_writer.close().unwrap();

        let reader = StoreReader::open(directory.open_read(path).unwrap(), 0).unwrap();
        let doc = reader.get::<TantivyDocument>(0).unwrap();

        assert_eq!(doc.field_values().count(), 2);
        assert_eq!(
            doc.get_all(text_field).next().unwrap().as_value().as_str(),
            Some("A")
        );
        assert_eq!(
            doc.get_all(text_field).nth(1).unwrap().as_value().as_str(),
            Some("title")
        );
    }
    #[test]
    fn test_simple_json_indexing() {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer_for_tests().unwrap();
        writer
            .add_document(doc!(json_field=>json!({"my_field": "b"})))
            .unwrap();
        writer
            .add_document(doc!(json_field=>json!({"my_field": "a"})))
            .unwrap();
        writer
            .add_document(doc!(json_field=>json!({"my_field": "b"})))
            .unwrap();
        writer.commit().unwrap();

        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let text_query = query_parser.parse_query("my_field:a").unwrap();
        let score_docs: Vec<(_, DocAddress)> = index
            .reader()
            .unwrap()
            .searcher()
            .search(&text_query, &TopDocs::with_limit(4))
            .unwrap();
        assert_eq!(score_docs.len(), 1);

        let text_query = query_parser.parse_query("my_field:b").unwrap();
        let score_docs: Vec<(_, DocAddress)> = index
            .reader()
            .unwrap()
            .searcher()
            .search(&text_query, &TopDocs::with_limit(4))
            .unwrap();
        assert_eq!(score_docs.len(), 2);
    }

    #[test]
    fn test_flat_json_indexing() {
        // A JSON Object that contains mixed values on the first level
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer_for_tests().unwrap();
        // Text, i64, u64
        writer.add_document(doc!(json_field=>"b")).unwrap();
        writer
            .add_document(doc!(json_field=>OwnedValue::I64(10i64)))
            .unwrap();
        writer
            .add_document(doc!(json_field=>OwnedValue::U64(55u64)))
            .unwrap();
        writer
            .add_document(doc!(json_field=>json!({"my_field": "a"})))
            .unwrap();
        writer.commit().unwrap();

        let search_and_expect = |query| {
            let query_parser = QueryParser::for_index(&index, vec![json_field]);
            let text_query = query_parser.parse_query(query).unwrap();
            let score_docs: Vec<(_, DocAddress)> = index
                .reader()
                .unwrap()
                .searcher()
                .search(&text_query, &TopDocs::with_limit(4))
                .unwrap();
            assert_eq!(score_docs.len(), 1);
        };

        search_and_expect("my_field:a");
        search_and_expect("b");
        search_and_expect("10");
        search_and_expect("55");
    }

    #[test]
    fn test_json_indexing() {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | TEXT);
        let schema = schema_builder.build();
        let json_val: serde_json::Value = serde_json::from_str(
            r#"{
            "toto": "titi",
            "float": -0.2,
            "bool": true,
            "unsigned": 1,
            "signed": -2,
            "complexobject": {
                "field.with.dot": 1
            },
            "date": "1985-04-12T23:20:50.52Z",
            "my_arr": [2, 3, {"my_key": "two tokens"}, 4]
        }"#,
        )
        .unwrap();
        let doc = doc!(json_field=>json_val.clone());
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let doc = searcher
            .doc::<TantivyDocument>(DocAddress {
                segment_ord: 0u32,
                doc_id: 0u32,
            })
            .unwrap();
        let serdeser_json_val = serde_json::from_str::<serde_json::Value>(&doc.to_json(&schema))
            .unwrap()
            .get("json")
            .unwrap()[0]
            .clone();
        assert_eq!(json_val, serdeser_json_val);
        let segment_reader = searcher.segment_reader(0u32);
        let inv_idx = segment_reader.inverted_index(json_field).unwrap();
        let term_dict = inv_idx.terms();

        let mut term_stream = term_dict.stream().unwrap();

        let term_from_path =
            |path: &str| -> Term { Term::from_field_json_path(json_field, path, false) };

        fn set_fast_val<T: FastValue>(val: T, mut term: Term) -> Term {
            term.append_type_and_fast_value(val);
            term
        }
        fn set_str(val: &str, mut term: Term) -> Term {
            term.append_type_and_str(val);
            term
        }

        let term = term_from_path("bool");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(true, term).serialized_value_bytes()
        );

        let term = term_from_path("complexobject.field\\.with\\.dot");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(1i64, term).serialized_value_bytes()
        );

        // Date
        let term = term_from_path("date");

        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(
                DateTime::from_utc(
                    OffsetDateTime::parse("1985-04-12T23:20:50.52Z", &Rfc3339).unwrap(),
                )
                .truncate(DATE_TIME_PRECISION_INDEXED),
                term
            )
            .serialized_value_bytes()
        );

        // Float
        let term = term_from_path("float");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(-0.2f64, term).serialized_value_bytes()
        );

        // Number In Array
        let term = term_from_path("my_arr");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(2i64, term).serialized_value_bytes()
        );

        let term = term_from_path("my_arr");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(3i64, term).serialized_value_bytes()
        );

        let term = term_from_path("my_arr");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(4i64, term).serialized_value_bytes()
        );

        // El in Array
        let term = term_from_path("my_arr.my_key");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_str("tokens", term).serialized_value_bytes()
        );
        let term = term_from_path("my_arr.my_key");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_str("two", term).serialized_value_bytes()
        );

        // Signed
        let term = term_from_path("signed");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(-2i64, term).serialized_value_bytes()
        );

        let term = term_from_path("toto");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_str("titi", term).serialized_value_bytes()
        );
        // Unsigned
        let term = term_from_path("unsigned");
        assert!(term_stream.advance());
        assert_eq!(
            term_stream.key(),
            set_fast_val(1i64, term).serialized_value_bytes()
        );

        assert!(!term_stream.advance());
    }

    #[test]
    fn test_json_tokenized_with_position() {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | TEXT);
        let schema = schema_builder.build();
        let mut doc = TantivyDocument::default();
        let json_val: BTreeMap<String, crate::schema::OwnedValue> =
            serde_json::from_str(r#"{"mykey": "repeated token token"}"#).unwrap();
        doc.add_object(json_field, json_val);
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0u32);
        let inv_index = segment_reader.inverted_index(json_field).unwrap();
        let mut term = Term::from_field_json_path(json_field, "mykey", false);
        term.append_type_and_str("token");
        let term_info = inv_index.get_term_info(&term).unwrap().unwrap();
        assert_eq!(
            term_info,
            TermInfo {
                doc_freq: 1,
                postings_range: 2..4,
                positions_range: 2..5
            }
        );
        let mut postings = inv_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            .unwrap()
            .unwrap();
        assert_eq!(postings.doc(), 0);
        assert_eq!(postings.term_freq(), 2);
        let mut positions = Vec::new();
        postings.positions(&mut positions);
        assert_eq!(&positions[..], &[1, 2]);
        assert_eq!(postings.advance(), TERMINATED);
    }

    #[test]
    fn test_json_raw_no_position() {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STRING);
        let schema = schema_builder.build();
        let json_val: serde_json::Value =
            serde_json::from_str(r#"{"mykey": "two tokens"}"#).unwrap();
        let doc = doc!(json_field=>json_val);
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0u32);
        let inv_index = segment_reader.inverted_index(json_field).unwrap();
        let mut term = Term::from_field_json_path(json_field, "mykey", false);
        term.append_type_and_str("two tokens");
        let term_info = inv_index.get_term_info(&term).unwrap().unwrap();
        assert_eq!(
            term_info,
            TermInfo {
                doc_freq: 1,
                postings_range: 0..1,
                positions_range: 0..0
            }
        );
        let mut postings = inv_index
            .read_postings(&term, IndexRecordOption::WithFreqs)
            .unwrap()
            .unwrap();
        assert_eq!(postings.doc(), 0);
        assert_eq!(postings.term_freq(), 1);
        let mut positions = Vec::new();
        postings.positions(&mut positions);
        assert_eq!(postings.advance(), TERMINATED);
    }

    #[test]
    fn test_position_overlapping_path() {
        // This test checks that we do not end up detecting phrase query due
        // to several string literal in the same json object being overlapping.
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", TEXT);
        let schema = schema_builder.build();
        let json_val: serde_json::Value = serde_json::from_str(
            r#"{"mykey": [{"field": "hello happy tax payer"}, {"field": "nothello"}]}"#,
        )
        .unwrap();
        let doc = doc!(json_field=>json_val);
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let term = Term::from_field_json_path(json_field, "mykey.field", false);

        let mut hello_term = term.clone();
        hello_term.append_type_and_str("hello");

        let mut nothello_term = term.clone();
        nothello_term.append_type_and_str("nothello");

        let mut happy_term = term.clone();
        happy_term.append_type_and_str("happy");

        let phrase_query = PhraseQuery::new(vec![hello_term, happy_term.clone()]);
        assert_eq!(searcher.search(&phrase_query, &Count).unwrap(), 1);
        let phrase_query = PhraseQuery::new(vec![nothello_term, happy_term]);
        assert_eq!(searcher.search(&phrase_query, &Count).unwrap(), 0);
    }

    #[test]
    fn test_json_fast() {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let json_val: serde_json::Value = serde_json::from_str(
            r#"{
            "toto": "titi",
            "float": -0.2,
            "bool": true,
            "unsigned": 1,
            "signed": -2,
            "complexobject": {
                "field.with.dot": 1
            },
            "date": "1985-04-12T23:20:50.52Z",
            "my_arr": [2, 3, {"my_key": "two tokens"}, 4]
        }"#,
        )
        .unwrap();
        let doc = doc!(json_field=>json_val.clone());
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0u32);

        fn assert_type(reader: &SegmentReader, field: &str, typ: ColumnType) {
            let cols = reader.fast_fields().dynamic_column_handles(field).unwrap();
            assert_eq!(cols.len(), 1, "{}", field);
            assert_eq!(cols[0].column_type(), typ, "{}", field);
        }
        assert_type(segment_reader, "json.toto", ColumnType::Str);
        assert_type(segment_reader, "json.float", ColumnType::F64);
        assert_type(segment_reader, "json.bool", ColumnType::Bool);
        assert_type(segment_reader, "json.unsigned", ColumnType::I64);
        assert_type(segment_reader, "json.signed", ColumnType::I64);
        assert_type(
            segment_reader,
            "json.complexobject.field\\.with\\.dot",
            ColumnType::I64,
        );
        assert_type(segment_reader, "json.date", ColumnType::DateTime);
        assert_type(segment_reader, "json.my_arr", ColumnType::I64);
        assert_type(segment_reader, "json.my_arr.my_key", ColumnType::Str);

        fn assert_empty(reader: &SegmentReader, field: &str) {
            let cols = reader.fast_fields().dynamic_column_handles(field).unwrap();
            assert_eq!(cols.len(), 0);
        }
        assert_empty(segment_reader, "unknown");
        assert_empty(segment_reader, "json");
        assert_empty(segment_reader, "json.toto.titi");

        let sub_columns = segment_reader
            .fast_fields()
            .dynamic_subpath_column_handles("json")
            .unwrap();
        assert_eq!(sub_columns.len(), 9);

        let subsub_columns = segment_reader
            .fast_fields()
            .dynamic_subpath_column_handles("json.complexobject")
            .unwrap();
        assert_eq!(subsub_columns.len(), 1);
    }

    #[test]
    fn test_json_term_with_numeric_merge_panic_regression_bug_2283() {
        // https://github.com/quickwit-oss/tantivy/issues/2283
        let mut schema_builder = Schema::builder();
        let json = schema_builder.add_json_field("json", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        let doc = json!({"field": "a"});
        writer.add_document(doc!(json=>doc)).unwrap();
        writer.commit().unwrap();
        let doc = json!({"field": "a", "id": 1});
        writer.add_document(doc!(json=>doc.clone())).unwrap();
        writer.commit().unwrap();

        // Force Merge
        writer.wait_merging_threads().unwrap();
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        let segment_ids = index
            .searchable_segment_ids()
            .expect("Searchable segments failed.");
        index_writer.merge(&segment_ids).wait().unwrap();
        assert!(index_writer.wait_merging_threads().is_ok());
    }

    #[test]
    fn test_bug_regression_1629_position_when_array_with_a_field_value_that_does_not_contain_any_token(
    ) {
        // We experienced a bug where we would have a position underflow when computing position
        // delta in an horrible corner case.
        //
        // See the commit with this unit test if you want the details.
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let doc = TantivyDocument::parse_json(&schema, r#"{"text": [ "bbb", "aaa", "", "aaa"]}"#)
            .unwrap();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer.add_document(doc).unwrap();
        // On debug this did panic on the underflow
        index_writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);
        let inv_index = seg_reader.inverted_index(text).unwrap();
        let term = Term::from_field_text(text, "aaa");
        let mut postings = inv_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            .unwrap()
            .unwrap();
        assert_eq!(postings.doc(), 0u32);
        let mut positions = Vec::new();
        postings.positions(&mut positions);
        // On release this was [2, 1]. (< note the decreasing values)
        assert_eq!(positions, &[2, 5]);
    }

    #[test]
    fn test_multiple_field_value_and_long_tokens() {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let mut doc = TantivyDocument::default();
        // This is a bit of a contrived example.
        let tokens = PreTokenizedString {
            text: "roller-coaster".to_string(),
            tokens: vec![Token {
                offset_from: 0,
                offset_to: 14,
                position: 0,
                text: "rollercoaster".to_string(),
                position_length: 2,
            }],
        };
        doc.add_pre_tokenized_text(text, tokens.clone());
        doc.add_pre_tokenized_text(text, tokens);
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer.add_document(doc).unwrap();
        index_writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);
        let inv_index = seg_reader.inverted_index(text).unwrap();
        let term = Term::from_field_text(text, "rollercoaster");
        let mut postings = inv_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            .unwrap()
            .unwrap();
        assert_eq!(postings.doc(), 0u32);
        let mut positions = Vec::new();
        postings.positions(&mut positions);
        assert_eq!(positions, &[0, 3]); //< as opposed to 0, 2 if we had a position length of 1.
    }

    #[test]
    fn test_last_token_not_ending_last() {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let mut doc = TantivyDocument::default();
        // This is a bit of a contrived example.
        let tokens = PreTokenizedString {
            text: "contrived-example".to_string(), //< I can't think of a use case where this corner case happens in real life.
            tokens: vec![
                Token {
                    // Not the last token, yet ends after the last token.
                    offset_from: 0,
                    offset_to: 14,
                    position: 0,
                    text: "long_token".to_string(),
                    position_length: 3,
                },
                Token {
                    offset_from: 0,
                    offset_to: 14,
                    position: 1,
                    text: "short".to_string(),
                    position_length: 1,
                },
            ],
        };
        doc.add_pre_tokenized_text(text, tokens);
        doc.add_text(text, "hello");
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer.add_document(doc).unwrap();
        index_writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);
        let inv_index = seg_reader.inverted_index(text).unwrap();
        let term = Term::from_field_text(text, "hello");
        let mut postings = inv_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            .unwrap()
            .unwrap();
        assert_eq!(postings.doc(), 0u32);
        let mut positions = Vec::new();
        postings.positions(&mut positions);
        assert_eq!(positions, &[4]); //< as opposed to 3 if we had a position length of 1.
    }

    #[test]
    fn test_show_error_when_tokenizer_not_registered() {
        let text_field_indexing = TextFieldIndexing::default()
            .set_tokenizer("custom_en")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default()
            .set_indexing_options(text_field_indexing)
            .set_stored();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("title", text_options);
        let schema = schema_builder.build();
        let tempdir = TempDir::new().unwrap();
        let tempdir_path = PathBuf::from(tempdir.path());
        Index::create_in_dir(&tempdir_path, schema).unwrap();
        let index = Index::open_in_dir(tempdir_path).unwrap();
        let schema = index.schema();
        let mut index_writer = index.writer(50_000_000).unwrap();
        let title = schema.get_field("title").unwrap();
        let mut document = TantivyDocument::default();
        document.add_text(title, "The Old Man and the Sea");
        index_writer.add_document(document).unwrap();
        let error = index_writer.commit().unwrap_err();
        assert_eq!(
            error.to_string(),
            "Schema error: 'Error getting tokenizer for field: title'"
        );
    }
}

//! parse_json_for_nested_sorted.rs
//!
//! Completely rewritten nested-JSON expansion for Tantivy, ensuring that
//! arrays of objects expand into multiple child docs, never merging them in
//! one single doc. Includes debug logs in every function.

use serde_json::{Map as JsonMap, Value as SerdeValue};

use crate::schema::{
    document::{DocParsingError, OwnedValue},
    Field, FieldType, JsonObjectOptions, NestedJsonObjectOptions, NestedOptions, Schema,
    TantivyDocument,
};
use crate::TantivyError;
use std::collections::HashMap;

/// A small debug container that accumulates string tokens for “NestedJson” fields
/// so we can store them sorted in each doc.
#[derive(Default, Debug)]
struct DocBuffers {
    /// For each field, we collect a list of tokens. We only
    /// finalize them at the end by sorting + joining.
    tokens_per_field: HashMap<Field, Vec<String>>,
}

impl DocBuffers {
    fn new() -> Self {
        println!("[DocBuffers::new] Creating empty DocBuffers");
        DocBuffers {
            tokens_per_field: HashMap::new(),
        }
    }

    /// Appends a set of string tokens to the field’s buffer
    fn push_tokens(&mut self, field: Field, tokens: &[String]) {
        println!(
            "[DocBuffers::push_tokens] field={:?}, adding {} tokens",
            field,
            tokens.len()
        );
        self.tokens_per_field
            .entry(field)
            .or_default()
            .extend_from_slice(tokens);
    }

    /// Once we’re done with a doc, we finalize these tokens by sorting,
    /// joining, and adding them to the doc as a single “joined string”.
    fn flush_to_tantivy_doc(&self, doc: &mut TantivyDocument) {
        println!("[DocBuffers::flush_to_tantivy_doc] Start flushing tokens");
        for (&field, token_list) in &self.tokens_per_field {
            println!(
                "  - Flushing field={:?} with {} tokens => sorting + joining",
                field,
                token_list.len()
            );
            let mut sorted = token_list.clone();
            sorted.sort();
            let joined = sorted.join(" ");
            doc.add_field_value(field, &OwnedValue::Str(joined.clone()));
            println!(
                "  - Wrote joined string to doc for field={:?}: '{}'",
                field, joined
            );
        }
        println!("[DocBuffers::flush_to_tantivy_doc] Done!");
    }
}

/// Parse a top-level JSON string into multiple child docs plus a final parent doc.
/// - Each array-of-objects spawns child docs, so that no single doc merges data
///   from different objects in that array. This prevents “cross talk” in queries.
/// - The parent doc is returned last in the final `Vec<TantivyDocument>`.
/// - We store debug logs in every function, so you can track exactly what is happening.
pub fn parse_json_for_nested_sorted(
    schema: &Schema,

    json_str: &str,
) -> Result<Vec<TantivyDocument>, DocParsingError> {
    println!(
        "[parse_json_for_nested_sorted] Entered with JSON length={}",
        json_str.len()
    );

    // 1) Parse
    let top_val: SerdeValue = match serde_json::from_str(json_str) {
        Ok(v) => {
            println!("[parse_json_for_nested_sorted] Successfully parsed JSON into Value");
            v
        }
        Err(e) => {
            println!("[parse_json_for_nested_sorted] ERROR: {}", e);
            return Err(DocParsingError::InvalidJson(e.to_string()));
        }
    };

    // Expect top-level object
    let objmap = top_val.as_object().ok_or_else(|| {
        println!("[parse_json_for_nested_sorted] top-level JSON is not an object => error");
        DocParsingError::InvalidJson("top-level JSON must be object".to_string())
    })?;

    println!(
        "[parse_json_for_nested_sorted] Found top-level object with {} fields",
        objmap.len()
    );

    // Prepare final results, plus a “parent doc” with buffer
    let mut child_docs: Vec<TantivyDocument> = Vec::new();
    let mut parent_doc = TantivyDocument::default();
    let mut parent_bufs = DocBuffers::new();

    // 2) Recursively parse the top-level object with no prefix
    parse_object_with_prefix(
        schema,
        "",
        objmap,
        &mut parent_doc,
        &mut parent_bufs,
        &mut child_docs,
    )?;

    // 3) final flush => parent doc
    println!("[parse_json_for_nested_sorted] flush parent doc buffers");
    parent_bufs.flush_to_tantivy_doc(&mut parent_doc);

    // 4) push parent doc last
    child_docs.push(parent_doc);

    println!(
        "[parse_json_for_nested_sorted] Completed => returning {} docs",
        child_docs.len()
    );
    Ok(child_docs)
}

/// Recursively parse an object’s fields. If a field is “Nested” or “NestedJson” and has
/// an array-of-objects, we spawn child docs. Otherwise we parse normally.
fn parse_object_with_prefix(
    schema: &Schema,
    prefix: &str,
    obj: &JsonMap<String, SerdeValue>,
    parent_doc: &mut TantivyDocument,
    parent_bufs: &mut DocBuffers,
    child_docs: &mut Vec<TantivyDocument>,
) -> Result<(), DocParsingError> {
    println!(
        "[parse_object_with_prefix] prefix='{}', {} fields",
        prefix,
        obj.len()
    );

    // For each key => build a “full_key”
    for (k, v) in obj {
        let full_key = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{}.{}", prefix, k)
        };
        println!(
            "[parse_object_with_prefix] Processing key='{}' => full_key='{}'",
            k, full_key
        );

        // Check if schema has that field
        let field_opt = schema.get_field(&full_key);
        let field = match field_opt {
            Ok(f) => f,
            Err(_) => {
                println!(
                    "[parse_object_with_prefix] field='{}' not in schema => skip",
                    full_key
                );
                continue;
            }
        };
        let fentry = schema.get_field_entry(field);
        let ftype = fentry.field_type();
        println!(
            "[parse_object_with_prefix] field='{}' found => type={:?}",
            full_key, ftype
        );

        match ftype {
            FieldType::Nested(nested_opts) => {
                println!(
                    "[parse_object_with_prefix] => Nested => calling expand_nested_array_of_objects"
                );
                expand_nested_array_of_objects(
                    schema,
                    &full_key,
                    v,
                    nested_opts.include_in_parent,
                    nested_opts.store_parent_flag,
                    parent_doc,
                    parent_bufs,
                    child_docs,
                )?;
            }
            FieldType::NestedJson(nested_json_opts) => {
                println!(
                    "[parse_object_with_prefix] => NestedJson => calling expand_nested_array_of_objects"
                );
                expand_nested_array_of_objects(
                    schema,
                    &full_key,
                    v,
                    nested_json_opts.nested_opts.include_in_parent,
                    nested_json_opts.nested_opts.store_parent_flag,
                    parent_doc,
                    parent_bufs,
                    child_docs,
                )?;
            }
            _ => {
                println!("[parse_object_with_prefix] => normal field => parse_regular_field");
                parse_regular_field(schema, field, v, parent_doc)?;
            }
        }
    }

    println!("[parse_object_with_prefix] done with prefix='{}'", prefix);
    Ok(())
}

/// The key routine that expands an array-of-objects into separate child docs.
fn expand_nested_array_of_objects(
    schema: &Schema,
    full_key: &str,
    val: &SerdeValue,
    include_in_parent: bool,
    store_parent_flag: bool,
    parent_doc: &mut TantivyDocument,
    parent_bufs: &mut DocBuffers,
    child_docs: &mut Vec<TantivyDocument>,
) -> Result<(), DocParsingError> {
    println!(
        "[expand_nested_array_of_objects] field='{}', include_in_parent={}, store_parent_flag={}",
        full_key, include_in_parent, store_parent_flag
    );

    // If store_parent_flag => set `_is_parent_<full_key>`=true in the parent doc
    if store_parent_flag {
        let parent_flag_name = format!("_is_parent_{}", full_key);
        if let Ok(flag_field) = schema.get_field(&parent_flag_name) {
            println!(
                "[expand_nested_array_of_objects] => set parent doc bool '{}' = true",
                parent_flag_name
            );
            parent_doc.add_field_value(flag_field, &OwnedValue::from(true));
        } else {
            println!(
                "[expand_nested_array_of_objects] => no field '{}' => skip parent_flag",
                parent_flag_name
            );
        }
    }

    // If val is an array => expand each array item into a child doc if it’s an object
    if let Some(arr) = val.as_array() {
        println!(
            "[expand_nested_array_of_objects] => array with {} elements => scanning",
            arr.len()
        );
        for (idx, element) in arr.iter().enumerate() {
            println!(
                "[expand_nested_array_of_objects] array elem #{} => value={:?}",
                idx, element
            );
            if let Some(obj) = element.as_object() {
                // If include_in_parent => flatten a copy into parent doc
                if include_in_parent {
                    println!(
                        "[expand_nested_array_of_objects] => include_in_parent => storing text for field='{}'",
                        full_key
                    );
                    parent_doc.add_text(field_for_key(schema, full_key)?, &element.to_string());
                    gather_nestedjson_strings_if_applicable(schema, full_key, element, parent_bufs);
                }
                // Build a child doc
                println!(
                    "[expand_nested_array_of_objects] => building child doc #{}",
                    idx
                );
                let mut child_doc = TantivyDocument::default();
                let mut child_bufs = DocBuffers::new();

                // For NestedJson with “stored: true”, store the raw JSON in the child doc as well
                store_raw_json_if_nestedjson(schema, full_key, element, &mut child_doc);

                // Gather tokens if it’s NestedJson
                gather_nestedjson_strings_if_applicable(schema, full_key, element, &mut child_bufs);

                // Then parse subfields recursively
                parse_object_with_prefix(
                    schema,
                    full_key,
                    obj,
                    &mut child_doc,
                    &mut child_bufs,
                    child_docs,
                )?;

                // finalize child’s tokens
                child_bufs.flush_to_tantivy_doc(&mut child_doc);

                // push child
                child_docs.push(child_doc);
            } else {
                // If array item is scalar => store in parent if “include_in_parent”
                println!("[expand_nested_array_of_objects] => scalar => store in parent if needed");
                if include_in_parent {
                    let field = field_for_key(schema, full_key)?;
                    parent_doc.add_text(field, &element.to_string());
                }
            }
        }
    }
    // If single object => one child doc
    else if let Some(obj) = val.as_object() {
        println!("[expand_nested_array_of_objects] => single object => building child doc");
        if include_in_parent {
            let field = field_for_key(schema, full_key)?;
            parent_doc.add_text(field, &val.to_string());
        }

        let mut child_doc = TantivyDocument::default();
        let mut child_bufs = DocBuffers::new();

        store_raw_json_if_nestedjson(schema, full_key, val, &mut child_doc);
        gather_nestedjson_strings_if_applicable(schema, full_key, val, &mut child_bufs);

        parse_object_with_prefix(
            schema,
            full_key,
            obj,
            &mut child_doc,
            &mut child_bufs,
            child_docs,
        )?;
        child_bufs.flush_to_tantivy_doc(&mut child_doc);

        child_docs.push(child_doc);
    }
    // If it’s a scalar => store in parent doc if “include_in_parent”
    else {
        println!("[expand_nested_array_of_objects] => scalar => store in parent if needed");
        if include_in_parent {
            let field = field_for_key(schema, full_key)?;
            parent_doc.add_text(field, &val.to_string());
        }
    }

    println!(
        "[expand_nested_array_of_objects] done for field='{}'",
        full_key
    );
    Ok(())
}

/// For a “field name” => retrieve the Field from schema or fail
fn field_for_key(schema: &Schema, key: &str) -> Result<Field, DocParsingError> {
    println!("[field_for_key] => key='{}'", key);
    schema
        .get_field(key)
        .map_err(|e| DocParsingError::InvalidJson(e.to_string()))
}

/// If the field is a `NestedJson` with “stored: true”, store the raw JSON string into the doc.
fn store_raw_json_if_nestedjson(
    schema: &Schema,
    full_key: &str,
    val: &SerdeValue,
    child_doc: &mut TantivyDocument,
) {
    println!(
        "[store_raw_json_if_nestedjson] => checking if field='{}' is NestedJson + stored",
        full_key
    );
    if let Ok(f) = schema.get_field(full_key) {
        if let FieldType::NestedJson(nj) = schema.get_field_entry(f).field_type() {
            if nj.json_opts.is_stored() {
                println!(
                    "[store_raw_json_if_nestedjson] => yes, storing raw JSON => field={:?}",
                    f
                );
                child_doc.add_text(f, &val.to_string());
            }
        }
    }
}

/// If it’s NestedJson => gather tokens from subobject recursively
fn gather_nestedjson_strings_if_applicable(
    schema: &Schema,
    full_key: &str,
    val: &SerdeValue,
    doc_bufs: &mut DocBuffers,
) {
    println!(
        "[gather_nestedjson_strings_if_applicable] => field='{}', val={:?}",
        full_key, val
    );
    if let Ok(f) = schema.get_field(full_key) {
        if let FieldType::NestedJson(nj) = schema.get_field_entry(f).field_type() {
            println!(
                "  => It's NestedJson => gather all strings recursively for field={:?}",
                f
            );
            let mut v = Vec::new();
            gather_strings_recursively(val, &mut v);
            doc_bufs.push_tokens(f, &v);
        } else {
            println!("  => not NestedJson => skip gather");
        }
    } else {
        println!("  => field not found => skip gather");
    }
}

/// Actually gather all strings (recursively) from a JSON value, ignoring numeric/bool, etc.
fn gather_strings_recursively(val: &SerdeValue, out: &mut Vec<String>) {
    println!("[gather_strings_recursively] => val={:?}", val);
    match val {
        SerdeValue::String(s) => {
            println!("  => found string='{}'", s);
            out.push(s.clone());
        }
        SerdeValue::Array(arr) => {
            println!("  => found array with {} elems => rec", arr.len());
            for elem in arr {
                gather_strings_recursively(elem, out);
            }
        }
        SerdeValue::Object(obj) => {
            println!("  => found object with {} keys => rec", obj.len());
            for (_k, subval) in obj {
                gather_strings_recursively(subval, out);
            }
        }
        _ => {
            println!("  => ignoring non-string scalar");
        }
    }
}

/// For normal (non-nested) fields => parse once, store once
fn parse_regular_field(
    schema: &Schema,
    field: crate::schema::Field,
    val: &SerdeValue,
    parent_doc: &mut TantivyDocument,
) -> Result<(), DocParsingError> {
    println!("[parse_regular_field] => field={:?}, val={:?}", field, val);
    let fentry = schema.get_field_entry(field);
    // Convert from JSON => OwnedValue
    let typed_val = fentry
        .field_type()
        .value_from_json_non_nested(val.clone())
        .map_err(|e| {
            println!("[parse_regular_field] => error converting => {}", e);
            DocParsingError::ValueError(schema.get_field_name(field).to_string(), e)
        })?;
    println!("[parse_regular_field] => succeeded => adding to doc");
    parent_doc.add_field_value(field, &typed_val);
    Ok(())
}

#[cfg(test)]
mod test_sorted_expansion {
    use crate::collector::Count;
    use crate::query::{ParentBitSetProducer, QueryParser, ScoreMode, ToParentBlockJoinQuery};
    use crate::schema::JsonObjectOptions;
    use crate::schema::{
        document::TantivyDocument, IndexRecordOption, NestedJsonObjectOptions, NestedOptions,
        SchemaBuilder, TextFieldIndexing, STORED,
    };
    use crate::tokenizer::{SimpleTokenizer, TextAnalyzer};
    use crate::{DocSet, Index, IndexWriter, Term};
    use serde_json::json;

    // We re-import our own parse function
    use super::parse_json_for_nested_sorted;

    /// Minimal “ParentBitSetProducer” that picks out `_is_parent_user=true` docs as parents
    pub struct NestedParentBitSetProducer {
        parent_field: crate::schema::Field,
    }

    impl NestedParentBitSetProducer {
        pub fn new(parent_field: crate::schema::Field) -> Self {
            Self { parent_field }
        }
    }

    impl ParentBitSetProducer for NestedParentBitSetProducer {
        fn produce(&self, reader: &crate::SegmentReader) -> crate::Result<common::BitSet> {
            let max_doc = reader.max_doc();
            let mut bitset = common::BitSet::with_max_value(max_doc);
            let inverted = reader.inverted_index(self.parent_field)?;
            let term_true = Term::from_field_bool(self.parent_field, true);
            if let Some(mut postings) =
                inverted.read_postings(&term_true, IndexRecordOption::Basic)?
            {
                while postings.doc() != crate::TERMINATED {
                    bitset.insert(postings.doc());
                    postings.advance();
                }
            }
            Ok(bitset)
        }
    }

    #[test]
    fn test_avoids_out_of_order_tokens_fixed() -> crate::Result<()> {
        // 1) Schema with user => NestedJson

        let mut builder = SchemaBuilder::default();
        let nested_opts = NestedOptions::new()
            .set_include_in_parent(true)
            .set_store_parent_flag(true);

        let json_opts = JsonObjectOptions::default()
            .set_stored()
            .set_indexing_options(TextFieldIndexing::default());

        let nested_json_opts = NestedJsonObjectOptions {
            nested_opts,
            json_opts,
        };

        let user_field = builder.add_nested_json_field("user", nested_json_opts);

        let schema = builder.build();

        // 2) Create index in memory

        let index = Index::create_in_ram(schema.clone());

        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));

        // 3) We'll have user => array-of-objects => ["white"] etc.

        let json_doc = json!({
            "user": [
                { "names": ["white", "alice"] }
            ]
        });
        let json_str = serde_json::to_string(&json_doc).unwrap();

        {
            let mut writer: IndexWriter<TantivyDocument> = index.writer_for_tests()?;

            // Expand into child+parent docs

            let expanded_docs = parse_json_for_nested_sorted(&schema, &json_str)?;

            // Add them, ensuring the child doc is doc #0, parent doc is doc #1
            let docs: Vec<_> = expanded_docs.into_iter().map(Into::into).collect();

            writer.add_documents(docs)?;

            writer.commit()?;
        }

        // 4) We'll do child-level queries => block-join => confirm the parent doc is found

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let qp = QueryParser::for_index(&index, vec![user_field]);

        for token in &["white", "alice", "names:white", "names:alice"] {
            let child_query = qp.parse_query(token)?;

            let parent_field = schema
                .get_field("_is_parent_user")
                .expect("auto-created by store_parent_flag=true");

            let producer = std::sync::Arc::new(NestedParentBitSetProducer::new(parent_field));

            let block_join = ToParentBlockJoinQuery::new(child_query, producer, ScoreMode::None);

            let count = searcher.search(&block_join, &Count)?;

            assert_eq!(
                1, count,
                "Expected 1 parent doc matching child token '{}'",
                token
            );
        }

        Ok(())
    }
}

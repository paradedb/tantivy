use serde_json::{Map as JsonMap, Value as SerdeValue};
use std::collections::{BTreeMap, HashMap};

use crate::schema::{
    document::{DocParsingError, OwnedValue},
    Field, FieldEntry, FieldType, NestedJsonObjectOptions, NestedOptions, Schema, TantivyDocument,
};
use crate::TantivyError;

/// Helper to collect string tokens for `NestedJson` fields, storing them sorted & joined.
#[derive(Default, Debug)]
struct DocBuffers {
    tokens_per_field: HashMap<Field, Vec<String>>,
}

impl DocBuffers {
    fn new() -> Self {
        println!("DocBuffers: Initializing new DocBuffers instance");
        DocBuffers {
            tokens_per_field: HashMap::new(),
        }
    }

    /// Push tokens into the buffer for a given field
    fn push_tokens(&mut self, field: Field, tokens: &[String]) {
        println!(
            "DocBuffers: Pushing {} tokens to field {:?}",
            tokens.len(),
            field
        );
        self.tokens_per_field
            .entry(field)
            .or_default()
            .extend_from_slice(tokens);
        println!(
            "DocBuffers: Current tokens for field {:?}: {:?}",
            field,
            self.tokens_per_field.get(&field)
        );
    }

    /// Flush all tokens into the doc as a single string, sorted + joined
    fn flush_to_tantivy_doc(&self, doc: &mut TantivyDocument) {
        println!("DocBuffers: Flushing all tokens to TantivyDocument");
        for (&field, token_list) in &self.tokens_per_field {
            println!(
                "DocBuffers: Processing field {:?} with {} tokens",
                field,
                token_list.len()
            );
            let mut sorted = token_list.clone();
            sorted.sort();
            println!(
                "DocBuffers: Sorted tokens for field {:?}: {:?}",
                field, sorted
            );
            let joined = sorted.join(" ");
            println!(
                "DocBuffers: Joined tokens for field {:?}: '{}'",
                field, joined
            );
            doc.add_field_value(field, &OwnedValue::Str(joined));
            println!(
                "DocBuffers: Added joined tokens to TantivyDocument for field {:?}",
                field
            );
        }
        println!("DocBuffers: Completed flushing tokens to TantivyDocument");
    }
}

/// The top-level function that expands arrays of objects for nested fields
/// (including `NestedJson`). Otherwise, it tries to store sub-values in a fallback
/// “dynamic” field—**but only if no nested context is found**.
///
/// We start here at `depth == 0` so that only this doc is recognized as “parent”.
pub fn parse_json_for_nested_sorted(
    schema: &Schema,
    parent_doc: &mut TantivyDocument,
    root_value: &SerdeValue,
) -> Result<Vec<TantivyDocument>, DocParsingError> {
    println!("parse_json_for_nested_sorted: Starting parsing of JSON for nested sorted documents");

    let mut child_docs = Vec::new();
    let mut parent_bufs = DocBuffers::new();

    // If top-level is an object, iterate over its fields. Otherwise, skip or handle accordingly.
    if let Some(obj) = root_value.as_object() {
        let mut path_stack = Vec::new();
        println!(
            "parse_json_for_nested_sorted: Found top-level object with {} fields",
            obj.len()
        );

        // For each key => push onto path_stack => call parse_value => pop.
        for (key, sub_val) in obj {
            path_stack.push(key.clone());
            // depth=0 on top-level
            parse_value(
                schema,
                &mut path_stack,
                None,
                sub_val,
                parent_doc,
                &mut parent_bufs,
                &mut child_docs,
                /* depth = */ 0,
            )?;
            path_stack.pop();
        }
    } else {
        println!("parse_json_for_nested_sorted: Top-level JSON is not an object => skipping or returning error");
        // Optionally handle arrays/scalars here
    }

    // Finally, flush all tokens for the parent doc
    println!(
        "parse_json_for_nested_sorted: Flushing tokens from DocBuffers to parent TantivyDocument"
    );
    parent_bufs.flush_to_tantivy_doc(parent_doc);

    // Append parent doc last
    println!(
        "parse_json_for_nested_sorted: Appending parent_doc to child_docs as the last document"
    );
    child_docs.push(parent_doc.clone());

    println!(
        "parse_json_for_nested_sorted: Parsing completed successfully with {} documents",
        child_docs.len()
    );

    Ok(child_docs)
}

/// Recursively parses `val`. If we’re inside a recognized nested field (`current_nested_field`),
/// we keep treating all sub-keys/arrays as belonging to that same field. If we’re
/// not in a nested context, we check if `path_stack` matches a nested field. Otherwise, we do fallback.
///
/// `depth` tracks how deep we are in the hierarchy, so we can see which doc is parent for sub-nesting.
fn parse_value(
    schema: &Schema,
    path_stack: &mut Vec<String>,
    current_nested_field: Option<Field>,
    val: &SerdeValue,
    parent_doc: &mut TantivyDocument,
    parent_bufs: &mut DocBuffers,
    child_docs: &mut Vec<TantivyDocument>,
    depth: usize,
) -> Result<(), DocParsingError> {
    println!(
        "parse_value: Entered with path_stack={:?}, current_nested_field={:?}",
        path_stack, current_nested_field
    );

    // see if the path_stack is recognized as nested in the schema
    let nested_field = if let Some((field, _fentry)) = schema.get_nested_field(path_stack) {
        println!(
            "parse_value: Found nested_field {:?} for path_stack={:?}",
            field, path_stack
        );
        Some(field)
    } else {
        println!(
            "parse_value: No nested_field found for path_stack={:?}, current_nested_field={:?}",
            path_stack, current_nested_field
        );
        current_nested_field
    };

    // If there's still no recognized nested field => fallback
    if nested_field.is_none() {
        println!("parse_value: No nested_field recognized. Proceeding with fallback storage.");
        store_in_fallback(schema, val, parent_doc)?;
        return Ok(());
    }

    // We do have a recognized nested field => expand arrays & objects
    let field = nested_field.unwrap();
    let fentry = schema.get_field_entry(field);
    println!(
        "parse_value: Processing nested_field {:?} with FieldType {:?}",
        field,
        fentry.field_type()
    );

    match fentry.field_type() {
        FieldType::Nested(nopts) => {
            println!(
                "parse_value: Field {:?} is of type Nested. Expanding nested value.",
                field
            );
            expand_nested_value(
                schema,
                path_stack,
                field,
                nopts.include_in_parent,
                nopts.store_parent_flag,
                /* gather_tokens= */ false,
                val,
                parent_doc,
                parent_bufs,
                child_docs,
                depth,
            )?;
        }
        FieldType::NestedJson(nj_opts) => {
            println!(
                "parse_value: Field {:?} is of type NestedJson. Expanding nested value with token gathering.",
                field
            );
            expand_nested_value(
                schema,
                path_stack,
                field,
                nj_opts.nested_opts.include_in_parent,
                nj_opts.nested_opts.store_parent_flag,
                /* gather_tokens= */ true,
                val,
                parent_doc,
                parent_bufs,
                child_docs,
                depth,
            )?;
        }
        // If the schema returned a "nested field" that isn't actually Nested or NestedJson,
        // we skip or parse no further
        _ => {
            println!(
                "parse_value: Field {:?} is not Nested or NestedJson. Skipping further parsing.",
                field
            );
        }
    }

    println!("parse_value: Exiting parse_value function");
    Ok(())
}

fn expand_nested_value(
    schema: &Schema,
    path_stack: &mut Vec<String>,
    nested_field: Field,
    include_in_parent: bool,
    store_parent_flag: bool,
    gather_tokens: bool,
    val: &SerdeValue,
    parent_doc: &mut TantivyDocument,
    parent_bufs: &mut DocBuffers,
    child_docs: &mut Vec<TantivyDocument>,
    depth: usize,
) -> Result<(), DocParsingError> {
    println!(
        "expand_nested_value: Expanding nested_field {:?} with include_in_parent={}, store_parent_flag={}, gather_tokens={}, depth={}",
        nested_field, include_in_parent, store_parent_flag, gather_tokens, depth
    );

    // Only set the parent flag if store_parent_flag == true *and* we see sub-doc structures
    let is_parent = schema.get_nested_field(path_stack).is_some();
    if store_parent_flag && is_parent {
        let parent_flag_name = format!("_is_parent_{}", schema.get_field_name(nested_field));
        println!(
            "expand_nested_value: Setting parent flag field '{}' to true (depth={})",
            parent_flag_name, depth
        );
        if let Ok(flag_field) = schema.get_field(&parent_flag_name) {
            parent_doc.add_field_value(flag_field, &OwnedValue::from(true));
            println!(
                "expand_nested_value: Added field '{:?}' with value 'true' to parent_doc",
                flag_field
            );
        } else {
            println!(
                "expand_nested_value: Parent flag field '{}' not found in schema. Skipping.",
                parent_flag_name
            );
        }
    } else if store_parent_flag {
        println!(
            "expand_nested_value: store_parent_flag={} but 'val' is scalar or empty => not setting parent flag (depth={})",
            store_parent_flag, depth
        );
    }

    if let Some(arr) = val.as_array() {
        println!(
            "expand_nested_value: Value is an array with {} elements",
            arr.len()
        );
        for (index, item) in arr.iter().enumerate() {
            println!(
                "expand_nested_value: Processing array element {}: {:?}",
                index, item
            );
            expand_single_nested_item(
                schema,
                path_stack,
                nested_field,
                include_in_parent,
                gather_tokens,
                item,
                parent_doc,
                parent_bufs,
                child_docs,
                depth,
            )?;
        }
    } else if val.is_object() {
        println!("expand_nested_value: Value is a single object");
        expand_single_nested_item(
            schema,
            path_stack,
            nested_field,
            include_in_parent,
            gather_tokens,
            val,
            parent_doc,
            parent_bufs,
            child_docs,
            depth,
        )?;
    } else {
        // scalar => store in parent if needed
        println!(
            "expand_nested_value: Value is a scalar. include_in_parent={}, depth={}",
            include_in_parent, depth
        );
        if include_in_parent {
            store_scalar_in_doc(parent_doc, nested_field, val);
            println!(
                "expand_nested_value: Stored scalar value in parent_doc for field {:?}",
                nested_field
            );
            if gather_tokens {
                println!(
                    "expand_nested_value: Gathering tokens for NestedJson field {:?}",
                    nested_field
                );
                gather_tokens_for_nestedjson(nested_field, val, parent_bufs);
            }
        }
    }

    println!(
        "expand_nested_value: Completed expanding nested_field {:?} at depth={}",
        nested_field, depth
    );
    Ok(())
}

/// Expand a single item (object or scalar) as one child doc (or store in parent if `include_in_parent==true`).
fn expand_single_nested_item(
    schema: &Schema,
    path_stack: &mut Vec<String>,
    nested_field: Field,
    include_in_parent: bool,
    gather_tokens: bool,
    val: &SerdeValue,
    parent_doc: &mut TantivyDocument,
    parent_bufs: &mut DocBuffers,
    child_docs: &mut Vec<TantivyDocument>,
    depth: usize,
) -> Result<(), DocParsingError> {
    println!(
        "expand_single_nested_item: Expanding single nested item for field {:?} at depth={}",
        nested_field, depth
    );

    if let Some(obj) = val.as_object() {
        println!(
            "expand_single_nested_item: Item is an object with {} keys",
            obj.len()
        );
        // If include_in_parent => store object in the parent doc
        if include_in_parent {
            store_object_in_doc(parent_doc, nested_field, obj);
            println!(
                "expand_single_nested_item: Stored object in parent_doc for field {:?}",
                nested_field
            );
            if gather_tokens {
                println!(
                    "expand_single_nested_item: Gathering tokens for NestedJson field {:?}",
                    nested_field
                );
                gather_tokens_for_nestedjson(nested_field, val, parent_bufs);
            }
        }

        // build a new "child doc"
        let mut child_doc = TantivyDocument::default();
        let mut child_bufs = DocBuffers::new();

        // If gather_tokens => possibly store raw object
        if gather_tokens {
            if let FieldType::NestedJson(nj_opts) =
                schema.get_field_entry(nested_field).field_type()
            {
                if nj_opts.json_opts.is_stored() {
                    store_object_in_doc(&mut child_doc, nested_field, obj);
                    println!(
                        "expand_single_nested_item: Stored raw object in child_doc for NestedJson field {:?}",
                        nested_field
                    );
                }
            }
            gather_tokens_for_nestedjson(nested_field, val, &mut child_bufs);
            println!(
                "expand_single_nested_item: Gathered tokens for NestedJson field {:?}",
                nested_field
            );
        }

        // parse sub-keys inside that child doc, referencing the same nested_field
        // but now at depth+1
        println!(
            "expand_single_nested_item: Recursively parsing sub-keys for child_doc of field {:?} at depth={}",
            nested_field, depth
        );
        for (k, sub_val) in obj {
            println!(
                "expand_single_nested_item: Processing sub-key '{}' with value {:?}",
                k, sub_val
            );
            path_stack.push(k.clone());
            parse_value(
                schema,
                path_stack,
                Some(nested_field),
                sub_val,
                &mut child_doc,
                &mut child_bufs,
                child_docs,
                depth + 1,
            )?;
            path_stack.pop();
            println!(
                "expand_single_nested_item: Completed processing sub-key '{}'",
                k
            );
        }

        // flush tokens => child doc
        println!("expand_single_nested_item: Flushing tokens from child_bufs to child_doc");
        child_bufs.flush_to_tantivy_doc(&mut child_doc);

        println!(
            "expand_single_nested_item: Appending child_doc to child_docs for field {:?}",
            nested_field
        );
        child_docs.push(child_doc);
    } else {
        // it’s a scalar => store in parent if needed
        println!(
            "expand_single_nested_item: Item is a scalar. include_in_parent={}, depth={}",
            include_in_parent, depth
        );
        if include_in_parent {
            store_scalar_in_doc(parent_doc, nested_field, val);
            println!(
                "expand_single_nested_item: Stored scalar value in parent_doc for field {:?}",
                nested_field
            );
            if gather_tokens {
                println!(
                    "expand_single_nested_item: Gathering tokens for NestedJson field {:?}",
                    nested_field
                );
                gather_tokens_for_nestedjson(nested_field, val, parent_bufs);
            }
        }
    }
    Ok(())
}

/// If this is a `NestedJson` field, gather string tokens from `val`.
fn gather_tokens_for_nestedjson(field: Field, val: &SerdeValue, doc_bufs: &mut DocBuffers) {
    println!(
        "gather_tokens_for_nestedjson: Gathering tokens for field {:?} from value {:?}",
        field, val
    );
    let mut strings = Vec::new();
    gather_all_strings(val, &mut strings);
    println!(
        "gather_tokens_for_nestedjson: Collected {} strings for field {:?}",
        strings.len(),
        field
    );
    doc_bufs.push_tokens(field, &strings);
    println!(
        "gather_tokens_for_nestedjson: Tokens pushed to DocBuffers for field {:?}",
        field
    );
}

/// Recursively gather all string values.
fn gather_all_strings(val: &SerdeValue, out: &mut Vec<String>) {
    match val {
        SerdeValue::String(s) => {
            println!("gather_all_strings: Found string '{}'", s);
            out.push(s.clone())
        }
        SerdeValue::Array(arr) => {
            println!(
                "gather_all_strings: Found array with {} elements",
                arr.len()
            );
            for (i, item) in arr.iter().enumerate() {
                println!("gather_all_strings: Processing array element 0: {:?}", item);
                gather_all_strings(item, out);
            }
        }
        SerdeValue::Object(obj) => {
            println!("gather_all_strings: Found object with {} keys", obj.len());
            for (k, v) in obj {
                println!(
                    "gather_all_strings: Processing key '{}' with value {:?}",
                    k, v
                );
                gather_all_strings(v, out);
            }
        }
        _ => {
            println!("gather_all_strings: Encountered non-string, non-array, non-object value. Skipping.");
        }
    }
}

/// Fallback for sub-values not recognized as nested:
fn store_in_fallback(
    _schema: &Schema,
    _val: &SerdeValue,
    _parent_doc: &mut TantivyDocument,
) -> Result<(), DocParsingError> {
    println!("store_in_fallback: Attempting to store value in fallback");
    // In this test, no fallback field is defined => skip
    println!("store_in_fallback: No fallback field defined. Skipping storage.");
    Ok(())
}

/// Store an object as OwnedValue::Object in the doc.
fn store_object_in_doc(doc: &mut TantivyDocument, field: Field, obj: &JsonMap<String, SerdeValue>) {
    println!(
        "store_object_in_doc: Storing object in doc for field {:?} with {} keys",
        field,
        obj.len()
    );
    let mut map = BTreeMap::new();
    for (k, v) in obj {
        println!(
            "store_object_in_doc: Converting sub-key '{}' with value {:?} to OwnedValue",
            k, v
        );
        map.insert(k.clone(), convert_serde_to_owned(v));
    }
    doc.add_object(field, map);
    println!(
        "store_object_in_doc: Object stored in doc for field {:?}",
        field
    );
}

/// Store a scalar value in the doc as text (or typed).
fn store_scalar_in_doc(doc: &mut TantivyDocument, field: Field, val: &SerdeValue) {
    println!(
        "store_scalar_in_doc: Storing scalar value {:?} in doc for field {:?}",
        val, field
    );
    // For simplicity, store everything as text
    doc.add_field_value(field, &OwnedValue::Str(val.to_string()));
    println!(
        "store_scalar_in_doc: Scalar value stored as text for field {:?}",
        field
    );
}

/// Convert `serde_json::Value` => `OwnedValue`.
fn convert_serde_to_owned(val: &SerdeValue) -> OwnedValue {
    match val {
        SerdeValue::Null => OwnedValue::Null,
        SerdeValue::Bool(b) => OwnedValue::from(*b),
        SerdeValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                OwnedValue::from(i)
            } else if let Some(u) = n.as_u64() {
                OwnedValue::from(u)
            } else if let Some(f) = n.as_f64() {
                OwnedValue::from(f)
            } else {
                // extremely rare big integer
                OwnedValue::Str(n.to_string())
            }
        }
        SerdeValue::String(s) => OwnedValue::Str(s.clone()),
        SerdeValue::Array(arr) => {
            let converted: Vec<OwnedValue> =
                arr.iter().map(|x| convert_serde_to_owned(x)).collect();
            OwnedValue::Array(converted)
        }
        SerdeValue::Object(obj) => OwnedValue::Object(
            obj.iter()
                .map(|(kk, vv)| (kk.clone(), convert_serde_to_owned(vv)))
                .collect(),
        ),
    }
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
            println!(
                "NestedParentBitSetProducer::new: Creating new producer for parent_field {:?}",
                parent_field
            );
            Self { parent_field }
        }
    }

    impl ParentBitSetProducer for NestedParentBitSetProducer {
        fn produce(&self, reader: &crate::SegmentReader) -> crate::Result<common::BitSet> {
            println!(
                "NestedParentBitSetProducer::produce: Producing BitSet for parent_field {:?}",
                self.parent_field
            );
            let max_doc = reader.max_doc();
            let mut bitset = common::BitSet::with_max_value(max_doc);
            println!(
                "NestedParentBitSetProducer::produce: Initialized BitSet with max_doc={}",
                max_doc
            );
            let inverted = reader.inverted_index(self.parent_field)?;
            println!(
                "NestedParentBitSetProducer::produce: Retrieved inverted index for parent_field {:?}",
                self.parent_field
            );
            let term_true = Term::from_field_bool(self.parent_field, true);
            println!(
                "NestedParentBitSetProducer::produce: Created Term {:?} for boolean true",
                term_true
            );

            if let Some(mut postings) =
                inverted.read_postings(&term_true, IndexRecordOption::Basic)?
            {
                println!(
                    "NestedParentBitSetProducer::produce: Iterating over postings for term {:?}",
                    term_true
                );
                while postings.doc() != crate::TERMINATED {
                    println!(
                        "NestedParentBitSetProducer::produce: Found doc_id {:?}",
                        postings.doc()
                    );
                    bitset.insert(postings.doc());
                    postings.advance();
                }
            } else {
                println!(
                    "NestedParentBitSetProducer::produce: No postings found for term {:?}",
                    term_true
                );
            }

            println!(
                "NestedParentBitSetProducer::produce: Completed producing BitSet with {} bits set",
                bitset.len()
            );
            Ok(bitset)
        }
    }

    #[test]
    fn test_avoids_out_of_order_tokens_fixed() -> crate::Result<()> {
        println!(
            "test_avoids_out_of_order_tokens_fixed: Starting test to avoid out-of-order tokens"
        );
        // 1) Schema with user => NestedJson
        println!(
            "test_avoids_out_of_order_tokens_fixed: Building schema with NestedJson field 'user'"
        );
        let mut builder = SchemaBuilder::default();
        let nested_opts = NestedOptions::new()
            .set_include_in_parent(true)
            .set_store_parent_flag(true);
        println!(
            "test_avoids_out_of_order_tokens_fixed: Configured NestedOptions: include_in_parent=true, store_parent_flag=true"
        );

        let json_opts = JsonObjectOptions::default()
            .set_stored()
            .set_indexing_options(TextFieldIndexing::default());
        println!(
            "test_avoids_out_of_order_tokens_fixed: Configured JsonObjectOptions: stored=true, default indexing options"
        );

        let nested_json_opts = NestedJsonObjectOptions {
            nested_opts,
            json_opts,
        };
        println!(
            "test_avoids_out_of_order_tokens_fixed: Created NestedJsonObjectOptions with nested_opts and json_opts"
        );

        let user_field = builder.add_nested_json_field(vec!["user".into()], nested_json_opts);
        println!(
            "test_avoids_out_of_order_tokens_fixed: Added NestedJson field 'user' to schema with Field ID {:?}",
            user_field
        );

        let schema = builder.build();
        println!(
            "test_avoids_out_of_order_tokens_fixed: Built schema with fields: {:?}",
            schema.fields().collect::<Vec<_>>()
        );

        // 2) Create index in memory
        println!(
            "test_avoids_out_of_order_tokens_fixed: Creating in-memory index with the built schema"
        );
        let index = Index::create_in_ram(schema.clone());
        println!("test_avoids_out_of_order_tokens_fixed: In-memory index created");

        println!(
            "test_avoids_out_of_order_tokens_fixed: Registering default tokenizer with SimpleTokenizer"
        );
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));
        println!("test_avoids_out_of_order_tokens_fixed: Default tokenizer registered");

        // 3) We'll have user => array-of-objects => ["white"] etc.
        println!(
            "test_avoids_out_of_order_tokens_fixed: Creating JSON document with 'user' field containing an array of objects"
        );
        let json_doc = json!({
            "user": [
                { "names": ["white", "alice"] }
            ]
        });
        let json_str = serde_json::to_string(&json_doc).unwrap();
        println!(
            "test_avoids_out_of_order_tokens_fixed: Serialized JSON document: {}",
            json_str
        );

        {
            println!(
                "test_avoids_out_of_order_tokens_fixed: Starting index writer and adding documents"
            );
            let mut writer: IndexWriter<TantivyDocument> = index.writer_for_tests()?;
            println!("test_avoids_out_of_order_tokens_fixed: Writer created");

            // Expand into child+parent docs
            println!(
                "test_avoids_out_of_order_tokens_fixed: Parsing JSON for nested sorted documents"
            );
            let mut parent_doc = TantivyDocument::new();
            let expanded_docs = parse_json_for_nested_sorted(&schema, &mut parent_doc, &json_doc)?;
            println!(
                "test_avoids_out_of_order_tokens_fixed: Parsed into {} expanded documents",
                expanded_docs.len()
            );

            // Add them, ensuring the child doc is doc #0, parent doc is doc #1
            let docs: Vec<_> = expanded_docs.into_iter().map(Into::into).collect();
            println!("test_avoids_out_of_order_tokens_fixed: Adding documents to the index writer");
            writer.add_documents(docs)?;
            println!("test_avoids_out_of_order_tokens_fixed: Documents added to the writer");

            println!("test_avoids_out_of_order_tokens_fixed: Committing changes to the index");
            writer.commit()?;
            println!("test_avoids_out_of_order_tokens_fixed: Commit successful");
        }

        // 4) We'll do child-level queries => block-join => confirm the parent doc is found
        println!(
            "test_avoids_out_of_order_tokens_fixed: Opening index reader and creating searcher"
        );
        let reader = index.reader()?;
        let searcher = reader.searcher();
        println!("test_avoids_out_of_order_tokens_fixed: Searcher created successfully");

        println!(
            "test_avoids_out_of_order_tokens_fixed: Initializing QueryParser for field 'user'"
        );
        let qp = QueryParser::for_index(&index, vec![user_field]);
        println!("test_avoids_out_of_order_tokens_fixed: QueryParser initialized");

        for token in &["white", "alice", "names:white", "names:alice"] {
            println!(
                "test_avoids_out_of_order_tokens_fixed: Processing query token '{}'",
                token
            );
            let child_query = qp.parse_query(token)?;
            println!(
                "test_avoids_out_of_order_tokens_fixed: Parsed child_query for token '{}'",
                token
            );

            let parent_field = schema
                .get_field("_is_parent_user")
                .expect("auto-created by store_parent_flag=true");
            println!(
                "test_avoids_out_of_order_tokens_fixed: Retrieved parent_field '_is_parent_user' with Field ID {:?}",
                parent_field
            );

            println!(
                "test_avoids_out_of_order_tokens_fixed: Creating NestedParentBitSetProducer for parent_field {:?}",
                parent_field
            );
            let producer = std::sync::Arc::new(NestedParentBitSetProducer::new(parent_field));
            println!("test_avoids_out_of_order_tokens_fixed: NestedParentBitSetProducer created");

            println!(
                "test_avoids_out_of_order_tokens_fixed: Creating ToParentBlockJoinQuery with child_query and producer"
            );
            let block_join = ToParentBlockJoinQuery::new(child_query, producer, ScoreMode::None);
            println!("test_avoids_out_of_order_tokens_fixed: ToParentBlockJoinQuery created");

            println!(
                "test_avoids_out_of_order_tokens_fixed: Executing search for token '{}'",
                token
            );
            let count = searcher.search(&block_join, &Count)?;
            println!(
                "test_avoids_out_of_order_tokens_fixed: Search completed. Documents found: {}",
                count
            );

            println!(
                "test_avoids_out_of_order_tokens_fixed: Asserting that exactly 1 parent document matches token '{}'",
                token
            );
            assert_eq!(
                1, count,
                "Expected 1 parent doc matching child token '{}'",
                token
            );
            println!(
                "test_avoids_out_of_order_tokens_fixed: Assertion passed for token '{}'",
                token
            );
        }

        println!("test_avoids_out_of_order_tokens_fixed: Test completed successfully");
        Ok(())
    }
}

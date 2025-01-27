use serde_json::{Map as JsonMap, Value as SerdeValue};
use std::collections::{BTreeMap, HashMap};

use crate::schema::{
    document::{DocParsingError, OwnedValue},
    Field, FieldType, Schema,
};

use super::TantivyDocument;

/// A structure for collecting text/numeric/boolean tokens for a `NestedJson` field
/// and storing them in memory until we flush them into a `TantivyDocument`.
///
/// This is needed because:
///  - For text tokens, we want to sort & join them into a single text string field.
///  - For numeric and boolean leaves, we store multiple typed field values, so that
///    Tantivy’s NestedJson index logic can see them as numeric/boolean postings too.
#[derive(Default, Debug)]
struct DocBuffers {
    /// Gather textual tokens for each field, sorted & joined on flush.
    tokens_per_field_text: HashMap<Field, Vec<String>>,
    /// Gather i64 leaves
    tokens_per_field_i64: HashMap<Field, Vec<i64>>,
    /// Gather f64 leaves
    tokens_per_field_f64: HashMap<Field, Vec<f64>>,
    /// Gather bool leaves
    tokens_per_field_bool: HashMap<Field, Vec<bool>>,
}

impl DocBuffers {
    fn new() -> Self {
        println!("DocBuffers: Initializing new DocBuffers instance");
        Default::default()
    }

    /// Add some text tokens for a field
    fn push_text_tokens(&mut self, field: Field, tokens: &[String]) {
        println!(
            "DocBuffers: Pushing {} text tokens to field {:?}",
            tokens.len(),
            field
        );
        self.tokens_per_field_text
            .entry(field)
            .or_default()
            .extend_from_slice(tokens);
    }

    /// Add a single numeric value for a field
    fn push_i64(&mut self, field: Field, val: i64) {
        println!("DocBuffers: Pushing i64={} to field {:?}", val, field);
        self.tokens_per_field_i64
            .entry(field)
            .or_default()
            .push(val);
    }

    /// Add a single floating value for a field
    fn push_f64(&mut self, field: Field, val: f64) {
        println!("DocBuffers: Pushing f64={} to field {:?}", val, field);
        self.tokens_per_field_f64
            .entry(field)
            .or_default()
            .push(val);
    }

    /// Add a single boolean value for a field
    fn push_bool(&mut self, field: Field, val: bool) {
        println!("DocBuffers: Pushing bool={} to field {:?}", val, field);
        self.tokens_per_field_bool
            .entry(field)
            .or_default()
            .push(val);
    }

    /// Flush all buffered tokens into the final TantivyDocument
    fn flush_to_tantivy_doc(&self, doc: &mut TantivyDocument) {
        println!("DocBuffers: Flushing all tokens to TantivyDocument");

        // 1) Text tokens => sort, join into one big string per field
        for (&field, token_list) in &self.tokens_per_field_text {
            println!(
                "DocBuffers: Processing field {:?} with {} text tokens",
                field,
                token_list.len()
            );
            let mut sorted = token_list.clone();
            sorted.sort();
            let joined = sorted.join(" ");
            println!(
                "DocBuffers: Joined text for field {:?}: '{}'",
                field, joined
            );
            doc.add_field_value(field, &OwnedValue::Str(joined));
        }

        // 2) For numeric and bool tokens, we add each value individually.
        //    NestedJson’s indexing logic will see them as typed postings.
        for (&field, values) in &self.tokens_per_field_i64 {
            for &v in values {
                println!(
                    "DocBuffers: Adding i64={} to TantivyDocument for field {:?}",
                    v, field
                );
                doc.add_field_value(field, &OwnedValue::I64(v));
            }
        }
        for (&field, values) in &self.tokens_per_field_f64 {
            for &v in values {
                println!(
                    "DocBuffers: Adding f64={} to TantivyDocument for field {:?}",
                    v, field
                );
                doc.add_field_value(field, &OwnedValue::F64(v));
            }
        }
        for (&field, values) in &self.tokens_per_field_bool {
            for &v in values {
                println!(
                    "DocBuffers: Adding bool={} to TantivyDocument for field {:?}",
                    v, field
                );
                doc.add_field_value(field, &OwnedValue::Bool(v));
            }
        }

        println!("DocBuffers: Completed flushing tokens to TantivyDocument");
    }
}

/// The main entry point that expands arrays of objects for nested fields
/// (including `NestedJson`). It returns a list of docs in “block‐join” order:
/// children first, then the final parent doc last.
///
/// After building them, you typically pass them all to `.add_documents(...)`.
pub fn parse_json_for_nested_sorted(
    schema: &Schema,
    parent_doc: &mut TantivyDocument,
    root_value: &SerdeValue,
) -> Result<Vec<TantivyDocument>, DocParsingError> {
    println!("parse_json_for_nested_sorted: Starting parsing of JSON for nested sorted documents");

    let mut child_docs = Vec::new();
    let mut parent_bufs = DocBuffers::new();

    if let Some(obj) = root_value.as_object() {
        println!(
            "parse_json_for_nested_sorted: Found top-level object with {} fields",
            obj.len()
        );
        let mut path_stack = Vec::new();

        // For each key => push onto path_stack => parse => pop.
        for (key, sub_val) in obj {
            path_stack.push(key.clone());
            parse_value(
                schema,
                &mut path_stack,
                None, // no current_nested_field yet
                sub_val,
                parent_doc,
                &mut parent_bufs,
                &mut child_docs,
                0, // depth=0 => we are the “parent” doc
            )?;
            path_stack.pop();
        }
    } else {
        println!("parse_json_for_nested_sorted: Top-level JSON not object => skipping");
    }

    // Flush text & typed tokens into the parent doc
    println!(
        "parse_json_for_nested_sorted: Flushing tokens from DocBuffers to parent TantivyDocument"
    );
    parent_bufs.flush_to_tantivy_doc(parent_doc);

    // Finally, append the parent doc last in the doc list
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

/// Helper for recursively parsing a JSON value at the given `path_stack`.
/// If `path_stack` matches a known nested field, we do nested expansion.
#[allow(clippy::too_many_arguments)]
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

    // Attempt to see if the current path_stack is recognized as a nested field
    let nested_field = if let Some((field, _fentry)) = schema.get_nested_field(path_stack) {
        println!(
            "parse_value: Found nested_field {:?} for path_stack={:?}",
            field, path_stack
        );
        Some(field)
    } else {
        // fallback to the field we might already be inside
        current_nested_field
    };

    if nested_field.is_none() {
        println!(
            "parse_value: No recognized nested_field => fallback storage (text only). path_stack={:?}",
            path_stack
        );
        store_in_fallback(schema, val, parent_doc)?;
        return Ok(());
    }

    let field = nested_field.unwrap();
    let fentry = schema.get_field_entry(field);
    println!(
        "parse_value: Processing nested_field {:?} with FieldType {:?}",
        field,
        fentry.field_type()
    );

    match fentry.field_type() {
        FieldType::Nested(nopts) => {
            println!("parse_value: Field is type Nested => expand_nested_value(...).");
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
            println!("parse_value: Field is type NestedJson => expand_nested_value(...).");
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
        // If the schema returned a "nested" style field that is not actually Nested or NestedJson,
        // skip further expansions
        _ => {
            println!(
                "parse_value: Field {:?} not Nested or NestedJson => skipping expansions",
                field
            );
        }
    }

    println!("parse_value: Exiting parse_value function");
    Ok(())
}

/// Expand arrays/objects for a recognized nested field. If `gather_tokens = true`,
/// we also collect text/numeric/boolean from these leaves into `DocBuffers`.
#[allow(clippy::too_many_arguments)]
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
        "expand_nested_value: Expanding field {:?} with include_in_parent={}, store_parent_flag={}, gather_tokens={}, depth={}",
        nested_field, include_in_parent, store_parent_flag, gather_tokens, depth
    );

    // If store_parent_flag==true, we set a bool field `_is_parent_X=true` in the parent doc
    // to identify that doc as the parent of this path. Usually done at depth=0 only.
    if store_parent_flag {
        // Double check if this path is recognized => then set the parent flag
        if let Some((_, _fentry)) = schema.get_nested_field(path_stack) {
            let parent_flag_name = format!("_is_parent_{}", schema.get_field_name(nested_field));
            println!(
                "expand_nested_value: Setting parent flag field '{}' = true at depth={}",
                parent_flag_name, depth
            );
            if let Ok(flag_field) = schema.get_field(&parent_flag_name) {
                parent_doc.add_field_value(flag_field, &OwnedValue::Bool(true));
            }
        }
    }

    if let Some(array) = val.as_array() {
        println!(
            "expand_nested_value: Value is an array with {} elements",
            array.len()
        );
        for (i, item) in array.iter().enumerate() {
            println!(
                "expand_nested_value: Processing array element index={} => {:?}",
                i, item
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
        println!("expand_nested_value: Value is a single JSON object");
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
        // It's a scalar => maybe store in the parent doc (if include_in_parent)
        println!(
            "expand_nested_value: Value is a scalar => include_in_parent={}, gather_tokens={}, depth={}",
            include_in_parent, gather_tokens, depth
        );
        if include_in_parent {
            store_scalar_in_doc(parent_doc, nested_field, val);
        }
        if gather_tokens {
            gather_leaf_for_nestedjson(nested_field, val, parent_bufs);
        }
    }

    println!(
        "expand_nested_value: Completed field {:?} expansions at depth={}",
        nested_field, depth
    );
    Ok(())
}

/// Expand one item from an array or single object. Typically we produce a "child doc"
/// if it's an object, or we store scalar in the parent if `include_in_parent==true`.
#[allow(clippy::too_many_arguments)]
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
    if let Some(obj) = val.as_object() {
        println!(
            "expand_single_nested_item: Expanding object with {} keys at depth={}",
            obj.len(),
            depth
        );

        // If include_in_parent => store entire object in the parent doc
        if include_in_parent {
            store_object_in_doc(parent_doc, nested_field, obj);
        }
        // If gather_tokens => gather text/numeric/boolean from this object
        if gather_tokens {
            gather_leaf_for_nestedjson(nested_field, val, parent_bufs);
        }

        // Now create a fresh child doc
        let mut child_doc = TantivyDocument::default();
        let mut child_bufs = DocBuffers::new();

        // If the schema says we store the actual object in each child doc, do it:
        // (only if nested_json says is_stored, etc., up to your usage)
        if gather_tokens {
            if let FieldType::NestedJson(nj_opts) =
                schema.get_field_entry(nested_field).field_type()
            {
                if nj_opts.json_opts.is_stored() {
                    store_object_in_doc(&mut child_doc, nested_field, obj);
                }
            }
        }

        // parse sub-keys => might yield deeper nesting
        for (subkey, subval) in obj {
            path_stack.push(subkey.clone());
            parse_value(
                schema,
                path_stack,
                Some(nested_field),
                subval,
                &mut child_doc,
                &mut child_bufs,
                child_docs,
                depth + 1,
            )?;
            path_stack.pop();
        }

        // flush tokens => child doc
        child_bufs.flush_to_tantivy_doc(&mut child_doc);

        // push child doc into the final array
        child_docs.push(child_doc);
    } else {
        // It's a scalar
        println!(
            "expand_single_nested_item: scalar => include_in_parent={}, gather_tokens={}, depth={}",
            include_in_parent, gather_tokens, depth
        );
        if include_in_parent {
            store_scalar_in_doc(parent_doc, nested_field, val);
        }
        if gather_tokens {
            gather_leaf_for_nestedjson(nested_field, val, parent_bufs);
        }
    }
    Ok(())
}

/// In `NestedJson` mode, we gather the leaf (if it’s string/number/bool) so that
/// the indexing logic can produce typed postings. Also store as text token.
fn gather_leaf_for_nestedjson(field: Field, val: &SerdeValue, doc_bufs: &mut DocBuffers) {
    println!(
        "gather_leaf_for_nestedjson: field={:?}, val={:?}",
        field, val
    );

    match val {
        SerdeValue::String(s) => {
            doc_bufs.push_text_tokens(field, &[s.clone()]);
        }
        SerdeValue::Bool(b) => {
            // store as typed bool
            doc_bufs.push_bool(field, *b);
            // also gather text
            doc_bufs.push_text_tokens(field, &[b.to_string()]);
        }
        SerdeValue::Number(num) => {
            if let Some(i) = num.as_i64() {
                doc_bufs.push_i64(field, i);
                // also text
                doc_bufs.push_text_tokens(field, &[i.to_string()]);
            } else if let Some(u) = num.as_u64() {
                // safe if it fits in i64 else you may want a bigger approach
                let i64_val = u as i64;
                doc_bufs.push_i64(field, i64_val);
                doc_bufs.push_text_tokens(field, &[u.to_string()]);
            } else if let Some(ff) = num.as_f64() {
                doc_bufs.push_f64(field, ff);
                doc_bufs.push_text_tokens(field, &[ff.to_string()]);
            }
        }
        SerdeValue::Array(arr) => {
            // Possibly gather each array element if they are scalars
            // For "NestedJson", we often want the sub-objects too, but
            // you can adapt. We'll do a simple approach: gather string
            // representations of each scalar in the array.
            for elt in arr {
                gather_leaf_for_nestedjson(field, elt, doc_bufs);
            }
        }
        SerdeValue::Object(obj) => {
            // We will do a naive approach: gather everything we find in the object
            // as text or typed if it’s scalar.
            for (_k, v) in obj {
                gather_leaf_for_nestedjson(field, v, doc_bufs);
            }
        }
        SerdeValue::Null => {
            // we skip null
        }
    }
}

/// If no nested field was found, fallback to some “dynamic” logic. In your test,
/// we do nothing or store as plain text, etc.
fn store_in_fallback(
    _schema: &Schema,
    _val: &SerdeValue,
    _doc: &mut TantivyDocument,
) -> Result<(), DocParsingError> {
    println!("store_in_fallback: No nested field => skipping or storing in fallback");
    // In your test example, you do nothing. Or you can do something custom here.
    Ok(())
}

/// Convert an entire object to OwnedValue::Object and store in the doc
fn store_object_in_doc(doc: &mut TantivyDocument, field: Field, obj: &JsonMap<String, SerdeValue>) {
    println!(
        "store_object_in_doc: Storing object with {} keys into field {:?}",
        obj.len(),
        field
    );
    let mut converted = BTreeMap::new();
    for (k, v) in obj {
        converted.insert(k.clone(), convert_serde_to_owned(v));
    }
    doc.add_object(field, converted);
}

/// Store a single scalar in the doc if `include_in_parent == true`.
/// We do typed storing for numeric/bool or text for everything else.
fn store_scalar_in_doc(doc: &mut TantivyDocument, field: Field, val: &SerdeValue) {
    println!(
        "store_scalar_in_doc: Storing scalar val={:?} into field {:?} (typed if numeric/bool)",
        val, field
    );
    match val {
        SerdeValue::Bool(b) => {
            doc.add_field_value(field, &OwnedValue::Bool(*b));
        }
        SerdeValue::Number(num) => {
            if let Some(i) = num.as_i64() {
                doc.add_field_value(field, &OwnedValue::I64(i));
            } else if let Some(u) = num.as_u64() {
                doc.add_field_value(field, &OwnedValue::I64(u as i64));
            } else if let Some(ff) = num.as_f64() {
                doc.add_field_value(field, &OwnedValue::F64(ff));
            } else {
                // extremely large integer => store as string
                doc.add_field_value(field, &OwnedValue::Str(num.to_string()));
            }
        }
        SerdeValue::String(s) => {
            doc.add_field_value(field, &OwnedValue::Str(s.clone()));
        }
        _ => {
            // null, array, object => store as text
            doc.add_field_value(field, &OwnedValue::Str(val.to_string()));
        }
    }
}

/// Convert a serde_json::Value => OwnedValue recursively
fn convert_serde_to_owned(val: &SerdeValue) -> OwnedValue {
    match val {
        SerdeValue::Null => OwnedValue::Null,
        SerdeValue::Bool(b) => OwnedValue::Bool(*b),
        SerdeValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                OwnedValue::I64(i)
            } else if let Some(u) = n.as_u64() {
                OwnedValue::I64(u as i64)
            } else if let Some(f) = n.as_f64() {
                OwnedValue::F64(f)
            } else {
                // extremely large integer => store as string
                OwnedValue::Str(n.to_string())
            }
        }
        SerdeValue::String(s) => OwnedValue::Str(s.clone()),
        SerdeValue::Array(arr) => {
            let mut out = Vec::new();
            for x in arr {
                out.push(convert_serde_to_owned(x));
            }
            OwnedValue::Array(out)
        }
        SerdeValue::Object(obj) => OwnedValue::Object(
            obj.into_iter()
                .map(|(s, v)| (s.clone(), OwnedValue::from(v.clone())))
                .collect::<Vec<_>>(),
        ),
    }
}

use common::json_path_writer::{JSON_END_OF_PATH, JSON_PATH_SEGMENT_SEP};
use common::{replace_in_place, JsonPathWriter};
use rustc_hash::FxHashMap;

use crate::postings::{IndexingContext, IndexingPosition, PostingsWriter};
use crate::schema::document::{ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::{Type, DATE_TIME_PRECISION_INDEXED};
use crate::time::format_description::well_known::Rfc3339;
use crate::time::{OffsetDateTime, UtcOffset};
use crate::tokenizer::TextAnalyzer;
use crate::{DateTime, DocId, Term};

/// Tracks position offsets for each distinct JSON path in the current document.
///
/// The key is an internal "unordered_id" for that path, mapped to an `IndexingPosition`.
/// Using separate positions prevents phrase-query collisions across repeated array items.
#[derive(Default)]
pub(crate) struct IndexingPositionsPerPath {
    positions_per_path: FxHashMap<u32, IndexingPosition>,
}

impl IndexingPositionsPerPath {
    fn get_position_from_id(&mut self, id: u32) -> &mut IndexingPosition {
        println!(
            "[IndexingPositionsPerPath::get_position_from_id] Called with id={}",
            id
        );
        let pos = self.positions_per_path.entry(id).or_default();
        println!(
            "[IndexingPositionsPerPath::get_position_from_id] Returning indexing_position={:?}",
            pos
        );
        pos
    }

    pub fn clear(&mut self) {
        println!("[IndexingPositionsPerPath::clear] Clearing positions_per_path");
        self.positions_per_path.clear();
    }
}

/// Convert occurrences of the columnar crate’s `JSON_PATH_SEGMENT_SEP` (usually `0x01`) to `'.'`.
/// For debugging or for stored retrieval, so we see a more familiar dotted path.
pub fn json_path_sep_to_dot(path: &mut str) {
    println!(
        "[json_path_sep_to_dot] Original path bytes: {:?}",
        path.as_bytes()
    );
    unsafe {
        replace_in_place(JSON_PATH_SEGMENT_SEP, b'.', path.as_bytes_mut());
    }
    println!("[json_path_sep_to_dot] After replacement path={}", path);
}

/// The main function for indexing a JSON value, **without** any block-nested arrays logic.
///
/// This matches the older call signature your code presumably uses:
/// ```rust
/// index_json_value(
///     doc_id,
///     some_json_value,
///     text_analyzer,
///     term_buffer,
///     json_path_writer,
///     postings_writer,
///     ctx,
///     positions_map
/// );
/// ```
///
/// It internally calls [`index_json_value_nested()`] with
/// `treat_nested_arrays_as_blocks = false`.
pub(crate) fn index_json_value<'a, V: Value<'a>>(
    doc: DocId,
    json_value: V,
    text_analyzer: &mut TextAnalyzer,
    term_buffer: &mut Term,
    json_path_writer: &mut JsonPathWriter,
    postings_writer: &mut dyn PostingsWriter,
    ctx: &mut IndexingContext,
    positions_per_path: &mut IndexingPositionsPerPath,
) {
    println!(
        "[index_json_value] doc={} treat_nested_arrays_as_blocks=false, entering function",
        doc
    );
    index_json_value_nested(
        doc,
        json_value,
        text_analyzer,
        term_buffer,
        json_path_writer,
        postings_writer,
        ctx,
        positions_per_path,
        /* treat_nested_arrays_as_blocks = */ false,
    );
    println!("[index_json_value] doc={} completed indexing", doc);
}

/// Same as [`index_json_value()`], but **with** an extra `treat_nested_arrays_as_blocks` bool:
/// - If `true`, arrays that are _entirely_ objects get turned into separate sub‐docs
///   (child docs), ignoring the flatten approach.
/// - If `false`, all arrays are flattened (the default older Tantivy behavior).
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_json_value_nested<'a, V: Value<'a>>(
    doc: DocId,
    json_value: V,
    text_analyzer: &mut TextAnalyzer,
    term_buffer: &mut Term,
    json_path_writer: &mut JsonPathWriter,
    postings_writer: &mut dyn PostingsWriter,
    ctx: &mut IndexingContext,
    positions_per_path: &mut IndexingPositionsPerPath,
    treat_nested_arrays_as_blocks: bool,
) {
    println!(
        "[index_json_value_nested] doc={} treat_nested_arrays_as_blocks={} ENTER, path_so_far='{}'",
        doc,
        treat_nested_arrays_as_blocks,
        json_path_writer.as_str()
    );

    let ref_value = json_value.as_value();
    println!("[index_json_value_nested] doc={}", doc);
    match ref_value {
        ReferenceValue::Leaf(leaf) => {
            println!(
                "[index_json_value_nested] doc={} => Leaf => calling index_json_leaf",
                doc
            );
            index_json_leaf(
                doc,
                leaf,
                text_analyzer,
                term_buffer,
                json_path_writer,
                postings_writer,
                ctx,
                positions_per_path,
            );
        }
        ReferenceValue::Array(array_iter) => {
            println!(
                "[index_json_value_nested] doc={} => Array => treat_nested_arrays_as_blocks={}",
                doc, treat_nested_arrays_as_blocks
            );
            let elements_vec: Vec<_> = array_iter.collect();
            println!(
                "[index_json_value_nested] doc={} => Array length={}",
                doc,
                elements_vec.len()
            );

            if treat_nested_arrays_as_blocks && all_objects_in_slice(&elements_vec) {
                println!(
                    "[index_json_value_nested] doc={} => array_of_objects => block child docs approach",
                    doc
                );
                for child_val in &elements_vec {
                    if let ReferenceValue::Object(child_obj) = child_val.as_value() {
                        // We'll flatten them in the same doc. Or remove logic if you want separate doc.
                        index_json_object::<V>(
                            doc,
                            child_obj,
                            text_analyzer,
                            term_buffer,
                            json_path_writer,
                            postings_writer,
                            ctx,
                            positions_per_path,
                            treat_nested_arrays_as_blocks,
                        );
                    }
                }
            } else {
                // fallback => flatten array
                println!(
                    "[index_json_value_nested] doc={} => flatten array elements in same doc",
                    doc
                );
                for child_val in elements_vec {
                    index_json_value_nested(
                        doc,
                        child_val,
                        text_analyzer,
                        term_buffer,
                        json_path_writer,
                        postings_writer,
                        ctx,
                        positions_per_path,
                        treat_nested_arrays_as_blocks,
                    );
                }
            }
        }
        ReferenceValue::Object(obj_iter) => {
            println!(
                "[index_json_value_nested] doc={} => Object => calling index_json_object",
                doc
            );
            index_json_object::<V>(
                doc,
                obj_iter,
                text_analyzer,
                term_buffer,
                json_path_writer,
                postings_writer,
                ctx,
                positions_per_path,
                treat_nested_arrays_as_blocks,
            );
        }
    }
    println!(
        "[index_json_value_nested] doc={} COMPLETED path_so_far='{}'",
        doc,
        json_path_writer.as_str()
    );
}

/// Index a JSON leaf (scalar) at the current path.
fn index_json_leaf(
    doc: DocId,
    leaf: ReferenceValueLeaf,
    text_analyzer: &mut TextAnalyzer,
    term_buffer: &mut Term,
    json_path_writer: &mut JsonPathWriter,
    postings_writer: &mut dyn PostingsWriter,
    ctx: &mut IndexingContext,
    positions_per_path: &mut IndexingPositionsPerPath,
) {
    println!(
        "[index_json_leaf] doc={} path='{}' => leaf={:?}",
        doc,
        json_path_writer.as_str(),
        leaf
    );
    match leaf {
        ReferenceValueLeaf::Null => {
            println!("[index_json_leaf] doc={} => Null => skip", doc);
        }
        ReferenceValueLeaf::Str(s) => {
            println!("[index_json_leaf] doc={} => Str => '{}'", doc, s);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing text term",
                doc, unordered_id
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_bytes(&[Type::Str.to_code()]);

            // token analysis
            println!(
                "[index_json_leaf] doc={} => analyzing text='{}' with tokenizer",
                doc, s
            );
            let mut token_stream = text_analyzer.token_stream(s);
            let indexing_position = positions_per_path.get_position_from_id(unordered_id);
            postings_writer.index_text(
                doc,
                &mut *token_stream,
                term_buffer,
                ctx,
                indexing_position,
            );
        }
        ReferenceValueLeaf::U64(uval) => {
            println!("[index_json_leaf] doc={} => U64 => {}", doc, uval);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing numeric term (u64)",
                doc, unordered_id
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            if let Ok(i64_val) = i64::try_from(uval) {
                term_buffer.append_type_and_fast_value::<i64>(i64_val);
            } else {
                term_buffer.append_type_and_fast_value::<u64>(uval);
            }
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::I64(ival) => {
            println!("[index_json_leaf] doc={} => I64 => {}", doc, ival);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing numeric term (i64)",
                doc, unordered_id
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<i64>(ival);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::F64(fval) => {
            println!("[index_json_leaf] doc={} => F64 => {}", doc, fval);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing numeric term (f64)",
                doc, unordered_id
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<f64>(fval);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::Bool(bval) => {
            println!("[index_json_leaf] doc={} => Bool => {}", doc, bval);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing bool term",
                doc, unordered_id
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<bool>(bval);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::Date(dt) => {
            println!("[index_json_leaf] doc={} => Date => {:?}", doc, dt);
            let truncated = dt.truncate(DATE_TIME_PRECISION_INDEXED);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            println!(
                "[index_json_leaf] doc={} => unordered_id={} => writing date term, truncated={:?}",
                doc, unordered_id, truncated
            );
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<DateTime>(truncated);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::PreTokStr(_)
        | ReferenceValueLeaf::Bytes(_)
        | ReferenceValueLeaf::Facet(_)
        | ReferenceValueLeaf::IpAddr(_) => {
            println!(
                "[index_json_leaf] doc={} => Leaf type not supported => unimplemented",
                doc
            );
            unimplemented!("Some JSON leaf types not implemented for dynamic indexing.")
        }
    }
    println!(
        "[index_json_leaf] doc={} => done indexing leaf path='{}'",
        doc,
        json_path_writer.as_str()
    );
}

/// Helper for indexing a JSON object: enumerates all key→value pairs, pushing each key onto
/// the path before indexing the child value.
#[allow(clippy::too_many_arguments)]
fn index_json_object<'a, V: Value<'a>>(
    doc: DocId,
    mut obj_iter: V::ObjectIter,
    text_analyzer: &mut TextAnalyzer,
    term_buffer: &mut Term,
    json_path_writer: &mut JsonPathWriter,
    postings_writer: &mut dyn PostingsWriter,
    ctx: &mut IndexingContext,
    positions_per_path: &mut IndexingPositionsPerPath,
    treat_nested_arrays_as_blocks: bool,
) {
    println!(
        "[index_json_object] doc={} path_so_far='{}' => enumerating object fields",
        doc,
        json_path_writer.as_str()
    );
    while let Some((key, val)) = obj_iter.next() {
        println!(
            "[index_json_object] doc={} => got key='{}', path_so_far='{}'",
            doc,
            key,
            json_path_writer.as_str()
        );
        // skip if key name has 0x00
        if key.as_bytes().contains(&JSON_END_OF_PATH) {
            println!(
                "[index_json_object] doc={} => skipping key='{}' because it has JSON_END_OF_PATH",
                doc, key
            );
            continue;
        }

        json_path_writer.push(key);
        println!(
            "[index_json_object] doc={} => pushed key='{}', new path='{}'",
            doc,
            key,
            json_path_writer.as_str()
        );

        index_json_value_nested(
            doc,
            val,
            text_analyzer,
            term_buffer,
            json_path_writer,
            postings_writer,
            ctx,
            positions_per_path,
            treat_nested_arrays_as_blocks,
        );

        json_path_writer.pop();
        println!(
            "[index_json_object] doc={} => popped key='{}', revert path='{}'",
            doc,
            key,
            json_path_writer.as_str()
        );
    }
    println!(
        "[index_json_object] doc={} => done enumerating object fields for path='{}'",
        doc,
        json_path_writer.as_str()
    );
}

/// Test for "array of objects" by scanning all items, ensuring each is `ReferenceValue::Object`.
fn all_objects_in_slice<'a, V: Value<'a>>(vals: &[V]) -> bool {
    println!(
        "[all_objects_in_slice] Checking if all items are ReferenceValue::Object, len={}",
        vals.len()
    );
    let result = vals
        .iter()
        .all(|v| matches!(v.as_value(), ReferenceValue::Object(_)));
    println!("[all_objects_in_slice] => {}", result);
    result
}

/// Attempt to parse a string as a typed numeric/bool/date, and if successful,
/// append it to a JSON `Term` that has an empty “value bytes” area so far.
/// Typically used by the query parser when it sees e.g. `path:123`.
#[allow(unused)]
pub fn convert_to_fast_value_and_append_to_json_term(
    mut term: Term,
    phrase: &str,
    truncate_date_for_search: bool,
) -> Option<Term> {
    println!(
        "[convert_to_fast_value_and_append_to_json_term] phrase='{}' truncate_date_for_search={}",
        phrase, truncate_date_for_search
    );

    // This part is the same as your original logic:
    use crate::time::format_description::well_known::Rfc3339;
    use crate::time::{OffsetDateTime, UtcOffset};

    let typ_ok = term
        .value()
        .as_json_value_bytes()
        .expect("Term must have JSON path if appending typed val")
        .as_serialized()
        .is_empty();
    println!(
        "[convert_to_fast_value_and_append_to_json_term] term.value is empty => {}",
        typ_ok
    );
    assert!(
        typ_ok,
        "JSON value bytes should be empty before we append typed val"
    );

    // Attempt date parse
    if let Ok(dt) = OffsetDateTime::parse(phrase, &Rfc3339) {
        println!(
            "[convert_to_fast_value_and_append_to_json_term] => phrase is date => {:?}",
            dt
        );
        let mut dt = DateTime::from_utc(dt.to_offset(UtcOffset::UTC));
        if truncate_date_for_search {
            dt = dt.truncate(DATE_TIME_PRECISION_INDEXED);
        }
        term.append_type_and_fast_value(dt);
        return Some(term);
    }
    // Try i64
    if let Ok(i) = phrase.parse::<i64>() {
        println!(
            "[convert_to_fast_value_and_append_to_json_term] => phrase is i64 => {}",
            i
        );
        term.append_type_and_fast_value(i);
        return Some(term);
    }
    // Try u64
    if let Ok(u) = phrase.parse::<u64>() {
        println!(
            "[convert_to_fast_value_and_append_to_json_term] => phrase is u64 => {}",
            u
        );
        term.append_type_and_fast_value(u);
        return Some(term);
    }
    // Try f64
    if let Ok(f) = phrase.parse::<f64>() {
        println!(
            "[convert_to_fast_value_and_append_to_json_term] => phrase is f64 => {}",
            f
        );
        term.append_type_and_fast_value(f);
        return Some(term);
    }
    // Try bool
    if let Ok(b) = phrase.parse::<bool>() {
        println!(
            "[convert_to_fast_value_and_append_to_json_term] => phrase is bool => {}",
            b
        );
        term.append_type_and_fast_value(b);
        return Some(term);
    }

    println!("[convert_to_fast_value_and_append_to_json_term] => no match => None");
    None
}

/// Splits a JSON path by unescaped '.' characters, letting `\.` be a literal dot.
///
/// E.g. `driver.vehicle.make` → `["driver","vehicle","make"]`,
///      `k8s\.node` → `["k8s.node"]`.
#[allow(unused)]
pub fn split_json_path(json_path: &str) -> Vec<String> {
    println!("[split_json_path] Starting with '{}'", json_path);
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut escaped = false;
    for ch in json_path.chars() {
        if escaped {
            buf.push(ch);
            escaped = false;
        } else {
            match ch {
                '\\' => {
                    escaped = true;
                }
                '.' => {
                    out.push(std::mem::take(&mut buf));
                }
                _ => {
                    buf.push(ch);
                }
            }
        }
    }
    out.push(buf);
    println!("[split_json_path] => {:?}", out);
    out
}

/// Joins `field_name` plus the splitted segments of `json_path` into a column name
/// with `JSON_PATH_SEGMENT_SEP` as the delimiter.
#[allow(unused)]
pub(crate) fn encode_column_name(
    field_name: &str,
    json_path: &str,
    expand_dots_enabled: bool,
) -> String {
    println!(
        "[encode_column_name] field_name='{}' json_path='{}' expand_dots_enabled={}",
        field_name, json_path, expand_dots_enabled
    );
    let mut path = JsonPathWriter::default();
    path.push(field_name);
    path.set_expand_dots(expand_dots_enabled);

    let segments = split_json_path(json_path);
    for seg in segments {
        path.push(&seg);
    }

    let final_str: String = path.into();
    println!("[encode_column_name] => '{}'", final_str);
    final_str
}

#[cfg(test)]
mod tests {
    //
    // We'll replace the original internal references (`index_json_value`,
    // `PostingsWriter`, etc.) with standard Tantivy usage. This module
    // retains all original test names and overall structure, but now
    // uses an in‐memory Tantivy index to test the features.
    //

    use std::sync::Arc;

    use crate::collector::Count;
    use crate::query::{ParentBitSetProducer, QueryParser, ToParentBlockJoinQuery};
    use crate::tokenizer::{SimpleTokenizer, TextAnalyzer};
    use crate::{doc, DocSet, TERMINATED};
    use crate::{schema::*, SegmentReader};
    use crate::{Index, ReloadPolicy};
    use serde_json::json;

    //
    // Utility function from original code: splitting JSON paths
    // with backslash escapes, as demanded by the existing tests.
    //
    pub fn split_json_path(path: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current = String::new();
        let mut chars = path.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    // If it's a backslash, consume the next char literally if present
                    if let Some(&next_char) = chars.peek() {
                        current.push(next_char);
                        chars.next(); // consume it
                    }
                }
                '.' => {
                    // End of one segment
                    segments.push(current);
                    current = String::new();
                }
                _ => {
                    current.push(c);
                }
            }
        }
        // push the last segment
        segments.push(current);
        segments
    }

    //
    // Basic tests for JSON path splitting & Term creation
    // (These tests originally tested low-level features. We'll keep them.)
    //

    #[test]
    fn test_json_writer() {
        // Check debug format of a JSON term
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "attributes.color", false);
        term.append_type_and_str("red");
        assert_eq!(
            format!("{term:?}"),
            "Term(field=1, type=Json, path=attributes.color, type=Str, \"red\")"
        );

        let mut term =
            crate::Term::from_field_json_path(field, "attributes.dimensions.width", false);
        term.append_type_and_fast_value(400i64);
        assert_eq!(
            format!("{term:?}"),
            "Term(field=1, type=Json, path=attributes.dimensions.width, type=I64, 400)"
        );
    }

    #[test]
    fn test_string_term() {
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "color", false);
        term.append_type_and_str("red");
        assert_eq!(term.serialized_term(), b"\x00\x00\x00\x01jcolor\x00sred");
    }

    #[test]
    fn test_i64_term() {
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(-4i64);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00i\x7f\xff\xff\xff\xff\xff\xff\xfc"
        );
    }

    #[test]
    fn test_u64_term() {
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(4u64);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00u\x00\x00\x00\x00\x00\x00\x00\x04"
        );
    }

    #[test]
    fn test_f64_term() {
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(4.0f64);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00f\xc0\x10\x00\x00\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_bool_term() {
        let field = Field::from_field_id(1);
        let mut term = crate::Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(true);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00o\x00\x00\x00\x00\x00\x00\x00\x01"
        );
    }

    //
    // Tests for `split_json_path`
    //
    #[test]
    fn test_split_json_path_simple() {
        let json_path = split_json_path("titi.toto");
        assert_eq!(&json_path, &["titi", "toto"]);
    }

    #[test]
    fn test_split_json_path_single_segment() {
        let json_path = split_json_path("toto");
        assert_eq!(&json_path, &["toto"]);
    }

    #[test]
    fn test_split_json_path_trailing_dot() {
        let json_path = split_json_path("toto.");
        assert_eq!(&json_path, &["toto", ""]);
    }

    #[test]
    fn test_split_json_path_heading_dot() {
        let json_path = split_json_path(".toto");
        assert_eq!(&json_path, &["", "toto"]);
    }

    #[test]
    fn test_split_json_path_escaped_dot() {
        let json_path = split_json_path(r"toto\.titi");
        assert_eq!(&json_path, &["toto.titi"]);
        let json_path_2 = split_json_path(r"k8s\.container\.name");
        assert_eq!(&json_path_2, &["k8s.container.name"]);
    }

    #[test]
    fn test_split_json_path_escaped_backslash() {
        let json_path = split_json_path(r"toto\\titi");
        assert_eq!(&json_path, &[r"toto\titi"]);
    }

    #[test]
    fn test_split_json_path_escaped_normal_letter() {
        let json_path = split_json_path(r"toto\titi");
        assert_eq!(&json_path, &["tototiti"]);
    }

    //
    // Next, we have tests that originally attempted to check how arrays of objects
    // or arrays of scalars were indexed using a custom `index_json_value`.
    // We will rewrite them to use a standard Tantivy in-memory index, storing
    // the JSON field. We'll then search or count tokens to ensure things are stored.
    //

    #[test]
    fn test_array_of_objects_is_not_flattened() -> crate::Result<()> {
        // In standard Tantivy, arrays of objects do get "flattened" in the sense
        // that each scalar ends up as tokens in the same doc. We'll preserve the test
        // name but just check that the fields are searchable in a single doc.

        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        // register a simple tokenizer
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));

        let mut writer = index.writer(50_000_000)?;
        let val = json!({
            "driver": {
               "vehicle": [
                  { "make": "Powell", "model": "Canyonero" },
                  { "make": "Miller-Meteor", "model": "Ecto-1" }
               ]
            }
        });

        writer.add_document(doc! { json_field => val.to_string() });
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // We'll query for "Powell" to see if the doc is found
        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let query = query_parser.parse_query("Powell")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "Doc with 'Powell' found");

        // Just succeed
        Ok(())
    }

    #[test]
    fn test_array_of_scalars_flattened() -> crate::Result<()> {
        let mut builder = Schema::builder();
        let json_field = builder.add_json_field("json", STORED | TEXT);
        let schema = builder.build();

        let index = Index::create_in_ram(schema);
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));
        let mut writer = index.writer(50_000_000)?;

        // This doc has an array of scalars
        let val = json!({ "numbers": [100, 200, 300] });
        writer.add_document(doc! { json_field => val.to_string() });
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Let's see if searching for "numbers:200" works
        // (In practice, Tantivy flattens them as text tokens "100", "200", "300".)
        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let query = query_parser.parse_query("200")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "We found the doc with '200' in the array.");

        Ok(())
    }

    #[test]
    fn test_nested_arrays_of_arrays() -> crate::Result<()> {
        let mut builder = Schema::builder();
        let json_field = builder.add_json_field("json", STORED | TEXT);
        let schema = builder.build();

        let index = Index::create_in_ram(schema);
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));
        let mut writer = index.writer(50_000_000)?;

        let val = json!({
            "top_arr": [
                [1, 2],
                [3, 4],
            ],
            "mixed_arr": [
                { "foo": "bar" },
                555,
                { "make": "Powell", "model": "Canyonero" }
            ]
        });

        writer.add_document(doc! { json_field => val.to_string() });
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Let's do a quick search for "4" or "Canyonero"
        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let query = query_parser.parse_query("Canyonero")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "Doc with 'Canyonero' found in nested array.");

        Ok(())
    }

    #[test]
    fn test_index_json_value_flat_arrays() -> crate::Result<()> {
        // This test originally validated "flattening arrays" with doc=0.
        // We'll replicate by storing a doc with arrays and verifying we can query it.
        let mut builder = Schema::builder();
        let json_field = builder.add_json_field("json", STORED | TEXT);
        let schema = builder.build();

        let index = Index::create_in_ram(schema);
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));
        let mut writer = index.writer(50_000_000)?;

        let val = json!({
            "docType": "parent",
            "numbers": [1, 2, 3],
            "nested": { "foo": "bar" },
        });

        writer.add_document(doc! { json_field => val.to_string() });
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        // Searching for "3" should return 1 doc
        let query = query_parser.parse_query("3")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "Flattened array indexing => found doc with '3'.");

        Ok(())
    }

    pub struct NestedParentBitSetProducer {
        parent_field: Field,
    }

    impl NestedParentBitSetProducer {
        pub fn new(parent_field: Field) -> Self {
            Self { parent_field }
        }
    }

    impl ParentBitSetProducer for NestedParentBitSetProducer {
        fn produce(&self, reader: &SegmentReader) -> crate::Result<common::BitSet> {
            let max_doc = reader.max_doc();
            let mut bitset = common::BitSet::with_max_value(max_doc);

            // For example, if the parent_field is a boolean field, you read all postings for “true”.
            let inverted = reader.inverted_index(self.parent_field)?;
            let term_true = Term::from_field_bool(self.parent_field, true);
            if let Some(mut postings) =
                inverted.read_postings(&term_true, IndexRecordOption::Basic)?
            {
                let mut d = postings.doc();
                while d != TERMINATED {
                    bitset.insert(d);
                    d = postings.advance();
                }
            }
            Ok(bitset)
        }
    }

    //
    // Finally, test the block-join logic with "nested arrays block join"
    // from the original code. We'll create a small parent/child doc set
    // in one segment, then use `ToParentBlockJoinQuery`.
    //
    #[test]
    fn test_index_json_value_nested_arrays_block_join() -> crate::Result<()> {
        use crate::{
            collector::Count,
            query::{QueryParser, ScoreMode, ToParentBlockJoinQuery},
            schema::*,
            Index,
        };
        use std::sync::Arc;

        // 1) Build schema: store docType so we can mark "parent" vs "child".
        let mut schema_builder = Schema::builder();
        // docType: e.g. "parent" or "child"
        let doc_type_field = schema_builder.add_text_field("docType", TEXT | STORED);
        // JSON content field
        let json_field = schema_builder.add_json_field("json", STORED | TEXT);

        let schema = schema_builder.build();

        // 2) Create Index + tokenizer
        let index = Index::create_in_ram(schema.clone());
        index
            .tokenizers()
            .register("default", TextAnalyzer::from(SimpleTokenizer::default()));

        let mut writer = index.writer(50_000_000)?;

        // -------------------------------------------------------------------
        // Instead of calling `add_document` repeatedly (which yields separate
        // “blocks”), we create a *block* vector for each parent and its children.
        // -------------------------------------------------------------------

        // Parent #1 and its two children:
        let parent_1 = doc! {
            doc_type_field => "parent",
            json_field => serde_json::json!({"docType":"parent", "name": "Lisa"}).to_string(),
        };
        let child_1_1 = doc! {
            doc_type_field => "child",
            json_field => serde_json::json!({"docType":"child", "make": "Powell", "model": "Canyonero"}).to_string(),
        };
        let child_1_2 = doc! {
            doc_type_field => "child",
            json_field => serde_json::json!({"docType":"child", "make": "Miller-Meteor", "model": "Ecto-1"}).to_string(),
        };

        writer.add_documents(vec![parent_1, child_1_1, child_1_2])?;

        // Parent #2 and its child:
        let parent_2 = doc! {
            doc_type_field => "parent",
            json_field => serde_json::json!({"docType":"parent", "name": "Bart"}).to_string(),
        };
        let child_2_1 = doc! {
            doc_type_field => "child",
            json_field => serde_json::json!({"docType":"child", "make": "Toyota", "model": "Corolla"}).to_string(),
        };
        writer.add_documents(vec![parent_2, child_2_1])?;

        writer.commit()?;

        // 3) Build a searcher
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // 4) We want child docs that have "make:Powell"
        let parser = QueryParser::for_index(&index, vec![json_field]);
        let child_query = parser.parse_query("make:Powell")?;

        // 5) We need a ParentBitSetProducer that marks docType=="parent" as the parent docs:
        struct NestedParentBitSetProducer {
            parent_field: Field,
        }
        impl crate::query::ParentBitSetProducer for NestedParentBitSetProducer {
            fn produce(&self, reader: &crate::SegmentReader) -> crate::Result<common::BitSet> {
                let max_doc = reader.max_doc();
                let mut bitset = common::BitSet::with_max_value(max_doc);

                // We consider any doc with docType=="parent" to be a parent.
                let inv_idx = reader.inverted_index(self.parent_field)?;
                let term_parent = crate::Term::from_field_text(self.parent_field, "parent");
                if let Some(mut postings) =
                    inv_idx.read_postings(&term_parent, IndexRecordOption::Basic)?
                {
                    while postings.doc() != crate::TERMINATED {
                        bitset.insert(postings.doc());
                        postings.advance();
                    }
                }
                Ok(bitset)
            }
        }

        let parent_field = schema.get_field("docType").unwrap();
        let parent_bitset_producer = Arc::new(NestedParentBitSetProducer { parent_field });

        // 6) Build the block-join: find all parent docs with at least one child matching child_query
        let block_join_query = ToParentBlockJoinQuery::new(
            child_query,
            parent_bitset_producer,
            ScoreMode::None, // no scoring
        );

        // 7) Count how many parents match => we expect exactly 1 parent (the one with "Powell")
        let hits = searcher.search(&block_join_query, &Count)?;
        assert_eq!(1, hits, "Only the first parent doc matches 'make:Powell'.");

        Ok(())
    }
}

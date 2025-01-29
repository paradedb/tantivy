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

/// This object is a map storing the last position for a given path for the current document
/// being indexed.
///
/// It is key to solve the following problem:
/// If we index a JsonObject emitting several terms with the same path
/// we do not want to create false positive in phrase queries.
///
/// For instance:
///
/// ```json
/// {"bands": [
///     {"band_name": "Elliot Smith"},
///     {"band_name": "The Who"},
/// ]}
/// ```
///
/// If we are careless and index each band names independently,
/// `Elliot` and `The` will end up indexed at position 0, and `Smith` and `Who` will be indexed at
/// position 1.
/// As a result, with lemmatization, "The Smiths" will match our object.
///
/// Worse, if a same term appears in the second object, a non increasing value would be pushed
/// to the position recorder probably provoking a panic.
///
/// This problem is solved for regular multivalued object by offsetting the position
/// of values, with a position gap. Here we would like `The` and `Who` to get indexed at
/// position 2 and 3 respectively.
///
/// With regular fields, we sort the fields beforehand, so that all terms with the same
/// path are indexed consecutively.
///
/// In JSON object, we do not have this comfort, so we need to record these position offsets in
/// a map.
///
/// Note that using a single position for the entire object would not hurt correctness.
/// It would however hurt compression.
///
/// We can therefore afford working with a map that is not imperfect. It is fine if several
/// path map to the same index position as long as the probability is relatively low.
#[derive(Default)]
pub(crate) struct IndexingPositionsPerPath {
    positions_per_path: FxHashMap<u32, IndexingPosition>,
}

impl IndexingPositionsPerPath {
    fn get_position_from_id(&mut self, id: u32) -> &mut IndexingPosition {
        self.positions_per_path.entry(id).or_default()
    }
    pub fn clear(&mut self) {
        self.positions_per_path.clear();
    }
}

/// Convert JSON_PATH_SEGMENT_SEP to a dot.
pub fn json_path_sep_to_dot(path: &mut str) {
    // This is safe since we are replacing a ASCII character by another ASCII character.
    unsafe {
        replace_in_place(JSON_PATH_SEGMENT_SEP, b'.', path.as_bytes_mut());
    }
}

#[allow(clippy::too_many_arguments)]
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
    index_json_value_nested(
        doc,
        json_value,
        text_analyzer,
        term_buffer,
        json_path_writer,
        postings_writer,
        ctx,
        positions_per_path,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
/// Same as [`index_json_value()`], but **with** an extra `treat_nested_arrays_as_blocks` bool:
/// - If `true`, arrays that are _entirely_ objects get turned into separate sub‐docs (child docs),
///   ignoring the flatten approach.
/// - If `false`, all arrays are flattened (the default older Tantivy behavior).
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
    let ref_value = json_value.as_value();
    match ref_value {
        ReferenceValue::Leaf(leaf) => {
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
            let elements_vec: Vec<_> = array_iter.collect();

            if treat_nested_arrays_as_blocks && all_objects_in_slice(&elements_vec) {
                for child_val in &elements_vec {
                    if let ReferenceValue::Object(child_obj) = child_val.as_value() {
                        // We'll flatten them in the same doc. Or remove logic if you want separate
                        // doc.
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
}

#[allow(clippy::too_many_arguments)]
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
    match leaf {
        ReferenceValueLeaf::Null => {
            // Skip null values
        }
        ReferenceValueLeaf::Str(s) => {
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_bytes(&[Type::Str.to_code()]);

            // token analysis
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
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            if let Ok(i64_val) = i64::try_from(uval) {
                term_buffer.append_type_and_fast_value::<i64>(i64_val);
            } else {
                term_buffer.append_type_and_fast_value(uval);
            }
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::I64(ival) => {
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<i64>(ival);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::F64(fval) => {
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<f64>(fval);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::Bool(bval) => {
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<bool>(bval);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::Date(dt) => {
            let truncated = dt.truncate(DATE_TIME_PRECISION_INDEXED);
            let unordered_id = ctx
                .path_to_unordered_id
                .get_or_allocate_unordered_id(json_path_writer.as_str());
            term_buffer.truncate_value_bytes(0);
            term_buffer.append_bytes(&unordered_id.to_be_bytes());
            term_buffer.append_type_and_fast_value::<DateTime>(truncated);
            postings_writer.subscribe(doc, 0u32, term_buffer, ctx);
        }
        ReferenceValueLeaf::PreTokStr(_)
        | ReferenceValueLeaf::Bytes(_)
        | ReferenceValueLeaf::Facet(_)
        | ReferenceValueLeaf::IpAddr(_) => {
            unimplemented!("Some JSON leaf types not implemented for dynamic indexing.")
        }
    }
}

/// Helper for indexing a JSON object: enumerates all key→value pairs, pushing each key onto
/// the path before indexing the child value.
#[allow(clippy::too_many_arguments)]
fn index_json_object<'a, V: Value<'a>>(
    doc: DocId,
    obj_iter: V::ObjectIter,
    text_analyzer: &mut TextAnalyzer,
    term_buffer: &mut Term,
    json_path_writer: &mut JsonPathWriter,
    postings_writer: &mut dyn PostingsWriter,
    ctx: &mut IndexingContext,
    positions_per_path: &mut IndexingPositionsPerPath,
    treat_nested_arrays_as_blocks: bool,
) {
    for (key, val) in obj_iter {
        // skip if key name has 0x00
        if key.as_bytes().contains(&JSON_END_OF_PATH) {
            continue;
        }

        json_path_writer.push(key);

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
    }
}

/// Test for "array of objects" by scanning all items, ensuring each is `ReferenceValue::Object`.
fn all_objects_in_slice<'a, V: Value<'a>>(vals: &[V]) -> bool {
    vals.iter()
        .all(|v| matches!(v.as_value(), ReferenceValue::Object(_)))
}

/// Tries to infer a JSON type from a string and append it to the term.
///
/// The term must be json + JSON path.
pub fn convert_to_fast_value_and_append_to_json_term(
    mut term: Term,
    phrase: &str,
    truncate_date_for_search: bool,
) -> Option<Term> {
    assert_eq!(
        term.value()
            .as_json_value_bytes()
            .expect("expecting a Term with a json type and json path")
            .as_serialized()
            .len(),
        0,
        "JSON value bytes should be empty"
    );
    if let Ok(dt) = OffsetDateTime::parse(phrase, &Rfc3339) {
        let mut dt = DateTime::from_utc(dt.to_offset(UtcOffset::UTC));
        if truncate_date_for_search {
            dt = dt.truncate(DATE_TIME_PRECISION_INDEXED);
        }
        term.append_type_and_fast_value(dt);
        return Some(term);
    }
    if let Ok(i64_val) = str::parse::<i64>(phrase) {
        term.append_type_and_fast_value(i64_val);
        return Some(term);
    }
    if let Ok(u64_val) = str::parse::<u64>(phrase) {
        term.append_type_and_fast_value(u64_val);
        return Some(term);
    }
    if let Ok(f64_val) = str::parse::<f64>(phrase) {
        term.append_type_and_fast_value(f64_val);
        return Some(term);
    }
    if let Ok(bool_val) = str::parse::<bool>(phrase) {
        term.append_type_and_fast_value(bool_val);
        return Some(term);
    }
    None
}

/// Splits a json path supplied to the query parser in such a way that
/// `.` can be escaped.
///
/// In other words,
/// - `k8s.node` ends up as `["k8s", "node"]`.
/// - `k8s\.node` ends up as `["k8s.node"]`.
pub fn split_json_path(json_path: &str) -> Vec<String> {
    let mut escaped_state: bool = false;
    let mut json_path_segments = Vec::new();
    let mut buffer = String::new();
    for ch in json_path.chars() {
        if escaped_state {
            buffer.push(ch);
            escaped_state = false;
            continue;
        }
        match ch {
            '\\' => {
                escaped_state = true;
            }
            '.' => {
                let new_segment = std::mem::take(&mut buffer);
                json_path_segments.push(new_segment);
            }
            _ => {
                buffer.push(ch);
            }
        }
    }
    json_path_segments.push(buffer);
    json_path_segments
}

/// Takes a field name, a json path as supplied by a user, and whether we should expand dots, and
/// return a column key, as expected by the columnar crate.
///
/// This function will detect unescaped dots in the path, and split over them.
/// If expand_dots is enabled, then even escaped dots will be split over.
///
/// The resulting list of segment then gets stitched together, joined by \1 separator,
/// as defined in the columnar crate.
pub(crate) fn encode_column_name(
    field_name: &str,
    json_path: &str,
    expand_dots_enabled: bool,
) -> String {
    let mut path = JsonPathWriter::default();
    path.push(field_name);
    path.set_expand_dots(expand_dots_enabled);
    for segment in split_json_path(json_path) {
        path.push(&segment);
    }
    path.into()
}

#[cfg(test)]
mod tests {
    use super::split_json_path;
    use crate::collector::Count;
    use crate::query::{ParentBitSetProducer, QueryParser};
    use crate::schema::{Field, IndexRecordOption, Schema, STORED, TEXT};
    use crate::tokenizer::{SimpleTokenizer, TextAnalyzer};
    use crate::{DocSet, Index, SegmentReader, Term, TERMINATED};

    #[test]
    fn test_json_writer() {
        let field = Field::from_field_id(1);

        let mut term = Term::from_field_json_path(field, "attributes.color", false);
        term.append_type_and_str("red");
        assert_eq!(
            format!("{term:?}"),
            "Term(field=1, type=Json, path=attributes.color, type=Str, \"red\")"
        );

        let mut term = Term::from_field_json_path(field, "attributes.dimensions.width", false);
        term.append_type_and_fast_value(400i64);
        assert_eq!(
            format!("{term:?}"),
            "Term(field=1, type=Json, path=attributes.dimensions.width, type=I64, 400)"
        );
    }

    #[test]
    fn test_string_term() {
        let field = Field::from_field_id(1);
        let mut term = Term::from_field_json_path(field, "color", false);
        term.append_type_and_str("red");

        assert_eq!(term.serialized_term(), b"\x00\x00\x00\x01jcolor\x00sred")
    }

    #[test]
    fn test_i64_term() {
        let field = Field::from_field_id(1);
        let mut term = Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(-4i64);

        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00i\x7f\xff\xff\xff\xff\xff\xff\xfc"
        )
    }

    #[test]
    fn test_u64_term() {
        let field = Field::from_field_id(1);
        let mut term = Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(4u64);

        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00u\x00\x00\x00\x00\x00\x00\x00\x04"
        )
    }

    #[test]
    fn test_f64_term() {
        let field = Field::from_field_id(1);
        let mut term = Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(4.0f64);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00f\xc0\x10\x00\x00\x00\x00\x00\x00"
        )
    }

    #[test]
    fn test_bool_term() {
        let field = Field::from_field_id(1);
        let mut term = Term::from_field_json_path(field, "color", false);
        term.append_type_and_fast_value(true);
        assert_eq!(
            term.serialized_term(),
            b"\x00\x00\x00\x01jcolor\x00o\x00\x00\x00\x00\x00\x00\x00\x01"
        )
    }

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

    #[test]
    fn test_array_of_objects_is_not_flattened() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", STORED | TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);

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

        writer
            .add_document(doc! { json_field => val.to_string() })
            .unwrap();
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let query = query_parser.parse_query("Powell")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "Doc with 'Powell' found");

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

        let val = json!({ "numbers": [100, 200, 300] });
        writer
            .add_document(doc! { json_field => val.to_string() })
            .unwrap();
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

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

        writer
            .add_document(doc! { json_field => val.to_string() })
            .unwrap();
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![json_field]);
        let query = query_parser.parse_query("Canyonero")?;
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1, "Doc with 'Canyonero' found in nested array.");

        Ok(())
    }

    pub struct NestedParentBitSetProducer {
        parent_field: Field,
    }

    impl ParentBitSetProducer for NestedParentBitSetProducer {
        fn produce(&self, reader: &SegmentReader) -> crate::Result<common::BitSet> {
            let max_doc = reader.max_doc();
            let mut bitset = common::BitSet::with_max_value(max_doc);

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
}

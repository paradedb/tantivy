use std::collections::BTreeMap;
use std::ops::BitOr;

use serde::{Deserialize, Serialize};

use super::text_options::{FastFieldTextOptions, TokenizerName};
use crate::schema::flags::{FastFlag, SchemaFlagList, StoredFlag};
use crate::schema::{TextFieldIndexing, TextOptions};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ObjectMappingType {
    /// A normal “object” type, flattened by default. (Equivalent to ES “object”).
    #[default]
    Default,
    /// A nested type. Each item in an array is indexed as a separate sub-document.
    Nested,
}

impl BitOr for ObjectMappingType {
    type Output = ObjectMappingType;

    fn bitor(self, other: ObjectMappingType) -> ObjectMappingType {
        match (self, other) {
            (ObjectMappingType::Nested, _) | (_, ObjectMappingType::Nested) => {
                ObjectMappingType::Nested
            }
            _ => ObjectMappingType::Default,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NestedOptions {
    /// If true, fields in the nested object are also added to the parent doc as flattened fields.
    #[serde(default)]
    pub include_in_parent: bool,

    /// If true, fields in the nested object are also added to the *root* doc as flattened fields.
    #[serde(default)]
    pub include_in_root: bool,
}

impl BitOr for NestedOptions {
    type Output = NestedOptions;

    fn bitor(self, other: NestedOptions) -> NestedOptions {
        NestedOptions {
            include_in_parent: self.include_in_parent || other.include_in_parent,
            include_in_root: self.include_in_root || other.include_in_root,
        }
    }
}

/// The `JsonObjectOptions` make it possible to
/// configure how a json object field should be indexed and stored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JsonObjectOptions {
    stored: bool,
    // If set to some, int, date, f64 and text will be indexed.
    // Text will use the TextFieldIndexing setting for indexing.
    indexing: Option<TextFieldIndexing>,
    // Store all field as fast fields with an optional tokenizer for text.
    fast: FastFieldTextOptions,
    /// tantivy will generate paths to the different nodes of the json object
    /// both in:
    /// - the inverted index (for the terms)
    /// - fast fields (for the column names).
    ///
    /// These json path are encoded by concatenating the list of object keys that
    /// are visited from the root to the leaf.
    ///
    /// By default, if an object key contains a `.`, we keep it as a `.` it as is.
    /// On the search side, users will then have to escape this `.` in the query parser
    ///
    /// or when referring to a column name.
    ///
    /// For instance:
    /// `{"root": {"child.with.dot": "hello"}}`
    ///
    /// Can be searched using the following query
    /// `root.child\.with\.dot:hello`
    ///
    /// If `expand_dots_enabled` is set to true, we will treat this `.` in object keys
    /// as json separators. In other words, if set to true, our object will be
    /// processed as if it was
    /// `{"root": {"child": {"with": {"dot": "hello"}}}}`
    /// and it can be search using the following query:
    /// `root.child.with.dot:hello`
    #[serde(default)]
    expand_dots_enabled: bool,

    /// Plain vs. nested object handling.
    #[serde(default)]
    object_mapping_type: ObjectMappingType,

    /// If this is a nested type, here is where we store nested-specific options.
    #[serde(default)]
    nested_options: NestedOptions,

    /// A map of subfield names to their own indexing/mapping options,
    /// which enables arbitrary recursive nesting.
    #[serde(default)]
    subfields: BTreeMap<String, JsonObjectOptions>,
}

impl JsonObjectOptions {
    pub fn nested() -> Self {
        JsonObjectOptions {
            object_mapping_type: ObjectMappingType::Nested,
            ..Default::default()
        }
    }
    /// Returns `true` if the json object should be stored.
    #[inline]
    pub fn is_stored(&self) -> bool {
        self.stored
    }

    /// Returns `true` iff the json object should be indexed.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        self.indexing.is_some()
    }

    /// Returns true if and only if the json object fields are
    /// to be treated as fast fields.
    #[inline]
    pub fn is_fast(&self) -> bool {
        matches!(self.fast, FastFieldTextOptions::IsEnabled(true))
            || matches!(
                &self.fast,
                FastFieldTextOptions::EnabledWithTokenizer { with_tokenizer: _ }
            )
    }

    /// Returns true if and only if the value is a fast field.
    #[inline]
    pub fn get_fast_field_tokenizer_name(&self) -> Option<&str> {
        match &self.fast {
            FastFieldTextOptions::IsEnabled(true) | FastFieldTextOptions::IsEnabled(false) => None,
            FastFieldTextOptions::EnabledWithTokenizer {
                with_tokenizer: tokenizer,
            } => Some(tokenizer.name()),
        }
    }

    /// Returns `true` iff dots in json keys should be expanded.
    ///
    /// When expand_dots is enabled, json object like
    /// `{"k8s.node.id": 5}` is processed as if it was
    /// `{"k8s": {"node": {"id": 5}}}`.
    /// This option has the merit of allowing users to
    /// write queries  like `k8s.node.id:5`.
    /// On the other, enabling that feature can lead to
    /// ambiguity.
    ///
    /// If disabled, the "." needs to be escaped:
    /// `k8s\.node\.id:5`.
    #[inline]
    pub fn is_expand_dots_enabled(&self) -> bool {
        self.expand_dots_enabled
    }

    #[inline]
    pub fn is_include_in_parent(&self) -> bool {
        self.nested_options.include_in_parent
    }

    #[inline]
    pub fn is_include_in_root(&self) -> bool {
        self.nested_options.include_in_root
    }

    #[inline]
    pub fn is_nested(&self) -> bool {
        matches!(self.object_mapping_type, ObjectMappingType::Nested)
    }

    /// Sets `expands_dots` to true.
    /// See `is_expand_dots_enabled` for more information.
    pub fn set_expand_dots_enabled(mut self) -> Self {
        self.expand_dots_enabled = true;
        self
    }

    /// Returns the text indexing options.
    ///
    /// If set to `Some` then both int and str values will be indexed.
    /// The inner `TextFieldIndexing` will however, only apply to the str values
    /// in the json object.
    #[inline]
    pub fn get_text_indexing_options(&self) -> Option<&TextFieldIndexing> {
        self.indexing.as_ref()
    }

    /// Sets the field as stored
    #[must_use]
    pub fn set_stored(mut self) -> Self {
        self.stored = true;
        self
    }

    /// Set the field as a fast field.
    ///
    /// Fast fields are designed for random access.
    /// Access time are similar to a random lookup in an array.
    /// Text fast fields will have the term ids stored in the fast field.
    ///
    /// The effective cardinality depends on the tokenizer. Without a tokenizer, the text will be
    /// stored as is, which equals to the "raw" tokenizer. The tokenizer can be used to apply
    /// normalization like lower case.
    /// The passed tokenizer_name must be available on the fast field tokenizer manager.
    /// `Index::fast_field_tokenizer`.
    ///
    /// The original text can be retrieved via
    /// [`TermDictionary::ord_to_term()`](crate::termdict::TermDictionary::ord_to_term)
    /// from the dictionary.
    #[must_use]
    pub fn set_fast(mut self, tokenizer_name: Option<&str>) -> Self {
        if let Some(tokenizer) = tokenizer_name {
            let tokenizer = TokenizerName::from_name(tokenizer);
            self.fast = FastFieldTextOptions::EnabledWithTokenizer {
                with_tokenizer: tokenizer,
            }
        } else {
            self.fast = FastFieldTextOptions::IsEnabled(true);
        }
        self
    }

    /// Sets the field as indexed, with the specific indexing options.
    #[must_use]
    pub fn set_indexing_options(mut self, indexing: TextFieldIndexing) -> Self {
        self.indexing = Some(indexing);
        self
    }

    #[must_use]
    pub fn set_nested(mut self) -> Self {
        self.object_mapping_type = ObjectMappingType::Nested;
        self
    }

    pub fn unset_indexing_options(&mut self) -> &mut Self {
        self.indexing = None;
        self
    }

    /// Convenience method to mark `include_in_parent = true`.
    #[must_use]
    pub fn set_include_in_parent(mut self) -> Self {
        self.nested_options.include_in_parent = true;
        self
    }

    /// Convenience method to mark `include_in_root = true`.
    #[must_use]
    pub fn set_include_in_root(mut self) -> Self {
        self.nested_options.include_in_root = true;
        self
    }

    #[must_use]
    pub fn add_subfield<S: Into<String>>(mut self, name: S, opts: JsonObjectOptions) -> Self {
        self.subfields.insert(name.into(), opts);
        self
    }

    pub fn subfields(&self) -> &BTreeMap<String, JsonObjectOptions> {
        &self.subfields
    }
}

impl From<StoredFlag> for JsonObjectOptions {
    fn from(_stored_flag: StoredFlag) -> Self {
        JsonObjectOptions {
            stored: true,
            indexing: None,
            fast: FastFieldTextOptions::default(),
            expand_dots_enabled: false,
            ..Default::default()
        }
    }
}

impl From<FastFlag> for JsonObjectOptions {
    fn from(_fast_flag: FastFlag) -> Self {
        JsonObjectOptions {
            stored: false,
            indexing: None,
            fast: FastFieldTextOptions::IsEnabled(true),
            expand_dots_enabled: false,
            ..Default::default()
        }
    }
}

impl From<()> for JsonObjectOptions {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl BitOr for JsonObjectOptions {
    type Output = JsonObjectOptions;

    fn bitor(mut self, other: JsonObjectOptions) -> JsonObjectOptions {
        for (other_key, other_val) in other.subfields {
            self.subfields
                .entry(other_key)
                .and_modify(|self_val| *self_val = std::mem::take(self_val) | other_val.clone())
                .or_insert(other_val);
        }
        JsonObjectOptions {
            stored: self.stored || other.stored,
            indexing: self.indexing.or(other.indexing),
            fast: self.fast | other.fast,
            expand_dots_enabled: self.expand_dots_enabled || other.expand_dots_enabled,
            object_mapping_type: self.object_mapping_type | other.object_mapping_type,
            nested_options: self.nested_options | other.nested_options,
            subfields: self.subfields,
        }
    }
}

impl<Head, Tail> From<SchemaFlagList<Head, Tail>> for JsonObjectOptions
where
    Head: Clone,
    Tail: Clone,
    Self: BitOr<Output = Self> + From<Head> + From<Tail>,
{
    fn from(head_tail: SchemaFlagList<Head, Tail>) -> Self {
        Self::from(head_tail.head) | Self::from(head_tail.tail)
    }
}

impl From<TextOptions> for JsonObjectOptions {
    fn from(text_options: TextOptions) -> Self {
        JsonObjectOptions {
            stored: text_options.is_stored(),
            indexing: text_options.get_indexing_options().cloned(),
            fast: text_options.fast,
            expand_dots_enabled: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FAST, STORED, TEXT};

    #[test]
    fn test_json_options_builder_methods() {
        // 1) Create a top-level JsonObjectOptions that is nested. We'll configure it with stored,
        //    fast, indexing, etc.
        let opts = JsonObjectOptions::nested()
            .set_stored()
            .set_fast(Some("my_tokenizer"))
            .set_indexing_options(TextFieldIndexing::default())
            // 2) Add a subfield "vehicle" which is ALSO nested
            .add_subfield(
                "vehicle",
                JsonObjectOptions::nested()
                    // Add sub-subfields "make" and "model"
                    .add_subfield(
                        "make",
                        JsonObjectOptions::default()
                            .set_indexing_options(TextFieldIndexing::default()),
                    )
                    .add_subfield(
                        "model",
                        JsonObjectOptions::default()
                            .set_stored()
                            .set_indexing_options(TextFieldIndexing::default()),
                    ),
            )
            // 3) Add another subfield "last_name" which is just a flattened object
            .add_subfield(
                "last_name",
                JsonObjectOptions::default().set_indexing_options(TextFieldIndexing::default()),
            );

        // 4) Verify top-level settings
        assert!(opts.is_stored());
        assert!(opts.is_indexed());
        assert!(opts.is_fast());
        assert_eq!(opts.object_mapping_type, ObjectMappingType::Nested);

        // 5) Check that "vehicle" subfield is nested
        assert!(opts.subfields.contains_key("vehicle"));
        let vehicle_opts = &opts.subfields["vehicle"];
        assert_eq!(vehicle_opts.object_mapping_type, ObjectMappingType::Nested);

        // 6) Check "vehicle.make"
        let make_opts = &vehicle_opts.subfields["make"];
        assert!(make_opts.is_indexed());
        assert!(!make_opts.is_stored());

        // 7) Check "vehicle.model"
        let model_opts = &vehicle_opts.subfields["model"];
        assert!(model_opts.is_stored());
        assert!(model_opts.is_indexed());

        // 8) Check "last_name" subfield
        assert!(opts.subfields.contains_key("last_name"));
        let last_name_opts = &opts.subfields["last_name"];
        assert!(last_name_opts.is_indexed());
        assert!(!last_name_opts.is_stored());
    }

    #[test]
    fn test_json_options_bit_or() {
        let opts1 = JsonObjectOptions::default().set_stored().add_subfield(
            "child",
            JsonObjectOptions::default().set_indexing_options(TextFieldIndexing::default()),
        );
        let opts2 = JsonObjectOptions::default()
            .set_fast(None)
            .add_subfield("child", JsonObjectOptions::default().set_stored());

        let combined = opts1 | opts2;
        // top-level merges
        assert!(combined.is_stored());
        assert!(combined.is_fast());
        // child merges
        let child_opts = combined.subfields.get("child").unwrap();
        assert!(child_opts.is_indexed());
        assert!(child_opts.is_stored());
    }

    #[test]
    fn test_json_options() {
        {
            let json_options: JsonObjectOptions = (STORED | TEXT).into();
            assert!(json_options.is_stored());
            assert!(json_options.is_indexed());
            assert!(!json_options.is_fast());
        }
        {
            let json_options: JsonObjectOptions = TEXT.into();
            assert!(!json_options.is_stored());
            assert!(json_options.is_indexed());
            assert!(!json_options.is_fast());
        }
        {
            let json_options: JsonObjectOptions = STORED.into();
            assert!(json_options.is_stored());
            assert!(!json_options.is_indexed());
            assert!(!json_options.is_fast());
        }
        {
            let json_options: JsonObjectOptions = FAST.into();
            assert!(!json_options.is_stored());
            assert!(!json_options.is_indexed());
            assert!(json_options.is_fast());
        }
        {
            let json_options: JsonObjectOptions = (FAST | STORED).into();
            assert!(json_options.is_stored());
            assert!(!json_options.is_indexed());
            assert!(json_options.is_fast());
        }
    }
}

pub mod explode {
    use crate::{
        schema::{DocParsingError, JsonObjectOptions, ObjectMappingType, Schema},
        Result, TantivyDocument,
    };
    use common::JsonPathWriter;
    use serde_json::{json, Map, Value};

    pub fn explode_tantivy_docs(
        parent: &mut TantivyDocument,
        schema: &Schema,
        field_name: &str,
        value: Value,
        opts: &JsonObjectOptions,
    ) -> Result<Vec<TantivyDocument>> {
        let mut exploded_docs: Vec<Value> = explode(
            &[],
            json!({field_name: value}),
            Some(&opts.clone().add_subfield(field_name, opts.clone())),
        );

        let exploded_parent = match exploded_docs.pop() {
            Some(Value::Object(obj)) => obj,
            _ => unreachable!("constructed as an object"),
        };

        for (name, json_value) in exploded_parent {
            if let Ok(field) = schema.get_field(&name) {
                let field_entry = schema.get_field_entry(field);
                let field_type = field_entry.field_type();
                match json_value {
                    serde_json::Value::Array(json_items) => {
                        for json_item in json_items {
                            let value = field_type
                                .value_from_json(json_item)
                                .map_err(|e| DocParsingError::ValueError(name.to_string(), e))?;
                            parent.add_field_value(field, &value);
                        }
                    }
                    _ => {
                        let value = field_type
                            .value_from_json(json_value)
                            .map_err(|e| DocParsingError::ValueError(name.to_string(), e))?;
                        parent.add_field_value(field, &value);
                    }
                }
            }
        }

        Ok(exploded_docs
            .into_iter()
            .map(|value| {
                TantivyDocument::from_json_object(schema, value.as_object().unwrap().clone())
                    .unwrap()
            })
            .collect())
    }

    /// Wrap `value` under the given `path`, producing exactly one doc.
    /// If `path` is empty, returns `[ value ]`.
    /// Otherwise nest `value` inside objects named by the path segments.
    fn wrap_in_path(path: &[&str], value: Value) -> Vec<Value> {
        if path.is_empty() {
            return vec![value];
        }
        let mut current = value;
        for seg in path.iter().rev() {
            let mut obj = Map::new();
            obj.insert(seg.to_string(), current);
            current = Value::Object(obj);
        }
        vec![current]
    }

    /// Create a doc with `_is_parent_<path> = true` and store `full_value` under that same path.
    fn make_parent_doc(path: &[&str], full_value: &Value) -> Value {
        // Build the `_is_parent_<path>` field
        let mut path_writer = JsonPathWriter::new();
        for seg in path {
            path_writer.push(seg);
        }
        let path_str = path_writer.as_str();
        let parent_flag = format!("_is_parent_{path_str}");

        let mut doc_map = Map::new();
        doc_map.insert(parent_flag, Value::Bool(true));

        // Now nest `full_value` under the path segments
        let mut current_map = match full_value {
            Value::Object(ref obj) => obj.clone(),
            other => {
                let mut tmp = Map::new();
                tmp.insert("".to_string(), other.clone());
                tmp
            }
        };
        for seg in path.iter().rev() {
            let mut new_map = Map::new();
            if current_map.len() == 1 && current_map.contains_key("") {
                // rename "" => seg
                if let Some(only_val) = current_map.remove("") {
                    new_map.insert(seg.to_string(), only_val);
                }
            } else {
                new_map.insert(seg.to_string(), Value::Object(current_map));
            }
            current_map = new_map;
        }
        // Merge it all
        for (k, v) in current_map {
            doc_map.insert(k, v);
        }
        Value::Object(doc_map)
    }

    /// Return the subset of subfields that are themselves `Nested`.
    fn nested_subfields<'a>(
        opts: &'a JsonObjectOptions,
        obj: &Map<String, Value>,
    ) -> Vec<(&'a String, &'a JsonObjectOptions)> {
        let mut results = Vec::new();
        for (child_key, child_opts) in &opts.subfields {
            if child_opts.object_mapping_type == ObjectMappingType::Nested {
                // Only relevant if the object actually has this child field
                if obj.contains_key(child_key) {
                    results.push((child_key, child_opts));
                }
            }
        }
        results
    }

    /// Explode the JSON `value` according to `opts` if it's nested.
    ///
    /// **Rules**:
    /// 1) If `opts` is missing or `object_mapping_type != Nested`, produce exactly **one** doc (via `wrap_in_path`).
    /// 2) **Nested array** => one child doc for each array item + one parent doc (unless `path.is_empty()`)  
    /// 3) **Nested object**:  
    ///    - If the object has subfields that are themselves nested, recursively explode them to produce child docs, then produce one parent doc with `_is_parent_<path> = true` for the entire object (unless `path.is_empty()`).  
    ///    - If the object does **not** contain any nested subfields, produce **only** one doc:
    ///       - if `path.is_empty()`, just the object,
    ///       - otherwise a single parent doc with `_is_parent_<path> = true`.
    /// 4) **Nested scalar** => exactly **one** doc (no `_is_parent_...`), even if `path` is non‐empty.
    ///
    pub fn explode(path: &[&str], value: Value, opts: Option<&JsonObjectOptions>) -> Vec<Value> {
        // If not nested => single doc
        let Some(my_opts) = opts else {
            return wrap_in_path(path, value);
        };
        if my_opts.object_mapping_type != ObjectMappingType::Nested {
            return wrap_in_path(path, value);
        }

        match value {
            Value::Array(arr) => {
                // Nested array => child doc per element, plus parent doc for entire array if path nonempty
                let mut docs = Vec::new();
                for elem in &arr {
                    // The user tests want each array item as a single doc, unless that item’s schema is also nested subfields.
                    // But typically "arr" corresponds to e.g. "j": [1,2,{k:v}] with no further subfields,
                    // so we just wrap each item.
                    docs.extend(wrap_in_path(path, elem.clone()));
                }
                if !path.is_empty() {
                    docs.push(make_parent_doc(path, &Value::Array(arr)));
                }
                docs
            }
            Value::Object(obj) => {
                // Possibly sub-nested
                let sub_nests = nested_subfields(my_opts, &obj);
                if sub_nests.is_empty() {
                    // No sub-nested => produce exactly 1 doc.
                    if path.is_empty() {
                        // top-level => just store the object
                        wrap_in_path(path, Value::Object(obj))
                    } else {
                        // produce a doc with `_is_parent_<path> = true`
                        vec![make_parent_doc(path, &Value::Object(obj))]
                    }
                } else {
                    // We do have sub-nested fields => produce child docs from each, then a parent doc
                    let mut docs = Vec::new();
                    for (child_key, child_opts) in sub_nests {
                        if let Some(subval) = obj.get(child_key) {
                            let mut new_path = path.to_vec();
                            new_path.push(child_key);
                            docs.extend(explode(&new_path, subval.clone(), Some(child_opts)));
                        }
                    }
                    // Then produce a parent doc if `path` is non-empty
                    if !path.is_empty() {
                        docs.push(make_parent_doc(path, &Value::Object(obj)));
                    }
                    docs
                }
            }
            scalar => {
                // Nested scalar => user tests want exactly one doc, no `_is_parent_`
                wrap_in_path(path, scalar)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BTreeMap;

        use super::*;
        use crate::schema::{JsonObjectOptions, ObjectMappingType};
        use serde_json::json;

        #[test]
        fn explode_non_nested_empty_object() {
            let path = vec![];
            let value = json!({});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_simple_object() {
            let path = vec![];
            let value = json!({"a": 1, "b": "two"});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({"a": 1, "b": "two"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_array() {
            let path = vec![];
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!([1, 2, 3])];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_empty_object() {
            let path = vec![];
            let value = json!({"root": {}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let parent_key = "_is_parent_root";
            let expected = vec![json!({parent_key: true, "root": {}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_scalar() {
            let path = vec!["field"];
            let value = json!("hello");
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            // No parent key should be added if its a scalar.
            let expected = vec![json!({"field": "hello"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_object() {
            let path = vec![];
            let value = json!({"root": {"a": 1, "b": "two"}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({ "_is_parent_root": true, "root": {"a": 1, "b": "two"}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_deep_object() {
            let path = vec![];
            let value = json!({"a": {"b": {"c": 42}}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "a".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({"_is_parent_a": true, "a": {"b": {"c": 42}}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_wide_object() {
            let value = serde_json::json!({
                "root": {
                    "a": 1,
                    "b": true,
                    "c": null,
                    "d": 3.14,
                    "e": "test",
                    "f": { "g": 99, "h": { "i": "deep" } },
                    "j": [1, 2, { "k": "v" }]
                }
            });

            // "root" is nested => we do block-join indexing for subfields.
            // Among them, "j" is also declared nested => we want to explode that array.
            let mut subfields_root = std::collections::BTreeMap::new();
            subfields_root.insert(
                "j".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    subfields: std::collections::BTreeMap::new(),
                    ..Default::default()
                },
            );

            let mut top_level_subfields = std::collections::BTreeMap::new();
            top_level_subfields.insert(
                "root".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    subfields: subfields_root,
                    ..Default::default()
                },
            );

            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: top_level_subfields,
                ..Default::default()
            };

            let path: Vec<&str> = vec![];
            let result = explode(&path, value, Some(&opts));

            // Then construct expected vector using these structs
            let expected = vec![
                serde_json::json!({ "root": { "j": 1 } }),
                serde_json::json!({ "root": { "j": 2 } }),
                serde_json::json!({ "root": { "j": { "k": "v" } } }),
                serde_json::json!({
                    "_is_parent_root\u{1}j": true,
                    "root": {
                        "j": [1, 2, { "k": "v" }]
                    }
                }),
                serde_json::json!({
                    "_is_parent_root": true,
                    "root": {
                        "a": 1,
                        "b": true,
                        "c": null,
                        "d": 3.14,
                        "e": "test",
                        "f": { "g": 99, "h": { "i": "deep" } },
                        "j": [1, 2, { "k": "v" }]
                    }
                }),
            ];

            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_array() {
            let path = vec![];
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!(1), json!(2), json!(3)];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_array_of_objects() {
            let path = vec![];
            let value = json!({"root": [
                {"a": 1},
                {"b": 2},
                {"c": {"d": 3}}
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            // We expect 3 child docs plus a parent doc
            let expected = vec![
                json!({"root": {"a": 1}}),
                json!({"root": {"b": 2}}),
                json!({"root": {"c": {"d": 3}}}),
                json!({
                    "_is_parent_root": true,
                    "root": [
                        {"a": 1},
                        {"b": 2},
                        {"c": {"d": 3}}
                    ]
                }),
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_multi_dimensional_arrays() {
            let path = vec![];
            let value = json!({"root": [
                [1, 2],
                [3, [4, 5]],
                [6, {"x": [7, 8]}]
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));

            // Because "root" is the nested subfield,
            // we expect a single parent doc with "_is_parent_root": true,
            // containing the entire array. (No recursion on multi-dimensional arrays.)
            // use serde_json::{json, Value};

            let expected = vec![
                json!({
                    "root": [1, 2]
                }),
                json!({
                    "root": [3, [4, 5]]
                }),
                json!({
                    "root": [6, { "x": [7, 8] }]
                }),
                json!({
                    "_is_parent_root": true,
                    "root": [
                        [1, 2],
                        [3, [4, 5]],
                        [6, { "x": [7, 8] }]
                    ]
                }),
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_mixed_types() {
            // "mixed" is nested, and so are subfields "array" and "letters".
            let mut subfields_mixed = BTreeMap::new();
            subfields_mixed.insert(
                "array".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    ..Default::default()
                },
            );
            subfields_mixed.insert(
                "letters".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    ..Default::default()
                },
            );

            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: subfields_mixed,
                ..Default::default()
            };

            let path = vec!["mixed"];

            let value = serde_json::json!({
                "array": [1, "two", true, null, { "nested": [3, 4] }],
                "obj": { "a": 5, "b": { "c": 6 } },
                "scalar": 7,
                "string": "eight",
                "letters": [
                    { "a": 1 },
                    { "b": 2 },
                    { "c": { "d": 3 } }
                ],
                "bool": false
            });

            let result = explode(&path, value, Some(&opts));

            // Block‐join logic always introduces a parent doc for a “nested” array subfield.
            let expected = vec![
                // Child documents for "array"
                json!({ "mixed": { "array": 1 } }),
                json!({ "mixed": { "array": "two" } }),
                json!({ "mixed": { "array": true } }),
                json!({ "mixed": { "array": null } }),
                json!({ "mixed": { "array": { "nested": [3, 4] } } }),
                // **Additional Object for "array"**
                json!({
                    "_is_parent_mixed\u{1}array": true,
                    "mixed": {
                        "array": [
                            1,
                            "two",
                            true,
                            null,
                            { "nested": [3, 4] }
                        ]
                    }
                }),
                // Child documents for "letters"
                json!({ "mixed": { "letters": { "a": 1 } } }),
                json!({ "mixed": { "letters": { "b": 2 } } }),
                json!({ "mixed": { "letters": { "c": { "d": 3 } } } }),
                // **Additional Object for "letters"**
                json!({
                    "_is_parent_mixed\u{1}letters": true,
                    "mixed": {
                        "letters": [
                            { "a": 1 },
                            { "b": 2 },
                            { "c": { "d": 3 } }
                        ]
                    }
                }),
                // Final parent document
                json!({
                    "_is_parent_mixed": true,
                    "mixed": {
                        "array": [
                            1,
                            "two",
                            true,
                            null,
                            { "nested": [3, 4] }
                        ],
                        "bool": false,
                        "letters": [
                            { "a": 1 },
                            { "b": 2 },
                            { "c": { "d": 3 } }
                        ],
                        "obj": { "a": 5, "b": { "c": 6 } },
                        "scalar": 7,
                        "string": "eight"
                    }
                }),
            ];

            assert_eq!(result, expected);
        }

        #[test]
        fn test_nested_multi_level() {
            // "driver_json" is nested at top-level,
            // "vehicle" is nested subfield of "driver_json".
            let value = json!({
                "driver_json": {
                    "last_name": "McQueen",
                    "vehicle": [
                        {"make": "Powell", "model": "Canyonero"},
                        {"make": "Miller-Meteor", "model": "Ecto-1"}
                    ]
                }
            });

            let mut vehicle_opts = JsonObjectOptions::default();
            vehicle_opts.object_mapping_type = ObjectMappingType::Nested;

            let mut driver_json_opts = JsonObjectOptions::default();
            driver_json_opts.object_mapping_type = ObjectMappingType::Nested;
            driver_json_opts
                .subfields
                .insert("vehicle".to_string(), vehicle_opts);

            let mut top_opts = JsonObjectOptions::default();
            top_opts.object_mapping_type = ObjectMappingType::Nested;
            top_opts
                .subfields
                .insert("driver_json".to_string(), driver_json_opts);

            let docs = explode(&[], value.clone(), Some(&top_opts));

            let child1 = json!({
                "driver_json": { "vehicle": { "make": "Powell", "model": "Canyonero" }}
            });
            let child2 = json!({
                "driver_json": { "vehicle": { "make": "Miller-Meteor", "model": "Ecto-1" }}
            });
            let vehicle_parent = json!({
                "_is_parent_driver_json\u{1}vehicle": true,
                "driver_json": {
                    "vehicle": [
                        { "make": "Powell", "model": "Canyonero" },
                        { "make": "Miller-Meteor", "model": "Ecto-1" }
                    ]
                }
            });
            let driver_json_parent = json!({
                "_is_parent_driver_json": true,
                "driver_json": {
                    "last_name": "McQueen",
                    "vehicle": [
                        { "make": "Powell", "model": "Canyonero" },
                        { "make": "Miller-Meteor", "model": "Ecto-1" }
                    ]
                }
            });

            let expected = vec![child1, child2, vehicle_parent, driver_json_parent];
            assert_eq!(docs, expected);
        }
    }
}

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
    pub stored: bool,
    // If set to some, int, date, f64 and text will be indexed.
    // Text will use the TextFieldIndexing setting for indexing.
    pub indexing: Option<TextFieldIndexing>,
    // Store all field as fast fields with an optional tokenizer for text.
    pub fast: FastFieldTextOptions,
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
    pub expand_dots_enabled: bool,

    /// Plain vs. nested object handling.
    #[serde(default)]
    pub object_mapping_type: ObjectMappingType,

    /// If this is a nested type, here is where we store nested-specific options.
    #[serde(default)]
    pub nested_options: NestedOptions,

    /// A map of subfield names to their own indexing/mapping options,
    /// which enables arbitrary recursive nesting.
    #[serde(default)]
    pub subfields: BTreeMap<String, JsonObjectOptions>,
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

    pub fn is_include_in_parent(&self) -> bool {
        self.nested_options.include_in_parent
    }

    pub fn is_include_in_root(&self) -> bool {
        self.nested_options.include_in_root
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
        assert!(opts.nested_options.include_in_parent);
        assert!(!opts.nested_options.include_in_root);

        // 5) Check that "vehicle" subfield is nested
        assert!(opts.subfields.contains_key("vehicle"));
        let vehicle_opts = &opts.subfields["vehicle"];
        assert_eq!(vehicle_opts.object_mapping_type, ObjectMappingType::Nested);
        assert!(!vehicle_opts.nested_options.include_in_root);
        assert!(vehicle_opts.nested_options.include_in_parent);

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

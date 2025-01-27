use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use common::JsonPathWriter;
use serde::de::{SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::nested_options::NestedOptions;
use super::*;
use crate::json_utils::split_json_path;
use crate::TantivyError;

/// Tantivy has a very strict schema.
/// You need to specify in advance whether a field is indexed or not,
/// stored or not, and RAM-based or not.
///
/// This is done by creating a schema object, and
/// setting up the fields one by one.
/// It is for the moment impossible to remove fields.
///
/// # Examples
///
/// ```
/// use tantivy::schema::*;
///
/// let mut schema_builder = Schema::builder();
/// let id_field = schema_builder.add_text_field("id", STRING);
/// let title_field = schema_builder.add_text_field("title", TEXT);
/// let body_field = schema_builder.add_text_field("body", TEXT);
/// let schema = schema_builder.build();
/// ```
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: Vec<FieldEntry>,
    fields_map: HashMap<String, Field>,
    nested_paths: HashMap<Vec<String>, Field>,
}

impl SchemaBuilder {
    /// Create a new `SchemaBuilder`
    pub fn new() -> SchemaBuilder {
        println!("Creating new SchemaBuilder");
        let builder = SchemaBuilder::default();
        println!("Created empty SchemaBuilder");
        builder
    }

    /// Adds a new u64 field.
    /// Returns the associated field handle
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_u64_field<T: Into<NumericOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_u64_field => Adding u64 field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_u64(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_u64_field => Added u64 field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new i64 field.
    /// Returns the associated field handle
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_i64_field<T: Into<NumericOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_i64_field => Adding i64 field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_i64(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_i64_field => Added i64 field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new f64 field.
    /// Returns the associated field handle
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_f64_field<T: Into<NumericOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_f64_field => Adding f64 field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_f64(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_f64_field => Added f64 field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new bool field.
    /// Returns the associated field handle
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_bool_field<T: Into<NumericOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_bool_field => Adding bool field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_bool(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_bool_field => Added bool field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new date field.
    /// Returns the associated field handle
    /// Internally, Tantivy simply stores dates as i64 UTC timestamps,
    /// while the user supplies DateTime values for convenience.
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_date_field<T: Into<DateOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_date_field => Adding date field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_date(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_date_field => Added date field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a ip field.
    /// Returns the associated field handle.
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_ip_addr_field<T: Into<IpAddrOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_ip_addr_field => Adding IP address field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_ip_addr(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_ip_addr_field => Added IP address field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new text field.
    /// Returns the associated field handle
    ///
    /// # Panics
    ///
    /// Panics when field already exists.
    pub fn add_text_field<T: Into<TextOptions>>(
        &mut self,
        field_name_str: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_text_field => Adding text field '{}'",
            field_name_str
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_text(field_name.clone(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_text_field => Added text field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a facet field to the schema.
    pub fn add_facet_field(
        &mut self,
        field_name: &str,
        facet_options: impl Into<FacetOptions>,
    ) -> Field {
        println!(
            "SchemaBuilder::add_facet_field => Adding facet field '{}'",
            field_name
        );
        let field_entry = FieldEntry::new_facet(field_name.to_string(), facet_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_facet_field => Added facet field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a fast bytes field to the schema.
    ///
    /// Bytes field are not searchable and are only used
    /// as fast field, to associate any kind of payload
    /// to a document.
    ///
    /// For instance, learning-to-rank often requires to access
    /// some document features at scoring time.
    /// These can be serializing and stored as a bytes field to
    /// get access rapidly when scoring each document.
    pub fn add_bytes_field<T: Into<BytesOptions>>(
        &mut self,
        field_name: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_bytes_field => Adding bytes field '{}'",
            field_name
        );
        let field_entry = FieldEntry::new_bytes(field_name.to_string(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_bytes_field => Added bytes field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a json object field to the schema.
    pub fn add_json_field<T: Into<JsonObjectOptions>>(
        &mut self,
        field_name: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_json_field => Adding JSON object field '{}'",
            field_name
        );
        let field_entry = FieldEntry::new_json(field_name.to_string(), field_options.into());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_json_field => Added JSON object field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Adds a new nested field to the schema, with the given name and NestedOptions.
    /// Also, if `NestedOptions::store_parent_flag` is true, we create an internal field
    /// named `_is_parent_<field_name>` so we can easily detect parent docs.
    pub fn add_nested_field(
        &mut self,
        field_path: Vec<String>,
        nested_opts: NestedOptions,
    ) -> Field {
        let mut field_name_builder = JsonPathWriter::new();
        for seg in &field_path {
            field_name_builder.push(&seg);
        }
        let field_name = field_name_builder.as_str();
        let field_entry = FieldEntry::new_nested(field_name.to_string(), nested_opts.clone());
        let field = self.add_field(field_entry);

        if nested_opts.store_parent_flag {
            let parent_field_name = format!("_is_parent_{}", field_name);

            // We'll store it as an indexed bool, for the parent doc
            let bool_options = NumericOptions::default().set_indexed();
            let bool_field_entry =
                FieldEntry::new(parent_field_name.clone(), FieldType::Bool(bool_options));
            let parent_field = self.add_field(bool_field_entry);

            // Record the mapping in nested_paths
            self.nested_paths.insert(field_path, field);
        }

        field
    }

    /// Adds a single "nested" field that also stores a JSON object in the same field,
    /// flattening all subkeys into text tokens via `TextOptions`.
    /// The `NestedOptions` control how child docs are expanded (include_in_parent, etc.).
    /// If `NestedOptions::store_parent_flag == true`, we create an internal boolean field
    /// named `"_is_parent_<field_name>"` to mark parent docs.
    pub fn add_nested_json_field(
        &mut self,
        field_path: Vec<String>,
        nested_opts: NestedJsonObjectOptions,
    ) -> Field {
        println!(
            "SchemaBuilder::add_nested_json_field => Adding nested JSON field with path '{:?}' and options {:?}",
            field_path, nested_opts
        );
        let mut field_name_builder = JsonPathWriter::new();
        for seg in &field_path {
            field_name_builder.push(&seg);
        }
        let field_name = field_name_builder.as_str();
        println!(
            "SchemaBuilder::add_nested_json_field => Constructed field name '{}'",
            field_name
        );
        let field_entry = FieldEntry::new_nested_json(field_name.to_string(), nested_opts.clone());
        let field = self.add_field(field_entry);
        println!(
            "SchemaBuilder::add_nested_json_field => Added nested JSON field '{}' with ID {}",
            field_name,
            field.field_id()
        );

        // If `store_parent_flag` is set, also create a parent-flag field
        if nested_opts.nested_opts.store_parent_flag {
            let parent_field_name = format!("_is_parent_{}", field_name);
            println!(
                "SchemaBuilder::add_nested_json_field => Creating parent flag field '{}'",
                parent_field_name
            );

            // We'll store this flag as an indexed bool so we can find parent docs easily.
            let bool_options = NumericOptions::default().set_indexed();
            let bool_field_entry =
                FieldEntry::new(parent_field_name.clone(), FieldType::Bool(bool_options));
            let parent_field = self.add_field(bool_field_entry);
            println!(
                "SchemaBuilder::add_nested_json_field => Added parent flag field '{}' with ID {}",
                parent_field_name,
                parent_field.field_id()
            );

            // Also record the mapping in `nested_paths` so queries can look it up
            self.nested_paths.insert(field_path, field);
            println!(
                "SchemaBuilder::add_nested_json_field => Recorded nested path mapping for field '{}'",
                field_name
            );
        }

        field
    }

    /// Adds a field entry to the schema in build.
    pub fn add_field(&mut self, field_entry: FieldEntry) -> Field {
        println!(
            "SchemaBuilder::add_field => Adding field '{}'",
            field_entry.name()
        );
        let field = Field::from_field_id(self.fields.len() as u32);
        println!(
            "SchemaBuilder::add_field => Assigned field ID: {}",
            field.field_id()
        );

        let field_name = field_entry.name().to_string();
        if let Some(_previous_value) = self.fields_map.insert(field_name.clone(), field) {
            panic!(
                "SchemaBuilder::add_field => Field '{}' already exists in schema",
                field_name
            );
        };
        self.fields.push(field_entry);
        println!(
            "SchemaBuilder::add_field => Successfully added field '{}' with ID {}",
            field_name,
            field.field_id()
        );
        field
    }

    /// Finalize the creation of a `Schema`
    /// This will consume your `SchemaBuilder`
    pub fn build(self) -> Schema {
        println!(
            "SchemaBuilder::build => Finalizing schema with {} fields",
            self.fields.len()
        );
        let built_schema = Schema(Arc::new(InnerSchema {
            fields: self.fields,
            fields_map: self.fields_map,
            nested_paths: self.nested_paths, // <==== store it
        }));
        println!("SchemaBuilder::build => Schema finalized successfully");
        built_schema
    }
}

#[derive(Debug)]
struct InnerSchema {
    fields: Vec<FieldEntry>,
    fields_map: HashMap<String, Field>, // transient
    nested_paths: HashMap<Vec<String>, Field>,
}

impl PartialEq for InnerSchema {
    fn eq(&self, other: &InnerSchema) -> bool {
        self.fields == other.fields
    }
}

impl Eq for InnerSchema {}

/// Tantivy has a very strict schema.
/// You need to specify in advance, whether a field is indexed or not,
/// stored or not, and RAM-based or not.
///
/// This is done by creating a schema object, and
/// setting up the fields one by one.
/// It is for the moment impossible to remove fields.
///
/// # Examples
///
/// ```
/// use tantivy::schema::*;
///
/// let mut schema_builder = Schema::builder();
/// let id_field = schema_builder.add_text_field("id", STRING);
/// let title_field = schema_builder.add_text_field("title", TEXT);
/// let body_field = schema_builder.add_text_field("body", TEXT);
/// let schema = schema_builder.build();
/// ```
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Schema(Arc<InnerSchema>);

// Returns the position (in byte offsets) of the unescaped '.' in the `field_path`.
//
// This function operates directly on bytes (as opposed to codepoint), relying
// on a encoding property of utf-8 for its correctness.
fn locate_splitting_dots(field_path: &str) -> Vec<usize> {
    println!(
        "locate_splitting_dots => Locating splitting dots in field_path '{}'",
        field_path
    );
    let mut splitting_dots_pos = Vec::new();
    let mut escape_state = false;
    for (pos, b) in field_path.bytes().enumerate() {
        if escape_state {
            println!(
                "locate_splitting_dots => Escaped character '{}' at position {}",
                b as char, pos
            );
            escape_state = false;
            continue;
        }
        match b {
            b'\\' => {
                println!(
                    "locate_splitting_dots => Found escape character '\\' at position {}",
                    pos
                );
                escape_state = true;
            }
            b'.' => {
                println!(
                    "locate_splitting_dots => Found splitting dot '.' at position {}",
                    pos
                );
                splitting_dots_pos.push(pos);
            }
            _ => {
                // No action needed
            }
        }
    }
    println!(
        "locate_splitting_dots => Splitting dots found at positions: {:?}",
        splitting_dots_pos
    );
    splitting_dots_pos
}

impl Schema {
    /// Return the `FieldEntry` associated with a `Field`.
    #[inline]
    pub fn get_field_entry(&self, field: Field) -> &FieldEntry {
        &self.0.fields[field.field_id() as usize]
    }

    /// Return the field name for a given `Field`.
    pub fn get_field_name(&self, field: Field) -> &str {
        self.get_field_entry(field).name()
    }

    /// Returns the number of fields in the schema.
    pub fn num_fields(&self) -> usize {
        self.0.fields.len()
    }

    /// Return the list of all the `Field`s.
    pub fn fields(&self) -> impl Iterator<Item = (Field, &FieldEntry)> {
        self.0
            .fields
            .iter()
            .enumerate()
            .map(|(field_id, field_entry)| (Field::from_field_id(field_id as u32), field_entry))
    }

    /// Return the list of all the `Field`s.
    pub fn nested_fields(&self) -> impl Iterator<Item = (&Vec<String>, Field, &FieldEntry)> {
        self.0
            .nested_paths
            .iter()
            .enumerate()
            .map(|(i, (path, field))| (path, *field, &self.0.fields[i]))
    }

    /// Creates a new builder.
    pub fn builder() -> SchemaBuilder {
        println!("Schema::builder => Creating new SchemaBuilder");
        SchemaBuilder::default()
    }

    /// Returns the field option associated with a given name.
    pub fn get_field(&self, field_name: &str) -> crate::Result<Field> {
        println!("Schema::get_field => Looking up field '{}'", field_name);
        let result = self
            .0
            .fields_map
            .get(field_name)
            .cloned()
            .ok_or_else(|| TantivyError::FieldNotFound(field_name.to_string()));

        match &result {
            Ok(field) => println!(
                "Schema::get_field => Found field '{}' with ID {}",
                field_name,
                field.field_id()
            ),
            Err(_) => println!("Schema::get_field => Field '{}' not found", field_name),
        }
        result
    }

    pub fn get_nested_field(&self, path: &Vec<String>) -> Option<(Field, FieldEntry)> {
        println!(
            "Schema::get_nested_field => Looking up nested field with path '{:?}'",
            path
        );
        let path_vec: Vec<String> = path.iter().map(|s| s.clone()).collect();
        self.0
            .nested_paths
            .get(&path_vec)
            .and_then(|&f| Some((f, self.0.fields[f.field_id() as usize].clone())))
    }

    /// Searches for a full_path in the schema, returning the field name and a JSON path.
    ///
    /// This function works by checking if the field exists for the exact given full_path.
    /// If it's not, it splits the full_path at non-escaped '.' chars and tries to match the
    /// prefix with the field names, favoring the longest field names.
    ///
    /// This does not check if field is a JSON field. It is possible for this functions to
    /// return a non-empty JSON path with a non-JSON field.
    pub fn find_field<'a>(&self, full_path: &'a str) -> Option<(Field, &'a str)> {
        println!(
            "Schema::find_field => Searching for full_path '{}'",
            full_path
        );
        if let Some(field) = self.0.fields_map.get(full_path) {
            println!(
                "Schema::find_field => Exact match found for '{}', returning field ID {} with empty JSON path",
                full_path,
                field.field_id()
            );
            return Some((*field, ""));
        }

        let mut splitting_period_pos: Vec<usize> = locate_splitting_dots(full_path);
        println!(
            "Schema::find_field => Initial splitting dots positions: {:?}",
            splitting_period_pos
        );

        while let Some(pos) = splitting_period_pos.pop() {
            println!("Schema::find_field => Processing split at position {}", pos);
            let (prefix, suffix) = full_path.split_at(pos);

            if let Some(field) = self.0.fields_map.get(prefix) {
                println!(
                    "Schema::find_field => Found field '{:?}' matching prefix '{}', JSON path '{}'",
                    field,
                    prefix,
                    &suffix[1..]
                );
                return Some((*field, &suffix[1..]));
            }
            // JSON path may contain a dot, for now we try both variants to find the field.
            let prefix_split = split_json_path(prefix).join(".");
            if let Some(field) = self.0.fields_map.get(&prefix_split) {
                println!(
                    "Schema::find_field => Found field '{:?}' matching split prefix '{}', JSON path '{}'",
                    field,
                    prefix_split,
                    &suffix[1..]
                );
                return Some((*field, &suffix[1..]));
            }
        }
        println!(
            "Schema::find_field => No matching field found for full_path '{}'",
            full_path
        );
        None
    }

    /// Transforms a user-supplied fast field name into a column name.
    ///
    /// This is similar to `.find_field` except it includes some fallback logic to
    /// a default json field. This functionality is used in Quickwit.
    ///
    /// If the remaining path is empty and seems to target JSON field, we return None.
    /// If the remaining path is non-empty and seems to target a non-JSON field, we return None.
    #[doc(hidden)]
    pub fn find_field_with_default<'a>(
        &self,
        full_path: &'a str,

        default_field_opt: Option<Field>,
    ) -> Option<(Field, &'a str)> {
        println!(
            "Schema::find_field_with_default => Searching with full_path '{}' and default_field_opt {:?}",
            full_path, default_field_opt
        );
        let (field, json_path) = self
            .find_field(full_path)
            .or(default_field_opt.map(|field| (field, full_path)))?;
        let field_entry = self.get_field_entry(field);
        let is_json = field_entry.field_type().value_type() == Type::Json;
        println!(
            "Schema::find_field_with_default => Field '{}' is_json: {}",
            field_entry.name(),
            is_json
        );
        if !is_json && !json_path.is_empty() {
            println!(
                "Schema::find_field_with_default => Non-JSON field '{}' has non-empty JSON path '{}', returning None",
                field_entry.name(),
                json_path
            );
            return None;
        }
        println!(
            "Schema::find_field_with_default => Returning field '{}' with JSON path '{}'",
            field_entry.name(),
            json_path
        );
        Some((field, json_path))
    }
}

impl Serialize for Schema {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        println!(
            "Schema::serialize => Serializing schema with {} fields",
            self.0.fields.len()
        );
        let mut seq = serializer.serialize_seq(Some(self.0.fields.len()))?;
        for (i, e) in self.0.fields.iter().enumerate() {
            println!(
                "Schema::serialize => Serializing field {}: {} ({:?})",
                i,
                e.name(),
                e.field_type()
            );
            seq.serialize_element(e)?;
        }
        println!("Schema::serialize => Serialization complete");
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        println!("Schema::deserialize => Starting deserialization");
        struct SchemaVisitor;

        impl<'de> Visitor<'de> for SchemaVisitor {
            type Value = Schema;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of field entries representing a Schema")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                println!("SchemaVisitor::visit_seq => Visiting sequence of fields");
                // Pre-size our vectors/maps based on the size hint if available
                let capacity = seq.size_hint().unwrap_or(0);
                println!(
                    "SchemaVisitor::visit_seq => Sequence size hint: {}",
                    capacity
                );

                // Build a temporary SchemaBuilder
                let mut schema_builder = SchemaBuilder {
                    fields: Vec::with_capacity(capacity),
                    fields_map: HashMap::with_capacity(capacity),
                    nested_paths: HashMap::new(),
                };
                println!("SchemaVisitor::visit_seq => Initialized SchemaBuilder");

                // Read each FieldEntry from the sequence
                while let Some(field_entry) = seq.next_element::<FieldEntry>()? {
                    println!(
                        "SchemaVisitor::visit_seq => Adding FieldEntry '{}' of type {:?}",
                        field_entry.name(),
                        field_entry.field_type()
                    );
                    schema_builder.add_field(field_entry.clone());
                }

                // Finally, build the actual Schema
                println!("SchemaVisitor::visit_seq => Finalizing Schema");
                Ok(schema_builder.build())
            }
        }

        // We delegate to our `SchemaVisitor` to parse a sequence of FieldEntry
        let schema = deserializer.deserialize_seq(SchemaVisitor)?;
        println!("Schema::deserialize => Deserialization complete");
        Ok(schema)
    }
}

mod test_nested_code {
    use super::*;
    use crate::collector::Count;
    use crate::index::Index;
    use crate::query::QueryParser;
    use crate::schema::document::DocParsingError;
    use crate::schema::field_type::ValueParsingError;
    use crate::schema::{
        self, nested_options::NestedOptions, BytesOptions, DateOptions, FacetOptions, Field,
        FieldEntry, FieldType, IndexRecordOption, JsonObjectOptions, NumericOptions, Schema,
        SchemaBuilder, TantivyDocument, TextOptions, Type, STRING, TEXT,
    };
    use crate::tokenizer::TokenizerManager;
    use crate::TantivyError;

    // Test that `FieldEntry::new_nested` creates an entry that is recognized as Nested,
    // is correctly stored or indexed depending on its NestedOptions, etc.
    #[test]
    fn test_field_entry_new_nested() {
        let nested_opts = NestedOptions::new().set_include_in_parent(false);
        let field_entry = FieldEntry::new_nested("my_nested".to_string(), nested_opts.clone());
        assert_eq!(field_entry.name(), "my_nested");
        assert!(matches!(field_entry.field_type(), FieldType::Nested(_)));
        // by default, Nested fields might not be "stored" or "indexed" unless you define logic for that:
        // Here we rely on nested_options' is_stored/is_indexed methods to see what it does.
        assert_eq!(field_entry.is_indexed(), nested_opts.is_indexed());
        assert_eq!(field_entry.is_stored(), nested_opts.is_stored());
        // etc.
    }

    /// This test demonstrates that if you delete a parent document (by some ID term),
    /// the child documents added by a nested field also get removed from the index.
    #[test]
    fn test_delete_parent_also_deletes_nested_children() -> crate::Result<()> {
        // 1. Build schema with:
        //    - A textual `id_field` to identify docs for deletion
        //    - A nested field "cart" with `store_parent_flag = true`.
        let mut schema_builder = SchemaBuilder::new();

        // We'll store an 'id' field to delete the parent by exact term.
        let id_field = schema_builder.add_text_field("id", STRING);

        // Build some basic text field for child items to demonstrate search.
        let item_title_opts = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            )
            .set_stored();

        // Mark the nested field with store_parent_flag = true, so that a hidden
        // "_is_parent_cart" field is created for parent docs.
        let nested_opts = NestedOptions::default().set_store_parent_flag(true);
        schema_builder.add_nested_field(vec!["cart".into()], nested_opts);

        // Also, let's add an example child field explicitly. (In practice you might
        // rely on JSON sub-objects inside the nested field. This test simply shows how
        // child docs are correlated.)
        let item_title_field = schema_builder.add_text_field("cart.title", item_title_opts);

        // 2. Build the schema and create an in-memory index.
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        // 3. Create an index writer.
        let mut index_writer = index.writer(50_000_000)?;

        // 4. Add a parent doc with 2 child docs, for instance:
        //    We'll manually build them here, but you could also use JSON doc parsing
        //    and rely on `Schema::add_nested_field` to create child docs automatically.
        //
        //    For demonstration, let's store them as if we had a single parent doc
        //    that included the array `cart: [ { "title": "item1" }, { "title": "item2" } ]`.
        //    The actual flattening is up to your indexing logic, but conceptually
        //    each child gets its own Doc with field "cart.title".

        // PARENT doc
        let mut parent_doc = TantivyDocument::new();
        parent_doc.add_text(id_field, "parent-123");
        // The parent doc must also have the hidden `_is_parent_cart` field = true, but if
        // you used `add_nested_field(...)` and parse a JSON doc in your real code,
        // Tantivy could set it automatically. For demonstration, we’ll leave it implied.

        // CHILD doc #1
        let mut child_doc1 = TantivyDocument::new();
        // same ID so we can link them
        child_doc1.add_text(id_field, "parent-123");
        child_doc1.add_text(item_title_field, "item1");

        // CHILD doc #2
        let mut child_doc2 = TantivyDocument::new();
        child_doc2.add_text(id_field, "parent-123");
        child_doc2.add_text(item_title_field, "item2");

        // Index them as if they are logically “parent + children”
        index_writer.add_document(parent_doc)?;
        index_writer.add_document(child_doc1)?;
        index_writer.add_document(child_doc2)?;

        // 5. Commit so they become visible to searches
        index_writer.commit()?;

        // 6. Check we can find the children. We’ll do a simple query on item_title_field.
        {
            let searcher = index.reader()?.searcher();
            let query_parser = QueryParser::for_index(&index, vec![item_title_field]);

            // For "item1"
            let query_item1 = query_parser.parse_query("item1")?;
            let count_item1 = searcher.search(&query_item1, &Count)?;
            assert_eq!(count_item1, 1, "Expected exactly 1 child doc for item1");

            // For "item2"
            let query_item2 = query_parser.parse_query("item2")?;
            let count_item2 = searcher.search(&query_item2, &Count)?;
            assert_eq!(count_item2, 1, "Expected exactly 1 child doc for item2");
        }

        // 7. Now delete the parent doc by ID.
        //    This will remove both the parent and all child docs sharing that ID.
        index_writer.delete_term(Term::from_field_text(id_field, "parent-123"));
        index_writer.commit()?;

        // 8. Confirm no docs remain for "item1" or "item2".
        {
            let searcher = index.reader()?.searcher();
            let query_parser = QueryParser::for_index(&index, vec![item_title_field]);

            let query_item1 = query_parser.parse_query("item1")?;
            let count_item1 = searcher.search(&query_item1, &Count)?;
            assert_eq!(
                count_item1, 0,
                "Child doc1 should be gone after parent delete"
            );

            let query_item2 = query_parser.parse_query("item2")?;
            let count_item2 = searcher.search(&query_item2, &Count)?;
            assert_eq!(
                count_item2, 0,
                "Child doc2 should be gone after parent delete"
            );
        }

        Ok(())
    }
}

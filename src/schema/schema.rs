use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::de::{SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
/// ```rust
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
        println!("SchemaBuilder::new called.");
        SchemaBuilder::default()
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
            "SchemaBuilder::add_u64_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_u64(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_i64_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_i64(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_f64_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_f64(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_bool_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_bool(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_date_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_date(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_ip_addr_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_ip_addr(field_name, field_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_text_field called with field_name: '{}'",
            field_name_str,
        );
        let field_name = String::from(field_name_str);
        let field_entry = FieldEntry::new_text(field_name, field_options.into());
        self.add_field(field_entry)
    }

    /// Adds a facet field to the schema.
    pub fn add_facet_field(
        &mut self,
        field_name: &str,
        facet_options: impl Into<FacetOptions>,
    ) -> Field {
        println!(
            "SchemaBuilder::add_facet_field called with field_name: '{}'",
            field_name,
        );
        let field_entry = FieldEntry::new_facet(field_name.to_string(), facet_options.into());
        self.add_field(field_entry)
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
            "SchemaBuilder::add_bytes_field called with field_name: '{}'",
            field_name,
        );
        let field_entry = FieldEntry::new_bytes(field_name.to_string(), field_options.into());
        self.add_field(field_entry)
    }

    /// Adds a json object field to the schema.
    pub fn add_json_field<T: Into<JsonObjectOptions>>(
        &mut self,
        field_name: &str,
        field_options: T,
    ) -> Field {
        println!(
            "SchemaBuilder::add_json_field called with field_name: '{}'",
            field_name,
        );
        let field_entry = FieldEntry::new_json(field_name.to_string(), field_options.into());
        self.add_field(field_entry)
    }

    /// Adds a field entry to the schema in build.
    pub fn add_field(&mut self, field_entry: FieldEntry) -> Field {
        println!(
            "SchemaBuilder::add_field called with field_entry: {:?}",
            field_entry
        );
        let field = Field::from_field_id(self.fields.len() as u32);
        let field_name = field_entry.name().to_string();
        if let Some(_previous_value) = self.fields_map.insert(field_name.clone(), field) {
            panic!(
                "SchemaBuilder::add_field panic: Field '{}' already exists in schema.",
                field_entry.name()
            );
        };
        println!(
            "SchemaBuilder::add_field: Field '{}' added with Field ID {}.",
            field_entry.name(),
            field.field_id()
        );
        self.fields.push(field_entry);
        field
    }

    /// Finalize the creation of a `Schema`
    /// This will consume your `SchemaBuilder`
    pub fn build(self) -> Schema {
        println!("SchemaBuilder::build called. Finalizing schema.");
        let schema = Schema(Arc::new(InnerSchema {
            fields: self.fields,
            fields_map: self.fields_map,
            nested_paths: self.nested_paths,
        }));
        println!("SchemaBuilder::build: Schema finalized.");
        schema
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
        println!(
            "InnerSchema::eq called. Comparing fields: {:?} with {:?}",
            self.fields, other.fields
        );
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
/// ```rust
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
        "locate_splitting_dots called with field_path: '{}'",
        field_path
    );
    let mut splitting_dots_pos = Vec::new();
    let mut escape_state = false;
    for (pos, b) in field_path.bytes().enumerate() {
        if escape_state {
            println!("Skipping escaped character at position {}", pos);
            escape_state = false;
            continue;
        }
        match b {
            b'\\' => {
                println!("Found escape character '\\' at position {}", pos);
                escape_state = true;
            }
            b'.' => {
                println!("Found splitting dot '.' at position {}", pos);
                splitting_dots_pos.push(pos);
            }
            _ => {}
        }
    }
    println!(
        "locate_splitting_dots: Found splitting dots at positions: {:?}",
        splitting_dots_pos
    );
    splitting_dots_pos
}

impl Schema {
    /// Return the `FieldEntry` associated with a `Field`.
    #[inline]
    pub fn get_field_entry(&self, field: Field) -> &FieldEntry {
        println!(
            "Schema::get_field_entry called with Field ID: {}",
            field.field_id()
        );
        &self.0.fields[field.field_id() as usize]
    }

    /// Return the field name for a given `Field`.
    pub fn get_field_name(&self, field: Field) -> &str {
        let field_name = self.get_field_entry(field).name();
        println!(
            "Schema::get_field_name called for Field ID {}: '{}'",
            field.field_id(),
            field_name
        );
        field_name
    }

    /// Returns the number of fields in the schema.
    pub fn num_fields(&self) -> usize {
        let count = self.0.fields.len();
        println!("Schema::num_fields called. Number of fields: {}", count);
        count
    }

    /// Return the list of all the `Field`s.
    pub fn fields(&self) -> impl Iterator<Item = (Field, &FieldEntry)> {
        println!("Schema::fields called. Iterating over fields.");
        self.0
            .fields
            .iter()
            .enumerate()
            .map(|(field_id, field_entry)| (Field::from_field_id(field_id as u32), field_entry))
    }

    pub fn nested_fields(&self) -> impl Iterator<Item = (&Vec<String>, Field, &FieldEntry)> {
        println!("Schema::nested_fields called. Iterating over nested fields.");
        self.0
            .nested_paths
            .iter()
            .enumerate()
            .map(|(i, (path, field))| (path, *field, &self.0.fields[i]))
    }

    /// Creates a new builder.
    pub fn builder() -> SchemaBuilder {
        println!("Schema::builder called. Creating new SchemaBuilder.");
        SchemaBuilder::default()
    }

    /// Returns the field option associated with a given name.
    pub fn get_field(&self, field_name: &str) -> crate::Result<Field> {
        println!("Schema::get_field called with field_name: '{}'", field_name);
        match self.0.fields_map.get(field_name) {
            Some(field) => {
                println!(
                    "Schema::get_field: Found Field '{}' with Field ID {}.",
                    field_name,
                    field.field_id()
                );
                Ok(*field)
            }
            None => {
                println!(
                    "Schema::get_field: Field '{}' not found. Returning error.",
                    field_name
                );
                Err(TantivyError::FieldNotFound(field_name.to_string()))
            }
        }
    }

    pub fn get_nested_field(&self, path: &[String]) -> Option<(Field, FieldEntry)> {
        println!("Schema::get_nested_field called with path: {:?}", path);
        let path_vec: Vec<String> = path.to_vec();
        match self.0.nested_paths.get(&path_vec) {
            Some(&f) => {
                let field_entry = self.0.fields[f.field_id() as usize].clone();
                println!(
                    "Schema::get_nested_field: Found nested field for path {:?}: Field ID {}",
                    path,
                    f.field_id()
                );
                Some((f, field_entry))
            }
            None => {
                println!(
                    "Schema::get_nested_field: No nested field found for path {:?}.",
                    path
                );
                None
            }
        }
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
        println!("Schema::find_field called with full_path: '{}'", full_path);
        if let Some(field) = self.0.fields_map.get(full_path) {
            println!(
                "Schema::find_field: Exact match found for '{}'. Returning Field ID {}.",
                full_path,
                field.field_id()
            );
            return Some((*field, ""));
        }

        let mut splitting_period_pos: Vec<usize> = locate_splitting_dots(full_path);
        while let Some(pos) = splitting_period_pos.pop() {
            let (prefix, suffix) = full_path.split_at(pos);
            println!(
                "Schema::find_field: Trying prefix '{}', suffix '{}'",
                prefix,
                &suffix[1..]
            );

            if let Some(field) = self.0.fields_map.get(prefix) {
                println!(
                    "Schema::find_field: Prefix '{}' matched Field ID {}. Returning with suffix '{}'.",
                    prefix,
                    field.field_id(),
                    &suffix[1..]
                );
                return Some((*field, &suffix[1..]));
            }
            // JSON path may contain a dot, for now we try both variants to find the field.
            let prefix_json = split_json_path(prefix).join(".");
            println!(
                "Schema::find_field: Trying JSON split prefix '{}'.",
                prefix_json
            );
            if let Some(field) = self.0.fields_map.get(&prefix_json) {
                println!(
                    "Schema::find_field: JSON split prefix '{}' matched Field ID {}. Returning with suffix '{}'.",
                    prefix_json,
                    field.field_id(),
                    &suffix[1..]
                );
                return Some((*field, &suffix[1..]));
            }
        }
        println!(
            "Schema::find_field: No matching field found for full_path '{}'.",
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
            "Schema::find_field_with_default called with full_path: '{}', default_field_opt: {:?}",
            full_path, default_field_opt
        );
        let result = self
            .find_field(full_path)
            .or(default_field_opt.map(|field| (field, full_path)));
        match result {
            Some((field, json_path)) => {
                println!(
                    "Schema::find_field_with_default: Initial find_field result: Field ID {}, json_path: '{}'",
                    field.field_id(),
                    json_path
                );
                let field_entry = self.get_field_entry(field);
                let is_json = field_entry.field_type().value_type() == Type::Json;
                println!(
                    "Schema::find_field_with_default: Field '{}' is_json: {}",
                    field_entry.name(),
                    is_json
                );
                if !is_json && !json_path.is_empty() {
                    println!(
                        "Schema::find_field_with_default: Field '{}' is not JSON and json_path '{}' is not empty. Returning None.",
                        field_entry.name(),
                        json_path
                    );
                    return None;
                }
                println!(
                    "Schema::find_field_with_default: Returning Field ID {}, json_path: '{}'",
                    field.field_id(),
                    json_path
                );
                Some((field, json_path))
            }
            None => {
                println!(
                    "Schema::find_field_with_default: No initial find_field result and no default_field_opt. Returning None."
                );
                None
            }
        }
    }
}

impl Serialize for Schema {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        println!("Schema::serialize called.");
        let mut seq = serializer.serialize_seq(Some(self.0.fields.len()))?;
        for e in &self.0.fields {
            println!("Schema::serialize: Serializing field_entry: {:?}", e);
            seq.serialize_element(e)?;
        }
        println!("Schema::serialize: Serialization completed.");
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        println!("Schema::deserialize called.");
        struct SchemaVisitor;

        impl<'de> Visitor<'de> for SchemaVisitor {
            type Value = Schema;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct Schema")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                println!("SchemaVisitor::visit_seq called. Starting deserialization.");
                let mut schema = SchemaBuilder {
                    fields: Vec::with_capacity(seq.size_hint().unwrap_or(0)),
                    fields_map: HashMap::with_capacity(seq.size_hint().unwrap_or(0)),
                    nested_paths: HashMap::new(),
                };

                while let Some(value) = seq.next_element()? {
                    println!("SchemaVisitor::visit_seq: Adding field_entry: {:?}", value);
                    schema.add_field(value);
                }

                println!("SchemaVisitor::visit_seq: Finalizing schema.");
                Ok(schema.build())
            }
        }

        let schema = deserializer.deserialize_seq(SchemaVisitor)?;
        println!("Schema::deserialize completed.");
        Ok(schema)
    }
}

#[cfg(test)]
mod tests {

    use std::collections::BTreeMap;

    use matches::{assert_matches, matches};
    use pretty_assertions::assert_eq;

    use crate::schema::field_type::ValueParsingError;
    use crate::schema::schema::DocParsingError::InvalidJson;
    use crate::schema::*;

    #[test]
    fn test_locate_splitting_dots() {
        println!("Running test_locate_splitting_dots.");
        assert_eq!(&super::locate_splitting_dots("a.b.c"), &[1, 3]);
        assert_eq!(&super::locate_splitting_dots(r"a\.b.c"), &[4]);
        assert_eq!(&super::locate_splitting_dots(r"a\..b.c"), &[3, 5]);
        println!("Completed test_locate_splitting_dots.");
    }

    #[test]
    pub fn is_indexed_test() {
        println!("Running test_is_indexed_test.");
        let mut schema_builder = Schema::builder();
        let field_str = schema_builder.add_text_field("field_str", STRING);
        let schema = schema_builder.build();
        assert!(schema.get_field_entry(field_str).is_indexed());
        println!("Completed test_is_indexed_test.");
    }

    #[test]
    pub fn test_schema_serialization() {
        println!("Running test_schema_serialization.");
        let mut schema_builder = Schema::builder();
        let count_options = NumericOptions::default().set_stored().set_fast();
        let popularity_options = NumericOptions::default().set_stored().set_fast();
        let score_options = NumericOptions::default()
            .set_indexed()
            .set_fieldnorm()
            .set_fast();
        let is_read_options = NumericOptions::default().set_stored().set_fast();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field(
            "author",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_fieldnorms(false),
            ),
        );
        schema_builder.add_u64_field("count", count_options);
        schema_builder.add_i64_field("popularity", popularity_options);
        schema_builder.add_f64_field("score", score_options);
        schema_builder.add_bool_field("is_read", is_read_options);
        let schema = schema_builder.build();
        let schema_json = serde_json::to_string_pretty(&schema).unwrap();
        let expected = r#"[
  {
    "name": "title",
    "type": "text",
    "options": {
      "indexing": {
        "record": "position",
        "fieldnorms": true,
        "tokenizer": "default"
      },
      "stored": false,
      "fast": false
    }
  },
  {
    "name": "author",
    "type": "text",
    "options": {
      "indexing": {
        "record": "basic",
        "fieldnorms": false,
        "tokenizer": "raw"
      },
      "stored": false,
      "fast": false
    }
  },
  {
    "name": "count",
    "type": "u64",
    "options": {
      "indexed": false,
      "fieldnorms": false,
      "fast": true,
      "stored": true
    }
  },
  {
    "name": "popularity",
    "type": "i64",
    "options": {
      "indexed": false,
      "fieldnorms": false,
      "fast": true,
      "stored": true
    }
  },
  {
    "name": "score",
    "type": "f64",
    "options": {
      "indexed": true,
      "fieldnorms": true,
      "fast": true,
      "stored": false
    }
  },
  {
    "name": "is_read",
    "type": "bool",
    "options": {
      "indexed": false,
      "fieldnorms": false,
      "fast": true,
      "stored": true
    }
  }
]"#;
        println!(
            "test_schema_serialization: Serialized schema_json:\n{}",
            schema_json
        );
        assert_eq!(schema_json, expected);

        let schema: Schema = serde_json::from_str(expected).unwrap();

        let mut fields = schema.fields();
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("title", field_entry.name());
            assert_eq!(0, field.field_id());
            println!(
                "test_schema_serialization: Field 0 - name: '{}'",
                field_entry.name()
            );
        }
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("author", field_entry.name());
            assert_eq!(1, field.field_id());
            println!(
                "test_schema_serialization: Field 1 - name: '{}'",
                field_entry.name()
            );
        }
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("count", field_entry.name());
            assert_eq!(2, field.field_id());
            println!(
                "test_schema_serialization: Field 2 - name: '{}'",
                field_entry.name()
            );
        }
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("popularity", field_entry.name());
            assert_eq!(3, field.field_id());
            println!(
                "test_schema_serialization: Field 3 - name: '{}'",
                field_entry.name()
            );
        }
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("score", field_entry.name());
            assert_eq!(4, field.field_id());
            println!(
                "test_schema_serialization: Field 4 - name: '{}'",
                field_entry.name()
            );
        }
        {
            let (field, field_entry) = fields.next().unwrap();
            assert_eq!("is_read", field_entry.name());
            assert_eq!(5, field.field_id());
            println!(
                "test_schema_serialization: Field 5 - name: '{}'",
                field_entry.name()
            );
        }
        assert!(fields.next().is_none());
        println!("Completed test_schema_serialization.");
    }

    #[test]
    pub fn test_document_to_json() {
        println!("Running test_document_to_json.");
        let mut schema_builder = Schema::builder();
        let count_options = NumericOptions::default().set_stored().set_fast();
        let is_read_options = NumericOptions::default().set_stored().set_fast();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field("author", STRING);
        schema_builder.add_u64_field("count", count_options);
        schema_builder.add_ip_addr_field("ip", FAST | STORED);
        schema_builder.add_bool_field("is_read", is_read_options);
        let schema = schema_builder.build();
        let doc_json = r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": 4,
                "ip": "127.0.0.1",
                "is_read": true
        }"#;
        println!("test_document_to_json: Parsing document JSON: {}", doc_json);
        let doc = TantivyDocument::parse_json(&schema, doc_json).unwrap();

        let doc_to_json = doc.to_json(&schema);
        println!(
            "test_document_to_json: Serialized document to JSON: {}",
            doc_to_json
        );
        let doc_serdeser = TantivyDocument::parse_json(&schema, &doc_to_json).unwrap();
        println!("test_document_to_json: Deserialized document JSON and compared.");
        assert_eq!(doc, doc_serdeser);
        println!("Completed test_document_to_json.");
    }

    #[test]
    pub fn test_document_to_ipv4_json() {
        println!("Running test_document_to_ipv4_json.");
        let mut schema_builder = Schema::builder();
        schema_builder.add_ip_addr_field("ip", FAST | STORED);
        let schema = schema_builder.build();

        // IpV4 loopback
        let doc_json = r#"{
                "ip": "127.0.0.1"
        }"#;
        println!(
            "test_document_to_ipv4_json: Parsing IPv4 loopback JSON: {}",
            doc_json
        );
        let doc = TantivyDocument::parse_json(&schema, doc_json).unwrap();
        let value: serde_json::Value = serde_json::from_str(&doc.to_json(&schema)).unwrap();
        assert_eq!(value["ip"][0], "127.0.0.1");
        println!("test_document_to_ipv4_json: Verified IPv4 loopback address.");

        // Special case IpV6 loopback. We don't want to map that to IPv4
        let doc_json = r#"{
                "ip": "::1"
        }"#;
        println!(
            "test_document_to_ipv4_json: Parsing IPv6 loopback JSON: {}",
            doc_json
        );
        let doc = TantivyDocument::parse_json(&schema, doc_json).unwrap();

        let value: serde_json::Value = serde_json::from_str(&doc.to_json(&schema)).unwrap();
        assert_eq!(value["ip"][0], "::1");
        println!("test_document_to_ipv4_json: Verified IPv6 loopback address.");

        // testing ip address of every router in the world
        let doc_json = r#"{
                "ip": "192.168.0.1"
        }"#;
        println!(
            "test_document_to_ipv4_json: Parsing IPv4 address JSON: {}",
            doc_json
        );
        let doc = TantivyDocument::parse_json(&schema, doc_json).unwrap();

        let value: serde_json::Value = serde_json::from_str(&doc.to_json(&schema)).unwrap();
        assert_eq!(value["ip"][0], "192.168.0.1");
        println!("test_document_to_ipv4_json: Verified IPv4 address '192.168.0.1'.");
        println!("Completed test_document_to_ipv4_json.");
    }

    #[test]
    pub fn test_document_from_nameddoc() {
        println!("Running test_document_from_nameddoc.");
        let mut schema_builder = Schema::builder();
        let title = schema_builder.add_text_field("title", TEXT);
        let val = schema_builder.add_i64_field("val", INDEXED);
        let schema = schema_builder.build();
        let mut named_doc_map = BTreeMap::default();
        named_doc_map.insert(
            "title".to_string(),
            vec![OwnedValue::from("title1"), OwnedValue::from("title2")],
        );
        named_doc_map.insert(
            "val".to_string(),
            vec![OwnedValue::from(14u64), OwnedValue::from(-1i64)],
        );
        println!(
            "test_document_from_nameddoc: Creating NamedFieldDocument with map: {:?}",
            named_doc_map
        );
        let doc =
            TantivyDocument::convert_named_doc(&schema, NamedFieldDocument(named_doc_map)).unwrap();
        println!("test_document_from_nameddoc: Converted NamedFieldDocument to TantivyDocument.");

        let title_values: Vec<_> = doc.get_all(title).map(OwnedValue::from).collect();
        let expected_title = vec![
            OwnedValue::from("title1".to_string()),
            OwnedValue::from("title2".to_string()),
        ];
        println!(
            "test_document_from_nameddoc: Comparing title_values: {:?} with expected: {:?}",
            title_values, expected_title
        );
        assert_eq!(title_values, expected_title);

        let val_values: Vec<_> = doc.get_all(val).map(OwnedValue::from).collect();
        let expected_val = vec![OwnedValue::from(14u64), OwnedValue::from(-1i64)];
        println!(
            "test_document_from_nameddoc: Comparing val_values: {:?} with expected: {:?}",
            val_values, expected_val
        );
        assert_eq!(val_values, expected_val);
        println!("Completed test_document_from_nameddoc.");
    }

    #[test]
    pub fn test_document_missing_field_no_error() {
        println!("Running test_document_missing_field_no_error.");
        let schema = Schema::builder().build();
        let mut named_doc_map = BTreeMap::default();
        named_doc_map.insert(
            "title".to_string(),
            vec![OwnedValue::from("title1"), OwnedValue::from("title2")],
        );
        println!(
            "test_document_missing_field_no_error: Creating NamedFieldDocument with map: {:?}",
            named_doc_map
        );
        TantivyDocument::convert_named_doc(&schema, NamedFieldDocument(named_doc_map)).unwrap();
        println!(
            "test_document_missing_field_no_error: Successfully handled document with missing fields."
        );
    }

    #[test]
    pub fn test_parse_document() {
        println!("Running test_parse_document.");
        let mut schema_builder = Schema::builder();
        let count_options = NumericOptions::default().set_stored().set_fast();
        let popularity_options = NumericOptions::default().set_stored().set_fast();
        let score_options = NumericOptions::default().set_indexed().set_fast();
        let title_field = schema_builder.add_text_field("title", TEXT);
        let author_field = schema_builder.add_text_field("author", STRING);
        let count_field = schema_builder.add_u64_field("count", count_options);
        let popularity_field = schema_builder.add_i64_field("popularity", popularity_options);
        let score_field = schema_builder.add_f64_field("score", score_options);
        let schema = schema_builder.build();

        {
            println!("test_parse_document: Parsing empty JSON document.");
            let doc = TantivyDocument::parse_json(&schema, "{}").unwrap();
            assert!(doc.field_values().next().is_none());
            println!("test_parse_document: Successfully parsed empty document.");
        }
        {
            let doc_json = r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": 4,
                "popularity": 10,
                "score": 80.5
            }"#;
            println!(
                "test_parse_document: Parsing valid JSON document: {}",
                doc_json
            );
            let doc = TantivyDocument::parse_json(&schema, doc_json).unwrap();
            println!("test_parse_document: Parsed document: {:?}", doc);

            println!(
                "test_parse_document: Verifying field 'title'. Expected: 'my title', Found: {:?}",
                doc.get_first(title_field).unwrap().as_str()
            );
            assert_eq!(
                doc.get_first(title_field).unwrap().as_str(),
                Some("my title")
            );
            println!(
                "test_parse_document: Verifying field 'author'. Expected: 'fulmicoton', Found: {:?}",
                doc.get_first(author_field).unwrap().as_str()
            );
            assert_eq!(
                doc.get_first(author_field).unwrap().as_str(),
                Some("fulmicoton")
            );
            println!(
                "test_parse_document: Verifying field 'count'. Expected: 4, Found: {:?}",
                doc.get_first(count_field).unwrap().as_u64()
            );
            assert_eq!(doc.get_first(count_field).unwrap().as_u64(), Some(4));
            println!(
                "test_parse_document: Verifying field 'popularity'. Expected: 10, Found: {:?}",
                doc.get_first(popularity_field).unwrap().as_i64()
            );
            assert_eq!(doc.get_first(popularity_field).unwrap().as_i64(), Some(10));
            println!(
                "test_parse_document: Verifying field 'score'. Expected: 80.5, Found: {:?}",
                doc.get_first(score_field).unwrap().as_f64()
            );
            assert_eq!(doc.get_first(score_field).unwrap().as_f64(), Some(80.5f64));
            println!("test_parse_document: Successfully parsed and verified valid JSON document.");
        }
        {
            let res = TantivyDocument::parse_json(
                &schema,
                r#"{
                "thisfieldisnotdefinedintheschema": "my title",
                "title": "my title",
                "author": "fulmicoton",
                "count": 4,
                "popularity": 10,
                "score": 80.5,
                "jambon": "bayonne"
            }"#,
            );
            println!(
                "test_parse_document: Parsing JSON with undefined fields. Result: {:?}",
                res
            );
            assert!(res.is_ok());
            println!("test_parse_document: Successfully handled document with undefined fields.");
        }
        {
            let json_err = TantivyDocument::parse_json(
                &schema,
                r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": "5",
                "popularity": "10",
                "score": "80.5",
                "jambon": "bayonne"
            }"#,
            );
            println!(
                "test_parse_document: Parsing JSON with incorrect field types. Result: {:?}",
                json_err
            );
            assert_matches!(
                json_err,
                Err(DocParsingError::ValueError(
                    _,
                    ValueParsingError::TypeError { .. }
                ))
            );
            println!("test_parse_document: Correctly detected type errors in JSON document.");
        }
        {
            let json_err = TantivyDocument::parse_json(
                &schema,
                r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": -5,
                "popularity": 10,
                "score": 80.5
            }"#,
            );
            println!(
                "test_parse_document: Parsing JSON with overflow error. Result: {:?}",
                json_err
            );
            assert_matches!(
                json_err,
                Err(DocParsingError::ValueError(
                    _,
                    ValueParsingError::OverflowError { .. }
                ))
            );
            println!("test_parse_document: Correctly detected overflow error in JSON document.");
        }
        {
            let json_err = TantivyDocument::parse_json(
                &schema,
                r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": 9223372036854775808,
                "popularity": 10,
                "score": 80.5
            }"#,
            );
            println!(
                "test_parse_document: Parsing JSON with large count value. Result: {:?}",
                json_err
            );
            assert!(!matches!(
                json_err,
                Err(DocParsingError::ValueError(
                    _,
                    ValueParsingError::OverflowError { .. }
                ))
            ));
            println!("test_parse_document: Handled large count value without overflow error.");
        }
        {
            let json_err = TantivyDocument::parse_json(
                &schema,
                r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": 50,
                "popularity": 9223372036854775808,
                "score": 80.5
            }"#,
            );
            println!(
                "test_parse_document: Parsing JSON with large popularity value. Result: {:?}",
                json_err
            );
            assert_matches!(
                json_err,
                Err(DocParsingError::ValueError(
                    _,
                    ValueParsingError::OverflowError { .. }
                ))
            );
            println!(
                "test_parse_document: Correctly detected overflow error for popularity field."
            );
        }
        {
            // Short JSON, under the 20 char take.
            let json_err = TantivyDocument::parse_json(&schema, r#"{"count": 50,}"#);
            println!(
                "test_parse_document: Parsing malformed JSON with trailing comma. Result: {:?}",
                json_err
            );
            assert_matches!(json_err, Err(InvalidJson(_)));
            println!("test_parse_document: Correctly detected invalid JSON with trailing comma.");
        }
        {
            let json_err = TantivyDocument::parse_json(
                &schema,
                r#"{
                "title": "my title",
                "author": "fulmicoton",
                "count": 50,
            }"#,
            );
            println!(
                "test_parse_document: Parsing malformed JSON with trailing comma. Result: {:?}",
                json_err
            );
            assert_matches!(json_err, Err(InvalidJson(_)));
            println!("test_parse_document: Correctly detected invalid JSON with trailing comma.");
        }
        println!("Completed test_parse_document.");
    }

    #[test]
    pub fn test_schema_add_field() {
        println!("Running test_schema_add_field.");
        let mut schema_builder = SchemaBuilder::default();
        let id_options = TextOptions::default().set_stored().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::Basic),
        );
        let timestamp_options = DateOptions::default()
            .set_stored()
            .set_indexed()
            .set_fieldnorm()
            .set_fast();
        println!(
            "test_schema_add_field: Adding '_id' field with options: {:?}",
            id_options
        );
        schema_builder.add_text_field("_id", id_options);
        println!(
            "test_schema_add_field: Adding '_timestamp' field with options: {:?}",
            timestamp_options
        );
        schema_builder.add_date_field("_timestamp", timestamp_options);

        let schema_content = r#"[
  {
    "name": "text",
    "type": "text",
    "options": {
      "indexing": {
        "record": "position",
        "fieldnorms": true,
        "tokenizer": "default"
      },
      "stored": false,
      "fast": false
    }
  },
  {
    "name": "popularity",
    "type": "i64",
    "options": {
      "indexed": false,
      "fieldnorms": false,
      "fast": true,
      "stored": true
    }
  }
]"#;
        println!(
            "test_schema_add_field: Deserializing schema_content:\n{}",
            schema_content
        );
        let tmp_schema: Schema =
            serde_json::from_str(schema_content).expect("error while reading json");
        for (_field, field_entry) in tmp_schema.fields() {
            println!(
                "test_schema_add_field: Adding field_entry from tmp_schema: {:?}",
                field_entry
            );
            schema_builder.add_field(field_entry.clone());
        }

        let schema = schema_builder.build();
        println!(
            "test_schema_add_field: Serialized schema to JSON: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
        let schema_json = serde_json::to_string_pretty(&schema).unwrap();
        let expected = r#"[
  {
    "name": "_id",
    "type": "text",
    "options": {
      "indexing": {
        "record": "basic",
        "fieldnorms": true,
        "tokenizer": "raw"
      },
      "stored": true,
      "fast": false
    }
  },
  {
    "name": "_timestamp",
    "type": "date",
    "options": {
      "indexed": true,
      "fieldnorms": true,
      "fast": true,
      "stored": true,
      "precision": "seconds"
    }
  },
  {
    "name": "text",
    "type": "text",
    "options": {
      "indexing": {
        "record": "position",
        "fieldnorms": true,
        "tokenizer": "default"
      },
      "stored": false,
      "fast": false
    }
  },
  {
    "name": "popularity",
    "type": "i64",
    "options": {
      "indexed": false,
      "fieldnorms": false,
      "fast": true,
      "stored": true
    }
  }
]"#;
        println!(
            "test_schema_add_field: Comparing serialized schema_json with expected.\nExpected:\n{}",
            expected
        );
        assert_eq!(schema_json, expected);
        println!("Completed test_schema_add_field.");
    }

    #[test]
    fn test_find_field() {
        println!("Running test_find_field.");
        let mut schema_builder = Schema::builder();
        println!("test_find_field: Adding 'foo' JSON field.");
        schema_builder.add_json_field("foo", STRING);

        println!("test_find_field: Adding 'bar' text field.");
        schema_builder.add_text_field("bar", STRING);
        println!("test_find_field: Adding 'foo.bar' text field.");
        schema_builder.add_text_field("foo.bar", STRING);
        println!("test_find_field: Adding 'foo.bar.baz' text field.");
        schema_builder.add_text_field("foo.bar.baz", STRING);
        println!("test_find_field: Adding 'bar.a.b.c' text field.");
        schema_builder.add_text_field("bar.a.b.c", STRING);
        let schema = schema_builder.build();

        println!("test_find_field: Performing find_field queries.");
        assert_eq!(
            schema.find_field("foo.bar"),
            Some((schema.get_field("foo.bar").unwrap(), ""))
        );
        println!("test_find_field: find_field('foo.bar') passed.");
        assert_eq!(
            schema.find_field("foo.bar.bar"),
            Some((schema.get_field("foo.bar").unwrap(), "bar"))
        );
        println!("test_find_field: find_field('foo.bar.bar') passed.");
        assert_eq!(
            schema.find_field("foo.bar.baz"),
            Some((schema.get_field("foo.bar.baz").unwrap(), ""))
        );
        println!("test_find_field: find_field('foo.bar.baz') passed.");
        assert_eq!(
            schema.find_field("foo.toto"),
            Some((schema.get_field("foo").unwrap(), "toto"))
        );
        println!("test_find_field: find_field('foo.toto') passed.");
        assert_eq!(
            schema.find_field("foo.bar"),
            Some((schema.get_field("foo.bar").unwrap(), ""))
        );
        println!("test_find_field: find_field('foo.bar') (second time) passed.");
        assert_eq!(
            schema.find_field("bar.toto.titi"),
            Some((schema.get_field("bar").unwrap(), "toto.titi"))
        );
        println!("test_find_field: find_field('bar.toto.titi') passed.");

        assert_eq!(schema.find_field("hello"), None);
        println!("test_find_field: find_field('hello') correctly returned None.");
        assert_eq!(schema.find_field(""), None);
        println!("test_find_field: find_field('') correctly returned None.");
        assert_eq!(schema.find_field("thiswouldbeareallylongfieldname"), None);
        println!("test_find_field: find_field('thiswouldbeareallylongfieldname') correctly returned None.");
        assert_eq!(schema.find_field("baz.bar.foo"), None);
        println!("test_find_field: find_field('baz.bar.foo') correctly returned None.");
        println!("Completed test_find_field.");
    }

    #[test]
    fn test_find_field_with_default() {
        println!("Running test_find_field_with_default.");
        let mut schema_builder = Schema::builder();
        println!("test_find_field_with_default: Adding '_default' JSON field.");
        schema_builder.add_json_field("_default", JsonObjectOptions::default());
        let default = Field::from_field_id(0);
        println!("test_find_field_with_default: Adding 'foo' JSON field.");
        schema_builder.add_json_field("foo", STRING);
        let foo = Field::from_field_id(1);
        println!("test_find_field_with_default: Adding 'foo.bar' text field.");
        schema_builder.add_text_field("foo.bar", STRING);
        let foo_bar = Field::from_field_id(2);
        println!("test_find_field_with_default: Adding 'bar' text field.");
        schema_builder.add_text_field("bar", STRING);
        let bar = Field::from_field_id(3);
        println!("test_find_field_with_default: Adding 'baz' JSON field.");
        schema_builder.add_json_field("baz", JsonObjectOptions::default());
        let baz = Field::from_field_id(4);
        let schema = schema_builder.build();

        println!("test_find_field_with_default: Performing find_field_with_default queries.");
        assert_eq!(schema.find_field_with_default("foo", None), Some((foo, "")));
        println!("test_find_field_with_default: find_field_with_default('foo', None) passed.");
        assert_eq!(
            schema.find_field_with_default("foo.bar", None),
            Some((foo_bar, ""))
        );
        println!("test_find_field_with_default: find_field_with_default('foo.bar', None) passed.");
        assert_eq!(schema.find_field_with_default("bar", None), Some((bar, "")));
        println!("test_find_field_with_default: find_field_with_default('bar', None) passed.");
        assert_eq!(schema.find_field_with_default("bar.baz", None), None);
        println!("test_find_field_with_default: find_field_with_default('bar.baz', None) correctly returned None.");
        assert_eq!(schema.find_field_with_default("baz", None), Some((baz, "")));
        println!("test_find_field_with_default: find_field_with_default('baz', None) passed.");
        assert_eq!(
            schema.find_field_with_default("baz.foobar", None),
            Some((baz, "foobar"))
        );
        println!(
            "test_find_field_with_default: find_field_with_default('baz.foobar', None) passed."
        );
        assert_eq!(schema.find_field_with_default("foobar", None), None);
        println!("test_find_field_with_default: find_field_with_default('foobar', None) correctly returned None.");

        assert_eq!(
            schema.find_field_with_default("foo", Some(default)),
            Some((foo, ""))
        );
        println!(
            "test_find_field_with_default: find_field_with_default('foo', Some(default)) passed."
        );
        assert_eq!(
            schema.find_field_with_default("foo.bar", Some(default)),
            Some((foo_bar, ""))
        );
        println!("test_find_field_with_default: find_field_with_default('foo.bar', Some(default)) passed.");
        assert_eq!(
            schema.find_field_with_default("bar", Some(default)),
            Some((bar, ""))
        );
        println!(
            "test_find_field_with_default: find_field_with_default('bar', Some(default)) passed."
        );
        // still None, we are under an existing field
        assert_eq!(
            schema.find_field_with_default("bar.baz", Some(default)),
            None
        );
        println!("test_find_field_with_default: find_field_with_default('bar.baz', Some(default)) correctly returned None.");
        assert_eq!(
            schema.find_field_with_default("baz", Some(default)),
            Some((baz, ""))
        );
        println!(
            "test_find_field_with_default: find_field_with_default('baz', Some(default)) passed."
        );
        assert_eq!(
            schema.find_field_with_default("baz.foobar", Some(default)),
            Some((baz, "foobar"))
        );
        println!("test_find_field_with_default: find_field_with_default('baz.foobar', Some(default)) passed.");
        assert_eq!(
            schema.find_field_with_default("foobar", Some(default)),
            Some((default, "foobar"))
        );
        println!("test_find_field_with_default: find_field_with_default('foobar', Some(default)) passed.");
        println!("Completed test_find_field_with_default.");
    }
}

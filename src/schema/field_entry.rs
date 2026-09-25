use serde::{Deserialize, Serialize};

use super::ip_options::IpAddrOptions;
use crate::schema::bytes_options::BytesOptions;
use crate::schema::custom_options::CustomOptions;
use crate::schema::{
    is_valid_field_name, DateOptions, FacetOptions, FieldType, JsonObjectOptions, NumericOptions,
    TextOptions, VectorOptions,
};

/// A `FieldEntry` represents a field and its configuration.
/// `Schema` are a collection of `FieldEntry`
///
/// It consists of
/// - a field name
/// - a field type, itself wrapping up options describing how the field should be indexed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldEntry {
    name: String,
    #[serde(flatten)]
    field_type: FieldType,
}

impl FieldEntry {
    /// Creates a new field entry given a name and a field type
    pub fn new(field_name: String, field_type: FieldType) -> FieldEntry {
        assert!(is_valid_field_name(&field_name));
        FieldEntry {
            name: field_name,
            field_type,
        }
    }

    /// Creates a new text field entry.
    pub fn new_text(field_name: String, text_options: TextOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Str(text_options))
    }

    /// Creates a new u64 field entry.
    pub fn new_u64(field_name: String, int_options: NumericOptions) -> FieldEntry {
        Self::new(field_name, FieldType::U64(int_options))
    }

    /// Creates a new i64 field entry.
    pub fn new_i64(field_name: String, int_options: NumericOptions) -> FieldEntry {
        Self::new(field_name, FieldType::I64(int_options))
    }

    /// Creates a new f64 field entry.
    pub fn new_f64(field_name: String, f64_options: NumericOptions) -> FieldEntry {
        Self::new(field_name, FieldType::F64(f64_options))
    }

    /// Creates a new bool field entry.
    pub fn new_bool(field_name: String, bool_options: NumericOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Bool(bool_options))
    }

    /// Creates a new date field entry.
    pub fn new_date(field_name: String, date_options: DateOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Date(date_options))
    }

    /// Creates a new ip address field entry.
    pub fn new_ip_addr(field_name: String, ip_options: IpAddrOptions) -> FieldEntry {
        Self::new(field_name, FieldType::IpAddr(ip_options))
    }

    /// Creates a field entry for a facet.
    pub fn new_facet(field_name: String, facet_options: FacetOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Facet(facet_options))
    }

    /// Creates a field entry for a bytes field
    pub fn new_bytes(field_name: String, bytes_options: BytesOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Bytes(bytes_options))
    }

    /// Creates a field entry for a json field
    pub fn new_json(field_name: String, json_object_options: JsonObjectOptions) -> FieldEntry {
        Self::new(field_name, FieldType::JsonObject(json_object_options))
    }

    /// Creates a field entry for a plugin-defined custom field.
    pub fn new_custom(field_name: String, custom_options: CustomOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Custom(custom_options))
    }

    /// Creates a field entry for a brute-force vector field.
    pub fn new_vector(field_name: String, vector_options: VectorOptions) -> FieldEntry {
        Self::new(field_name, FieldType::Vector(vector_options))
    }

    /// Returns the name of the field
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the field type
    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    /// Returns true if the field is indexed.
    ///
    /// An indexed field is searchable.
    pub fn is_indexed(&self) -> bool {
        self.field_type.is_indexed()
    }

    /// Returns true if the field is normed
    pub fn has_fieldnorms(&self) -> bool {
        self.field_type.has_fieldnorms()
    }

    /// Returns true if posting-local norms are enabled for this field.
    pub fn has_posting_norms(&self) -> bool {
        self.field_type.has_posting_norms()
    }

    /// Returns true if the field is a fast field
    pub fn is_fast(&self) -> bool {
        self.field_type.is_fast()
    }

    /// Returns true if the field has the expand dots option set (for json fields)
    pub fn is_expand_dots_enabled(&self) -> bool {
        match self.field_type {
            FieldType::JsonObject(ref options) => options.is_expand_dots_enabled(),
            _ => false,
        }
    }

    /// Returns true if the field is stored
    #[inline]
    pub fn is_stored(&self) -> bool {
        match self.field_type {
            FieldType::U64(ref options)
            | FieldType::I64(ref options)
            | FieldType::F64(ref options)
            | FieldType::Bool(ref options) => options.is_stored(),
            FieldType::Date(ref options) => options.is_stored(),
            FieldType::Str(ref options) => options.is_stored(),
            FieldType::Facet(ref options) => options.is_stored(),
            FieldType::Bytes(ref options) => options.is_stored(),
            FieldType::JsonObject(ref options) => options.is_stored(),
            FieldType::IpAddr(ref options) => options.is_stored(),
            FieldType::Custom(_) => false,
            FieldType::Vector(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::schema::{Schema, TextFieldIndexing, TEXT};
    use crate::Index;

    #[test]
    #[should_panic]
    fn test_invalid_field_name_should_panic() {
        FieldEntry::new_text("-hello".to_string(), TEXT);
    }

    #[test]
    fn test_json_serialization() {
        let field_value = FieldEntry::new_text(String::from("title"), TEXT);

        let expected = r#"{
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
}"#;
        let field_value_json = serde_json::to_string_pretty(&field_value).unwrap();

        assert_eq!(expected, &field_value_json);

        let field_value: FieldEntry = serde_json::from_str(expected).unwrap();

        assert_eq!("title", field_value.name);

        match field_value.field_type {
            FieldType::Str(_) => {}
            _ => panic!("expected FieldType::Str"),
        }
    }

    #[test]
    fn test_json_deserialization() {
        let json_str = r#"{
  "name": "title",
  "options": {
    "indexing": {
      "record": "position",
      "fieldnorms": true,
      "tokenizer": "default"
    },
    "stored": false
  },
  "type": "text"
}"#;
        let field_entry: FieldEntry = serde_json::from_str(json_str).unwrap();
        match field_entry.field_type {
            FieldType::Str(_) => {}
            _ => panic!("expected FieldType::Str"),
        }
    }

    #[test]
    fn test_posting_norm_options_roundtrip_and_defaults() {
        use crate::schema::{
            BytesOptions, DateOptions, IpAddrOptions, NumericOptions, FAST, INDEXED,
        };

        let field_type = FieldType::Str(
            TEXT.set_indexing_options(TextFieldIndexing::default().set_posting_norms(true)) | FAST,
        );
        assert!(field_type.has_posting_norms());
        let mut serialized = serde_json::to_value(&field_type).unwrap();
        assert_eq!(
            serde_json::from_value::<FieldType>(serialized.clone()).unwrap(),
            field_type
        );
        let options = &mut serialized["options"]["indexing"];
        assert_eq!(options["posting_norms"], true);
        options.as_object_mut().unwrap().remove("posting_norms");
        let legacy: FieldType = serde_json::from_value(serialized.clone()).unwrap();
        assert!(!legacy.has_posting_norms());
        assert_eq!(serde_json::to_value(&legacy).unwrap(), serialized);
        assert!(!TextFieldIndexing::default()
            .set_posting_norms(true)
            .set_fieldnorms(false)
            .posting_norms());

        for field_type in [
            FieldType::U64(NumericOptions::from(INDEXED)),
            FieldType::I64(NumericOptions::from(INDEXED)),
            FieldType::F64(NumericOptions::from(INDEXED)),
            FieldType::Bool(NumericOptions::from(INDEXED)),
            FieldType::Date(DateOptions::from(INDEXED)),
            FieldType::Bytes(BytesOptions::from(INDEXED)),
            FieldType::IpAddr(IpAddrOptions::from(INDEXED)),
        ] {
            assert!(field_type.has_fieldnorms());
            assert!(!field_type.has_posting_norms());
            let serialized = serde_json::to_value(&field_type).unwrap();
            assert!(serialized["options"].get("posting_norms").is_none());
        }
    }

    #[test]
    fn test_missing_fieldnorms() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let no_field_norm = TextOptions::default()
            .set_indexing_options(TextFieldIndexing::default().set_fieldnorms(false));
        let text = schema_builder.add_text_field("text", no_field_norm);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;
        index_writer.add_document(doc!(text=>"abc"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let err = searcher.segment_reader(0u32).get_fieldnorms_reader(text);
        assert!(matches!(err, Err(crate::TantivyError::SchemaError(_))));
        Ok(())
    }
}

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Read, Write};
use std::net::Ipv6Addr;

use columnar::MonotonicallyMappableToU128;
use common::{read_u32_vint_no_advance, serialize_vint_u32, BinarySerializable, DateTime, VInt};
use serde_json::Map;
pub use CompactDoc as TantivyDocument;

use super::{JsonObjectOptions, ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::document::{
    DeserializeError, Document, DocumentDeserialize, DocumentDeserializer,
};
use crate::schema::field_type::ValueParsingError;
use crate::schema::{Facet, Field, NamedFieldDocument, OwnedValue, Schema};
use crate::tokenizer::PreTokenizedString;

#[repr(packed)]
#[derive(Debug, Clone)]
/// A field value pair in the compact tantivy document
struct FieldValueAddr {
    pub field: u16,
    pub value_addr: ValueAddr,
}

#[derive(Debug, Clone)]
/// The default document in tantivy. It encodes data in a compact form.
pub struct CompactDoc {
    /// `node_data` is a vec of bytes, where each value is serialized into bytes and stored. It
    /// includes all the data of the document and also metadata like where the nodes are located
    /// in an object or array.
    pub node_data: Vec<u8>,
    /// The root (Field, Value) pairs
    field_values: Vec<FieldValueAddr>,
    pub is_parent: bool,
}

impl Default for CompactDoc {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactDoc {
    /// Creates a new, empty document object
    /// The reserved capacity is for the total serialized data
    pub fn with_capacity(bytes: usize) -> CompactDoc {
        CompactDoc {
            node_data: Vec::with_capacity(bytes),
            field_values: Vec::with_capacity(4),
            is_parent: false,
        }
    }

    /// Creates a new, empty document object
    pub fn new() -> CompactDoc {
        CompactDoc::with_capacity(1024)
    }

    /// Skrinks the capacity of the document to fit the data
    pub fn shrink_to_fit(&mut self) {
        self.node_data.shrink_to_fit();
        self.field_values.shrink_to_fit();
    }

    /// Returns the length of the document.
    pub fn len(&self) -> usize {
        self.field_values.len()
    }

    /// Adding a facet to the document.
    pub fn add_facet<F>(&mut self, field: Field, path: F)
    where
        Facet: From<F>,
    {
        let facet = Facet::from(path);
        self.add_leaf_field_value(field, ReferenceValueLeaf::Facet(facet.encoded_str()));
    }

    /// Add a text field.
    pub fn add_text<S: AsRef<str>>(&mut self, field: Field, text: S) {
        self.add_leaf_field_value(field, ReferenceValueLeaf::Str(text.as_ref()));
    }

    /// Add a pre-tokenized text field.
    pub fn add_pre_tokenized_text(&mut self, field: Field, pre_tokenized_text: PreTokenizedString) {
        self.add_leaf_field_value(field, pre_tokenized_text);
    }

    /// Add a u64 field
    pub fn add_u64(&mut self, field: Field, value: u64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a IP address field. Internally only Ipv6Addr is used.
    pub fn add_ip_addr(&mut self, field: Field, value: Ipv6Addr) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a i64 field
    pub fn add_i64(&mut self, field: Field, value: i64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a f64 field
    pub fn add_f64(&mut self, field: Field, value: f64) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a bool field
    pub fn add_bool(&mut self, field: Field, value: bool) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a date field with unspecified time zone offset
    pub fn add_date(&mut self, field: Field, value: DateTime) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a bytes field
    pub fn add_bytes(&mut self, field: Field, value: &[u8]) {
        self.add_leaf_field_value(field, value);
    }

    /// Add a dynamic object field
    pub fn add_object(&mut self, field: Field, object: BTreeMap<String, OwnedValue>) {
        self.add_field_value(field, &OwnedValue::from(object));
    }

    /// Add a dynamic object field
    pub fn add_nested_object(
        &mut self,
        schema: &Schema,
        field: Field,
        value: serde_json::Value,
        opts: &JsonObjectOptions,
    ) -> crate::Result<Vec<Self>> {
        let field_name = schema.get_field_name(field);
        explode::explode_tantivy_docs(self, schema, field_name, value, opts)
    }

    /// Add a (field, value) to the document.
    ///
    /// `OwnedValue` implements Value, which should be easiest to use, but is not the most
    /// performant.
    pub fn add_field_value<'a, V: Value<'a>>(&mut self, field: Field, value: V) {
        let field_value = FieldValueAddr {
            field: field
                .field_id()
                .try_into()
                .expect("support only up to u16::MAX field ids"),
            value_addr: self.add_value(value),
        };
        self.field_values.push(field_value);
    }

    /// Add a (field, leaf value) to the document.

    /// Leaf values don't have nested values.
    pub fn add_leaf_field_value<'a, T: Into<ReferenceValueLeaf<'a>>>(
        &mut self,
        field: Field,
        typed_val: T,
    ) {
        let value = typed_val.into();
        let field_value = FieldValueAddr {
            field: field
                .field_id()
                .try_into()
                .expect("support only up to u16::MAX field ids"),
            value_addr: self.add_value_leaf(value),
        };
        self.field_values.push(field_value);
    }

    /// field_values accessor
    pub fn field_values(&self) -> impl Iterator<Item = (Field, CompactDocValue<'_>)> {
        self.field_values.iter().map(|field_val| {
            let field = Field::from_field_id(field_val.field as u32);
            let val = self.get_compact_doc_value(field_val.value_addr);
            (field, val)
        })
    }

    /// Returns all of the `ReferenceValue`s associated the given field
    pub fn get_all(&self, field: Field) -> impl Iterator<Item = CompactDocValue<'_>> + '_ {
        self.field_values
            .iter()
            .filter(move |field_value| Field::from_field_id(field_value.field as u32) == field)
            .map(|val| self.get_compact_doc_value(val.value_addr))
    }

    /// Returns the first `ReferenceValue` associated the given field
    pub fn get_first(&self, field: Field) -> Option<CompactDocValue<'_>> {
        self.get_all(field).next()
    }

    /// Create document from a named doc.
    pub fn convert_named_doc(
        schema: &Schema,
        named_doc: NamedFieldDocument,
    ) -> Result<Self, DocParsingError> {
        let mut document = Self::new();
        for (field_name, values) in named_doc.0 {
            if let Ok(field) = schema.get_field(&field_name) {
                for value in values {
                    document.add_field_value(field, &value);
                }
            }
        }
        Ok(document)
    }

    /// Build a document object from a json-object.
    pub fn parse_json(schema: &Schema, doc_json: &str) -> Result<Self, DocParsingError> {
        let json_obj: Map<String, serde_json::Value> =
            serde_json::from_str(doc_json).map_err(|_| DocParsingError::invalid_json(doc_json))?;
        Self::from_json_object(schema, json_obj)
    }

    /// Build a document object from a json-object.
    pub fn from_json_object(
        schema: &Schema,
        json_obj: Map<String, serde_json::Value>,
    ) -> Result<Self, DocParsingError> {
        let mut doc = Self::default();
        for (field_name, json_value) in json_obj {
            if let Ok(field) = schema.get_field(&field_name) {
                let field_entry = schema.get_field_entry(field);
                let field_type = field_entry.field_type();
                match json_value {
                    serde_json::Value::Array(json_items) => {
                        for json_item in json_items {
                            let value = field_type
                                .value_from_json(json_item)
                                .map_err(|e| DocParsingError::ValueError(field_name.clone(), e))?;
                            doc.add_field_value(field, &value);
                        }
                    }
                    _ => {
                        let value = field_type
                            .value_from_json(json_value)
                            .map_err(|e| DocParsingError::ValueError(field_name.clone(), e))?;
                        doc.add_field_value(field, &value);
                    }
                }
            }
        }
        Ok(doc)
    }

    fn add_value_leaf(&mut self, leaf: ReferenceValueLeaf) -> ValueAddr {
        let type_id = ValueType::from(&leaf);
        // Write into `node_data` and return u32 position as its address
        // Null and bool are inlined into the address
        let val_addr = match leaf {
            ReferenceValueLeaf::Null => 0,
            ReferenceValueLeaf::Str(bytes) => {
                write_bytes_into(&mut self.node_data, bytes.as_bytes())
            }
            ReferenceValueLeaf::Facet(bytes) => {
                write_bytes_into(&mut self.node_data, bytes.as_bytes())
            }
            ReferenceValueLeaf::Bytes(bytes) => write_bytes_into(&mut self.node_data, bytes),
            ReferenceValueLeaf::U64(num) => write_into(&mut self.node_data, num),
            ReferenceValueLeaf::I64(num) => write_into(&mut self.node_data, num),
            ReferenceValueLeaf::F64(num) => write_into(&mut self.node_data, num),
            ReferenceValueLeaf::Bool(b) => b as u32,
            ReferenceValueLeaf::Date(date) => {
                write_into(&mut self.node_data, date.into_timestamp_nanos())
            }
            ReferenceValueLeaf::IpAddr(num) => write_into(&mut self.node_data, num.to_u128()),
            ReferenceValueLeaf::PreTokStr(pre_tok) => write_into(&mut self.node_data, *pre_tok),
        };
        ValueAddr { type_id, val_addr }
    }
    /// Adds a value and returns in address into the
    fn add_value<'a, V: Value<'a>>(&mut self, value: V) -> ValueAddr {
        let value = value.as_value();
        let type_id = ValueType::from(&value);
        match value {
            ReferenceValue::Leaf(leaf) => self.add_value_leaf(leaf),
            ReferenceValue::Array(elements) => {
                // addresses of the elements in node_data
                // Reusing a vec would be nicer, but it's not easy because of the recursion
                // A global vec would work if every writer get it's discriminator
                let mut addresses = Vec::new();
                for elem in elements {
                    let value_addr = self.add_value(elem);
                    write_into(&mut addresses, value_addr);
                }
                ValueAddr {
                    type_id,
                    val_addr: write_bytes_into(&mut self.node_data, &addresses),
                }
            }
            ReferenceValue::Object(entries) => {
                // addresses of the elements in node_data
                let mut addresses = Vec::new();
                for (key, value) in entries {
                    let key_addr = self.add_value_leaf(ReferenceValueLeaf::Str(key));
                    let value_addr = self.add_value(value);
                    write_into(&mut addresses, key_addr);
                    write_into(&mut addresses, value_addr);
                }
                ValueAddr {
                    type_id,
                    val_addr: write_bytes_into(&mut self.node_data, &addresses),
                }
            }
        }
    }

    /// Get CompactDocValue for address
    fn get_compact_doc_value(&self, value_addr: ValueAddr) -> CompactDocValue<'_> {
        CompactDocValue {
            container: self,
            value_addr,
        }
    }

    /// get &[u8] reference from node_data
    fn extract_bytes(&self, addr: Addr) -> &[u8] {
        binary_deserialize_bytes(self.get_slice(addr))
    }

    /// get &str reference from node_data
    fn extract_str(&self, addr: Addr) -> &str {
        let data = self.extract_bytes(addr);
        // Utf-8 checks would have a noticeable performance overhead here
        unsafe { std::str::from_utf8_unchecked(data) }
    }

    /// deserialized owned value from node_data
    fn read_from<T: BinarySerializable>(&self, addr: Addr) -> io::Result<T> {
        let data_slice = &self.node_data[addr as usize..];
        let mut cursor = std::io::Cursor::new(data_slice);
        T::deserialize(&mut cursor)
    }

    /// get slice from address. The returned slice is open ended
    fn get_slice(&self, addr: Addr) -> &[u8] {
        &self.node_data[addr as usize..]
    }

    pub fn set_is_parent(&mut self, field: Field, is_parent: bool) {
        // or store a fieldvalue for the boolean
        self.is_parent = is_parent;
        if is_parent {
            self.add_field_value(field, &OwnedValue::from(true));
        }
    }

    pub fn debug_str(&self, root_field: Field, parent_flag_field: Field) -> String {
        // Look up the JSON string in `root_field`
        let mut json_val = "<none>".to_string();
        let mut parent_str = String::new();
        for (fld, val) in self.field_values() {
            if fld == root_field {
                json_val = format!("{val:?}");
            }
            if fld == parent_flag_field && val.as_bool() == Some(true) {
                parent_str = ", is_parent=true".to_string();
            }
        }
        format!("{{json={json_val}{parent_str}}}")
    }
}

/// BinarySerializable alternative to read references
fn binary_deserialize_bytes(data: &[u8]) -> &[u8] {
    let (len, bytes_read) = read_u32_vint_no_advance(data);
    &data[bytes_read..bytes_read + len as usize]
}

/// Write bytes and return the position of the written data.
///
/// BinarySerializable alternative to write references
fn write_bytes_into(vec: &mut Vec<u8>, data: &[u8]) -> u32 {
    let pos = vec.len() as u32;
    let mut buf = [0u8; 8];
    let len_vint_bytes = serialize_vint_u32(data.len() as u32, &mut buf);
    vec.extend_from_slice(len_vint_bytes);
    vec.extend_from_slice(data);
    pos
}

/// Serialize and return the position
fn write_into<T: BinarySerializable>(vec: &mut Vec<u8>, value: T) -> u32 {
    let pos = vec.len() as u32;
    value.serialize(vec).unwrap();
    pos
}

impl PartialEq for CompactDoc {
    fn eq(&self, other: &Self) -> bool {
        // super slow, but only here for tests
        let convert_to_comparable_map = |doc: &CompactDoc| {
            let mut field_value_set: HashMap<Field, HashSet<String>> = Default::default();
            for field_value in doc.field_values.iter() {
                let value: OwnedValue = doc.get_compact_doc_value(field_value.value_addr).into();
                let value = serde_json::to_string(&value).unwrap();
                field_value_set
                    .entry(Field::from_field_id(field_value.field as u32))
                    .or_default()
                    .insert(value);
            }
            field_value_set
        };
        let self_field_values: HashMap<Field, HashSet<String>> = convert_to_comparable_map(self);
        let other_field_values: HashMap<Field, HashSet<String>> = convert_to_comparable_map(other);
        self_field_values.eq(&other_field_values)
    }
}

impl Eq for CompactDoc {}

impl DocumentDeserialize for CompactDoc {
    fn deserialize<'de, D>(mut deserializer: D) -> Result<Self, DeserializeError>
    where
        D: DocumentDeserializer<'de>,
    {
        let mut doc = CompactDoc::default();
        // TODO: Deserializing into OwnedValue is wasteful. The deserializer should be able to work
        // on slices and referenced data.
        while let Some((field, value)) = deserializer.next_field::<OwnedValue>()? {
            doc.add_field_value(field, &value);
        }
        Ok(doc)
    }
}

/// A value of Compact Doc needs a reference to the container to extract its payload
#[derive(Debug, Clone, Copy)]
pub struct CompactDocValue<'a> {
    container: &'a CompactDoc,
    value_addr: ValueAddr,
}
impl PartialEq for CompactDocValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        let value1: OwnedValue = (*self).into();
        let value2: OwnedValue = (*other).into();
        value1 == value2
    }
}
impl From<CompactDocValue<'_>> for OwnedValue {
    fn from(value: CompactDocValue) -> Self {
        value.as_value().into()
    }
}
impl<'a> Value<'a> for CompactDocValue<'a> {
    type ArrayIter = CompactDocArrayIter<'a>;

    type ObjectIter = CompactDocObjectIter<'a>;

    fn as_value(&self) -> ReferenceValue<'a, Self> {
        self.get_ref_value().unwrap()
    }
}
impl<'a> CompactDocValue<'a> {
    fn get_ref_value(&self) -> io::Result<ReferenceValue<'a, CompactDocValue<'a>>> {
        let addr = self.value_addr.val_addr;
        match self.value_addr.type_id {
            ValueType::Null => Ok(ReferenceValueLeaf::Null.into()),
            ValueType::Str => {
                let str_ref = self.container.extract_str(addr);
                Ok(ReferenceValueLeaf::Str(str_ref).into())
            }
            ValueType::Facet => {
                let str_ref = self.container.extract_str(addr);
                Ok(ReferenceValueLeaf::Facet(str_ref).into())
            }
            ValueType::Bytes => {
                let data = self.container.extract_bytes(addr);
                Ok(ReferenceValueLeaf::Bytes(data).into())
            }
            ValueType::U64 => self
                .container
                .read_from::<u64>(addr)
                .map(ReferenceValueLeaf::U64)
                .map(Into::into),
            ValueType::I64 => self
                .container
                .read_from::<i64>(addr)
                .map(ReferenceValueLeaf::I64)
                .map(Into::into),
            ValueType::F64 => self
                .container
                .read_from::<f64>(addr)
                .map(ReferenceValueLeaf::F64)
                .map(Into::into),
            ValueType::Bool => Ok(ReferenceValueLeaf::Bool(addr != 0).into()),
            ValueType::Date => self
                .container
                .read_from::<i64>(addr)
                .map(|ts| ReferenceValueLeaf::Date(DateTime::from_timestamp_nanos(ts)))
                .map(Into::into),
            ValueType::IpAddr => self
                .container
                .read_from::<u128>(addr)
                .map(|num| ReferenceValueLeaf::IpAddr(Ipv6Addr::from_u128(num)))
                .map(Into::into),
            ValueType::PreTokStr => self
                .container
                .read_from::<PreTokenizedString>(addr)
                .map(Into::into)
                .map(ReferenceValueLeaf::PreTokStr)
                .map(Into::into),
            ValueType::Object => Ok(ReferenceValue::Object(CompactDocObjectIter::new(
                self.container,
                addr,
            )?)),
            ValueType::Array => Ok(ReferenceValue::Array(CompactDocArrayIter::new(
                self.container,
                addr,
            )?)),
        }
    }
}

/// The address in the vec
type Addr = u32;

#[derive(Clone, Copy, Default)]
#[repr(packed)]
/// The value type and the address to its payload in the container.
struct ValueAddr {
    type_id: ValueType,
    /// This is the address to the value in the vec, except for bool and null, which are inlined
    val_addr: Addr,
}
impl BinarySerializable for ValueAddr {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.type_id.serialize(writer)?;
        VInt(self.val_addr as u64).serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let type_id = ValueType::deserialize(reader)?;
        let val_addr = VInt::deserialize(reader)?.0 as u32;
        Ok(ValueAddr { type_id, val_addr })
    }
}
impl std::fmt::Debug for ValueAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val_addr = self.val_addr;
        f.write_fmt(format_args!("{:?} at {:?}", self.type_id, val_addr))
    }
}

/// A enum representing a value for tantivy to index.
///
/// ** Any changes need to be reflected in `BinarySerializable` for `ValueType` **
///
/// We can't use [schema::Type] or [columnar::ColumnType] here, because they are missing
/// some items like Array and PreTokStr.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum ValueType {
    /// A null value.
    #[default]
    Null = 0,
    /// The str type is used for any text information.
    Str = 1,
    /// Unsigned 64-bits Integer `u64`
    U64 = 2,
    /// Signed 64-bits Integer `i64`
    I64 = 3,
    /// 64-bits Float `f64`
    F64 = 4,
    /// Date/time with nanoseconds precision
    Date = 5,
    /// Facet
    Facet = 6,
    /// Arbitrarily sized byte array
    Bytes = 7,
    /// IpV6 Address. Internally there is no IpV4, it needs to be converted to `Ipv6Addr`.
    IpAddr = 8,
    /// Bool value
    Bool = 9,
    /// Pre-tokenized str type,
    PreTokStr = 10,
    /// Object
    Object = 11,
    /// Pre-tokenized str type,
    Array = 12,
}

impl BinarySerializable for ValueType {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u8).serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let num = u8::deserialize(reader)?;
        let type_id = if (0..=12).contains(&num) {
            unsafe { std::mem::transmute::<u8, ValueType>(num) }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid value type id: {num}"),
            ));
        };
        Ok(type_id)
    }
}

impl<'a, V: Value<'a>> From<&ReferenceValue<'a, V>> for ValueType {
    fn from(value: &ReferenceValue<'a, V>) -> Self {
        match value {
            ReferenceValue::Leaf(leaf) => leaf.into(),
            ReferenceValue::Array(_) => ValueType::Array,
            ReferenceValue::Object(_) => ValueType::Object,
        }
    }
}
impl<'a> From<&ReferenceValueLeaf<'a>> for ValueType {
    fn from(value: &ReferenceValueLeaf<'a>) -> Self {
        match value {
            ReferenceValueLeaf::Null => ValueType::Null,
            ReferenceValueLeaf::Str(_) => ValueType::Str,
            ReferenceValueLeaf::U64(_) => ValueType::U64,
            ReferenceValueLeaf::I64(_) => ValueType::I64,
            ReferenceValueLeaf::F64(_) => ValueType::F64,
            ReferenceValueLeaf::Bool(_) => ValueType::Bool,
            ReferenceValueLeaf::Date(_) => ValueType::Date,
            ReferenceValueLeaf::IpAddr(_) => ValueType::IpAddr,
            ReferenceValueLeaf::PreTokStr(_) => ValueType::PreTokStr,
            ReferenceValueLeaf::Facet(_) => ValueType::Facet,
            ReferenceValueLeaf::Bytes(_) => ValueType::Bytes,
        }
    }
}

#[derive(Debug, Clone)]
/// The Iterator for the object values in the compact document
pub struct CompactDocObjectIter<'a> {
    container: &'a CompactDoc,
    node_addresses_slice: &'a [u8],
}

impl<'a> CompactDocObjectIter<'a> {
    fn new(container: &'a CompactDoc, addr: Addr) -> io::Result<Self> {
        // Objects are `&[ValueAddr]` serialized into bytes
        let node_addresses_slice = container.extract_bytes(addr);
        Ok(Self {
            container,
            node_addresses_slice,
        })
    }
}

impl<'a> Iterator for CompactDocObjectIter<'a> {
    type Item = (&'a str, CompactDocValue<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_addresses_slice.is_empty() {
            return None;
        }
        let key_addr = ValueAddr::deserialize(&mut self.node_addresses_slice).ok()?;
        let key = self.container.extract_str(key_addr.val_addr);
        let value = ValueAddr::deserialize(&mut self.node_addresses_slice).ok()?;
        let value = CompactDocValue {
            container: self.container,
            value_addr: value,
        };
        Some((key, value))
    }
}

#[derive(Debug, Clone)]
/// The Iterator for the array values in the compact document
pub struct CompactDocArrayIter<'a> {
    container: &'a CompactDoc,
    node_addresses_slice: &'a [u8],
}

impl<'a> CompactDocArrayIter<'a> {
    fn new(container: &'a CompactDoc, addr: Addr) -> io::Result<Self> {
        // Arrays are &[ValueAddr] serialized into bytes
        let node_addresses_slice = container.extract_bytes(addr);
        Ok(Self {
            container,
            node_addresses_slice,
        })
    }
}

impl<'a> Iterator for CompactDocArrayIter<'a> {
    type Item = CompactDocValue<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_addresses_slice.is_empty() {
            return None;
        }
        let value = ValueAddr::deserialize(&mut self.node_addresses_slice).ok()?;
        let value = CompactDocValue {
            container: self.container,
            value_addr: value,
        };
        Some(value)
    }
}

impl Document for CompactDoc {
    type Value<'a> = CompactDocValue<'a>;
    type FieldsValuesIter<'a> = FieldValueIterRef<'a>;

    fn iter_fields_and_values(&self) -> Self::FieldsValuesIter<'_> {
        FieldValueIterRef {
            slice: self.field_values.iter(),
            container: self,
        }
    }
}

/// A helper wrapper for creating an iterator over the field values
pub struct FieldValueIterRef<'a> {
    slice: std::slice::Iter<'a, FieldValueAddr>,
    container: &'a CompactDoc,
}

impl<'a> Iterator for FieldValueIterRef<'a> {
    type Item = (Field, CompactDocValue<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next().map(|field_value| {
            (
                Field::from_field_id(field_value.field as u32),
                CompactDocValue::<'a> {
                    container: self.container,
                    value_addr: field_value.value_addr,
                },
            )
        })
    }
}

/// Error that may happen when deserializing
/// a document from JSON.
#[derive(Debug, Error, PartialEq)]
pub enum DocParsingError {
    /// The payload given is not valid JSON.
    #[error("The provided string is not valid JSON")]
    InvalidJson(String),
    /// One of the value node could not be parsed.
    #[error("The field '{0:?}' could not be parsed: {1:?}")]
    ValueError(String, ValueParsingError),
}

impl DocParsingError {
    /// Builds a NotJson DocParsingError
    fn invalid_json(invalid_json: &str) -> Self {
        let sample = invalid_json.chars().take(20).collect();
        DocParsingError::InvalidJson(sample)
    }
}

mod explode {
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
                TantivyDocument::from_json_object(&schema, value.as_object().unwrap().clone())
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

#[cfg(test)]
mod tests {
    use crate::schema::*;

    #[test]
    fn test_doc() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("title", TEXT);
        let mut doc = TantivyDocument::default();
        doc.add_text(text_field, "My title");
        assert_eq!(doc.field_values().count(), 1);

        let schema = schema_builder.build();
        let _val = doc.get_first(text_field).unwrap();
        let _json = doc.to_named_doc(&schema);
    }

    #[test]
    fn test_json_value() {
        let json_str = r#"{ 
            "toto": "titi",
            "float": -0.2,
            "bool": true,
            "unsigned": 1,
            "signed": -2,
            "complexobject": {
                "field.with.dot": 1
            },
            "date": "1985-04-12T23:20:50.52Z",
            "my_arr": [2, 3, {"my_key": "two tokens"}, 4, {"nested_array": [2, 5, 6, [7, 8, {"a": [{"d": {"e":[99]}}, 9000]}, 9, 10], [5, 5]]}]
        }"#;
        let json_val: std::collections::BTreeMap<String, OwnedValue> =
            serde_json::from_str(json_str).unwrap();

        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", TEXT);
        let mut doc = TantivyDocument::default();
        doc.add_object(json_field, json_val);

        let schema = schema_builder.build();
        let json = doc.to_json(&schema);
        let actual_json: serde_json::Value = serde_json::from_str(&json).unwrap();
        let expected_json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(actual_json["json"][0], expected_json);
    }

    // TODO: Should this be re-added with the serialize method
    //       technically this is no longer useful since the doc types
    //       do not implement BinarySerializable due to orphan rules.
    // #[test]
    // fn test_doc_serialization_issue() {
    //     let mut doc = Document::default();
    //     doc.add_json_object(
    //         Field::from_field_id(0),
    //         serde_json::json!({"key": 2u64})
    //             .as_object()
    //             .unwrap()
    //             .clone(),
    //     );
    //     doc.add_text(Field::from_field_id(1), "hello");
    //     assert_eq!(doc.field_values().len(), 2);
    //     let mut payload: Vec<u8> = Vec::new();
    //     doc_binary_wrappers::serialize(&doc, &mut payload).unwrap();
    //     assert_eq!(payload.len(), 26);
    //     doc_binary_wrappers::deserialize::<Document, _>(&mut &payload[..]).unwrap();
    // }
}

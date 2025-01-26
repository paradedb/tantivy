// src/schema/term.rs
use std::hash::{Hash, Hasher};
use std::net::Ipv6Addr;
use std::{fmt, str};

use columnar::MonotonicallyMappableToU128;
use common::json_path_writer::{JSON_END_OF_PATH, JSON_PATH_SEGMENT_SEP_STR};
use common::JsonPathWriter;

use super::date_time_options::DATE_TIME_PRECISION_INDEXED;
use super::{Field, Schema};
use crate::fastfield::FastValue;
use crate::json_utils::split_json_path;
use crate::schema::{Facet, Type};
use crate::DateTime;

/// Term represents the value that the token can take.
/// It's a serialized representation over different types.
///
/// It actually wraps a `Vec<u8>`. The first 5 bytes are metadata.
/// 4 bytes are the field id, and the last byte is the type.
///
/// The serialized value `ValueBytes` is considered everything after the 4 first bytes (term id).
#[derive(Clone)]
pub struct Term<B = Vec<u8>>(B)
where
    B: AsRef<[u8]>;

/// The number of bytes used as metadata by `Term`.
const TERM_METADATA_LENGTH: usize = 5;

impl Term {
    /// Create a new Term with a buffer with a given capacity.
    pub fn with_capacity(capacity: usize) -> Term {
        println!(
            "Term::with_capacity: Creating Term with capacity {}.",
            capacity
        );
        let mut data = Vec::with_capacity(TERM_METADATA_LENGTH + capacity);
        data.resize(TERM_METADATA_LENGTH, 0u8);
        println!(
            "Term::with_capacity: Initialized Term with TERM_METADATA_LENGTH {} and additional capacity {}.",
            TERM_METADATA_LENGTH, capacity
        );
        Term(data)
    }

    /// Creates a term from a json path.
    ///
    /// The json path can address a nested value in a JSON object.
    /// e.g. `{"k8s": {"node": {"id": 5}}}` can be addressed via `k8s.node.id`.
    ///
    /// In case there are dots in the field name, and the `expand_dots_enabled` parameter is not
    /// set they need to be escaped with a backslash.
    /// e.g. `{"k8s.node": {"id": 5}}` can be addressed via `k8s\.node.id`.
    pub fn from_field_json_path(field: Field, json_path: &str, expand_dots_enabled: bool) -> Term {
        println!(
            "Term::from_field_json_path: Creating Term from field {:?}, json_path '{}', expand_dots_enabled {}.",
            field, json_path, expand_dots_enabled
        );
        let paths = split_json_path(json_path);
        println!(
            "Term::from_field_json_path: Split json_path into {} segments.",
            paths.len()
        );
        let mut json_path_writer = JsonPathWriter::with_expand_dots(expand_dots_enabled);
        for (i, path) in paths.iter().enumerate() {
            println!(
                "Term::from_field_json_path: Pushing path segment {}: '{}'.",
                i + 1,
                path
            );
            json_path_writer.push(&path);
        }
        json_path_writer.set_end();
        println!("Term::from_field_json_path: Set JSON path end.");
        let mut term = Term::with_type_and_field(Type::Json, field);
        println!(
            "Term::from_field_json_path: Initialized Term with Type::Json and field {:?}.",
            field
        );

        term.append_bytes(json_path_writer.as_str().as_bytes());
        println!(
            "Term::from_field_json_path: Appended JSON path bytes to Term: {:?}.",
            json_path_writer.as_str()
        );

        println!(
            "Term::from_field_json_path: Created Term from field {:?} and json_path '{}'.",
            field, json_path
        );
        term
    }

    /// Gets the full path of the field name + optional json path.
    pub fn get_full_path(&self, schema: &Schema) -> String {
        println!("Term::get_full_path: Retrieving full path.");
        let field = self.field();
        println!("Term::get_full_path: Retrieved field {:?}.", field);
        let mut full_path = schema.get_field_name(field).to_string();
        println!(
            "Term::get_full_path: Field name '{}' obtained from schema.",
            full_path
        );
        if let Some(json_path) = self.get_json_path() {
            println!("Term::get_full_path: Found JSON path '{}'.", json_path);
            full_path.push('.');
            full_path.push_str(&json_path);
            println!(
                "Term::get_full_path: Appended JSON path to full path: '{}'.",
                full_path
            );
        } else {
            println!("Term::get_full_path: No JSON path found.");
        };
        println!("Term::get_full_path: Final full path is '{}'.", full_path);
        full_path
    }

    /// Gets the json path if the type is JSON
    pub fn get_json_path(&self) -> Option<String> {
        println!("Term::get_json_path: Checking if Term type is JSON.");
        let value = self.value();
        if let Some((json_path, _)) = value.as_json() {
            println!("Term::get_json_path: JSON path found. Extracting...");
            Some(unsafe {
                std::str::from_utf8_unchecked(&json_path[..json_path.len() - 1]).to_string()
            })
        } else {
            println!("Term::get_json_path: Term type is not JSON or JSON path not found.");
            None
        }
    }

    pub(crate) fn with_type_and_field(typ: Type, field: Field) -> Term {
        println!(
            "Term::with_type_and_field: Creating Term with Type {:?} and Field {:?}.",
            typ, field
        );
        let mut term = Self::with_capacity(8);
        term.set_field_and_type(field, typ);
        println!(
            "Term::with_type_and_field: Set field and type for Term: {:?}.",
            term.field()
        );
        term
    }

    fn with_bytes_and_field_and_payload(typ: Type, field: Field, bytes: &[u8]) -> Term {
        println!(
            "Term::with_bytes_and_field_and_payload: Creating Term with Type {:?}, Field {:?}, and payload bytes: {:?}.",
            typ, field, bytes
        );
        let mut term = Self::with_capacity(bytes.len());
        term.set_field_and_type(field, typ);
        term.0.extend_from_slice(bytes);
        println!(
            "Term::with_bytes_and_field_and_payload: Extended Term with payload bytes. Current Term bytes: {:?}.",
            term.0
        );
        term
    }

    pub(crate) fn from_fast_value<T: FastValue>(field: Field, val: &T) -> Term {
        println!(
            "Term::from_fast_value: Creating Term from Field {:?} and FastValue {:?}.",
            field, val
        );
        let mut term = Self::with_type_and_field(T::to_type(), field);
        println!(
            "Term::from_fast_value: Initialized Term with Type {:?} and Field {:?}.",
            T::to_type(),
            field
        );
        term.set_u64(val.to_u64());
        println!(
            "Term::from_fast_value: Set u64 value {} in Term.",
            val.to_u64()
        );
        term
    }

    /// Panics when the term is not empty... ie: some value is set.
    /// Use `clear_with_field_and_type` in that case.
    ///
    /// Sets field and the type.
    pub(crate) fn set_field_and_type(&mut self, field: Field, typ: Type) {
        println!(
            "Term::set_field_and_type: Setting field {:?} and type {:?}.",
            field, typ
        );
        assert!(
            self.is_empty(),
            "Term is not empty. Use clear_with_field_and_type instead."
        );
        let field_bytes = field.field_id().to_be_bytes();
        println!(
            "Term::set_field_and_type: Field ID bytes: {:?}.",
            field_bytes
        );
        self.0[0..4].clone_from_slice(field_bytes.as_ref());
        self.0[4] = typ.to_code();
        println!(
            "Term::set_field_and_type: Set field ID and type code in Term bytes: {:?}.",
            self.0
        );
    }

    /// Is empty if there are no value bytes.
    pub fn is_empty(&self) -> bool {
        let empty = self.0.len() == TERM_METADATA_LENGTH;
        println!(
            "Term::is_empty: Term is {}.",
            if empty { "empty" } else { "not empty" }
        );
        empty
    }

    /// Builds a term given a field, and a `Ipv6Addr`-value
    pub fn from_field_ip_addr(field: Field, ip_addr: Ipv6Addr) -> Term {
        println!(
            "Term::from_field_ip_addr: Creating Term from Field {:?} and Ipv6Addr {:?}.",
            field, ip_addr
        );
        let mut term = Self::with_type_and_field(Type::IpAddr, field);
        println!(
            "Term::from_field_ip_addr: Initialized Term with Type IpAddr and Field {:?}.",
            field
        );
        term.set_ip_addr(ip_addr);
        println!(
            "Term::from_field_ip_addr: Set Ipv6Addr {:?} in Term.",
            ip_addr
        );
        term
    }

    /// Builds a term given a field, and a `u64`-value
    pub fn from_field_u64(field: Field, val: u64) -> Term {
        println!(
            "Term::from_field_u64: Creating Term from Field {:?} and u64 value {}.",
            field, val
        );
        Term::from_fast_value(field, &val)
    }

    /// Builds a term given a field, and a `i64`-value
    pub fn from_field_i64(field: Field, val: i64) -> Term {
        println!(
            "Term::from_field_i64: Creating Term from Field {:?} and i64 value {}.",
            field, val
        );
        Term::from_fast_value(field, &val)
    }

    /// Builds a term given a field, and a `f64`-value
    pub fn from_field_f64(field: Field, val: f64) -> Term {
        println!(
            "Term::from_field_f64: Creating Term from Field {:?} and f64 value {}.",
            field, val
        );
        Term::from_fast_value(field, &val)
    }

    /// Builds a term given a field, and a `bool`-value
    pub fn from_field_bool(field: Field, val: bool) -> Term {
        println!(
            "Term::from_field_bool: Creating Term from Field {:?} and bool value {}.",
            field, val
        );
        Term::from_fast_value(field, &val)
    }

    /// Builds a term given a field, and a `DateTime` value.
    ///
    /// The contained value may not match the value, due to the truncation used
    /// for indexed data [super::DATE_TIME_PRECISION_INDEXED].
    /// To create a term used for search use `from_field_date_for_search`.
    pub fn from_field_date(field: Field, val: DateTime) -> Term {
        println!(
            "Term::from_field_date: Creating Term from Field {:?} and DateTime value {:?}.",
            field, val
        );
        Term::from_fast_value(field, &val)
    }

    /// Builds a term given a field, and a `DateTime` value to be used in searching the inverted
    /// index.
    /// It truncates the `DateTime` to the precision used in the index
    /// ([super::DATE_TIME_PRECISION_INDEXED]).
    pub fn from_field_date_for_search(field: Field, val: DateTime) -> Term {
        println!(
            "Term::from_field_date_for_search: Creating Term from Field {:?} and truncated DateTime value {:?}.",
            field, val.truncate(DATE_TIME_PRECISION_INDEXED)
        );
        Term::from_fast_value(field, &val.truncate(DATE_TIME_PRECISION_INDEXED))
    }

    /// Creates a `Term` given a facet.
    pub fn from_facet(field: Field, facet: &Facet) -> Term {
        println!(
            "Term::from_facet: Creating Term from Field {:?} and Facet {:?}.",
            field, facet
        );
        let facet_encoded_str = facet.encoded_str();
        println!(
            "Term::from_facet: Facet encoded string: '{}'.",
            facet_encoded_str
        );
        Term::with_bytes_and_field_and_payload(Type::Facet, field, facet_encoded_str.as_bytes())
    }

    /// Builds a term given a field, and a string value
    pub fn from_field_text(field: Field, text: &str) -> Term {
        println!(
            "Term::from_field_text: Creating Term from Field {:?} and text '{}'.",
            field, text
        );
        Term::with_bytes_and_field_and_payload(Type::Str, field, text.as_bytes())
    }

    /// Builds a term bytes.
    pub fn from_field_bytes(field: Field, bytes: &[u8]) -> Term {
        println!(
            "Term::from_field_bytes: Creating Term from Field {:?} and bytes {:?}.",
            field, bytes
        );
        Term::with_bytes_and_field_and_payload(Type::Bytes, field, bytes)
    }

    /// Removes the value_bytes and set the field and type code.
    pub(crate) fn clear_with_field_and_type(&mut self, typ: Type, field: Field) {
        println!(
            "Term::clear_with_field_and_type: Clearing Term and setting Type {:?}, Field {:?}.",
            typ, field
        );
        self.truncate_value_bytes(0);
        self.set_field_and_type(field, typ);
        println!(
            "Term::clear_with_field_and_type: Term cleared and field/type set. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Removes the value_bytes and set the type code.
    pub fn clear_with_type(&mut self, typ: Type) {
        println!(
            "Term::clear_with_type: Clearing value bytes and setting Type {:?}.",
            typ
        );
        self.truncate_value_bytes(0);
        self.0[4] = typ.to_code();
        println!(
            "Term::clear_with_type: Type code set to {:?}. Current Term bytes: {:?}.",
            typ.to_code(),
            self.0
        );
    }

    /// Sets a u64 value in the term.
    ///
    /// U64 are serialized using (8-byte) BigEndian
    /// representation.
    /// The use of BigEndian has the benefit of preserving
    /// the natural order of the values.
    pub fn set_u64(&mut self, val: u64) {
        println!("Term::set_u64: Setting u64 value {}.", val);
        self.set_fast_value(val);
        println!(
            "Term::set_u64: u64 value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Sets a `i64` value in the term.
    pub fn set_i64(&mut self, val: i64) {
        println!("Term::set_i64: Setting i64 value {}.", val);
        self.set_fast_value(val);
        println!(
            "Term::set_i64: i64 value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Sets a `DateTime` value in the term.
    pub fn set_date(&mut self, date: DateTime) {
        println!("Term::set_date: Setting DateTime value {:?}.", date);
        self.set_fast_value(date);
        println!(
            "Term::set_date: DateTime value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Sets a `f64` value in the term.
    pub fn set_f64(&mut self, val: f64) {
        println!("Term::set_f64: Setting f64 value {}.", val);
        self.set_fast_value(val);
        println!(
            "Term::set_f64: f64 value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Sets a `bool` value in the term.
    pub fn set_bool(&mut self, val: bool) {
        println!("Term::set_bool: Setting bool value {}.", val);
        self.set_fast_value(val);
        println!(
            "Term::set_bool: bool value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    fn set_fast_value<T: FastValue>(&mut self, val: T) {
        println!(
            "Term::set_fast_value: Converting FastValue {:?} to u64 and setting bytes.",
            val
        );
        self.set_bytes(val.to_u64().to_be_bytes().as_ref());
        println!(
            "Term::set_fast_value: FastValue bytes set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Append a type marker + fast value to a term.
    /// This is used in JSON type to append a fast value after the path.
    ///
    /// It will not clear existing bytes.
    pub fn append_type_and_fast_value<T: FastValue>(&mut self, val: T) {
        println!(
            "Term::append_type_and_fast_value: Appending Type {:?} and FastValue {:?} to Term.",
            T::to_type(),
            val
        );
        self.0.push(T::to_type().to_code());
        let value = val.to_u64();
        self.0.extend(value.to_be_bytes().as_ref());
        println!(
            "Term::append_type_and_fast_value: Appended Type {:?} and FastValue {}. Current Term bytes: {:?}.",
            T::to_type(),
            value,
            self.0
        );
    }

    /// Append a string type marker + string to a term.
    /// This is used in JSON type to append a str after the path.
    ///
    /// It will not clear existing bytes.
    pub fn append_type_and_str(&mut self, val: &str) {
        println!(
            "Term::append_type_and_str: Appending Type {:?} and string '{}' to Term.",
            Type::Str,
            val
        );
        self.0.push(Type::Str.to_code());
        self.0.extend(val.as_bytes().as_ref());
        println!(
            "Term::append_type_and_str: Appended Type {:?} and string '{}'. Current Term bytes: {:?}.",
            Type::Str,
            val,
            self.0
        );
    }

    /// Sets a `Ipv6Addr` value in the term.
    pub fn set_ip_addr(&mut self, val: Ipv6Addr) {
        println!("Term::set_ip_addr: Setting Ipv6Addr value {:?}.", val);
        self.set_bytes(val.to_u128().to_be_bytes().as_ref());
        println!(
            "Term::set_ip_addr: Ipv6Addr value set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Sets the value of a `Bytes` field.
    pub fn set_bytes(&mut self, bytes: &[u8]) {
        println!("Term::set_bytes: Setting bytes {:?} in Term.", bytes);
        self.truncate_value_bytes(0);
        self.0.extend(bytes);
        println!(
            "Term::set_bytes: Bytes set successfully. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// Truncates the value bytes of the term. Value and field type stays the same.
    pub fn truncate_value_bytes(&mut self, len: usize) {
        println!(
            "Term::truncate_value_bytes: Truncating value bytes to length {}.",
            len
        );
        self.0.truncate(len + TERM_METADATA_LENGTH);
        println!(
            "Term::truncate_value_bytes: Truncated Term bytes. Current Term bytes: {:?}.",
            self.0
        );
    }

    /// The length of the bytes.
    pub fn len_bytes(&self) -> usize {
        let len = self.0.len() - TERM_METADATA_LENGTH;
        println!("Term::len_bytes: Length of value bytes is {}.", len);
        len
    }

    /// Appends value bytes to the Term.
    ///
    /// This function returns the segment that has just been added.
    #[inline]
    pub fn append_bytes(&mut self, bytes: &[u8]) -> &mut [u8] {
        let cloned = self.0.clone();
        println!("Term::append_bytes: Appending bytes {:?} to Term.", bytes);
        let len_before = self.0.len();
        self.0.extend_from_slice(bytes);
        let appended_segment = &mut self.0[len_before..];
        println!(
            "Term::append_bytes: Appended bytes: {:?}.",
            appended_segment
        );
        appended_segment
    }

    /// Appends json path bytes to the Term.
    /// If the path contains 0 bytes, they are replaced by a "0" string.
    /// The 0 byte is used to mark the end of the path.
    ///
    /// This function returns the segment that has just been added.
    #[inline]
    pub fn append_path(&mut self, bytes: &[u8]) -> &mut [u8] {
        let cloned = self.clone();
        println!(
            "Term::append_path: Appending JSON path bytes {:?} to Term.",
            bytes
        );
        let len_before = self.0.len();
        assert!(
            !bytes.contains(&JSON_END_OF_PATH),
            "JSON path bytes contain the JSON_END_OF_PATH byte."
        );
        println!(
            "Term::append_path: Asserted that JSON path bytes do not contain JSON_END_OF_PATH."
        );
        self.0.extend_from_slice(bytes);
        let appended_segment = &mut self.0[len_before..];
        println!(
            "Term::append_path: Appended JSON path bytes: {:?}.",
            appended_segment
        );
        appended_segment
    }
}

impl<B> Term<B>
where
    B: AsRef<[u8]>,
{
    /// Wraps a object holding bytes
    pub fn wrap(data: B) -> Term<B> {
        println!(
            "Term::wrap: Wrapping data into Term. Data length: {} bytes.",
            data.as_ref().len()
        );
        Term(data)
    }

    /// Return the type of the term.
    pub fn typ(&self) -> Type {
        println!("Term::typ: Retrieving Type from Term.");
        let term_type = self.value().typ();
        println!("Term::typ: Term Type is {:?}.", term_type);
        term_type
    }

    /// Returns the field.
    pub fn field(&self) -> Field {
        println!("Term::field: Retrieving Field from Term.");
        let field_id_bytes: [u8; 4] = (&self.0.as_ref()[..4]).try_into().unwrap();
        let field = Field::from_field_id(u32::from_be_bytes(field_id_bytes));
        println!("Term::field: Retrieved Field {:?} from Term.", field);
        field
    }

    /// Returns the serialized representation of the value.
    /// (this does neither include the field id nor the value type.)
    ///
    /// If the term is a string, its value is utf-8 encoded.
    /// If the term is a u64, its value is encoded according
    /// to `byteorder::BigEndian`.
    pub fn serialized_value_bytes(&self) -> &[u8] {
        println!("Term::serialized_value_bytes: Retrieving serialized value bytes from Term.");
        &self.0.as_ref()[TERM_METADATA_LENGTH..]
    }

    /// Returns the value of the term.
    /// address or JSON path + value. (this does not include the field.)
    pub fn value(&self) -> ValueBytes<&[u8]> {
        println!("Term::value: Wrapping Term bytes into ValueBytes.");
        ValueBytes::wrap(&self.0.as_ref()[4..])
    }

    /// Returns the serialized representation of Term.
    /// This includes field_id, value type and value.
    ///
    /// Do NOT rely on this byte representation in the index.
    /// This value is likely to change in the future.
    #[inline]
    pub fn serialized_term(&self) -> &[u8] {
        println!("Term::serialized_term: Returning serialized Term bytes.");
        self.0.as_ref()
    }
}

/// ValueBytes represents a serialized value.
///
/// The value can be of any type of [`Type`] (e.g. string, u64, f64, bool, date, JSON).
/// The serialized representation matches the lexicographical order of the type.
///
/// The `ValueBytes` format is as follow:
/// `[type code: u8][serialized value]`
///
/// For JSON `ValueBytes` equals to:
/// `[type code=JSON][JSON path][JSON_END_OF_PATH][ValueBytes]`
///
/// The nested ValueBytes in JSON is never of type JSON. (there's no recursion)
#[derive(Clone)]
pub struct ValueBytes<B>(B)
where
    B: AsRef<[u8]>;

impl<B> ValueBytes<B>
where
    B: AsRef<[u8]>,
{
    /// Wraps a object holding bytes
    pub fn wrap(data: B) -> ValueBytes<B> {
        println!(
            "ValueBytes::wrap: Wrapping data into ValueBytes. Data length: {} bytes.",
            data.as_ref().len()
        );
        ValueBytes(data)
    }

    /// Wraps a object holding Vec<u8>
    pub fn to_owned(&self) -> ValueBytes<Vec<u8>> {
        println!("ValueBytes::to_owned: Cloning ValueBytes into owned Vec<u8>.");
        ValueBytes(self.0.as_ref().to_vec())
    }

    fn typ_code(&self) -> u8 {
        println!("ValueBytes::typ_code: Retrieving type code from ValueBytes.");
        self.0.as_ref()[0]
    }

    /// Return the type of the term.
    pub fn typ(&self) -> Type {
        println!(
            "ValueBytes::typ: Converting type code {} to Type.",
            self.typ_code()
        );
        let typ = Type::from_code(self.typ_code()).expect("The term has an invalid type code");
        println!("ValueBytes::typ: Type is {:?}.", typ);
        typ
    }

    /// Returns the `u64` value stored in a term.
    ///
    /// Returns `None` if the term is not of the u64 type, or if the term byte representation
    /// is invalid.
    pub fn as_u64(&self) -> Option<u64> {
        println!("ValueBytes::as_u64: Attempting to retrieve u64 value.");
        self.get_fast_type::<u64>()
    }

    fn get_fast_type<T: FastValue>(&self) -> Option<T> {
        println!(
            "ValueBytes::get_fast_type: Attempting to retrieve FastValue of Type {:?}.",
            T::to_type()
        );
        if self.typ() != T::to_type() {
            println!(
                "ValueBytes::get_fast_type: Term type {:?} does not match requested FastValue type {:?}.",
                self.typ(),
                T::to_type()
            );
            return None;
        }
        let value_bytes = self.raw_value_bytes_payload();
        println!(
            "ValueBytes::get_fast_type: Raw value bytes for FastValue: {:?}.",
            value_bytes
        );
        let value_u64 = u64::from_be_bytes(value_bytes.try_into().ok()?);
        println!(
            "ValueBytes::get_fast_type: Converted raw bytes to u64 value {}.",
            value_u64
        );
        let fast_val = T::from_u64(value_u64);
        println!(
            "ValueBytes::get_fast_type: Created FastValue {:?} from u64 value {}.",
            fast_val, value_u64
        );
        Some(fast_val)
    }

    /// Returns the `i64` value stored in a term.
    ///
    /// Returns `None` if the term is not of the i64 type, or if the term byte representation
    /// is invalid.
    pub fn as_i64(&self) -> Option<i64> {
        println!("ValueBytes::as_i64: Attempting to retrieve i64 value.");
        self.get_fast_type::<i64>()
    }

    /// Returns the `f64` value stored in a term.
    ///
    /// Returns `None` if the term is not of the f64 type, or if the term byte representation
    /// is invalid.
    pub fn as_f64(&self) -> Option<f64> {
        println!("ValueBytes::as_f64: Attempting to retrieve f64 value.");
        self.get_fast_type::<f64>()
    }

    /// Returns the `bool` value stored in a term.
    ///
    /// Returns `None` if the term is not of the bool type, or if the term byte representation
    /// is invalid.
    pub fn as_bool(&self) -> Option<bool> {
        println!("ValueBytes::as_bool: Attempting to retrieve bool value.");
        self.get_fast_type::<bool>()
    }

    /// Returns the `Date` value stored in a term.
    ///
    /// Returns `None` if the term is not of the Date type, or if the term byte representation
    /// is invalid.
    pub fn as_date(&self) -> Option<DateTime> {
        println!("ValueBytes::as_date: Attempting to retrieve DateTime value.");
        self.get_fast_type::<DateTime>()
    }

    /// Returns the text associated with the term.
    ///
    /// Returns `None` if the field is not of string type
    /// or if the bytes are not valid utf-8.
    pub fn as_str(&self) -> Option<&str> {
        println!("ValueBytes::as_str: Checking if Type is Str.");
        if self.typ() != Type::Str {
            println!("ValueBytes::as_str: Term type is not Str.");
            return None;
        }
        let bytes = self.raw_value_bytes_payload();
        println!(
            "ValueBytes::as_str: Attempting to convert bytes {:?} to &str.",
            bytes
        );
        match str::from_utf8(bytes) {
            Ok(s) => {
                println!(
                    "ValueBytes::as_str: Successfully converted bytes to &str: '{}'.",
                    s
                );
                Some(s)
            }
            Err(e) => {
                println!(
                    "ValueBytes::as_str: Failed to convert bytes to &str. Error: {:?}.",
                    e
                );
                None
            }
        }
    }

    /// Returns the facet associated with the term.
    ///
    /// Returns `None` if the field is not of facet type
    /// or if the bytes are not valid utf-8.
    pub fn as_facet(&self) -> Option<Facet> {
        println!("ValueBytes::as_facet: Checking if Type is Facet.");
        if self.typ() != Type::Facet {
            println!("ValueBytes::as_facet: Term type is not Facet.");
            return None;
        }
        let facet_encoded_str = str::from_utf8(self.raw_value_bytes_payload()).ok()?;
        println!(
            "ValueBytes::as_facet: Facet encoded string: '{}'.",
            facet_encoded_str
        );
        Some(Facet::from_encoded_string(facet_encoded_str.to_string()))
    }

    /// Returns the bytes associated with the term.
    ///
    /// Returns `None` if the field is not of bytes type.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        println!("ValueBytes::as_bytes: Checking if Type is Bytes.");
        if self.typ() != Type::Bytes {
            println!("ValueBytes::as_bytes: Term type is not Bytes.");
            return None;
        }
        println!(
            "ValueBytes::as_bytes: Retrieving bytes {:?} from Term.",
            self.raw_value_bytes_payload()
        );
        Some(self.raw_value_bytes_payload())
    }

    /// Returns a `Ipv6Addr` value from the term.
    pub fn as_ip_addr(&self) -> Option<Ipv6Addr> {
        println!("ValueBytes::as_ip_addr: Checking if Type is IpAddr.");
        if self.typ() != Type::IpAddr {
            println!("ValueBytes::as_ip_addr: Term type is not IpAddr.");
            return None;
        }
        let ip_u128 = u128::from_be_bytes(self.raw_value_bytes_payload().try_into().ok()?);
        println!(
            "ValueBytes::as_ip_addr: Converted bytes to u128 value {}.",
            ip_u128
        );
        let ip_addr = Ipv6Addr::from_u128(ip_u128);
        println!(
            "ValueBytes::as_ip_addr: Converted u128 to Ipv6Addr {:?}.",
            ip_addr
        );
        Some(ip_addr)
    }

    /// Returns the json path type.
    ///
    /// Returns `None` if the value is not JSON.
    pub fn json_path_type(&self) -> Option<Type> {
        println!("ValueBytes::json_path_type: Attempting to retrieve JSON path type.");
        let json_value_bytes = self.as_json_value_bytes()?;
        println!(
            "ValueBytes::json_path_type: Retrieved JSON ValueBytes with Type {:?}.",
            json_value_bytes.typ()
        );
        Some(json_value_bytes.typ())
    }

    /// Returns the json path bytes (including the JSON_END_OF_PATH byte),
    /// and the encoded ValueBytes after the json path.
    ///
    /// Returns `None` if the value is not JSON.
    pub(crate) fn as_json(&self) -> Option<(&[u8], ValueBytes<&[u8]>)> {
        println!("ValueBytes::as_json: Checking if Type is Json.");
        if self.typ() != Type::Json {
            println!("ValueBytes::as_json: Term type is not Json.");
            return None;
        }
        let bytes = self.raw_value_bytes_payload();
        println!(
            "ValueBytes::as_json: Retrieved raw value bytes: {:?}.",
            bytes
        );

        let pos = bytes.iter().cloned().position(|b| b == JSON_END_OF_PATH)?;
        println!(
            "ValueBytes::as_json: Found JSON_END_OF_PATH at position {}.",
            pos
        );
        // split at pos + 1, so that json_path_bytes includes the JSON_END_OF_PATH byte.
        let (json_path_bytes, term) = bytes.split_at(pos + 1);
        println!(
            "ValueBytes::as_json: Split bytes into json_path_bytes {:?} and term {:?}.",
            json_path_bytes, term
        );
        Some((json_path_bytes, ValueBytes::wrap(term)))
    }

    /// Returns the encoded ValueBytes after the json path.
    ///
    /// Returns `None` if the value is not JSON.
    pub(crate) fn as_json_value_bytes(&self) -> Option<ValueBytes<&[u8]>> {
        println!(
            "ValueBytes::as_json_value_bytes: Checking if Type is Json and retrieving ValueBytes."
        );
        if self.typ() != Type::Json {
            println!("ValueBytes::as_json_value_bytes: Term type is not Json. Returning None.");
            return None;
        }
        let bytes = self.raw_value_bytes_payload();
        println!(
            "ValueBytes::as_json_value_bytes: Retrieved raw value bytes: {:?}.",
            bytes
        );
        let pos = bytes.iter().cloned().position(|b| b == JSON_END_OF_PATH)?;
        println!(
            "ValueBytes::as_json_value_bytes: Found JSON_END_OF_PATH at position {}.",
            pos
        );
        let value_bytes = &bytes[pos + 1..];
        println!(
            "ValueBytes::as_json_value_bytes: Extracted value_bytes after JSON path: {:?}.",
            value_bytes
        );
        Some(ValueBytes::wrap(value_bytes))
    }

    /// Returns the raw value of ValueBytes payload, without the type tag.
    pub(crate) fn raw_value_bytes_payload(&self) -> &[u8] {
        println!("ValueBytes::raw_value_bytes_payload: Retrieving raw value bytes payload.");
        &self.0.as_ref()[1..]
    }

    /// Returns the serialized value of ValueBytes payload, without the type tag.
    pub(crate) fn value_bytes_payload(&self) -> Vec<u8> {
        println!("ValueBytes::value_bytes_payload: Retrieving serialized value bytes payload.");
        if let Some(value_bytes) = self.as_json_value_bytes() {
            println!(
                "ValueBytes::value_bytes_payload: Term is Json. Retrieving ValueBytes payload {:?}.",
                value_bytes.raw_value_bytes_payload()
            );
            value_bytes.raw_value_bytes_payload().to_vec()
        } else {
            println!(
                "ValueBytes::value_bytes_payload: Term is not Json. Retrieving raw value bytes payload {:?}.",
                self.raw_value_bytes_payload()
            );
            self.raw_value_bytes_payload().to_vec()
        }
    }

    /// Returns the serialized representation of Term.
    ///
    /// Do NOT rely on this byte representation in the index.
    /// This value is likely to change in the future.
    pub fn as_serialized(&self) -> &[u8] {
        println!(
            "ValueBytes::as_serialized: Returning serialized Term bytes: {:?}.",
            self.0.as_ref()
        );
        self.0.as_ref()
    }

    fn debug_value_bytes(&self, f: &mut fmt::Formatter) -> fmt::Result {
        println!("ValueBytes::debug_value_bytes: Starting to format ValueBytes for Debug.");
        let typ = self.typ();
        write!(f, "type={typ:?}, ")?;
        match typ {
            Type::Str => {
                println!("ValueBytes::debug_value_bytes: Term Type is Str.");
                let s = self.as_str();
                write_opt(f, s)?;
            }
            Type::U64 => {
                println!("ValueBytes::debug_value_bytes: Term Type is U64.");
                write_opt(f, self.as_u64())?;
            }
            Type::I64 => {
                println!("ValueBytes::debug_value_bytes: Term Type is I64.");
                write_opt(f, self.as_i64())?;
            }
            Type::F64 => {
                println!("ValueBytes::debug_value_bytes: Term Type is F64.");
                write_opt(f, self.as_f64())?;
            }
            Type::Bool => {
                println!("ValueBytes::debug_value_bytes: Term Type is Bool.");
                write_opt(f, self.as_bool())?;
            }
            // TODO pretty print these types too.
            Type::Date => {
                println!("ValueBytes::debug_value_bytes: Term Type is Date.");
                write_opt(f, self.as_date())?;
            }
            Type::Facet => {
                println!("ValueBytes::debug_value_bytes: Term Type is Facet.");
                write_opt(f, self.as_facet())?;
            }
            Type::Bytes => {
                println!("ValueBytes::debug_value_bytes: Term Type is Bytes.");
                write_opt(f, self.as_bytes())?;
            }
            Type::Json => {
                println!("ValueBytes::debug_value_bytes: Term Type is Json.");
                if let Some((path_bytes, sub_value_bytes)) = self.as_json() {
                    // Remove the JSON_END_OF_PATH byte & convert to utf8.
                    let path = str::from_utf8(&path_bytes[..path_bytes.len() - 1])
                        .map_err(|_| std::fmt::Error)?;
                    let path_pretty = path.replace(JSON_PATH_SEGMENT_SEP_STR, ".");
                    println!(
                        "ValueBytes::debug_value_bytes: JSON path pretty string: '{}'.",
                        path_pretty
                    );
                    write!(f, "path={path_pretty}, ")?;
                    sub_value_bytes.debug_value_bytes(f)?;
                }
            }
            Type::IpAddr => {
                println!("ValueBytes::debug_value_bytes: Term Type is IpAddr.");
                write_opt(f, self.as_ip_addr())?;
            }
        }
        println!("ValueBytes::debug_value_bytes: Completed formatting ValueBytes for Debug.");
        Ok(())
    }
}

impl<B> Ord for Term<B>
where
    B: AsRef<[u8]>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        println!(
            "Term::cmp: Comparing Term {:?} with Term {:?}.",
            self.serialized_term(),
            other.serialized_term()
        );
        let result = self.serialized_term().cmp(other.serialized_term());
        println!("Term::cmp: Comparison result is {:?}.", result);
        result
    }
}

impl<B> PartialOrd for Term<B>
where
    B: AsRef<[u8]>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        println!("Term::partial_cmp: Performing partial comparison.");
        Some(self.cmp(other))
    }
}

impl<B> PartialEq for Term<B>
where
    B: AsRef<[u8]>,
{
    fn eq(&self, other: &Self) -> bool {
        println!(
            "Term::eq: Checking equality between Term {:?} and Term {:?}.",
            self.serialized_term(),
            other.serialized_term()
        );
        let is_equal = self.serialized_term() == other.serialized_term();
        println!("Term::eq: Equality result is {}.", is_equal);
        is_equal
    }
}

impl<B> Eq for Term<B> where B: AsRef<[u8]> {}

impl<B> Hash for Term<B>
where
    B: AsRef<[u8]>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        println!("Term::hash: Hashing Term bytes {:?}.", self.0.as_ref());
        self.0.as_ref().hash(state)
    }
}

fn write_opt<T: std::fmt::Debug>(f: &mut fmt::Formatter, val_opt: Option<T>) -> fmt::Result {
    if let Some(val) = val_opt {
        println!("write_opt: Writing value {:?}.", val);
        write!(f, "{val:?}")?;
    } else {
        println!("write_opt: No value to write.");
    }
    Ok(())
}

impl<B> fmt::Debug for Term<B>
where
    B: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        println!("Term::fmt::Debug: Formatting Term for Debug.");
        let field_id = self.field().field_id();
        write!(f, "Term(field={field_id}, ")?;
        let value_bytes = ValueBytes::wrap(&self.0.as_ref()[4..]);
        value_bytes.debug_value_bytes(f)?;
        write!(f, ")",)?;
        println!("Term::fmt::Debug: Completed formatting Term for Debug.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::schema::*;

    #[test]
    pub fn test_term_str() {
        println!("test_term_str: Starting test for Term with string value.");
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("text", STRING);
        let title_field = schema_builder.add_text_field("title", STRING);
        println!("test_term_str: Added text fields 'text' and 'title' to schema.");
        let term = Term::from_field_text(title_field, "test");
        println!("test_term_str: Created Term from text 'test'.");
        assert_eq!(term.field(), title_field);
        println!("test_term_str: Asserted that Term.field() matches title_field.");
        assert_eq!(term.typ(), Type::Str);
        println!("test_term_str: Asserted that Term.typ() is Type::Str.");
        assert_eq!(term.value().as_str(), Some("test"));
        println!("test_term_str: Asserted that Term.value().as_str() returns Some('test').");
        println!("test_term_str: Test completed successfully.");
    }

    /// Size (in bytes) of the buffer of a fast value (u64, i64, f64, or date) term.
    /// <field> + <type byte> + <value len>
    ///
    /// - <field> is a big endian encoded u32 field id
    /// - <type_byte>'s most significant bit expresses whether the term is a json term or not The
    ///   remaining 7 bits are used to encode the type of the value. If this is a JSON term, the
    ///   type is the type of the leaf of the json.
    /// - <value> is,  if this is not the json term, a binary representation specific to the type.
    ///   If it is a JSON Term, then it is prepended with the path that leads to this leaf value.
    const FAST_VALUE_TERM_LEN: usize = 4 + 1 + 8;

    #[test]
    pub fn test_term_u64() {
        println!("test_term_u64: Starting test for Term with u64 value.");
        let mut schema_builder = Schema::builder();
        let count_field = schema_builder.add_u64_field("count", INDEXED);
        println!("test_term_u64: Added u64 field 'count' to schema.");
        let term = Term::from_field_u64(count_field, 983u64);
        println!("test_term_u64: Created Term from u64 value 983.");
        assert_eq!(term.field(), count_field);
        println!("test_term_u64: Asserted that Term.field() matches count_field.");
        assert_eq!(term.typ(), Type::U64);
        println!("test_term_u64: Asserted that Term.typ() is Type::U64.");
        assert_eq!(term.serialized_term().len(), FAST_VALUE_TERM_LEN);
        println!(
            "test_term_u64: Asserted that Term.serialized_term().len() is {}.",
            FAST_VALUE_TERM_LEN
        );
        assert_eq!(term.value().as_u64(), Some(983u64));
        println!("test_term_u64: Asserted that Term.value().as_u64() returns Some(983u64).");
        println!("test_term_u64: Test completed successfully.");
    }

    #[test]
    pub fn test_term_bool() {
        println!("test_term_bool: Starting test for Term with bool value.");
        let mut schema_builder = Schema::builder();
        let bool_field = schema_builder.add_bool_field("bool", INDEXED);
        println!("test_term_bool: Added bool field 'bool' to schema.");
        let term = Term::from_field_bool(bool_field, true);
        println!("test_term_bool: Created Term from bool value true.");
        assert_eq!(term.field(), bool_field);
        println!("test_term_bool: Asserted that Term.field() matches bool_field.");
        assert_eq!(term.typ(), Type::Bool);
        println!("test_term_bool: Asserted that Term.typ() is Type::Bool.");
        assert_eq!(term.serialized_term().len(), FAST_VALUE_TERM_LEN);
        println!(
            "test_term_bool: Asserted that Term.serialized_term().len() is {}.",
            FAST_VALUE_TERM_LEN
        );
        assert_eq!(term.value().as_bool(), Some(true));
        println!("test_term_bool: Asserted that Term.value().as_bool() returns Some(true).");
        println!("test_term_bool: Test completed successfully.");
    }
}

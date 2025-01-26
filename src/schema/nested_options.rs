// src/schema/nested_options.rs
use crate::schema::TextFieldIndexing;
use serde::{Deserialize, Serialize};

use super::JsonObjectOptions;

/// Options for a "nested" field in Tantivy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct NestedOptions {
    /// If true, nested child fields also appear in the parent doc.
    pub include_in_parent: bool,
    /// If true, nested child fields also appear in the root doc (if multiple levels).
    pub include_in_root: bool,
    /// If true, we store a hidden parent flag for each doc.
    /// Some users may prefer to do parent-detection in a different way.
    pub store_parent_flag: bool,
    #[serde(default)]
    expand_dots_enabled: bool,
}

impl NestedOptions {
    pub fn new() -> Self {
        println!("Creating new NestedOptions with defaults:");
        println!("  include_in_parent: false");
        println!("  include_in_root: false");
        println!("  store_parent_flag: true");
        NestedOptions {
            include_in_parent: false,
            include_in_root: false,
            store_parent_flag: true, // default to true
            expand_dots_enabled: false,
        }
    }

    pub fn set_include_in_parent(mut self, val: bool) -> Self {
        println!(
            "Setting include_in_parent: {} -> {}",
            self.include_in_parent, val
        );
        self.include_in_parent = val;
        self
    }
    pub fn set_include_in_root(mut self, val: bool) -> Self {
        println!(
            "Setting include_in_root: {} -> {}",
            self.include_in_root, val
        );
        self.include_in_root = val;
        self
    }
    pub fn set_store_parent_flag(mut self, val: bool) -> Self {
        println!(
            "Setting store_parent_flag: {} -> {}",
            self.store_parent_flag, val
        );
        self.store_parent_flag = val;
        self
    }

    pub fn is_indexed(&self) -> bool {
        // By default, we do not index the nested field itself
        // because we rely on expansions. If you want to index it,
        // you'd change this logic.
        println!("Checking is_indexed() -> false");
        false
    }

    pub fn is_stored(&self) -> bool {
        println!("Checking is_stored() -> false");
        false
    }

    pub fn is_fast(&self) -> bool {
        println!("Checking is_fast() -> false");
        false
    }

    pub fn fieldnorms(&self) -> bool {
        println!("Checking fieldnorms() -> false");
        false
    }

    pub fn get_text_indexing_options(&self) -> Option<&TextFieldIndexing> {
        println!("Getting text_indexing_options() -> None");
        None
    }
}

/// Options for a “nested JSON” field in Tantivy.
/// Combines `NestedOptions` for child/parent docs,
/// plus `JsonObjectOptions` for storing/indexing JSON text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NestedJsonObjectOptions {
    pub nested_opts: NestedOptions,
    pub json_opts: JsonObjectOptions,
}

impl Default for NestedJsonObjectOptions {
    fn default() -> Self {
        NestedJsonObjectOptions {
            nested_opts: NestedOptions::default(),
            json_opts: JsonObjectOptions::default(),
        }
    }
}

impl NestedJsonObjectOptions {
    /// Convenience constructor
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_indexed(&self) -> bool {
        // If you want to tokenize child fields in the parent doc,
        // rely on the underlying json_opts.
        let indexed = self.json_opts.is_indexed();
        println!("NestedJsonObjectOptions::is_indexed => {}", indexed);
        indexed
    }

    pub fn is_stored(&self) -> bool {
        let stored = self.json_opts.is_stored();
        println!("NestedJsonObjectOptions::is_stored => {}", stored);
        stored
    }

    pub fn is_fast(&self) -> bool {
        let fast = self.json_opts.is_fast();
        println!("NestedJsonObjectOptions::is_fast => {}", fast);
        fast
    }

    #[inline]
    pub fn is_expand_dots_enabled(&self) -> bool {
        self.nested_opts.expand_dots_enabled
    }

    pub fn fieldnorms(&self) -> bool {
        println!("Checking fieldnorms() -> false");
        false
    }

    pub fn get_text_indexing_options(&self) -> Option<&TextFieldIndexing> {
        let text_opts = self.json_opts.get_text_indexing_options();
        println!(
            "NestedJsonObjectOptions::get_text_indexing_options => {:?}",
            text_opts
        );
        text_opts
    }

    // -------------------
    // NESTED OPTIONS
    // -------------------
    pub fn set_include_in_parent(mut self, yes: bool) -> Self {
        self.nested_opts = self.nested_opts.set_include_in_parent(yes);
        self
    }
    pub fn set_store_parent_flag(mut self, yes: bool) -> Self {
        self.nested_opts = self.nested_opts.set_store_parent_flag(yes);
        self
    }

    // -------------------
    // JSON OPTIONS
    // -------------------
    /// If you want to index text inside the JSON subfields,
    /// pass in a `TextFieldIndexing` (tokenizer, record=position, etc.)
    pub fn set_indexing_options(mut self, indexing: TextFieldIndexing) -> Self {
        self.json_opts = self.json_opts.set_indexing_options(indexing);
        self
    }
    /// Mark the JSON as stored (the full JSON is retrievable).
    pub fn set_stored(mut self) -> Self {
        self.json_opts = self.json_opts.set_stored();
        self
    }
    /// Enable fast field indexing (for e.g. numeric subfields or raw text tokens).
    /// If `tokenizer_name` is Some("…"), the text is tokenized with that choice.
    pub fn set_fast(mut self, tokenizer_name: Option<&str>) -> Self {
        self.json_opts = self.json_opts.set_fast(tokenizer_name);
        self
    }
    /// Expand '.' in object keys into nested sub-objects (instead of literal dots).
    pub fn set_expand_dots_enabled(mut self) -> Self {
        self.json_opts = self.json_opts.set_expand_dots_enabled();
        self
    }
}

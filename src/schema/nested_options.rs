use serde::{Deserialize, Serialize};

use super::JsonObjectOptions;
use crate::schema::TextFieldIndexing;

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
        NestedOptions {
            include_in_parent: false,
            include_in_root: false,
            store_parent_flag: true, // default to true
            expand_dots_enabled: false,
        }
    }

    pub fn set_include_in_parent(mut self, val: bool) -> Self {
        self.include_in_parent = val;
        self
    }

    pub fn set_include_in_root(mut self, val: bool) -> Self {
        self.include_in_root = val;
        self
    }

    pub fn set_store_parent_flag(mut self, val: bool) -> Self {
        self.store_parent_flag = val;
        self
    }

    pub fn is_indexed(&self) -> bool {
        false
    }

    pub fn is_stored(&self) -> bool {
        false
    }

    pub fn is_fast(&self) -> bool {
        false
    }

    pub fn fieldnorms(&self) -> bool {
        false
    }

    pub fn get_text_indexing_options(&self) -> Option<&TextFieldIndexing> {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct NestedJsonObjectOptions {
    pub nested_opts: NestedOptions,
    pub json_opts: JsonObjectOptions,
}

impl NestedJsonObjectOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_indexed(&self) -> bool {
        // If you want to tokenize child fields in the parent doc,
        // rely on the underlying json_opts.
        self.json_opts.is_indexed()
    }

    pub fn is_stored(&self) -> bool {
        self.json_opts.is_stored()
    }

    pub fn is_fast(&self) -> bool {
        self.json_opts.is_fast()
    }

    #[inline]
    pub fn is_expand_dots_enabled(&self) -> bool {
        self.nested_opts.expand_dots_enabled
    }

    pub fn fieldnorms(&self) -> bool {
        false
    }

    pub fn get_text_indexing_options(&self) -> Option<&TextFieldIndexing> {
        self.json_opts.get_text_indexing_options()
    }

    pub fn set_include_in_parent(mut self, yes: bool) -> Self {
        self.nested_opts = self.nested_opts.set_include_in_parent(yes);
        self
    }

    pub fn set_store_parent_flag(mut self, yes: bool) -> Self {
        self.nested_opts = self.nested_opts.set_store_parent_flag(yes);
        self
    }

    // -------------------
    pub fn set_indexing_options(mut self, indexing: TextFieldIndexing) -> Self {
        self.json_opts = self.json_opts.set_indexing_options(indexing);
        self
    }

    pub fn set_stored(mut self) -> Self {
        self.json_opts = self.json_opts.set_stored();
        self
    }

    pub fn set_fast(mut self, tokenizer_name: Option<&str>) -> Self {
        self.json_opts = self.json_opts.set_fast(tokenizer_name);
        self
    }

    pub fn set_expand_dots_enabled(mut self) -> Self {
        self.json_opts = self.json_opts.set_expand_dots_enabled();
        self
    }
}

//! Character filters preprocess text before tokenization.
//!
//! Unlike token filters that operate on individual tokens, character filters
//! operate on the raw character stream and can add, remove, or modify characters.
//!
//! # Example
//!
//! ```rust,no_run
//! use tantivy::tokenizer::*;
//!
//! let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
//!     .char_filter(Some(HtmlStripCharacterFilter::default()))
//!     .filter(RemoveLongFilter::limit(40))
//!     .filter(LowerCaser)
//!     .build();
//!
//! let mut stream = analyzer.token_stream("<b>Hello World</b>");
//! // Tokens: ["hello", "world"]
//! ```

// Note: This module cannot use super::tokenizer::TextAnalyzer due to circular dependencies

/// A character filter preprocesses text before tokenization.
///
/// Character filters receive the original text as a stream of characters
/// and can transform the stream by adding, removing, or changing characters.
///
/// This is different from token filters, which operate on already tokenized text.
pub trait CharacterFilter: 'static + Send + Sync {
    /// Filter the input text, returning a new owned string.
    ///
    /// This method is called before tokenization. The returned string
    /// will be passed to the tokenizer.
    fn filter(&self, text: &str) -> String;

    fn box_clone(&self) -> Box<dyn CharacterFilter>;
}

pub trait BoxableCharacterFilter: CharacterFilter + Send + Sync {
    fn box_clone(&self) -> Box<dyn CharacterFilter>;
}

impl<T: CharacterFilter + Clone> BoxableCharacterFilter for T {
    fn box_clone(&self) -> Box<dyn CharacterFilter> {
        Box::new(self.clone())
    }
}

pub use super::html_strip::HtmlStripCharacterFilter;

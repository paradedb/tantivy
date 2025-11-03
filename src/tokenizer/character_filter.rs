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

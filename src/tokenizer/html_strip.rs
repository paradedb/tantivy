//! HTML Strip Character Filter
//!
//! The `html_strip` character filter strips out HTML elements like `<b>`
//! and decodes HTML entities like `&amp;` and `&lt;`.
//!
//! It also supports an `escaped_tags` option, which allows specific HTML
//! tags to be kept as literal text (e.g. `<b>` → `&lt;b&gt;` → `<b>`).
//!
//! # Example
//!
//! ```rust,no_run
//! use tantivy::tokenizer::*;
//!
//! let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
//!     .char_filter(Some(HtmlStripCharacterFilter::default()))
//!     .filter(LowerCaser)
//!     .build();
//!
//! let stream = analyzer.token_stream("<b>Hello</b> &amp; <i>World</i>");
//! // Tokens: ["hello", "&", "world"]
//! ```

use std::collections::HashSet;

use html_escape::{decode_html_entities, encode_text};

use super::character_filter::CharacterFilter;

/// Character filter that strips HTML elements and decodes HTML entities.
///
/// Supports an `escaped_tags` list to skip stripping specific tags.
#[derive(Clone, Default)]
pub struct HtmlStripCharacterFilter {
    escaped_tags: HashSet<String>,
}

impl HtmlStripCharacterFilter {
    /// Create a new filter with optional escaped tags.
    pub fn new<T: Into<String>>(escaped_tags: impl IntoIterator<Item = T>) -> Self {
        Self {
            escaped_tags: escaped_tags.into_iter().map(|t| t.into()).collect(),
        }
    }
}

impl CharacterFilter for HtmlStripCharacterFilter {
    fn filter(&self, text: &str) -> String {
        let stripped = strip_html(text, &self.escaped_tags);
        decode_html_entities(&stripped).to_string()
    }

    fn box_clone(&self) -> Box<dyn CharacterFilter> {
        Box::new(self.clone())
    }
}

/// Remove HTML tags while skipping whitelisted tags in `escaped_tags`.
fn strip_html(text: &str, escaped_tags: &HashSet<String>) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            let mut tag_buf = String::from("<");
            for c in chars.by_ref() {
                tag_buf.push(c);
                if c == '>' {
                    break;
                }
            }

            // collect tag name (e.g. "b" from "<b>", "div" from "</div class='x'>")
            let mut tag_name = String::new();
            let mut tag_chars = tag_buf.chars().peekable();

            // skip initial '<' or '</'
            if tag_chars.peek() == Some(&'<') {
                tag_chars.next();
                if tag_chars.peek() == Some(&'/') {
                    tag_chars.next();
                }
            }

            // collect until space, '>', or '/'
            while let Some(&c) = tag_chars.peek() {
                if c.is_whitespace() || c == '>' || c == '/' {
                    break;
                }
                tag_name.push(c);
                tag_chars.next();
            }

            let tag_name = tag_name.to_ascii_lowercase();

            // if this tag is in escaped_tags, output it escaped
            if escaped_tags.contains(&tag_name) {
                result.push_str(&encode_text(&tag_buf));
            }
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SimpleTokenizer, TextAnalyzer};

    #[test]
    fn test_basic_strip_and_decode() {
        let html = "<b>Hello</b> &amp; <i>World</i>";
        let filtered = HtmlStripCharacterFilter::default().filter(html);
        assert_eq!(filtered, "Hello & World");
    }

    #[test]
    fn test_escaped_tags() {
        let html = "<p>I'm <b>so</b> <i>happy</i>!</p>";
        let filter = HtmlStripCharacterFilter::new(["b", "i"]);
        let filtered = filter.filter(html);
        assert_eq!(filtered, "I'm <b>so</b> <i>happy</i>!");
    }

    #[test]
    fn test_nested_tags() {
        let html = "<p>Paragraph <em>with emphasis</em> text</p>";
        let filtered = HtmlStripCharacterFilter::default().filter(html);
        assert_eq!(filtered, "Paragraph with emphasis text");
    }

    #[test]
    fn test_multiple_entities() {
        let html = "&quot;Hello&quot; &amp; &lt;world&gt;";
        let filtered = HtmlStripCharacterFilter::default().filter(html);
        assert_eq!(filtered, "\"Hello\" & <world>");
    }

    #[test]
    fn test_with_tokenizer() {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .char_filter(Some(HtmlStripCharacterFilter::default()))
            .build();

        let mut stream = analyzer.token_stream("<b>Hello World</b>");
        let mut tokens = Vec::new();
        stream.process(&mut |t| tokens.push(t.clone()));

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "World");
    }

    #[test]
    fn test_escaped_tags_with_entities() {
        let html = "<p>I&apos;m <b>so</b> &amp; <i>happy</i>!</p>";
        let filter = HtmlStripCharacterFilter::new(["b"]);
        let filtered = filter.filter(html);
        assert_eq!(filtered, "I'm <b>so</b> & happy!");
    }

    #[test]
    fn test_ampersand_not_entity() {
        let html = "A & B";
        let filtered = HtmlStripCharacterFilter::default().filter(html);
        assert_eq!(filtered, "A & B");
    }

    #[test]
    fn test_malformed_tag_tolerant() {
        let html = "text <b unclosed text> more </p weird>";
        let filter = HtmlStripCharacterFilter::new(["b"]);
        let filtered = filter.filter(html);
        assert_eq!(filtered, "text <b unclosed text> more ");
    }
}

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

use fst::Set;
use html_escape::{decode_html_entities, encode_text};
use std::collections::HashSet;

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
        let escaped_tags_fst = Set::from_iter(self.escaped_tags.iter().map(|t| t.as_str())).expect("should be able to build fst from escaped_tags");
        let stripped = strip_html(text, &escaped_tags_fst);
        decode_html_entities(&stripped).to_string()
    }

    fn box_clone(&self) -> Box<dyn CharacterFilter> {
        Box::new(self.clone())
    }
}

/// Remove HTML tags while skipping whitelisted tags in `escaped_tags`.
pub fn strip_html(text: &str, escaped_tags_fst: &Set<Vec<u8>>) -> String {
    let fst = escaped_tags_fst.as_fst();
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars();

    while let Some(ch) = chars.next() {
        if ch != '<' {
            result.push(ch);
            continue;
        }

        // we're inside a tag now
        let mut tag_buf = String::from("<");
        let mut current_node = fst.root();
        let mut is_valid_path = true;
        let mut is_final = false;
        let mut in_tag_name = false;
        let mut after_slash = false;

        // consume characters until '>' or end-of-input
        while let Some(c) = chars.next() {
            tag_buf.push(c);

            if c == '>' {
                break;
            }

            // skip initial '<' and optional '/'
            if !in_tag_name {
                if c == '/' && !after_slash {
                    after_slash = true;
                    continue;
                }
                if c.is_ascii_alphabetic() {
                    in_tag_name = true;
                } else {
                    continue;
                }
            }

            // walk the tag name directly through the FST
            let lc = c.to_ascii_lowercase();
            if lc.is_whitespace() || lc == '>' || lc == '/' {
                in_tag_name = false;
                continue;
            }

            let byte = lc as u8;
            if let Some(trans) = current_node.transitions().find(|t| t.inp == byte) {
                current_node = fst.node(trans.addr);
                if current_node.is_final() {
                    is_final = true;
                }
            } else {
                is_valid_path = false;
                // no point continuing traversal; tag isn't whitelisted
                // but keep consuming until '>' so we don’t break parsing
            }
        }

        if is_valid_path && is_final {
            // this tag is in escaped_tags_fst, escape it literally
            result.push_str(&encode_text(&tag_buf));
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

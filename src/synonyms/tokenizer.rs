//! The tokenizer of a synonym rule field.

use super::rule::SynonymRule;
use crate::tokenizer::{PreTokenizedStream, PreTokenizedString, TextAnalyzer, Token, Tokenizer};

/// Joins the tokens of a key. A rule cannot contain this byte, so two keys never join to the
/// same term.
pub(crate) const KEY_SEPARATOR: u8 = b'\0';

pub(crate) fn join_key(tokens: &[String]) -> String {
    tokens.join("\0")
}

/// Indexes a Solr synonym rule, such as `couch, sofa, divan` or `sneakers => running shoes`, as
/// one term per key, the analyzed tokens of the key joined with a zero byte.
///
/// A line that is not a valid rule gives no term, and its document is still added. Use
/// [`SynonymRule::check`] to reject it instead.
#[derive(Clone)]
pub struct SynonymRuleTokenizer {
    analyzer: TextAnalyzer,
}

impl SynonymRuleTokenizer {
    /// `analyzer` is the analyzer of the fields that the rules expand, so a stemmer acts on a rule
    /// the way it acts on a query.
    pub fn new(analyzer: TextAnalyzer) -> Self {
        SynonymRuleTokenizer { analyzer }
    }
}

impl Tokenizer for SynonymRuleTokenizer {
    type TokenStream<'a> = PreTokenizedStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> PreTokenizedStream {
        let keys = match SynonymRule::parse(text) {
            Ok(Some(rule)) => match rule.compile(&mut self.analyzer) {
                Ok(Some(rule)) => rule.keys,
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        let tokens = keys
            .iter()
            .enumerate()
            .map(|(position, key)| Token {
                offset_from: 0,
                offset_to: text.len(),
                position,
                text: join_key(key),
                position_length: 1,
            })
            .collect();
        PreTokenizedStream::from(PreTokenizedString {
            text: text.to_string(),
            tokens,
        })
    }
}

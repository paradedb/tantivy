//! The tokenizer of a synonym rule field.

use super::rule::CompiledRule;
use crate::tokenizer::{PreTokenizedStream, PreTokenizedString, TextAnalyzer, Token, Tokenizer};

/// Joins the tokens of a key. A rule cannot contain it, so two keys never join to the same term.
pub(crate) const KEY_SEPARATOR: char = '\0';

/// Indexes a Solr synonym rule, such as `couch, sofa, divan` or `sneakers => running shoes`, as
/// one term per key, the analyzed tokens of the key joined with a zero byte.
///
/// A line that is not a valid rule gives no term, and its document is still added. Use
/// [`SynonymRule::check`](super::SynonymRule::check) to reject it instead.
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
        let keys = CompiledRule::from_line(text, &mut self.analyzer)
            .map(|rule| rule.keys)
            .unwrap_or_default();
        let separator = KEY_SEPARATOR.to_string();
        let tokens = keys
            .iter()
            .enumerate()
            .map(|(position, key)| Token {
                offset_from: 0,
                offset_to: text.len(),
                position,
                text: key.join(&separator),
                position_length: 1,
            })
            .collect();
        PreTokenizedStream::from(PreTokenizedString {
            text: text.to_string(),
            tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synonyms::SynonymRule;
    use crate::tokenizer::{SimpleTokenizer, StopWordFilter, TokenStream};

    /// The insert check and the tokenizer must agree. Otherwise a rule that passes the check is
    /// stored and never expands.
    #[test]
    fn a_rule_indexes_keys_exactly_when_the_check_accepts_it() {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(StopWordFilter::remove(vec!["the".to_string()]))
            .build();
        let mut tokenizer = SynonymRuleTokenizer::new(analyzer.clone());
        for rule in [
            "sofa, couch",
            "sofa",
            "sneakers => running shoes",
            "a => b => c",
            "the who => band",
            "who the => band",
            "the => sofa",
            " , ",
            "a\0b, c",
            "# a comment",
            "",
        ] {
            let accepted = match SynonymRule::parse(rule) {
                Ok(Some(parsed)) => parsed.check(&mut analyzer).is_ok(),
                _ => false,
            };
            let mut stream = tokenizer.token_stream(rule);
            let mut keys = 0;
            while stream.advance() {
                keys += 1;
            }
            assert_eq!(keys > 0, accepted, "{rule:?}");
        }
    }
}

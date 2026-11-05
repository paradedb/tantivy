//! Solr synonym rules, the format Elasticsearch and OpenSearch read from a synonym file.

use std::fmt;

use crate::tokenizer::TextAnalyzer;

/// Bounds the alternatives that one query term can expand to.
pub const MAX_RULE_TERMS: usize = 256;

/// One rule, split into its terms but not yet analyzed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SynonymRule {
    /// `a, b, c`. Every term expands to the whole group.
    Equivalent(Vec<String>),
    /// `a, b => c, d`. Every term on the left is replaced by the right side.
    Directed {
        /// The terms that are replaced.
        from: Vec<String>,
        /// The terms they are replaced with.
        to: Vec<String>,
    },
}

/// Why a line is not a Solr rule.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuleError {
    /// A side has more than [`MAX_RULE_TERMS`] terms.
    TooManyTerms,
    /// A side of `=>` has no term.
    EmptySide,
    /// The rule has more than one `=>`.
    TooManyArrows,
    /// The rule contains a zero byte, which joins the tokens of a key.
    ZeroByte,
    /// The rule has separators and no term.
    NoTerms,
    /// The analyzer drops a word of this term, such as a stopword, so the key would match more
    /// than the term does.
    DroppedWord(String),
}

impl fmt::Display for RuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleError::TooManyTerms => {
                write!(
                    f,
                    "a side of a rule can have at most {MAX_RULE_TERMS} terms"
                )
            }
            RuleError::EmptySide => write!(f, "both sides of '=>' need at least one term"),
            RuleError::TooManyArrows => write!(f, "a rule can have at most one '=>'"),
            RuleError::ZeroByte => write!(f, "a rule cannot contain a zero byte"),
            RuleError::NoTerms => write!(f, "a rule needs at least one term"),
            RuleError::DroppedWord(term) => {
                write!(f, "the analyzer drops a word of the term '{term}'")
            }
        }
    }
}

impl std::error::Error for RuleError {}

impl SynonymRule {
    /// Parses one line of a synonym file. A blank line or a `#` comment is `None`, so a file
    /// loads line by line.
    pub fn parse(rule: &str) -> Result<Option<SynonymRule>, RuleError> {
        let rule = rule.trim();
        if rule.is_empty() || rule.starts_with('#') {
            return Ok(None);
        }
        if rule.contains('\0') {
            return Err(RuleError::ZeroByte);
        }
        let mut sides = split_rule(rule);
        if sides.iter().any(|side| side.len() > MAX_RULE_TERMS) {
            return Err(RuleError::TooManyTerms);
        }
        match sides.len() {
            1 => {
                let terms = sides.pop().unwrap();
                if terms.is_empty() {
                    return Err(RuleError::NoTerms);
                }
                Ok(Some(SynonymRule::Equivalent(terms)))
            }
            2 => {
                let to = sides.pop().unwrap();
                let from = sides.pop().unwrap();
                if from.is_empty() || to.is_empty() {
                    return Err(RuleError::EmptySide);
                }
                Ok(Some(SynonymRule::Directed { from, to }))
            }
            _ => Err(RuleError::TooManyArrows),
        }
    }

    /// Checks the rule with the analyzer that the rule field's
    /// [`SynonymRuleTokenizer`](super::SynonymRuleTokenizer) wraps. The tokenizer indexes no key
    /// for an invalid rule, so a caller that wants the error checks the rule before it adds it.
    pub fn check(&self, analyzer: &mut TextAnalyzer) -> Result<(), RuleError> {
        self.compile(analyzer).map(drop)
    }

    /// Analyzes both sides of the rule. A rule whose alternatives all analyze to nothing, such as
    /// a right side made of stopwords, compiles to nothing, so the typed term stays instead of
    /// matching nothing.
    pub(crate) fn compile(
        &self,
        analyzer: &mut TextAnalyzer,
    ) -> Result<Option<CompiledRule>, RuleError> {
        let mut analyze = |terms: &[String]| -> Result<Vec<Vec<String>>, RuleError> {
            let mut analyzed: Vec<Vec<String>> = Vec::with_capacity(terms.len());
            for term in terms {
                let tokens = analyze_term(analyzer, term)?;
                if !tokens.is_empty() && !analyzed.contains(&tokens) {
                    analyzed.push(tokens);
                }
            }
            Ok(analyzed)
        };
        let (keys, alternatives) = match self {
            SynonymRule::Equivalent(terms) => {
                let group = analyze(terms)?;
                (group.clone(), group)
            }
            SynonymRule::Directed { from, to } => (analyze(from)?, analyze(to)?),
        };
        if keys.is_empty() || alternatives.is_empty() {
            return Ok(None);
        }
        Ok(Some(CompiledRule { keys, alternatives }))
    }
}

/// A rule after analysis. Every key of the rule expands to the same alternatives.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CompiledRule {
    pub(crate) keys: Vec<Vec<String>>,
    pub(crate) alternatives: Vec<Vec<String>>,
}

/// Splits a rule on `=>` and `,`. A backslash escapes the character after it, as in Solr, so a
/// term can contain a comma.
fn split_rule(rule: &str) -> Vec<Vec<String>> {
    fn push(side: &mut Vec<String>, term: &mut String) {
        let trimmed = term.trim();
        if !trimmed.is_empty() {
            side.push(trimmed.to_string());
        }
        term.clear();
    }

    let mut sides = vec![Vec::new()];
    let mut term = String::new();
    let mut chars = rule.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => term.push(chars.next().unwrap_or('\\')),
            ',' => push(sides.last_mut().unwrap(), &mut term),
            '=' if chars.peek() == Some(&'>') => {
                chars.next();
                push(sides.last_mut().unwrap(), &mut term);
                sides.push(Vec::new());
            }
            _ => term.push(c),
        }
    }
    push(sides.last_mut().unwrap(), &mut term);
    sides
}

/// A word the analyzer drops leaves a gap in the token positions. Lucene's `SynonymMap.Parser`
/// refuses such a term too, because its key would be shorter and match more.
fn analyze_term(analyzer: &mut TextAnalyzer, term: &str) -> Result<Vec<String>, RuleError> {
    let mut stream = analyzer.token_stream(term);
    let mut tokens = Vec::new();
    let mut next_position = 0;
    while stream.advance() {
        let token = stream.token();
        if token.position > next_position {
            return Err(RuleError::DroppedWord(term.to_string()));
        }
        next_position = token.position + 1;
        tokens.push(token.text.clone());
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{LowerCaser, SimpleTokenizer, StopWordFilter};

    fn strings(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|term| term.to_string()).collect()
    }

    #[test]
    fn parses_solr_rules() {
        assert_eq!(
            SynonymRule::parse("couch, sofa , divan").unwrap(),
            Some(SynonymRule::Equivalent(strings(&[
                "couch", "sofa", "divan"
            ])))
        );
        assert_eq!(
            SynonymRule::parse("sneakers => running shoes, trainers").unwrap(),
            Some(SynonymRule::Directed {
                from: strings(&["sneakers"]),
                to: strings(&["running shoes", "trainers"]),
            })
        );
        assert_eq!(SynonymRule::parse("  # a comment").unwrap(), None);
        assert_eq!(SynonymRule::parse("").unwrap(), None);
        assert_eq!(
            SynonymRule::parse(r"a\, b, c").unwrap(),
            Some(SynonymRule::Equivalent(strings(&["a, b", "c"])))
        );
        assert_eq!(
            SynonymRule::parse("a => b => c"),
            Err(RuleError::TooManyArrows)
        );
        assert_eq!(SynonymRule::parse("=> c"), Err(RuleError::EmptySide));
        assert_eq!(SynonymRule::parse("a\0b, c"), Err(RuleError::ZeroByte));
        assert_eq!(SynonymRule::parse(" , , "), Err(RuleError::NoTerms));
        assert_eq!(
            SynonymRule::parse(r"a \=> b, c\").unwrap(),
            Some(SynonymRule::Equivalent(strings(&["a => b", r"c\"])))
        );
        let long = (0..=MAX_RULE_TERMS)
            .map(|i| format!("t{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(SynonymRule::parse(&long), Err(RuleError::TooManyTerms));
    }

    #[test]
    fn compiles_the_keys_and_their_alternatives() {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(vec!["the".to_string()]))
            .build();
        let mut compile = |rule: &str| {
            SynonymRule::parse(rule)
                .unwrap()
                .unwrap()
                .compile(&mut analyzer)
                .unwrap()
        };
        let compiled = |keys: &[&[&str]], alternatives: &[&[&str]]| {
            Some(CompiledRule {
                keys: keys.iter().map(|key| strings(key)).collect(),
                alternatives: alternatives.iter().map(|alt| strings(alt)).collect(),
            })
        };
        assert_eq!(
            compile("Couch, sofa, SOFA"),
            compiled(&[&["couch"], &["sofa"]], &[&["couch"], &["sofa"]])
        );
        assert_eq!(
            compile("sneakers, trainers => running shoes"),
            compiled(&[&["sneakers"], &["trainers"]], &[&["running", "shoes"]])
        );
        // A key that only maps to itself still compiles, since another rule can add to it.
        assert_eq!(compile("sofa"), compiled(&[&["sofa"]], &[&["sofa"]]));
        assert_eq!(compile("sofa, the"), compiled(&[&["sofa"]], &[&["sofa"]]));
        assert_eq!(compile("sofa => the"), None);
        assert_eq!(compile("the => sofa"), None);
    }

    #[test]
    fn a_term_that_loses_a_word_to_the_analyzer_does_not_compile() {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(StopWordFilter::remove(vec!["the".to_string()]))
            .build();
        for rule in [
            "the who => band",
            "band => the who",
            "rock of the ages, rock",
        ] {
            let compiled = SynonymRule::parse(rule)
                .unwrap()
                .unwrap()
                .compile(&mut analyzer);
            assert!(
                matches!(compiled, Err(RuleError::DroppedWord(_))),
                "{rule}: {compiled:?}"
            );
        }
    }
}

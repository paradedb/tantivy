use std::collections::HashMap;
use std::sync::Arc;

use super::{Token, TokenFilter, TokenStream, Tokenizer};

/// Represents different types of synonym mappings.
/// 
/// Based on Lucence's synonym format:
/// - `Equivalent`: Bidirectional synonyms where all terms are equivalent
/// - `Explicit`: One-way mapping from source terms to target term
#[derive(Debug, Clone, PartialEq)]
pub enum SynonymRule {
    /// Equivalent synonyms (bidirectional).
    /// Example: "dog,puppy,hound" - any of these terms will expand to all terms
    Equivalent(Vec<String>),
    
    /// Explicit mapping (one-way).
    /// Example: "teh,hte => the" - left side terms map only to right side term
    Explicit {
        /// Source terms that trigger the mapping
        from: Vec<String>,
        /// Target term to emit
        to: String,
    },
}

impl SynonymRule {
    /// Creates an equivalent synonym rule from a list of terms.
    pub fn equivalent<S: AsRef<str>>(terms: Vec<S>) -> Self {
        SynonymRule::Equivalent(
            terms.into_iter().map(|s| s.as_ref().to_string()).collect()
        )
    }
    
    /// Creates an explicit mapping rule.
    pub fn explicit<S: AsRef<str>>(from: Vec<S>, to: S) -> Self {
        SynonymRule::Explicit {
            from: from.into_iter().map(|s| s.as_ref().to_string()).collect(),
            to: to.as_ref().to_string(),
        }
    }
    
    /// Parses a synonym rule from Lucence-style string format.
    /// 
    /// Examples:
    /// - "dog,puppy,hound" -> Equivalent rule
    /// - "teh,hte => the" -> Explicit rule
    pub fn parse(rule_str: &str) -> Result<Self, String> {
        if let Some((left, right)) = rule_str.split_once(" => ") {
            // Explicit mapping
            let from: Vec<String> = left.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let to = right.trim().to_string();
            
            if from.is_empty() || to.is_empty() {
                return Err("Invalid explicit mapping format".to_string());
            }
            
            Ok(SynonymRule::Explicit { from, to })
        } else {
            // Equivalent synonyms
            let terms: Vec<String> = rule_str.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            if terms.len() < 2 {
                return Err("Equivalent synonyms must have at least 2 terms".to_string());
            }
            
            Ok(SynonymRule::Equivalent(terms))
        }
    }
}

/// `TokenFilter` that expands tokens with their synonyms.
/// 
/// Supports both equivalent (bidirectional) and explicit (one-way) synonym mappings,
/// following Lucence's synonym format.
/// 
/// # Example
/// ```rust
/// use tantivy::tokenizer::*;
/// 
/// let rules = vec![
///     SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
///     SynonymRule::explicit(vec!["automobile", "car"], "vehicle"),
/// ];
/// 
/// let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
///     .filter(SynonymFilter::new(rules))
///     .build();
/// ```
#[derive(Clone)]
pub struct SynonymFilter {
    // Maps each word to its complete synonym set (including itself)
    synonym_map: Arc<HashMap<String, Arc<Vec<String>>>>,
}

impl SynonymFilter {
    /// Creates a new `SynonymFilter` from synonym rules.
    pub fn new(rules: Vec<SynonymRule>) -> Self {
        let mut synonym_map = HashMap::new();
        
        for rule in rules {
            match rule {
                SynonymRule::Equivalent(terms) => {
                    // Create shared synonym set
                    let shared_terms = Arc::new(terms);
                    // Each term maps to the shared set
                    for term in shared_terms.iter() {
                        synonym_map.insert(term.clone(), shared_terms.clone());
                    }
                }
                SynonymRule::Explicit { from, to } => {
                    // Each source term maps only to the target term
                    let target_vec = Arc::new(vec![to]);
                    for term in &from {
                        synonym_map.insert(term.clone(), target_vec.clone());
                    }
                }
            }
        }
        
        SynonymFilter {
            synonym_map: Arc::new(synonym_map),
        }
    }
    
    /// Creates a new `SynonymFilter` from Lucence-style rule strings.
    /// 
    /// # Example
    /// ```rust
    /// use tantivy::tokenizer::*;
    /// 
    /// let rules = vec![
    ///     "dog,puppy,hound",           // Equivalent synonyms
    ///     "teh,hte => the",        // Explicit mapping
    /// ];
    /// 
    /// let filter = SynonymFilter::from_strings(rules).unwrap();
    /// ```
    pub fn from_strings<S: AsRef<str>>(rule_strings: Vec<S>) -> Result<Self, String> {
        let mut rules = Vec::new();
        
        for rule_str in rule_strings {
            let rule = SynonymRule::parse(rule_str.as_ref())?;
            rules.push(rule);
        }
        
        Ok(Self::new(rules))
    }
}

impl TokenFilter for SynonymFilter {
    type Tokenizer<T: Tokenizer> = SynonymFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> SynonymFilterWrapper<T> {
        SynonymFilterWrapper {
            synonym_map: self.synonym_map,
            inner: tokenizer,
        }
    }
}

#[derive(Clone)]
pub struct SynonymFilterWrapper<T> {
    synonym_map: Arc<HashMap<String, Arc<Vec<String>>>>,
    inner: T,
}

impl<T: Tokenizer> Tokenizer for SynonymFilterWrapper<T> {
    type TokenStream<'a> = SynonymFilterStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        SynonymFilterStream {
            synonym_map: self.synonym_map.clone(),
            tail: self.inner.token_stream(text),
            current_synonyms: None,
            synonym_index: 0,
            current_token: None,
        }
    }
}

pub struct SynonymFilterStream<T> {
    synonym_map: Arc<HashMap<String, Arc<Vec<String>>>>,
    tail: T,
    current_synonyms: Option<Arc<Vec<String>>>,  // Shared reference to synonym list
    synonym_index: usize,                        // Current position in synonym list
    current_token: Option<Token>,
}

impl<T: TokenStream> TokenStream for SynonymFilterStream<T> {
    fn advance(&mut self) -> bool {
        // If we have pending synonyms to emit, emit the next one
        if let Some(ref synonyms) = self.current_synonyms {
            if self.synonym_index < synonyms.len() {
                if let Some(ref mut token) = self.current_token {
                    token.text = synonyms[self.synonym_index].clone();
                    self.synonym_index += 1;
                    return true;
                }
            }
            // Done with current synonym set
            self.current_synonyms = None;
        }

        // Get the next token from the underlying stream
        if !self.tail.advance() {
            return false;
        }

        // Clone the current token
        self.current_token = Some(self.tail.token().clone());
        
        // Check if this token has synonyms
        if let Some(ref token) = self.current_token {
            if let Some(synonyms) = self.synonym_map.get(&token.text) {
                // We found synonyms! Set up to emit all of them
                self.current_synonyms = Some(synonyms.clone()); // Rc clone (cheap)
                self.synonym_index = 0;
                // The first synonym becomes the current token
                if !synonyms.is_empty() {
                    if let Some(ref mut current_token) = self.current_token {
                        current_token.text = synonyms[0].clone();
                        self.synonym_index = 1;
                    }
                }
            }
        }

        true
    }

    fn token(&self) -> &Token {
        self.current_token.as_ref().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.current_token.as_mut().unwrap_or_else(|| self.tail.token_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SimpleTokenizer, TextAnalyzer, Token};

    #[test]
    fn test_synonym_rule_parsing() {
        // Test equivalent synonyms
        let rule = SynonymRule::parse("dog,puppy,hound").unwrap();
        assert_eq!(rule, SynonymRule::Equivalent(vec![
            "dog".to_string(), 
            "puppy".to_string(), 
            "hound".to_string()
        ]));
        
        // Test explicit mapping
        let rule = SynonymRule::parse("teh,hte => the").unwrap();
        assert_eq!(rule, SynonymRule::Explicit {
            from: vec!["teh".to_string(), "hte".to_string()],
            to: "the".to_string(),
        });
        
        // Test error cases
        assert!(SynonymRule::parse("single").is_err());
        assert!(SynonymRule::parse(" => the").is_err());
        assert!(SynonymRule::parse("teh => ").is_err());
    }

    #[test]
    fn test_equivalent_synonym_rules() {
        let rules = vec![
            SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
            SynonymRule::equivalent(vec!["cat", "kitten"]),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "dog");
        
        // Should emit all synonyms for "dog"
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "dog");
        assert_eq!(tokens[1].text, "puppy");
        assert_eq!(tokens[2].text, "hound");
    }

    #[test]
    fn test_explicit_synonym_rules() {
        let rules = vec![
            SynonymRule::explicit(vec!["automobile", "car"], "vehicle"),
            SynonymRule::explicit(vec!["teh", "hte"], "the"),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "automobile");
        
        // Should emit only the target term
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "vehicle");
        
        let tokens = token_stream_helper(&mut analyzer, "car");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "vehicle");
    }

    #[test]
    fn test_mixed_synonym_rules() {
        let rules = vec![
            SynonymRule::equivalent(vec!["cat", "kitty"]),           // Bidirectional
            SynonymRule::explicit(vec!["automobile"], "car"),        // One-way
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        // Test equivalent synonyms
        let tokens = token_stream_helper(&mut analyzer, "cat");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "cat");
        assert_eq!(tokens[1].text, "kitty");
        
        // Test explicit mapping
        let tokens = token_stream_helper(&mut analyzer, "automobile");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "car");
        
        // Test that target doesn't map back in explicit rules
        let tokens = token_stream_helper(&mut analyzer, "car");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "car"); // No expansion
    }

    #[test]
    fn test_from_strings() {
        let rule_strings = vec![
            "cat,kitty,puss",              // Equivalent
            "automobile,car => vehicle",   // Explicit  
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::from_strings(rule_strings).unwrap())
            .build();

        // Test equivalent
        let tokens = token_stream_helper(&mut analyzer, "cat");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().any(|t| t.text == "cat"));
        assert!(tokens.iter().any(|t| t.text == "kitty"));
        assert!(tokens.iter().any(|t| t.text == "puss"));
        
        // Test explicit
        let tokens = token_stream_helper(&mut analyzer, "automobile");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "vehicle");
    }

    #[test]
    fn test_new_method() {
        let rules = vec![
            SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
            SynonymRule::equivalent(vec!["cat", "kitten"]),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "dog");
        
        // Should emit all synonyms for "dog"
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "dog");
        assert_eq!(tokens[1].text, "puppy");
        assert_eq!(tokens[2].text, "hound");
    }

    #[test]
    fn test_synonym_with_regular_words() {
        let rules = vec![
            SynonymRule::equivalent(vec!["cat", "kitty"]),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "the cat runs");
        
        // Should be: "the", "cat", "kitty", "runs"
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "the");
        assert_eq!(tokens[1].text, "cat");
        assert_eq!(tokens[2].text, "kitty");
        assert_eq!(tokens[3].text, "runs");
    }

    #[test]
    fn test_no_synonyms() {
        let rules = vec![
            SynonymRule::equivalent(vec!["cat", "kitty"]),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "dog runs");
        
        // Should remain unchanged
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "dog");
        assert_eq!(tokens[1].text, "runs");
    }

    #[test]
    fn test_synonym_positions_and_offsets() {
        let rules = vec![
            SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
        ];

        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(SynonymFilter::new(rules))
            .build();

        let tokens = token_stream_helper(&mut analyzer, "the dog runs");
        
        // Should be: "the"(pos:0), "dog"(pos:1), "puppy"(pos:1), "hound"(pos:1), "runs"(pos:2)
        assert_eq!(tokens.len(), 5);
        
        // Check "the" token
        assert_eq!(tokens[0].text, "the");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].offset_from, 0);
        assert_eq!(tokens[0].offset_to, 3);
        
        // Check all synonym tokens have same position and offset
        assert_eq!(tokens[1].text, "dog");
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[1].offset_from, 4);
        assert_eq!(tokens[1].offset_to, 7);
        
        assert_eq!(tokens[2].text, "puppy");
        assert_eq!(tokens[2].position, 1);  // Same position as "dog"
        assert_eq!(tokens[2].offset_from, 4);  // Same offset as "dog"  
        assert_eq!(tokens[2].offset_to, 7);    // Same offset as "dog"
        
        assert_eq!(tokens[3].text, "hound");
        assert_eq!(tokens[3].position, 1);  // Same position as "dog"
        assert_eq!(tokens[3].offset_from, 4);  // Same offset as "dog"
        assert_eq!(tokens[3].offset_to, 7);    // Same offset as "dog"
        
        // Check "runs" token  
        assert_eq!(tokens[4].text, "runs");
        assert_eq!(tokens[4].position, 2);
        assert_eq!(tokens[4].offset_from, 8);
        assert_eq!(tokens[4].offset_to, 12);
    }

    fn token_stream_helper(analyzer: &mut TextAnalyzer, text: &str) -> Vec<Token> {
        let mut token_stream = analyzer.token_stream(text);
        let mut tokens: Vec<Token> = vec![];
        let mut add_token = |token: &Token| {
            tokens.push(token.clone());
        };
        token_stream.process(&mut add_token);
        tokens
    }
}

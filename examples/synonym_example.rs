use tantivy::tokenizer::*;

fn main() {
    println!("=== Synonym Filter Examples ===\n");
    
    // Example 1: Using SynonymRule for mixed synonym types
    println!("Example 1: Mixed synonym types using SynonymRule");
    let rules = vec![
        // Equivalent synonyms (bidirectional)
        SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
        SynonymRule::equivalent(vec!["cat", "kitten", "feline"]),
        
        // Explicit mappings (one-way)
        SynonymRule::explicit(vec!["automobile", "car"], "vehicle"),
        SynonymRule::explicit(vec!["teh", "hte"], "the"),
    ];

    let mut analyzer1 = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)  // Lowercase first for consistent matching
        .filter(SynonymFilter::new(rules))
        .build();

    let test_cases1 = vec![
        "I have a dog",           // Should expand to dog, puppy, hound
        "The automobile is fast", // Should map to vehicle only
        "I saw teh dog run",     // teh->the
    ];

    for text in test_cases1 {
        println!("\n  Input: '{}'", text);
        println!("  Tokens:");
        
        let tokens = get_tokens(&mut analyzer1, text);
        for (i, token) in tokens.iter().enumerate() {
            println!("    {}: '{}'", i, token.text);
        }
    }

    // Example 2: Using string parsing (Elasticsearch format)
    println!("\n\nExample 2: Using Elasticsearch-style string format");
    let rule_strings = vec![
        "dog,puppy,hound",              // Equivalent synonyms
        "happy,joyful,glad",           // Equivalent synonyms  
        "automobile,car => vehicle",   // Explicit mapping
        "teh,hte => the",         // Explicit mapping
    ];

    let mut analyzer2 = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(SynonymFilter::from_strings(rule_strings).unwrap())
        .build();

    let test_cases2 = vec![
        "Happy dog in the car",    // happy->all, dog->all, car->vehicle
        "I wrote teh email",       // teh->the, misspelling correction
    ];

    for text in test_cases2 {
        println!("\n  Input: '{}'", text);
        println!("  Tokens:");
        
        let tokens = get_tokens(&mut analyzer2, text);
        for (i, token) in tokens.iter().enumerate() {
            println!("    {}: '{}'", i, token.text);
        }
    }

    // Example 3: Equivalent synonyms using SynonymRule
    println!("\n\nExample 3: Equivalent synonyms using SynonymRule");
    let equiv_rules = vec![
        SynonymRule::equivalent(vec!["dog", "puppy", "hound"]),
        SynonymRule::equivalent(vec!["cat", "kitten", "feline"]),
    ];

    let mut analyzer3 = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(SynonymFilter::new(equiv_rules))
        .build();

    let tokens = get_tokens(&mut analyzer3, "The dog chased the cat");
    println!("\n  Input: 'The dog chased the cat'");
    println!("  Tokens:");
    for (i, token) in tokens.iter().enumerate() {
        println!("    {}: '{}'", i, token.text);
    }
    
    println!("\n=== Key Differences ===");
    println!("• Equivalent synonyms: bidirectional (any -> all)");
    println!("• Explicit mappings: one-way (source terms -> target term only)");
    println!("• String format follows Elasticsearch conventions");
    println!("• Only SynonymRule-based API is supported (no legacy Vec<Vec<String>>)");
}

fn get_tokens(analyzer: &mut TextAnalyzer, text: &str) -> Vec<Token> {
    let mut token_stream = analyzer.token_stream(text);
    let mut tokens = Vec::new();
    let mut add_token = |token: &Token| {
        tokens.push(token.clone());
    };
    token_stream.process(&mut add_token);
    tokens
}
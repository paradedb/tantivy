use crate::core::searcher::Searcher;
use crate::query::{EnableScoring, Explanation, Query, QueryClone, Scorer, Weight};
use crate::schema::Term;
use crate::{DocId, DocSet, Result, Score, SegmentReader, TERMINATED};
use common::BitSet;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreMode {
    Avg,
    Max,
    Total,
}

impl Default for ScoreMode {
    fn default() -> Self {
        ScoreMode::Avg
    }
}

pub struct NestedDocumentQuery {
    child_query: Box<dyn Query>,
    parents_filter: Box<dyn Query>,
    score_mode: ScoreMode,
}

impl Clone for NestedDocumentQuery {
    fn clone(&self) -> Self {
        NestedDocumentQuery {
            child_query: self.child_query.box_clone(),
            parents_filter: self.parents_filter.box_clone(),
            score_mode: self.score_mode,
        }
    }
}

impl NestedDocumentQuery {
    pub fn new(
        child_query: Box<dyn Query>,
        parents_filter: Box<dyn Query>,
        score_mode: ScoreMode,
    ) -> NestedDocumentQuery {
        NestedDocumentQuery {
            child_query,
            parents_filter,
            score_mode,
        }
    }
}

impl fmt::Debug for NestedDocumentQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NestedDocumentQuery(child_query: {:?}, parents_filter: {:?}, score_mode: {:?})",
            self.child_query, self.parents_filter, self.score_mode
        )
    }
}

impl Query for NestedDocumentQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        println!("Creating weights for NestedDocumentQuery");
        let child_weight = self.child_query.weight(enable_scoring)?;
        let parents_weight = self.parents_filter.weight(enable_scoring)?;

        Ok(Box::new(NestedDocumentWeight {
            child_weight,
            parents_weight,
            score_mode: self.score_mode,
        }))
    }

    fn explain(
        &self,
        _searcher: &Searcher,
        _doc_address: crate::DocAddress,
    ) -> Result<Explanation> {
        unimplemented!("Explain is not implemented for NestedDocumentQuery");
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        let weight = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut total_count = 0;
        for reader in searcher.segment_readers() {
            total_count += weight.count(reader)?;
        }
        Ok(total_count as usize)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.child_query.query_terms(visitor);
        self.parents_filter.query_terms(visitor);
    }
}

struct NestedDocumentWeight {
    child_weight: Box<dyn Weight>,
    parents_weight: Box<dyn Weight>,
    score_mode: ScoreMode,
}

impl Weight for NestedDocumentWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
        println!("\n=== Creating NestedDocumentScorer for segment ===");
        println!("Max doc in segment: {}", reader.max_doc());

        // Create parents bitset
        let mut parents_bitset = BitSet::with_max_value(reader.max_doc());

        println!("Building parents bitset with boost: {}", boost);
        let mut parents_scorer = self.parents_weight.scorer(reader, boost)?;

        // Collect all parent documents
        let mut found_parent = false;
        while parents_scorer.doc() != TERMINATED {
            let parent_doc = parents_scorer.doc();
            parents_bitset.insert(parent_doc);
            println!(
                "Found parent doc: {} (total parents: {})",
                parent_doc,
                parents_bitset.len()
            );
            found_parent = true;
            parents_scorer.advance();
        }

        // If no parents in this segment, return empty scorer
        if !found_parent {
            println!("No parents found in segment, returning empty scorer");
            return Ok(Box::new(EmptyScorer));
        }

        println!("Total parent docs found: {}", parents_bitset.len());

        // Get child scorer
        let child_scorer = self.child_weight.scorer(reader, boost)?;

        Ok(Box::new(NestedDocumentScorer {
            child_scorer,
            parent_docs: parents_bitset,
            score_mode: self.score_mode,
            current_parent: 0,
            current_score: 0.0,
            initialized: false,
            has_more: true,
        }))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> Result<Explanation> {
        unimplemented!("Explain is not implemented for NestedDocumentWeight");
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        let mut count = 0;
        let mut scorer = self.scorer(reader, 1.0)?;
        while scorer.doc() != TERMINATED {
            count += 1;
            scorer.advance();
        }
        Ok(count)
    }
}

struct EmptyScorer;

impl DocSet for EmptyScorer {
    fn advance(&mut self) -> DocId {
        TERMINATED
    }

    fn doc(&self) -> DocId {
        TERMINATED
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for EmptyScorer {
    fn score(&mut self) -> Score {
        0.0
    }
}

struct NestedDocumentScorer {
    child_scorer: Box<dyn Scorer>,
    parent_docs: BitSet,
    score_mode: ScoreMode,
    current_parent: DocId,
    current_score: Score,
    initialized: bool,
    has_more: bool,
}

impl DocSet for NestedDocumentScorer {
    fn advance(&mut self) -> DocId {
        if !self.initialized {
            println!("\n=== Initializing NestedDocumentScorer ===");
            self.initialized = true;
            self.current_parent = self.find_next_parent(0);
            if self.current_parent == TERMINATED {
                self.has_more = false;
                return TERMINATED;
            }
            return self.collect_matches();
        }

        if !self.has_more {
            return TERMINATED;
        }

        self.current_parent = self.find_next_parent(self.current_parent + 1);
        if self.current_parent == TERMINATED {
            self.has_more = false;
            return TERMINATED;
        }

        self.collect_matches()
    }

    fn doc(&self) -> DocId {
        if self.has_more {
            self.current_parent
        } else {
            TERMINATED
        }
    }

    fn size_hint(&self) -> u32 {
        self.parent_docs.len() as u32
    }
}

impl NestedDocumentScorer {
    fn collect_matches(&mut self) -> DocId {
        println!(
            "\n>>> Collecting matches for parent: {}",
            self.current_parent
        );

        // Initialize child scorer if not already initialized
        let mut child_doc = if !self.initialized {
            println!(">>> First time initialization of child scorer");
            let doc = self.child_scorer.advance();
            println!(">>> Initial child doc: {}", doc);
            self.initialized = true;
            doc
        } else {
            let doc = self.child_scorer.doc();
            println!(">>> Using existing child doc: {}", doc);
            doc
        };

        println!(">>> Current child doc: {}", child_doc);
        println!(">>> Current parent: {}", self.current_parent);
        println!(">>> Has more: {}", self.has_more);

        let mut child_scores = Vec::new();

        // Find all valid children for this parent
        while child_doc != TERMINATED && child_doc < self.current_parent {
            println!(
                ">>> Examining child doc {} for parent {}",
                child_doc, self.current_parent
            );

            // Check if this is a valid child
            let mut is_valid = true;
            for doc_id in (child_doc + 1)..self.current_parent {
                if self.parent_docs.contains(doc_id) {
                    println!(
                        ">>> Found intervening parent {} between child {} and parent {}",
                        doc_id, child_doc, self.current_parent
                    );
                    is_valid = false;
                    break;
                }
            }

            if is_valid {
                let score = self.child_scorer.score();
                println!(
                    ">>> Child {} is valid for parent {} with score {}",
                    child_doc, self.current_parent, score
                );
                child_scores.push(score);
            } else {
                println!(
                    ">>> Child {} is not valid for parent {}",
                    child_doc, self.current_parent
                );
            }

            child_doc = self.child_scorer.advance();
            println!(">>> Advanced child scorer to: {}", child_doc);
        }

        if !child_scores.is_empty() {
            println!(
                ">>> Found {} valid children for parent {}",
                child_scores.len(),
                self.current_parent
            );

            self.current_score = match self.score_mode {
                ScoreMode::Avg => {
                    let avg = child_scores.iter().sum::<Score>() / child_scores.len() as Score;
                    println!(
                        ">>> Average score for parent {}: {}",
                        self.current_parent, avg
                    );
                    avg
                }
                ScoreMode::Max => {
                    let max = child_scores.iter().cloned().fold(f32::MIN, f32::max);
                    println!(">>> Max score for parent {}: {}", self.current_parent, max);
                    max
                }
                ScoreMode::Total => {
                    let sum = child_scores.iter().sum();
                    println!(
                        ">>> Total score for parent {}: {}",
                        self.current_parent, sum
                    );
                    sum
                }
            };

            println!(
                ">>> Returning parent {} with score {}",
                self.current_parent, self.current_score
            );
            self.current_parent
        } else {
            println!(
                ">>> No valid children for parent {}, advancing",
                self.current_parent
            );
            // No children found, try next parent
            self.advance()
        }
    }

    fn advance(&mut self) -> DocId {
        if !self.initialized {
            println!(">>> Initial advance - not yet initialized");
            self.initialized = true;
            self.current_parent = self.find_next_parent(0);

            if self.current_parent == TERMINATED {
                println!(">>> No initial parent found, terminating");
                self.has_more = false;
                return TERMINATED;
            }

            println!(">>> Found initial parent: {}", self.current_parent);
            return self.collect_matches();
        }

        if !self.has_more {
            println!(">>> No more documents to process");
            return TERMINATED;
        }

        println!(
            ">>> Advancing from parent {} to next parent",
            self.current_parent
        );
        self.current_parent = self.find_next_parent(self.current_parent + 1);

        if self.current_parent == TERMINATED {
            println!(">>> No more parents found, terminating");
            self.has_more = false;
            return TERMINATED;
        }

        println!(">>> Found next parent: {}", self.current_parent);
        self.collect_matches()
    }

    fn find_next_parent(&self, from: DocId) -> DocId {
        println!(">>> Looking for next parent starting from {}", from);
        let mut current = from;
        while current < self.parent_docs.max_value() {
            if self.parent_docs.contains(current) {
                println!(">>> Found next parent: {}", current);
                return current;
            }
            current += 1;
        }
        println!(">>> No more parents found after {}", from);
        TERMINATED
    }
}

impl Scorer for NestedDocumentScorer {
    fn score(&mut self) -> Score {
        self.current_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocs;
    use crate::query::{BooleanQuery, Occur, RangeQuery, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, INDEXED, STORED, STRING};
    use crate::Term;
    use crate::{Index, Result};
    use std::ops::Bound;

    #[test]
    fn test_nested_document_query() -> Result<()> {
        println!("\n=== Starting test_nested_document_query ===");

        // Create schema
        let mut schema_builder = Schema::builder();
        let name = schema_builder.add_text_field("name", STORED);
        let country = schema_builder.add_text_field("country", STRING | STORED);
        let doc_type = schema_builder.add_text_field("doc_type", STRING | STORED);
        let skill = schema_builder.add_text_field("skill", STRING | STORED);
        let year = schema_builder.add_u64_field("year", STORED | INDEXED);
        let schema = schema_builder.build();

        println!("Created schema");

        // Create index
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(50_000_000)?;

        println!("Created index and writer");

        // Add documents in the correct order: children before parents
        writer.add_document(doc!(
            skill => "java",
            year => 2006u64,
        ))?;

        writer.add_document(doc!(
            skill => "python",
            year => 2010u64,
        ))?;

        writer.add_document(doc!(
            name => "Lisa",
            country => "United Kingdom",
            doc_type => "resume",
        ))?;

        writer.add_document(doc!(
            skill => "ruby",
            year => 2005u64,
        ))?;

        writer.add_document(doc!(
            skill => "java",
            year => 2007u64,
        ))?;

        writer.add_document(doc!(
            name => "Frank",
            country => "United States",
            doc_type => "resume",
        ))?;

        println!("Added all documents");
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        println!("Created reader and searcher");

        // Verify document count
        let total_docs: u32 = searcher
            .segment_readers()
            .iter()
            .map(|reader| reader.num_docs())
            .sum();
        println!("Total documents in index: {}", total_docs);
        assert_eq!(total_docs, 6, "Should have 6 documents total");

        // Create parent query
        let parent_query = TermQuery::new(
            Term::from_field_text(doc_type, "resume"),
            IndexRecordOption::Basic,
        );

        // Test parent query
        let parent_results = searcher.search(&parent_query, &TopDocs::with_limit(10))?;
        println!("Parent query found {} results", parent_results.len());
        assert_eq!(parent_results.len(), 2, "Should find 2 parent documents");

        // Create child query
        let child_query = BooleanQuery::new(vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(skill, "java"),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(RangeQuery::new(
                    Bound::Included(Term::from_field_u64(year, 2006u64)),
                    Bound::Included(Term::from_field_u64(year, 2011u64)),
                )),
            ),
        ]);

        // Test child query
        let child_results = searcher.search(&child_query, &TopDocs::with_limit(10))?;
        println!("Child query found {} results", child_results.len());
        assert_eq!(child_results.len(), 2, "Should find 2 child documents");

        // Create nested query
        let nested_query = NestedDocumentQuery::new(
            Box::new(child_query),
            Box::new(parent_query),
            ScoreMode::Avg,
        );

        // Test nested query
        let nested_results = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        println!("Nested query found {} results", nested_results.len());
        assert_eq!(
            nested_results.len(),
            2,
            "Should find 2 nested document matches"
        );

        Ok(())
    }
}

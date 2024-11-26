use common::BitSet;
use std::fmt;

use crate::error::TantivyError;
use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::DocId;
use crate::Score;
use crate::SegmentReader;
use crate::{DocSet, Result, TERMINATED};

/// Score mode for how to combine child document scores
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreMode {
    /// Use the average of all child document scores
    Avg,
    /// Use the maximum score from child documents
    Max,
    /// Use the sum of all child document scores
    Total,
}

/// Query that remaps matches on child documents to their parent document.
/// Parent documents are identified using a Filter.
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
            score_mode: self.score_mode.clone(),
        }
    }
}

impl NestedDocumentQuery {
    /// Create a new NestedDocumentQuery
    #[allow(dead_code)]
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
        let child_weight = self.child_query.weight(enable_scoring)?;
        let parent_weight = self.parents_filter.weight(enable_scoring)?;

        Ok(Box::new(NestedDocumentWeight {
            child_weight,
            parent_weight,
            score_mode: self.score_mode,
        }))
    }
}

struct NestedDocumentWeight {
    child_weight: Box<dyn Weight>,
    parent_weight: Box<dyn Weight>,
    score_mode: ScoreMode,
}

impl Weight for NestedDocumentWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        // Get parent document filter
        let mut parent_scorer = self.parent_weight.scorer(reader, boost)?;
        let mut parent_docs = BitSet::with_max_value(reader.max_doc());

        let mut doc = parent_scorer.doc();
        while doc != TERMINATED {
            parent_docs.insert(doc);
            doc = parent_scorer.advance();
        }

        // Get child document scorer
        let child_scorer = self.child_weight.scorer(reader, boost)?;

        Ok(Box::new(NestedDocumentScorer {
            child_scorer,
            parent_docs,
            score_mode: self.score_mode,
            current_doc: TERMINATED,
            current_score: 0.0,
        }))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(
                "No matching parent document found".to_string(),
            ));
        }
        Ok(Explanation::new("NestedDocumentQuery", 1.0))
    }
}

struct NestedDocumentScorer {
    child_scorer: Box<dyn Scorer>,
    parent_docs: BitSet,
    score_mode: ScoreMode,
    current_doc: DocId,
    current_score: Score,
}

impl DocSet for NestedDocumentScorer {
    fn advance(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }

        loop {
            let child_doc = self.child_scorer.doc();
            if child_doc == TERMINATED {
                self.current_doc = TERMINATED;
                return TERMINATED;
            }

            let parent_doc = find_prev_set_bit(&self.parent_docs, child_doc);
            if parent_doc == TERMINATED {
                let _ = self.child_scorer.advance();
                continue;
            }

            // Collect all child scores for this parent_doc
            let mut scores = Vec::new();
            let mut explanations = Vec::new();

            let current_score = self.child_scorer.score();
            scores.push(current_score);
            explanations.push(Explanation::new("Child match", current_score));

            // Advance to next child
            let next_doc = self.child_scorer.advance();

            // Collect all children that belong to the same parent_doc
            let mut temp_doc = next_doc;
            while temp_doc != TERMINATED {
                let next_parent_doc = find_prev_set_bit(&self.parent_docs, temp_doc);
                if next_parent_doc != parent_doc {
                    break;
                }
                let current_score = self.child_scorer.score();
                scores.push(current_score);
                explanations.push(Explanation::new("Child match", current_score));
                temp_doc = self.child_scorer.advance();
            }

            // Calculate aggregate score based on mode
            self.current_score = match self.score_mode {
                ScoreMode::Avg => {
                    let sum: Score = scores.iter().sum();
                    sum / scores.len() as Score
                }
                ScoreMode::Max => *scores
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                ScoreMode::Total => scores.iter().sum(),
            };

            self.current_doc = parent_doc;
            return self.current_doc;
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.child_scorer.size_hint()
    }
}

impl Scorer for NestedDocumentScorer {
    fn score(&mut self) -> Score {
        self.current_score
    }
}

// Helper function to find previous set bit in BitSet
fn find_prev_set_bit(bitset: &BitSet, doc: DocId) -> DocId {
    let mut current = doc;
    while current > 0 {
        if bitset.contains(current) {
            return current;
        }
        current -= 1;
    }
    if bitset.contains(0) {
        0
    } else {
        TERMINATED
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use super::*;
    use crate::collector::TopDocs;
    use crate::query::{BooleanQuery, Occur, RangeQuery, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Value, STORED, TEXT};
    use crate::Term;
    use crate::{Index, TantivyDocument};

    #[test]
    fn test_nested_document_query() -> Result<()> {
        let mut schema_builder = Schema::builder();
        let name = schema_builder.add_text_field("name", STORED);
        let country = schema_builder.add_text_field("country", TEXT);
        let doc_type = schema_builder.add_text_field("doc_type", TEXT);
        let skill = schema_builder.add_text_field("skill", TEXT);
        let year = schema_builder.add_u64_field("year", STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(50_000_000)?;

        // Add resume documents with nested job experiences
        {
            let mut doc: TantivyDocument = Default::default();
            doc.add_text(name, "Lisa");
            doc.add_text(country, "United Kingdom");
            doc.add_text(doc_type, "resume");
            writer.add_document(doc)?;

            let mut job1: TantivyDocument = Default::default();
            job1.add_text(skill, "java");
            job1.add_u64(year, 2006);
            writer.add_document(job1)?;

            let mut job2: TantivyDocument = Default::default();
            job2.add_text(skill, "python");
            job2.add_u64(year, 2010);
            writer.add_document(job2)?;
        }

        {
            let mut doc: TantivyDocument = Default::default();
            doc.add_text(name, "Frank");
            doc.add_text(country, "United States");
            doc.add_text(doc_type, "resume");
            writer.add_document(doc)?;

            let mut job1: TantivyDocument = Default::default();
            job1.add_text(skill, "ruby");
            job1.add_u64(year, 2005);
            writer.add_document(job1)?;

            let mut job2: TantivyDocument = Default::default();
            job2.add_text(skill, "java");
            job2.add_u64(year, 2007);
            writer.add_document(job2)?;
        }

        writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create parent filter for resume documents
        let parent_query = Box::new(TermQuery::new(
            Term::from_field_text(doc_type, "resume"),
            IndexRecordOption::Basic,
        ));

        // Create child query for relevant work experience
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
                    Bound::Included(Term::from_field_u64(year, 2006)),
                    Bound::Included(Term::from_field_u64(year, 2011)),
                )),
            ),
        ]);

        // Create parent query for UK residents
        let country_query = Box::new(TermQuery::new(
            Term::from_field_text(country, "United Kingdom"),
            IndexRecordOption::Basic,
        ));

        // Combine into nested query
        let nested_query = Box::new(NestedDocumentQuery::new(
            Box::new(child_query),
            parent_query,
            ScoreMode::Avg,
        ));

        // Combine with parent criteria
        let final_query = BooleanQuery::new(vec![
            (Occur::Must, country_query),
            (Occur::Must, nested_query),
        ]);

        let top_docs = searcher.search(&final_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 1);

        let doc: TantivyDocument = searcher.doc(top_docs[0].1)?;
        assert_eq!(doc.get_first(name).unwrap().as_str(), Some("Lisa"));

        Ok(())
    }
}

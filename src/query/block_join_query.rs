#![allow(unused)]
use crate::core::searcher::Searcher;
use crate::query::{EnableScoring, Explanation, Query, QueryClone, Scorer, Weight};
use crate::schema::Term;
use crate::{DocAddress, DocId, DocSet, Result, Score, SegmentReader, TERMINATED};
use common::BitSet;
use std::fmt;

/// How scores should be aggregated from child documents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreMode {
    /// Use the average of all child scores as the parent score.
    Avg,
    /// Use the maximum child score as the parent score.
    Max,
    /// Sum all child scores for the parent score.
    Total,
    /// Do not score parent docs from child docs. Just rely on parent scoring.
    None,
}

impl Default for ScoreMode {
    fn default() -> Self {
        ScoreMode::Avg
    }
}

/// `BlockJoinQuery` performs a join from child documents to parent documents,
/// based on a block structure: child documents are indexed before their parent.
/// The `parents_filter` identifies the parent documents in each segment.
pub struct BlockJoinQuery {
    child_query: Box<dyn Query>,
    parents_filter: Box<dyn Query>,
    score_mode: ScoreMode,
}

impl Clone for BlockJoinQuery {
    fn clone(&self) -> Self {
        BlockJoinQuery {
            child_query: self.child_query.box_clone(),
            parents_filter: self.parents_filter.box_clone(),
            score_mode: self.score_mode,
        }
    }
}

impl BlockJoinQuery {
    pub fn new(
        child_query: Box<dyn Query>,
        parents_filter: Box<dyn Query>,
        score_mode: ScoreMode,
    ) -> BlockJoinQuery {
        BlockJoinQuery {
            child_query,
            parents_filter,
            score_mode,
        }
    }
}

impl fmt::Debug for BlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlockJoinQuery(child_query: {:?}, parents_filter: {:?}, score_mode: {:?})",
            self.child_query, self.parents_filter, self.score_mode
        )
    }
}

impl Query for BlockJoinQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        println!("BlockJoinQuery::weight() - Creating weights");
        let child_weight = self.child_query.weight(enable_scoring.clone())?;
        println!("BlockJoinQuery::weight() - Created child weight");
        let parents_weight = self.parents_filter.weight(enable_scoring)?;
        println!("BlockJoinQuery::weight() - Created parent weight");

        Ok(Box::new(BlockJoinWeight {
            child_weight,
            parents_weight,
            score_mode: self.score_mode,
        }))
    }

    fn explain(&self, searcher: &Searcher, doc_address: DocAddress) -> Result<Explanation> {
        let reader = searcher.segment_reader(doc_address.segment_ord);
        let mut scorer = self
            .weight(EnableScoring::enabled_from_searcher(searcher))?
            .scorer(reader, 1.0)?;

        let mut current_doc = scorer.doc();
        while current_doc != TERMINATED && current_doc < doc_address.doc_id {
            current_doc = scorer.advance();
        }

        let score = if current_doc == doc_address.doc_id {
            scorer.score()
        } else {
            0.0
        };

        let mut explanation = Explanation::new("BlockJoinQuery", score);
        explanation.add_detail(Explanation::new("score", score));
        Ok(explanation)
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

struct BlockJoinWeight {
    child_weight: Box<dyn Weight>,
    parents_weight: Box<dyn Weight>,
    score_mode: ScoreMode,
}

impl Weight for BlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
        println!(
            "BlockJoinWeight::scorer() - Creating scorer with boost {}",
            boost
        );

        // Create parents bitset
        let max_doc = reader.max_doc();
        println!("BlockJoinWeight::scorer() - Max doc value: {}", max_doc);
        let mut parents_bitset = BitSet::with_max_value(max_doc);

        println!("BlockJoinWeight::scorer() - Creating parent scorer");
        let mut parents_scorer = self.parents_weight.scorer(reader, boost)?;
        println!("BlockJoinWeight::scorer() - Parent scorer created");

        // Collect all parent documents
        let mut found_parent = false;
        let mut parent_count = 0;
        while parents_scorer.doc() != TERMINATED {
            let parent_doc = parents_scorer.doc();
            println!(
                "BlockJoinWeight::scorer() - Found parent doc: {}",
                parent_doc
            );
            parents_bitset.insert(parent_doc);
            found_parent = true;
            parent_count += 1;
            parents_scorer.advance();
        }
        println!(
            "BlockJoinWeight::scorer() - Found {} parent documents",
            parent_count
        );

        // If no parents in this segment, return empty scorer
        if !found_parent {
            println!("BlockJoinWeight::scorer() - No parents found, returning empty scorer");
            return Ok(Box::new(EmptyScorer));
        }

        println!("BlockJoinWeight::scorer() - Creating child scorer");
        let child_scorer = self.child_weight.scorer(reader, boost)?;
        println!("BlockJoinWeight::scorer() - Child scorer created");

        println!("BlockJoinWeight::scorer() - Creating BlockJoinScorer");
        Ok(Box::new(BlockJoinScorer {
            child_scorer,
            parent_docs: parents_bitset,
            score_mode: self.score_mode,
            current_parent: TERMINATED,
            current_score: 0.0,
            initialized: false,
            has_more: true,
        }))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> Result<Explanation> {
        unimplemented!("Explain is not implemented for BlockJoinWeight");
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

struct BlockJoinScorer {
    child_scorer: Box<dyn Scorer>,
    parent_docs: BitSet,
    score_mode: ScoreMode,
    current_parent: DocId,
    current_score: Score,
    initialized: bool,
    has_more: bool,
}

impl DocSet for BlockJoinScorer {
    fn advance(&mut self) -> DocId {
        if !self.has_more {
            return TERMINATED;
        }

        if !self.initialized {
            self.child_scorer.advance();
            self.initialized = true;
        }

        let start = if self.current_parent == TERMINATED {
            0
        } else {
            self.current_parent + 1
        };

        self.current_parent = self.find_next_parent(start);

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

impl BlockJoinScorer {
    fn find_next_parent(&self, from: DocId) -> DocId {
        println!(
            "BlockJoinScorer::find_next_parent() - Starting from {}",
            from
        );
        let mut current = from;
        let max_value = self.parent_docs.max_value();
        println!(
            "BlockJoinScorer::find_next_parent() - Max value: {}",
            max_value
        );

        while current < max_value {
            if self.parent_docs.contains(current) {
                println!(
                    "BlockJoinScorer::find_next_parent() - Found parent at {}",
                    current
                );
                return current;
            }
            current += 1;
        }
        println!("BlockJoinScorer::find_next_parent() - No more parents found");
        TERMINATED
    }

    fn collect_matches(&mut self) -> DocId {
        let parent_id = self.current_parent;
        if parent_id == TERMINATED {
            return TERMINATED;
        }

        let mut child_doc = self.child_scorer.doc();
        let mut child_scores = Vec::new();
        let next_parent = self.find_next_parent(parent_id + 1);

        // Collect all child docs between current parent and next parent
        while child_doc != TERMINATED && (next_parent == TERMINATED || child_doc < next_parent) {
            if child_doc > parent_id {
                break;
            }
            if !self.parent_docs.contains(child_doc) {
                child_scores.push(self.child_scorer.score());
            }
            child_doc = self.child_scorer.advance();
        }

        // Compute parent score
        self.current_score = if child_scores.is_empty() {
            match self.score_mode {
                ScoreMode::None => 1.0,
                _ => 0.0,
            }
        } else {
            match self.score_mode {
                ScoreMode::Avg => child_scores.iter().sum::<Score>() / child_scores.len() as Score,
                ScoreMode::Max => child_scores.iter().cloned().fold(0.0, f32::max),
                ScoreMode::Total => child_scores.iter().sum(),
                ScoreMode::None => 1.0,
            }
        };
        parent_id
    }
}

impl Scorer for BlockJoinScorer {
    fn score(&mut self) -> Score {
        self.current_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocs;
    use crate::query::{Query, TermQuery};
    use crate::schema::*;
    use crate::{DocAddress, Index, IndexWriter, Term};

    fn create_test_index() -> crate::Result<(Index, Field, Field, Field, Field)> {
        let mut schema_builder = Schema::builder();
        let name_field = schema_builder.add_text_field("name", STRING | STORED);
        let country_field = schema_builder.add_text_field("country", STRING | STORED);
        let skill_field = schema_builder.add_text_field("skill", STRING | STORED);
        let doc_type_field = schema_builder.add_text_field("docType", STRING);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;

            // First block:
            // children docs first, parent doc last
            index_writer.add_documents(vec![
                doc!(skill_field => "java", doc_type_field => "job"),
                doc!(skill_field => "python", doc_type_field => "job"),
                doc!(skill_field => "java", doc_type_field => "job"),
                // parent last in this block
                doc!(name_field => "Lisa", country_field => "United Kingdom", doc_type_field => "resume"),
            ])?;

            // Second block:
            index_writer.add_documents(vec![
                doc!(skill_field => "ruby", doc_type_field => "job"),
                doc!(skill_field => "java", doc_type_field => "job"),
                // parent last in this block
                doc!(name_field => "Frank", country_field => "United States", doc_type_field => "resume"),
            ])?;

            index_writer.commit()?;
        }
        Ok((
            index,
            name_field,
            country_field,
            skill_field,
            doc_type_field,
        ))
    }

    #[test]
    pub fn test_simple_block_join() -> crate::Result<()> {
        let (index, name_field, _country_field, skill_field, doc_type_field) = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_query = TermQuery::new(
            Term::from_field_text(doc_type_field, "resume"),
            IndexRecordOption::Basic,
        );

        let child_query = TermQuery::new(
            Term::from_field_text(skill_field, "java"),
            IndexRecordOption::Basic,
        );

        let block_join_query = BlockJoinQuery::new(
            Box::new(child_query),
            Box::new(parent_query),
            ScoreMode::Avg,
        );

        let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 1);

        let doc: TantivyDocument = searcher.doc(top_docs[0].1)?;
        assert_eq!(doc.get_first(name_field).unwrap().as_str().unwrap(), "Lisa");

        Ok(())
    }

    #[test]
    pub fn test_block_join_no_matches() -> crate::Result<()> {
        let (index, _name_field, country_field, skill_field, doc_type_field) = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_query = TermQuery::new(
            Term::from_field_text(country_field, "Japan"), // Non-existent country
            IndexRecordOption::Basic,
        );

        let child_query = TermQuery::new(
            Term::from_field_text(skill_field, "java"),
            IndexRecordOption::Basic,
        );

        let block_join_query = BlockJoinQuery::new(
            Box::new(child_query),
            Box::new(parent_query),
            ScoreMode::Avg,
        );

        let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_block_join_scoring() -> crate::Result<()> {
        let (index, _name_field, _country_field, skill_field, doc_type_field) =
            create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_query = TermQuery::new(
            Term::from_field_text(doc_type_field, "resume"),
            IndexRecordOption::WithFreqs,
        );

        let child_query = TermQuery::new(
            Term::from_field_text(skill_field, "java"),
            IndexRecordOption::WithFreqs,
        );

        let block_join_query = BlockJoinQuery::new(
            Box::new(child_query),
            Box::new(parent_query),
            ScoreMode::Avg,
        );

        let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 1);

        // Score should be influenced by children, ensure it's not zero
        assert!(top_docs[0].0 > 0.0);

        Ok(())
    }

    #[test]
    pub fn test_explain_block_join() -> crate::Result<()> {
        let (index, _name_field, country_field, skill_field, _doc_type_field) =
            create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_query = TermQuery::new(
            Term::from_field_text(country_field, "United Kingdom"),
            IndexRecordOption::Basic,
        );

        let child_query = TermQuery::new(
            Term::from_field_text(skill_field, "java"),
            IndexRecordOption::Basic,
        );

        let block_join_query = BlockJoinQuery::new(
            Box::new(child_query),
            Box::new(parent_query),
            ScoreMode::Avg,
        );

        // The parent doc for "United Kingdom" is doc 3 in the first segment
        let explanation = block_join_query.explain(&searcher, DocAddress::new(0, 3))?;
        assert!(explanation.value() > 0.0);

        Ok(())
    }
}

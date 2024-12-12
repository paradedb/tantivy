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
///
/// Similar to Lucene's `BlockJoinQuery`, we wrap a "child query" and produce
/// matches in the "parent space".
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

    fn explain(&self, _searcher: &Searcher, _doc_address: DocAddress) -> Result<Explanation> {
        unimplemented!("Explain is not implemented for BlockJoinQuery");
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
        println!("BlockJoinWeight::scorer() - Creating scorer with boost {}", boost);
        
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
            println!("BlockJoinWeight::scorer() - Found parent doc: {}", parent_doc);
            parents_bitset.insert(parent_doc);
            found_parent = true;
            parent_count += 1;
            parents_scorer.advance();
        }
        println!("BlockJoinWeight::scorer() - Found {} parent documents", parent_count);

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
        println!("BlockJoinScorer::advance() - Starting advance");
        
        if !self.has_more {
            println!("BlockJoinScorer::advance() - No more documents available");
            return TERMINATED;
        }

        if !self.initialized {
            println!("BlockJoinScorer::advance() - Initializing child scorer");
            self.child_scorer.advance();
            self.initialized = true;
            println!("BlockJoinScorer::advance() - Child scorer initialized");
        }

        loop {
            let start = if self.current_parent == TERMINATED {
                println!("BlockJoinScorer::advance() - Starting from beginning");
                0
            } else {
                println!("BlockJoinScorer::advance() - Starting from parent {} + 1", self.current_parent);
                self.current_parent + 1
            };

            self.current_parent = self.find_next_parent(start);
            println!("BlockJoinScorer::advance() - Found next parent: {:?}", self.current_parent);
            
            if self.current_parent == TERMINATED {
                println!("BlockJoinScorer::advance() - No more parents found");
                self.has_more = false;
                return TERMINATED;
            }

            let doc_id = self.collect_matches();
            println!("BlockJoinScorer::advance() - Collected matches, doc_id: {:?}", doc_id);
            if doc_id != TERMINATED {
                return doc_id;
            }
            println!("BlockJoinScorer::advance() - No matches found for current parent, continuing...");
        }
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
        println!("BlockJoinScorer::find_next_parent() - Starting from {}", from);
        let mut current = from;
        let max_value = self.parent_docs.max_value();
        println!("BlockJoinScorer::find_next_parent() - Max value: {}", max_value);
        
        while current < max_value {
            if self.parent_docs.contains(current) {
                println!("BlockJoinScorer::find_next_parent() - Found parent at {}", current);
                return current;
            }
            current += 1;
        }
        println!("BlockJoinScorer::find_next_parent() - No more parents found");
        TERMINATED
    }

    fn collect_matches(&mut self) -> DocId {
        println!("BlockJoinScorer::collect_matches() - Starting collection for parent {}", self.current_parent);
        let parent_id = self.current_parent;
        let mut child_doc = self.child_scorer.doc();
        println!("BlockJoinScorer::collect_matches() - Initial child doc: {:?}", child_doc);
        let mut child_scores = Vec::new();

        while child_doc != TERMINATED && child_doc < parent_id {
            println!("BlockJoinScorer::collect_matches() - Processing child doc {} for parent {}", child_doc, parent_id);
            
            // Check if there's another parent in between:
            let mut is_valid = true;
            for doc_id in (child_doc + 1)..parent_id {
                if self.parent_docs.contains(doc_id) {
                    println!("BlockJoinScorer::collect_matches() - Found intervening parent at {}", doc_id);
                    is_valid = false;
                    break;
                }
            }

            if is_valid {
                let score = self.child_scorer.score();
                println!("BlockJoinScorer::collect_matches() - Valid child found with score {}", score);
                child_scores.push(score);
            }

            child_doc = self.child_scorer.advance();
            println!("BlockJoinScorer::collect_matches() - Advanced to next child: {:?}", child_doc);
        }

        if child_scores.is_empty() {
            println!("BlockJoinScorer::collect_matches() - No matching children found for parent {}", parent_id);
            TERMINATED
        } else {
            println!("BlockJoinScorer::collect_matches() - Found {} matching children", child_scores.len());
            self.current_score = match self.score_mode {
                ScoreMode::Avg => {
                    let avg = child_scores.iter().sum::<Score>() / child_scores.len() as Score;
                    println!("BlockJoinScorer::collect_matches() - Calculated average score: {}", avg);
                    avg
                },
                ScoreMode::Max => {
                    let max = child_scores.iter().cloned().fold(f32::MIN, f32::max);
                    println!("BlockJoinScorer::collect_matches() - Calculated max score: {}", max);
                    max
                },
                ScoreMode::Total => {
                    let total = child_scores.iter().sum();
                    println!("BlockJoinScorer::collect_matches() - Calculated total score: {}", total);
                    total
                },
                ScoreMode::None => {
                    println!("BlockJoinScorer::collect_matches() - Using no scoring mode");
                    0.0
                },
            };
            parent_id
        }
    }
}

impl Scorer for BlockJoinScorer {
    fn score(&mut self) -> Score {
        self.current_score
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::collector::TopDocs;
//     use crate::query::block_join_query::{BlockJoinQuery, ScoreMode};
//     use crate::query::TermQuery;
//     use crate::schema::{IndexRecordOption, Schema, INDEXED, STORED, STRING};
//     use crate::{Index, Term};

//     #[test]
//     fn test_block_join_query() -> crate::Result<()> {
//         // Build a schema:
//         let mut schema_builder = Schema::builder();
//         let name = schema_builder.add_text_field("name", STORED);
//         let country = schema_builder.add_text_field("country", STRING | STORED);
//         let doc_type = schema_builder.add_text_field("doc_type", STRING | STORED);
//         let skill = schema_builder.add_text_field("skill", STRING | STORED);
//         let year = schema_builder.add_u64_field("year", INDEXED | STORED);
//         let schema = schema_builder.build();

//         // Create index
//         let index = Index::create_in_ram(schema);
//         let mut writer = index.writer(50_000_000)?;

//         // Add a set of child docs followed by a parent doc
//         // child docs:
//         writer.add_documents(vec![
//             doc!(skill => "java",   year => 2006u64),
//             doc!(skill => "python", year => 2010u64),
//             doc!(name => "Lisa", country => "United Kingdom", doc_type => "resume"),
//             doc!(skill => "ruby",  year => 2005u64),
//             doc!(skill => "java",  year => 2007u64),
//             doc!(name => "Frank", country => "United States", doc_type => "resume"),
//         ])?;

//         writer.commit()?;

//         let reader = index.reader()?;
//         let searcher = reader.searcher();

//         // parent filter query
//         let parent_query = Box::new(TermQuery::new(
//             Term::from_field_text(doc_type, "resume"),
//             IndexRecordOption::Basic,
//         ));

//         // child query
//         let child_query = Box::new(crate::query::BooleanQuery::new_multiterms_query(vec![
//             Term::from_field_text(skill, "java"),
//             Term::from_field_u64(year, 2006),
//         ]));

//         // Wrap child query in BlockJoinQuery:
//         let block_join_query = BlockJoinQuery::new(child_query, parent_query, ScoreMode::Avg);

//         // Just test searching top docs:
//         let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(10))?;
//         assert_eq!(
//             top_docs.len(),
//             2,
//             "Should find 2 parent matches from children"
//         );

//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocs;
    use crate::query::{Query, Scorer, TermQuery};
    use crate::schema::*;
    use crate::{DocAddress, DocId, Index, IndexWriter, Score};

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

            // First resume block
            index_writer.add_document(doc!(
                skill_field => "java",
                doc_type_field => "job"
            ))?;
            index_writer.add_document(doc!(
                skill_field => "python",
                doc_type_field => "job"
            ))?;
            index_writer.add_document(doc!(
                name_field => "Lisa",
                country_field => "United Kingdom",
                doc_type_field => "resume"
            ))?;

            // Second resume block
            index_writer.add_document(doc!(
                skill_field => "ruby",
                doc_type_field => "job"
            ))?;
            index_writer.add_document(doc!(
                skill_field => "java",
                doc_type_field => "job"
            ))?;
            index_writer.add_document(doc!(
                name_field => "Frank",
                country_field => "United States",
                doc_type_field => "resume"
            ))?;

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

    #[derive(Debug)]
    struct BlockJoinQuery {
        parent_query: Box<dyn Query>,
        child_query: Box<dyn Query>,
        doc_type_field: Field,
        parent_doc_type: String,
    }

    impl BlockJoinQuery {
        fn new<Q1: Query + 'static, Q2: Query + 'static>(
            parent_query: Q1,
            child_query: Q2,
            doc_type_field: Field,
            parent_doc_type: &str,
        ) -> Self {
            BlockJoinQuery {
                parent_query: Box::new(parent_query),
                child_query: Box::new(child_query),
                doc_type_field,
                parent_doc_type: parent_doc_type.to_string(),
            }
        }
    }

    // Implement QueryClone manually instead of deriving Clone
    impl QueryClone for BlockJoinQuery {
        fn box_clone(&self) -> Box<dyn Query> {
            Box::new(BlockJoinQuery {
                parent_query: self.parent_query.box_clone(),
                child_query: self.child_query.box_clone(),
                doc_type_field: self.doc_type_field,
                parent_doc_type: self.parent_doc_type.clone(),
            })
        }
    }

    impl Query for BlockJoinQuery {
        fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
            let parent_weight = self.parent_query.weight(enable_scoring.clone())?;
            let child_weight = self.child_query.weight(enable_scoring)?;

            Ok(Box::new(BlockJoinWeight {
                parent_weight,
                child_weight,
                doc_type_field: self.doc_type_field,
                parent_doc_type: self.parent_doc_type.clone(),
            }))
        }

        fn explain(
            &self,
            searcher: &Searcher,
            doc_address: DocAddress,
        ) -> crate::Result<Explanation> {
            let weight = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
            weight.explain(
                &searcher.segment_reader(doc_address.segment_ord),
                doc_address.doc_id,
            )
        }
    }

    struct BlockJoinWeight {
        parent_weight: Box<dyn Weight>,
        child_weight: Box<dyn Weight>,
        doc_type_field: Field,
        parent_doc_type: String,
    }

    impl Weight for BlockJoinWeight {
        fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
            let parent_scorer = self.parent_weight.scorer(reader, boost)?;
            let child_scorer = self.child_weight.scorer(reader, boost)?;

            Ok(Box::new(BlockJoinScorer {
                parent_scorer,
                child_scorer,
                current_doc: 0,
                score: 0.0,
            }))
        }

        fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
            let mut scorer = self.scorer(reader, 1.0)?;

            if scorer.seek(doc) != doc {
                return Ok(Explanation::new("No match", 0.0));
            }

            Ok(Explanation::new("Block join score", scorer.score()))
        }
    }

    struct BlockJoinScorer {
        parent_scorer: Box<dyn Scorer>,
        child_scorer: Box<dyn Scorer>,
        current_doc: DocId,
        score: Score,
    }

    impl DocSet for BlockJoinScorer {
        fn advance(&mut self) -> DocId {
            let parent_doc = self.parent_scorer.advance();
            if parent_doc == TERMINATED {
                return TERMINATED;
            }

            // Find child docs between current and next parent
            let mut found_child = false;
            while self.child_scorer.doc() < parent_doc {
                found_child = true;
                if self.child_scorer.advance() == TERMINATED {
                    break;
                }
            }

            self.current_doc = if found_child { parent_doc } else { TERMINATED };
            self.current_doc
        }

        fn doc(&self) -> DocId {
            self.current_doc
        }

        fn size_hint(&self) -> u32 {
            self.parent_scorer.size_hint()
        }
    }

    impl Scorer for BlockJoinScorer {
        fn score(&mut self) -> Score {
            self.score
        }
    }

    #[test]
    pub fn test_simple_block_join() -> crate::Result<()> {
        let (index, name_field, country_field, skill_field, doc_type_field) = create_test_index()?;
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

        let block_join_query =
            BlockJoinQuery::new(parent_query, child_query, doc_type_field, "resume");

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

        let block_join_query =
            BlockJoinQuery::new(parent_query, child_query, doc_type_field, "resume");

        let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_block_join_scoring() -> crate::Result<()> {
        let (index, _name_field, country_field, skill_field, doc_type_field) = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_query = TermQuery::new(
            Term::from_field_text(country_field, "United Kingdom"),
            IndexRecordOption::WithFreqs,
        );

        let child_query = TermQuery::new(
            Term::from_field_text(skill_field, "java"),
            IndexRecordOption::WithFreqs,
        );

        let block_join_query =
            BlockJoinQuery::new(parent_query, child_query, doc_type_field, "resume");

        let top_docs = searcher.search(&block_join_query, &TopDocs::with_limit(1))?;
        assert_eq!(top_docs.len(), 1);

        // Score should be influenced by both parent and child matches
        assert!(top_docs[0].0 > 0.0);

        Ok(())
    }

    #[test]
    pub fn test_explain_block_join() -> crate::Result<()> {
        let (index, _name_field, country_field, skill_field, doc_type_field) = create_test_index()?;
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

        let block_join_query =
            BlockJoinQuery::new(parent_query, child_query, doc_type_field, "resume");

        let explanation = block_join_query.explain(&searcher, DocAddress::new(0, 2))?;
        assert!(explanation.value() > 0.0);

        Ok(())
    }
}

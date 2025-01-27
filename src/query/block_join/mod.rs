mod parent_children_block_join_query;
mod to_child_block_join_query;
mod to_parent_block_join_query;

use crate::core::searcher::Searcher;
use crate::schema::{Field, Value};
use crate::{DocAddress, TantivyDocument};

pub use to_parent_block_join_query::{ParentBitSetProducer, ScoreMode, ToParentBlockJoinQuery};

#[allow(unused)]
pub fn doc_string_field(searcher: &Searcher, doc_addr: DocAddress, field: Field) -> String {
    // Retrieve the stored document
    if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_addr) {
        if let Some(v) = doc.get_first(field) {
            if let Some(text) = v.as_str() {
                return text.to_string();
            }
        }
    }
    "".to_string()
}

#[cfg(test)]
mod scorer_tests {
    use common::BitSet;

    use crate::collector::TopDocs;
    use crate::query::block_join::doc_string_field;
    use crate::query::AllQuery;
    use crate::schema::{Field, IndexRecordOption, SchemaBuilder, STORED, STRING};
    use crate::{doc, DocSet, SegmentReader};
    use crate::{Index, IndexSettings};
    use std::sync::Arc;

    use crate::directory::RamDirectory;
    use crate::query::{ParentBitSetProducer, ScoreMode, ToParentBlockJoinQuery};

    use crate::Result;

    pub struct ParentBitsForScorerTest {
        doc_type_field: Field,
    }

    impl ParentBitsForScorerTest {
        pub fn new(doc_type_field: Field) -> Self {
            ParentBitsForScorerTest { doc_type_field }
        }
    }

    impl ParentBitSetProducer for ParentBitsForScorerTest {
        fn produce(&self, reader: &SegmentReader) -> crate::Result<BitSet> {
            let max_doc = reader.max_doc();
            let mut bitset = BitSet::with_max_value(max_doc);

            let inverted = reader.inverted_index(self.doc_type_field)?;
            // Now look for "parent" instead of "resume"!
            let term = crate::Term::from_field_text(self.doc_type_field, "parent");

            if let Some(mut postings) = inverted.read_postings(&term, IndexRecordOption::Basic)? {
                let mut doc_id = postings.doc();
                while doc_id != crate::TERMINATED {
                    bitset.insert(doc_id);
                    doc_id = postings.advance();
                }
            }
            Ok(bitset)
        }
    }

    /// This test emulates the Java testScoreNone scenario:
    /// We build 10 "blocks."  Block i has i child docs, then 1 parent doc (with docType="parent").
    /// The child query matches all child docs, but ScoreMode::None => the parent's score is 0.0,
    /// and the doc iteration returns only the parent docs in order.
    #[test]
    fn test_score_none() -> Result<()> {
        // 1) Build schema
        let mut sb = SchemaBuilder::default();
        let value_f = sb.add_text_field("value", STRING | STORED);
        let doctype_f = sb.add_text_field("docType", STRING);
        let schema = sb.build();

        // 2) Create index
        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;

        // 3) Index 10 blocks: block i has i child docs + 1 parent
        {
            let mut writer = index.writer_for_tests()?;
            for i in 0..10 {
                let mut block_docs = Vec::new();
                // Add i child docs
                for j in 0..i {
                    // child: "value" => j
                    let child_doc = doc! {
                        value_f => j.to_string(),
                    };
                    block_docs.push(child_doc);
                }
                // parent doc
                let parent_doc = doc! {
                    doctype_f => "parent",
                    value_f   => i.to_string(),
                };
                block_docs.push(parent_doc);
                writer.add_documents(block_docs)?;
            }
            writer.commit()?;
        }

        // 4) Create a searcher
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // 5) "parents filter"
        let parent_bits = Arc::new(ParentBitsForScorerTest::new(doctype_f));

        // 6) Child query => matches all docs. We only want to match child docs, but it's okay to use AllQuery
        //    Because the block-join logic enforces the docType=parent is not in child query (not indexed).
        let child_query = AllQuery;

        // 7) Wrap with ToParentBlockJoinQuery => ScoreMode::None
        let join_q = ToParentBlockJoinQuery::new(
            Box::new(child_query),
            parent_bits.clone(),
            ScoreMode::None,
        );

        // 8) Search for top 20 hits
        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(20))?;

        // Expect 10 parent docs
        assert_eq!(10, top_docs.len(), "We should find exactly 10 parents.");

        // Scores must be zero with ScoreMode::None
        for (score, _addr) in &top_docs {
            assert!(
                (*score - 0.0).abs() < f32::EPSILON,
                "ScoreMode::None => parent's score must be 0.0"
            );
        }

        // Optionally, confirm the "value" field is in ascending order
        // i.e. the parent blocks should appear in the same order we inserted them
        let mut last_i: i64 = -1;
        for (_, addr) in &top_docs {
            let val_str = doc_string_field(&searcher, *addr, value_f);
            let this_i = val_str.parse::<i64>().unwrap_or(-99);
            assert!(
                this_i > last_i,
                "Parent doc IDs not in ascending order! last={}, current={}",
                last_i,
                this_i
            );
            last_i = this_i;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use std::ops::Bound;
    use std::sync::Arc;

    use common::BitSet;

    use crate::collector::TopDocs;
    use crate::directory::RamDirectory;
    use crate::docset::DocSet;
    use crate::index::{Index, IndexSettings};
    use crate::query::block_join::doc_string_field;
    use crate::query::block_join::to_child_block_join_query::ToChildBlockJoinQuery;
    use crate::query::{BooleanQuery, BoostQuery, Occur, RangeQuery, TermQuery};
    use crate::query::{ParentBitSetProducer, ScoreMode, ToParentBlockJoinQuery};
    use crate::schema::{
        Field, IndexRecordOption, SchemaBuilder, TantivyDocument, TextFieldIndexing, TextOptions,
        FAST, STORED, STRING, TEXT,
    };
    use crate::tokenizer::{RawTokenizer, TextAnalyzer, TokenizerManager};
    use crate::{doc, IndexWriter, Term};
    use crate::{Result, SegmentReader, TERMINATED};

    // --------------------------------------------------------------------------
    // A small helper for building test doc arrays (resumes vs children).
    // --------------------------------------------------------------------------
    fn make_resume(name_f: Field, country_f: Field, name: &str, country: &str) -> TantivyDocument {
        doc! {
            name_f => name,
            country_f => country,
        }
    }

    fn make_job(skill_f: Field, year_f: Field, skill: &str, year: i64) -> TantivyDocument {
        doc! {
            skill_f => skill,
            year_f => year,
        }
    }

    fn make_qualification(qual_f: Field, year_f: Field, qual: &str, year: i64) -> TantivyDocument {
        doc! {
            qual_f => qual,
            year_f => year,
        }
    }

    // --------------------------------------------------------------------------
    // A test-specific "parent filter" that marks docType="resume" as parent.
    // --------------------------------------------------------------------------
    pub struct ResumeParentBitSetProducer {
        doc_type_field: Field,
    }

    impl ResumeParentBitSetProducer {
        pub fn new(doc_type_field: Field) -> Self {
            ResumeParentBitSetProducer { doc_type_field }
        }
    }

    impl ParentBitSetProducer for ResumeParentBitSetProducer {
        fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
            let max_doc = reader.max_doc();
            let mut bitset = BitSet::with_max_value(max_doc);

            // Inverted index
            let inverted = reader.inverted_index(self.doc_type_field)?;
            let term = crate::Term::from_field_text(self.doc_type_field, "resume");
            if let Some(mut postings) = inverted.read_postings(&term, IndexRecordOption::Basic)? {
                let mut doc = postings.doc();
                while doc != TERMINATED {
                    bitset.insert(doc);
                    doc = postings.advance();
                }
            }

            Ok(bitset)
        }
    }

    fn strset<T: IntoIterator<Item = String>>(items: T) -> HashSet<String> {
        items.into_iter().collect()
    }

    // --------------------------------------------------------------------------
    // Now the test functions follow
    // --------------------------------------------------------------------------

    /// This test checks that if we have a child filter that matches zero child docs,
    /// then ToParentBlockJoinQuery should produce no parent documents.
    #[test]
    fn test_empty_child_filter() -> crate::Result<()> {
        // 1) Set up the schema and index as before
        let mut sb = SchemaBuilder::default();
        let skill_f = sb.add_text_field("skill", STRING | STORED);
        let year_f = sb.add_i64_field("year", FAST | STORED);
        let doctype_f = sb.add_text_field("docType", STRING);
        let name_f = sb.add_text_field("name", STORED);
        let country_f = sb.add_text_field("country", STRING | STORED);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;

        {
            let mut writer = index.writer_for_tests()?;

            // block #1
            writer.add_documents(vec![
                // children
                make_job(skill_f, year_f, "java", 2007),
                make_job(skill_f, year_f, "python", 2010),
                // parent
                {
                    let mut d = make_resume(name_f, country_f, "Lisa", "United Kingdom");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;

            // block #2
            writer.add_documents(vec![
                // children
                make_job(skill_f, year_f, "ruby", 2005),
                make_job(skill_f, year_f, "java", 2006),
                // parent
                {
                    let mut d = make_resume(name_f, country_f, "Frank", "United States");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;

            writer.commit()?;
        }

        // 2) Build the searcher
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // 3) Build a ParentBitSetProducer: which docs are parents?
        let parent_bits = Arc::new(ResumeParentBitSetProducer::new(doctype_f));

        // 4) Child filter: skill=java AND year in [2006..2011]
        let q_java = TermQuery::new(
            Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );

        let q_year = RangeQuery::new(
            Bound::Included(Term::from_field_i64(year_f, 2006)),
            Bound::Included(Term::from_field_i64(year_f, 2011)),
        );

        let child_bq = BooleanQuery::intersection(vec![Box::new(q_java), Box::new(q_year)]);

        // 5) Wrap that child query in a ToParentBlockJoinQuery
        let join_q =
            ToParentBlockJoinQuery::new(Box::new(child_bq), parent_bits.clone(), ScoreMode::Avg);

        // 6) Search and confirm we get two parents (Lisa, Frank)
        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        assert_eq!(
            2,
            top_docs.len(),
            "Expected 2 parents from the child->parent join!"
        );

        // Optionally verify that those parents are Lisa and Frank
        let found_names: HashSet<String> = top_docs
            .iter()
            .map(|(_, addr)| doc_string_field(&searcher, *addr, name_f))
            .collect();

        let expected = strset(vec!["Lisa".to_string(), "Frank".to_string()]);
        assert_eq!(
            expected, found_names,
            "Should have matched the two parents: Lisa and Frank"
        );

        Ok(())
    }

    #[test]
    fn test_bq_should_joined_child() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let skill_f = sb.add_text_field("skill", STRING | STORED);
        let year_f = sb.add_i64_field("year", FAST | STORED);
        let doctype_f = sb.add_text_field("docType", STRING);
        let name_f = sb.add_text_field("name", STORED);
        let country_f = sb.add_text_field("country", STRING | STORED);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block1
            writer.add_documents(vec![
                make_job(skill_f, year_f, "java", 2007),
                make_job(skill_f, year_f, "python", 2010),
                {
                    let mut d = make_resume(name_f, country_f, "Lisa", "United Kingdom");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            // block2
            writer.add_documents(vec![
                make_job(skill_f, year_f, "ruby", 2005),
                make_job(skill_f, year_f, "java", 2006),
                {
                    let mut d = make_resume(name_f, country_f, "Frank", "United States");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_bits = Arc::new(ResumeParentBitSetProducer::new(doctype_f));

        // child => skill=java, year=[2006..2011]
        let q_java = TermQuery::new(
            crate::Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let q_year = RangeQuery::new(
            Bound::Included(crate::Term::from_field_i64(year_f, 2006)),
            Bound::Included(crate::Term::from_field_i64(year_f, 2011)),
        );
        let child_bq = BooleanQuery::intersection(vec![Box::new(q_java), Box::new(q_year)]);
        let child_join =
            ToParentBlockJoinQuery::new(Box::new(child_bq), parent_bits.clone(), ScoreMode::Avg);

        // parent => country=UK
        let parent_query = TermQuery::new(
            crate::Term::from_field_text(country_f, "United Kingdom"),
            IndexRecordOption::Basic,
        );

        // SHOULD => union
        let or_query = BooleanQuery::new(vec![
            (Occur::Should, Box::new(parent_query)),
            (Occur::Should, Box::new(child_join)),
        ]);

        let top_docs = searcher.search(&or_query, &TopDocs::with_limit(5))?;
        // Expect 2 => Lisa + Frank
        assert_eq!(2, top_docs.len());

        let found: HashSet<String> = top_docs
            .iter()
            .map(|(_, addr)| doc_string_field(&searcher, *addr, name_f))
            .collect();
        assert_eq!(strset(vec!["Lisa".to_string(), "Frank".to_string()]), found);

        Ok(())
    }

    #[test]
    fn test_simple() -> Result<()> {
        // 1) Define a custom TextOptions with the raw tokenizer.
        let raw_stored_indexed = TextOptions::default().set_stored().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::Basic),
        );

        // 2) Build schema, specifying raw tokenizer for skill, docType, country
        //    while "year" is i64, "name" is just stored text, etc.
        let mut sb = SchemaBuilder::default();

        // skill => raw tokenizer (stored, indexed)
        let skill_f = sb.add_text_field("skill", raw_stored_indexed.clone());

        // year => i64
        let year_f = sb.add_i64_field("year", FAST | STORED);

        // docType => raw tokenizer (not stored in this example—unless you want it)
        let doc_type_options = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::Basic),
        );
        let doctype_f = sb.add_text_field("docType", doc_type_options);

        // name => just stored text (for retrieval), no indexing
        let name_f = sb.add_text_field("name", STORED);

        // country => raw tokenizer (stored + indexed)
        let country_f = sb.add_text_field("country", raw_stored_indexed);

        let schema = sb.build();

        // 3) Create the index, add documents in parent-child blocks
        let _ram = RamDirectory::create();
        // Create index as before
        //
        // Create our custom TokenizerManager
        let my_tokenizers = TokenizerManager::default();
        my_tokenizers.register("raw", TextAnalyzer::from(RawTokenizer::default()));
        let index = Index::builder()
            .schema(schema)
            .tokenizers(my_tokenizers) // <--- register them here
            .settings(IndexSettings::default())
            .create_in_ram()?;

        {
            let mut writer = index.writer_for_tests()?;
            // block1
            writer.add_documents(vec![
                make_job(skill_f, year_f, "java", 2007),
                make_job(skill_f, year_f, "python", 2010),
                {
                    let mut d = make_resume(name_f, country_f, "Lisa", "United Kingdom");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            // block2
            writer.add_documents(vec![
                make_job(skill_f, year_f, "ruby", 2005),
                make_job(skill_f, year_f, "java", 2006),
                {
                    let mut d = make_resume(name_f, country_f, "Frank", "United States");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            writer.commit()?;
        }

        // 4) Build a searcher
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Create a "parent bitset" for docs with docType="resume"
        let parent_bits = Arc::new(ResumeParentBitSetProducer::new(doctype_f));

        // child => skill=java, year in [2006..2011]
        let q_java = TermQuery::new(
            Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let q_year = RangeQuery::new(
            Bound::Included(Term::from_field_i64(year_f, 2006)),
            Bound::Included(Term::from_field_i64(year_f, 2011)),
        );
        let child_bq = BooleanQuery::intersection(vec![Box::new(q_java.clone()), Box::new(q_year)]);

        // parent => country="United Kingdom"
        let parent_q = TermQuery::new(
            Term::from_field_text(country_f, "United Kingdom"),
            IndexRecordOption::Basic,
        );

        // 5) child->parent join
        let child_join =
            ToParentBlockJoinQuery::new(Box::new(child_bq), parent_bits.clone(), ScoreMode::Avg);
        let and_query = BooleanQuery::intersection(vec![Box::new(parent_q), Box::new(child_join)]);
        let top_docs = searcher.search(&and_query, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs.len());
        let name_val = doc_string_field(&searcher, top_docs[0].1, name_f);
        assert_eq!("Lisa", name_val);

        // 6) Now parent->child join
        let up_join = ToChildBlockJoinQuery::new(
            Box::new(TermQuery::new(
                Term::from_field_text(country_f, "United Kingdom"),
                IndexRecordOption::Basic,
            )),
            parent_bits.clone(),
        );
        // child => skill=java
        let child_again = BooleanQuery::intersection(vec![Box::new(up_join), Box::new(q_java)]);
        let child_hits = searcher.search(&child_again, &TopDocs::with_limit(10))?;
        assert_eq!(1, child_hits.len());
        let skill_val = doc_string_field(&searcher, child_hits[0].1, skill_f);
        assert_eq!("java", skill_val);

        Ok(())
    }

    #[test]
    fn test_simple_filter() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let skill_f = sb.add_text_field("skill", STRING | STORED);
        let year_f = sb.add_i64_field("year", FAST | STORED);
        let doctype_f = sb.add_text_field("docType", STRING);
        let name_f = sb.add_text_field("name", STORED);
        let country_f = sb.add_text_field("country", STRING | STORED);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block #1
            writer.add_documents(vec![
                make_job(skill_f, year_f, "java", 2007),
                make_job(skill_f, year_f, "python", 2010),
                {
                    let mut d = make_resume(name_f, country_f, "Lisa", "United Kingdom");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            // skillless doc
            writer.add_document({
                let mut d = doc! {};
                d.add_text(doctype_f, "resume");
                d.add_text(name_f, "Skillless");
                d
            })?;
            // block #2
            writer.add_documents(vec![
                make_job(skill_f, year_f, "ruby", 2005),
                make_job(skill_f, year_f, "java", 2006),
                {
                    let mut d = make_resume(name_f, country_f, "Frank", "United States");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            // another skillless
            writer.add_document({
                let mut d = doc! {};
                d.add_text(doctype_f, "resume");
                d.add_text(name_f, "Skillless2");
                d
            })?;

            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parents = Arc::new(ResumeParentBitSetProducer::new(doctype_f));

        // child => skill=java, year=2006..2011
        let q_java = TermQuery::new(
            crate::Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let q_year = RangeQuery::new(
            Bound::Included(crate::Term::from_field_i64(year_f, 2006)),
            Bound::Included(crate::Term::from_field_i64(year_f, 2011)),
        );
        let child_bq = BooleanQuery::intersection(vec![Box::new(q_java.clone()), Box::new(q_year)]);
        let child_join = ToParentBlockJoinQuery::new(
            Box::new(child_bq.clone()),
            parents.clone(),
            ScoreMode::Avg,
        );

        // no filter => should find 2
        let no_filter_docs = searcher.search(&child_join, &TopDocs::with_limit(10))?;
        assert_eq!(2, no_filter_docs.len());

        // filter => docType=resume
        let filter_query = TermQuery::new(
            crate::Term::from_field_text(doctype_f, "resume"),
            IndexRecordOption::Basic,
        );
        let bq2 =
            BooleanQuery::intersection(vec![Box::new(child_join.clone()), Box::new(filter_query)]);
        let docs2 = searcher.search(&bq2, &TopDocs::with_limit(10))?;
        assert_eq!(2, docs2.len());

        // filter => country=Oz => 0
        let q_oz = TermQuery::new(
            crate::Term::from_field_text(country_f, "Oz"),
            IndexRecordOption::Basic,
        );
        let bq_oz = BooleanQuery::intersection(vec![Box::new(child_join.clone()), Box::new(q_oz)]);
        let oz_docs = searcher.search(&bq_oz, &TopDocs::with_limit(10))?;
        assert_eq!(0, oz_docs.len());

        // filter => country=UK => Lisa only
        let q_uk = TermQuery::new(
            crate::Term::from_field_text(country_f, "United Kingdom"),
            IndexRecordOption::Basic,
        );
        let bq_uk = BooleanQuery::intersection(vec![Box::new(child_join), Box::new(q_uk)]);
        let uk_docs = searcher.search(&bq_uk, &TopDocs::with_limit(10))?;
        assert_eq!(1, uk_docs.len());
        let nm = doc_string_field(&searcher, uk_docs[0].1, name_f);
        assert_eq!("Lisa", nm);

        Ok(())
    }

    #[test]
    fn test_child_query_never_match() -> crate::Result<()> {
        // Give the schema at least one field:
        let mut schema_builder = SchemaBuilder::default();
        let dummy_field = schema_builder.add_text_field("dummy", STRING);
        let schema = schema_builder.build();

        // Create the index in RAM
        let index = Index::create_in_ram(schema);
        {
            // Just commit an empty segment so the index isn't empty
            let mut writer: IndexWriter = index.writer_for_tests()?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // A minimal ParentBitSetProducer that always returns an empty BitSet:
        struct DummyParent;
        impl ParentBitSetProducer for DummyParent {
            fn produce(&self, _reader: &SegmentReader) -> crate::Result<BitSet> {
                // no docs marked as parents
                Ok(BitSet::with_max_value(0))
            }
        }
        let dummy_arc = Arc::new(DummyParent);

        // Child query that will never match
        let child_q = TermQuery::new(
            crate::Term::from_field_text(dummy_field, "no-match"),
            IndexRecordOption::Basic,
        );

        // Wrap in a ToParentBlockJoinQuery
        let join_q = ToParentBlockJoinQuery::new(Box::new(child_q), dummy_arc, ScoreMode::Avg);

        // Then boost it
        let boosted = BoostQuery::new(Box::new(join_q), 2.0);

        // Execute and verify we get 0 hits without panic
        let top_docs = searcher.search(&boosted, &TopDocs::with_limit(10))?;
        assert_eq!(0, top_docs.len(), "Expected no hits, but got some");

        Ok(())
    }

    #[test]
    fn test_multi_child_types() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let doctype_f = sb.add_text_field("docType", STRING);
        let skill_f = sb.add_text_field("skill", STRING | STORED);
        let qual_f = sb.add_text_field("qualification", STRING | STORED);
        let year_f = sb.add_i64_field("year", FAST | STORED);
        let name_f = sb.add_text_field("name", STORED);
        let country_f = sb.add_text_field("country", STRING | STORED);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // single block
            writer.add_documents(vec![
                make_job(skill_f, year_f, "java", 2007),
                make_job(skill_f, year_f, "python", 2010),
                make_qualification(qual_f, year_f, "maths", 1999),
                {
                    let mut d = make_resume(name_f, country_f, "Lisa", "United Kingdom");
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let parent_bits = Arc::new(ResumeParentBitSetProducer::new(doctype_f));

        // child #1 => skill=java, year in [2006..2011]
        let c1_skill = TermQuery::new(
            crate::Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let c1_year = RangeQuery::new(
            Bound::Included(crate::Term::from_field_i64(year_f, 2006)),
            Bound::Included(crate::Term::from_field_i64(year_f, 2011)),
        );
        let child1 = BooleanQuery::intersection(vec![Box::new(c1_skill), Box::new(c1_year)]);

        // child #2 => qualification=maths, year in [1980..2000]
        let c2_qual = TermQuery::new(
            crate::Term::from_field_text(qual_f, "maths"),
            IndexRecordOption::Basic,
        );
        let c2_year = RangeQuery::new(
            Bound::Included(crate::Term::from_field_i64(year_f, 1980)),
            Bound::Included(crate::Term::from_field_i64(year_f, 2000)),
        );
        let child2 = BooleanQuery::intersection(vec![Box::new(c2_qual), Box::new(c2_year)]);

        // parent => country=UK
        let parent_q = TermQuery::new(
            crate::Term::from_field_text(country_f, "United Kingdom"),
            IndexRecordOption::Basic,
        );

        let join1 =
            ToParentBlockJoinQuery::new(Box::new(child1), parent_bits.clone(), ScoreMode::Avg);
        let join2 =
            ToParentBlockJoinQuery::new(Box::new(child2), parent_bits.clone(), ScoreMode::Avg);

        let big_bq =
            BooleanQuery::intersection(vec![Box::new(parent_q), Box::new(join1), Box::new(join2)]);
        let top_docs = searcher.search(&big_bq, &TopDocs::with_limit(10))?;
        // should be 1 => Lisa
        assert_eq!(1, top_docs.len());
        let nm = doc_string_field(&searcher, top_docs[0].1, name_f);
        assert_eq!("Lisa", nm);

        Ok(())
    }

    #[test]
    fn test_advance_single_parent_single_child() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let child_f = sb.add_text_field("child", STRING);
        let parent_f = sb.add_text_field("parent", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_documents(vec![doc!(child_f => "1"), doc!(parent_f => "1")])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // parent filter => parent="1"
        struct MyParentBitset(Field, &'static str);
        impl ParentBitSetProducer for MyParentBitset {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, self.1);
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parents = Arc::new(MyParentBitset(parent_f, "1"));
        let child_q = TermQuery::new(
            crate::Term::from_field_text(child_f, "1"),
            IndexRecordOption::Basic,
        );
        let join_q = ToParentBlockJoinQuery::new(Box::new(child_q), parents, ScoreMode::Avg);

        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        // We expect 1 parent match
        assert_eq!(1, top_docs.len());

        Ok(())
    }

    #[test]
    fn test_advance_single_parent_no_child() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let parent_f = sb.add_text_field("parent", STRING);
        let child_f = sb.add_text_field("child", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block1 => parent only
            writer.add_documents(vec![doc!(parent_f => "1")])?;
            // block2 => child + parent
            writer.add_documents(vec![doc!(child_f => "2"), doc!(parent_f => "2")])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct MyParents(Field, &'static str);
        impl ParentBitSetProducer for MyParents {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, self.1);
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parent_bits = Arc::new(MyParents(parent_f, "2"));
        let child_q = TermQuery::new(
            crate::Term::from_field_text(child_f, "2"),
            IndexRecordOption::Basic,
        );
        let join_q = ToParentBlockJoinQuery::new(Box::new(child_q), parent_bits, ScoreMode::Avg);

        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        // expect 1 => parent=2
        assert_eq!(1, top_docs.len());
        Ok(())
    }

    #[test]
    fn test_child_query_never_matches() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let doctype_f = sb.add_text_field("docType", STRING);
        let child_f = sb.add_text_field("childText", TEXT);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block1 => child + parent
            writer.add_documents(vec![
                doc!(child_f => "some text"),
                doc!(doctype_f => "resume"),
            ])?;
            // block2 => parent only
            writer.add_documents(vec![doc!(doctype_f => "resume")])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct ResumeBits(Field);
        impl ParentBitSetProducer for ResumeBits {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, "resume");
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parent_bits = Arc::new(ResumeBits(doctype_f));
        // child => never matches
        let child_q = TermQuery::new(
            crate::Term::from_field_text(child_f, "bogusbogus"),
            IndexRecordOption::Basic,
        );
        let join_q = ToParentBlockJoinQuery::new(Box::new(child_q), parent_bits, ScoreMode::Avg);
        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        assert_eq!(0, top_docs.len());
        Ok(())
    }

    #[test]
    fn test_advance_single_deleted_parent_no_child() -> crate::Result<()> {
        let mut sb = SchemaBuilder::default();
        let skill_f = sb.add_text_field("skill", STRING);
        let doctype_f = sb.add_text_field("docType", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;

            // block1 => child=java + parent
            writer.add_documents(vec![doc!(skill_f => "java"), doc!(doctype_f => "isparent")])?;

            // single parent => isparent
            writer.add_documents(vec![doc!(doctype_f => "isparent")])?;
            writer.commit()?;

            // delete by doctype=isparent (this removes *both* parents)
            let del_t = Term::from_field_text(doctype_f, "isparent");
            writer.delete_term(del_t);

            // re-add block => parent= isparent (but now it has no child)
            writer.add_documents(vec![doc!(doctype_f => "isparent")])?;

            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct PBits(Field);
        impl ParentBitSetProducer for PBits {
            fn produce(&self, reader: &SegmentReader) -> crate::Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = Term::from_field_text(self.0, "isparent");
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parents = Arc::new(PBits(doctype_f));

        // child => skill=java
        let cq = TermQuery::new(
            Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let join_q = ToParentBlockJoinQuery::new(Box::new(cq), parents, ScoreMode::Avg);

        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        // Because the original parent got deleted and the new parent has no child,
        // we now expect 0 hits.
        assert_eq!(0, top_docs.len());

        Ok(())
    }

    #[test]
    fn test_parent_scoring_bug() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let skill_f = sb.add_text_field("skill", STRING | STORED);
        let doctype_f = sb.add_text_field("docType", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block1 => (java, python), parent=resume
            writer.add_documents(vec![
                doc!(skill_f => "java"),
                doc!(skill_f => "python"),
                doc!(doctype_f => "resume"),
            ])?;
            // block2 => (java, ruby), parent=resume
            writer.add_documents(vec![
                doc!(skill_f => "java"),
                doc!(skill_f => "ruby"),
                doc!(doctype_f => "resume"),
            ])?;

            // delete all skill=java
            let del_term = crate::Term::from_field_text(skill_f, "java");
            writer.delete_term(del_term);

            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct ResumeBits(Field);
        impl ParentBitSetProducer for ResumeBits {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, "resume");
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parents = Arc::new(ResumeBits(doctype_f));

        // child => skill=java (but all were deleted)
        let cq = TermQuery::new(
            crate::Term::from_field_text(skill_f, "java"),
            IndexRecordOption::Basic,
        );
        let join_q = ToParentBlockJoinQuery::new(Box::new(cq), parents, ScoreMode::Avg);

        let top_docs = searcher.search(&join_q, &TopDocs::with_limit(10))?;
        // Probably 0 hits now. Just ensure no panic or weird zero-score bug
        for (sc, _) in &top_docs {
            assert_ne!(*sc, 0.0, "Unexpected zero aggregator bug");
        }

        Ok(())
    }

    #[test]
    fn test_to_child_initial_advance_parent_but_no_kids() -> Result<()> {
        // The first block => parent only, second => child + parent
        let mut sb = SchemaBuilder::default();
        let doctype_f = sb.add_text_field("docType", STRING);
        let skill_f = sb.add_text_field("skill", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block1 => parent only
            writer.add_documents(vec![{
                let mut d = doc!();
                d.add_text(doctype_f, "resume");
                d
            }])?;
            // block2 => child + parent
            writer.add_documents(vec![doc!(skill_f => "java"), {
                let mut d = doc!();
                d.add_text(doctype_f, "resume");
                d
            }])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct BitsetProd(Field);
        impl ParentBitSetProducer for BitsetProd {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, "resume");
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parents = Arc::new(BitsetProd(doctype_f));
        let parent_query = TermQuery::new(
            crate::Term::from_field_text(doctype_f, "resume"),
            IndexRecordOption::Basic,
        );
        let to_child = ToChildBlockJoinQuery::new(Box::new(parent_query), parents);
        let top_docs = searcher.search(&to_child, &TopDocs::with_limit(10))?;
        // we expect 1 child => skill=java
        assert_eq!(1, top_docs.len());
        Ok(())
    }

    #[test]
    fn test_score_mode() -> Result<()> {
        let mut sb = SchemaBuilder::default();
        let doctype_f = sb.add_text_field("docType", STRING);
        let skill_f = sb.add_text_field("skill", STRING);
        let schema = sb.build();

        let ram = RamDirectory::create();
        let index = Index::create(ram, schema.clone(), IndexSettings::default())?;
        {
            let mut writer = index.writer_for_tests()?;
            // block => 3 child docs skill=bar, then 1 parent => docType=resume
            writer.add_documents(vec![
                doc!(skill_f => "bar"),
                doc!(skill_f => "bar"),
                doc!(skill_f => "bar"),
                {
                    let mut d = doc!();
                    d.add_text(doctype_f, "resume");
                    d
                },
            ])?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        struct ResumeBits(Field);
        impl ParentBitSetProducer for ResumeBits {
            fn produce(&self, reader: &SegmentReader) -> Result<BitSet> {
                let max_doc = reader.max_doc();
                let mut bs = BitSet::with_max_value(max_doc);
                let inv = reader.inverted_index(self.0)?;
                let term = crate::Term::from_field_text(self.0, "resume");
                if let Some(mut postings) = inv.read_postings(&term, IndexRecordOption::Basic)? {
                    let mut d = postings.doc();
                    while d != TERMINATED {
                        bs.insert(d);
                        d = postings.advance();
                    }
                }
                Ok(bs)
            }
        }

        let parents = Arc::new(ResumeBits(doctype_f));
        let tq = TermQuery::new(
            crate::Term::from_field_text(skill_f, "bar"),
            IndexRecordOption::Basic,
        );

        for mode in &[
            ScoreMode::None,
            ScoreMode::Avg,
            ScoreMode::Max,
            ScoreMode::Min,
            ScoreMode::Total,
        ] {
            let join_q = ToParentBlockJoinQuery::new(Box::new(tq.clone()), parents.clone(), *mode);
            let hits = searcher.search(&join_q, &TopDocs::with_limit(10))?;
            // should yield 1 parent
            assert_eq!(1, hits.len(), "ScoreMode={:?} mismatch", mode);
        }

        Ok(())
    }
}

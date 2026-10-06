use super::Collector;
use crate::collector::SegmentCollector;
use crate::query::Weight;
use crate::{DocId, Score, SegmentOrdinal, SegmentReader};

/// `CountCollector` collector only counts how many
/// documents match the query.
///
/// ```rust
/// use tantivy::collector::Count;
/// use tantivy::query::QueryParser;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index};
///
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
///
/// let mut index_writer = index.writer(15_000_000).unwrap();
/// index_writer.add_document(doc!(title => "The Name of the Wind")).unwrap();
/// index_writer.add_document(doc!(title => "The Diary of Muadib")).unwrap();
/// index_writer.add_document(doc!(title => "A Dairy Cow")).unwrap();
/// index_writer.add_document(doc!(title => "The Diary of a Young Girl")).unwrap();
/// assert!(index_writer.commit().is_ok());
///
/// let reader = index.reader().unwrap();
/// let searcher = reader.searcher();
///
/// // Here comes the important part
/// let query_parser = QueryParser::for_index(&index, vec![title]);
/// let query = query_parser.parse_query("diary").unwrap();
/// let count = searcher.search(&query, &Count).unwrap();
///
/// assert_eq!(count, 2);
/// ```
pub struct Count;

impl Collector for Count {
    type Fruit = usize;

    type Child = SegmentCountCollector;

    fn for_segment(
        &self,
        _: SegmentOrdinal,
        _: &SegmentReader,
    ) -> crate::Result<SegmentCountCollector> {
        Ok(SegmentCountCollector::default())
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, segment_counts: Vec<usize>) -> crate::Result<usize> {
        Ok(segment_counts.into_iter().sum())
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        _segment_ord: u32,
        reader: &SegmentReader,
    ) -> crate::Result<usize> {
        Ok(weight.count(reader)? as usize)
    }
}

#[derive(Default)]
pub struct SegmentCountCollector {
    count: usize,
}

impl SegmentCollector for SegmentCountCollector {
    type Fruit = usize;

    fn collect(&mut self, _: DocId, _: Score) {
        self.count += 1;
    }

    fn collect_block(&mut self, docs: &[DocId]) {
        self.count += docs.len();
    }

    fn supports_bitmap_collection(&self) -> bool {
        true
    }

    fn collect_bitmap(&mut self, _base: DocId, mask: &crate::DocIdBitmap) {
        self.count += mask.iter().map(|word| word.len() as usize).sum::<usize>();
    }

    fn harvest(self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::{Count, SegmentCountCollector};
    use crate::collector::{Collector, SegmentCollector};

    struct BitmapOnlyCount;
    impl SegmentCollector for BitmapOnlyCount {
        type Fruit = ();
        fn collect(&mut self, _: crate::DocId, _: crate::Score) {
            panic!("count enumerated a bitmap");
        }
        fn collect_block(&mut self, _: &[crate::DocId]) {
            panic!("count enumerated a bitmap");
        }
        fn supports_bitmap_collection(&self) -> bool {
            true
        }

        fn collect_bitmap(&mut self, _: crate::DocId, _: &crate::DocIdBitmap) {}
        fn harvest(self) {}
    }

    #[test]
    fn bitmap_count_collectors_apply_deletes_without_enumeration() -> crate::Result<()> {
        use crate::aggregation::{AggContextParams, AggregationCollector};
        use crate::query::{EnableScoring, QueryParser};
        use crate::schema::{Schema, FAST, INDEXED, TEXT};
        use crate::{Index, Term};
        let mut schema = Schema::builder();
        let text = schema.add_text_field(
            "text",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_bitmap_postings(true),
            ),
        );
        let id = schema.add_u64_field("id", FAST | INDEXED);
        let optional = schema.add_u64_field("optional", FAST);
        let multiple = schema.add_u64_field("multiple", FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for doc in 0..2303 {
            let mut record = doc!(text => if doc % 2 == 0 { "a" } else { "b" }, id => doc as u64,
                multiple => 1u64, multiple => 2u64);
            if doc % 5 == 0 {
                record.add_u64(optional, 1);
            }
            writer.add_document(record)?;
        }
        writer.commit()?;
        for doc in (0..2303).step_by(7) {
            writer.delete_term(Term::from_field_u64(id, doc));
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let query = QueryParser::for_index(&index, vec![text]).parse_query("a OR b")?;
        let expected = (0..2303).filter(|doc| doc % 7 != 0).count();
        assert_eq!(searcher.search(&query, &Count)?, expected);
        assert_eq!(
            searcher.search(&query, &(Count, Some(Count)))?,
            (expected, Some(expected))
        );
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let term_query = QueryParser::for_index(&index, vec![text]).parse_query("a")?;
        let term_weight = term_query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut ordinary_reader = searcher.segment_reader(0).clone();
        assert!(term_weight.scorer(&ordinary_reader, 1.0)?.has_fast_bitset());
        ordinary_reader.bitmap_postings_enabled = false;
        assert!(!term_weight.scorer(&ordinary_reader, 1.0)?.has_fast_bitset());
        let mut proof = BitmapOnlyCount;
        super::super::default_collect_segment_impl(
            &mut proof,
            weight.as_ref(),
            searcher.segment_reader(0),
            false,
        )?;
        let request = serde_json::from_value(serde_json::json!({
            "documents": {"filter": "*"},
            "filtered": {"filter": "text:a"},
            "nested": {"filter": "text:a", "aggs": {
                "sum": {"sum": {"field": "id"}},
                "max": {"max": {"field": "id"}}
            }},
            "values": {"value_count": {"field": "id"}},
            "optional": {"value_count": {"field": "optional"}},
            "multiple": {"value_count": {"field": "multiple"}}
        }))?;
        let result = searcher.search(
            &query,
            &AggregationCollector::from_aggs(request, AggContextParams::default()),
        )?;
        let json = serde_json::to_value(result)?;
        assert_eq!(json["documents"]["doc_count"], expected);
        let filtered: Vec<u64> = (0..2303)
            .filter(|doc| doc % 7 != 0 && doc % 2 == 0)
            .collect();
        assert_eq!(json["filtered"]["doc_count"], filtered.len());
        assert_eq!(json["nested"]["doc_count"], filtered.len());
        assert_eq!(
            json["nested"]["sum"]["value"],
            filtered.iter().sum::<u64>() as f64
        );
        assert_eq!(
            json["nested"]["max"]["value"],
            *filtered.last().unwrap() as f64
        );
        assert_eq!(json["values"]["value"], expected as f64);
        assert_eq!(
            json["optional"]["value"],
            (0..2303).filter(|doc| doc % 7 != 0 && doc % 5 == 0).count() as f64
        );
        assert_eq!(json["multiple"]["value"], (expected * 2) as f64);
        Ok(())
    }

    #[test]
    fn aggregation_bitmap_execution_is_count_only() -> crate::Result<()> {
        use crate::aggregation::{AggContextParams, AggregationCollector};
        use crate::query::{AllWeight, Explanation, Scorer, Weight};
        use crate::schema::{Schema, FAST};
        use crate::{DocId, Index, Score, SegmentReader};

        struct CheckedWeight(bool);
        impl Weight for CheckedWeight {
            fn scorer(
                &self,
                reader: &SegmentReader,
                boost: Score,
            ) -> crate::Result<Box<dyn Scorer>> {
                assert_eq!(reader.bitmap_postings_enabled, self.0);
                AllWeight.scorer(reader, boost)
            }
            fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
                AllWeight.explain(reader, doc)
            }
            fn for_each_no_score_batch(
                &self,
                reader: &SegmentReader,
                callback: &mut dyn FnMut(crate::DocSetBatch<'_>),
            ) -> crate::Result<()> {
                assert!(self.0, "non-count request entered bitmap execution");
                assert!(reader.bitmap_postings_enabled);
                AllWeight.for_each_no_score_batch(reader, callback)
            }
        }
        let mut schema = Schema::builder();
        let value = schema.add_u64_field("value", FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(value => 7u64))?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        for (request, bitmap) in [
            (serde_json::json!({"a":{"filter":"*"}}), true),
            (
                serde_json::json!({"a":{"filter":"*"}, "b":{"filter":"*"}}),
                true,
            ),
            (serde_json::json!({"a":{"sum":{"field":"value"}}}), false),
            (
                serde_json::json!({"a":{"value_count":{"field":"value"}}}),
                false,
            ),
            (
                serde_json::json!({"a":{"filter":"*"}, "b":{"sum":{"field":"value"}}}),
                false,
            ),
            (
                serde_json::json!({"a":{"filter":"*", "aggs":{"b":{"sum":{"field":"value"}}}}}),
                false,
            ),
            (serde_json::json!({"a":{"terms":{"field":"value"}}}), false),
        ] {
            let collector = AggregationCollector::from_aggs(
                serde_json::from_value(request)?,
                AggContextParams::default(),
            );
            collector.collect_segment(&CheckedWeight(bitmap), 0, reader)??;
            let mut multi = crate::collector::MultiCollector::new();
            multi.add_collector(Count);
            multi.add_collector(collector);
            multi.collect_segment(&CheckedWeight(bitmap), 0, reader)?;
        }
        (Count, Some(Count)).collect_segment(&CheckedWeight(true), 0, reader)?;
        assert!(reader.bitmap_postings_enabled);
        Ok(())
    }

    #[test]
    fn test_count_collect_does_not_requires_scoring() {
        assert!(!Count.requires_scoring());
    }

    #[test]
    fn test_segment_count_collector() {
        {
            let count_collector = SegmentCountCollector::default();
            assert_eq!(count_collector.harvest(), 0);
        }
        {
            let mut count_collector = SegmentCountCollector::default();
            count_collector.collect(0u32, 1.0);
            assert_eq!(count_collector.harvest(), 1);
        }
        {
            let mut count_collector = SegmentCountCollector::default();
            count_collector.collect(0u32, 1.0);
            assert_eq!(count_collector.harvest(), 1);
        }
        {
            let mut count_collector = SegmentCountCollector::default();
            count_collector.collect(0u32, 1.0);
            count_collector.collect(1u32, 1.0);
            assert_eq!(count_collector.harvest(), 2);
        }
    }
}

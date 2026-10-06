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
        let mut proof = BitmapOnlyCount;
        super::super::default_collect_segment_impl(
            &mut proof,
            weight.as_ref(),
            searcher.segment_reader(0),
            false,
        )?;
        let request = serde_json::from_value(serde_json::json!({
            "documents": {"filter": "*"},
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
        assert_eq!(json["values"]["value"], expected as f64);
        assert_eq!(
            json["optional"]["value"],
            (0..2303).filter(|doc| doc % 7 != 0 && doc % 5 == 0).count() as f64
        );
        assert_eq!(json["multiple"]["value"], (expected * 2) as f64);
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

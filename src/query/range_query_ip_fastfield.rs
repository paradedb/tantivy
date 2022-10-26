//! IP Fastfields support efficient scanning for range queries.
//! We use this variant only if the fastfield exists, otherwise the default in `range_query` is
//! used, which uses the term dictionary + postings.

use std::net::Ipv6Addr;
use std::ops::{Bound, RangeInclusive};
use std::sync::Arc;

use common::BinarySerializable;
use fastfield_codecs::{Column, MonotonicallyMappableToU128};

use super::range_query::map_bound;
use super::{ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Cardinality, Field};
use crate::{DocId, DocSet, Score, SegmentReader, TantivyError, TERMINATED};

/// `IPFastFieldRangeWeight` uses the ip address fast field to execute range queries.
pub struct IPFastFieldRangeWeight {
    field: Field,
    left_bound: Bound<Ipv6Addr>,
    right_bound: Bound<Ipv6Addr>,
}

impl IPFastFieldRangeWeight {
    pub fn new(field: Field, left_bound: &Bound<Vec<u8>>, right_bound: &Bound<Vec<u8>>) -> Self {
        let ip_from_bound_raw_data = |data: &Vec<u8>| {
            let left_ip_u128: u128 =
                u128::from_be(BinarySerializable::deserialize(&mut &data[..]).unwrap());
            Ipv6Addr::from_u128(left_ip_u128)
        };
        let left_bound = map_bound(left_bound, &ip_from_bound_raw_data);
        let right_bound = map_bound(right_bound, &ip_from_bound_raw_data);
        Self {
            field,
            left_bound,
            right_bound,
        }
    }
}

impl Weight for IPFastFieldRangeWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let field_type = reader.schema().get_field_entry(self.field).field_type();
        match field_type.fastfield_cardinality().unwrap() {
            Cardinality::SingleValue => {
                let ip_addr_fast_field = reader.fast_fields().ip_addr(self.field)?;
                let value_range = bound_to_value_range(
                    &self.left_bound,
                    &self.right_bound,
                    ip_addr_fast_field.as_ref(),
                );
                let docset = IpRangeDocSet::new(value_range, ip_addr_fast_field);
                Ok(Box::new(ConstScorer::new(docset, boost)))
            }
            Cardinality::MultiValues => unimplemented!(),
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({}) does not match",
                doc
            )));
        }
        let explanation = Explanation::new("Const", scorer.score());

        Ok(explanation)
    }
}

fn bound_to_value_range(
    left_bound: &Bound<Ipv6Addr>,
    right_bound: &Bound<Ipv6Addr>,
    column: &dyn Column<Ipv6Addr>,
) -> RangeInclusive<Ipv6Addr> {
    let start_value = match left_bound {
        Bound::Included(ip_addr) => *ip_addr,
        Bound::Excluded(ip_addr) => Ipv6Addr::from(ip_addr.to_u128() + 1),
        Bound::Unbounded => column.min_value(),
    };

    let end_value = match right_bound {
        Bound::Included(ip_addr) => *ip_addr,
        Bound::Excluded(ip_addr) => Ipv6Addr::from(ip_addr.to_u128() - 1),
        Bound::Unbounded => column.max_value(),
    };
    start_value..=end_value
}

/// Helper to have a cursor over a vec of docids
struct VecCursor {
    docs: Vec<u32>,
    current_pos: usize,
}
impl VecCursor {
    fn new() -> Self {
        Self {
            docs: Vec::with_capacity(32),
            current_pos: 0,
        }
    }
    fn next(&mut self) -> Option<u32> {
        self.current_pos += 1;
        self.current()
    }
    #[inline]
    fn current(&self) -> Option<u32> {
        self.docs.get(self.current_pos).map(|el| *el as u32)
    }

    fn set_data(&mut self, data: Vec<u32>) {
        self.docs = data;
        self.current_pos = 0;
    }
    fn is_empty(&self) -> bool {
        self.current_pos >= self.docs.len()
    }
}

struct IpRangeDocSet {
    /// The range filter on the values.
    value_range: RangeInclusive<Ipv6Addr>,
    ip_addr_fast_field: Arc<dyn Column<Ipv6Addr>>,
    /// The last docid (exclusive) that has been fetched.
    fetched_until_doc: u32,
    /// Current batch of loaded docs.
    loaded_docs: VecCursor,
}
impl IpRangeDocSet {
    fn new(
        value_range: RangeInclusive<Ipv6Addr>,
        ip_addr_fast_field: Arc<dyn Column<Ipv6Addr>>,
    ) -> Self {
        let mut ip_range_docset = Self {
            value_range,
            ip_addr_fast_field,
            loaded_docs: VecCursor::new(),
            fetched_until_doc: 0,
        };
        ip_range_docset.fetch_block();
        ip_range_docset
    }

    /// Returns true if more data could be fetched
    fn fetch_block(&mut self) {
        let mut horizon: u32 = 1;
        const MAX_HORIZON: u32 = 100_000;
        while self.loaded_docs.is_empty() {
            let finished_to_end = self.fetch_horizon(horizon);
            if finished_to_end {
                break;
            }
            // Fetch more data, increase horizon
            horizon = (horizon * 2).min(MAX_HORIZON);
        }
    }

    /// Fetches a block for docid range [fetched_until_doc .. fetched_until_doc + HORIZON]
    fn fetch_horizon(&mut self, horizon: u32) -> bool {
        let mut end = self.fetched_until_doc + horizon;
        let mut finished_to_end = false;

        let limit = self.ip_addr_fast_field.num_vals();
        if end >= limit {
            end = limit;
            finished_to_end = true;
        }

        let data = self
            .ip_addr_fast_field
            .get_positions_for_value_range(self.value_range.clone(), self.fetched_until_doc..end);
        self.loaded_docs.set_data(data);
        self.fetched_until_doc = end;
        finished_to_end
    }
}

impl DocSet for IpRangeDocSet {
    fn advance(&mut self) -> DocId {
        if let Some(docid) = self.loaded_docs.next() {
            docid as u32
        } else {
            if self.fetched_until_doc >= self.ip_addr_fast_field.num_vals() as u32 {
                return TERMINATED;
            }
            self.fetch_block();
            self.loaded_docs.current().unwrap_or(TERMINATED)
        }
    }

    fn doc(&self) -> DocId {
        self.loaded_docs
            .current()
            .map(|el| el as u32)
            .unwrap_or(TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0 // heuristic possible by checking number of hits when fetching a block
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::ProptestConfig;
    use proptest::strategy::Strategy;
    use proptest::{prop_oneof, proptest};

    use super::*;
    use crate::collector::Count;
    use crate::query::QueryParser;
    use crate::schema::{Schema, FAST, INDEXED, STORED, TEXT};
    use crate::Index;

    #[derive(Clone, Debug)]
    struct Doc {
        id: String,
        ip: Ipv6Addr,
    }

    fn operation_strategy() -> impl Strategy<Value = Doc> {
        prop_oneof![
            (0u64..100u64).prop_map(doc_from_id_1),
            (1u64..100u64).prop_map(doc_from_id_2),
        ]
    }

    fn doc_from_id_1(id: u64) -> Doc {
        Doc {
            // ip != id
            id: id.to_string(),
            ip: Ipv6Addr::from_u128(id as u128),
        }
    }
    fn doc_from_id_2(id: u64) -> Doc {
        Doc {
            // ip != id
            id: (id - 1).to_string(),
            ip: Ipv6Addr::from_u128(id as u128),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        #[test]
        fn test_ip_range_for_docs_prop(ops in proptest::collection::vec(operation_strategy(), 1..100)) {
            assert!(test_ip_range_for_docs(ops).is_ok());
        }
    }

    #[test]
    fn ip_range_regression1_test() {
        let ops = vec![
            doc_from_id_1(52),
            doc_from_id_1(63),
            doc_from_id_1(12),
            doc_from_id_2(91),
            doc_from_id_2(33),
        ];
        assert!(test_ip_range_for_docs(ops).is_ok());
    }

    #[test]
    fn ip_range_regression2_test() {
        let ops = vec![doc_from_id_1(0)];
        assert!(test_ip_range_for_docs(ops).is_ok());
    }

    fn test_ip_range_for_docs(docs: Vec<Doc>) -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let ip_field = schema_builder.add_ip_addr_field("ip", INDEXED | STORED | FAST);
        let text_field = schema_builder.add_text_field("id", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        {
            let mut index_writer = index.writer(3_000_000).unwrap();
            for doc in &docs {
                index_writer
                    .add_document(doc!(
                        ip_field => doc.ip,
                        text_field => doc.id.to_string()
                    ))
                    .unwrap();
            }

            index_writer.commit().unwrap();
        }
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let get_num_hits = |query| searcher.search(&query, &(Count)).unwrap();
        let query_from_text = |text: &str| {
            QueryParser::for_index(&index, vec![])
                .parse_query(text)
                .unwrap()
        };

        let gen_query_inclusive = |from: Ipv6Addr, to: Ipv6Addr| {
            format!("ip:[{} TO {}]", &from.to_string(), &to.to_string())
        };

        let test_sample = |sample_docs: Vec<Doc>| {
            let mut ips: Vec<Ipv6Addr> = sample_docs.iter().map(|doc| doc.ip).collect();
            ips.sort();
            let expected_num_hits = docs
                .iter()
                .filter(|doc| (ips[0]..=ips[1]).contains(&doc.ip))
                .count();

            let query = gen_query_inclusive(ips[0], ips[1]);
            assert_eq!(get_num_hits(query_from_text(&query)), expected_num_hits);

            // Intersection search
            let id_filter = sample_docs[0].id.to_string();
            let expected_num_hits = docs
                .iter()
                .filter(|doc| (ips[0]..=ips[1]).contains(&doc.ip) && doc.id == id_filter)
                .count();
            let query = format!("{} AND id:{}", query, &id_filter);
            assert_eq!(get_num_hits(query_from_text(&query)), expected_num_hits);
        };

        test_sample(vec![docs[0].clone(), docs[0].clone()]);
        if docs.len() > 1 {
            test_sample(vec![docs[0].clone(), docs[1].clone()]);
            test_sample(vec![docs[1].clone(), docs[1].clone()]);
        }
        if docs.len() > 2 {
            test_sample(vec![docs[1].clone(), docs[2].clone()]);
        }

        Ok(())
    }
}
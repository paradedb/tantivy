use binggan::{black_box, BenchGroup};
use rand::prelude::*;
use rand::rngs::StdRng;
use tantivy::collector::Collector;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, FAST, TEXT};
use tantivy::{doc, DocId, Index, ReloadPolicy, Searcher, SegmentOrdinal};

#[derive(Clone)]
pub struct BenchIndex {
    #[allow(dead_code)]
    pub index: Index,
    pub searcher: Searcher,
    pub query_parser: QueryParser,
}

pub fn format_pct(p: f32) -> String {
    let pct = (p as f64) * 100.0;
    let rounded = (pct * 1_000_000.0).round() / 1_000_000.0;
    if rounded.fract() <= 0.001 {
        format!("{}%", rounded as u64)
    } else {
        format!("{}%", rounded)
    }
}

pub fn query_label(query_str: &str, term_pcts: &[(&str, String)]) -> String {
    let mut label = query_str.to_string();
    for (term, pct) in term_pcts {
        label = label.replace(term, pct);
    }
    label.replace(' ', "_")
}

#[allow(dead_code)]
pub trait FruitCount {
    fn count(&self) -> usize;
}

impl FruitCount for usize {
    fn count(&self) -> usize {
        *self
    }
}

impl<T> FruitCount for Vec<T> {
    fn count(&self) -> usize {
        self.len()
    }
}

impl<A: FruitCount, B> FruitCount for (A, B) {
    fn count(&self) -> usize {
        self.0.count()
    }
}

#[allow(dead_code)]
pub fn add_bench_task<C: Collector + 'static>(
    bench_group: &mut BenchGroup,
    bench_index: &BenchIndex,
    query_str: &str,
    collector: C,
    collector_name: &str,
) where
    C::Fruit: FruitCount,
{
    let query = bench_index.query_parser.parse_query(query_str).unwrap();
    let searcher = bench_index.searcher.clone();
    bench_group.register(collector_name.to_string(), move |_| {
        black_box(searcher.search(&query, &collector).unwrap().count())
    });
}

pub fn build_index(
    num_docs: usize,
    num_segments: usize,
    terms: &[(&str, f32)],
    mut doc_generator: impl FnMut(SegmentOrdinal, DocId, &mut StdRng) -> (usize, usize),
) -> (BenchIndex, BenchIndex) {
    let mut schema_builder = Schema::builder();
    let f_title = schema_builder.add_text_field("title", TEXT);
    let f_body = schema_builder.add_text_field("body", TEXT);
    let f_score = schema_builder.add_u64_field("score", FAST);
    let f_score2 = schema_builder.add_u64_field("score2", FAST);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());

    let mut rng = StdRng::from_seed([7u8; 32]);

    {
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        let docs_per_segment = num_docs / num_segments;
        for seg_ord in 0..num_segments {
            for doc_id in 0..docs_per_segment {
                let (target_multiplier, filler_count) = doc_generator(seg_ord as SegmentOrdinal, doc_id as DocId, &mut rng);
                let score = rng.random_range(0u64..100_000u64);
                let score2 = rng.random_range(0u64..100_000u64);
                let mut title_tokens: Vec<&str> = Vec::new();
                let mut body_tokens: Vec<&str> = Vec::new();
                for &(tok, prob) in terms {
                    if rng.random_bool(prob as f64) {
                        for _ in 0..target_multiplier {
                            if rng.random_bool(0.1) {
                                title_tokens.push(tok);
                            } else {
                                body_tokens.push(tok);
                            }
                        }
                    }
                }
                for _ in 0..filler_count {
                    body_tokens.push("z");
                }
                if title_tokens.is_empty() && body_tokens.is_empty() {
                    body_tokens.push("z");
                }
                writer
                    .add_document(doc!(
                        f_title=>title_tokens.join(" "),
                        f_body=>body_tokens.join(" "),
                        f_score=>score,
                        f_score2=>score2,
                    ))
                    .unwrap();
            }
            writer.commit().unwrap();
        }
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();

    let qp_single = QueryParser::for_index(&index, vec![f_body]);
    let qp_multi = QueryParser::for_index(&index, vec![f_title, f_body]);

    let only_title = BenchIndex {
        index: index.clone(),
        searcher: searcher.clone(),
        query_parser: qp_single,
    };
    let title_and_body = BenchIndex {
        index,
        searcher,
        query_parser: qp_multi,
    };
    (only_title, title_and_body)
}

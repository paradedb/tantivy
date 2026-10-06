use std::hint::black_box;
use std::time::Instant;

use serde_json::json;
use tantivy::aggregation::AggregationCollector;
use tantivy::collector::{Count, TopDocs};
use tantivy::index::BitmapPostingsConfig;
use tantivy::query::{EnableScoring, Query, QueryParser};
use tantivy::schema::{Schema, FAST, TEXT};
use tantivy::{doc, DocSetBatch, Index, IndexSettings, Searcher};

fn build(num_docs: u32, bitmaps: bool, density: u8) -> tantivy::Result<Index> {
    let mut schema = Schema::builder();
    let field = schema.add_text_field(
        "text",
        TEXT.set_indexing_options(
            TEXT.get_indexing_options()
                .unwrap()
                .clone()
                .set_bitmap_postings(bitmaps),
        ),
    );
    let number = schema.add_u64_field("number", FAST);
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            bitmap_postings: BitmapPostingsConfig {
                min_density_percent: density,
                ..Default::default()
            },
            ..Default::default()
        })
        .create_in_ram()?;
    let start = Instant::now();
    let mut writer = index.writer_with_num_threads(1, 64_000_000)?;
    let mut seed = 17u64;
    for ordinal in 0..num_docs {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut terms = vec!["all"];
        for (term, shift, threshold) in
            [("a", 0, 50), ("b", 8, 33), ("c", 16, 12), ("medium", 24, 6)]
        {
            if (seed >> shift) % 100 < threshold {
                terms.push(term);
            }
        }
        if ordinal % 997 == 0 {
            terms.push("rare");
        }
        writer.add_document(
            doc!(field => terms.join(" "), number => (ordinal as u64 * 104729) % num_docs as u64),
        )?;
    }
    writer.commit()?;
    drop(writer);
    let searcher = index.reader()?.searcher();
    let usage = searcher.space_usage()?;
    let bitmap_bytes: u64 = usage
        .segments()
        .iter()
        .map(|segment| {
            segment
                .component(tantivy::index::SegmentComponent::PostingBitmaps)
                .total()
                .get_bytes()
        })
        .sum();
    println!(
        "{}",
        json!({"build": if bitmaps {"bitmap"} else {"ordinary"}, "docs":num_docs,
        "density_percent": density, "seconds":start.elapsed().as_secs_f64(), "bytes":usage.total().get_bytes(), "bitmap_bytes":bitmap_bytes})
    );
    Ok(index)
}

fn run(searcher: &Searcher, query: &dyn Query, collector: &str) -> tantivy::Result<usize> {
    Ok(match collector {
        "count" => searcher.search(query, &Count)?,
        "wrapped" => searcher.search(query, &Some(Count))?.unwrap(),
        "aggregation" => {
            let aggs = serde_json::from_value(json!({"count":{"filter":"*"}}))?;
            let result = searcher.search(
                query,
                &AggregationCollector::from_aggs(aggs, Default::default()),
            )?;
            serde_json::to_value(result)?["count"]["doc_count"]
                .as_u64()
                .unwrap() as usize
        }
        "top10" => searcher
            .search(query, &TopDocs::with_limit(10).order_by_score())?
            .len(),
        _ => unreachable!(),
    })
}

fn main() -> tantivy::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let num_docs = args.get(1).map_or(200_000, |value| value.parse().unwrap());
    let rounds = args
        .get(2)
        .map_or(31, |value| value.parse::<usize>().unwrap());
    let density = args.get(3).map_or(10, |value| value.parse().unwrap());
    let ordinary = build(num_docs, false, density)?;
    let mut bitmap = build(num_docs, true, density)?;
    let ordinary_reader = ordinary.reader()?;
    bitmap.settings_mut().bitmap_postings.use_for_queries = false;
    let disabled_reader = bitmap.reader()?;
    bitmap.settings_mut().bitmap_postings.use_for_queries = true;
    let enabled_reader = bitmap.reader()?;
    let searchers = [
        ordinary_reader.searcher(),
        disabled_reader.searcher(),
        enabled_reader.searcher(),
    ];
    let names = ["ordinary", "bitmap_disabled", "bitmap_enabled"];
    let parser = QueryParser::for_index(&bitmap, vec![bitmap.schema().get_field("text")?]);
    for expression in [
        "a OR b",
        "a AND b",
        "(a OR b) AND c",
        "(a AND b) OR c",
        "(a OR b) -c",
        "rare AND a",
        "rare OR a",
        "rare AND medium",
        "medium AND c",
        "a AND number:[100 TO 1000]",
        "a OR number:[100 TO 1000]",
        "a AND number:[100 TO 100000]",
    ] {
        if args.get(4).is_some_and(|filter| filter != expression) {
            continue;
        }
        let query = parser.parse_query(expression)?;
        for (name, searcher) in names.iter().zip(&searchers) {
            let weight = query.weight(EnableScoring::disabled_from_searcher(searcher))?;
            let mut bitmap_batches = 0usize;
            let mut enumerated_docs = 0usize;
            for segment in searcher.segment_readers() {
                weight.for_each_no_score_batch(segment, &mut |batch| match batch {
                    DocSetBatch::Docs(docs) => enumerated_docs += docs.len(),
                    DocSetBatch::Bitmap(..) => bitmap_batches += 1,
                })?;
            }
            println!(
                "{}",
                json!({"mechanism":expression,"variant":name,"bitmap_batches":bitmap_batches,"enumerated_docs":enumerated_docs})
            );
        }
        for collector in ["count", "wrapped", "aggregation", "top10"] {
            if args.get(6).is_some_and(|filter| filter != collector) {
                continue;
            }
            let expected = run(&searchers[0], query.as_ref(), collector)?;
            let mut timings: [Vec<f64>; 3] = Default::default();
            for round in 0..rounds + 3 {
                for offset in 0..3 {
                    let i = (round + offset) % 3;
                    if args.get(5).is_some_and(|filter| filter != names[i]) {
                        continue;
                    }
                    let start = Instant::now();
                    let result = black_box(run(&searchers[i], query.as_ref(), collector)?);
                    let micros = start.elapsed().as_secs_f64() * 1e6;
                    assert_eq!(result, expected, "{expression}, {collector}, {}", names[i]);
                    if round >= 3 {
                        timings[i].push(micros);
                    }
                }
            }
            for (i, times) in timings.iter_mut().enumerate() {
                if times.is_empty() {
                    continue;
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"query":expression,"collector":collector,"variant":names[i],
                    "count":expected,"rounds":rounds,"median_us":times[times.len()/2],"p95_us":times[(times.len()*95/100).min(times.len()-1)]})
                );
            }
        }
    }
    Ok(())
}

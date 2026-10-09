use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::json;
use tantivy::collector::{Count, TopDocs};
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, INDEXED, TEXT};
use tantivy::{doc, Index};

fn main() -> tantivy::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(
        args.get(1)
            .expect("usage: sparse_disjunction_count INDEX [LABEL]"),
    );
    let label = args.get(2).map(String::as_str).unwrap_or("unknown");
    if !path.join("meta.json").exists() {
        std::fs::create_dir_all(path)?;
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let other = schema.add_text_field("other", TEXT);
        let id = schema.add_u64_field("id", INDEXED);
        let index = Index::create_in_dir(path, schema.build())?;
        let mut writer = index.writer_with_num_threads(1, 100_000_000)?;
        for i in 0..2_000_003u64 {
            let mut terms = vec!["all".to_owned()];
            if i % 5 != 0 {
                terms.push("dense".into());
            }
            if i % 3 != 0 {
                terms.push("second".into());
            }
            if i % 8 == 0 {
                terms.push("border".into());
            }
            if i % 100 == 0 {
                terms.push("medium".into());
            }
            if i % 2003 == 0 {
                terms.push("sparse".into());
            }
            if i % 20011 == 0 {
                terms.push("rare0".into());
                terms.push("shared".into());
                terms.push("unusual phrase".into());
                terms.push(if i % 5 == 0 { "outside" } else { "inside" }.into());
            }
            if i % 1009 == 0 {
                terms.push(format!("rare{}", 1 + (i / 1009) % 63));
            }
            if (10_000..10_100).contains(&i) {
                terms.push("clustered".into());
            }
            writer.add_document(doc!(text => terms.join(" "), other => if i % 30011 == 0 { "rare" } else { "filler" }, id => i))?;
        }
        writer.commit()?;
        let segments = index.searchable_segment_ids()?;
        writer.merge(&segments).wait()?;
        writer.wait_merging_threads()?;
    }
    let mut index = Index::open_in_dir(path)?;
    let text = index.schema().get_field("text")?;
    let parser = QueryParser::for_index(&index, vec![text]);
    index.settings_mut().bitmap_postings.use_for_queries = false;
    let ordinary = index.reader()?.searcher();
    index.settings_mut().bitmap_postings.use_for_queries = true;
    let bitmap = index.reader()?.searcher();
    let mut cases: Vec<(String, String)> = [
        ("one_rare", "dense OR rare0"),
        ("no_overlap", "dense OR outside"),
        ("complete_overlap", "dense OR inside"),
        ("rare_overlap", "dense OR rare0 OR shared"),
        ("duplicates", "dense OR rare0 OR rare0"),
        ("clustered", "dense OR clustered"),
        ("nested", "(dense OR rare0) OR (rare1 OR rare2)"),
        ("phrase", "dense OR \"unusual phrase\""),
        ("conjunction", "dense OR (rare0 AND shared)"),
        ("cross_field", "dense OR other:rare"),
        ("boost", "dense^3 OR rare0"),
        ("exclusion", "+(dense OR rare0 OR rare1) -shared"),
        ("dense_exclusion", "+dense -shared"),
        ("sparse_boundary", "dense OR sparse"),
        ("medium_control", "dense OR medium"),
        ("two_dense_control", "dense OR second OR rare0"),
        ("bitmap_boundary_control", "dense OR border"),
        ("all_sparse_control", "rare0 OR rare1 OR rare2"),
        ("intersection_control", "dense AND rare0"),
        ("match_all", "all OR rare0"),
        ("missing", "dense OR nonexistent"),
    ]
    .into_iter()
    .map(|(name, query)| (name.into(), query.into()))
    .collect();
    for n in [3, 15, 63] {
        let query = std::iter::once("dense".to_owned())
            .chain((0..n).map(|i| format!("rare{i}")))
            .collect::<Vec<_>>()
            .join(" OR ");
        cases.push((format!("{n}_rare"), query));
    }
    for (name, expression) in cases {
        let query = parser.parse_query(&expression)?;
        let expected = ordinary.search(&query, &Count)?;
        assert_eq!(bitmap.search(&query, &Count)?, expected, "{expression}");
        let mut times = Vec::new();
        for round in 0..25 {
            let start = Instant::now();
            let mut iterations = 0;
            while start.elapsed() < Duration::from_millis(20) {
                assert_eq!(
                    black_box(bitmap.search(black_box(&query), &Count)?),
                    expected
                );
                iterations += 1;
            }
            if round >= 4 {
                times.push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"label":label,"case":name,"query":expression,"count":expected,"median_us":times[times.len()/2],"p90_us":times[times.len()*9/10],"samples_us":times,"segments":bitmap.segment_readers().len()})
        );
    }
    let query = parser.parse_query("dense OR rare0 OR rare1")?;
    let expected = ordinary.search(&query, &TopDocs::with_limit(10).order_by_score())?;
    let mut times = Vec::new();
    for _ in 0..25 {
        let start = Instant::now();
        assert_eq!(
            bitmap.search(&query, &TopDocs::with_limit(10).order_by_score())?,
            expected
        );
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"label":label,"case":"scored_control","median_us":times[12]})
    );
    Ok(())
}

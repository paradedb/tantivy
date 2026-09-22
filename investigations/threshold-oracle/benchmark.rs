use super::*;
use crate::fieldnorm::threshold_trace as trace;

#[test]
#[ignore]
fn replay_hn_threshold_oracle() {
    let directory = std::env::var("MIN_NORM_EXPORT").expect("MIN_NORM_EXPORT");
    let dir = Path::new(&directory);
    let output = std::env::var("THRESHOLD_OUTPUT").expect("THRESHOLD_OUTPUT");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let segments = manifest["segments"].as_array().unwrap();
    let total_docs = manifest["global_num_docs"].as_u64().unwrap();
    let avg = manifest["global_total_tokens"].as_u64().unwrap() as f32 / total_docs as f32;
    let payload_size = manifest["storage_payload_bytes"].as_u64().unwrap() as usize;
    let field = dir.file_name().unwrap().to_str().unwrap();
    let k = 10;
    let mut results = Vec::new();
    for (term, name) in manifest["terms"].as_array().unwrap().iter().enumerate() {
        if name != "database" && !(field == "text" && name == "postgres") {
            continue;
        }
        let weight = Bm25Weight::for_one_term(
            manifest["global_doc_freqs"][term].as_u64().unwrap(),
            total_docs,
            avg,
            Bm25Params::default(),
        );
        let mut expected = Vec::new();
        let mut inputs = Vec::new();
        let mut all_scores = Vec::new();
        let mut full_blocks = 0;
        for (ordinal, segment) in segments.iter().enumerate() {
            let info = &segment["terms"][term];
            let count = info["doc_freq"].as_u64().unwrap() as usize;
            assert!(count > 0);
            full_blocks += count / 128;
            let stem = info["stem"].as_str().unwrap();
            let records = std::fs::read(dir.join(format!("{stem}.records"))).unwrap();
            assert_eq!(records.len(), count * 24);
            let raw = std::fs::read(dir.join(format!("{stem}.raw"))).unwrap();
            let norm_bytes = OwnedBytes::new(
                std::fs::read(dir.join(format!("{}.norms", segment["stem"].as_str().unwrap())))
                    .unwrap(),
            );
            for record in records.chunks_exact(24) {
                let doc = u32_at(record, 0);
                let score = f32::from_le_bytes(record[16..20].try_into().unwrap());
                assert_eq!(score, weight.score(record[20], u32_at(record, 4)));
                assert_eq!(record[20], norm_bytes[doc as usize]);
                assert_eq!(record[21], 1);
                collect(&mut expected, (score, ordinal, doc), k);
                all_scores.push(score);
            }
            inputs.push((records, raw, norm_bytes));
        }
        let final_threshold = expected.last().unwrap().0;
        let mut runs = Vec::new();
        for format in ["original", "zero", "tf_class"] {
            let encoded: Vec<Vec<u8>> = inputs
                .iter()
                .map(|(records, raw, _)| {
                    rewrite(raw, records, if format == "tf_class" { 2 } else { 0 })
                })
                .collect();
            for policy in [
                "normal_strict",
                "normal_tie_safe",
                "oracle",
                "seeded",
                "oracle_address",
                "seeded_address",
            ] {
                let fixed = policy.starts_with("oracle") || policy.starts_with("seeded");
                let seeded = policy.starts_with("seeded");
                let address_aware = policy.ends_with("address");
                let last_winner = *expected.last().unwrap();
                let mut heap = if seeded { expected.clone() } else { Vec::new() };
                let mut threshold = if fixed {
                    final_threshold.next_down()
                } else {
                    Score::MIN
                };
                let reads = Arc::new(Mutex::new(Reads::default()));
                let mut returned = 0usize;
                let mut seeded_duplicates = 0usize;
                trace::start(format == "original", threshold);
                for (ordinal, segment) in segments.iter().enumerate() {
                    if address_aware {
                        threshold = if ordinal <= last_winner.1 {
                            final_threshold.next_down()
                        } else {
                            final_threshold
                        };
                    }
                    trace::segment(ordinal, threshold);
                    let norms = FieldNormReader::open(FileSlice::new(Arc::new(Norms {
                        data: inputs[ordinal].2.clone(),
                        reads: reads.clone(),
                        segment: ordinal,
                        base: segment["norm_base"].as_u64().unwrap() as usize,
                        payload_size,
                    })));
                    let block = BlockSegmentPostings::open(
                        (inputs[ordinal].0.len() / 24) as u32,
                        OwnedBytes::new(encoded[ordinal].clone()),
                        IndexRecordOption::WithFreqsAndPositions,
                        IndexRecordOption::WithFreqs,
                    )
                    .unwrap();
                    let scorer = TermScorer::new(
                        SegmentPostings::from_block_postings(block, None),
                        norms,
                        weight.clone(),
                    );
                    block_wand_single_scorer(scorer, threshold, &mut |doc, score| {
                        returned += 1;
                        let previous = heap.clone();
                        if heap.iter().any(|row| row.1 == ordinal && row.2 == doc) {
                            assert!(seeded);
                            seeded_duplicates += 1;
                        } else {
                            collect(&mut heap, (score, ordinal, doc), k);
                        }
                        if !fixed && heap.len() == k {
                            threshold = heap.last().unwrap().0;
                            if policy == "normal_tie_safe" {
                                threshold = threshold.next_down();
                            }
                        }
                        if address_aware && (ordinal, doc) >= (last_winner.1, last_winner.2) {
                            threshold = final_threshold;
                        }
                        if heap != previous {
                            trace::heap_update(threshold);
                        }
                        threshold
                    });
                }
                let mut stats = trace::finish();
                assert_eq!(heap, expected, "{field}:{name} {format} {policy}");
                let reads = reads.lock().unwrap();
                assert_eq!(stats["norm_reads"].as_u64().unwrap(), reads.count);
                assert_eq!(
                    stats["norm_pages"].as_u64().unwrap(),
                    reads.pages.len() as u64
                );
                assert_eq!(
                    stats["full_blocks_visited"].as_u64().unwrap()
                        + stats["full_blocks_skipped_without_candidates"]
                            .as_u64()
                            .unwrap(),
                    full_blocks as u64
                );
                if fixed {
                    let expected_returned = if address_aware {
                        k
                    } else {
                        all_scores
                            .iter()
                            .filter(|score| **score >= final_threshold)
                            .count()
                    };
                    assert_eq!(returned, expected_returned);
                }
                if seeded {
                    assert_eq!(seeded_duplicates, k);
                    assert_eq!(stats["heap_updates"], 0);
                }
                stats["format"] = json!(format);
                stats["policy"] = json!(policy);
                stats["returned_candidates"] = json!(returned);
                stats["seeded_duplicates"] = json!(seeded_duplicates);
                stats["same_results"] = json!(true);
                let mut compact = stats.clone();
                compact.as_object_mut().unwrap().remove("timeline");
                println!(
                    "{}",
                    json!({"field": field, "term": name, "stats": compact})
                );
                runs.push(stats);
            }
        }
        results.push(json!({
            "field": field, "term": name, "k": k,
            "postings": all_scores.len(), "full_blocks": full_blocks,
            "final_threshold": final_threshold,
            "docs_equal_cutoff": all_scores.iter().filter(|score| **score == final_threshold).count(),
            "docs_above_cutoff": all_scores.iter().filter(|score| **score > final_threshold).count(),
            "expected": expected, "runs": runs,
        }));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

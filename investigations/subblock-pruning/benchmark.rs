use super::*;
use crate::fieldnorm::threshold_trace as trace;
use crate::postings::serializer::PostingsSerializer;
use crate::postings::subblock::{SUBBLOCK_SIZE, SubblockSummary, read_summaries, write_summaries};

#[test]
#[ignore]
fn replay_hn_subblock_pruning() {
    let directory = std::env::var("MIN_NORM_EXPORT").unwrap();
    let dir = Path::new(&directory);
    let output = std::env::var("SUBBLOCK_OUTPUT").unwrap();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let field = dir.file_name().unwrap().to_str().unwrap();
    let total_docs = manifest["global_num_docs"].as_u64().unwrap();
    let avg = manifest["global_total_tokens"].as_u64().unwrap() as f32 / total_docs as f32;
    let payload_size = manifest["storage_payload_bytes"].as_u64().unwrap() as usize;
    let mut results = Vec::new();
    for (term, name) in manifest["terms"].as_array().unwrap().iter().enumerate() {
        if name == "the" && std::env::var_os("SUBBLOCK_COMMON").is_none() {
            continue;
        }
        let weight = Bm25Weight::for_one_term(
            manifest["global_doc_freqs"][term].as_u64().unwrap(),
            total_docs,
            avg,
            Bm25Params::default(),
        );
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        let mut added_bytes = 0usize;
        let mut posting_bytes = 0usize;
        let mut summary_pages = 0usize;
        for (ordinal, segment) in manifest["segments"].as_array().unwrap().iter().enumerate() {
            let info = &segment["terms"][term];
            let count = info["doc_freq"].as_u64().unwrap() as usize;
            if count == 0 {
                continue;
            }
            let stem = info["stem"].as_str().unwrap();
            let records = std::fs::read(dir.join(format!("{stem}.records"))).unwrap();
            let raw = std::fs::read(dir.join(format!("{stem}.raw"))).unwrap();
            let norms = OwnedBytes::new(
                std::fs::read(dir.join(format!("{}.norms", segment["stem"].as_str().unwrap())))
                    .unwrap(),
            );
            let mut summaries = Vec::new();
            for group in records.chunks(24 * SUBBLOCK_SIZE) {
                let mut summary = SubblockSummary::default();
                for record in group.chunks_exact(24) {
                    let score = weight.score(record[20], u32_at(record, 4));
                    assert_eq!(
                        score,
                        f32::from_le_bytes(record[16..20].try_into().unwrap())
                    );
                    assert_eq!(record[20], norms[u32_at(record, 0) as usize]);
                    assert_eq!(record[21], 1);
                    summary.record(record[20], u32_at(record, 4));
                    collect(&mut expected, (score, ordinal, u32_at(record, 0)), 10);
                }
                assert!(
                    group.chunks_exact(24).all(|record| summary.bound(&weight)
                        >= weight.score(record[20], u32_at(record, 4)))
                );
                summaries.push(summary);
            }
            let mut encoded = Vec::new();
            write_summaries(&summaries, &mut encoded).unwrap();
            added_bytes += encoded.len();
            summary_pages += encoded.len().div_ceil(payload_size);
            encoded.extend_from_slice(&raw);
            posting_bytes += raw.len();
            if cfg!(feature = "subblock-pruning") {
                let norm_reader = FieldNormReader::open(FileSlice::from(norms.as_slice().to_vec()));
                let mut serializer = PostingsSerializer::new(
                    avg,
                    IndexRecordOption::WithFreqsAndPositions,
                    Some(norm_reader),
                    Bm25Params::default(),
                );
                serializer.new_term(count as u32, true);
                for record in records.chunks_exact(24) {
                    serializer.write_doc(u32_at(record, 0), u32_at(record, 4));
                }
                let mut serialized = Vec::new();
                serializer
                    .close_term(count as u32, &mut serialized)
                    .unwrap();
                let (actual, _) =
                    read_summaries(count as u32, OwnedBytes::new(serialized)).unwrap();
                let (reference, _) =
                    read_summaries(count as u32, OwnedBytes::new(encoded.clone())).unwrap();
                assert_eq!(actual.as_slice(), reference.as_slice());
            }
            inputs.push((
                ordinal,
                count,
                raw,
                encoded,
                norms,
                segment["norm_base"].as_u64().unwrap() as usize,
            ));
        }
        let last = *expected.last().unwrap();
        let k = expected.len();
        let mut runs = Vec::new();
        for format in ["legacy", "subblock16"] {
            for policy in ["growing", "growing_ties", "oracle"] {
                let mut heap = Vec::new();
                let mut threshold = Score::MIN;
                let reads = Arc::new(Mutex::new(Reads::default()));
                trace::start(false, threshold);
                for (ordinal, count, raw, encoded, norm_bytes, base) in &inputs {
                    if policy == "oracle" {
                        threshold = if *ordinal <= last.1 {
                            last.0.next_down()
                        } else {
                            last.0
                        };
                    }
                    trace::segment(*ordinal, threshold);
                    let norms = FieldNormReader::open(FileSlice::new(Arc::new(Norms {
                        data: norm_bytes.clone(),
                        reads: reads.clone(),
                        segment: *ordinal,
                        base: *base,
                        payload_size,
                    })));
                    let block = BlockSegmentPostings::open(
                        *count as u32,
                        OwnedBytes::new(if format == "legacy" {
                            raw.clone()
                        } else {
                            encoded.clone()
                        }),
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
                        let before = heap.clone();
                        collect(&mut heap, (score, *ordinal, doc), k);
                        if policy == "oracle" {
                            if (*ordinal, doc) >= (last.1, last.2) {
                                threshold = last.0;
                            }
                        } else if heap.len() == k {
                            threshold = heap.last().unwrap().0;
                            if policy == "growing_ties" {
                                threshold = threshold.next_down();
                            }
                        }
                        if heap != before {
                            trace::heap_update(threshold);
                        }
                        threshold
                    });
                }
                let mut stats = trace::finish();
                assert_eq!(heap, expected, "{field}:{name} {format} {policy}");
                assert_eq!(
                    stats["norm_reads"].as_u64().unwrap(),
                    reads.lock().unwrap().count
                );
                assert_eq!(
                    stats["norm_pages"].as_u64().unwrap(),
                    reads.lock().unwrap().pages.len() as u64
                );
                stats.as_object_mut().unwrap().remove("timeline");
                stats["format"] = json!(format);
                stats["policy"] = json!(policy);
                stats["same_results"] = json!(true);
                runs.push(stats);
            }
        }
        let result = json!({"field":field,"term":name,"k":k,"postings":manifest["global_doc_freqs"][term],
            "added_bytes":added_bytes,"original_posting_bytes":posting_bytes,"packed_summary_pages_per_segment":summary_pages,
            "header_bytes":inputs.len()*10,"expected":expected,"runs":runs});
        println!("{result}");
        results.push(result);
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

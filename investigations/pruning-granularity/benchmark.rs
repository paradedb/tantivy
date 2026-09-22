use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::TERMINATED;
use crate::fieldnorm::threshold_trace as trace;

#[derive(Clone, Copy)]
struct Record {
    doc: u32,
    tf: u32,
    score: Score,
    norm: u8,
}

struct Input {
    records: Vec<Record>,
    raw_records: Vec<u8>,
    raw: Vec<u8>,
    norms: OwnedBytes,
    base: usize,
    minima: Vec<[u8; 3]>,
}

fn simulate(
    inputs: &[Input],
    weight: &Bm25Weight,
    expected: &[(Score, usize, u32)],
    payload_size: usize,
    group_size: usize,
    format: &str,
) -> Value {
    let last = *expected.last().unwrap();
    let mut heap = Vec::new();
    let mut pages = BTreeSet::new();
    let mut reads = 0usize;
    let mut tail_reads = 0usize;
    let mut candidates = 0usize;
    let mut rejected = 0usize;
    let mut groups = 0usize;
    let mut entered = 0usize;
    let mut group_rows = Vec::new();
    for (ordinal, input) in inputs.iter().enumerate() {
        let mut threshold = if ordinal <= last.1 {
            last.0.next_down()
        } else {
            last.0
        };
        for (group_idx, group) in input.records.chunks(group_size).enumerate() {
            groups += 1;
            let bound = group.iter().map(|r| r.score).reduce(f32::max).unwrap();
            if bound <= threshold {
                continue;
            }
            entered += 1;
            let before = reads;
            let mut group_pages = BTreeSet::new();
            for (offset, record) in group.iter().enumerate() {
                if bound <= threshold {
                    break;
                }
                candidates += 1;
                let posting_idx = group_idx * group_size + offset;
                let min_norm = if format == "tf_class" {
                    input.minima[posting_idx / 128][record.tf.saturating_sub(1).min(2) as usize]
                } else {
                    0
                };
                if format != "original" && weight.score(min_norm, record.tf) <= threshold {
                    rejected += 1;
                    continue;
                }
                reads += 1;
                tail_reads += usize::from(posting_idx >= input.records.len() / 128 * 128);
                let page = (ordinal, (input.base + record.doc as usize) / payload_size);
                pages.insert(page);
                group_pages.insert(page);
                let score = weight.score(input.norms[record.doc as usize], record.tf);
                assert_eq!(score, record.score);
                if score > threshold {
                    collect(&mut heap, (score, ordinal, record.doc), expected.len());
                    if (ordinal, record.doc) >= (last.1, last.2) {
                        threshold = last.0;
                    }
                }
            }
            group_rows.push(
                json!({"segment":ordinal,"start_posting":group_idx*group_size,
                "len":group.len(),"bound":bound,"norm_reads":reads-before,
                "norm_pages":group_pages.len()}),
            );
        }
    }
    assert_eq!(heap, expected, "group_size={group_size} format={format}");
    assert_eq!(reads + rejected, candidates);
    json!({"kind":"simulation","format":format,"group_size":group_size,
        "groups":groups,"groups_entered":entered,"groups_skipped":groups-entered,
        "candidates":candidates,"tf_rejected":rejected,"norm_reads":reads,
        "norm_pages":pages.len(),"scoring_norm_reads_in_original_tail":tail_reads,
        "same_results":true,"entered_groups":group_rows})
}

#[test]
#[ignore]
fn replay_hn_pruning_granularity() {
    let directory = std::env::var("MIN_NORM_EXPORT").unwrap();
    let dir = Path::new(&directory);
    let output = std::env::var("GRANULARITY_OUTPUT").unwrap();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let segments = manifest["segments"].as_array().unwrap();
    let total_docs = manifest["global_num_docs"].as_u64().unwrap();
    let avg = manifest["global_total_tokens"].as_u64().unwrap() as f32 / total_docs as f32;
    let payload_size = manifest["storage_payload_bytes"].as_u64().unwrap() as usize;
    let field = dir.file_name().unwrap().to_str().unwrap();
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
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        let mut full_bounds = BTreeMap::new();
        let mut all_bounds = BTreeMap::new();
        let mut bound_rows = Vec::new();
        for (ordinal, segment) in segments.iter().enumerate() {
            let info = &segment["terms"][term];
            let stem = info["stem"].as_str().unwrap();
            let raw_records = std::fs::read(dir.join(format!("{stem}.records"))).unwrap();
            let raw = std::fs::read(dir.join(format!("{stem}.raw"))).unwrap();
            let norms = OwnedBytes::new(
                std::fs::read(dir.join(format!("{}.norms", segment["stem"].as_str().unwrap())))
                    .unwrap(),
            );
            let records: Vec<Record> = raw_records
                .chunks_exact(24)
                .map(|bytes| {
                    let record = Record {
                        doc: u32_at(bytes, 0),
                        tf: u32_at(bytes, 4),
                        score: f32::from_le_bytes(bytes[16..20].try_into().unwrap()),
                        norm: bytes[20],
                    };
                    assert_eq!(bytes[21], 1);
                    assert_eq!(record.norm, norms[record.doc as usize]);
                    assert_eq!(record.score, weight.score(record.norm, record.tf));
                    collect(&mut expected, (record.score, ordinal, record.doc), 10);
                    record
                })
                .collect();
            assert_eq!(records.len(), info["doc_freq"].as_u64().unwrap() as usize);
            assert!(records.windows(2).all(|pair| pair[0].doc < pair[1].doc));
            let mut rest = &raw[..];
            let skip_len = VInt::deserialize_u64(&mut rest).unwrap() as usize;
            let skip = &rest[..skip_len];
            assert_eq!(skip_len, records.len() / 128 * 12);
            let mut minima = Vec::new();
            for (block_idx, group) in records.chunks(128).enumerate() {
                let exact = group.iter().map(|r| r.score).reduce(f32::max).unwrap();
                let end = if group.len() == 128 {
                    group.last().unwrap().doc
                } else {
                    TERMINATED
                };
                all_bounds.insert((ordinal, end), exact);
                let mut mins = [255; 3];
                if group.len() == 128 {
                    full_bounds.insert((ordinal, end), exact);
                    for record in group {
                        let class = record.tf.saturating_sub(1).min(2) as usize;
                        mins[class] = mins[class].min(record.norm);
                    }
                    let entry = &skip[block_idx * 12..(block_idx + 1) * 12];
                    assert_eq!(u32_at(entry, 0), end);
                    let stored = weight.score(entry[10], decode_block_wand_max_tf(entry[11]));
                    bound_rows.push(json!({"segment":ordinal,"block":block_idx,
                        "stored":stored,"exact":exact,"stored_norm":entry[10],"stored_tf_code":entry[11]}));
                } else {
                    mins = [0; 3];
                }
                minima.push(mins);
            }
            inputs.push(Input {
                records,
                raw_records,
                raw,
                norms,
                base: segment["norm_base"].as_u64().unwrap() as usize,
                minima,
            });
        }
        let last = *expected.last().unwrap();
        let mut runs = Vec::new();
        for format in ["original", "zero", "tf_class"] {
            for bound_mode in ["stored", "exact_full", "exact_all"] {
                let reads = Arc::new(Mutex::new(Reads::default()));
                let mut heap = Vec::new();
                trace::start(format == "original", last.0.next_down());
                trace::set_bounds(match bound_mode {
                    "exact_full" => full_bounds.clone(),
                    "exact_all" => all_bounds.clone(),
                    _ => BTreeMap::new(),
                });
                for (ordinal, input) in inputs.iter().enumerate() {
                    let mut threshold = if ordinal <= last.1 {
                        last.0.next_down()
                    } else {
                        last.0
                    };
                    trace::segment(ordinal, threshold);
                    let norms = FieldNormReader::open(FileSlice::new(Arc::new(Norms {
                        data: input.norms.clone(),
                        reads: reads.clone(),
                        segment: ordinal,
                        base: input.base,
                        payload_size,
                    })));
                    let raw = rewrite(
                        &input.raw,
                        &input.raw_records,
                        if format == "tf_class" { 2 } else { 0 },
                    );
                    let block = BlockSegmentPostings::open(
                        input.records.len() as u32,
                        OwnedBytes::new(raw),
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
                        collect(&mut heap, (score, ordinal, doc), 10);
                        if (ordinal, doc) >= (last.1, last.2) {
                            threshold = last.0;
                        }
                        threshold
                    });
                }
                assert_eq!(heap, expected, "{field}:{name} {format} {bound_mode}");
                let mut stats = trace::finish();
                assert_eq!(
                    stats["norm_reads"].as_u64().unwrap(),
                    reads.lock().unwrap().count
                );
                assert_eq!(
                    stats["norm_pages"].as_u64().unwrap(),
                    reads.lock().unwrap().pages.len() as u64
                );
                stats.as_object_mut().unwrap().remove("timeline");
                stats["kind"] = json!("native");
                stats["format"] = json!(format);
                stats["bound_mode"] = json!(bound_mode);
                stats["same_results"] = json!(true);
                if bound_mode == "exact_all" {
                    let sim = simulate(&inputs, &weight, &expected, payload_size, 128, format);
                    for key in ["norm_reads", "norm_pages", "candidates", "tf_rejected"] {
                        assert_eq!(
                            stats[key], sim[key],
                            "native/simulation mismatch: {field}:{name} {format} {key}"
                        );
                    }
                    stats["simulation_verified"] = json!(true);
                }
                runs.push(stats);
            }
            for group_size in [128, 64, 32, 16, 8, 1] {
                runs.push(simulate(
                    &inputs,
                    &weight,
                    &expected,
                    payload_size,
                    group_size,
                    format,
                ));
            }
        }
        println!(
            "{}",
            json!({"field":field,"term":name,"runs":runs.iter().map(|r| {
            let mut r=r.clone();r.as_object_mut().unwrap().remove("entered_groups");r
        }).collect::<Vec<_>>()})
        );
        results.push(
            json!({"field":field,"term":name,"postings":manifest["global_doc_freqs"][term],
            "cutoff":last,"expected":expected,"blocks":bound_rows,"runs":runs}),
        );
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

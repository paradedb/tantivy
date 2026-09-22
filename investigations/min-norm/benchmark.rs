use std::collections::HashSet;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use common::{BinarySerializable, HasLen, VInt};
use serde_json::{json, Value};

use super::block_wand_union::block_wand_single_scorer;
use crate::directory::{FileHandle, FileSlice, OwnedBytes};
use crate::fieldnorm::FieldNormReader;
use crate::postings::skip::{decode_bitwidth, decode_block_wand_max_tf, SkipSerializer};
use crate::postings::{BlockSegmentPostings, SegmentPostings};
use crate::query::term_query::TermScorer;
use crate::query::Bm25Weight;
use crate::schema::IndexRecordOption;
use crate::{Bm25Params, Score};

#[path = "../threshold-oracle/benchmark.rs"]
mod threshold_oracle;

#[path = "../pruning-granularity/benchmark.rs"]
mod pruning_granularity;

#[path = "../subblock-pruning/benchmark.rs"]
mod subblock_pruning;

#[derive(Debug, Default)]
struct Reads {
    count: u64,
    pages: HashSet<(usize, usize)>,
}

#[derive(Debug)]
struct Norms {
    data: OwnedBytes,
    reads: Arc<Mutex<Reads>>,
    segment: usize,
    base: usize,
    payload_size: usize,
}

impl HasLen for Norms {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl FileHandle for Norms {
    fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
        Ok(self.data.slice(range))
    }
    fn read_byte(&self, offset: usize) -> std::io::Result<u8> {
        let mut reads = self.reads.lock().unwrap();
        reads.count += 1;
        reads
            .pages
            .insert((self.segment, (self.base + offset) / self.payload_size));
        crate::fieldnorm::threshold_trace::norm_read(
            self.segment,
            offset as u32,
            (self.base + offset) / self.payload_size,
        );
        Ok(self.data[offset])
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn rewrite(raw: &[u8], records: &[u8], mode: usize) -> Vec<u8> {
    let count = records.len() / 24;
    if count < 128 || mode == 0 {
        return raw.to_vec();
    }
    let mut rest = raw;
    let skip_len = VInt::deserialize_u64(&mut rest).unwrap() as usize;
    let (skip, payload) = rest.split_at(skip_len);
    assert_eq!(skip_len, count / 128 * 12);
    let mut writer = SkipSerializer::new();
    for (block, entry) in skip.chunks_exact(12).enumerate() {
        let (bits, delta) = decode_bitwidth(entry[4]);
        assert!(delta);
        writer.write_doc(u32_at(entry, 0), bits);
        writer.write_term_freq(entry[5]);
        writer.write_total_term_freq(u32_at(entry, 6));
        writer.write_blockwand_max(entry[10], decode_block_wand_max_tf(entry[11]));
        let mut minima = [255; 3];
        for record in records[block * 128 * 24..(block + 1) * 128 * 24].chunks_exact(24) {
            let class = u32_at(record, 4).saturating_sub(1).min(2) as usize;
            minima[class] = minima[class].min(record[20]);
        }
        writer.write_min_fieldnorms(minima, mode == 2);
    }
    let mut out = Vec::new();
    VInt(writer.data().len() as u64)
        .serialize(&mut out)
        .unwrap();
    out.write_all(writer.data()).unwrap();
    out.write_all(payload).unwrap();
    out
}

fn collect(heap: &mut Vec<(Score, usize, u32)>, row: (Score, usize, u32), k: usize) -> Score {
    if heap.len() < k
        || row.0 > heap.last().unwrap().0
        || (row.0 == heap.last().unwrap().0
            && (row.1, row.2) < (heap.last().unwrap().1, heap.last().unwrap().2))
    {
        heap.push(row);
        heap.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        heap.truncate(k);
    }
    if heap.len() == k {
        heap.last().unwrap().0
    } else {
        Score::MIN
    }
}

#[test]
#[ignore]
fn replay_hn_minimum_norm_formats() {
    let directory = std::env::var("MIN_NORM_EXPORT").expect("MIN_NORM_EXPORT");
    let dir = Path::new(&directory);
    let scope = std::env::var("MIN_NORM_THRESHOLD_SCOPE").unwrap_or_else(|_| "global".into());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let segments = manifest["segments"].as_array().unwrap();
    let total_docs = manifest["global_num_docs"].as_u64().unwrap();
    let avg = manifest["global_total_tokens"].as_u64().unwrap() as f32 / total_docs as f32;
    let payload_size = manifest["storage_payload_bytes"].as_u64().unwrap() as usize;
    let mut results = Vec::new();
    for (term, name) in manifest["terms"].as_array().unwrap().iter().enumerate() {
        let weight = Bm25Weight::for_one_term(
            manifest["global_doc_freqs"][term].as_u64().unwrap(),
            total_docs,
            avg,
            Bm25Params::default(),
        );
        let ks: &[usize] = if name == "database" {
            &[1, 10, 100]
        } else {
            &[10]
        };
        for &k in ks {
            let mut expected = Vec::new();
            let mut mode_results = Vec::new();
            for mode in 0..3 {
                let reads = Arc::new(Mutex::new(Reads::default()));
                let mut heap = Vec::new();
                let mut threshold = Score::MIN;
                let mut raw_bytes = 0usize;
                let mut encoded_bytes = 0usize;
                let mut blocks = 0usize;
                for (ordinal, segment) in segments.iter().enumerate() {
                    let info = &segment["terms"][term];
                    let count = info["doc_freq"].as_u64().unwrap() as usize;
                    if count == 0 {
                        continue;
                    }
                    let stem = info["stem"].as_str().unwrap();
                    let records = std::fs::read(dir.join(format!("{stem}.records"))).unwrap();
                    assert_eq!(records.len(), count * 24);
                    let raw = std::fs::read(dir.join(format!("{stem}.raw"))).unwrap();
                    let bytes = rewrite(&raw, &records, mode);
                    raw_bytes += raw.len();
                    encoded_bytes += bytes.len();
                    blocks += count / 128;
                    let norm_bytes = OwnedBytes::new(
                        std::fs::read(
                            dir.join(format!("{}.norms", segment["stem"].as_str().unwrap())),
                        )
                        .unwrap(),
                    );
                    if mode == 0 {
                        for record in records.chunks_exact(24) {
                            let score = f32::from_le_bytes(record[16..20].try_into().unwrap());
                            assert_eq!(score, weight.score(record[20], u32_at(record, 4)));
                            assert_eq!(record[21], 1, "replay requires all postings live");
                            collect(&mut expected, (score, ordinal, u32_at(record, 0)), k);
                        }
                    }
                    let norms = FieldNormReader::open(FileSlice::new(Arc::new(Norms {
                        data: norm_bytes,
                        reads: reads.clone(),
                        segment: ordinal,
                        base: segment["norm_base"].as_u64().unwrap() as usize,
                        payload_size,
                    })));
                    let block = BlockSegmentPostings::open(
                        count as u32,
                        OwnedBytes::new(bytes),
                        IndexRecordOption::WithFreqsAndPositions,
                        IndexRecordOption::WithFreqs,
                    )
                    .unwrap();
                    let scorer = TermScorer::new(
                        SegmentPostings::from_block_postings(block, None),
                        norms,
                        weight.clone(),
                    );
                    if scope == "segment" {
                        let mut local = Vec::new();
                        block_wand_single_scorer(scorer, Score::MIN, &mut |doc, score| {
                            collect(&mut local, (score, ordinal, doc), k)
                        });
                        for row in local {
                            collect(&mut heap, row, k);
                        }
                    } else {
                        block_wand_single_scorer(scorer, threshold, &mut |doc, score| {
                            threshold = collect(&mut heap, (score, ordinal, doc), k);
                            threshold
                        });
                    }
                }
                assert_eq!(heap, expected, "term={name} k={k} mode={mode}");
                let reads = reads.lock().unwrap();
                mode_results.push(json!({"mode":(["zero","block","tf_class"][mode]),"norm_reads":reads.count,"norm_pages":reads.pages.len(),"added_bytes":encoded_bytes-raw_bytes,"full_blocks":blocks,"same_results":true}));
            }
            println!("{}", json!({"term":name,"k":k,"results":mode_results}));
            results.push(json!({"term":name,"k":k,"results":mode_results}));
        }
    }
    std::fs::write(
        dir.join(format!("native-results-{scope}.json")),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
}

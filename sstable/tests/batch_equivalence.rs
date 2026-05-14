//! Equivalence tests for `Dictionary::batch_term_info_exact`.
//!
//! For each test cell we build a `Dictionary<MonotonicU64SSTable>`, then
//! compare the batched iterator's output against a hand-rolled baseline
//! that does K individual `dict.get(key)` calls. Both must yield the
//! same `(input_index, value)` pairs in the same order.
//!
//! `MonotonicU64SSTable` is enough — the `batch_term_info_exact` algorithm
//! is fully agnostic to the value type (it only does `value.clone()`). If
//! the value-reader plumbing works for `u64`, it works for `TermInfo`.

use std::io;

use common::OwnedBytes;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use tantivy_sstable::{Dictionary, MonotonicU64SSTable, SortedTermSlice, sort_and_dedupe_terms};

/// Build a dictionary of `n` u64-encoded BE keys with values `i as u64`.
/// Optionally override the target block length so we can force more
/// block transitions for small dicts.
fn build_dict(n: usize, block_len: Option<usize>) -> Dictionary<MonotonicU64SSTable> {
    let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
    if let Some(bl) = block_len {
        builder.set_block_len(bl);
    }
    for i in 0..n as u64 {
        let key = i.to_be_bytes();
        builder.insert(&key[..], &i).unwrap();
    }
    let bytes = builder.finish().unwrap();
    Dictionary::<MonotonicU64SSTable>::from_bytes_for_tests(OwnedBytes::new(bytes)).unwrap()
}

/// Build a dictionary of `n` arbitrary byte-string keys (deterministic
/// per `seed`), with values `i as u64`. Returns the dictionary plus the
/// sorted-deduped key list (which the dictionary preserves).
fn build_dict_strings(n: usize, seed: u64) -> (Dictionary<MonotonicU64SSTable>, Vec<Vec<u8>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut keys: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let len = rng.random_range(1..16usize);
            (0..len).map(|_| rng.random()).collect()
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();

    let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
    for (i, key) in keys.iter().enumerate() {
        builder.insert(&key[..], &(i as u64)).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let dict =
        Dictionary::<MonotonicU64SSTable>::from_bytes_for_tests(OwnedBytes::new(bytes)).unwrap();
    (dict, keys)
}

/// Compute the baseline `(idx, value)` pairs by calling `dict.get(key)`
/// on each input key.
fn baseline<K: AsRef<[u8]>>(
    dict: &Dictionary<MonotonicU64SSTable>,
    keys: &[K],
) -> Vec<(usize, u64)> {
    keys.iter()
        .enumerate()
        .filter_map(|(i, k)| dict.get(k.as_ref()).unwrap().map(|v| (i, v)))
        .collect()
}

/// Drive the batched iterator and collect the result.
fn batched<K: AsRef<[u8]>>(
    dict: &Dictionary<MonotonicU64SSTable>,
    sorted: SortedTermSlice<'_, K>,
) -> Vec<(usize, u64)> {
    dict.batch_term_info_exact(sorted)
        .collect::<io::Result<Vec<_>>>()
        .unwrap()
}

/// Assert that the batched iterator and `dict.get`-baseline produce the
/// same result on a given sorted input.
fn assert_equivalent_on_sorted(dict: &Dictionary<MonotonicU64SSTable>, sorted_keys: &[Vec<u8>]) {
    let sorted = SortedTermSlice::new(sorted_keys).expect("test input must be sorted");
    let base = baseline(dict, sorted_keys);
    let batch = batched(dict, sorted);
    assert_eq!(
        base,
        batch,
        "batched output diverged from dict.get() baseline (n_keys={})",
        sorted_keys.len()
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_input_yields_none() {
    let dict = build_dict(100, None);
    let keys: Vec<Vec<u8>> = vec![];
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn empty_dictionary_yields_none_for_any_input() {
    let dict = build_dict(0, None);
    let keys: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn all_inputs_past_last_key() {
    // Dict has keys 0..100 (big-endian u64); query keys far past those.
    let dict = build_dict(100, None);
    let keys: Vec<Vec<u8>> = (0u64..5)
        .map(|i| (i + 10_000).to_be_bytes().to_vec())
        .collect();
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn all_inputs_before_first_key() {
    // Dict starts at key `0x01u64` (BE), query keys are all zero-prefixed
    // sub-keys that sort strictly before that.
    let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
    for i in 1u64..50 {
        builder.insert(&i.to_be_bytes()[..], &i).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let dict =
        Dictionary::<MonotonicU64SSTable>::from_bytes_for_tests(OwnedBytes::new(bytes)).unwrap();

    let keys: Vec<Vec<u8>> = vec![vec![0u8], vec![0, 0]];
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn single_input_present() {
    let dict = build_dict(100, None);
    let keys: Vec<Vec<u8>> = vec![42u64.to_be_bytes().to_vec()];
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn single_input_absent() {
    // Dict has 0..100; query for 999 which is past the last key.
    let dict = build_dict(100, None);
    let keys: Vec<Vec<u8>> = vec![999u64.to_be_bytes().to_vec()];
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn all_present() {
    // Query every dict key.
    let dict = build_dict(500, None);
    let keys: Vec<Vec<u8>> = (0u64..500).map(|i| i.to_be_bytes().to_vec()).collect();
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn all_missing_interspersed_with_dict_range() {
    // Dict has even-numbered keys 0, 2, 4, ..., 198 (n=100). Query for
    // the odd keys in the same range — all absent but interleaved with
    // the dictionary keys, so the iterator must Greater-skip per target.
    let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
    for i in 0u64..100 {
        let v = i * 2;
        builder.insert(&v.to_be_bytes()[..], &v).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let dict =
        Dictionary::<MonotonicU64SSTable>::from_bytes_for_tests(OwnedBytes::new(bytes)).unwrap();

    let keys: Vec<Vec<u8>> = (0u64..100)
        .map(|i| (i * 2 + 1).to_be_bytes().to_vec())
        .collect();
    // The last query (199) is past the dict's max (198), so the
    // `get_block_with_key` short-circuit fires for it; the rest must
    // resolve via Greater-skip within blocks.
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn mixed_present_and_absent_alternating() {
    // Same construction as above; query for both evens (present) and
    // odds (absent), alternating.
    let mut builder = Dictionary::<MonotonicU64SSTable>::builder(Vec::new()).unwrap();
    for i in 0u64..100 {
        let v = i * 2;
        builder.insert(&v.to_be_bytes()[..], &v).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let dict =
        Dictionary::<MonotonicU64SSTable>::from_bytes_for_tests(OwnedBytes::new(bytes)).unwrap();

    let keys: Vec<Vec<u8>> = (0u64..50).map(|i| i.to_be_bytes().to_vec()).collect();
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn inputs_concentrated_in_one_block() {
    // Tiny block_len forces many blocks; pick a query subset clustered
    // in a tight key range so all map to the same block.
    let dict = build_dict(1000, Some(64));
    // 10 consecutive keys starting at 250.
    let keys: Vec<Vec<u8>> = (250u64..260).map(|i| i.to_be_bytes().to_vec()).collect();
    assert_equivalent_on_sorted(&dict, &keys);
}

#[test]
fn inputs_spanning_many_blocks() {
    // Tiny block_len + queries spread evenly across the key range
    // exercises the block-transition logic many times.
    let dict = build_dict(1000, Some(64));
    let keys: Vec<Vec<u8>> = (0u64..1000)
        .step_by(37)
        .map(|i| i.to_be_bytes().to_vec())
        .collect();
    assert_equivalent_on_sorted(&dict, &keys);
}

// ---------------------------------------------------------------------------
// Random equivalence across dictionary sizes + query subsets
// ---------------------------------------------------------------------------

fn random_equivalence_at(n_dict: usize, n_query: usize, seed: u64) {
    let (dict, dict_keys) = build_dict_strings(n_dict, seed);
    let n_dict_actual = dict_keys.len();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xdead));

    // Sample `n_query` keys: half present (drawn from dict_keys), half
    // absent (random fresh bytes).
    let mut query: Vec<Vec<u8>> = Vec::with_capacity(n_query);
    for i in 0..n_query {
        if i % 2 == 0 && n_dict_actual > 0 {
            let idx = rng.random_range(0..n_dict_actual);
            query.push(dict_keys[idx].clone());
        } else {
            // Random bytes — overwhelmingly likely to be absent from dict.
            let len = rng.random_range(1..32usize);
            query.push((0..len).map(|_| rng.random()).collect());
        }
    }
    sort_and_dedupe_terms(&mut query);
    assert_equivalent_on_sorted(&dict, &query);
}

#[test]
fn random_small_dict_zero_query() {
    random_equivalence_at(10, 0, 1);
}

#[test]
fn random_small_dict_single_query() {
    random_equivalence_at(10, 1, 2);
}

#[test]
fn random_small_dict_ten_query() {
    random_equivalence_at(10, 10, 3);
}

#[test]
fn random_medium_dict_ten_query() {
    random_equivalence_at(1_000, 10, 4);
}

#[test]
fn random_medium_dict_hundred_query() {
    random_equivalence_at(1_000, 100, 5);
}

#[test]
fn random_medium_dict_full_query() {
    random_equivalence_at(1_000, 1_000, 6);
}

#[test]
fn random_large_dict_hundred_query() {
    random_equivalence_at(10_000, 100, 7);
}

#[test]
fn random_large_dict_thousand_query() {
    random_equivalence_at(10_000, 1_000, 8);
}

#[test]
fn random_large_dict_full_query() {
    random_equivalence_at(10_000, 10_000, 9);
}

// ---------------------------------------------------------------------------
// Several seeds at each shape (lightweight fuzzing)
// ---------------------------------------------------------------------------

#[test]
fn random_multi_seed_medium() {
    for seed in 100..120u64 {
        random_equivalence_at(1_000, 50, seed);
    }
}

#[test]
fn random_multi_seed_large() {
    for seed in 200..205u64 {
        random_equivalence_at(10_000, 500, seed);
    }
}

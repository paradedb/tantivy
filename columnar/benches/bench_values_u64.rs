use std::ops::RangeInclusive;
use std::sync::Arc;

use binggan::{black_box, InputGroup};
use common::file_slice::FileSlice;
use rand::prelude::*;
use tantivy_columnar::column_values::{
    load_u64_based_column_values, serialize_u64_based_column_values, CodecType,
};
use tantivy_columnar::*;

// Warning: this generates the same permutation at each call
fn generate_permutation() -> Vec<u64> {
    let mut permutation: Vec<u64> = (0u64..100_000u64).collect();
    permutation.shuffle(&mut StdRng::from_seed([1u8; 32]));
    permutation
}

// Warning: this generates the same permutation at each call
fn generate_permutation_gcd() -> Vec<u64> {
    let mut permutation: Vec<u64> = (1u64..100_000u64).map(|el| el * 1000).collect();
    permutation.shuffle(&mut StdRng::from_seed([1u8; 32]));
    permutation
}

// Generate bounded precision data (financial-style, good for BUFF)
fn generate_bounded_precision() -> Vec<u64> {
    let mut data: Vec<u64> = (0u64..100_000u64)
        .map(|i| {
            // Simulate prices: $10.00 to $1000.00 with 2 decimal places
            // Stored as cents: 1000 to 100000
            1000 + (i % 99000)
        })
        .collect();
    data.shuffle(&mut StdRng::from_seed([1u8; 32]));
    data
}

/// Helper function to serialize and load column values with a specific codec
pub fn serialize_and_load(column: &[u64], codec_type: CodecType) -> Arc<dyn ColumnValues<u64>> {
    let mut buffer = Vec::new();
    serialize_u64_based_column_values(&column, &[codec_type], &mut buffer).unwrap();
    load_u64_based_column_values::<u64>(FileSlice::from(buffer)).unwrap()
}

/// Returns list of available codecs (includes BUFF when feature is enabled)
#[allow(unused_mut)] // mut is needed when buff-compression feature is enabled
fn get_all_codecs() -> Vec<(&'static str, CodecType)> {
    let mut codecs = vec![
        ("Bitpacked", CodecType::Bitpacked),
        ("Linear", CodecType::Linear),
        ("BlockwiseLinear", CodecType::BlockwiseLinear),
    ];
    #[cfg(feature = "buff-compression")]
    codecs.push(("BUFF", CodecType::Buff));
    codecs
}

const FIFTY_PERCENT_RANGE: RangeInclusive<u64> = 1..=50;
const SINGLE_ITEM: u64 = 90;
const SINGLE_ITEM_RANGE: RangeInclusive<u64> = 90..=90;
const ONE_PERCENT_ITEM_RANGE: RangeInclusive<u64> = 49..=49;

fn get_data_50percent_item() -> Vec<u128> {
    let mut rng = StdRng::from_seed([1u8; 32]);

    let mut data = vec![];
    for _ in 0..300_000 {
        let val = rng.gen_range(1..=100);
        data.push(val);
    }
    data.push(SINGLE_ITEM);

    data.shuffle(&mut rng);
    data.iter().map(|el| *el as u128).collect::<Vec<_>>()
}

type VecCol = (Vec<u64>, Arc<dyn ColumnValues<u64>>);

fn bench_access() {
    let permutation = generate_permutation();
    let column_perm: Arc<dyn ColumnValues<u64>> =
        serialize_and_load(&permutation, CodecType::Bitpacked);

    let permutation_gcd = generate_permutation_gcd();
    let column_perm_gcd: Arc<dyn ColumnValues<u64>> =
        serialize_and_load(&permutation_gcd, CodecType::Bitpacked);

    let mut group: InputGroup<VecCol> = InputGroup::new_with_inputs(vec![
        (
            "access".to_string(),
            (permutation.clone(), column_perm.clone()),
        ),
        (
            "access_gcd".to_string(),
            (permutation_gcd.clone(), column_perm_gcd.clone()),
        ),
    ]);

    group.register("stride7_vec", |inp: &VecCol| {
        let n = inp.0.len();
        let mut a = 0u64;
        for i in (0..n / 7).map(|val| val * 7) {
            a += inp.0[i];
        }
        black_box(a);
    });

    group.register("fullscan_vec", |inp: &VecCol| {
        let mut a = 0u64;
        for i in 0..inp.0.len() {
            a += inp.0[i];
        }
        black_box(a);
    });

    group.register("stride7_column_values", |inp: &VecCol| {
        let n = inp.1.num_vals() as usize;
        let mut a = 0u64;
        for i in (0..n / 7).map(|val| val * 7) {
            a += inp.1.get_val(i as u32);
        }
        black_box(a);
    });

    group.register("fullscan_column_values", |inp: &VecCol| {
        let mut a = 0u64;
        let n = inp.1.num_vals() as usize;
        for i in 0..n {
            a += inp.1.get_val(i as u32);
        }
        black_box(a);
    });

    group.run();
}

fn bench_range() {
    let data_50 = get_data_50percent_item();
    let data_u64 = data_50.iter().map(|el| *el as u64).collect::<Vec<_>>();
    let column_data: Arc<dyn ColumnValues<u64>> =
        serialize_and_load(&data_u64, CodecType::Bitpacked);

    let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> =
        InputGroup::new_with_inputs(vec![("dist_50pct_item".to_string(), column_data.clone())]);

    group.register(
        "fastfield_getrange_u64_50percent_hit",
        |col: &Arc<dyn ColumnValues<u64>>| {
            let mut positions = Vec::new();
            col.get_row_ids_for_value_range(FIFTY_PERCENT_RANGE, 0..col.num_vals(), &mut positions);
            black_box(positions.len());
        },
    );

    group.register(
        "fastfield_getrange_u64_1percent_hit",
        |col: &Arc<dyn ColumnValues<u64>>| {
            let mut positions = Vec::new();
            col.get_row_ids_for_value_range(
                ONE_PERCENT_ITEM_RANGE,
                0..col.num_vals(),
                &mut positions,
            );
            black_box(positions.len());
        },
    );

    group.register(
        "fastfield_getrange_u64_single_hit",
        |col: &Arc<dyn ColumnValues<u64>>| {
            let mut positions = Vec::new();
            col.get_row_ids_for_value_range(SINGLE_ITEM_RANGE, 0..col.num_vals(), &mut positions);
            black_box(positions.len());
        },
    );

    group.register(
        "fastfield_getrange_u64_hit_all",
        |col: &Arc<dyn ColumnValues<u64>>| {
            let mut positions = Vec::new();
            col.get_row_ids_for_value_range(0..=u64::MAX, 0..col.num_vals(), &mut positions);
            black_box(positions.len());
        },
    );

    group.run();
}

/// Benchmark codec comparison across different data patterns
fn bench_codec_comparison() {
    let permutation = generate_permutation();
    let permutation_gcd = generate_permutation_gcd();
    let bounded_precision = generate_bounded_precision();

    println!("\n=== Codec Comparison Benchmark ===\n");

    // Build inputs for each codec and data pattern
    let codecs = get_all_codecs();

    // Test 1: Random permutation data
    {
        let inputs: Vec<_> = codecs
            .iter()
            .map(|(name, codec)| {
                let col = serialize_and_load(&permutation, *codec);
                (format!("perm_{}", name), col)
            })
            .collect();

        let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> =
            InputGroup::new_with_inputs(inputs);

        group.register("fullscan", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut sum = 0u64;
            for i in 0..col.num_vals() {
                sum = sum.wrapping_add(col.get_val(i));
            }
            black_box(sum);
        });

        group.register("random_access", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut sum = 0u64;
            let mut rng = StdRng::from_seed([42u8; 32]);
            for _ in 0..1000 {
                let idx = rng.gen_range(0..col.num_vals());
                sum = sum.wrapping_add(col.get_val(idx));
            }
            black_box(sum);
        });

        group.run();
    }

    // Test 2: GCD pattern data
    {
        let inputs: Vec<_> = codecs
            .iter()
            .map(|(name, codec)| {
                let col = serialize_and_load(&permutation_gcd, *codec);
                (format!("gcd_{}", name), col)
            })
            .collect();

        let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> =
            InputGroup::new_with_inputs(inputs);

        group.register("fullscan_gcd", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut sum = 0u64;
            for i in 0..col.num_vals() {
                sum = sum.wrapping_add(col.get_val(i));
            }
            black_box(sum);
        });

        group.run();
    }

    // Test 3: Bounded precision data (BUFF's sweet spot)
    {
        let inputs: Vec<_> = codecs
            .iter()
            .map(|(name, codec)| {
                let col = serialize_and_load(&bounded_precision, *codec);
                (format!("bounded_{}", name), col)
            })
            .collect();

        let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> =
            InputGroup::new_with_inputs(inputs);

        group.register("fullscan_bounded", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut sum = 0u64;
            for i in 0..col.num_vals() {
                sum = sum.wrapping_add(col.get_val(i));
            }
            black_box(sum);
        });

        group.register("range_query_bounded", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut positions = Vec::new();
            // Query for values in the "middle" range
            col.get_row_ids_for_value_range(40000..=60000, 0..col.num_vals(), &mut positions);
            black_box(positions.len());
        });

        group.run();
    }
}

fn main() {
    bench_access();
    bench_range();
    bench_codec_comparison();
}

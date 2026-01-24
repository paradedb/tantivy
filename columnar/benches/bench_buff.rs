//! BUFF codec benchmarks
//!
//! This benchmark compares BUFF codec against other codecs for various data patterns,
//! focusing on scenarios where BUFF is expected to excel (bounded precision data).
//!
//! Run with: cargo bench --features buff-compression --bench bench_buff

use std::sync::Arc;

use binggan::{black_box, InputGroup};
use common::file_slice::FileSlice;
use rand::prelude::*;
use tantivy_columnar::column_values::{
    load_u64_based_column_values, serialize_u64_based_column_values, CodecType, ColumnValues,
};

/// Helper function to serialize and load column values with a specific codec
fn serialize_and_load(column: &[u64], codec_type: CodecType) -> Arc<dyn ColumnValues<u64>> {
    let mut buffer = Vec::new();
    serialize_u64_based_column_values(&column, &[codec_type], &mut buffer).unwrap();
    load_u64_based_column_values::<u64>(FileSlice::from(buffer)).unwrap()
}

/// Get compressed size for a codec
fn get_compressed_size(column: &[u64], codec_type: CodecType) -> usize {
    let mut buffer = Vec::new();
    serialize_u64_based_column_values(&column, &[codec_type], &mut buffer).unwrap();
    buffer.len()
}

/// Returns list of all codecs to compare
fn get_all_codecs() -> Vec<(&'static str, CodecType)> {
    vec![
        ("Bitpacked", CodecType::Bitpacked),
        ("Linear", CodecType::Linear),
        ("BlockwiseLinear", CodecType::BlockwiseLinear),
        ("BUFF", CodecType::Buff),
    ]
}

// ============================================================================
// Data generation functions
// ============================================================================

/// Financial price data: values like $10.00 to $1000.00 stored as cents
/// This is ideal for BUFF - bounded precision with limited range
fn generate_price_data(n: usize) -> Vec<u64> {
    let mut rng = StdRng::from_seed([1u8; 32]);
    (0..n)
        .map(|_| rng.gen_range(1000u64..100000u64)) // $10.00 to $1000.00 in cents
        .collect()
}

/// Sensor data: values in a bounded range with noise
/// Common in IoT applications
fn generate_sensor_data(n: usize) -> Vec<u64> {
    let mut rng = StdRng::from_seed([2u8; 32]);
    let base = 5000u64;
    (0..n)
        .map(|i| {
            // Sinusoidal pattern with noise
            let sin_component = ((i as f64 * 0.01).sin() * 1000.0) as i64;
            let noise = rng.gen_range(-100i64..100i64);
            (base as i64 + sin_component + noise).max(0) as u64
        })
        .collect()
}

/// Metric data with GCD pattern (e.g., timestamps rounded to seconds)
fn generate_gcd_data(n: usize) -> Vec<u64> {
    let mut data: Vec<u64> = (0..n as u64).map(|i| i * 1000).collect();
    data.shuffle(&mut StdRng::from_seed([3u8; 32]));
    data
}

/// Random permutation (challenging for all codecs)
fn generate_random_data(n: usize) -> Vec<u64> {
    let mut data: Vec<u64> = (0..n as u64).collect();
    data.shuffle(&mut StdRng::from_seed([4u8; 32]));
    data
}

/// Sequential data (ideal for linear interpolation)
fn generate_sequential_data(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

/// Constant data (best case for compression)
fn generate_constant_data(n: usize) -> Vec<u64> {
    vec![42u64; n]
}

// ============================================================================
// Benchmark functions
// ============================================================================

/// Print compression ratios for all codecs
fn print_compression_ratios() {
    println!("\n=== Compression Ratios (100k values) ===\n");

    let datasets: Vec<(&str, Vec<u64>)> = vec![
        ("Price (bounded)", generate_price_data(100_000)),
        ("Sensor (bounded)", generate_sensor_data(100_000)),
        ("GCD pattern", generate_gcd_data(100_000)),
        ("Random perm", generate_random_data(100_000)),
        ("Sequential", generate_sequential_data(100_000)),
        ("Constant", generate_constant_data(100_000)),
    ];

    let codecs = get_all_codecs();

    // Print header
    print!("{:<20}", "Dataset");
    for (name, _) in &codecs {
        print!("{:>15}", name);
    }
    println!();
    println!("{}", "-".repeat(20 + 15 * codecs.len()));

    // Print sizes for each dataset
    for (dataset_name, data) in &datasets {
        print!("{:<20}", dataset_name);
        let raw_size = data.len() * 8; // 8 bytes per u64

        for (_, codec) in &codecs {
            let compressed_size = get_compressed_size(data, *codec);
            let ratio = raw_size as f64 / compressed_size as f64;
            print!("{:>14.2}x", ratio);
        }
        println!();
    }
    println!();
}

/// Benchmark access patterns across codecs
fn bench_access_patterns() {
    let n = 100_000;
    let price_data = generate_price_data(n);
    let codecs = get_all_codecs();

    // Create columns for all codecs
    let columns: Vec<_> = codecs
        .iter()
        .map(|(name, codec)| {
            let col = serialize_and_load(&price_data, *codec);
            (format!("price_{}", name), col)
        })
        .collect();

    let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> = InputGroup::new_with_inputs(columns);

    // Sequential scan (common in aggregations)
    group.register("sequential_scan", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut sum = 0u64;
        for i in 0..col.num_vals() {
            sum = sum.wrapping_add(col.get_val(i));
        }
        black_box(sum);
    });

    // Strided access (common in sampling)
    group.register("strided_access", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut sum = 0u64;
        let stride = 7;
        for i in (0..col.num_vals()).step_by(stride) {
            sum = sum.wrapping_add(col.get_val(i));
        }
        black_box(sum);
    });

    // Random access (common in joins)
    group.register("random_access", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut sum = 0u64;
        let mut rng = StdRng::from_seed([99u8; 32]);
        for _ in 0..1000 {
            let idx = rng.gen_range(0..col.num_vals());
            sum = sum.wrapping_add(col.get_val(idx));
        }
        black_box(sum);
    });

    // Bulk access
    group.register("bulk_access", |col: &Arc<dyn ColumnValues<u64>>| {
        let indexes: Vec<u32> = (0..1000).collect();
        let mut output = vec![0u64; 1000];
        col.get_vals(&indexes, &mut output);
        black_box(output);
    });

    group.run();
}

/// Benchmark range queries (analytics workload)
fn bench_range_queries() {
    let n = 100_000;
    let price_data = generate_price_data(n);
    let codecs = get_all_codecs();

    let columns: Vec<_> = codecs
        .iter()
        .map(|(name, codec)| {
            let col = serialize_and_load(&price_data, *codec);
            (format!("range_{}", name), col)
        })
        .collect();

    let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> = InputGroup::new_with_inputs(columns);

    // Narrow range (selective query)
    group.register("narrow_range", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut positions = Vec::new();
        col.get_row_ids_for_value_range(49000..=51000, 0..col.num_vals(), &mut positions);
        black_box(positions.len());
    });

    // Wide range (less selective)
    group.register("wide_range", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut positions = Vec::new();
        col.get_row_ids_for_value_range(30000..=70000, 0..col.num_vals(), &mut positions);
        black_box(positions.len());
    });

    // Single value
    group.register("single_value", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut positions = Vec::new();
        col.get_row_ids_for_value_range(50000..=50000, 0..col.num_vals(), &mut positions);
        black_box(positions.len());
    });

    group.run();
}

/// Benchmark with different data sizes
fn bench_scalability() {
    let sizes = [1_000, 10_000, 100_000];
    let codecs = get_all_codecs();

    for size in sizes {
        println!("\n--- Size: {} values ---", size);

        let data = generate_price_data(size);
        let columns: Vec<_> = codecs
            .iter()
            .map(|(name, codec)| {
                let col = serialize_and_load(&data, *codec);
                (format!("n{}_{}", size, name), col)
            })
            .collect();

        let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> =
            InputGroup::new_with_inputs(columns);

        group.register("fullscan", |col: &Arc<dyn ColumnValues<u64>>| {
            let mut sum = 0u64;
            for i in 0..col.num_vals() {
                sum = sum.wrapping_add(col.get_val(i));
            }
            black_box(sum);
        });

        group.run();
    }
}

/// Benchmark aggregation-like operations
fn bench_aggregations() {
    let n = 100_000;
    let price_data = generate_price_data(n);
    let codecs = get_all_codecs();

    let columns: Vec<_> = codecs
        .iter()
        .map(|(name, codec)| {
            let col = serialize_and_load(&price_data, *codec);
            (format!("agg_{}", name), col)
        })
        .collect();

    let mut group: InputGroup<Arc<dyn ColumnValues<u64>>> = InputGroup::new_with_inputs(columns);

    // Sum aggregation
    group.register("sum", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut sum = 0u64;
        for i in 0..col.num_vals() {
            sum = sum.wrapping_add(col.get_val(i));
        }
        black_box(sum);
    });

    // Min/Max aggregation
    group.register("min_max", |col: &Arc<dyn ColumnValues<u64>>| {
        let mut min = u64::MAX;
        let mut max = 0u64;
        for i in 0..col.num_vals() {
            let val = col.get_val(i);
            min = min.min(val);
            max = max.max(val);
        }
        black_box((min, max));
    });

    // Count distinct (approximation by checking uniqueness in sample)
    group.register("sample_distinct", |col: &Arc<dyn ColumnValues<u64>>| {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let sample_size = col.num_vals().min(10_000);
        for i in 0..sample_size {
            seen.insert(col.get_val(i));
        }
        black_box(seen.len());
    });

    group.run();
}

fn main() {
    println!("=== BUFF Codec Benchmark Suite ===\n");

    // Print compression ratio summary
    print_compression_ratios();

    // Run access pattern benchmarks
    println!("\n=== Access Pattern Benchmarks ===\n");
    bench_access_patterns();

    // Run range query benchmarks
    println!("\n=== Range Query Benchmarks ===\n");
    bench_range_queries();

    // Run scalability benchmarks
    println!("\n=== Scalability Benchmarks ===");
    bench_scalability();

    // Run aggregation benchmarks
    println!("\n=== Aggregation Benchmarks ===\n");
    bench_aggregations();
}

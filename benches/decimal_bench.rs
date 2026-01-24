//! Benchmark for Decimal fields with BUFF compression
//!
//! This benchmark measures the performance of Decimal fields stored as
//! fixed-point integers in fast fields, including:
//! - Indexing decimal values
//! - Reading decimal values
//! - Aggregations (sum, min, max, avg)
//! - Range queries on decimal data
//!
//! Run with:
//!   cargo bench --bench decimal_bench --features columnar-buff-compression

use std::str::FromStr;
use std::sync::Arc;

use binggan::{InputGroup, black_box};
use tantivy::fastfield::fixed_point_to_decimal_string;
use tantivy::schema::{DecimalOptions, DecimalValue, OwnedValue, Schema, FAST, STORED};
use tantivy::{Index, IndexWriter, TantivyDocument};

/// Generate test decimal values (prices with 4 decimal places)
fn generate_decimal_values(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            // Prices from $10.0000 to $999.9999
            let dollars = 10 + (i % 990);
            let cents = i % 10000;
            format!("{}.{:04}", dollars, cents)
        })
        .collect()
}

/// Generate high-precision decimal values (6 decimal places, like crypto)
fn generate_high_precision_values(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            // Values from 0.000001 to 999.999999
            let whole = i % 1000;
            let fraction = (i * 7) % 1_000_000;
            format!("{}.{:06}", whole, fraction)
        })
        .collect()
}

/// Create an index with decimal field and return (index, field, scale)
fn create_decimal_index(values: &[String], precision: u32, scale: i32) -> (Index, tantivy::schema::Field, i32) {
    let mut schema_builder = Schema::builder();
    let decimal_options = DecimalOptions::default()
        .set_fast()
        .set_stored()
        .set_precision(precision)
        .set_scale(scale);
    let decimal_field = schema_builder.add_decimal_field("amount", decimal_options);
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema);
    let mut index_writer: IndexWriter = index.writer(50_000_000).unwrap();

    for val_str in values {
        let decimal = DecimalValue::from_str(val_str).unwrap();
        let mut doc = TantivyDocument::default();
        doc.add_field_value(decimal_field, &OwnedValue::Decimal(decimal));
        index_writer.add_document(doc).unwrap();
    }
    index_writer.commit().unwrap();

    (index, decimal_field, scale)
}

/// Benchmark decimal field indexing
fn bench_decimal_indexing() {
    println!("--- Decimal Indexing Benchmarks ---\n");

    let sizes = [1_000, 10_000, 100_000];

    for &size in &sizes {
        let values = generate_decimal_values(size);
        
        let inputs: Vec<_> = vec![
            (format!("index_decimal_{}", size), values.clone()),
        ];

        let mut group: InputGroup<Vec<String>> = InputGroup::new_with_inputs(inputs);

        group.register("indexing", |values: &Vec<String>| {
            let mut schema_builder = Schema::builder();
            let decimal_options = DecimalOptions::default()
                .set_fast()
                .set_stored()
                .set_precision(12)
                .set_scale(4);
            let decimal_field = schema_builder.add_decimal_field("amount", decimal_options);
            let schema = schema_builder.build();

            let index = Index::create_in_ram(schema);
            let mut index_writer: IndexWriter = index.writer(50_000_000).unwrap();

            for val_str in values {
                let decimal = DecimalValue::from_str(val_str).unwrap();
                let mut doc = TantivyDocument::default();
                doc.add_field_value(decimal_field, &OwnedValue::Decimal(decimal));
                index_writer.add_document(doc).unwrap();
            }
            index_writer.commit().unwrap();
            black_box(index);
        });

        group.run();
    }
}

/// Benchmark decimal field reading
fn bench_decimal_reading() {
    println!("\n--- Decimal Reading Benchmarks ---\n");

    let sizes = [10_000, 100_000];

    for &size in &sizes {
        let values = generate_decimal_values(size);
        let (index, _field, scale) = create_decimal_index(&values, 12, 4);

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0);
        let fast_fields = segment_reader.fast_fields();
        let col = fast_fields.decimal_i64("amount").unwrap();

        let inputs: Vec<_> = vec![(format!("read_decimal_{}", size), Arc::new(col))];

        type ColType = Arc<tantivy::columnar::Column<i64>>;
        let mut group: InputGroup<ColType> = InputGroup::new_with_inputs(inputs);

        // Sequential read
        group.register("sequential_read", |col: &ColType| {
            let mut sum = 0i64;
            for doc_id in 0..col.num_docs() {
                if let Some(val) = col.first(doc_id) {
                    sum += val;
                }
            }
            black_box(sum);
        });

        // Random access read
        group.register("random_access", |col: &ColType| {
            let mut sum = 0i64;
            let n = col.num_docs();
            for i in (0..n).step_by(10) {
                if let Some(val) = col.first(i) {
                    sum += val;
                }
            }
            black_box(sum);
        });

        // Read with conversion to decimal string
        group.register("read_with_conversion", move |col: &ColType| {
            let mut results = Vec::with_capacity(100);
            for doc_id in 0..100u32.min(col.num_docs()) {
                if let Some(val) = col.first(doc_id) {
                    results.push(fixed_point_to_decimal_string(val, scale));
                }
            }
            black_box(results);
        });

        group.run();
    }
}

/// Benchmark decimal aggregations
fn bench_decimal_aggregations() {
    println!("\n--- Decimal Aggregation Benchmarks ---\n");

    let values = generate_decimal_values(100_000);
    let (index, _field, _scale) = create_decimal_index(&values, 12, 4);

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let segment_reader = searcher.segment_reader(0);
    let fast_fields = segment_reader.fast_fields();
    let col = fast_fields.decimal_i64("amount").unwrap();

    let inputs: Vec<_> = vec![("decimal_100k".to_string(), Arc::new(col))];

    type ColType = Arc<tantivy::columnar::Column<i64>>;
    let mut group: InputGroup<ColType> = InputGroup::new_with_inputs(inputs);

    // Sum aggregation
    group.register("sum", |col: &ColType| {
        let mut sum = 0i64;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                sum += val;
            }
        }
        black_box(sum);
    });

    // Min aggregation
    group.register("min", |col: &ColType| {
        let mut min = i64::MAX;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                min = min.min(val);
            }
        }
        black_box(min);
    });

    // Max aggregation
    group.register("max", |col: &ColType| {
        let mut max = i64::MIN;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                max = max.max(val);
            }
        }
        black_box(max);
    });

    // Full stats (sum, min, max, count)
    group.register("full_stats", |col: &ColType| {
        let mut sum = 0i64;
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        let mut count = 0u32;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                sum += val;
                min = min.min(val);
                max = max.max(val);
                count += 1;
            }
        }
        let avg = sum as f64 / count as f64;
        black_box((sum, min, max, avg, count));
    });

    group.run();
}

/// Benchmark high-precision decimal operations
fn bench_high_precision_decimal() {
    println!("\n--- High-Precision Decimal (scale=6) Benchmarks ---\n");

    let values = generate_high_precision_values(100_000);
    let (index, _field, _scale) = create_decimal_index(&values, 15, 6);

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let segment_reader = searcher.segment_reader(0);
    let fast_fields = segment_reader.fast_fields();
    let col = fast_fields.decimal_i64("amount").unwrap();

    let inputs: Vec<_> = vec![("highprec_100k".to_string(), Arc::new(col))];

    type ColType = Arc<tantivy::columnar::Column<i64>>;
    let mut group: InputGroup<ColType> = InputGroup::new_with_inputs(inputs);

    // Sum with high precision
    group.register("highprec_sum", |col: &ColType| {
        let mut sum = 0i64;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                sum += val;
            }
        }
        black_box(sum);
    });

    // Filter scan
    group.register("highprec_filter", |col: &ColType| {
        let mut count = 0u32;
        let threshold = 500_000_000i64; // 500.000000
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                if val > threshold {
                    count += 1;
                }
            }
        }
        black_box(count);
    });

    group.run();
}

/// Compare Decimal vs f64 performance
fn bench_decimal_vs_f64() {
    println!("\n--- Decimal vs f64 Comparison ---\n");

    let n = 100_000usize;

    // Create decimal index
    let decimal_values = generate_decimal_values(n);
    let (decimal_index, _, _) = create_decimal_index(&decimal_values, 12, 4);

    // Create f64 index
    let mut schema_builder = Schema::builder();
    let f64_field = schema_builder.add_f64_field("amount", FAST | STORED);
    let schema = schema_builder.build();
    let f64_index = Index::create_in_ram(schema);
    let mut index_writer: IndexWriter = f64_index.writer(50_000_000).unwrap();
    for val_str in &decimal_values {
        let f64_val: f64 = val_str.parse().unwrap();
        let mut doc = TantivyDocument::default();
        doc.add_f64(f64_field, f64_val);
        index_writer.add_document(doc).unwrap();
    }
    index_writer.commit().unwrap();

    // Get fast field readers
    let decimal_reader = decimal_index.reader().unwrap();
    let decimal_searcher = decimal_reader.searcher();
    let decimal_col = decimal_searcher
        .segment_reader(0)
        .fast_fields()
        .decimal_i64("amount")
        .unwrap();

    let f64_reader = f64_index.reader().unwrap();
    let f64_searcher = f64_reader.searcher();
    let f64_col = f64_searcher
        .segment_reader(0)
        .fast_fields()
        .f64("amount")
        .unwrap()
        .first_or_default_col(0.0);

    // Benchmark sum on both types
    println!("Sum aggregation comparison:");

    let inputs_decimal: Vec<_> = vec![("decimal".to_string(), Arc::new(decimal_col))];
    type DecimalColType = Arc<tantivy::columnar::Column<i64>>;
    let mut group_decimal: InputGroup<DecimalColType> = InputGroup::new_with_inputs(inputs_decimal);
    group_decimal.register("decimal_sum", |col: &DecimalColType| {
        let mut sum = 0i64;
        for doc_id in 0..col.num_docs() {
            if let Some(val) = col.first(doc_id) {
                sum += val;
            }
        }
        black_box(sum);
    });
    group_decimal.run();

    let inputs_f64: Vec<_> = vec![("f64".to_string(), f64_col)];
    type F64ColType = Arc<dyn tantivy::columnar::ColumnValues<f64>>;
    let mut group_f64: InputGroup<F64ColType> = InputGroup::new_with_inputs(inputs_f64);
    group_f64.register("f64_sum", |col: &F64ColType| {
        let mut sum = 0.0f64;
        for i in 0..col.num_vals() {
            sum += col.get_val(i);
        }
        black_box(sum);
    });
    group_f64.run();
}

fn main() {
    println!("=== Decimal Field Benchmark Suite ===\n");

    bench_decimal_indexing();
    bench_decimal_reading();
    bench_decimal_aggregations();
    bench_high_precision_decimal();
    bench_decimal_vs_f64();

    println!("\n=== Benchmark Complete ===");
}

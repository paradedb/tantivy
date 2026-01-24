// # BUFF Compression Example
//
// This example demonstrates how to use BUFF (Byte-sliced) compression for
// numeric fast fields. BUFF is particularly effective for bounded-precision
// numeric data like financial values, sensor readings, and metrics.
//
// Run with:
//   cargo run --example buff_compression --features columnar-buff-compression
// ---

use tantivy::schema::{Schema, FAST, STORED};
use tantivy::{Index, IndexWriter, TantivyDocument};

fn main() -> tantivy::Result<()> {
    // # Create Schema
    //
    // We'll create a schema for a stock trading system with:
    // - ticker: text field for stock symbol
    // - price: f64 field for stock price (ideal for BUFF - bounded precision)
    // - volume: u64 field for trading volume
    // - change_percent: f64 field for daily change percentage

    let mut schema_builder = Schema::builder();
    let ticker_field = schema_builder.add_text_field("ticker", STORED);
    let price_field = schema_builder.add_f64_field("price", FAST | STORED);
    let volume_field = schema_builder.add_u64_field("volume", FAST | STORED);
    let change_field = schema_builder.add_f64_field("change_percent", FAST | STORED);

    let schema = schema_builder.build();

    // # Create Index
    //
    // When the `columnar-buff-compression` feature is enabled, the columnar
    // storage will automatically consider BUFF as a codec option for numeric
    // columns. The best codec is selected based on data characteristics.

    let index = Index::create_in_ram(schema.clone());

    // # Index Documents
    //
    // We'll add sample stock market data. This type of data - with bounded
    // precision decimals - is exactly what BUFF compression is designed for.

    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // Sample stock data
    let stocks = vec![
        ("AAPL", 185.92, 45_000_000u64, 1.25),
        ("GOOGL", 141.80, 22_000_000u64, -0.83),
        ("MSFT", 378.91, 18_000_000u64, 0.42),
        ("AMZN", 178.25, 35_000_000u64, 2.15),
        ("TSLA", 248.50, 95_000_000u64, -1.67),
        ("META", 505.75, 12_000_000u64, 0.98),
        ("NVDA", 875.28, 42_000_000u64, 3.45),
        ("AMD", 162.33, 55_000_000u64, -0.22),
        ("INTC", 43.21, 28_000_000u64, -2.10),
        ("NFLX", 628.90, 8_000_000u64, 1.05),
    ];

    for (ticker, price, volume, change) in &stocks {
        let mut doc = TantivyDocument::default();
        doc.add_text(ticker_field, ticker);
        doc.add_f64(price_field, *price);
        doc.add_u64(volume_field, *volume);
        doc.add_f64(change_field, *change);
        index_writer.add_document(doc)?;
    }

    // Add more data to demonstrate BUFF compression benefits at scale
    // Generate 10,000 synthetic stock entries
    for i in 0..10_000 {
        let mut doc = TantivyDocument::default();
        doc.add_text(ticker_field, format!("SYM{:05}", i));
        // Prices between $10 and $1000 with 2 decimal precision
        doc.add_f64(price_field, 10.0 + (i as f64 % 990.0) + (i as f64 % 100.0) / 100.0);
        // Volume between 1M and 100M
        doc.add_u64(volume_field, 1_000_000 + (i as u64 * 9900));
        // Change between -5% and +5%
        doc.add_f64(change_field, -5.0 + (i as f64 % 1000.0) / 100.0);
        index_writer.add_document(doc)?;
    }

    index_writer.commit()?;

    // # Read Fast Fields
    //
    // Access the fast fields to demonstrate round-trip correctness

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment_reader = searcher.segment_reader(0);
    let fast_fields = segment_reader.fast_fields();

    // Access compressed numeric columns
    let prices = fast_fields.f64("price")?.first_or_default_col(0.0);
    let volumes = fast_fields.u64("volume")?.first_or_default_col(0);
    let changes = fast_fields.f64("change_percent")?.first_or_default_col(0.0);

    println!("=== BUFF Compression Example ===\n");
    println!("Indexed {} documents\n", prices.num_vals());

    // Show statistics for the price column
    println!("Price column statistics:");
    println!("  Min: ${:.2}", prices.min_value());
    println!("  Max: ${:.2}", prices.max_value());
    println!();

    // Show first 10 documents from the fast fields
    println!("Sample data (first 10 documents):");
    println!(
        "{:<8} {:>10} {:>12} {:>10}",
        "Doc ID", "Price", "Volume", "Change%"
    );
    println!("{}", "-".repeat(44));

    for i in 0..10u32 {
        println!(
            "Doc {:>3}  ${:>9.2} {:>12} {:>+9.2}%",
            i,
            prices.get_val(i),
            volumes.get_val(i),
            changes.get_val(i)
        );
    }

    // # Fast Field Range Query
    //
    // Direct fast field range query (efficient for analytics on compressed data)

    println!("\n=== Fast Field Range Query: High Volume (>50M) ===\n");

    let mut high_volume_docs = Vec::new();
    volumes.get_row_ids_for_value_range(50_000_000..=u64::MAX, 0..10, &mut high_volume_docs);

    println!(
        "Found {} high-volume stocks in first 10 docs:",
        high_volume_docs.len()
    );
    for doc_id in high_volume_docs {
        println!("  Doc {}: volume = {}", doc_id, volumes.get_val(doc_id));
    }

    // # Price Range Query on Fast Field
    //
    // Query prices in the $100-$200 range
    // Note: For f64 columns, we scan and filter since range queries
    // work best on integer types with the fast field API

    println!("\n=== Price Range Query: Prices $100-$200 ===\n");

    let mut mid_price_docs = Vec::new();
    for doc_id in 0..20u32 {
        let price = prices.get_val(doc_id);
        if price >= 100.0 && price <= 200.0 {
            mid_price_docs.push(doc_id);
        }
    }

    println!(
        "Found {} stocks priced $100-$200 in first 20 docs:",
        mid_price_docs.len()
    );
    for doc_id in mid_price_docs.iter().take(5) {
        println!(
            "  Doc {}: price = ${:.2}",
            doc_id,
            prices.get_val(*doc_id)
        );
    }
    if mid_price_docs.len() > 5 {
        println!("  ... and {} more", mid_price_docs.len() - 5);
    }

    // # Bulk Value Retrieval
    //
    // Efficient batch access to compressed values

    println!("\n=== Bulk Value Retrieval ===\n");

    let doc_ids: Vec<u32> = (0..5).collect();
    let mut price_buffer = vec![0.0f64; 5];
    prices.get_vals(&doc_ids, &mut price_buffer);

    println!("Batch retrieved prices for docs 0-4:");
    for (i, price) in price_buffer.iter().enumerate() {
        println!("  Doc {}: ${:.2}", i, price);
    }

    println!("\n=== BUFF Compression Benefits ===\n");
    println!("BUFF compression is particularly effective for:");
    println!("  - Financial data with fixed decimal precision");
    println!("  - Sensor readings with bounded ranges");
    println!("  - Metrics with known precision requirements");
    println!();
    println!("The codec automatically optimizes storage based on:");
    println!("  - Value distribution and range");
    println!("  - GCD (Greatest Common Divisor) patterns");
    println!("  - Byte-sliced storage for efficient queries");

    Ok(())
}

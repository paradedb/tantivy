#!/usr/bin/env python3
"""PDXearch (DuckDB) benchmark for comparison with tantivy vector search.

Uses GIST-1M dataset for unfiltered search, Cohere 1M with scalar labels for filtered.
"""

import os
import sys
import time
import struct
import subprocess

DUCKDB = os.path.expanduser("~/bin/duckdb")
PDXEARCH_EXT = os.path.expanduser("~/PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension")
GIST_DIR = os.path.expanduser("~/rabitq-rs-bench/data/gist/gist")
COHERE_DIR = os.path.expanduser("~/.cache/tantivy_bench/cohere_10m")
DB_PATH = os.path.expanduser("~/.cache/tantivy_bench/pdxearch_bench.duckdb")


def run_duckdb(sql, db=DB_PATH):
    """Run SQL in DuckDB and return output."""
    result = subprocess.run(
        [DUCKDB, db, "-c", sql],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", flush=True)
    return result.stdout


def bench_gist_unfiltered():
    """Benchmark PDXearch on GIST-1M unfiltered."""
    print("=== PDXearch GIST-1M Unfiltered ===", flush=True)

    # Load extension
    run_duckdb(f"LOAD '{PDXEARCH_EXT}';")

    # Create table and load GIST vectors
    print("Loading GIST-1M vectors...", flush=True)
    # DuckDB can't read fvecs directly — need to convert via Python first
    # For now, use the parquet approach or skip if too complex
    print("  (GIST fvecs loading requires conversion — skipping for now)")
    print("  Use Cohere parquet benchmark instead.")


def bench_cohere_filtered():
    """Benchmark PDXearch on Cohere 1M with scalar label filter."""
    print("=== PDXearch Cohere 1M Filtered ===", flush=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Load extension and create table
    setup_sql = f"""
    LOAD '{PDXEARCH_EXT}';

    CREATE TABLE vectors AS
    SELECT id, emb, label
    FROM (
        SELECT v.id, v.emb, l.labels as label
        FROM read_parquet('{COHERE_DIR}/train-00-of-10.parquet') v
        JOIN read_parquet('{COHERE_DIR}/scalar_labels.parquet') l
        ON v.id = l.id
    );

    SELECT COUNT(*) as num_rows FROM vectors;
    """
    print("Creating table with vectors + labels...", flush=True)
    t0 = time.time()
    output = run_duckdb(setup_sql)
    print(f"  Table created in {time.time()-t0:.1f}s", flush=True)
    print(f"  {output.strip()}", flush=True)

    # Create PDXearch index
    print("Creating PDXearch index...", flush=True)
    t0 = time.time()
    index_sql = f"""
    LOAD '{PDXEARCH_EXT}';
    CREATE INDEX vec_idx ON vectors USING pdxearch (emb)
    WITH (metric = 'l2', num_clusters = 400);
    """
    run_duckdb(index_sql)
    print(f"  Index created in {time.time()-t0:.1f}s", flush=True)

    # Load queries
    print("Loading queries...", flush=True)
    # Read first 100 queries from test.parquet
    query_sql = f"""
    LOAD '{PDXEARCH_EXT}';
    SELECT emb FROM read_parquet('{COHERE_DIR}/test.parquet') LIMIT 100;
    """

    # Benchmark unfiltered
    print("\nBenchmark (top-100, 100 queries):", flush=True)

    # Unfiltered search
    bench_sql = f"""
    LOAD '{PDXEARCH_EXT}';
    SET pdxearch_n_probe = 10;

    .timer on

    SELECT v.id
    FROM vectors v
    ORDER BY array_distance(v.emb, (SELECT emb FROM read_parquet('{COHERE_DIR}/test.parquet') LIMIT 1)::FLOAT[768])
    LIMIT 100;
    """
    print("Running unfiltered query...", flush=True)
    t0 = time.time()
    for i in range(10):
        run_duckdb(bench_sql)
    elapsed = time.time() - t0
    print(f"  Unfiltered: {10/elapsed:.0f} QPS ({elapsed/10*1000:.0f}ms avg)", flush=True)

    # Filtered search (1% filter)
    bench_filtered_sql = f"""
    LOAD '{PDXEARCH_EXT}';
    SET pdxearch_n_probe = 10;

    SELECT v.id
    FROM vectors v
    WHERE v.label = 'label_1p'
    ORDER BY array_distance(v.emb, (SELECT emb FROM read_parquet('{COHERE_DIR}/test.parquet') LIMIT 1)::FLOAT[768])
    LIMIT 100;
    """
    print("Running filtered (1%) query...", flush=True)
    t0 = time.time()
    for i in range(10):
        run_duckdb(bench_filtered_sql)
    elapsed = time.time() - t0
    print(f"  Filtered 1%: {10/elapsed:.0f} QPS ({elapsed/10*1000:.0f}ms avg)", flush=True)

    # Filtered search (50% filter)
    bench_filtered_50_sql = f"""
    LOAD '{PDXEARCH_EXT}';
    SET pdxearch_n_probe = 10;

    SELECT v.id
    FROM vectors v
    WHERE v.label = 'label_50p'
    ORDER BY array_distance(v.emb, (SELECT emb FROM read_parquet('{COHERE_DIR}/test.parquet') LIMIT 1)::FLOAT[768])
    LIMIT 100;
    """
    print("Running filtered (50%) query...", flush=True)
    t0 = time.time()
    for i in range(10):
        run_duckdb(bench_filtered_50_sql)
    elapsed = time.time() - t0
    print(f"  Filtered 50%: {10/elapsed:.0f} QPS ({elapsed/10*1000:.0f}ms avg)", flush=True)


if __name__ == "__main__":
    bench_cohere_filtered()

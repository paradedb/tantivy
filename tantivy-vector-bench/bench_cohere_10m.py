#!/usr/bin/env python3
"""Benchmark tantivy vector search on Cohere 10M (768d, cosine/L2)."""

import os
import sys
import time
import tempfile
import shutil

import pyarrow.parquet as pq

from tantivy_vector_bench import TantivyVectorIndex

CACHE_DIR = os.path.expanduser("~/.cache/tantivy_bench/cohere_10m")
DIM = 768
NUM_SHARDS = 10
BATCH_SIZE = 5000


def iter_train_batches(shard_idx, batch_size=BATCH_SIZE):
    path = os.path.join(CACHE_DIR, f"train-{shard_idx:02d}-of-10.parquet")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "emb"]):
        ids = batch.column("id").to_pylist()
        embs = [v.as_py() for v in batch.column("emb")]
        yield ids, embs


def load_test_queries():
    path = os.path.join(CACHE_DIR, "test.parquet")
    table = pq.read_table(path, columns=["id", "emb"])
    return [row.as_py() for row in table.column("emb")]


def load_ground_truth():
    path = os.path.join(CACHE_DIR, "neighbors.parquet")
    table = pq.read_table(path)
    return [row.as_py() for row in table.column("neighbors_id")]


def compute_recall(results, ground_truth, k):
    return len(set(results[:k]) & set(ground_truth[:k])) / k


def main():
    num_shards = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SHARDS
    print(f"=== Cohere {num_shards}M Benchmark (dim={DIM}) ===", flush=True)

    for f in ["train-00-of-10.parquet", "test.parquet", "neighbors.parquet"]:
        if not os.path.exists(os.path.join(CACHE_DIR, f)):
            print(f"ERROR: {f} not found in {CACHE_DIR}")
            sys.exit(1)

    data_dir = os.path.expanduser(f"~/.cache/tantivy_bench/index_cohere_{num_shards}m")
    skip_indexing = os.path.exists(os.path.join(data_dir, "meta.json"))
    print(f"Index dir: {data_dir}", flush=True)
    if skip_indexing:
        print("  (index exists, skipping indexing)", flush=True)

    if not skip_indexing:
        # --- Indexing ---
        print(f"\n--- Indexing ({num_shards} shards) ---", flush=True)
        idx = TantivyVectorIndex(
            DIM, "l2", data_dir, CACHE_DIR,
            num_shards=10, vectors_per_shard=1_000_000, num_clusters_per_1k=1,
        )

        total_inserted = 0
        t_index_start = time.time()

        for shard in range(num_shards):
            shard_count = 0
            t0 = time.time()
            for ids, embs in iter_train_batches(shard):
                count = idx.insert(ids, embs)
                shard_count += count
                total_inserted += count
            idx.commit()
            elapsed = time.time() - t0
            vps = shard_count / elapsed if elapsed > 0 else 0
            print(
                f"  shard {shard:2d}: {shard_count:>8,} vectors in {elapsed:.1f}s "
                f"({vps:,.0f} vec/s)  total={total_inserted:>10,}",
                flush=True,
            )

        t_insert_total = time.time() - t_index_start

        print(f"\nFinalizing...", flush=True)
        t0 = time.time()
        idx.finalize()
        t_finalize = time.time() - t0

        t_total = t_insert_total + t_finalize
        vecs_per_sec = total_inserted / t_total
        print(f"\n  Vectors:    {total_inserted:>10,}")
        print(f"  Insert:     {t_insert_total:>10.1f}s")
        print(f"  Finalize:   {t_finalize:>10.1f}s")
        print(f"  Total:      {t_total:>10.1f}s")
        print(f"  Throughput: {vecs_per_sec:>10,.0f} vectors/sec", flush=True)

    # --- Search ---
    if skip_indexing:
        idx = TantivyVectorIndex.open(DIM, "l2", data_dir)
    print(f"\n--- Search ---", flush=True)
    queries = load_test_queries()
    ground_truth = load_ground_truth()
    num_queries = len(queries)
    print(f"  {num_queries} queries loaded", flush=True)

    configs = [
        (1, 1.0),
        (5, 2.0),
        (10, 3.0),
        (20, 5.0),
        (50, 10.0),
        (100, 100.0),
    ]

    k = 100
    print(f"\n  {'config':>20s}  {'recall@10':>10s}  {'recall@100':>10s}  {'latency':>10s}  {'QPS':>8s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for max_probe, ratio in configs:
        recalls_10 = []
        recalls_100 = []
        t0 = time.time()
        for qi in range(num_queries):
            results = idx.search(queries[qi], k=k, max_probe=max_probe, distance_ratio=ratio)
            recalls_10.append(compute_recall(results, ground_truth[qi], 10))
            recalls_100.append(compute_recall(results, ground_truth[qi], k))
        elapsed = time.time() - t0

        avg_r10 = sum(recalls_10) / len(recalls_10)
        avg_r100 = sum(recalls_100) / len(recalls_100)
        avg_latency = elapsed / num_queries * 1000
        qps = num_queries / elapsed

        label = f"probe({max_probe},{ratio:.0f})"
        print(f"  {label:>20s}  {avg_r10:>10.3f}  {avg_r100:>10.3f}  {avg_latency:>8.1f}ms  {qps:>8.0f}", flush=True)

    print(f"\nIndex persisted at {data_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

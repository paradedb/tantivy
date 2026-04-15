#!/usr/bin/env python3
"""Filtered vector search benchmark using Cohere 1M + scalar labels."""

import os
import sys
import time
import pickle

import pyarrow.parquet as pq

from tantivy_vector_bench import TantivyVectorIndex

CACHE_DIR = os.path.expanduser("~/.cache/tantivy_bench")
PARQUET_DIR = os.path.join(CACHE_DIR, "cohere_10m")
DIM = 768
K = 100
BATCH_SIZE = 5000


def main():
    total_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    label_filter = sys.argv[2] if len(sys.argv) > 2 else "label_1p"

    data_dir = os.path.join(CACHE_DIR, f"index_cohere_filtered_{total_bits}b")
    skip_indexing = os.path.exists(os.path.join(data_dir, "meta.json"))

    print(f"=== Filtered Vector Search Benchmark ===", flush=True)
    print(f"total_bits={total_bits}, filter_label={label_filter}", flush=True)

    # Load queries + unfiltered GT
    print("Loading queries...", flush=True)
    queries_table = pq.read_table(f"{PARQUET_DIR}/test.parquet", columns=["emb"])
    queries = [row.as_py() for row in queries_table.column("emb")]
    print(f"  {len(queries)} queries", flush=True)

    # Load scalar labels for shard 0 (1M vectors)
    print("Loading labels...", flush=True)
    labels_table = pq.read_table(f"{PARQUET_DIR}/scalar_labels.parquet", columns=["id", "labels"])
    all_labels = dict(zip(
        labels_table.column("id").to_pylist(),
        labels_table.column("labels").to_pylist(),
    ))

    if not skip_indexing:
        print(f"\nIndexing shard 0 (1M vectors) with labels...", flush=True)
        idx = TantivyVectorIndex(DIM, "l2", data_dir, PARQUET_DIR,
            num_shards=1, vectors_per_shard=1_000_000, num_clusters_per_1k=1,
            total_bits=total_bits)

        COMMIT_EVERY = 100_000
        t0 = time.time()
        since_commit = 0
        pf = pq.ParquetFile(f"{PARQUET_DIR}/train-00-of-10.parquet")
        total = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["id", "emb"]):
            ids = batch.column("id").to_pylist()
            embs = [v.as_py() for v in batch.column("emb")]
            labels = [all_labels.get(i, "") for i in ids]
            idx.insert(ids, embs, labels)
            total += len(ids)
            since_commit += len(ids)
            if since_commit >= COMMIT_EVERY:
                idx.commit()
                since_commit = 0
        if since_commit > 0:
            idx.commit()
        idx.finalize()
        t_index = time.time() - t0
        print(f"Indexed {total} vectors in {t_index:.1f}s ({total/t_index:.0f} vec/s)", flush=True)
    else:
        print(f"  (index exists, skipping indexing)", flush=True)
        idx = TantivyVectorIndex.open(DIM, "l2", data_dir, total_bits)

    # Count matching docs
    label_count = sum(1 for i in range(1_000_000) if all_labels.get(i, "") == label_filter)
    selectivity = label_count / 1_000_000 * 100
    print(f"\nFilter: label='{label_filter}' matches {label_count:,} docs ({selectivity:.1f}%)", flush=True)

    # Compute brute-force ground truth for filtered search
    gt_cache = os.path.join(CACHE_DIR, f"gt_filtered_{label_filter}.pkl")
    if os.path.exists(gt_cache):
        print("Loading cached filtered ground truth...", flush=True)
        with open(gt_cache, "rb") as f:
            ground_truth = pickle.load(f)
    else:
        print("Computing filtered ground truth (brute force)...", flush=True)
        import numpy as np

        # Load shard 0 vectors
        vecs_cache = os.path.join(CACHE_DIR, "cohere_1m_vecs.pkl")
        if os.path.exists(vecs_cache):
            with open(vecs_cache, "rb") as f:
                _, all_embs = pickle.load(f)
        else:
            all_embs = []
            pf = pq.ParquetFile(f"{PARQUET_DIR}/train-00-of-10.parquet")
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["emb"]):
                all_embs.extend([v.as_py() for v in batch.column("emb")])

        # Filter to matching docs
        filtered_ids = [i for i in range(len(all_embs)) if all_labels.get(i, "") == label_filter]
        filtered_vecs = np.array([all_embs[i] for i in filtered_ids], dtype=np.float32)
        q_arr = np.array(queries, dtype=np.float32)

        ground_truth = []
        for qi in range(len(queries)):
            diffs = filtered_vecs - q_arr[qi]
            dists = np.sum(diffs * diffs, axis=1)
            top_k_idx = np.argpartition(dists, min(K, len(dists)-1))[:K]
            top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
            ground_truth.append([filtered_ids[j] for j in top_k_idx])
            if (qi + 1) % 100 == 0:
                print(f"  {qi+1}/{len(queries)}", flush=True)

        with open(gt_cache, "wb") as f:
            pickle.dump(ground_truth, f)
        print(f"  Done.", flush=True)

    def recall_at_k(results, gt, k):
        return len(set(results[:k]) & set(gt[:k])) / min(k, len(gt))

    # Benchmark
    num_queries = len(queries)
    nprobe = 10
    print(f"\nBenchmark (nprobe={nprobe}, top-{K}, {num_queries} queries):", flush=True)
    print(f"{'':>10}  {'QPS':>8}  {'Recall@10':>10}  {'Recall@100':>10}", flush=True)
    print("-" * 45, flush=True)

    # Unfiltered
    r10s, r100s = [], []
    t0 = time.time()
    for qi in range(num_queries):
        results = idx.search(queries[qi], k=K, max_probe=nprobe, distance_ratio=1000.0)
        # Can't compute recall without unfiltered GT — skip
    elapsed = time.time() - t0
    qps = num_queries / elapsed
    print(f"{'unfiltered':>10}  {qps:>8.0f}  {'n/a':>10}  {'n/a':>10}", flush=True)

    # Filtered
    r10s, r100s = [], []
    t0 = time.time()
    for qi in range(num_queries):
        results = idx.search_filtered(queries[qi], label_filter, k=K, max_probe=nprobe, distance_ratio=1000.0)
        r10s.append(recall_at_k(results, ground_truth[qi], 10))
        r100s.append(recall_at_k(results, ground_truth[qi], K))
    elapsed = time.time() - t0
    qps = num_queries / elapsed
    avg_r10 = sum(r10s) / len(r10s)
    avg_r100 = sum(r100s) / len(r100s)
    print(f"{'filtered':>10}  {qps:>8.0f}  {avg_r10:>10.3f}  {avg_r100:>10.3f}", flush=True)

    print(f"\nIndex at {data_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

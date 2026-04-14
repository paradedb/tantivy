#!/usr/bin/env python3
"""Self-contained 1M Cohere benchmark with cached vectors and ground truth."""

import os
import sys
import time
import tempfile
import shutil
import pickle

import numpy as np
import pyarrow.parquet as pq

from tantivy_vector_bench import TantivyVectorIndex

CACHE_DIR = os.path.expanduser("~/.cache/tantivy_bench")
PARQUET_DIR = os.path.join(CACHE_DIR, "cohere_10m")
VEC_CACHE = os.path.join(CACHE_DIR, "cohere_1m_vecs.pkl")
GT_CACHE = os.path.join(CACHE_DIR, "cohere_1m_gt.pkl")
DIM = 768
NUM_QUERIES = 100
K = 100


def load_vectors():
    if os.path.exists(VEC_CACHE):
        print("Loading cached vectors...", flush=True)
        with open(VEC_CACHE, "rb") as f:
            return pickle.load(f)

    print("Loading shard 0 from parquet...", flush=True)
    t0 = time.time()
    pf = pq.ParquetFile(f"{PARQUET_DIR}/train-00-of-10.parquet")
    all_ids = []
    all_embs = []
    for batch in pf.iter_batches(batch_size=5000, columns=["id", "emb"]):
        all_ids.extend(batch.column("id").to_pylist())
        all_embs.extend([v.as_py() for v in batch.column("emb")])
    print(f"Loaded {len(all_embs)} vectors in {time.time()-t0:.1f}s", flush=True)

    print("Caching vectors...", flush=True)
    with open(VEC_CACHE, "wb") as f:
        pickle.dump((all_ids, all_embs), f)

    return all_ids, all_embs


def compute_ground_truth(queries, corpus):
    if os.path.exists(GT_CACHE):
        print("Loading cached ground truth...", flush=True)
        with open(GT_CACHE, "rb") as f:
            return pickle.load(f)

    print("Computing ground truth with numpy...", flush=True)
    t0 = time.time()
    q_arr = np.array(queries, dtype=np.float32)
    c_arr = np.array(corpus, dtype=np.float32)

    ground_truth = []
    for qi in range(len(queries)):
        diffs = c_arr - q_arr[qi]
        dists = np.sum(diffs * diffs, axis=1)
        top_k_idx = np.argpartition(dists, K)[:K]
        top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
        ground_truth.append(top_k_idx.tolist())
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(queries)}", flush=True)

    print(f"Ground truth computed in {time.time()-t0:.1f}s", flush=True)
    with open(GT_CACHE, "wb") as f:
        pickle.dump(ground_truth, f)

    return ground_truth


def recall_at_k(results, gt, k):
    return len(set(results[:k]) & set(gt[:k])) / k


def main():
    all_ids, all_embs = load_vectors()

    queries = all_embs[-NUM_QUERIES:]
    corpus = all_embs[:-NUM_QUERIES]
    corpus_ids = all_ids[:-NUM_QUERIES]

    # Ground truth uses corpus indices (0..999900), not parquet IDs
    ground_truth = compute_ground_truth(queries, corpus)

    data_dir = tempfile.mkdtemp(prefix="tantivy_1m_")
    print(f"\nIndex dir: {data_dir}", flush=True)
    print(f"Indexing {len(corpus)} vectors...", flush=True)

    idx = TantivyVectorIndex(DIM, "l2", data_dir, PARQUET_DIR,
        num_shards=1, vectors_per_shard=1_000_000, num_clusters_per_1k=1)

    COMMIT_EVERY = 100_000
    t0 = time.time()
    since_commit = 0
    for start in range(0, len(corpus), 5000):
        end = min(start + 5000, len(corpus))
        # Use sequential IDs matching corpus index so GT lines up
        idx.insert(list(range(start, end)), corpus[start:end])
        since_commit += end - start
        if since_commit >= COMMIT_EVERY:
            idx.commit()
            since_commit = 0
    if since_commit > 0:
        idx.commit()
    idx.finalize()
    t_index = time.time() - t0
    print(f"Indexed in {t_index:.1f}s ({len(corpus)/t_index:.0f} vec/s)", flush=True)

    print(f"\nSearching ({NUM_QUERIES} queries, top-{K})...", flush=True)
    configs = [(1, 1.0), (5, 2.0), (10, 3.0), (20, 5.0), (50, 10.0), (100, 100.0)]

    print(f"\n  {'config':>20s}  {'recall@10':>10s}  {'recall@100':>10s}  {'latency':>10s}  {'QPS':>8s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for max_probe, ratio in configs:
        r10s, r100s = [], []
        t0 = time.time()
        for qi in range(NUM_QUERIES):
            results = idx.search(queries[qi], k=K, max_probe=max_probe, distance_ratio=ratio)
            r10s.append(recall_at_k(results, ground_truth[qi], 10))
            r100s.append(recall_at_k(results, ground_truth[qi], K))
        elapsed = time.time() - t0
        label = f"probe({max_probe},{ratio:.0f})"
        print(f"  {label:>20s}  {sum(r10s)/len(r10s):>10.3f}  {sum(r100s)/len(r100s):>10.3f}  {elapsed/NUM_QUERIES*1000:>8.1f}ms  {NUM_QUERIES/elapsed:>8.0f}", flush=True)

    shutil.rmtree(data_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()

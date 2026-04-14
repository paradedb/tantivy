#!/usr/bin/env python3
"""GIST-1M benchmark for tantivy vector search. Apples-to-apples with rabitq-rs."""

import os
import sys
import time
import struct
import tempfile
import shutil
import pickle

import numpy as np

from tantivy_vector_bench import TantivyVectorIndex

GIST_DIR = os.path.expanduser("~/rabitq-rs-bench/data/gist/gist")
CACHE_DIR = os.path.expanduser("~/.cache/tantivy_bench")
DIM = 960
K = 100


def read_fvecs(path, limit=None):
    vectors = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            d = struct.unpack("<i", buf)[0]
            vec = struct.unpack(f"<{d}f", f.read(d * 4))
            vectors.append(list(vec))
            if limit and len(vectors) >= limit:
                break
    return vectors


def read_ivecs(path, limit=None):
    vectors = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            d = struct.unpack("<i", buf)[0]
            vec = struct.unpack(f"<{d}i", f.read(d * 4))
            vectors.append(list(vec))
            if limit and len(vectors) >= limit:
                break
    return vectors


def recall_at_k(results, gt, k):
    return len(set(results[:k]) & set(gt[:k])) / k


def main():
    cache_vecs = os.path.join(CACHE_DIR, "gist_1m_vecs.pkl")
    cache_queries = os.path.join(CACHE_DIR, "gist_1m_queries.pkl")
    cache_gt = os.path.join(CACHE_DIR, "gist_1m_gt.pkl")

    # Load data (with caching)
    if os.path.exists(cache_queries) and os.path.exists(cache_gt):
        print("Loading cached queries + ground truth...", flush=True)
        with open(cache_queries, "rb") as f:
            queries = pickle.load(f)
        with open(cache_gt, "rb") as f:
            gt = pickle.load(f)
        # Only load base vectors if we need to index
        total_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 7
        data_dir = os.path.join(CACHE_DIR, f"index_gist_{total_bits}b")
        if os.path.exists(os.path.join(data_dir, "meta.json")):
            base = []  # not needed, index is persisted
        elif os.path.exists(cache_vecs):
            print("Loading cached base vectors...", flush=True)
            with open(cache_vecs, "rb") as f:
                base = pickle.load(f)
        else:
            base = None  # will trigger full load below
    else:
        base = None

    if base is None:
        print("Loading GIST-1M from fvecs...", flush=True)
        t0 = time.time()
        base = read_fvecs(f"{GIST_DIR}/gist_base.fvecs")
        print(f"  Base: {len(base)} vectors, dim={len(base[0])}, {time.time()-t0:.1f}s", flush=True)

        queries = read_fvecs(f"{GIST_DIR}/gist_query.fvecs")
        print(f"  Queries: {len(queries)}", flush=True)

        gt = read_ivecs(f"{GIST_DIR}/gist_groundtruth.ivecs")
        print(f"  Ground truth: {len(gt)} entries", flush=True)

        print("Caching...", flush=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_vecs, "wb") as f:
            pickle.dump(base, f)
        with open(cache_queries, "wb") as f:
            pickle.dump(queries, f)
        with open(cache_gt, "wb") as f:
            pickle.dump(gt, f)

    num_vecs = len(base) if base else 0
    num_queries = len(queries)
    print(f"\nGIST-1M: {num_vecs} base vectors, {num_queries} queries, dim={DIM}", flush=True)

    # Index
    if not isinstance(total_bits, int):
        total_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    data_dir = os.path.join(CACHE_DIR, f"index_gist_{total_bits}b")
    skip_indexing = os.path.exists(os.path.join(data_dir, "meta.json"))
    print(f"total_bits={total_bits}", flush=True)
    if skip_indexing:
        print(f"  (index exists at {data_dir}, skipping indexing)", flush=True)
        idx = TantivyVectorIndex.open(DIM, "l2", data_dir, total_bits)
    else:
        print(f"\nIndexing {num_vecs} vectors...", flush=True)
        idx = TantivyVectorIndex(DIM, "l2", data_dir, data_dir,
            num_shards=1, vectors_per_shard=num_vecs, num_clusters_per_1k=4,
            total_bits=total_bits)

        COMMIT_EVERY = 100_000
        t0 = time.time()
        since_commit = 0
        for start in range(0, num_vecs, 5000):
            end = min(start + 5000, num_vecs)
            idx.insert(list(range(start, end)), base[start:end])
            since_commit += end - start
            if since_commit >= COMMIT_EVERY:
                idx.commit()
                since_commit = 0
        if since_commit > 0:
            idx.commit()
        idx.finalize()
        t_index = time.time() - t0
        print(f"Indexed in {t_index:.1f}s ({num_vecs/t_index:.0f} vec/s)", flush=True)

    # Benchmark
    print(f"\nBenchmark (top-{K}, {num_queries} queries):", flush=True)
    print(f"{'nprobe':>8} | {'QPS':>10} | {'Recall@100':>10}", flush=True)
    print("-" * 35, flush=True)

    for nprobe in [5, 10, 20, 50, 100]:
        t0 = time.time()
        total_recall = 0.0
        for qi in range(num_queries):
            results = idx.search(queries[qi], k=K, max_probe=nprobe, distance_ratio=1000.0)
            total_recall += recall_at_k(results, gt[qi], K)
        elapsed = time.time() - t0
        qps = num_queries / elapsed
        avg_recall = total_recall / num_queries
        print(f"{nprobe:>8} | {qps:>10.0f} | {avg_recall*100:>9.1f}%", flush=True)

    print(f"\nIndex at {data_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

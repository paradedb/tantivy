import json
import statistics
import sys
from pathlib import Path

import psycopg

root = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "packed-comparison"
queries = {}
for field in ["title", "text"]:
    for shape, predicate in {
        "single": f"{field} === 'database'",
        "and": f"{field} === 'postgres' AND {field} === 'database'",
        "or": f"{field} === 'postgres' OR {field} === 'database'",
    }.items():
        queries[f"{field}-{shape}"] = f"SELECT id, title, by, score FROM hn_items WHERE {predicate} ORDER BY pdb.score(id) DESC LIMIT 10"
results = {}
for name, query in queries.items():
    out = {"query": query, "runs": [], "rows": {}, "summary": {}}
    results[name] = out
    with psycopg.connect("postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable", autocommit=True) as c:
        c.execute("SET max_parallel_workers_per_gather=0")
        c.execute("SELECT public.diagnostic_subblock_pruning(true)")
        out["storage"] = c.execute("SELECT public.diagnostic_posting_norm_storage('hn_items_idx')").fetchone()[0]
        assert all(s["posting_norm_bytes"] and s["packed_posting_norm_bytes"] for s in out["storage"])
        out["index_size_bytes"] = c.execute("SELECT pg_relation_size('hn_items_idx')").fetchone()[0]
        for iteration in range(25):
            for label in (["raw", "packed"] if iteration % 2 == 0 else ["packed", "raw"]):
                c.execute("SELECT public.diagnostic_posting_norms(true)")
                c.execute("SELECT public.diagnostic_packed_posting_norms(%s)", (label == "packed",))
                plan = c.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF, FORMAT JSON) " + query).fetchone()[0]
                count = c.execute("SELECT public.diagnostic_posting_norm_reads()").fetchone()[0]
                out["runs"].append({"label": label, "iteration": iteration, "plan": plan, "local_norm_lookups": count})
        for label in ["raw", "packed"]:
            c.execute("SELECT public.diagnostic_posting_norms(true)")
            c.execute("SELECT public.diagnostic_packed_posting_norms(%s)", (label == "packed",))
            out["rows"][label] = c.execute(query.replace("id, title, by, score", "id, title, by, score, pdb.score(id) AS bm25")).fetchall()
            plan = c.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF) " + query).fetchall()
            (root / f"{tag}-{name}-{label}-plan.txt").write_text("\n".join(row[0] for row in plan) + "\n")
            values = []
            for run in out["runs"]:
                if run["label"] != label or run["iteration"] < 5:
                    continue
                plan = run["plan"][0]
                io = json.loads(plan["Plan"]["Plans"][0]["IO Breakdown"])
                values.append({
                    "execution_ms": plan["Execution Time"],
                    "shared_hits": plan["Plan"]["Shared Hit Blocks"],
                    "shared_reads": plan["Plan"]["Shared Read Blocks"],
                    "fieldnorm_hits": io["components"].get("fieldnorm", {}).get("hits", 0),
                    "posting_norm_hits": io["components"].get("pnorm", {}).get("hits", 0) + io["components"].get("bpnorm", {}).get("hits", 0),
                    "fieldnorm_reads": io["components"].get("fieldnorm", {}).get("reads", 0),
                    "posting_norm_reads": io["components"].get("pnorm", {}).get("reads", 0) + io["components"].get("bpnorm", {}).get("reads", 0),
                    "packed_stream_hits": io["components"].get("bpnorm", {}).get("hits", 0),
                    "global_norm_lookups": io["file_read_calls"].get("fieldnorm/read_byte", 0),
                    "local_norm_lookups": run["local_norm_lookups"],
                })
            out["summary"][label] = {key: statistics.median(v[key] for v in values) for key in values[0]}
        out["identical_rows_and_scores"] = out["rows"]["raw"] == out["rows"]["packed"]
        out["identical_norm_lookup_count"] = out["summary"]["raw"]["local_norm_lookups"] == out["summary"]["packed"]["local_norm_lookups"]
        (root / f"{tag}-results.json").write_text(json.dumps(results, indent=2))
        assert out["summary"]["packed"]["packed_stream_hits"] > 0
        assert out["summary"]["raw"]["packed_stream_hits"] == 0
        assert out["identical_rows_and_scores"]
        assert out["identical_norm_lookup_count"]
        assert out["summary"]["packed"]["global_norm_lookups"] == 0
        print(name, json.dumps(out["summary"]), flush=True)

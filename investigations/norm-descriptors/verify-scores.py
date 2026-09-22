import hashlib
import json
import struct
import sys
from pathlib import Path

import psycopg

root = Path(__file__).resolve().parent
matrix = json.loads((root / (sys.argv[1] if len(sys.argv) > 1 else "comparison.json")).read_text())
results = {}
with psycopg.connect("postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable", autocommit=True) as c:
    c.execute("SET max_parallel_workers_per_gather=0")
    c.execute("SET statement_timeout='120s'")
    for name, measured in matrix.items():
        predicate = measured["query"].split("WHERE ")[1].split(" ORDER BY")[0]
        query = f"WITH matched AS MATERIALIZED (SELECT id, pdb.score(id) AS bm25 FROM hn_items WHERE {predicate}) SELECT id, bm25 FROM matched ORDER BY id"
        result = {}
        results[name] = result
        for label in ["raw", "packed"]:
            c.execute("SELECT public.diagnostic_posting_norms(true)")
            c.execute("SELECT public.diagnostic_packed_posting_norms(%s)", (label == "packed",))
            rows = c.execute(query).fetchall()
            digest = hashlib.sha256()
            for doc, score in rows:
                digest.update(struct.pack("!qf", doc, score))
            result[label] = {"count": len(rows), "score_digest": digest.hexdigest()}
            by_id = dict(rows)
            top = measured["rows"][{"raw": "rewritten_raw", "packed": "embedded"}[label]]
            result[label]["matches_exhaustive_top_scores"] = sorted([r[-1] for r in top], reverse=True) == sorted(by_id.values(), reverse=True)[:10]
            result[label]["exact_selected_scores"] = all(by_id[r[0]] == r[-1] for r in top)
            assert result[label]["matches_exhaustive_top_scores"]
            assert result[label]["exact_selected_scores"]
        assert result["raw"] == result["packed"]
        c.execute("SELECT public.diagnostic_posting_norms(true)")
        unscored = measured["query"].replace(" ORDER BY pdb.score(id) DESC", "")
        result["unscored_plan"] = c.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF, FORMAT JSON) " + unscored).fetchone()[0]
        result["unscored_norm_lookups"] = c.execute("SELECT public.diagnostic_posting_norm_reads()").fetchone()[0]
        (root / "descriptor-correctness.json").write_text(json.dumps(results, indent=2))
        assert result["unscored_norm_lookups"] == 0
        c.execute("SELECT public.diagnostic_posting_norms(true)")
        assert c.execute(f"SELECT count(*) FROM hn_items WHERE {predicate}").fetchone()[0] == result["packed"]["count"]
        result["count_norm_lookups"] = c.execute("SELECT public.diagnostic_posting_norm_reads()").fetchone()[0]
        assert result["count_norm_lookups"] == 0
        print(name, result["packed"], "unscored and count: zero norm reads", flush=True)
        (root / "descriptor-correctness.json").write_text(json.dumps(results, indent=2))

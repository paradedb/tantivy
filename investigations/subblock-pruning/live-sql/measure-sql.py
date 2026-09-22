import json
import statistics
import sys
from pathlib import Path

import psycopg

root = Path(__file__).resolve().parent
query = """SELECT id, title, by, score
FROM hn_items
WHERE title === 'database'
ORDER BY pdb.score(id) DESC
LIMIT 10"""
explain = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF, FORMAT JSON) "
mode = sys.argv[1]
out = {"query": query, "mode": mode, "runs": []}
with psycopg.connect("postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable", autocommit=True) as c:
    with c.transaction(force_rollback=True):
        c.execute("SET max_parallel_workers_per_gather=0")
        c.execute("SET statement_timeout='60s'")
        out["indexes"] = c.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename='hn_items' ORDER BY indexname").fetchall()
        if not mode.startswith("original"):
            print("Building temporary title index inside a transaction that will be rolled back", flush=True)
            c.execute("SET LOCAL statement_timeout=0")
            c.execute("SET LOCAL lock_timeout='10s'")
            c.execute("DROP INDEX public.hn_items_idx")
            c.execute("CREATE INDEX hn_title_subblock_experiment_idx ON public.hn_items USING bm25 (id, title) WITH (target_segment_count=10)")
            c.execute("SET LOCAL statement_timeout='60s'")
            out["experimental_index_size_bytes"] = c.execute("SELECT pg_relation_size('hn_title_subblock_experiment_idx')").fetchone()[0]
            print("Index built; measuring exact SQL with pruning off and on", flush=True)
        labels = ["original"] if mode.startswith("original") else ["off", "on"]
        for iteration in range(25):
            for label in labels if iteration % 2 == 0 else labels[::-1]:
                if not mode.startswith("original"):
                    c.execute("SELECT public.diagnostic_subblock_pruning(%s)", (label == "on",))
                plan = c.execute(explain + query).fetchone()[0]
                out["runs"].append({"label": label, "iteration": iteration, "warmup": iteration < 5, "plan": plan})
        out["rows"] = {}
        out["scored_rows"] = {}
        for label in labels:
            if not mode.startswith("original"):
                c.execute("SELECT public.diagnostic_subblock_pruning(%s)", (label == "on",))
            out["rows"][label] = c.execute(query).fetchall()
            out["scored_rows"][label] = c.execute(query.replace("id, title, by, score", "id, title, by, score, pdb.score(id) AS bm25")).fetchall()
            text_plan = c.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF) " + query).fetchall()
            (root / f"sql-{mode}-{label}-plan.txt").write_text("\n".join(row[0] for row in text_plan) + "\n")
        out["summary"] = {}
        for label in labels:
            values = []
            for run in out["runs"]:
                if run["label"] != label or run["warmup"]:
                    continue
                plan = run["plan"][0]
                scan = plan["Plan"]["Plans"][0]
                io = json.loads(scan["IO Breakdown"])
                values.append({"execution_ms": plan["Execution Time"], "shared_hits": plan["Plan"]["Shared Hit Blocks"], "shared_reads": plan["Plan"]["Shared Read Blocks"], "fieldnorm_hits": io["components"]["fieldnorm"]["hits"], "fieldnorm_reads": io["components"]["fieldnorm"]["reads"], "fieldnorm_calls": io["file_read_calls"].get("fieldnorm/read_byte", 0)})
            out["summary"][label] = {key: {"min": min(v[key] for v in values), "median": statistics.median(v[key] for v in values), "max": max(v[key] for v in values)} for key in values[0]}
        if not mode.startswith("original"):
            out["identical_rows_and_scores"] = out["scored_rows"]["off"] == out["scored_rows"]["on"] and out["rows"]["off"] == out["rows"]["on"]
            exhaustive_query = """WITH matched AS MATERIALIZED (
    SELECT id, pdb.score(id) AS bm25 FROM hn_items WHERE title === 'database'
    ) SELECT id, bm25 FROM matched ORDER BY bm25 DESC, id"""
            exhaustive = c.execute(exhaustive_query).fetchall()
            out["exhaustive_plan"] = c.execute(explain + exhaustive_query).fetchone()[0]
            expected_scores = dict(exhaustive)
            out["exhaustive_match_count"] = len(exhaustive)
            out["exhaustive_top10_scores"] = [score for _, score in exhaustive[:10]]
            out["matches_exhaustive_scores"] = all(
                [row[-1] for row in rows] == out["exhaustive_top10_scores"]
                and all(expected_scores[row[0]] == row[-1] for row in rows)
                for rows in out["scored_rows"].values()
            )
        (root / f"sql-{mode}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out["summary"], indent=2))
        if not mode.startswith("original"):
            assert out["identical_rows_and_scores"], "Pruning changed the result rows or scores"
            assert out["matches_exhaustive_scores"], "Pruned results do not match exhaustive scoring"

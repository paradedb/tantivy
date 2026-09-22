import json
import time
from pathlib import Path

import psycopg

root = Path(__file__).resolve().parent
url = "postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable"
with psycopg.connect(url, autocommit=True) as c:
    c.execute("SET lock_timeout='10s'")
    c.execute("SET statement_timeout=0")
    c.execute("SELECT diagnostic_packed_posting_norms(false)")
    query = "SELECT id, pdb.score(id) FROM diagnostic_bitpacked_norms_smoke WHERE title === 'database' AND body === 'postgres' ORDER BY id"
    expected = c.execute(query).fetchall()
    smoke = c.execute("SELECT diagnostic_pack_posting_norms('diagnostic_bitpacked_norms_smoke_idx')").fetchone()[0]
    assert smoke
with psycopg.connect(url, autocommit=True) as c:
    c.execute("SELECT diagnostic_packed_posting_norms(true)")
    actual = c.execute(query).fetchall()
    plan = c.execute("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, TIMING OFF, FORMAT JSON) " + query).fetchone()[0]
    assert actual == expected
    assert 'bpnorm' in json.dumps(plan)
    (root / 'conversion-smoke.json').write_text(json.dumps({'storage': smoke, 'rows_match': True, 'plan': plan}, indent=2))
    print('Conversion smoke passed, 600 exact scores match', flush=True)
    c.execute("DROP TABLE diagnostic_bitpacked_norms_smoke")
    c.execute("SET lock_timeout='10s'")
    c.execute("SET statement_timeout=0")
    started = time.monotonic()
    print('Packing full existing HN norm streams', flush=True)
    result = c.execute("SELECT diagnostic_pack_posting_norms('hn_items_idx')").fetchone()[0]
    seconds = time.monotonic() - started
    (root / 'conversion.json').write_text(json.dumps({'seconds': seconds, 'segments': result}, indent=2))
    print('Finished', seconds, 'seconds; raw bytes:', sum(s['raw_bytes'] for s in result), 'packed bytes:', sum(s['packed_bytes'] for s in result), flush=True)

import json
import time
from pathlib import Path
import psycopg
from psycopg.types.json import Jsonb
root=Path(__file__).resolve().parent
url='postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable'
with psycopg.connect(url,autocommit=True) as c:
    c.execute('SET max_parallel_workers_per_gather=0')
    c.execute('CREATE TABLE diagnostic_inline_norms_smoke(id bigint PRIMARY KEY,title text,body text)')
    c.execute("INSERT INTO diagnostic_inline_norms_smoke SELECT i, 'database ' || repeat('other ',i%300), 'postgres ' || repeat('more ',i%20) FROM generate_series(1,600) i")
    c.execute("CREATE INDEX diagnostic_inline_norms_smoke_idx ON diagnostic_inline_norms_smoke USING bm25(id,title,body) WITH(key_field='id')")
    storage=c.execute("SELECT diagnostic_posting_norm_storage('diagnostic_inline_norms_smoke_idx')").fetchone()[0]
    assert all(s['posting_norm_bytes'] is None and s['packed_posting_norm_bytes'] is None for s in storage)
    query="SELECT id,pdb.score(id) FROM diagnostic_inline_norms_smoke WHERE title === 'database' AND body === 'postgres' ORDER BY id"
    c.execute('SELECT diagnostic_posting_norms(false)')
    expected=c.execute(query).fetchall()
    c.execute('SELECT diagnostic_posting_norms(true)')
    actual=c.execute(query).fetchall()
    assert expected==actual
    plan=c.execute('EXPLAIN (ANALYZE,BUFFERS,VERBOSE,TIMING OFF,FORMAT JSON) '+query).fetchone()[0]
    (root/'smoke.json').write_text(json.dumps({'storage':storage,'exact_600_scores':True,'plan':plan},indent=2))
    print('New inline index: no norm sidecars; all 600 scores match global norms',flush=True)
    c.execute('DROP TABLE diagnostic_inline_norms_smoke')

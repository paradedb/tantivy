import json
import time
from pathlib import Path
import psycopg
root = Path(__file__).resolve().parent
url = 'postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable'
with psycopg.connect(url, autocommit=True) as c:
    c.execute('SET max_parallel_workers_per_gather=0')
    c.execute('CREATE TABLE diagnostic_norm_directory_smoke(id bigint PRIMARY KEY, title text, body text)')
    c.execute("INSERT INTO diagnostic_norm_directory_smoke SELECT i, 'database ' || repeat('other ', i % 300), 'postgres ' || repeat('more ', i % 20) FROM generate_series(1,600) i")
    c.execute("CREATE INDEX diagnostic_norm_directory_smoke_idx ON diagnostic_norm_directory_smoke USING bm25(id,title,body) WITH(key_field='id')")
    query = "SELECT id,pdb.score(id) FROM diagnostic_norm_directory_smoke WHERE title === 'database' AND body === 'postgres' ORDER BY id"
    before = c.execute(query).fetchall()
    segments = c.execute("SELECT diagnostic_posting_norm_storage('diagnostic_norm_directory_smoke_idx')").fetchone()[0]
    changes = [c.execute("SELECT diagnostic_rewrite_norm_directory('diagnostic_norm_directory_smoke_idx',%s)", (s['segment'],)).fetchone()[0] for s in segments]
    for enabled in [False, True]:
        c.execute('SELECT diagnostic_embedded_norm_directory(%s)', (enabled,))
        assert c.execute(query).fetchall() == before
        plan = c.execute('EXPLAIN (ANALYZE,BUFFERS,VERBOSE,TIMING OFF,FORMAT JSON) '+query).fetchone()[0]
        (root/f'smoke-{enabled}-plan.json').write_text(json.dumps(plan,indent=2))
    c.execute("SELECT diagnostic_select_norm_components('diagnostic_norm_directory_smoke_idx',%s)", (psycopg.types.json.Jsonb([r['old'] for r in changes]),))
    assert c.execute(query).fetchall() == before
    c.execute("SELECT diagnostic_select_norm_components('diagnostic_norm_directory_smoke_idx',%s)", (psycopg.types.json.Jsonb([r['new'] for r in changes]),))
    assert c.execute(query).fetchall() == before
    (root/'smoke-conversion.json').write_text(json.dumps(changes,indent=2))
    print('Smoke conversion and component-switch round trip: all 600 scores identical',flush=True)
    c.execute('DROP TABLE diagnostic_norm_directory_smoke')
    c.execute("SET lock_timeout='10s'")
    segments = c.execute("SELECT diagnostic_posting_norm_storage('hn_items_idx')").fetchone()[0]
    (root/'before-storage.json').write_text(json.dumps(segments,indent=2))
    changes=[]
    for s in segments:
        started=time.monotonic()
        result=c.execute("SELECT diagnostic_rewrite_norm_directory('hn_items_idx',%s)",(s['segment'],)).fetchone()[0]
        result['seconds']=time.monotonic()-started
        changes.append(result)
        (root/'conversion.json').write_text(json.dumps(changes,indent=2))
        print(s['segment'],result['seconds'],'seconds',flush=True)
        c.execute('CHECKPOINT')
    (root/'after-storage.json').write_text(json.dumps(c.execute("SELECT diagnostic_posting_norm_storage('hn_items_idx')").fetchone()[0],indent=2))

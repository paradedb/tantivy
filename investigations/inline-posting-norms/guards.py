import json,statistics
from pathlib import Path
import psycopg
from psycopg.types.json import Jsonb
root=Path(__file__).resolve().parent
changes=json.loads((root/'conversion.json').read_text())
queries={}
for field in ['title','text']:
    for term in ['database','the']:
        queries[f'{field}-{term}-unscored']=f"SELECT id FROM hn_items WHERE {field} === '{term}' LIMIT 10"
        queries[f'{field}-{term}-count']=f"SELECT count(*) FROM hn_items WHERE {field} === '{term}'"
    queries[f'{field}-the-top10']=f"SELECT id,pdb.score(id) FROM hn_items WHERE {field} === 'the' ORDER BY pdb.score(id) DESC LIMIT 10"
    queries[f'{field}-the-database-top10']=f"SELECT id,pdb.score(id) FROM hn_items WHERE {field} === 'the' AND {field} === 'database' ORDER BY pdb.score(id) DESC LIMIT 10"
results={k:{'query':v,'runs':[],'rows':{},'summary':{}} for k,v in queries.items()}
with psycopg.connect('postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable',autocommit=True) as c:
    c.execute('SET max_parallel_workers_per_gather=0')
    c.execute("SET statement_timeout='120s'")
    c.execute('SELECT diagnostic_posting_norms(true)')
    c.execute('SELECT diagnostic_packed_posting_norms(false)')
    for iteration in range(12):
        for label in (['raw','inline'] if iteration%2==0 else ['inline','raw']):
            c.execute("SELECT diagnostic_select_norm_components('hn_items_idx',%s)",(Jsonb([s['old' if label=='raw' else 'new'] for s in changes]),))
            for name,out in results.items():
                c.execute('SELECT diagnostic_posting_norms(true)')
                plan=c.execute('EXPLAIN (ANALYZE,BUFFERS,VERBOSE,TIMING OFF,FORMAT JSON) '+out['query']).fetchone()[0]
                reads=c.execute('SELECT diagnostic_posting_norm_reads()').fetchone()[0]
                out['runs'].append({'label':label,'iteration':iteration,'plan':plan,'norm_lookups':reads})
                if 'top10' not in name: assert reads==0
                if iteration==11: out['rows'][label]=c.execute(out['query']).fetchall()
        print('Iteration',iteration+1,flush=True)
    c.execute("SELECT diagnostic_select_norm_components('hn_items_idx',%s)",(Jsonb([s['new'] for s in changes]),))
for name,out in results.items():
    assert out['rows']['raw']==out['rows']['inline']
    for label in ['raw','inline']:
        runs=[r for r in out['runs'] if r['label']==label and r['iteration']>=2]
        out['summary'][label]={'execution_ms':statistics.median(r['plan'][0]['Execution Time'] for r in runs),'shared_hits':statistics.median(r['plan'][0]['Plan']['Shared Hit Blocks'] for r in runs),'shared_reads':statistics.median(r['plan'][0]['Plan']['Shared Read Blocks'] for r in runs)}
    print(name,out['summary'],flush=True)
(root/'guards.json').write_text(json.dumps(results,indent=2))

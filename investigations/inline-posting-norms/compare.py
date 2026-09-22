import json
import statistics
from pathlib import Path
import psycopg
from psycopg.types.json import Jsonb
root=Path(__file__).resolve().parent
url='postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable'
changes=json.loads((root/'conversion.json').read_text())
queries={}
for field in ['title','text']:
    for shape,predicate in {'single':f"{field} === 'database'",'and':f"{field} === 'postgres' AND {field} === 'database'",'or':f"{field} === 'postgres' OR {field} === 'database'"}.items():
        queries[f'{field}-{shape}']=f'SELECT id,title,by,score FROM hn_items WHERE {predicate} ORDER BY pdb.score(id) DESC LIMIT 10'
modes={'original_raw':('old',False,False),'original_packed':('old',True,False),'inline':('new',True,True)}
results={name:{'query':query,'runs':[],'rows':{},'summary':{}} for name,query in queries.items()}
current=None
with psycopg.connect(url,autocommit=True) as c:
    c.execute('SET max_parallel_workers_per_gather=0')
    c.execute("SET lock_timeout='10s'")
    c.execute('SELECT diagnostic_subblock_pruning(true)')
    for iteration in range(35):
        labels=list(modes)
        labels=labels[iteration%len(labels):]+labels[:iteration%len(labels)]
        if iteration%2: labels.reverse()
        for label in labels:
            components,packed,embedded=modes[label]
            if current!=components:
                c.execute("SELECT diagnostic_select_norm_components('hn_items_idx',%s)",(Jsonb([s[components] for s in changes]),))
                current=components
            c.execute('SELECT diagnostic_packed_posting_norms(%s)',(packed,))
            c.execute('SELECT diagnostic_embedded_norm_directory(%s)',(embedded,))
            for name,query in queries.items():
                c.execute('SELECT diagnostic_posting_norms(true)')
                plan=c.execute('EXPLAIN (ANALYZE,BUFFERS,VERBOSE,SETTINGS,TIMING OFF,FORMAT JSON) '+query).fetchone()[0]
                count=c.execute('SELECT diagnostic_posting_norm_reads()').fetchone()[0]
                results[name]['runs'].append({'label':label,'iteration':iteration,'plan':plan,'norm_lookups':count})
                if iteration==34:
                    results[name]['rows'][label]=c.execute(query.replace('id,title,by,score','id,title,by,score,pdb.score(id)')).fetchall()
                    text=c.execute('EXPLAIN (ANALYZE,BUFFERS,VERBOSE,SETTINGS,TIMING OFF) '+query).fetchall()
                    (root/f'{name}-{label}-plan.txt').write_text('\n'.join(r[0] for r in text)+'\n')
        if iteration%5==4: print('Completed',iteration+1,'iterations',flush=True)
    c.execute("SELECT diagnostic_select_norm_components('hn_items_idx',%s)",(Jsonb([s['new'] for s in changes]),))
for name,out in results.items():
    for label in modes:
        values=[]
        for run in out['runs']:
            if run['iteration']<5 or run['label']!=label: continue
            plan=run['plan'][0]
            io=json.loads(plan['Plan']['Plans'][0]['IO Breakdown'])
            values.append({'execution_ms':plan['Execution Time'],'shared_hits':plan['Plan']['Shared Hit Blocks'],'shared_reads':plan['Plan']['Shared Read Blocks'],'fieldnorm_hits':io['components'].get('fieldnorm',{}).get('hits',0),'norm_stream_hits':sum(io['components'].get(ext,{}).get('hits',0) for ext in ['pnorm','bpnorm']),'postings_hits':io['components'].get('idx',{}).get('hits',0),'global_norm_lookups':io['file_read_calls'].get('fieldnorm/read_byte',0),'norm_lookups':run['norm_lookups']})
        out['summary'][label]={key:statistics.median(v[key] for v in values) for key in values[0]}
        assert out['summary'][label]['global_norm_lookups']==0
        assert out['rows'][label]==out['rows']['original_raw']
        if label=='inline': assert out['summary'][label]['norm_stream_hits']==0
        assert out['summary'][label]['norm_lookups']==out['summary']['original_raw']['norm_lookups']
    print(name,json.dumps(out['summary']),flush=True)
(root/'comparison.json').write_text(json.dumps(results,indent=2))
(root/'summary.json').write_text(json.dumps({name:{key:value for key,value in out.items() if key!='runs'} for name,out in results.items()},indent=2))

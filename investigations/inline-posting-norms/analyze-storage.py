import json
from pathlib import Path
root=Path(__file__).resolve().parent
before=json.loads((root/'before-storage.json').read_text())
after=json.loads((root/'after-storage.json').read_text())
changes=json.loads((root/'conversion.json').read_text())
raw=sum(s['posting_norm_bytes'] for s in before)
packed=sum(s['packed_posting_norm_bytes'] for s in before)
base=sum(sum(s['component_bytes'].values()) for s in before)-packed
inline=sum(sum(s['component_bytes'].values()) for s in after)-raw-packed
fields={}
for segment in changes:
    for field in segment['fields']:
        if not field['raw_norm_bytes']: continue
        out=fields.setdefault(field['field'],{'raw_norm_bytes':0,'raw_header_bytes':0,'inline_bytes':0})
        out['raw_norm_bytes']+=field['raw_norm_bytes']
        out['raw_header_bytes']+=18*field['terms']
        out['inline_bytes']+=field['new_postings']-field['old_postings']+18*field['terms']
for out in fields.values():
    out['old_norm_representation_bytes']=out['raw_norm_bytes']+out['raw_header_bytes']
    out['saved_bytes']=out['old_norm_representation_bytes']-out['inline_bytes']
    out['percent_saved']=100*out['saved_bytes']/out['old_norm_representation_bytes']
old_norms=sum(f['old_norm_representation_bytes'] for f in fields.values())
new_norms=sum(f['inline_bytes'] for f in fields.values())
result={'raw_sidecar_bytes':raw,'old_packed_sidecar_bytes':packed,'postings_delta_bytes':sum(s['new']['postings']['total_bytes']-s['old']['postings']['total_bytes'] for s in changes),'terms_delta_bytes':sum(s['new']['terms']['total_bytes']-s['old']['terms']['total_bytes'] for s in changes),'saved_component_bytes':base-inline,'raw_format_component_bytes':base,'inline_format_component_bytes':inline,'total_component_percent_saved':100*(base-inline)/base,'fields':fields,'old_norm_representation_bytes':old_norms,'inline_norm_representation_bytes':new_norms,'norm_representation_percent_saved':100*(old_norms-new_norms)/old_norms}
(root/'storage-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

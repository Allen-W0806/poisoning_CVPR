"""CPU bridge: SDPA forward/output adjoints/input adjoints and LoRA-B VJP."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put
from scripts.backward_boundary_capture_20261102 import decode

def main(out):
 import torch
 torch.set_num_threads(2);a=load(out/'worker0/witness.pt');b=load(out/'worker1/witness.pt');x,y=a['boundaries'],b['boundaries'];sdpa=[];bridge=[]
 assert len(x['sdpa'])==len(y['sdpa'])==6
 for rx,ry in zip(x['sdpa'],y['sdpa']):
  for field in ['q','k','v','output','grad_output','dq','dk','dv']:
   d=compare(decode(x,rx[field]),decode(y,ry[field]));sdpa.append(dict(transition=rx['transition'],call=rx['call'],boundary=field,comparison=d,layout0=rx[field]['layout'],layout1=ry[field]['layout']))
 ra={(r['parameter'],r['transition'],r['call']):r for r in x['records']};rb={(r['parameter'],r['transition'],r['call']):r for r in y['records']}
 for key in ra:
  if key[0]!=x['target_B'] or key[1]>=2:continue
  for field in ['input','output','grad_output','local_input_VJP','local_weight_VJP']:
   d=compare(decode(x,ra[key][field]),decode(y,rb[key][field]));bridge.append(dict(parameter=key[0],transition=key[1],call=key[2],boundary=field,comparison=d))
 raw=compare(a['clip']['raw'],b['clip']['raw']);summary=dict(initial_exact=compare(a['before'],b['before'])['exact'],conditioning_exact=compare(a['conditioning'],b['conditioning'])['exact'],raw_exact=raw['exact'],raw_max_abs=raw['max_abs'],SDPA_unequal=[dict(transition=r['transition'],call=r['call'],boundary=r['boundary'],max_abs=r['comparison']['max_abs'],elements=r['comparison']['affected_elements'],relative_L2=r['comparison']['relative_L2']) for r in sdpa if not r['comparison']['exact']],LoRA_B_unequal=[dict(transition=r['transition'],call=r['call'],boundary=r['boundary'],max_abs=r['comparison']['max_abs']) for r in bridge if not r['comparison']['exact']],root_cause_established=False)
 put(out/'attention_bridge_comparisons.json',dict(sdpa=sdpa,LoRA_B=bridge,raw=raw,summary=summary));put(out/'attention_bridge_summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main(ROOT/'results/numerical_fix_20261102_v1'/sys.argv[1])

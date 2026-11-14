"""CPU launch-context factor comparison, no tolerance substituted for exact."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put
def main():
 import torch
 torch.set_num_threads(2);out=ROOT/'results/numerical_fix_20261102_v1/h3_launch';modes=['baseline0','blocked0','blocked1','baseline1'];s={m:load(out/m/'update0.pt') for m in modes};result={}
 for a,b in [('baseline0','baseline1'),('blocked0','blocked1'),('baseline0','blocked0'),('baseline1','blocked1')]:
  x,y=s[a],s[b];r=dict(initial=compare(x['before'],y['before']))
  for key in ['raw','postclip','runtime_total_norm']:r[key]=compare(x['capture'][key],y['capture'][key])
  for key in ['adapter','optimizer','scheduler','rng_torch','rng_cuda','rng_numpy','rng_python']:r['after_'+key]=compare(x['after'][key],y['after'][key])
  result[a+'__'+b]=r
 supported=all(v['exact'] for v in result['blocked0__blocked1'].values()) and not result['baseline0__baseline1']['raw']['exact']
 summary={k:{n:{q:v[q] for q in ['exact','tolerance_pass','affected_elements','max_abs','relative_L2']} for n,v in row.items() if n in ['initial','raw','postclip','after_adapter']} for k,row in result.items()}
 put(out/'H3_comparisons.json',result);put(out/'H3_summary.json',dict(pairs=summary,H3_launch_blocking_supported=supported,ordinary_uninstrumented_verification=False,root_cuda_primitive_established=False));print(json.dumps(dict(pairs=summary,H3_launch_blocking_supported=supported)))
if __name__=='__main__':main()

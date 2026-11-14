"""CPU ABBA H1 comparison; exact and tolerance gates stay separate."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put

def main():
 import torch
 torch.set_num_threads(2);out=ROOT/'results/numerical_fix_20261102_v1/h1_order';modes=['baseline0','single0','single1','baseline1'];states={m:load(out/m/'update0.pt') for m in modes};pairs={}
 for a,b in [('baseline0','baseline1'),('single0','single1'),('baseline0','single0'),('baseline1','single1')]:
  x,y=states[a],states[b];pairs[a+'__'+b]=dict(initial=compare(x['before'],y['before']),raw=compare(x['capture']['raw'],y['capture']['raw']),postclip=compare(x['capture']['postclip'],y['capture']['postclip']),norm=compare(x['capture']['runtime_total_norm'],y['capture']['runtime_total_norm']),after=compare(x['after'],y['after']))
 summary={k:{n:dict(exact=r['exact'],tolerance=r['tolerance_pass'],max_abs=r['max_abs'],relative_L2=r['relative_L2'],affected_elements=r['affected_elements']) for n,r in v.items()} for k,v in pairs.items()};supported=all(pairs['single0__single1'][k]['exact'] for k in pairs['single0__single1']) and not pairs['baseline0__baseline1']['raw']['exact']
 put(out/'H1_comparisons.json',pairs);put(out/'H1_summary.json',dict(pairs=summary,H1_single_thread_supported=supported,scientific_loss_changed=False,ordinary_uninstrumented_verification=False));print(json.dumps(dict(pairs={k:{n:v[n] for n in ['raw','norm','after']} for k,v in summary.items()},H1_single_thread_supported=supported)))
if __name__=='__main__':main()

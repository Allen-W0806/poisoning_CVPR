"""CPU-only exact per-parameter analysis of the existing88433 witnesses."""
import csv,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put,sha

def main():
 import torch
 torch.set_num_threads(2);old=ROOT/'results/overnight_benign_20261102_v2/numerical_a0';out=ROOT/'results/numerical_fix_20261102_v1';out.mkdir(exist_ok=True)
 files=dict(fresh0=old/'fresh0/update0.pt',fresh1=old/'fresh1/update0.pt',resident0=old/'resident/update0.pt',resident1=old/'resident/update1.pt');data={k:load(p) for k,p in files.items()};rows=[];sets={};initial={}
 for a,b in [('fresh0','fresh1'),('resident0','resident1'),('fresh0','resident0'),('fresh0','resident1')]:
  pair=a+'__'+b;sets[pair]=[];initial[pair]=compare(data[a]['before'],data[b]['before'])['exact'];ra=data[a]['capture']['raw'];rb=data[b]['capture']['raw'];assert [n for n,g in ra]==[n for n,g in rb]
  for (name,x),(_,y) in zip(ra,rb):
   if torch.equal(x,y):continue
   d=compare(x,y);module=name.rsplit('.',1)[0];parts=name.split('.');att=next((parts[i+1] for i,p in enumerate(parts) if p=='attentions'),None);kind=next((p for p in parts if p in {'attn1','attn2'}),None);sets[pair].append(name)
   rows.append(dict(pair=pair,parameter=name,module=module,layer='down_blocks.0.attentions.'+str(att),attention=kind,shape='x'.join(map(str,x.shape)),dtype=str(x.dtype),elements=x.numel(),affected_elements=d['affected_elements'],max_abs=d['max_abs'],relative_L2=d['relative_L2'],exact=d['exact'],tolerance_pass=d['tolerance_pass']))
 all_same=all(set(v)==set(next(iter(sets.values()))) for v in sets.values());names=sorted(set(n for v in sets.values() for n in v));assert len(names)==32
 for row in rows:row['same_affected_set_across_all_four_pairs']=all_same
 with (out/'affected_gradients32.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 put(out/'CPU_existing_evidence.json',dict(source_job='88433',source_files={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in files.items()},all_starts_exact=initial,affected_sets=sets,same_affected_set=all_same,unique_affected_names=names,per_parameter=rows,production_fix='NONE',first_cuda_operation='NOT ESTABLISHED',new_GPU_seconds=0))
 report='# Existing88433 affected-gradient map\n\nAll32 affected parameters recur in all four comparisons: '+str(all_same)+'.\n\n| Exact parameter | Shape | Dtype | Fresh/fresh differing elements | Max abs | Relative L2 |\n|---|---|---|---:|---:|---:|\n'
 for r in [v for v in rows if v['pair']=='fresh0__fresh1']:report+=f"|`{r['parameter']}`|{r['shape']}|{r['dtype']}|{r['affected_elements']}|{r['max_abs']:.12g}|{r['relative_L2']:.12g}|\n"
 (out/'affected_gradients32.md').write_text(report,encoding='utf8');print(json.dumps(dict(affected=32,same_set=all_same,first=names[0],layers=sorted({r['layer'] for r in rows}),attention_kinds=sorted({r['attention'] for r in rows}),first_shapes=[r['shape'] for r in rows[:4]])))
if __name__=='__main__':main()

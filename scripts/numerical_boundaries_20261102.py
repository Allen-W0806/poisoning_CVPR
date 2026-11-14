"""CPU-only, tensor-valued provenance comparisons; never infer missing states."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):
 Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def compare(a,b):
 import torch,numpy as np
 rows=[];struct=[]
 def walk(x,y,path):
  if isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor):
   if x.shape!=y.shape or x.dtype!=y.dtype:struct.append(path+':shape/dtype');return
   v=x.detach().cpu();w=y.detach().cpu();d=v.double()-w.double();affected=int((v!=w).sum());den=float(torch.linalg.vector_norm(v.double()));err=float(torch.linalg.vector_norm(d))
   rows.append(dict(path=path,exact=torch.equal(v,w),tolerance_pass=bool((d.abs() <= 1e-7+1e-5*v.double().abs()).all()),elements=v.numel(),affected_elements=affected,max_abs=float(d.abs().max()) if v.numel() else 0.,reference_L2=den,error_L2=err,relative_L2=err/den if den else (0. if err==0 else None),zero_reference_nonzero_error=bool(not den and err)))
  elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):walk(torch.from_numpy(x),torch.from_numpy(y),path)
  elif isinstance(x,dict) and isinstance(y,dict):
   if set(x)!=set(y):struct.append(path+':keys')
   for k in x.keys()&y.keys():walk(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)) and isinstance(y,(list,tuple)):
   if len(x)!=len(y):struct.append(path+':length')
   for i,(v,w) in enumerate(zip(x,y)):walk(v,w,path+'/'+str(i))
  elif x!=y:struct.append(path+':value')
 walk(a,b,'root')
 den=math.sqrt(sum(r['reference_L2']**2 for r in rows));err=math.sqrt(sum(r['error_L2']**2 for r in rows))
 undefined=sum(r['relative_L2'] is None for r in rows)
 return dict(exact=not struct and all(r['exact'] for r in rows),tolerance_pass=not struct and all(r['tolerance_pass'] for r in rows),tensor_count=len(rows),affected_tensors=sum(not r['exact'] for r in rows),affected_elements=sum(r['affected_elements'] for r in rows),max_abs=max((r['max_abs'] for r in rows),default=0.),relative_L2=err/den if den else (0. if not err else None),undefined_tensor_relative_L2_count=undefined,max_tensor_relative_L2=None if undefined else max((r['relative_L2'] for r in rows),default=0.),structural_differences=struct,tensors=rows)
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def historical(out):
 import torch
 torch.set_num_threads(2)
 p=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';s=ROOT/'results/execution_remediation_20261102_v1_bf16_adjoints'
 result={'live_pilot':{},'saved_replay':{},'missing_live_boundaries':['internal trajectories','training forward activations/loss input tensors','raw preclip gradients','per-backward contributions','original outer scheduler object']}
 a=load(p/'ordinary_scheduler/start/training_state.pt');b=load(p/'rsi/start/training_state.pt');result['live_pilot']['initial_all']=compare(a,b)
 a=load(p/'ordinary_scheduler/candidate/training_state.pt');b=load(p/'rsi/candidate/training_state.pt')
 for k in ['adapter','optimizer','scheduler','rng_torch','rng_cuda','rng_numpy','rng_python']:result['live_pilot'][k]=compare(a[k],b[k])
 for pair,label in [((0,3),'ordinary'),((1,2),'capture')]:
  i,j=pair;mi='ordinary' if i==0 else 'capture';mj='ordinary' if j==3 else 'capture'
  da=s/f'{i}_{mi}';db=s/f'{j}_{mj}'
  row={'initial':compare(load(da/'before/training_state.pt'),load(db/'before/training_state.pt')),'final':compare(load(da/'after/training_state.pt'),load(db/'after/training_state.pt')),'gradients':compare(load(da/'gradient_witness.pt'),load(db/'gradient_witness.pt'))}
  result['saved_replay'][label]=row
 put(out/'existing_tensor_comparisons.json',result)
 # Reverify immutable candidate bytes and human-review assets; do not infer admission.
 base=ROOT/'results/execution_remediation_20261102_v1';mp=base/'prospective/candidate_content_manifest.json';manifest=json.loads(mp.read_text());rows=manifest['records'];verified=[]
 for r in rows:
  image=ROOT/r['image_path'];verified.append(dict(image_id=r['source_image_id'],bytes=image.stat().st_size,sha256=sha(image),matches=sha(image)==r['image_sha256']))
 kit=base/'semantic_review_v3_r2';hm=json.loads((kit/'blinded_manifest.json').read_text());assets=[]
 for r in hm['cases']:
  for k,v in r.items():
   if isinstance(v,str) and v.startswith('images/'):
    image=kit/v;assets.append(dict(path=v,sha256=sha(image),bytes=image.stat().st_size))
 packages=[]
 for suffix in ['A','B']:
  z=base/f'independent_review32_v3_r2_{suffix}.zip';packages.append(dict(path=str(z.relative_to(ROOT)),sha256=sha(z),bytes=z.stat().st_size))
 prior=json.loads((base/'real_review_ingestion_empty/summary.json').read_text())
 put(out/'candidate_and_review_preservation.json',dict(candidate_manifest_sha256=sha(mp),candidate_count=len(rows),unique_IDs=len({r['source_image_id'] for r in rows}),all_candidate_bytes_match=all(r['matches'] for r in verified),candidate_bytes=sum(r['bytes'] for r in verified),records=verified,human_cases=len(hm['cases']),assets=assets,packages=packages,real_ingestion_receipt=prior,new_model_calls=0,new_annotations=0,final_holdout_admission='NOT RELEASED; unresolved historical lineage and protocol/power'))
 print(json.dumps(dict(live_initial=result['live_pilot']['initial_all']['exact'],adapter_affected=result['live_pilot']['adapter']['affected_elements'],adapter_max=result['live_pilot']['adapter']['max_abs'],saved_replay={k:all(v[f]['exact'] for f in v) for k,v in result['saved_replay'].items()},candidate_bytes_verified=all(r['matches'] for r in verified),human_cases=len(hm['cases']))))
def diagnostic(out):
 import torch
 torch.set_num_threads(2);runs={k:load(out/k/'witness.pt') for k in ['A1','A2','B1','B2']};res={}
 coverage={}
 for a,b,label in [('A1','A2','ordinary_repeat'),('B1','B2','fixed_replay_repeat'),('A1','B1','ordinary_vs_capture'),('A2','B1','second_ordinary_vs_capture')]:
  res[label]={k:compare(runs[a][k],runs[b][k]) for k in runs[a].keys()&runs[b].keys()}
  coverage[label]=dict(compared=sorted(res[label]),not_comparable_in_A=sorted(runs[b].keys()-runs[a].keys()),not_comparable_in_B=sorted(runs[a].keys()-runs[b].keys()),meaning='Missing boundaries are NOT OBSERVED, never equality evidence')
  for key in ['adapter','optimizer','scheduler','rng_torch','rng_cuda','rng_numpy','rng_python']:
   res[label]['after_'+key]=compare(runs[a]['after'][key],runs[b]['after'][key])
  for key in ['exp_avg','exp_avg_sq']:
   aa={k:v[key] for k,v in runs[a]['after']['optimizer']['state'].items()};bb={k:v[key] for k,v in runs[b]['after']['optimizer']['state'].items()};res[label]['after_'+key]=compare(aa,bb)
  aa=json.loads((out/a/'summary.json').read_text())['update'];bb=json.loads((out/b/'summary.json').read_text())['update']
  res[label]['preclip_norm_scalar']=compare(torch.tensor(aa['gradient_norm'],dtype=torch.float32),torch.tensor(bb['gradient_norm'],dtype=torch.float32));res[label]['scalar_loss']=compare(torch.tensor(aa['loss'],dtype=torch.float64),torch.tensor(bb['loss'],dtype=torch.float64))
 first=[]
 for label in res:
  for k in ['before','before_update_rng','trajectories','forward','scalar_loss','raw','preclip_norm_scalar','postclip','after']:
   if k in res[label] and not res[label][k]['exact']:first.append(dict(pair=label,boundary=k));break
 exact=all(v['exact'] for row in res.values() for v in row.values())
 put(out/'new_tensor_comparisons.json',dict(pairs=res,coverage=coverage,first_observed_by_pair=first,all_saved_boundaries_exact=exact,tolerance_definition='Per-element abs <=1e-7 +1e-5*abs(reference); separately evaluated, never substitutes exact',tolerance_pass=all(v['tolerance_pass'] for row in res.values() for v in row.values())))
 print(json.dumps(dict(exact=exact,first=first,details='new_tensor_comparisons.json')))
if __name__=='__main__':
 mode=sys.argv[1];out=ROOT/'results/numerical_root_cause_20261102_v1';out.mkdir(exist_ok=True)
 (historical if mode=='historical' else diagnostic)(out)

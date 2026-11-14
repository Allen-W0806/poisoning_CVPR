"""New CPU-only spatial/support and clipping-propagation analysis of saved 88099.

No model construction, CUDA call, optimizer step, tolerance change, or GPU replay.
"""
import argparse,collections,csv,hashlib,json,math,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARENT=ROOT/'results/overnight_supervisor_20261101_v1_P1_ABBA_v1_a0'
FORENSICS=ROOT/'results/overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a1'
GUARD='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvout(p,rows):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(run):
 import torch
 torch.set_num_threads(2);started=time.perf_counter();assert os.getenv('CUDA_VISIBLE_DEVICES')=='','CPU allocation must mask CUDA'
 assert sha(ROOT/'src/rl/phase5_backend.py')==GUARD
 out=run/'bf16_spatial_clipping';out.mkdir(exist_ok=False)
 inputs={}
 for entry in read(PARENT/'artifact_inventory.json')['files']:
  p=(PARENT/entry['path']).resolve();assert p.is_relative_to(PARENT.resolve())
  digest=sha(p);assert digest==entry['sha256'],'Stored first-divergence evidence changed: '+str(p);inputs[str(p.relative_to(ROOT))]=digest
 names=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/generator_ownership.json')['parameter_order']
 previous=read(FORENSICS/'summary.json');inputs[str((FORENSICS/'summary.json').relative_to(ROOT))]=sha(FORENSICS/'summary.json')
 data=[torch.load(PARENT/n/'gradients.pt',map_location='cpu',weights_only=False) for n in ['1_instrumented','2_instrumented']]
 raw=[x['raw_gradients'][0] for x in data];post=[x['postclip_gradients'] for x in data]
 summaries=[read(PARENT/n/'summary.json') for n in ['1_instrumented','2_instrumented']]
 factors=[torch.clamp(torch.tensor(1.,dtype=torch.float32)/(torch.tensor(s['update']['gradient_norm'],dtype=torch.float32)+1e-6),max=1.) for s in summaries]
 rows=[];groups=collections.defaultdict(lambda:dict(tensors=0,raw_different=0,post_different=0,raw_error_squared=0.,max_raw_abs=0.,raw_elements=0,different_elements=0))
 residuals=[0.,0.];exact_reconstruction=[0,0];raw_equal_post_changed=[];counterfactual_common_changed=0
 for i,n in enumerate(names):
  a,b=raw[0][i],raw[1][i];pa,pb=post[0][i],post[1][i];assert all(x.device.type=='cpu' for x in [a,b,pa,pb]);assert a.shape==b.shape
  delta=b.double()-a.double();equal=torch.equal(a,b);post_equal=torch.equal(pa,pb)
  predicted=[a*factors[0],b*factors[1]]
  errs=[float((predicted[j]-post[j][i]).abs().max()) for j in [0,1]]
  for j in [0,1]:residuals[j]=max(residuals[j],errs[j]);exact_reconstruction[j]+=int(torch.equal(predicted[j],post[j][i]))
  common_changed=not torch.equal(a*factors[0],b*factors[0]);counterfactual_common_changed+=common_changed
  if equal and not post_equal:raw_equal_post_changed.append(n)
  prefix=n.split('.transformer_blocks')[0]
  g=groups[prefix];g['tensors']+=1;g['raw_different']+=not equal;g['post_different']+=not post_equal;g['raw_error_squared']+=float(delta.square().sum());g['max_raw_abs']=max(g['max_raw_abs'],float(delta.abs().max()));g['raw_elements']+=a.numel();g['different_elements']+=int((a!=b).sum())
  rows.append(dict(parameter=n,block=prefix,raw_equal=equal,post_equal=post_equal,raw_error_squared=float(delta.square().sum()),different_elements=int((a!=b).sum()),max_raw_abs=float(delta.abs().max()),post_max_abs=float((pa-pb).abs().max()),CPU_scale_replay_A_exact=torch.equal(predicted[0],pa),CPU_scale_replay_B_exact=torch.equal(predicted[1],pb),common_factor_counterfactual_different=common_changed))
 group_rows=[dict(block=k,**v) for k,v in groups.items()];total_error=sum(g['raw_error_squared'] for g in group_rows)
 for g in group_rows:g['raw_error_energy_fraction']=g['raw_error_squared']/total_error if total_error else 0.
 states=[torch.load(PARENT/n/'after/training_state.pt',map_location='cpu',weights_only=False) for n in ['1_instrumented','2_instrumented']]
 adapters=list(states[0]['adapter']);assert adapters==list(states[1]['adapter']);assert len(adapters)==len(names)==256
 param_ids=states[0]['optimizer']['param_groups'][0]['params'];assert len(param_ids)==256
 optimizer=[]
 for i,n in enumerate(names):
  pid=param_ids[i];x,y=[s['optimizer']['state'][pid] for s in states]
  optimizer.append(dict(parameter=n,raw_equal=rows[i]['raw_equal'],post_equal=rows[i]['post_equal'],adapter_equal=torch.equal(states[0]['adapter'][adapters[i]],states[1]['adapter'][adapters[i]]),exp_avg_equal=torch.equal(x['exp_avg'],y['exp_avg']),exp_avg_sq_equal=torch.equal(x['exp_avg_sq'],y['exp_avg_sq']),exp_avg_delta_max=float((x['exp_avg']-y['exp_avg']).abs().max()),exp_avg_sq_delta_max=float((x['exp_avg_sq']-y['exp_avg_sq']).abs().max())))
 bad=[x for x in rows if not x['raw_equal']]
 assert len(bad)==previous['different_raw_tensors']
 result=dict(status='PASS_NEW_CPU_ANALYSIS_ONLY',created_UTC=datetime.now(timezone.utc).isoformat(),Slurm_job_id=os.getenv('SLURM_JOB_ID'),parent_GPU_job='88099',parent_CPU_job='88105',raw_different_tensors=len(bad),total_tensors=len(names),raw_different_blocks=[x['block'] for x in group_rows if x['raw_different']],different_raw_elements=sum(x['different_elements'] for x in rows),total_parameter_elements=sum(g['raw_elements'] for g in group_rows),postclip_different_tensors=sum(not x['post_equal'] for x in rows),raw_equal_but_postclip_different_tensors=len(raw_equal_post_changed),preclip_norms=[s['update']['gradient_norm'] for s in summaries],CPU_float32_clip_factors=[float(x) for x in factors],clip_factor_delta=float(factors[1]-factors[0]),CPU_scale_reconstruction_exact_tensors=exact_reconstruction,CPU_scale_reconstruction_max_abs=residuals,CPU_common_factor_counterfactual_changed_tensors=counterfactual_common_changed,raw_error_energy_by_block=group_rows,adapter_different_tensors=sum(not x['adapter_equal'] for x in optimizer),optimizer_different_tensors={k:sum(not x[k+'_equal'] for x in optimizer) for k in ['exp_avg','exp_avg_sq']},all_gradients_finite=all(bool(torch.isfinite(x).all()) for seq in raw+post for x in seq),root_CUDA_primitive='NOT IDENTIFIED',scientific_numerical_fix=False,production_BF16_release=False,scientific_improvement='NOT ESTABLISHED',new_GPU_seconds=0,new_model_calls=0,new_optimizer_steps=0,CPU_seconds=time.perf_counter()-started,interpretation='Spatial support and postclip scaling propagation only. CPU multiplication replay is an explanatory check, not acceptance of an altered training algorithm. Registered module order is not backward temporal order; no per-transition/module-intermediate adjoints were saved.')
 csvout(out/'per_parameter_propagation.csv',rows);csvout(out/'block_error_support.csv',group_rows);csvout(out/'optimizer_propagation.csv',optimizer);put(out/'input_hashes.json',inputs);put(out/'summary.json',result)
 assert not torch.cuda.is_initialized(),'Analysis must not initialize CUDA'
 put(out/'checks.json',dict(status='PASS',scope='new CPU-only saved-evidence analysis',input_artifacts_unchanged=True,CUDA_initialized=False,root_cause_fixed=False,new_GPU_seconds=0))
 print(json.dumps({k:v for k,v in result.items() if k!='raw_error_energy_by_block'}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();main(Path(a.run))

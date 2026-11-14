"""One preregistered O/C/C/O saved-trajectory adjoint-localization experiment.

Only constant coefficient1. No VLM, new images, semantic reward, attack objective,
threshold change or repeated seed search. Instrumentation is not a fix.
"""
import argparse, copy, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def strict_pair_acceptance(pairs,raw):
 return bool(pairs) and raw['exact'] and all(p['initial_state_exact'] and p['postclip']['exact'] and p['adapter_exact'] and p['optimizer_exact'] and p['scheduler_exact'] and p['RNG_exact'] for p in pairs)

def worker(run,index,mode):
 import numpy as np,torch
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from scripts.run_ip2pix_checkpoint_diagnostic import frozen_hash,same
 from src.training.generator_adapter import adapter_state
 from src.utils.seed import seed_everything
 cfg=json.loads((run/'protocol.json').read_text());out=run/f'{index}_{mode}';out.mkdir();handles=[];original_clip=torch.nn.utils.clip_grad_norm_;clock=time.perf_counter();records=[];contributions={};raw=[];tag={'transition':-1};checks={}
 def tensor_info(x):
  y=x.detach().contiguous().cpu();v=y.float()
  return dict(sha256=hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest(),shape=list(y.shape),dtype=str(y.dtype),finite=bool(torch.isfinite(v).all()),L2=float(torch.linalg.vector_norm(v.double())))
 try:
  seed_everything(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h,n
  parent=ROOT/cfg['parent'];assert sha(parent/'before_next/training_state.pt')==cfg['checkpoint_sha256'];assert sha(ROOT/cfg['saved_forward'])==cfg['saved_forward_sha256']
  effective=json.loads((parent/'effective_config.json').read_text())['production_config']
  editor=load_pretrained_editor(ROOT,device='cuda');policy=DiffusionPolicy(editor,effective);trainer=DiffusionRLTrainer(policy);trainer.restore(parent/'before_next')
  saved=torch.load(ROOT/cfg['saved_forward'],map_location='cpu',weights_only=False);condition={k:v.cuda() for k,v in saved['conditioning'].items()};first=saved['episodes'][0];condition.update(sample_id=first['sample_id'],image_id=first['image_id'],instruction=first['instruction'])
  episodes=[(t,condition,1.) for t in saved['episodes']]
  rng=saved['before_update_rng'];torch.set_rng_state(rng['rng_torch']);torch.cuda.set_rng_state_all(rng['rng_cuda']);np.random.set_state(rng['rng_numpy']);__import__('random').setstate(rng['rng_python'])
  trainer.save(out/'before');basehash=frozen_hash(policy.pipe)
  names={n:p for n,p in policy.pipe.unet.named_parameters() if p.requires_grad};target_names=[n for n in names if n.startswith('down_blocks.0.attentions.')];assert len(names)==256 and len(target_names)==32
  if mode=='capture':
   predict=policy.predict
   def tagged_predict(*a,**kw):tag['transition']+=1;return predict(*a,**kw)
   policy.predict=tagged_predict
   for pname in target_names:
    module_name=pname.removesuffix('.weight');module=policy.pipe.unet.get_submodule(module_name);calls={'n':0}
    def forward_hook(m,args,output,name=module_name,counter=calls):
     if not output.requires_grad:return
     call=counter['n'];counter['n']+=1;transition=tag['transition'];input_info=tensor_info(args[0]);weight_info=tensor_info(m.weight)
     def adjoint_hook(g):
      records.append(dict(module=name,call=call,transition=transition,input=input_info,weight=weight_info,upstream_adjoint=tensor_info(g)));return None
     output.register_hook(adjoint_hook)
    handles.append(module.register_forward_hook(forward_hook))
    def contribution(g,name=pname):
     contributions.setdefault(name,[]).append(dict(transition=tag['transition'],gradient=g.detach().cpu().clone()));return None
    handles.append(names[pname].register_hook(contribution))
   def preclip(parameters,max_norm,*a,**kw):
    ps=list(parameters);raw.extend([p.grad.detach().cpu().clone() for p in ps]);return original_clip(ps,max_norm,*a,**kw)
   torch.nn.utils.clip_grad_norm_=preclip
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):receipt=trainer.update(episodes)
  torch.nn.utils.clip_grad_norm_=original_clip;trainer.save(out/'after')
  torch.save(dict(raw=raw,postclip=trainer.last_gradients,contributions=contributions),out/'gradient_witness.pt')
  put(out/'adjoint_boundaries.json',records)
  checks=dict(finite_loss=__import__('math').isfinite(receipt['loss']),finite_gradients=receipt['finite_gradients'],all_adapter_tensors_updated=receipt['changes']['updated_tensors']==256,frozen_base_unchanged=frozen_hash(policy.pipe)==basehash,coefficient_constant=receipt['individual_rewards']==[1.,1.],optimizer_ownership={id(p) for g in trainer.optimizer.param_groups for p in g['params']}=={id(p) for p in names.values()},captured_all_requested=mode=='ordinary' or len(contributions)==32,training_guard_unchanged=sha(ROOT/'src/rl/phase5_backend.py')==cfg['guard_sha256'])
  put(out/'summary.json',dict(mode=mode,checks=checks,update=receipt,seconds=time.perf_counter()-clock,UNet_forward_calls=policy.unet_calls,gradient_checkpointing=effective['gradient_checkpointing'],device=torch.cuda.get_device_name(),node=os.getenv('SLURMD_NODENAME'),job=os.getenv('SLURM_JOB_ID'),torch=torch.__version__,CUDA=torch.version.cuda,ordinary_path_release=False,new_images=0,new_VLM_calls=0))
  return 0 if all(checks.values()) else 1
 except Exception:(out/'failure.txt').write_text(traceback.format_exc());return 1
 finally:
  torch.nn.utils.clip_grad_norm_=original_clip
  for h in handles:h.remove()

def main(run):
 import torch
 from scripts.run_ip2pix_checkpoint_diagnostic import same,discrepancy
 cfg=json.loads((run/'protocol.json').read_text());executions=[];clock=time.perf_counter()
 for i,mode in enumerate(cfg['variants']):
  with (run/f'worker_{i}.log').open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(run),'--worker',str(i),'--mode',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  executions.append(dict(index=i,mode=mode,returncode=p.returncode))
  if p.returncode:break
 put(run/'worker_executions.json',executions)
 if len(executions)!=4 or any(x['returncode'] for x in executions):
  put(run/'checks.json',dict(status='FAIL',diagnostic_complete=False,failure='worker contract/implementation failure',numerical_retry_allowed=False,executions=executions));return 1
 entries=[]
 for i,m in enumerate(cfg['variants']):
  d=run/f'{i}_{m}';entries.append(dict(before=torch.load(d/'before/training_state.pt',map_location='cpu',weights_only=False),after=torch.load(d/'after/training_state.pt',map_location='cpu',weights_only=False),g=torch.load(d/'gradient_witness.pt',map_location='cpu',weights_only=False),boundaries=json.loads((d/'adjoint_boundaries.json').read_text()),summary=json.loads((d/'summary.json').read_text())))
 pair_rows=[]
 for a,b,label in [(0,3,'ordinary_repeat'),(1,2,'capture_repeat')]:
  x,y=entries[a],entries[b];pair_rows.append(dict(pair=label,initial_state_exact=same(x['before'],y['before']),postclip=discrepancy(x['g']['postclip'],y['g']['postclip']),adapter_exact=same(x['after']['adapter'],y['after']['adapter']),optimizer_exact=same(x['after']['optimizer'],y['after']['optimizer']),scheduler_exact=same(x['after']['scheduler'],y['after']['scheduler']),RNG_exact=all(same(x['after'][k],y['after'][k]) for k in ['rng_torch','rng_cuda','rng_numpy','rng_python'])))
 x,y=entries[1],entries[2];a={(v['module'],v['transition'],v['call']):v for v in x['boundaries']};b={(v['module'],v['transition'],v['call']):v for v in y['boundaries']};assert set(a)==set(b)
 boundary_diff=[]
 for k in sorted(a):
  fields={f:a[k][f]['sha256']==b[k][f]['sha256'] for f in ['input','weight','upstream_adjoint']}
  if not all(fields.values()):boundary_diff.append(dict(module=k[0],transition=k[1],call=k[2],equal=fields))
 contribution_diff=[]
 for name,vals in x['g']['contributions'].items():
  other=y['g']['contributions'][name];assert [v['transition'] for v in vals]==[v['transition'] for v in other]
  unequal=[v['transition'] for v,w in zip(vals,other) if not torch.equal(v['gradient'],w['gradient'])]
  if unequal:contribution_diff.append(dict(parameter=name,unequal_transitions=unequal))
 raw=discrepancy(x['g']['raw'],y['g']['raw']);exact=strict_pair_acceptance(pair_rows,raw) and all(same(entries[0]['before'],e['before']) for e in entries[1:])
 interpretation='Mismatch not reproduced in this fixed pair; paired-boundary hypothesis remains inconclusive, never a repair' if raw['exact'] else ('Observed unequal incoming adjoints before local parameter accumulation; temporal first CUDA primitive still unresolved' if any(not v['equal']['upstream_adjoint'] for v in boundary_diff) else 'No unequal captured incoming adjoints; inspect uncaptured boundary/local reduction/leaf accumulation; not a confirmed cause')
 put(run/'summary.json',dict(status='PASS_BOUNDED_DIAGNOSTIC_ONLY' if exact else 'NUMERICAL_MISMATCH_PRESERVED',pairs=pair_rows,capture_raw_error=raw,boundary_count=len(a),boundary_differences=boundary_diff,contribution_differences=contribution_diff,hypothesis_result=interpretation,scientific_release=False,numerical_fix_confirmed=False,seconds=time.perf_counter()-clock,source_checkpoint=cfg['checkpoint_sha256'],variants=cfg['variants'],new_VLM_calls=0,new_images=0))
 put(run/'checks.json',dict(status='PASS' if exact else 'FAIL',diagnostic_complete=True,numerical_retry_allowed=False,ordinary_full_graph_release=False))
 put(run/'artifact_inventory.json',dict(files=[dict(path=str(p.relative_to(run)),sha256=sha(p),bytes=p.stat().st_size) for p in run.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']))
 return 0 if exact else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--worker',type=int);p.add_argument('--mode');a=p.parse_args();raise SystemExit(worker(a.run,a.worker,a.mode) if a.worker is not None else main(a.run))

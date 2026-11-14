"""Predeclared same-allocation O/I/I/O replay, constant benign coefficient only.

Ordinary workers do not attach hooks or replace operations. Instrumented workers
use the historical preclip CPU capture. A local PASS cannot certify B10/T10.
"""
import argparse,copy,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def worker(run,index,mode):
 import numpy as np,torch
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.training.generator_adapter import adapter_state
 from src.utils.seed import seed_everything
 from scripts.run_ip2pix_checkpoint_diagnostic import same,frozen_hash
 cfg=json.loads((run/'protocol.json').read_text());out=run/f'{index}_{mode}';out.mkdir();started=time.perf_counter();clip=torch.nn.utils.clip_grad_norm_;raw=[]
 try:
  torch.set_num_threads(4);seed_everything(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  parent=ROOT/cfg['parent_checkpoint_directory'];effective=json.loads((parent/'effective_config.json').read_text())['production_config']
  assert {n:sha(ROOT/n) for n in cfg['protected_source']}==cfg['protected_source']
  assert sha(parent/'before_next/training_state.pt')==cfg['checkpoint_sha256']
  assert sha(parent/'reference_rollout.pt')==cfg['rollout_sha256']
  record=json.loads((ROOT/'results/replication_monitor_20261029_v1_ip2pix_a0/protocol.json').read_text())['source_record'];assert sha(ROOT/record['image_path'])==record['image_sha256']
  assert torch.cuda.device_count()==1 and os.getenv('SLURM_JOB_ID')
  editor=load_pretrained_editor(ROOT,device='cuda');policy=DiffusionPolicy(editor,effective);trainer=DiffusionRLTrainer(policy);trainer.restore(parent/'before_next');frozen=frozen_hash(policy.pipe)
  trainer.save(out/'restored_before')
  with Image.open(ROOT/record['image_path']) as im:im=im.convert('RGB')
  saved=torch.load(parent/'reference_rollout.pt',map_location='cpu',weights_only=False)
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   condition=policy.condition(dict(image=im,sample_id=record['sample_id'],image_id=record['image_id']));conditions={k:v.cpu() for k,v in condition.items() if isinstance(v,torch.Tensor)};episodes=[];pixels=[];seeds=[]
   for j in range(2):
    seed=int(torch.randint(0,2**30,(1,)).item());img,tr=policy.sample(condition,seed,f'next_{j}');episodes.append((tr,condition,1.));pixels.append(np.array(img));seeds.append(seed)
   before_update=dict(rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=__import__('random').getstate())
   torch.save(dict(episodes=[{k:v for k,v in t.items() if k!='generation_seconds'} for t,c,r in episodes],pixels=pixels,seeds=seeds,conditioning=conditions,before_update_rng=before_update),out/'forward.pt')
   if mode=='instrumented':
    def capture(parameters,max_norm,*args,**kwargs):
     ps=list(parameters);raw.append([p.grad.detach().cpu().clone() if p.grad is not None else None for p in ps]);return clip(ps,max_norm,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=capture
   receipt=trainer.update(episodes)
  torch.nn.utils.clip_grad_norm_=clip
  trainer.save(out/'after');torch.save(dict(postclip_gradients=trainer.last_gradients,raw_gradients=raw),out/'gradients.pt')
  forward_equal=all(same(a['steps'],b['steps']) for (a,c,r),b in zip(episodes,saved['episodes'])) and same(pixels,saved['pixels']) and seeds==saved['seeds'] and same(conditions,saved['conditioning'])
  checks=dict(checkpoint_contract_restored=True,forward_equals_preserved_parent=forward_equal,frozen_weights_invariant=frozen_hash(policy.pipe)==frozen,finite_gradients=receipt['finite_gradients'],parameters_updated=receipt['changes']['updated_tensors']==256,reward_is_constant_and_non_adversarial=receipt['individual_rewards']==[1.,1.])
  identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip()
  metadata=dict(node=os.getenv('SLURMD_NODENAME'),job=os.getenv('SLURM_JOB_ID'),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),device_uuid_driver=identity,capability=list(torch.cuda.get_device_capability()),deterministic=torch.are_deterministic_algorithms_enabled(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,CUBLAS_WORKSPACE_CONFIG=os.getenv('CUBLAS_WORKSPACE_CONFIG'),parameter_dtypes=sorted({str(p.dtype) for p in policy.pipe.unet.parameters()}),source_sha256=sha(Path(__file__)),instrumentation='historical preclip CPU capture' if mode=='instrumented' else 'no added hook or operation replacement')
  put(out/'summary.json',dict(status='PASS_WORKER_CONTRACTS_ONLY' if all(checks.values()) else 'FAIL',checks=checks,mode=mode,metadata=metadata,update=receipt,seconds=time.perf_counter()-started,full_historical_B10_T10='NOT VALIDATED'))
  return 0 if all(checks.values()) else 1
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());return 1
 finally:torch.nn.utils.clip_grad_norm_=clip

def run_parent(run):
 import torch
 from scripts.run_ip2pix_checkpoint_diagnostic import same,discrepancy
 started=time.perf_counter();cfg=json.loads((run/'protocol.json').read_text());executions=[]
 for i,mode in enumerate(cfg['order']):
  t=time.perf_counter()
  with (run/f'worker_{i}.log').open('x') as f:p=subprocess.run([sys.executable,'-B',str(Path(__file__)),str(run),'--worker',str(i),'--mode',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  executions.append(dict(index=i,mode=mode,returncode=p.returncode,seconds=time.perf_counter()-t))
 put(run/'worker_executions.json',executions)
 if any(x['returncode'] for x in executions):
  put(run/'checks.json',dict(status='FAIL_WORKER_CONTRACT',executions=executions,numerical_retry_allowed=False));return 1
 entries=[]
 for i,m in enumerate(cfg['order']):
  p=run/f'{i}_{m}';entries.append(dict(summary=json.loads((p/'summary.json').read_text()),before=torch.load(p/'restored_before/training_state.pt',map_location='cpu',weights_only=False),forward=torch.load(p/'forward.pt',map_location='cpu',weights_only=False),after=torch.load(p/'after/training_state.pt',map_location='cpu',weights_only=False),grads=torch.load(p/'gradients.pt',map_location='cpu',weights_only=False)['postclip_gradients']))
 pairs=[]
 for a,b,label in [(0,3,'ordinary_repeat'),(1,2,'instrumented_repeat'),(0,1,'ordinary_instrumented_first'),(3,2,'ordinary_instrumented_last')]:
  x,y=entries[a],entries[b];fields=dict(initial_checkpoint=same(x['before'],y['before']),forward_and_preupdate_rng=same(x['forward'],y['forward']),postclip_backward=same(x['grads'],y['grads']),adapter=same(x['after']['adapter'],y['after']['adapter']),optimizer=same(x['after']['optimizer'],y['after']['optimizer']),scheduler=same(x['after']['scheduler'],y['after']['scheduler']),postupdate_rng=all(same(x['after'][k],y['after'][k]) for k in ['rng_torch','rng_cuda','rng_numpy','rng_python']))
  pairs.append(dict(pair=label,indices=[a,b],comparisons=fields,earliest_observed_mismatch=next((k for k,v in fields.items() if not v),None),gradient_error=discrepancy(x['grads'],y['grads'])))
 md=[e['summary']['metadata'] for e in entries];hardware_equal=len({(x['node'],x['device_uuid_driver']) for x in md})==1;complete=hardware_equal and all(all(p['comparisons'].values()) for p in pairs)
 put(run/'summary.json',dict(status='PASS_BOUNDED_PAIR_ONLY' if complete else 'NUMERICAL_MISMATCH_PRESERVED',pairs=pairs,hardware_equal=hardware_equal,metadata=md,seconds=time.perf_counter()-started,ordinary_reproducible_in_this_pair=all(pairs[0]['comparisons'].values()),causal_root_cause='NOT CONFIRMED; postclip observations cannot identify an unobserved temporal primitive',instrumentation_deviation='Instrumented mode copies raw gradients to CPU before clipping; ordinary mode leaves core unmodified. All modes save forward/state outputs at existing phase boundaries.',historical_scope='B2/T2 constant-coefficient checkpoint diagnostic; not B10/T10 scientific learning; one matched-hardware batch, no repeat-until-PASS',scientific_improvement='NOT ESTABLISHED',reward_release=False))
 put(run/'artifact_inventory.json',dict(files=[dict(path=p.relative_to(run).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(run.rglob('*')) if p.is_file()]))
 put(run/'checks.json',dict(status='PASS' if complete else 'FAIL',diagnostic_complete=True,ordinary_full_graph_scientific_release=False,numerical_retry_allowed=False))
 return 0 if complete else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--worker',type=int);p.add_argument('--mode',choices=['ordinary','instrumented']);a=p.parse_args();run=Path(a.run);raise SystemExit(worker(run,a.worker,a.mode) if a.worker is not None else run_parent(run))

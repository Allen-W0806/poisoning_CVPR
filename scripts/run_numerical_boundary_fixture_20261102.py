"""One bounded constant-coefficient experiment. No production math changes.

A: uninstrumented canonical ordinary update, twice. C: their live collections,
saved separately. B: two instrumented replays of C1 from the same checkpoint.
All B10/T2, one device/process, original sequential CFG/MATH/checkpointing config.
"""
import copy,hashlib,json,os,signal,sys,time,traceback,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def main(out):
 started=time.perf_counter()
 def deadline(*_):raise TimeoutError('165-second process deadline; no retry')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(165)
 import numpy as np,torch
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.training.generator_adapter import adapter_state
 from scripts.run_ip2pix_checkpoint_diagnostic import frozen_hash,same
 from src.utils.seed import seed_everything
 import src.rl.diffusion_rl_trainer as tm
 cfg=json.loads((out/'protocol.json').read_text());old=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';pc=json.loads((old/'protocol.json').read_text());rows=json.loads((old/'source_manifest.json').read_text())['train'][10:20]
 assert torch.__version__.startswith('2.6.0') and torch.cuda.device_count()==1
 for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h,n
 seed_everything(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 editor=load_pretrained_editor(ROOT,device='cuda');p=DiffusionPolicy(editor,pc['diffusion_config']);t=DiffusionRLTrainer(p);p.contract['pilot_task']=pc['engineering_proxy'];p.contract['source_manifest_sha256']=sha(old/'source_manifest.json')
 names={n:v for n,v in p.pipe.unet.named_parameters() if v.requires_grad};ownership={id(v) for v in names.values()}=={id(v) for g in t.optimizer.param_groups for v in g['params']};assert ownership and len(names)==256
 conditions=[]
 for r in rows:
  assert sha(ROOT/r['image_path'])==r['image_sha256']
  with Image.open(ROOT/r['image_path']) as im:conditions.append(p.condition({**r,'image':im.convert('RGB')}))
 basehash=frozen_hash(p.pipe);original_predict=p.predict;original_transition=tm.transition;original_clip=torch.nn.utils.clip_grad_norm_;live=None;live_rng=None;summaries=[]
 def rng():return dict(rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate())
 def state():return dict(adapter=adapter_state(p.pipe.unet),optimizer=copy.deepcopy(t.optimizer.state_dict()),scheduler=copy.deepcopy(t.scheduler.state_dict()),step=t.step,version=p.version,consumed=copy.deepcopy(t.consumed),**rng())
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
  return x
 def setrng(s):
  torch.set_rng_state(s['rng_torch']);torch.cuda.set_rng_state_all(s['rng_cuda']);np.random.set_state(s['rng_numpy']);random.setstate(s['rng_python'])
 try:
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   for mode in ['A1','A2','B1','B2']:
    folder=out/mode;folder.mkdir();t.restore(old/'shared_prefix');p.config['learning_rate']=5e-6;t.optimizer.param_groups[0]['lr']=5e-6;t.scheduler.base_lrs=[5e-6]
    before=cpu(state());raw=[];forward=[];adjoints=[];contributions={};hooks=[];tag={'transition':-1}
    if mode.startswith('A'):
     episodes=[]
     for i,c in enumerate(conditions):
      _,tr=p.sample(c,4210000+i,f'constant_fixture_{i}');episodes.append((tr,c,1.))
     trajectories=[{k:v for k,v in tr.items() if k!='generation_seconds'} for tr,c,r in episodes];update_rng=cpu(rng())
     if mode=='A1':live=episodes;live_rng=update_rng
     torch.save(dict(trajectories=trajectories,before_update_rng=update_rng),out/('C1_live.pt' if mode=='A1' else 'C2_live.pt'))
    else:
     episodes=live;trajectories=[{k:v for k,v in tr.items() if k!='generation_seconds'} for tr,c,r in episodes];setrng(live_rng);update_rng=cpu(rng())
     def predict(*a,**kw):
      tag['transition']+=1;value=original_predict(*a,**kw);forward.append(dict(boundary='guided_prediction',transition=tag['transition'],state=cpu(a[0]),t=int(a[1]),epsilon=cpu(value)));return value
     def transition(*a,**kw):
      value=original_transition(*a,**kw);forward.append(dict(boundary='loss_inputs',transition=tag['transition'],action=cpu(value[0]),logp=cpu(value[1]),mean=cpu(value[2]),std=cpu(value[3])));return value
     def preclip(parameters,max_norm,*a,**kw):
      ps=list(parameters);raw.extend(cpu(v.grad) for v in ps);return original_clip(ps,max_norm,*a,**kw)
     p.predict=predict;tm.transition=transition;torch.nn.utils.clip_grad_norm_=preclip
     for name,param in names.items():
      if not name.startswith('down_blocks.0.attentions.'):continue
      def contribution(g,n=name):
       contributions.setdefault(n,[]).append(dict(transition=tag['transition'],gradient=cpu(g)))
      hooks.append(param.register_hook(contribution))
      module=p.pipe.unet.get_submodule(name.removesuffix('.weight'))
      def hook(m,args,value,n=name):
       if tag['transition']!=0:return
       trans=tag['transition'];index=len(adjoints);entry=dict(parameter=n,transition=trans,input=cpu(args[0]),output=cpu(value));adjoints.append(entry)
       value.register_hook(lambda g,e=entry:e.update(upstream_adjoint=cpu(g)))
      hooks.append(module.register_forward_hook(hook))
    receipt=t.update(episodes)
    p.predict=original_predict;tm.transition=original_transition;torch.nn.utils.clip_grad_norm_=original_clip
    for h in hooks:h.remove()
    after=cpu(state());witness=dict(before=before,before_update_rng=update_rng,trajectories=trajectories,postclip=cpu(t.last_gradients),after=after)
    if mode.startswith('B'):witness.update(raw=raw,forward=forward,first_transition_downblock=adjoints,contributions=contributions)
    torch.save(witness,folder/'witness.pt');t.save(folder/'after');t.restore(folder/'after');resume_exact=same(cpu(state()),after)
    summary=dict(mode=mode,update=receipt,resume_exact=resume_exact,seconds=time.perf_counter()-started);put(folder/'summary.json',summary);summaries.append(summary)
    print(json.dumps(dict(mode=mode,loss=receipt['loss'],norm=receipt['gradient_norm'],elapsed=summary['seconds'],resume_exact=resume_exact)),flush=True)
   invariants=dict(frozen_base_unchanged=frozen_hash(p.pipe)==basehash,ownership_exact=ownership,all_resume_exact=all(s['resume_exact'] for s in summaries),training_guard_unchanged=sha(ROOT/'src/rl/phase5_backend.py')==cfg['guard_sha256'],coefficient_constant=all(s['update']['individual_rewards']==[1.]*10 for s in summaries))
   put(out/'execution_summary.json',dict(summaries=summaries,invariants=invariants,seconds=time.perf_counter()-started,torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),job=os.getenv('SLURM_JOB_ID'),NFE=p.unet_calls,settings_changed=[],instrumentation='B only: forward/adjoint/raw copies; A has no added hooks',original_scientific_loss_unchanged=True))
   put(out/'checks.json',dict(status='PASS' if all(invariants.values()) else 'FAIL',diagnostic_complete=True,numerical_retry_allowed=False,scope='Execution invariants only; CPU exact tensor comparisons pending; not BF16 release'))
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());put(out/'checks.json',dict(status='FAIL',diagnostic_complete=False,numerical_retry_allowed=False));raise
 finally:
  signal.alarm(0);p.predict=original_predict;tm.transition=original_transition;torch.nn.utils.clip_grad_norm_=original_clip
if __name__=='__main__':main(Path(sys.argv[1]))

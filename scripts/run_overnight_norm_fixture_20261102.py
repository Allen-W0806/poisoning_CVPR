"""One predeclared process-residency experiment, minimal gradient capture."""
import argparse,copy,hashlib,inspect,json,os,subprocess,sys,time,traceback,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def worker(out,mode):
 import numpy as np,torch
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.training.generator_adapter import adapter_state
 from src.utils.seed import seed_everything
 from scripts.run_ip2pix_checkpoint_diagnostic import frozen_hash,same
 from scripts.gradient_boundary_capture_20261102 import CaptureClip,norms
 cfg=json.loads((out/'protocol.json').read_text());pilot=ROOT/cfg['pilot'];pc=json.loads((pilot/'protocol.json').read_text());saved=torch.load(ROOT/cfg['trajectory_file'],map_location='cpu',weights_only=False);rows=json.loads((pilot/'source_manifest.json').read_text())['train'][10:20]
 seed_everything(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 assert torch.__version__.startswith('2.6.0') and torch.cuda.device_count()==1
 for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h,n
 assert sha(pilot/'shared_prefix/training_state.pt')==cfg['checkpoint_sha256'];assert sha(ROOT/cfg['trajectory_file'])==cfg['trajectory_sha256']
 editor=load_pretrained_editor(ROOT,device='cuda');p=DiffusionPolicy(editor,pc['diffusion_config']);t=DiffusionRLTrainer(p);p.contract['pilot_task']=pc['engineering_proxy'];p.contract['source_manifest_sha256']=sha(pilot/'source_manifest.json');named=[(n,v) for n,v in p.pipe.unet.named_parameters() if v.requires_grad]
 conditions=[]
 for r in rows:
  assert sha(ROOT/r['image_path'])==r['image_sha256']
  with Image.open(ROOT/r['image_path']) as im:conditions.append(p.condition({**r,'image':im.convert('RGB')}))
 assert {id(v) for n,v in named}=={id(v) for g in t.optimizer.param_groups for v in g['params']} and len(named)==256
 frozen=frozen_hash(p.pipe);original=torch.nn.utils.clip_grad_norm_;folder=out/mode;folder.mkdir();torch.save(dict(conditioning=[{k:v.detach().cpu() for k,v in c.items() if isinstance(v,torch.Tensor)} for c in conditions],names=[n for n,v in named]),folder/'conditioning.pt')
 (folder/'torch_clip_grad_norm_source.py').write_text(inspect.getsource(original),encoding='utf8')
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
  return copy.deepcopy(x)
 def state():return cpu(dict(adapter=adapter_state(p.pipe.unet),optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),step=t.step,version=p.version,consumed=t.consumed,rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate()))
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
  for i in range(2 if mode=='resident' else 1):
   t.restore(pilot/'shared_prefix');p.config['learning_rate']=5e-6;t.optimizer.param_groups[0]['lr']=5e-6;t.scheduler.base_lrs=[5e-6]
   rng=saved['before_update_rng'];torch.set_rng_state(rng['rng_torch']);torch.cuda.set_rng_state_all(rng['rng_cuda']);np.random.set_state(rng['rng_numpy']);random.setstate(rng['rng_python'])
   before=state();cap=CaptureClip(named,original);torch.nn.utils.clip_grad_norm_=cap
   try:receipt=t.update([(tr,c,1.) for tr,c in zip(saved['trajectories'],conditions)])
   finally:torch.nn.utils.clip_grad_norm_=original
   assert len(cap.records)==1;record=cap.records[0];reference=norms(record['raw']);after=state();torch.save(dict(before=before,after=after,capture=record),folder/f'update{i}.pt')
   put(folder/f'update{i}.json',dict(update=receipt,FP64_reference=reference,runtime_total_norm=float(record['runtime_total_norm']),coefficient=float(record['runtime_coefficient_formula']),torch=torch.__version__,CUDA=torch.version.cuda,precision=pc['diffusion_config'],constant_coefficient=1.,checkpoint_roundtrip_only=False,ordinary_uninstrumented=False,scope='Canonical update with one minimal clipping wrapper; raw copied only after all backward calls',ownership_exact=True))
   print(json.dumps(dict(mode=mode,iteration=i,norm=float(record['runtime_total_norm']),FP64_norm=reference['total_L2_FP64'])),flush=True)
 assert frozen_hash(p.pipe)==frozen
 put(folder/'checks.json',dict(status='PASS',frozen_base_unchanged=True,ownership_exact=True,guard_unchanged=sha(ROOT/'src/rl/phase5_backend.py')==cfg['guard_sha256']))

def main(out):
 start=time.perf_counter();codes=[]
 for mode in ['fresh0','fresh1','resident']:
  with (out/(mode+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  codes.append(dict(mode=mode,returncode=r.returncode))
  if r.returncode:break
 put(out/'executions.json',dict(workers=codes,seconds=time.perf_counter()-start,scientific_reward_evaluated=False,fix_confirmed=False))
 if len(codes)!=3 or any(x['returncode'] for x in codes):raise RuntimeError('Bounded worker failed; preserve logs, never repeat numerical failure')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('out',type=Path);a.add_argument('--worker');v=a.parse_args()
 try:worker(v.out,v.worker) if v.worker else main(v.out)
 except Exception:
  (v.out/('failure_'+(v.worker or 'parent')+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise

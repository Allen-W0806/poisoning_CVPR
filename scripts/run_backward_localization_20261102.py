"""Matched two-fresh-process backward localization, constant coefficient1."""
import argparse,copy,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def worker(out,index):
 import torch,numpy as np
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.training.generator_adapter import adapter_state
 from src.utils.seed import seed_everything
 from scripts.run_ip2pix_checkpoint_diagnostic import frozen_hash
 from scripts.gradient_boundary_capture_20261102 import CaptureClip
 from scripts.backward_boundary_capture_20261102 import BoundaryCapture
 cfg=json.loads((out/'protocol.json').read_text());parent=ROOT/cfg['pilot'];pc=json.loads((parent/'protocol.json').read_text());saved=torch.load(ROOT/cfg['trajectory_file'],map_location='cpu',weights_only=False);rows=json.loads((parent/'source_manifest.json').read_text())['train'][10:20]
 for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h,n
 assert torch.__version__.startswith('2.6.0') and torch.version.cuda=='12.4';seed_everything(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 p=DiffusionPolicy(load_pretrained_editor(ROOT,device='cuda'),pc['diffusion_config']);t=DiffusionRLTrainer(p);p.contract['pilot_task']=pc['engineering_proxy'];p.contract['source_manifest_sha256']=sha(parent/'source_manifest.json');conditions=[]
 for r in rows:
  assert sha(ROOT/r['image_path'])==r['image_sha256']
  with Image.open(ROOT/r['image_path']) as im:conditions.append(p.condition({**r,'image':im.convert('RGB')}))
 t.restore(parent/'shared_prefix');p.config['learning_rate']=5e-6;t.optimizer.param_groups[0]['lr']=5e-6;t.scheduler.base_lrs=[5e-6];rng=saved['before_update_rng'];torch.set_rng_state(rng['rng_torch']);torch.cuda.set_rng_state_all(rng['rng_cuda']);np.random.set_state(rng['rng_numpy']);random.setstate(rng['rng_python'])
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
  return copy.deepcopy(x)
 def state():return cpu(dict(adapter=adapter_state(p.pipe.unet),optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),step=t.step,version=p.version,consumed=t.consumed,rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate()))
 folder=out/f'worker{index}';folder.mkdir();before=state();base=frozen_hash(p.pipe);named=[(n,v) for n,v in p.pipe.unet.named_parameters() if v.requires_grad];assert len(named)==256 and {id(v) for n,v in named}=={id(v) for g in t.optimizer.param_groups for v in g['params']}
 phase={'transition':-1};capture=BoundaryCapture(p.pipe.unet,cfg['target_parameter'],phase);original_predict=p.predict;original_backward=torch.Tensor.backward;original_clip=torch.nn.utils.clip_grad_norm_;clip=CaptureClip(named,original_clip)
 def predict(*a,**kw):phase['transition']+=1;return original_predict(*a,**kw)
 def backward(tensor,*a,**kw):
  result=original_backward(tensor,*a,**kw);capture.after_backward();return result
 p.predict=predict;torch.Tensor.backward=backward;torch.nn.utils.clip_grad_norm_=clip
 try:
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.autograd.set_multithreading_enabled(cfg['autograd_multithreading']):receipt=t.update([(tr,c,1.) for tr,c in zip(saved['trajectories'],conditions)])
 finally:
  p.predict=original_predict;torch.Tensor.backward=original_backward;torch.nn.utils.clip_grad_norm_=original_clip;capture.close()
 boundary=capture.export();put(folder/'capture_size.json',dict(storage_bytes=boundary['storage_bytes'],cap_bytes=260000000,full_forward_transitions=[0,1],upstream_and_local_VJP_transitions=list(range(20))));assert boundary['storage_bytes']<260_000_000,'Storage bound exceeded; do not expand dump';assert len(boundary['accumulated'])==20;assert len(boundary['records'])==32*3*20
 torch.save(dict(before=before,after=state(),clip=clip.records[0],boundaries=boundary,conditioning=[{k:v.detach().cpu() for k,v in c.items() if isinstance(v,torch.Tensor)} for c in conditions]),folder/'witness.pt')
 put(folder/'summary.json',dict(update=receipt,frozen_base_unchanged=frozen_hash(p.pipe)==base,ownership_exact=True,torch=torch.__version__,CUDA=torch.version.cuda,UNet_NFE=p.unet_calls,autograd_multithreading=cfg['autograd_multithreading'],capture_bytes=boundary['storage_bytes'],ordinary_uninstrumented=False,coefficient_constant1=True,guard_unchanged=sha(ROOT/'src/rl/phase5_backend.py')==cfg['guard_sha256']))
 print(json.dumps(dict(worker=index,loss=receipt['loss'],norm=receipt['gradient_norm'],capture_bytes=boundary['storage_bytes'])),flush=True)

def main(out):
 started=time.time()
 for index in [0,1]:
  with (out/f'worker{index}.log').open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',str(index)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  if p.returncode:raise RuntimeError('Worker failed; no automatic numerical retry')
 put(out/'execution.json',dict(status='EXECUTED_COMPARISON_PENDING_CPU',seconds=time.time()-started,job_id=os.getenv('SLURM_JOB_ID'),fix_confirmed=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--worker',type=int);a=p.parse_args()
 try:worker(a.out,a.worker) if a.worker is not None else main(a.out)
 except Exception:
  (a.out/('failure_'+str(a.worker)+'.txt')).write_text(traceback.format_exc());raise

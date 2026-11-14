"""Conditional ordinary uninstrumented repeat and resumed next-update check."""
import argparse,copy,json,os,random,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def worker(out,mode):
 import numpy as np,torch
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.utils.seed import seed_everything
 from scripts.run_ip2pix_checkpoint_diagnostic import frozen_hash
 cfg=json.loads((out/'protocol.json').read_text());pilot=ROOT/cfg['pilot'];pc=json.loads((pilot/'protocol.json').read_text());saved=torch.load(ROOT/cfg['trajectory_file'],map_location='cpu',weights_only=False);rows=json.loads((pilot/'source_manifest.json').read_text())['train'][10:20]
 assert torch.__version__.startswith('2.6.0')
 seed_everything(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h
 editor=load_pretrained_editor(ROOT,device='cuda');p=DiffusionPolicy(editor,pc['diffusion_config']);t=DiffusionRLTrainer(p);p.contract['pilot_task']=pc['engineering_proxy'];p.contract['source_manifest_sha256']=sha(pilot/'source_manifest.json');conditions=[]
 for r in rows:
  assert sha(ROOT/r['image_path'])==r['image_sha256']
  with Image.open(ROOT/r['image_path']) as im:conditions.append(p.condition({**r,'image':im.convert('RGB')}))
 frozen=frozen_hash(p.pipe);ownership={id(v) for v in p.pipe.unet.parameters() if v.requires_grad}=={id(v) for g in t.optimizer.param_groups for v in g['params']};assert ownership
 folder=out/mode;folder.mkdir();updates=[]
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
  if mode=='resume':t.restore(out/'ordinary0/after1');p.config['learning_rate']=5e-6
  else:
   t.restore(pilot/'shared_prefix');p.config['learning_rate']=5e-6;t.optimizer.param_groups[0]['lr']=5e-6;t.scheduler.base_lrs=[5e-6]
   rng=saved['before_update_rng'];torch.set_rng_state(rng['rng_torch']);torch.cuda.set_rng_state_all(rng['rng_cuda']);np.random.set_state(rng['rng_numpy']);random.setstate(rng['rng_python']);t.save(folder/'before')
   updates.append(t.update([(tr,c,1.) for tr,c in zip(saved['trajectories'],conditions)]));t.save(folder/'after1')
  episodes=[]
  for i,c in enumerate(conditions):
   _,tr=p.sample(c,4220000+i,'next_update_'+str(i));episodes.append((tr,c,1.))
  torch.save(dict(trajectories=[{k:v for k,v in tr.items() if k!='generation_seconds'} for tr,c,r in episodes],rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/'next_trajectory.pt')
  updates.append(t.update(episodes));t.save(folder/'after2');torch.save(t.last_gradients,folder/'postclip.pt')
  assert frozen_hash(p.pipe)==frozen
  put(folder/'receipt.json',dict(updates=updates,ownership_exact=ownership,frozen_base_unchanged=True,unmodified_canonical_update=True,clip_wrapper=False,original_guard_unchanged=sha(ROOT/'src/rl/phase5_backend.py')==cfg['guard_sha256'],UNet_NFE=p.unet_calls))
def main(out):
 for mode in ['ordinary0','ordinary1','resume']:
  with (out/(mode+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  if p.returncode:raise RuntimeError(mode+' failed; preserve original failure')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--worker');a=p.parse_args()
 try:worker(a.out,a.worker) if a.worker else main(a.out)
 except Exception:
  (a.out/('failure_'+(a.worker or 'parent')+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise

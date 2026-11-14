"""H1 single-factor ABBA: default vs single-thread autograd, no layer hooks."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put

def worker(out,mode,enabled):
 import torch
 from scripts.run_overnight_norm_fixture_20261102 import worker as ordinary_update
 from src.rl.diffusion_policy import DiffusionPolicy
 original=DiffusionPolicy.__init__;policies=[]
 def init(self,*a,**kw):original(self,*a,**kw);policies.append(self)
 DiffusionPolicy.__init__=init
 try:
  with torch.autograd.set_multithreading_enabled(enabled):
   assert torch.autograd.is_multithreading_enabled()==enabled;ordinary_update(out,mode)
  put(out/mode/'H1_runtime.json',dict(autograd_multithreading_enabled=enabled,UNet_NFE=policies[-1].unet_calls,layer_or_parameter_hooks=False,clip_capture_only=True,only_factor_changed='autograd multithreading',Torch=torch.__version__,CUDA=torch.version.cuda))
 finally:DiffusionPolicy.__init__=original
def main(out):
 for mode,enabled in [('baseline0',True),('single0',False),('single1',False),('baseline1',True)]:
  with (out/(mode+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',mode,'--enabled',str(int(enabled))],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  if p.returncode:raise RuntimeError('H1 controlled worker failed; no numerical retry')
 put(out/'execution.json',dict(job_id=os.getenv('SLURM_JOB_ID'),status='EXECUTED_H1_COMPARISONS_PENDING_CPU',variants=['baseline0','single0','single1','baseline1'],scientific_loss_changed=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--worker');p.add_argument('--enabled',type=int);a=p.parse_args();worker(a.out,a.worker,bool(a.enabled)) if a.worker else main(a.out)

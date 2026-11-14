"""Candidate verification: two raw captures, ordinary pair, resumed next update.

Ordinary/resume workers use the original canonical trainer with no added layer,
parameter, clipping or backward hooks. Only model loading opts into the guard.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def worker(out,mode):
 import torch
 from src.rl.cuda_execution_guard import configure_strict_diffusion_backward
 assert not torch.cuda.is_initialized();guard=configure_strict_diffusion_backward()
 cfg=json.loads((out/'protocol.json').read_text())
 for n,h in cfg['source_hashes'].items():assert sha(ROOT/n)==h,n
 import src.models.pretrained_editor_loader as loader
 original=loader.load_pretrained_editor
 def strict(*args,**kwargs):kwargs['strict_backward']=True;return original(*args,**kwargs)
 loader.load_pretrained_editor=strict
 if mode.startswith('raw'):
  from scripts.run_overnight_norm_fixture_20261102 import worker as run
 else:
  from scripts.run_overnight_ordinary_resume_20261102 import worker as run
 try:run(out,mode)
 finally:loader.load_pretrained_editor=original
 put(out/mode/'execution_guard.json',dict(**guard,actual_environment=os.environ['CUDA_LAUNCH_BLOCKING'],ordinary_uninstrumented=not mode.startswith('raw'),loader_optin=True))
def main(out):
 started=time.perf_counter()
 for mode in ['raw0','raw1','ordinary0','ordinary1','resume']:
  env={**os.environ,'CUDA_LAUNCH_BLOCKING':'0'} # The new source guard must set1.
  with (out/(mode+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
  if p.returncode:raise RuntimeError('Verification worker failed; never release or retry untilPASS')
 put(out/'execution.json',dict(job_id=os.getenv('SLURM_JOB_ID'),seconds=time.perf_counter()-started,workers=['raw0','raw1','ordinary0','ordinary1','resume'],candidate='explicit loader opt-in to pre-CUDA launch blocking'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--worker');a=p.parse_args();worker(a.out,a.worker) if a.worker else main(a.out)

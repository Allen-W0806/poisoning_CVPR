"""H3 ABBA: CUDA launch blocking only; all numerical settings stay fixed."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put
def main(out):
 for mode,blocking in [('baseline0',False),('blocked0',True),('blocked1',True),('baseline1',False)]:
  env={**os.environ,'CUDA_LAUNCH_BLOCKING':'1' if blocking else '0'}
  with (out/(mode+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-B',__file__,str(out),'--worker',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
  if p.returncode:raise RuntimeError('H3 worker failed; no automatic numerical retry')
 put(out/'execution.json',dict(job_id=os.getenv('SLURM_JOB_ID'),variants=['baseline0','blocked0','blocked1','baseline1'],only_factor='CUDA_LAUNCH_BLOCKING=0 versus1',scientific_loss_changed=False))
def worker(out,mode):
 import torch
 from scripts.run_backward_h1_20261102 import worker as run
 assert not torch.cuda.is_initialized();blocking=os.environ['CUDA_LAUNCH_BLOCKING'];run(out,mode,True)
 record=json.loads((out/mode/'H1_runtime.json').read_text());record.update(CUDA_LAUNCH_BLOCKING=blocking,only_factor_changed='CUDA launch blocking; autograd multithreading keptTrue',CUDA_environment_set_before_initialization=True);put(out/mode/'H3_runtime.json',record)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--worker');a=p.parse_args();worker(a.out,a.worker) if a.worker else main(a.out)

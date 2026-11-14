"""Held Slurm submission, live cumulative budget, no recurring monitor/retries."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def accounting():
 ids=set()
 for folder in ('gradient_feedback_summary_20261028','matched_feedback_20261028_v2','semantic_clean_20261029_v1'):
  for r in read(ROOT/'results'/folder/'resource_accounting.json')['jobs']:ids.add(r['job_id'])
 for r in read(ROOT/'results/replication_monitor_20261029_v1/resource_accounting.json')['registered_GPU_jobs']:ids.add(r['job_id'])
 raw=subprocess.check_output(['sacct','-X','-nP','-j',','.join(sorted(ids)),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode'],text=True);rows=[]
 for line in raw.splitlines():
  x=line.split('|')
  if len(x)<5 or x[0] not in ids:continue
  import re
  m=re.search(r'gres/gpu=(\d+)',x[3]);g=int(m[1]) if m else 0
  assert x[1].split()[0].rstrip('+') in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
  rows.append(dict(job_id=x[0],state=x[1],elapsed=int(x[2]),gpu_count=g,gpu_seconds=g*int(x[2]),exit_code=x[4]))
 assert {r['job_id'] for r in rows}==ids,'Accounting missing prior registered work'
 return dict(rows=rows,total=sum(r['gpu_seconds'] for r in rows),raw=raw)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--cpu-job',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id
 assert not (run/'submission_intent.json').exists(),'Never duplicate an existing/uncertain submission'
 assert read(run/'cpu_complete.json')['status']=='PASS'
 prior=accounting();assert prior['total']+900<=3600,'Global budget gate failed';assert read(run/'protocol.json')['gpu_cap_seconds']==900
 names=['scripts/small_scale_validation_20261031.py','scripts/run_qwen_replication.py','scripts/validate_ddpo_wiring.py','configs/gradient_feedback_validation_v2.json','configs/env-spec.yaml']
 names += [str(p.relative_to(ROOT)) for top in ('src/rl','src/rsi','src/training') for p in (ROOT/top).rglob('*.py')]
 write(run/'source_manifest.json',{n:sha(ROOT/n) for n in names})
 write(run/'budget_admission.json',dict(live_prior=prior,new_stage_max=900,global_max=3600,remaining_after_max=3600-prior['total']-900,no_budget_increase=True))
 command=['sbatch','--parsable','--hold','--no-requeue','--partition=epyc-gpu','--account=allusers','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time=00:15:00','--job-name=poisoning-small-benign','--comment=poisoning-small:'+a.run_id,'--output='+str(run/'logs/gpu-%j.out'),'--error='+str(run/'logs/gpu-%j.err'),'--wrap=cd '+str(ROOT)+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 '+str(ROOT/'.venv-editor/bin/python')+' -B scripts/small_scale_validation_20261031.py gpu --run-id '+a.run_id]
 write(run/'submission_intent.json',dict(command=command,at=time.time(),cpu_job=a.cpu_job))
 r=subprocess.run(command,capture_output=True,text=True);write(run/'submission_reply.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));r.check_returncode();job=r.stdout.strip().split(';')[0];assert job.isdigit()
 write(run/'job_ledger.json',dict(cpu_job=a.cpu_job,gpu_job=job,maximum_gpu_seconds=900,prior_live_seconds=prior['total'],recurring_monitor_created=False))
 subprocess.run(['scontrol','release',job],check=True)
 print(json.dumps(dict(gpu_job=job,cpu_job=a.cpu_job,global_prior_seconds=prior['total'],new_cap_seconds=900)))

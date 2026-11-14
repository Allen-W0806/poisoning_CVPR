"""Transfer only new scripts; use Slurm, bounded allocations and auditable ledger."""
import argparse,base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='next_validation_20261101T0040Z_v1'
def remote(code):
 r=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=80)
 print(r.stdout.decode(),end='');print(r.stderr.decode(),end='');r.check_returncode()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu','finalizer','report-source','sync']);a=p.parse_args()
 if a.mode in ('cpu','gpu'):
  files=['scripts/next_validation_20261031.py'] if a.mode=='cpu' else ['scripts/next_validation_20261031.py','scripts/integrated_benign_rsi_20261031.py']
  package={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in files}
  remote('import base64\nfrom pathlib import Path\nr=Path("/storage/home/ydk297/projects/poisoning")\nfor n,v in '+repr(package)+'.items():\n p=r/n\n if p.exists():assert p.read_bytes()==base64.b64decode(v),"Refuse overwrite changed deployed source"\n else:p.write_bytes(base64.b64decode(v))\n')
 if a.mode=='cpu':
  remote('''import subprocess,json
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');name='''+repr(RUN)+'''
assert not (r/'results'/name).exists()
previous=list((r/'reports').glob('CPU_SUBMIT_'+name+'*.json'))
assert all(json.loads(p.read_text())['returncode']!=0 and not json.loads(p.read_text())['stdout'].strip() for p in previous)
assert len(previous)<2
cmd=['sbatch','--parsable','--no-requeue','--partition=epyc-cpu','--account=allusers','--cpus-per-task=4','--mem=8G','--time=00:05:00','--job-name=poisoning-next-cpu','--output='+str(r/'logs'/(name+'-cpu-%j.out')),'--error='+str(r/'logs'/(name+'-cpu-%j.err')),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/next_validation_20261031.py prepare --run-id '+name]
p=subprocess.run(cmd,capture_output=True,text=True);receipt=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);(r/'reports'/('CPU_SUBMIT_'+name+'_a'+str(len(previous))+'.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));p.check_returncode()
''')
 elif a.mode=='finalizer':
  package={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in ['scripts/report_next_validation_20261031.py','src/progression/dashboard.py']}
  remote('''import base64,json,subprocess
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');name='''+repr(RUN)+''';run=r/'results'/name
files='''+repr(package)+'''
for n,v in files.items():
 p=r/n;content=base64.b64decode(v)
 if n=='src/progression/dashboard.py':
  old=p.read_bytes();assert b'from scripts.report_small_scale_20261031 import dashboard' in old and b'next_dashboard' not in old
  (run/'dashboard_entry_before.py').write_bytes(old)
 elif p.exists():assert p.read_bytes()==content
 p.write_bytes(content)
ledger=json.loads((run/'job_ledger.json').read_text());assert 'finalizer_job' not in ledger
cmd=['sbatch','--parsable','--no-requeue','--dependency=afterany:'+ledger['gpu_job'],'--partition=epyc-cpu','--account=allusers','--cpus-per-task=1','--mem=4G','--time=00:03:00','--job-name=poisoning-next-finalize','--output='+str(run/'logs/finalizer-%j.out'),'--error='+str(run/'logs/finalizer-%j.err'),'--wrap=cd '+str(r)+' && PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/report_next_validation_20261031.py']
p=subprocess.run(cmd,capture_output=True,text=True);(run/'finalizer_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2));p.check_returncode();ledger['finalizer_job']=p.stdout.strip().split(';')[0];(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
''')
 elif a.mode=='report-source':
  content=base64.b64encode((ROOT/'scripts/report_next_validation_20261031.py').read_bytes()).decode()
  remote('''import base64,hashlib
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');p=r/'scripts/report_next_validation_20261031.py';old=p.read_bytes();backup=r/'results/next_validation_20261101T0040Z_v1/audit'/('report_source_'+hashlib.sha256(old).hexdigest()[:12]+'.py');backup.parent.mkdir(parents=True,exist_ok=True)
if not backup.exists():backup.write_bytes(old)
t=p.with_suffix('.partial');t.write_bytes(base64.b64decode('''+repr(content)+'''));t.replace(p);print('Reporting source updated; experimental source unchanged')
''')
 elif a.mode=='sync':
  code='''import base64,json
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');name='''+repr(RUN)+''';files={}
def add(p):
 if p.is_file():files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
for folder in r.glob('results/'+name+'*'):
 for p in folder.rglob('*'):
  if p.suffix in {'.json','.jsonl','.csv','.md','.py','.out','.err','.log','.html'} or ('semantic_review32' in p.parts and p.suffix in {'.jpg','.png'}):add(p)
for name in ['NEXT_VALIDATION_DECISION_20261031.md','NEXT_VALIDATION_STATUS_20261031.json','NEXT_VALIDATION_DASHBOARD_QA_20261031.json','validation_gpu_reservations.json']:add(r/'reports'/name)
for p in (r/'reports').glob('CPU_SUBMIT_next_validation_20261101T0040Z_v1*.json'):add(p)
for p in (r/'dashboard').glob('*'):
 if p.suffix in {'.html','.json'}:add(p)
add(r/'EXPERIMENT_INDEX.md');add(r/'reports/monitoring/scheduler.json');add(r/'reports/rsi_stage_monitor/heartbeat.json')
assert sum(map(len,files.values()))<70_000_000
print(json.dumps(files))
'''
  p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=100);p.check_returncode();files=json.loads(p.stdout);total=0
  for rel,value in files.items():
   path=(ROOT/rel).resolve();assert path.is_relative_to(ROOT.resolve());path.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(value);path.write_bytes(data);total+=len(data)
  print(json.dumps(dict(files=len(files),bytes=total,model_weights_transferred=False)))
 elif a.mode=='gpu':
  remote('''import sys,subprocess,json,fcntl,hashlib,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');sys.path.insert(0,str(r));from scripts.submit_small_scale_20261031 import accounting
name='''+repr(RUN)+''';run=r/'results'/name
with (r/'reports/next_validation_budget.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((run/'cpu_complete.json').read_text())['status']=='PASS'
 assert not (run/'submission_intent.json').exists()
 assert hashlib.sha256((r/'src/rl/phase5_backend.py').read_bytes()).hexdigest()=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
 prior=accounting();raw=subprocess.check_output(['sacct','-X','-nP','-j','87921','--format=JobID,State,ElapsedRaw,ExitCode'],text=True);x=raw.strip().split('|');assert x[:2]==['87921','COMPLETED'] and x[3]=='0:0';total=prior['total']+int(x[2]);assert total+600<=3600
 active=subprocess.check_output(['squeue','-u',__import__('getpass').getuser(),'-h','-o','%i|%j|%k|%T|%l'],text=True);assert not any('poisoning' in s and 'monitor' not in s for s in active.splitlines()),'Other active Poisoning job requires explicit reservation reconciliation'
 names=['scripts/next_validation_20261031.py','scripts/integrated_benign_rsi_20261031.py','scripts/run_ip2pix_checkpoint_diagnostic.py','scripts/run_ip2pix_checkpoint_instrumented.py','configs/env-spec.yaml','configs/phase3_generator_rl.yaml']+[str(p.relative_to(r)) for top in ['src/rl','src/rsi','src/training'] for p in (r/top).rglob('*.py')]
 source={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names};(run/'source_manifest.json').write_text(json.dumps(source,indent=2))
 admission=dict(prior_live_seconds=total,maximum_reserved_seconds=600,global_cap=3600,remaining_after_maximum=3600-total-600,raw_sacct=raw,live_squeue=active,all_prior_jobs=prior['rows'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(run/'budget_admission.json').write_text(json.dumps(admission,indent=2))
 cmd=['sbatch','--parsable','--hold','--no-requeue','--partition=epyc-gpu','--account=allusers','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time=00:10:00','--job-name=poisoning-next-benign','--comment=poisoning-next:'+name,'--output='+str(run/'logs/gpu-%j.out'),'--error='+str(run/'logs/gpu-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/next_validation_20261031.py gpu --run-id '+name]
 (run/'submission_intent.json').write_text(json.dumps(cmd));p=subprocess.run(cmd,capture_output=True,text=True);(run/'submission_reply.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();ledger=dict(run_id=name,gpu_job=job,maximum_gpu_seconds=600,prior_live_seconds=total,recurring_monitor_created=False);(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));(r/'reports/validation_gpu_reservations.json').write_text(json.dumps(dict(cap=3600,consumed=total,reserved=600,reservations=[ledger]),indent=2));subprocess.run(['scontrol','release',job],check=True);print(json.dumps(ledger))
''')

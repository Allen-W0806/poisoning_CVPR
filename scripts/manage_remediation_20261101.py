"""One bounded benign job, source archive and live admission; no new monitor."""
import argparse,base64,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='audit_remediation_20261101T0120Z_v1'
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=100);print(p.stdout.decode(),end='');print(p.stderr.decode(),end='');p.check_returncode()
def gpu():
 run=ROOT/'results'/RUN;receipts=[]
 for name,script in [('BF16_ordinary','run_ip2pix_checkpoint_diagnostic.py'),('BF16_instrumented','run_ip2pix_checkpoint_instrumented.py'),('threeway','benign_threeway_20261101.py')]:
  arg=RUN if name=='threeway' else RUN+'_'+name
  (run/'live.json').write_text(json.dumps(dict(stage=name,time_unix=time.time(),job=__import__('os').environ.get('SLURM_JOB_ID'))))
  start=time.perf_counter()
  with (run/'logs'/(name+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-B','scripts/'+script,'--run-id',arg],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  row=dict(task=name,returncode=p.returncode,GPU_allocation_held_seconds=time.perf_counter()-start);receipts.append(row);(run/'executions'/(name+'.json')).write_text(json.dumps(row,indent=2))
 (run/'gpu_complete.json').write_text(json.dumps(dict(status='COMPLETED_WITH_LIMITATIONS',executions=receipts),indent=2));(run/'live.json').write_text(json.dumps(dict(stage='complete; inspect nested assertions',time_unix=time.time())))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['launch','resume-launch','gpu','cpu-audit','sync']);a=ap.parse_args()
 if a.mode=='gpu':gpu();raise SystemExit()
 if a.mode in ['launch','resume-launch']:
  files=['scripts/manage_remediation_20261101.py','scripts/benign_threeway_20261101.py','tests/test_fresh_batch_remediation.py','src/rl/diffusion_trajectory.py','src/rl/diffusion_rl_trainer.py'];package={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in files}
  code='''import sys,json,base64,subprocess,hashlib,shutil,fcntl,getpass,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');sys.path.insert(0,str(r));run=r/'results'/'''+repr(RUN)+''';assert not run.exists();run.mkdir();(run/'logs').mkdir();(run/'executions').mkdir();(run/'audit').mkdir()
old=r/'results/next_validation_20261101T0040Z_v1/audit/executed_source'
for n in ['src/rl/diffusion_trajectory.py','src/rl/diffusion_rl_trainer.py']:
 assert (r/n).read_bytes()==(old/n).read_bytes(),'Remote scientific source changed unexpectedly'
 p=run/'audit/before'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/n,p)
code="from src.rl.diffusion_trajectory import assert_fresh;import json;t={'trajectory_id':'same','policy_version':0,'contract':{'benign':1}};assert_fresh(t,0,{'benign':1},set());assert_fresh(t,0,{'benign':1},set());print(json.dumps({'duplicate_checks_both_accepted':True,'scope':'F23 prepatch CPU reproduction, no optimizer'}))"
p=subprocess.run([str(r/'.venv-editor/bin/python'),'-B','-c',code],cwd=r,capture_output=True,text=True);assert p.returncode==0;pout=json.loads(p.stdout);(run/'audit/F23_before.json').write_text(json.dumps(pout,indent=2))
files='''+repr(package)+'''
for n,v in files.items():
 p=r/n
 if p.exists() and not n.startswith('src/'):assert p.read_bytes()==base64.b64decode(v)
 p.write_bytes(base64.b64decode(v))
with (run/'logs/F23_regression.log').open('x') as f:p=subprocess.run([str(r/'.venv-editor/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_fresh_batch_remediation.py','-v'],cwd=r,stdout=f,stderr=subprocess.STDOUT)
(run/'audit/F23_after.json').write_text(json.dumps(dict(status='PASS' if p.returncode==0 else 'FAIL',returncode=p.returncode,tests=5,source_hashes={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files if n.startswith('src/')}),indent=2));assert p.returncode==0,'Preserve failed regression; no GPU admission'
protocol=dict(run_id=run.name,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_cap_seconds=600,global_validation_cap_seconds=3600,training_seeds=[42,123,3407],arms=['fixed','adaptive_scheduler','rsi'],objective='same benign terminal cos(first latent coordinate) as previous tiny test',pre_adaptation_steps=2,post_adaptation_steps=2,logical_batch_size=10,diffusion_steps=2,initial_lr=1e-5,ordinary_scheduler='ReduceLROnPlateau max, factor .5, patience0, threshold0; initial and preadaptation validation',RSI_proposal='cooldown .5 on no initial-to-prefix gain, else LR2x; candidate accepted iff validation mean strictly improves own prefix and cost gate passes; rejection restores OWN prefix',validation_conditions=dict(seed=100000,n=16),untouched_synthetic_test_conditions=dict(seed=200000,n=32),per_arm_evaluation_queries=80,per_arm_actual_forward_NFE=320,per_arm_optimizer_calls=4,common_initial_full_state_and_preadaptation_full_state_must_be_bitwise_equal=True,bootstrap='5000 crossed resamples of3 training seeds and32 shared synthetic conditions, seed20261101',BF16=dict(repetitions_per_path=1,paths=['ordinary','instrumented'],seed=42,acceptance='existing bitwise assertions unchanged; first mismatch preserved; no run-until-pass'),human_review='reuse existing predeclared32,0 genuine labels',source_guard_sha256='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712',no_VLM_reward=True,no_attack_metrics=True)
(run/'protocol.json').write_text(json.dumps(protocol,indent=2));base=json.loads((r/'results/replication_monitor_20261029_v1_ip2pix_a0/protocol.json').read_text())
for kind in ['ordinary','instrumented']:
 child=r/'results'/(run.name+'_BF16_'+kind);child.mkdir();(child/'protocol.json').write_text(json.dumps({**base,'parent_run':run.name},indent=2))
with (r/'reports/next_validation_budget.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from scripts.submit_small_scale_20261031 import accounting
 prior=accounting();raw=subprocess.check_output(['sacct','-X','-nP','-j','87921,88016','--format=JobID,State,ElapsedRaw,ExitCode'],text=True);extra=[]
 for line in raw.splitlines():
  x=line.split('|');assert x[1]=='COMPLETED' and x[3]=='0:0';extra.append(int(x[2]))
 assert len(extra)==2;total=prior['total']+sum(extra);globalledger=json.loads((r/'reports/validation_gpu_reservations.json').read_text());assert total==globalledger['consumed'] and globalledger['reserved']==0;assert total+600<=3600
 active=subprocess.check_output(['squeue','-u',getpass.getuser(),'-h','-o','%i|%j|%T|%l|%k'],text=True);assert not any('poisoning' in x and 'monitor' not in x for x in active.splitlines())
 assert hashlib.sha256((r/'src/rl/phase5_backend.py').read_bytes()).hexdigest()==protocol['source_guard_sha256']
 (run/'budget_before.json').write_text(json.dumps(globalledger,indent=2));(run/'budget_admission.json').write_text(json.dumps(dict(prior_live_seconds=total,reserved_seconds=600,remaining_after_max=3600-total-600,raw_sacct=raw,active_squeue=active),indent=2))
 names=list(files)+['scripts/run_ip2pix_checkpoint_diagnostic.py','scripts/run_ip2pix_checkpoint_instrumented.py','configs/env-spec.yaml','configs/phase3_generator_rl.yaml','src/rl/phase5_backend.py','src/rsi/controller.py','src/rsi/memory.py','src/rsi/revision.py'];manifest={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names};(run/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
 for n in names:
  p=run/'audit/executed_source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/n,p)
 cmd=['sbatch','--parsable','--hold','--no-requeue','--partition=epyc-gpu','--account=allusers','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time=00:10:00','--job-name=poisoning-audit-benign','--comment=poisoning-audit:'+run.name,'--output='+str(run/'logs/gpu-%j.out'),'--error='+str(run/'logs/gpu-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/manage_remediation_20261101.py gpu']
 (run/'submission_intent.json').write_text(json.dumps(cmd));p=subprocess.run(cmd,capture_output=True,text=True);(run/'submission_reply.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();ledger=dict(run_id=run.name,gpu_job=job,cap_seconds=600,prior_live_seconds=total,new_monitor_created=False);(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));(r/'reports/validation_gpu_reservations.json').write_text(json.dumps(dict(cap=3600,consumed=total,reserved=600,reservations=[ledger]),indent=2));subprocess.run(['scontrol','release',job],check=True);print(json.dumps(ledger))
'''
  if a.mode=='resume-launch':
   suffix=code[code.index("with (r/'reports/next_validation_budget.lock')"):]
   code='''import sys,json,base64,subprocess,hashlib,shutil,fcntl,getpass,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');sys.path.insert(0,str(r));run=r/'results'/'''+repr(RUN)+''';assert run.exists() and not (run/'submission_intent.json').exists();assert json.loads((run/'audit/F23_after.json').read_text())['status']=='PASS';protocol=json.loads((run/'protocol.json').read_text());files='''+repr(package)+'''
(run/'audit/launch_failure_a0.json').write_text(json.dumps(dict(error="TypeError unsupported dict+list in submission source manifest",GPU_job_submitted=False,fix='list(files) before concatenation'),indent=2))
for n,v in files.items():
 p=r/n;content=base64.b64decode(v)
 if p.read_bytes()!=content:
  assert n=='scripts/manage_remediation_20261101.py';shutil.copyfile(p,run/'audit/submission_wrapper_a0.py');p.write_bytes(content)
'''+suffix
  remote(code)
 elif a.mode=='cpu-audit':
  blob=base64.b64encode((ROOT/'scripts/audit_remediation_20261101.py').read_bytes()).decode()
  remote('''import base64,json,subprocess
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+''';p=r/'scripts/audit_remediation_20261101.py';content=base64.b64decode('''+repr(blob)+''')
if p.exists():assert p.read_bytes()==content
else:p.write_bytes(content)
assert not (run/'cpu_submission.json').exists()
cmd=['sbatch','--parsable','--no-requeue','--partition=epyc-cpu','--account=allusers','--cpus-per-task=2','--mem=8G','--time=00:04:00','--job-name=poisoning-audit-cpu','--output='+str(run/'logs/cpu-%j.out'),'--error='+str(run/'logs/cpu-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/audit_remediation_20261101.py --run-id '+run.name]
p=subprocess.run(cmd,capture_output=True,text=True);(run/'cpu_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2));p.check_returncode();ledger=json.loads((run/'job_ledger.json').read_text());ledger['cpu_job']=p.stdout.strip().split(';')[0];(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
''')
 elif a.mode=='sync':
  code='''import json,base64
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');files={}
def add(p):
 if p.is_file():files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
for folder in r.glob('results/'''+RUN+'''*'):
 for p in folder.rglob('*'):
  if p.suffix in {'.json','.jsonl','.csv','.md','.log','.out','.err','.py','.html'}:add(p)
for name in ['ROOT_CAUSE_AND_REMEDIATION.md','SCIENTIFIC_GATE_MATRIX.md','SMALL_SCALE_VALIDATION_NEXT.md','REMEDIATION_STATUS_20261101.json','validation_gpu_reservations.json']:add(r/'reports'/name)
for p in (r/'dashboard').glob('*'):
 if p.suffix in {'.html','.json'}:add(p)
add(r/'reports/monitoring/scheduler.json');add(r/'EXPERIMENT_INDEX.md');assert sum(map(len,files.values()))<25_000_000;print(json.dumps(files))
'''
  p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=100);p.check_returncode();files=json.loads(p.stdout)
  for rel,value in files.items():
   path=(ROOT/rel).resolve();assert path.is_relative_to(ROOT.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
  print(json.dumps(dict(files=len(files),bytes=sum(len(base64.b64decode(v)) for v in files.values()),weights_copied=False)))

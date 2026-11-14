"""Event-driven scoped controller. New budget, no recurring polling timer.

Every submission is held until durable registration and its afterany callback
are recorded. An uncertain submission is never blindly retried. Science guards
and the original 3600-second ledger are not changed.
"""
import argparse,copy,fcntl,hashlib,json,math,os,re,shlex,signal,subprocess,sys,time,uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.monitoring.common import FileLock
RUN='overnight_benign_20261102_v2';BASE=ROOT/'reports'/RUN;OUT=ROOT/'results'/RUN
GUARD='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','UNKNOWN'}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def now():return datetime.now(timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except FileNotFoundError:return default
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
 with q.open('w',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 q.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
@contextmanager
def serialized_controller(wait=False):
 # Event handlers wait on the kernel lock (bounded by their Slurm CPU walltime),
 # rather than consuming the only afterany notification as a nonblocking no-op.
 BASE.mkdir(parents=True,exist_ok=True)
 with (BASE/'controller.lock').open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|(0 if wait else fcntl.LOCK_NB))
  except BlockingIOError:yield False;return
  try:yield True
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def command(args,check=True):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=45,env={**os.environ,'TZ':'America/New_York'})
 if check:p.check_returncode()
 return p
def ledger(s):
 rows=[];used=reserved=0
 for t in s['GPU_jobs']:
  elapsed=int(t.get('elapsed_seconds',0));st=t.get('scheduler_state','UNKNOWN');cap=int(t['reservation_seconds']);g=int(t.get('allocated_GPUs',1));cost=elapsed*g
  remain=0 if st in TERMINAL else max(0,cap-cost)
  used+=cost;reserved+=remain;rows.append(dict(job_id=t.get('job_id'),stage=t['stage'],state=st,GPU_seconds=cost,reserved_GPU_seconds=remain,exit_code=t.get('exit_code'),reservation_seconds=cap))
 return dict(run_id=s['run_id'],cap=36000,consumed=used,reserved=reserved,remaining=36000-used-reserved,rows=rows,deadline_UTC=s['deadline_UTC'],initialized_UTC=s['initialized_UTC'],scope='NEW overnight allocations only; previous3600-second ledger is separate and unchanged')
def can_admit(s,cap,stage,clock=None):
 b=ledger(s);clock=time.time() if clock is None else clock
 if s.get('admissions_closed'):return False,'admissions closed'
 if clock+cap+120>s['deadline_epoch']:return False,'wall deadline/finish margin'
 if b['remaining']<cap:return False,'cumulative allocated GPU budget'
 if any(t.get('scheduler_state','UNKNOWN') in ACTIVE for t in s['GPU_jobs']):return False,'registered allocation or uncertain submission still active'
 if stage in {'numerical','ordinary_resume'} and sum(t.get('elapsed_seconds',0) if t.get('scheduler_state') in TERMINAL else t['reservation_seconds'] for t in s['GPU_jobs'] if t['stage'] in {'numerical','ordinary_resume'})+cap>s['numerical_budget_seconds']:return False,'predeclared numerical sub-budget'
 return True,'eligible resource envelope'
def collect(s):
 ids=[t['job_id'] for t in s['GPU_jobs'] if t.get('job_id')]
 if not ids:return
 p=command(['sacct','-X','-nP','-j',','.join(ids),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,NodeList,Start,End']);by={x[0]:x for line in p.stdout.splitlines() if len(x:=line.split('|'))>=8}
 for t in s['GPU_jobs']:
  v=by.get(t.get('job_id'))
  if not v:continue
  gpu=int((re.search(r'gres/gpu=(\d+)',v[3]) or [None,'0'])[1]);assert gpu==1 or v[1].startswith(('PENDING','CANCELLED'))
  t.update(scheduler_state=v[1].split()[0].rstrip('+'),elapsed_seconds=int(v[2]),allocated_GPUs=gpu or 1,exit_code=v[4],node=v[5],start_cluster_local=v[6],end_cluster_local=v[7])
 write(BASE/'budget.json',ledger(s))
def runtime_sources():
 names=['src/rl/diffusion_policy.py','src/rl/diffusion_rl_trainer.py','src/rl/diffusion_trajectory.py','src/rl/phase5_backend.py','scripts/gradient_boundary_capture_20261102.py','scripts/run_overnight_norm_fixture_20261102.py','scripts/run_overnight_ordinary_resume_20261102.py']
 return {n:sha(ROOT/n) for n in names}
def protocol(out,stage='numerical'):
 p=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';tr=ROOT/'results/numerical_root_cause_20261102_v1/C1_live.pt'
 value=dict(run_id=RUN,created_UTC=now(),pilot=str(p.relative_to(ROOT)),trajectory_file=str(tr.relative_to(ROOT)),trajectory_sha256=sha(tr),checkpoint_sha256=sha(p/'shared_prefix/training_state.pt'),source_manifest_sha256=sha(p/'source_manifest.json'),source_hashes=runtime_sources(),guard_sha256=GUARD,constant_coefficient=1.,model_settings='Pinned88392 B10/T2, sequential CFG, MATH SDPA, BF16 frozen forward/FP32 LoRA, no activation checkpointing, same source/order/trajectory/optimizer/LR5e-6',residency_order=['fresh0 one update in process0','fresh1 one update in process1','resident first and second updates in process2, restoring identical initial states'],sole_treatment='process residency; same minimal end-of-backward clipping witness in every update',capture='Ordered raw/postclip gradients, runtime norm and coefficient, FP64 CPU reference and per-tensor norms, before/after adapter/Adam/RNG; no activations',strict_exact=True,tolerance=dict(atol=1e-7,rtol=1e-5,reference='first tensor'),no_retry_until_PASS=True,production_repair='NONE PREAPPROVED; evidence must justify a versioned correction before any fix claim',estimated_cluster_evidence_bytes=300000000,numerical_budget_seconds=1200)
 value['stage']=stage
 if stage=='ordinary_resume':value.update(residency_order=['ordinary0 two canonical updates in process0','ordinary1 identical two-update repeat in process1','resume process2 restores ordinary0 after first update, collects second trajectory and performs next update'],sole_treatment='checkpoint recovery and process restart for next update; no clipping/forward hooks',capture='Canonical checkpoints and already-produced postclip gradient buffers; live next trajectories; no added gradient hooks',numerical_fix_claim=False)
 write(out/'protocol.json',value)
def submission_args(s,kind,token,wrap,wall=600,dependency=None,begin=None):
 args=['sbatch','--parsable','--hold','--no-requeue','--account=allusers','--partition='+('epyc-gpu' if kind=='gpu' else 'epyc-cpu'),'--cpus-per-task='+('4' if kind=='gpu' else '2'),'--mem='+('48G' if kind=='gpu' else '8G'),'--time='+str(wall//60),'--job-name=poisoning-night-'+kind,'--comment='+RUN+':'+token,'--output='+str(OUT/('%x-%j.out')),'--error='+str(OUT/('%x-%j.err'))]
 if kind=='gpu':args+=['--gres=gpu:tesla_a100_80gb:1','--deadline='+datetime.fromtimestamp(s['deadline_epoch'],ZoneInfo('America/New_York')).strftime('%Y-%m-%dT%H:%M:%S')]
 if dependency:args+=['--dependency=afterany:'+dependency]
 if begin:args+=['--begin='+begin]
 return args+['--wrap='+wrap]
def wrap(mode,*args):
 return 'cd '+shlex.quote(str(ROOT))+' && export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && '+shlex.quote(str(ROOT/'.venv-editor/bin/python'))+' -B scripts/overnight_benign_controller_20261102.py '+mode+' '+' '.join(shlex.quote(str(a)) for a in args)
def registered_submit(s,kind,token,args,record):
 registry=s['GPU_jobs'] if kind=='gpu' else s['CPU_jobs'];assert not any(t['token']==token for t in registry)
 record.update(token=token,job_id=None,submission_UTC=now(),scheduler_state='UNKNOWN',command=args);registry.append(record);write(BASE/'state.json',s)
 if kind=='gpu':write(BASE/'budget.json',ledger(s))
 write(OUT/('submission_intent_'+token+'.json'),record)
 p=command(args,check=False);write(OUT/('submission_reply_'+token+'.json'),dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();record.update(job_id=job,scheduler_state='PENDING');write(BASE/'state.json',s);return job
def submit_gpu(s,stage,attempt=0,exclude=None):
 cap=360 if stage=='numerical' else 480;ok,reason=can_admit(s,cap,stage);assert ok,reason
 q=command(['squeue','-h','-u',os.environ['USER'],'-o','%i|%T|%b']).stdout
 if any('gpu:' in x or 'gres/gpu' in x for x in q.splitlines()):raise RuntimeError('Another user GPU allocation/pending request exists; no concurrent allocation')
 assert sha(ROOT/'src/rl/phase5_backend.py')==GUARD
 token=stage+'_a'+str(attempt);out=OUT/token;out.mkdir(exist_ok=False);protocol(out,stage)
 args=submission_args(s,'gpu',token,wrap('gpu',token),wall=cap-60)
 if exclude:args.insert(-1,'--exclude='+exclude)
 t=dict(stage=stage,attempt=attempt,run_directory=str(out.relative_to(ROOT)),reservation_seconds=cap,wall_seconds=cap-60,allocated_GPUs=1,elapsed_seconds=0,processed=False)
 job=registered_submit(s,'gpu',token,args,t)
 cbtoken='complete_'+job;cb=registered_submit(s,'cpu',cbtoken,submission_args(s,'cpu',cbtoken,wrap('event',job),wall=600,dependency=job),dict(purpose='afterany completion',parent_GPU_job=job))
 t['completion_job_id']=cb;write(BASE/'state.json',s);write(BASE/'budget.json',ledger(s))
 command(['scontrol','release',cb]);command(['scontrol','release',job]);s['status']='WAIT_GPU_COMPLETION';s['active_stage']=stage;write(BASE/'state.json',s)
def render(s):
 from scripts.overnight_benign_reporting_20261102 import render
 render(s,ROOT,BASE,OUT)
def register_accounting_recovery(s,parent):
 token='accounting_recovery_'+parent;current=os.getenv('SLURM_JOB_ID','')
 if not current.isdigit() or any(t['token']==token for t in s['CPU_jobs']):return
 args=submission_args(s,'cpu',token,wrap('event',parent),wall=600,dependency=current);args.insert(-1,'--begin=now+1minute')
 job=registered_submit(s,'cpu',token,args,dict(purpose='one-shot delayed sacct reconciliation; not recurring polling',parent_GPU_job=parent));command(['scontrol','release',job]);write(BASE/'state.json',s)
def close_unused_deadline(s):
 if not s.get('admissions_closed') or any(t.get('scheduler_state') in ACTIVE for t in s['GPU_jobs']):return
 for t in s['CPU_jobs']:
  if t['token']!='deadline' or t.get('cancelled_after_early_close'):continue
  job=t.get('job_id');q=command(['squeue','-h','-j',job,'-o','%T'],check=False)
  if q.stdout.strip()=='PENDING':
   details=command(['scontrol','show','job','-o',job]).stdout;assert 'Comment='+RUN+':deadline' in details
   command(['scancel',job]);t.update(cancelled_after_early_close=True,cancelled_UTC=now(),cancellation_reason='No eligible task remains; final report written before10h ceiling')
  write(BASE/'state.json',s)
def advance(s):
 collect(s)
 for t in s['GPU_jobs']:
  if t.get('processed') or t.get('scheduler_state') not in TERMINAL:continue
  out=ROOT/t['run_directory'];s['completed_jobs'].append(t['job_id']);t['processed']=True
  if t['scheduler_state']=='COMPLETED' and t.get('exit_code')=='0:0':
   from scripts.overnight_benign_reporting_20261102 import analyze_numerical,analyze_resume
   try:result=analyze_numerical(out) if t['stage']=='numerical' else analyze_resume(out)
   except Exception as error:
    result=dict(exact_pass=False,tolerance_pass=False,classification='ARTIFACT_VALIDATION_FAILURE',error=repr(error),first_boundary=None,fix_confirmed=False);write(out/'artifact_validation_failure.json',result)
   t['result']=result;t['classification']='NUMERICAL_PASS_SCOPED' if result['exact_pass'] else result.get('classification','NUMERICAL_MISMATCH_PRESERVED')
   if result['exact_pass']:s['passed_stages'].append(t['stage'])
   else:s['numerical_failed']=True;s['failure_history'].append(dict(job_id=t['job_id'],reason=t['classification'],first_boundary=result.get('first_boundary'),UTC=now()))
  else:
   t['classification']='INFRASTRUCTURE_OR_EXECUTION_FAILURE';s['failure_history'].append(dict(job_id=t['job_id'],state=t['scheduler_state'],exit_code=t.get('exit_code'),UTC=now()))
   # One narrowly justified repair: Slurm explicitly reports a failed node;
   # retry excludes that exact node, only if regression/policy guards pass.
   if t['scheduler_state']=='NODE_FAIL' and t['attempt']==0 and re.fullmatch(r'epyc\d+',t.get('node','')):
    ok,_=can_admit(s,t['reservation_seconds'],t['stage'])
    if ok:s['infrastructure_repair']=dict(failed_job=t['job_id'],identified_issue='Slurm NODE_FAIL',repair='exclude failed node '+t['node'],regression='test_one_node_retry');write(BASE/'state.json',s);submit_gpu(s,t['stage'],1,t['node']);render(s);return
   s['numerical_failed']=True
  write(BASE/'state.json',s)
 if any(t.get('scheduler_state') in ACTIVE for t in s['GPU_jobs']):render(s);return
 if s.get('numerical_failed'):
  s.update(status='STOPPED_NUMERICAL_OR_EXECUTION_FAILURE',admissions_closed=True,learning_eligible=False);render(s);write(BASE/'state.json',s);return
 if s.get('admissions_closed'):
  s.update(status='STOPPED_DEADLINE_WITH_RESULTS',learning_eligible=False);render(s);write(BASE/'state.json',s);return
 if 'numerical' not in s['passed_stages']:
  submit_gpu(s,'numerical');render(s);return
 if 'ordinary_resume' not in s['passed_stages']:
  submit_gpu(s,'ordinary_resume');render(s);return
 # No automatic override: closed independent-development and science-reward
 # admission gates remain closed even if this scoped numerical pair passes.
 s.update(status='STOPPED_NO_ELIGIBLE_LEARNING_STAGE',admissions_closed=True,learning_eligible=False,learning_blockers=['independent development evaluation NOT ADMITTED;160 final sources sealed','original scientific reward release remains denied','one local numerical PASS cannot erase historical failures'])
 render(s);write(BASE/'state.json',s)
def install():
 BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 with FileLock(BASE/'controller.lock') as acquired:
  assert acquired
  if (BASE/'state.json').exists():print(json.dumps(dict(status='ALREADY_INSTALLED',state=str(BASE/'state.json'))));return
  epoch=time.time();s=dict(run_id=RUN,initialized_UTC=datetime.fromtimestamp(epoch,timezone.utc).isoformat(),deadline_epoch=epoch+36000,deadline_UTC=datetime.fromtimestamp(epoch+36000,timezone.utc).isoformat(),GPU_jobs=[],CPU_jobs=[],completed_jobs=[],passed_stages=[],failure_history=[],admissions_closed=False,numerical_budget_seconds=1200,status='INITIALIZED',learning_eligible=False,independent_development_admitted=False,original_reward_release=False,previous_validation_ledger='reports/overnight_supervisor/state.json; kept separate',sealed_final_sources='results/execution_remediation_20261102_v1/prospective/candidate_content_manifest.json',sealed_manifest_sha256=sha(ROOT/'results/execution_remediation_20261102_v1/prospective/candidate_content_manifest.json'))
  before=OUT/'before';before.mkdir()
  for n in ['NUMERICAL_ROOT_CAUSE_LATEST.md','NEXT_ELIGIBLE_EXPERIMENT.md','EXECUTION_REMEDIATION_LATEST.md']:(before/n).write_bytes((ROOT/'reports'/n).read_bytes())
  write(BASE/'state.json',s);write(BASE/'budget.json',ledger(s));write(ROOT/'reports/OVERNIGHT_CONTROLLER_POINTER.json',dict(run_id=RUN,state_path=str((BASE/'state.json').relative_to(ROOT)),controller_script='scripts/overnight_benign_controller_20261102.py'))
  deadline=datetime.fromtimestamp(s['deadline_epoch'],ZoneInfo('America/New_York')).strftime('%Y-%m-%dT%H:%M:%S')
  endjob=registered_submit(s,'cpu','deadline',submission_args(s,'cpu','deadline',wrap('deadline'),wall=600,begin=deadline),dict(purpose='one-shot wall deadline finalization'))
  boot=registered_submit(s,'cpu','bootstrap',submission_args(s,'cpu','bootstrap',wrap('bootstrap'),wall=600),dict(purpose='CPU tests, dry run, stage admission'))
  command(['scontrol','release',endjob]);command(['scontrol','release',boot]);render(s);print(json.dumps(dict(run_id=RUN,bootstrap_job=boot,deadline_job=endjob,deadline_UTC=s['deadline_UTC'],budget=ledger(s))))
def controller(mode,job=None):
 with serialized_controller(wait=mode in {'event','bootstrap','deadline'}) as acquired:
  if not acquired:print(json.dumps(dict(status='LOCKED_NO_DUPLICATE',UTC=now())));return
  s=read(BASE/'state.json');assert s
  if mode in {'health','deadline'} and s.get('admissions_closed') and all(t.get('processed') for t in s['GPU_jobs']):
   close_unused_deadline(s)
   print(json.dumps(dict(status='TERMINAL_NOOP',run_id=s['run_id'],last_status=s['status'])));return
  s['last_controller_UTC']=now();s['last_controller_job_id']=os.getenv('SLURM_JOB_ID');s['last_event']=mode
  if mode=='bootstrap':
   with (OUT/'CPU_preparation_tests.log').open('x') as f:p=subprocess.run([str(ROOT/'.venv-editor/bin/python'),'-B','scripts/test_overnight_benign_20261102.py'],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   if p.returncode:s.update(status='STOPPED_CPU_PREPARATION_FAIL',admissions_closed=True);write(BASE/'state.json',s);render(s);raise RuntimeError('CPU regressions failed before GPU admission')
   s['CPU_preparation_PASS']=True;write(BASE/'state.json',s)
  if mode=='event':
   matches=[t for t in s['GPU_jobs'] if t.get('job_id')==job];assert len(matches)==1
   if matches[0].get('processed'):s['duplicate_callback_noops']=s.get('duplicate_callback_noops',0)+1;write(BASE/'state.json',s);print('IDEMPOTENT_ALREADY_PROCESSED');return
  if mode=='deadline' or time.time()>=s['deadline_epoch']:
   collect(s);s.update(admissions_closed=True,deadline_enforced_UTC=now(),status='DEADLINE_ADMISSIONS_CLOSED')
   for task in s['GPU_jobs']:
    if task.get('scheduler_state')=='PENDING' and task.get('job_id'):
     detail=command(['scontrol','show','job','-o',task['job_id']],check=False).stdout
     if 'Comment='+RUN+':'+task['token'] in detail and 'JobState=PENDING' in detail:command(['scancel',task['job_id']]);task['deadline_cancel_requested']=True
   write(BASE/'state.json',s);advance(s);close_unused_deadline(s);render(s);return
  if not s.get('CPU_preparation_PASS'):render(s);return
  advance(s)
  if mode=='event':
   parent=next(t for t in s['GPU_jobs'] if t.get('job_id')==job)
   if not parent.get('processed') and parent.get('scheduler_state') in ACTIVE:register_accounting_recovery(s,job)
  close_unused_deadline(s)
  # Fixed-model CPU evaluation is independent of failed train reproducibility.
  from scripts.overnight_benign_reporting_20261102 import fixed_model_metrics
  if not (OUT/'fixed_model_metrics.json').exists():fixed_model_metrics(ROOT,OUT);render(s)
  write(BASE/'state.json',s);print(json.dumps(dict(status=s['status'],budget=ledger(s))))
def gpu(token):
 s=read(BASE/'state.json');t=next(x for x in s['GPU_jobs'] if x['token']==token);out=ROOT/t['run_directory'];remaining=s['deadline_epoch']-time.time()
 if remaining<t['reservation_seconds']:write(out/'startup_guard.json',dict(status='REFUSED_LATE_START',remaining=remaining));return 75
 script='scripts/run_overnight_norm_fixture_20261102.py' if t['stage']=='numerical' else 'scripts/run_overnight_ordinary_resume_20261102.py';limit=min(t['wall_seconds']-20,int(remaining)-30)
 env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 with (out/'payload.log').open('x') as f:
  p=subprocess.Popen([str(ROOT/'.venv-editor/bin/python'),'-B',script,str(out)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
  try:code=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 write(out/'process_exit.json',dict(returncode=code,limit_seconds=limit,finished_UTC=now(),job_id=os.getenv('SLURM_JOB_ID')));return code if code>=0 else 128-code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['install','bootstrap','event','health','deadline','gpu']);p.add_argument('argument',nargs='?');a=p.parse_args()
 if a.mode=='install':install()
 elif a.mode=='gpu':raise SystemExit(gpu(a.argument))
 else:controller(a.mode,a.argument)

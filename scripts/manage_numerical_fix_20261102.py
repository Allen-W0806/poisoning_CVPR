"""Separate bounded urgent-debug authorization; never reopens the old controller."""
import argparse,json,os,shlex,subprocess,sys,time,uuid,signal
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.overnight_benign_controller_20261102 import command,read,write,now
from scripts.numerical_boundaries_20261102 import sha
from src.monitoring.common import FileLock
BASE=ROOT/'reports/numerical_fix_20261102_v1';OUT=ROOT/'results/numerical_fix_20261102_v1'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}

def account(s):
 ids=[j['job_id'] for j in s['jobs'] if j.get('job_id')]
 if ids:
  p=command(['sacct','-X','-nP','-j',','.join(ids),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode']);by={v[0]:v for line in p.stdout.splitlines() if len(v:=line.split('|'))>=5}
  for j in s['jobs']:
   if j.get('job_id') in by:
    v=by[j['job_id']];j.update(state=v[1].split()[0].rstrip('+'),elapsed=int(v[2]),exit_code=v[4]);assert 'gres/gpu=1' in v[3] or j['state'] in {'PENDING','CANCELLED'}
 consumed=sum(j.get('elapsed',0) for j in s['jobs']);reserved=sum(max(0,j['cap']-j.get('elapsed',0)) for j in s['jobs'] if j['state'] not in TERMINAL)
 result=dict(cap=960,consumed=consumed,reserved=reserved,remaining=960-consumed-reserved,deadline_UTC=s['deadline_UTC'],scope='New user-authorized urgent debug only; parent controller remains finalized',authorization_envelope_total_cap=36000,previous_overnight_spend=172,aggregate_envelope_spend=172+consumed);write(BASE/'budget.json',result);return result
def manage(mode,stage=None,autograd=True):
 BASE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 with FileLock(BASE/'lock') as acquired:
  assert acquired
  s=read(BASE/'state.json')
  if s is None:
   parent=read(ROOT/'reports/overnight_benign_20261102_v2/state.json');assert parent['admissions_closed'];deadline=parent['deadline_epoch'];assert time.time()<deadline
   s=dict(run_id='numerical_fix_20261102_v1',initialized_UTC=now(),deadline_epoch=deadline,deadline_UTC=parent['deadline_UTC'],cap=960,jobs=[],authorization='New urgent numerical-fix user request c1f1d8fe-6f07-483b-b04c-3ff9e9fa47de; bounded diagnostics, not old-controller reopening',parent_controller_remains_closed=True,previous_parent_numerical_spend=172,parent_numerical_cap=1200,scope='Nonadversarial constant-coefficient fixture only');write(BASE/'state.json',s)
   before=OUT/'before';before.mkdir(exist_ok=True)
   for n in ['NUMERICAL_ROOT_CAUSE_LATEST.md','OVERNIGHT_FINAL_REPORT.md','NEXT_ELIGIBLE_EXPERIMENT.md']:(before/n).write_bytes((ROOT/'reports'/n).read_bytes())
  b=account(s);write(BASE/'state.json',s)
  if mode=='close':s.update(admissions_closed=True,closed_UTC=now(),status='ROOT_CAUSE_PARTIALLY_LOCALIZED',reason='Ordinary guard verification failed and resumed optimizer update timed out; no further eligible bounded GPU test');write(BASE/'state.json',s);print(json.dumps(dict(status=s['status'],budget=b)));return
  if mode=='status':print(json.dumps(dict(budget=b,jobs=s['jobs'])));return
  assert not s.get('admissions_closed',False),'This debug run is finalized; do not silently reopen'
  assert mode=='submit' and stage not in {j['stage'] for j in s['jobs']};cap=300 if stage!='verification' else 420
  assert b['remaining']>=cap and time.time()+cap+120<s['deadline_epoch'];assert b['previous_overnight_spend']+b['consumed']+b['reserved']+cap<=1200
  assert not any(j['state'] not in TERMINAL for j in s['jobs'])
  active=command(['squeue','-h','-u',os.environ['USER'],'-o','%i|%T|%b']).stdout
  # Other explicitly allocated projects (e.g.RSI-Video) have independent scope.
  for line in active.splitlines():
   if 'gpu:' not in line and 'gres/gpu' not in line:continue
   job=line.split('|')[0];detail=command(['scontrol','show','job','-o',job]).stdout
   assert ('WorkDir='+str(ROOT)+' ') not in detail and ('Command='+str(ROOT)+'/') not in detail,'Another poisoning GPU allocation is active'
  out=OUT/stage;out.mkdir(exist_ok=False);old=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';trajectory=ROOT/'results/numerical_root_cause_20261102_v1/C1_live.pt'
  names=['src/rl/diffusion_policy.py','src/rl/diffusion_rl_trainer.py','src/rl/diffusion_trajectory.py','src/rl/phase5_backend.py','scripts/backward_boundary_capture_20261102.py','scripts/run_backward_localization_20261102.py','scripts/run_backward_h1_20261102.py','scripts/run_overnight_norm_fixture_20261102.py','scripts/run_backward_h3_20261102.py','src/models/pretrained_editor_loader.py','src/rl/cuda_execution_guard.py','scripts/run_numerical_fix_verification_20261102.py']
  cfg=dict(stage=stage,created_UTC=now(),source_hashes={n:sha(ROOT/n) for n in names},pilot=str(old.relative_to(ROOT)),checkpoint_sha256=sha(old/'shared_prefix/training_state.pt'),trajectory_file=str(trajectory.relative_to(ROOT)),trajectory_sha256=sha(trajectory),target_parameter='down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight',autograd_multithreading=autograd,guard_sha256=sha(ROOT/'src/rl/phase5_backend.py'),coefficient_constant=1.,B=10,T=2,LR=5e-6,ordinary_uninstrumented=False,estimated_evidence_bytes=600000000,change='No backend/precision/settings change; identical targeted node hooks' if autograd else 'H1 only: autograd multithreading disabled; same hooks,model,data,loss,precision and backend')
  if stage.startswith('h1'):cfg.update(variants=[['baseline0',True],['single0',False],['single1',False],['baseline1',True]],change='H1 only: default versus disabled autograd multithreading. No layer/parameter hooks; identical clipping-only end-of-backward capture in every worker.')
  if stage.startswith('h3'):cfg.update(variants=[['baseline0',0],['blocked0',1],['blocked1',1],['baseline1',0]],change='H3 only: CUDA_LAUNCH_BLOCKING0 versus1 set before CUDA initialization; MATH,deterministic algorithms,checkpointing,precision,CUBLAS workspace and autograd threads held fixed; no layer hooks.')
  if stage=='verification':cfg.update(variants=['raw0','raw1','ordinary0','ordinary1','resume'],change='Source-integrated optional loader guard sets CUDA_LAUNCH_BLOCKING1 before CUDA initialization. Two clipping-only raw captures; then original uninstrumented ordinary pair and actual resumed next update. No model/loss/precision/backend changes.')
  write(out/'protocol.json',cfg)
  wrap='cd '+shlex.quote(str(ROOT))+' && export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 && '+str(ROOT/'.venv-editor/bin/python')+' -B scripts/manage_numerical_fix_20261102.py gpu '+stage
  args=['sbatch','--parsable','--hold','--no-requeue','--account=allusers','--partition=epyc-gpu','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time='+str((cap-60)//60),'--deadline='+datetime.fromtimestamp(s['deadline_epoch'],ZoneInfo('America/New_York')).strftime('%Y-%m-%dT%H:%M:%S'),'--job-name=poisoning-fix-'+stage,'--comment=numerical_fix_20261102_v1:'+stage,'--output='+str(out/'stdout.log'),'--error='+str(out/'stderr.log'),'--wrap='+wrap]
  j=dict(stage=stage,state='UNKNOWN',cap=cap,job_id=None,submitted_UTC=now(),command=args);s['jobs'].append(j);write(BASE/'state.json',s);account(s);write(out/'submission_intent.json',j)
  p=command(args);write(out/'submission_reply.json',dict(stdout=p.stdout,stderr=p.stderr));job=p.stdout.strip().split(';')[0];assert job.isdigit();j.update(job_id=job,state='PENDING');write(BASE/'state.json',s);account(s);command(['scontrol','release',job]);print(json.dumps(dict(job_id=job,reserved=cap,deadline=s['deadline_UTC'],parent_controller='UNCHANGED_CLOSED')))
def gpu(stage):
 s=read(BASE/'state.json');j=next(j for j in s['jobs'] if j['stage']==stage);out=OUT/stage;remain=s['deadline_epoch']-time.time();assert remain>=j['cap'],'Late queued start refused before model load'
 with (out/'payload.log').open('x') as f:
  script='scripts/run_backward_h1_20261102.py' if stage.startswith('h1') else ('scripts/run_backward_h3_20261102.py' if stage.startswith('h3') else ('scripts/run_numerical_fix_verification_20261102.py' if stage=='verification' else 'scripts/run_backward_localization_20261102.py'))
  p=subprocess.Popen([str(ROOT/'.venv-editor/bin/python'),'-B',script,str(out)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
  try:code=p.wait(timeout=j['cap']-80)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 write(out/'process_exit.json',dict(returncode=code,finished_UTC=now(),job_id=os.getenv('SLURM_JOB_ID')));return code if code>=0 else 128-code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['status','submit','gpu','close']);p.add_argument('stage',nargs='?');p.add_argument('--single-thread',action='store_true');a=p.parse_args()
 if a.mode=='gpu':raise SystemExit(gpu(a.stage))
 else:manage(a.mode,a.stage,not a.single_thread)

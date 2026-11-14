"""Register exactly one requested diagnostic with the existing monitor/budget."""
import ast,fcntl,json,os,sys,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.overnight_supervisor_20261101 import accounting,inspect_tasks,read,write,now,sha,BASE
from src.monitoring.common import FileLock
OUT=ROOT/'results/numerical_root_cause_20261102_v1'

def main(mode):
 OUT.mkdir(exist_ok=True)
 with FileLock(BASE/'monitor.lock') as acquired:
  assert acquired,'Existing monitor owns lock; no duplicate action'
  with (ROOT/'reports/next_validation_budget.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   s=read(BASE/'state.json');inspect_tasks(s);b=accounting(s);write(OUT/('budget_'+mode+'.json'),dict(UTC=now(),**b));write(BASE/'state.json',s)
   if mode!='submit':
    print(json.dumps(dict(budget={k:b[k] for k in ['cap','consumed','reserved','remaining','unreconciled_project_GPU_jobs']},tasks=[{k:t.get(k) for k in ['task_key','job_id','scheduler_state','classification','elapsed_seconds']} for t in s['tasks'][-3:]],heartbeat=read(BASE/'heartbeat.json'))));return
   assert b['remaining']>=200 and not b['unreconciled_project_GPU_jobs'] and not b['reserved']
   assert not any(t['task_key']=='TARGETED_BOUNDARY_20261102_v1' for t in s['tasks']),'Already registered; never duplicate'
   paths=['scripts/run_numerical_boundary_fixture_20261102.py','src/rl/diffusion_policy.py','src/rl/diffusion_rl_trainer.py','src/rl/diffusion_trajectory.py','src/rl/phase5_backend.py']
   for n in paths:ast.parse((ROOT/n).read_text())
   protocol=dict(UTC=now(),task='TARGETED_BOUNDARY_20261102_v1',cap_A100_seconds=200,Slurm_wall_seconds=180,process_alarm_seconds=165,coefficient=1.,logical_batch=10,timesteps=2,source='existing development train[10:20], seed4210000+i',checkpoint='results/execution_remediation_20261102_v1_pretrained_pilot/shared_prefix/training_state.pt',checkpoint_sha256=sha(ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot/shared_prefix/training_state.pt'),active_LR=5e-6,order=['A1 ordinary live collection and update','A2 identical ordinary repeat','B1 fixed C1 replay with boundary capture','B2 identical fixed replay'],collection_witnesses=['C1_live.pt','C2_live.pt'],treatment='Instrumentation only on B; no backend/precision/determinism/checkpointing/accumulation setting changed',no_fresh_process_claim=True,source_hashes={n:sha(ROOT/n) for n in paths},guard_sha256=sha(ROOT/'src/rl/phase5_backend.py'),acceptance=dict(exact_required=True,tolerance_separate=dict(atol=1e-7,rtol=1e-5),historical_failures_retained=True,scientific_release=False),capture_scope='All live latent states/actions/noise/logp; all guided predictions/loss inputs and raw gradients in B; first32 parameter gradient contributions each transition; full first downblock inputs/output adjoints only transition0',numerical_retry_allowed=False,estimated_new_cluster_bytes=650_000_000,large_C_drive_writes=0)
   assert protocol['guard_sha256']=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
   write(OUT/'protocol.json',protocol)
   wrap='cd '+shlex.quote(str(ROOT))+' && export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && '+str(ROOT/'.venv-editor/bin/python')+' -B scripts/run_numerical_boundary_fixture_20261102.py '+str(OUT)
   args=['sbatch','--parsable','--hold','--no-requeue','--account=allusers','--partition=epyc-gpu','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time=00:03:00','--job-name=poisoning-boundary','--comment=poisoning-targeted-boundary-v1','--output='+str(OUT/'stdout.log'),'--error='+str(OUT/'stderr.log'),'--wrap='+wrap]
   task=dict(task_key=protocol['task'],kind='gpu',run_id=OUT.name,cap_seconds=200,attempt=0,job_id=None,scheduler_state='INTENT',submitted_UTC=now(),numerical_retry_allowed=False)
   s['tasks'].append(task);write(BASE/'state.json',s);write(OUT/'submission_intent.json',dict(task=task,command=args))
   reply=subprocess.run(args,capture_output=True,text=True);write(OUT/'submission_reply.json',dict(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr));reply.check_returncode();job=reply.stdout.strip().split(';')[0];assert job.isdigit();task.update(job_id=job,scheduler_state='PENDING');write(BASE/'state.json',s);write(OUT/'reservation.json',dict(job_id=job,reserved_seconds=200,budget_before=b))
   subprocess.run(['scontrol','release',job],check=True);print(json.dumps(dict(job_id=job,reserved_seconds=200,monitor_chain='unchanged; no timer or completion monitor created')))
if __name__=='__main__':main(sys.argv[1])

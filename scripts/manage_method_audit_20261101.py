"""Canonical-project-only deployment; warm runtime, held bounded Slurm probe."""
import argparse,base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='method_code_audit_20261101T0510Z_v1'
def remote(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-B','-'],input=code.encode(),capture_output=True,timeout=100);print(p.stdout.decode(),end='');print(p.stderr.decode(),end='');p.check_returncode()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu','sync']);a=p.parse_args()
 if a.mode=='prepare':
  sources=['scripts/method_verify_20261101.py','scripts/probe_ip2pix_vjp_20261101.py'];files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in sources}
  inputs=['output/cvpr2027_visuals/source/manuscript_integrated.tex','output/cvpr2027_visuals/source/manuscript_original.tex','output/cvpr2027_visuals/algorithm1.tex','output/cvpr2027_visuals/algorithm1.pdf','output/cvpr2027_visuals/manuscript_preview.pdf','output/rsi_figure1_final_20261031/source/figure_spec.json','output/rsi_figure1_final_20261031/RSI_Figure1_Final.png','output/rsi_figure1_final_20261031/RSI_Figure1_Final.pdf'];materials={Path(n).name:base64.b64encode((ROOT/n).read_bytes()).decode() for n in inputs}
  remote('''import base64,json,subprocess,hashlib,shutil,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+''';assert not run.exists();run.mkdir();(run/'logs').mkdir();(run/'audit').mkdir();(run/'audit/supplied_method_snapshot').mkdir()
files='''+repr(files)+'''
for n,v in files.items():
 p=r/n;assert not p.exists();p.write_bytes(base64.b64decode(v))
materials='''+repr(materials)+'''
for n,v in materials.items():(run/'audit/supplied_method_snapshot'/n).write_bytes(base64.b64decode(v))
(run/'protocol.json').write_text(json.dumps(dict(run_id=run.name,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_objective_unchanged=True,legacy_scientific_loss_unchanged=True,scope='CPU method/label-field/provenance audit; one local VJP diagnostic from preserved failed checkpoint',gpu_cap_seconds=180,global_cap_seconds=3600,target_selection='first differing raw-gradient parameter in registration order from existing failed run, not temporal kernel order',GPU_local_exact_repetitions=3,VJP_acceptance='bitwise identical repeated CUDA VJPs; finite gradients; no tolerance relaxation',FP64_CPU_reference='descriptive analytical g.T@x only; not a new PASS threshold',constant_diagnostic_coefficient=1.,human_review_status='await genuine independent existing submissions',four_arm_image_training='BLOCKED until ordinary full-graph numerical and independent semantic prerequisites pass',no_ASR_or_attack_optimization=True),indent=2))
for n in ['METHOD_IMPLEMENTATION_AUDIT.md','METRIC_AND_NORMALIZATION_AUDIT.md','RL_RSI_CONTROLLED_VALIDATION.md','NEXT_EXPERIMENT_DECISION.md']:
 p=r/'reports'/n
 if p.exists():
  d=run/'audit/preexisting_reports'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
cmd=['sbatch','--parsable','--no-requeue','--partition=epyc-cpu','--account=allusers','--cpus-per-task=2','--mem=8G','--time=00:05:00','--job-name=poisoning-method-cpu','--output='+str(run/'logs/cpu-%j.out'),'--error='+str(run/'logs/cpu-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/method_verify_20261101.py --run-id '+run.name]
p=subprocess.run(cmd,capture_output=True,text=True);(run/'cpu_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();ledger=dict(run_id=run.name,cpu_job=p.stdout.strip().split(';')[0],new_monitor_created=False);(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
''')
 elif a.mode=='gpu':
  probe=base64.b64encode((ROOT/'scripts/probe_ip2pix_vjp_20261101.py').read_bytes()).decode()
  remote('''import sys,json,subprocess,hashlib,fcntl,getpass,shutil,base64
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');sys.path.insert(0,str(r));run=r/'results'/'''+repr(RUN)+''';cpu=json.loads((run/'cpu_verification.json').read_text());assert cpu['status']=='PASS_CPU_VERIFICATION_ONLY' and cpu['GPU_probe_eligible'];assert not (run/'submission_intent.json').exists()
probe=r/'scripts/probe_ip2pix_vjp_20261101.py';content=base64.b64decode('''+repr(probe)+''')
if probe.read_bytes()!=content:
 shutil.copyfile(probe,run/'audit/probe_pre_launch_v0.py');probe.write_bytes(content)
with (r/'reports/next_validation_budget.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);from scripts.submit_small_scale_20261031 import accounting
 old=accounting();raw=subprocess.check_output(['sacct','-X','-nP','-j','87921,88016,88029,88032','--format=JobID,State,ElapsedRaw,ExitCode'],text=True);extra=[]
 for line in raw.splitlines():
  x=line.split('|');assert x[1]=='COMPLETED' and x[3]=='0:0';extra.append(int(x[2]))
 assert len(extra)==4;total=old['total']+sum(extra);budget=json.loads((r/'reports/validation_gpu_reservations.json').read_text());assert total==budget['consumed'] and budget['reserved']==0;assert total+180<=3600
 assert hashlib.sha256((r/'src/rl/phase5_backend.py').read_bytes()).hexdigest()=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
 names=['scripts/probe_ip2pix_vjp_20261101.py','scripts/method_verify_20261101.py','src/rl/diffusion_policy.py','src/rl/diffusion_rl_trainer.py','src/rl/diffusion_trajectory.py','src/rl/phase5_backend.py','configs/env-spec.yaml','configs/phase3_generator_rl.yaml'];manifest={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names};(run/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
 for n in names:
  d=run/'audit/executed_source'/n;d.parent.mkdir(parents=True,exist_ok=True)
  if d.exists():assert d.read_bytes()==(r/n).read_bytes()
  else:shutil.copyfile(r/n,d)
 (run/'budget_before.json').write_text(json.dumps(budget,indent=2));(run/'budget_admission.json').write_text(json.dumps(dict(prior_live_seconds=total,new_cap=180,remaining_after_max=3600-total-180,raw_sacct=raw),indent=2))
 cmd=['sbatch','--parsable','--hold','--no-requeue','--partition=epyc-gpu','--account=allusers','--gres=gpu:tesla_a100_80gb:1','--cpus-per-task=4','--mem=48G','--time=00:03:00','--job-name=poisoning-method-vjp','--comment=poisoning-method:'+run.name,'--output='+str(run/'logs/gpu-%j.out'),'--error='+str(run/'logs/gpu-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/probe_ip2pix_vjp_20261101.py --run-id '+run.name]
 (run/'submission_intent.json').write_text(json.dumps(cmd));p=subprocess.run(cmd,capture_output=True,text=True);(run/'submission_reply.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();job=p.stdout.strip().split(';')[0];ledger=json.loads((run/'job_ledger.json').read_text());ledger.update(gpu_job=job,prior_live_seconds=total,gpu_cap=180);(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));(r/'reports/validation_gpu_reservations.json').write_text(json.dumps(dict(cap=3600,consumed=total,reserved=180,reservations=[ledger]),indent=2));subprocess.run(['scontrol','release',job],check=True);print(json.dumps(ledger))
''')
 elif a.mode=='sync':
  code='''import base64,json
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+''';files={}
for p in run.rglob('*'):
 if p.is_file() and p.suffix in {'.json','.jsonl','.md','.csv','.py','.out','.err','.tex','.log'}:files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
for n in ['METHOD_IMPLEMENTATION_AUDIT.md','METRIC_AND_NORMALIZATION_AUDIT.md','RL_RSI_CONTROLLED_VALIDATION.md','NEXT_EXPERIMENT_DECISION.md','METHOD_AUDIT_STATUS_20261101.json','METHOD_AUDIT_CLOSURE_20261101.json','validation_gpu_reservations.json']:
 p=r/'reports'/n
 if p.exists():files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
for n in ['reports/monitoring/scheduler.json','EXPERIMENT_INDEX.md']:files[n]=base64.b64encode((r/n).read_bytes()).decode()
for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html','state.json','dashboard_data.json']:
 p=r/'dashboard'/n
 if p.exists():files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
files['src/progression/dashboard.py']=base64.b64encode((r/'src/progression/dashboard.py').read_bytes()).decode()
assert sum(map(len,files.values()))<20_000_000;print(json.dumps(files))
'''
  p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-B','-'],input=code.encode(),capture_output=True,timeout=100);p.check_returncode();files=json.loads(p.stdout)
  for n,v in files.items():
   path=(ROOT/n).resolve();assert path.is_relative_to(ROOT.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(v))
  print(json.dumps(dict(files=len(files),bytes=sum(len(base64.b64decode(v)) for v in files.values()),model_weights_copied=False)))

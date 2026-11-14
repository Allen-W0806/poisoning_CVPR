"""Read-only live Slurm/Git/artifact checks; writes only a new status receipt.

No controller cycle, admission, training, smoke test or scheduler mutation is
called. Slurm output is explicitly requested in UTC.
"""
import ast,csv,json,os,re,subprocess,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,sha,put

def read(p,default=None):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except FileNotFoundError:return default
def run(args,tz='UTC'):
 p=subprocess.run(args,cwd=ROOT,env={**os.environ,'TZ':tz},capture_output=True,text=True,timeout=60);p.check_returncode();return dict(command=args,TZ=tz,stdout=p.stdout,stderr=p.stderr)
def ast_without_docs(p):
 tree=ast.parse(Path(p).read_text(encoding='utf-8-sig'))
 for node in ast.walk(tree):
  if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
 return ast.dump(tree,include_attributes=False)
def short(c):return {k:c[k] for k in ['exact','tolerance_pass','affected_tensors','affected_elements','max_abs','relative_L2']}
def main():
 import torch
 torch.set_num_threads(2);started=datetime.now(timezone.utc);stamp=started.strftime('%Y%m%dT%H%M%SZ');out=ROOT/'results'/('live_progress_'+stamp);out.mkdir(exist_ok=False);queries={}
 fields=['JobIDRaw','JobName%80','State','Submit','Start','End','ElapsedRaw','AllocTRES%180','ExitCode','WorkDir%350','Comment%200','SubmitLine%2200']
 # Include the pending monitor whose eligible/start time is later than now.
 queries['sacct_all']=run(['sacct','-X','-nP','-u',os.environ['USER'],'--starttime=2026-11-02T00:00:00','--endtime=2026-11-04T00:00:00','--format='+','.join(fields)])
 queries['anchor_UTC']=run(['sacct','-X','-nP','-j','88433','--format=JobIDRaw,Submit,Start,End'])
 queries['anchor_local_crosscheck']=run(['sacct','-X','-nP','-j','88433','--format=JobIDRaw,Submit,Start,End'],tz='America/New_York')
 queries['squeue']=run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%M|%S|%E|%b'])
 jobs=[]
 keys=['job_id','name','state','submit_UTC','start_UTC','end_UTC','elapsed_seconds','allocTRES','exit_code','workdir','comment','submit_line']
 for line in queries['sacct_all']['stdout'].splitlines():
  parts=line.split('|')
  if len(parts)<12 or not parts[0].isdigit() or int(parts[0])<=88433:continue
  row=dict(zip(keys,parts));project=row['workdir'].rstrip('/')==str(ROOT) or str(ROOT)+'/' in row['submit_line'] or 'cd '+str(ROOT) in row['submit_line']
  if not project:continue
  g=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row['allocTRES']);row['allocated_GPUs']=int(g[1]) if g else 0;row['elapsed_seconds']=int(row['elapsed_seconds']);row['GPU_seconds']=row['allocated_GPUs']*row['elapsed_seconds'];row['state']=row['state'].split()[0].rstrip('+');jobs.append(row)
 active_project=[];excluded=[]
 for line in queries['squeue']['stdout'].splitlines():
  p=line.split('|')
  if not p[0].isdigit():continue
  q=run(['scontrol','show','job','-o',p[0]]);queries['scontrol_'+p[0]]=q
  get=lambda key:next(iter(re.findall(r'(?:^|\s)'+re.escape(key)+r'=([^\s]*)',q['stdout'])),None)
  work,cmd=get('WorkDir'),get('Command');is_project=(work or '').rstrip('/')==str(ROOT) or str(ROOT)+'/' in (cmd or '') or str(ROOT) in (get('SubmitLine') or '') or any(j['job_id']==p[0] for j in jobs)
  row=dict(job_id=p[0],name=p[1],state=p[2],runtime=p[3],start_from_squeue_UTC=p[4],dependency=p[5],TRES_per_node=p[6] if len(p)>6 else None,workdir=work,command=cmd)
  (active_project if is_project else excluded).append(row)
 queries['git_head']=run(['git','rev-parse','HEAD']);queries['git_branch']=run(['git','branch','--show-current']);queries['git_status']=run(['git','status','--short','--untracked-files=normal']);queries['git_tracked_diff_stat']=run(['git','diff','--stat']);queries['git_tracked_numstat']=run(['git','diff','--numstat']);queries['git_loader_diff']=run(['git','diff','--','src/models/pretrained_editor_loader.py'])
 protected=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json');source_checks={n:dict(current_sha256=sha(ROOT/n),matches_preserved_reference=sha(ROOT/n)==h) for n,h in protected.items()}
 guard=ROOT/'src/rl/cuda_execution_guard.py';loader=ROOT/'src/models/pretrained_editor_loader.py';candidate=ROOT/'results/numerical_fix_20261102_v1/candidate_failed_source';candidate_same={name:ast_without_docs(current)==ast_without_docs(candidate/name) for name,current in [('cuda_execution_guard.py',guard),('pretrained_editor_loader.py',loader)]}
 caller=ROOT/'scripts/run_pretrained_benign_pilot_20261102.py';call_lines=[dict(line=i,text=s.strip()) for i,s in enumerate(caller.read_text().splitlines(),1) if 'load_pretrained_editor(' in s];default_false=next(n for n in ast.walk(ast.parse(loader.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='load_pretrained_editor').args.defaults[0].value is False
 tracked=[line for line in queries['git_status']['stdout'].splitlines() if not line.startswith('??')];untracked=[line for line in queries['git_status']['stdout'].splitlines() if line.startswith('??')];put(out/'git_evidence.json',dict(queries={k:v for k,v in queries.items() if k.startswith('git_')},protected=source_checks,candidate_functional_code_same_as_failed=candidate_same,strict_backward_default_false=default_false,current_production_caller_lines=call_lines,guard_sha256=sha(guard),loader_sha256=sha(loader)))
 ver=ROOT/'results/numerical_fix_20261102_v1/verification';a=load(ver/'raw0/update0.pt');b=load(ver/'raw1/update0.pt');checks={}
 checks['raw_pair_initial']=compare(a['before'],b['before']);checks['captured_raw_pair']=compare(a['capture']['raw'],b['capture']['raw']);checks['captured_postclip_pair']=compare(a['capture']['postclip'],b['capture']['postclip']);checks['captured_after']=compare(a['after'],b['after'])
 for phase in ['before','after1','after2']:
  x,y=load(ver/'ordinary0'/phase/'training_state.pt'),load(ver/'ordinary1'/phase/'training_state.pt');checks['ordinary_'+phase]=compare(x,y)
  if phase!='before':
   checks[phase+'_adapter']=compare(x['adapter'],y['adapter'])
   for kind in ['exp_avg','exp_avg_sq']:checks[phase+'_'+kind]=compare({k:v[kind] for k,v in x['optimizer']['state'].items()},{k:v[kind] for k,v in y['optimizer']['state'].items()})
 checks['ordinary_postclip_update2']=compare(load(ver/'ordinary0/postclip.pt'),load(ver/'ordinary1/postclip.pt'));checks['saved_resumed_next_trajectory']=compare(load(ver/'ordinary0/next_trajectory.pt'),load(ver/'resume/next_trajectory.pt'))
 resume_paths=['after2/training_state.pt','postclip.pt','receipt.json','execution_guard.json'];resume={n:dict(exists=(ver/'resume'/n).exists(),mtime_UTC=datetime.fromtimestamp((ver/'resume'/n).stat().st_mtime,timezone.utc).isoformat() if (ver/'resume'/n).exists() else None) for n in resume_paths};process=read(ver/'process_exit.json');latest_summary=read(ver/'mandatory_verification.json')
 records={m:read(ver/m/'receipt.json') for m in ['ordinary0','ordinary1']};norms={m:[dict(step=u['step'],loss=u['loss'],gradient_norm=u['gradient_norm']) for u in r['updates']] for m,r in records.items()}
 put(out/'live_tensor_readback.json',dict(checks=checks,resume_artifacts=resume,process_exit=process,recorded_update_scalars=norms,saved_finalizer_status=latest_summary['status'],latest_finalizer_sha256=sha(ver/'mandatory_verification.json'),comparison_scope='Recomputed on CPU from existing tensors;no training or smoke test',torch=torch.__version__))
 old=read(ROOT/'reports/overnight_supervisor/state.json');night=read(ROOT/'reports/overnight_benign_20261102_v2/state.json');nb=read(ROOT/'reports/overnight_benign_20261102_v2/budget.json');debug=read(ROOT/'reports/numerical_fix_20261102_v1/state.json');db=read(ROOT/'reports/numerical_fix_20261102_v1/budget.json');scheduler=read(ROOT/'reports/monitoring/scheduler.json');heartbeat=read(ROOT/'reports/overnight_supervisor/heartbeat.json');fallback=read(ROOT/'reports/overnight_supervisor/overnight_fallback_health.json')
 expired=time.time()>=night['deadline_epoch'];debug_expired=time.time()>=debug['deadline_epoch'];assert expired and debug_expired
 registered_gpu={j['job_id'] for j in debug['jobs']}|{j['job_id'] for j in night['GPU_jobs']};unregistered=[j['job_id'] for j in jobs if j['allocated_GPUs']>0 and j['job_id'] not in registered_gpu];gpu=[j for j in jobs if j['allocated_GPUs']>0];active_gpu=[j for j in active_project if 'gpu:' in str(j['TRES_per_node']) or 'gres/gpu' in str(j['TRES_per_node'])]
 result=dict(PROJECT_ID='RSI-POISONING',checked_UTC=datetime.now(timezone.utc).isoformat(),started_UTC=started.isoformat(),evidence_directory=str(out.relative_to(ROOT)),jobs_submitted_after88433=jobs,active_project=active_project,excluded_active_other_projects=excluded,project_GPU_counts=dict(Counter(j['state'] for j in gpu)),project_GPU_seconds_after88433=sum(j['GPU_seconds'] for j in gpu),unregistered_project_GPU_jobs=unregistered,protected_source_checks=source_checks,candidate_functional_same_as_failed=candidate_same,strict_backward_default_false=default_false,caller_lines=call_lines,Git_HEAD=queries['git_head']['stdout'].strip(),git_tracked_changes=tracked,git_untracked_entries=untracked,checks={k:short(v) for k,v in checks.items()},resume_artifacts=resume,process_exit=process,update_scalars=norms,overnight=dict(state=night['status'],admissions_closed=night['admissions_closed'],budget=nb,authorization_expired=expired),urgent_debug=dict(state=debug['status'],admissions_closed=debug['admissions_closed'],budget=db,authorization_expired=debug_expired),legacy_validation_budget=old.get('GPU_budget'),scheduler=scheduler,last_heartbeat=heartbeat,last_fallback_health=fallback,active_project_GPU_jobs=active_gpu,currently_spendable_GPU_seconds=0,verified_production_numerical_fix=False,overall_numerical_status='ROOT_CAUSE_PARTIALLY_LOCALIZED')
 put(out/'live_snapshot.json',result);put(out/'raw_readonly_queries.json',queries)
 table='| Job | Name | Type | State / exit | Submit UTC | Start UTC | End UTC | Elapsed s | GPU s |\n|---|---|---|---|---|---|---|---:|---:|\n'
 for j in jobs:table+=f"|{j['job_id']}|{j['name']}|{'GPU diagnostic' if j['allocated_GPUs'] else 'CPU/monitor'}|{j['state']} / {j['exit_code']}|{j['submit_UTC']}|{j['start_UTC']}|{j['end_UTC']}|{j['elapsed_seconds']}|{j['GPU_seconds']}|\n"
 detail='| Live CPU re-readback | Exact | Tolerance | Differing elements | Max absolute | Relative L2 |\n|---|---|---|---:|---:|---:|\n'
 for k in ['captured_raw_pair','captured_postclip_pair','ordinary_before','after1_adapter','after1_exp_avg','after1_exp_avg_sq','after2_adapter','ordinary_postclip_update2','saved_resumed_next_trajectory']:
  c=checks[k];detail+=f"|{k}|{c['exact']}|{c['tolerance_pass']}|{c['affected_elements']:,}|{c['max_abs']:.12g}|{c['relative_L2']:.12g}|\n"
 legacy=old.get('GPU_budget',{});tracked_text='\n'.join('- `'+x+'`' for x in tracked);excluded_text='\n'.join('- '+r['job_id']+' / '+r['name']+' / '+str(r['command']) for r in excluded)
 report=f'''# Live RSI-Poisoning execution status

PROJECT_ID: RSI-POISONING. **Live checked UTC: {result['checked_UTC']}**. Canonical root: `/storage/home/ydk297/projects/poisoning`.

**No RSI-Poisoning GPU job is currently running or pending** ({len(active_gpu)} found). Current numerical status: **ROOT_CAUSE_PARTIALLY_LOCALIZED; no verified production fix**. The10-hour authorization expired at **{night['deadline_UTC']}**. All unused allowances are accounting balances,not current authorization. This check submitted/restarted no jobs and changed no controller/budget state.

## Live Slurm results after88433

Queried all user allocation records since2026-11-02,then selected numeric job IDs>88433 with canonical project WorkDir or SubmitLine. Also inspected current squeue jobs with scontrol;excluded other projects by command/workdir. Job88433 is the reference anchor and is not charged again. All Slurm times below were requested with TZ=UTC;the raw anchor was also queried in America/New_York to cross-check timezone conversion. Allocation rows only (`sacct -X`),no batch/extern double counting.

{table}

Post88433 project GPU allocations: {len(gpu)},states{json.dumps(result['project_GPU_counts'])},actual allocated GPU-seconds{result['project_GPU_seconds_after88433']}. These GPU jobs are numerical debugging/verification fixtures containing optimizer updates,not newly trained benign-efficacy arms. No later project GPU allocation was found. Unregistered project GPU IDs: {unregistered}.

Other active user jobs excluded from RSI-Poisoning accounting:

{excluded_text}

These other projects were read only at scheduler-metadata level;their jobs were not modified,canceled or charged here.

## Current code and implementation

Git HEAD `{result['Git_HEAD']}`. Tracked working-tree changes currently comprise{len(tracked)} paths:

{tracked_text}

Git dirtiness is not proof of a numerical fix. All8 preserved model-policy/trainer/trajectory/phase5-guard/config reference hashes still match;the core backward/loss implementation has not been corrected since the preserved failing numerical reference. The current loader has `strict_backward=False` by default and an optional call to `configure_strict_diffusion_backward`;the helper only sets CUDA_LAUNCH_BLOCKING1/checks initialization and explicitly calls itself experimental after ordinary verification failed. AST comparison excluding docstrings shows current helper/loader functionality matches the failed candidate: {candidate_same}.

Current actual benign training caller loads the editor as:

{chr(10).join('- line'+str(x['line'])+': `'+x['text']+'`' for x in call_lines)}

It does not opt in. The explicit opt-in exists in the diagnostic verification harness. No new verified fix or production enablement is present. Untracked entries and exact diffs/current hashes are retained in `git_evidence.json`;archived pre-fix and failed-candidate code remains in the numerical-debug evidence directory. This report-generation script is inspection tooling,not a numerical repair.

## Latest ordinary-path and resume evidence, re-read now

Latest relevant GPU verification is88504,Slurm FAILED,exit124:0,340 GPU-seconds. Its process-exit receipt records the bounded timeout at{process.get('finished_UTC')}. CPU comparisons below were recomputed in this live check from the existing .pt files;no model/smoke/training experiment was rerun.

{detail}

Ordinary recorded loss/norm by update:

{chr(10).join('- '+m+': '+json.dumps(rows) for m,rows in norms.items())}

The captured raw pair passes,but the ordinary uninstrumented first and second updates fail. Candidate acceptance therefore fails regardless of capture-enabled results. Some tiny moment differences pass tolerance while failing exact equality;they are not relabeled bitwise equal. Resume actually restored a checkpoint and saved an exact92-tensor next trajectory. Current existence checks still show no `resume/after2/training_state.pt`,no completed update receipt,no postclip and no final guard receipt: **next optimizer update after resume remains NOT VERIFIED/TIMEOUT**. No newer project GPU run was found to supersede this result.

## Controllers, monitor and authorization

- Overnight run `overnight_benign_20261102_v2`: {night['status']};admissions_closed={night['admissions_closed']};init{night['initialized_UTC']};deadline{night['deadline_UTC']};EXPIRED. Its sealed ledger records172/36000 GPU-seconds and35828 unused. Do not interpret that stale unused balance as permission.
- Separate urgent-debug run `numerical_fix_20261102_v1`: {debug['status']};admissions_closed={debug['admissions_closed']};852/960 GPU-seconds,reserved0,108 unused;deadline{debug['deadline_UTC']};EXPIRED. Shared authorization-envelope spend is1024=172+852;unspent34976 is also expired. Neither finalized controller was reopened.
- Older validation ledger remains separate: cap{legacy.get('cap')},consumed{legacy.get('consumed')},reserved{legacy.get('reserved')},unused{legacy.get('remaining')}. Its last action is{old.get('next_action')};this read-only request grants no new GPU allowance.
- Existing two-hour monitor: enabled={scheduler['enabled']},installed={scheduler['installed']},interval{scheduler['interval_hours']}h;last tick **{scheduler['last_tick_job_id']}**,last successful check **{scheduler['last_success_utc']}**;registered successor **{scheduler['next_job_id']}**,due **{scheduler['next_due_utc']}**. Current squeue confirms one matching monitor. No duplicate recurring monitor was installed. Latest fallback read the finalized controller and returned TERMINAL_NOOP at{fallback.get('checked_UTC')}.

**Currently spendable GPU budget:0 under the expired/closed authorizations.** No silent extension,restart or new admission was made.

## Blockers and next eligible action

The responsible CUDA primitive is still unresolved. Selected-module upstream adjoint divergence was localized,but launch-blocking failed ordinary verification;checkpoint-next-update proof is missing. Scientific reward release and independent development-evaluation admission also remain unresolved;160 prospective final sources remain sealed. No new learned-model efficacy test is eligible;ASR NOT TESTED.

The next eligible work is CPU-only analysis/preparation of minimal whole-UNet prediction,loss-input and output-adjoint witnesses and a single-factor pre-import versus late-bootstrap launch test. Any new GPU diagnostic needs an explicitly renewed budget/window and preregistered discriminating hypothesis;do not reuse the expired ledger. After a genuine repair,ordinary uninstrumented/raw/postclip/Adam/adapter/ownership and completed resume-next-update checks must all pass before a matched Frozen-versus-DDPO benign learning test can be admitted.

Evidence for this live check: `{out.relative_to(ROOT)}/live_snapshot.json`, `raw_readonly_queries.json`, `git_evidence.json`, `live_tensor_readback.json`. Slurm and artifact observations are current at the stated timestamp;Drive upload/readback verification is recorded separately. No raw tensors,images,weights,prompts,source or logs are uploaded with this aggregate report.
'''
 report=re.sub(r'([,;])(?=[A-Za-z0-9`])',r'\1 ',report)
 report=re.sub(r'\b(The|All|after|since|comprise|is|exact|records|and|unused|cap|consumed|reserved|at|line|states)(?=\d)',r'\1 ',report)
 report=report.replace('CUDA_LAUNCH_BLOCKING1','CUDA_LAUNCH_BLOCKING=1').replace('after88433','after 88433').replace('IDs>88433','IDs > 88433')
 report=report.replace('Queried all user allocation records since 2026-11-02,','Queried all user allocation records since 2026-11-02 with an end bound of 2026-11-04 to include future-eligible pending jobs,')
 (ROOT/'reports/LIVE_PROGRESS_LATEST.md').write_text(report,encoding='utf8');put(out/'report_manifest.json',dict(report='reports/LIVE_PROGRESS_LATEST.md',sha256=sha(ROOT/'reports/LIVE_PROGRESS_LATEST.md'),checked_UTC=result['checked_UTC'],no_jobs_submitted=True,no_controller_restart=True));print(json.dumps(dict(checked_UTC=result['checked_UTC'],evidence=str(out.relative_to(ROOT)),post_anchor_jobs=len(jobs),project_GPU_states=result['project_GPU_counts'],active_project=active_project,excluded_active=excluded,numerical_fix=False,budget_expired=True,next_monitor=scheduler['next_job_id'],report_sha256=sha(ROOT/'reports/LIVE_PROGRESS_LATEST.md'))))
if __name__=='__main__':main()

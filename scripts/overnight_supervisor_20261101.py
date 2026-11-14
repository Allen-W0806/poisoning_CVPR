"""Stateful extension of the EXISTING two-hour Slurm callback, never a timer.

Only two executable tasks are admitted: changed-evidence CPU audit and one
predeclared constant-coefficient O/I/I/O numerical diagnostic. No scientific
reward is released. Completed numerical failures never trigger rerun-until-PASS.
"""
import csv,hashlib,html,io,json,os,re,shlex,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.monitoring.common import FileLock
BASE=ROOT/'reports/overnight_supervisor';CAMPAIGN='overnight_supervisor_20261101_v1'
KIT='results/next_validation_20261101T0040Z_v1/semantic_review32'
GUARD='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
REPORT_NAMES=['CURRENT_STAGE.md','EXPERIMENT_SUMMARY.md','NEXT_ACTIONS.md','EXPERIMENT_AUDIT_SUMMARY.md','RESULTS_SUMMARY.csv','GPU_COST_SUMMARY.csv','STATUS.json','LATEST_REPORT_INDEX.md']
REUSE=['results/audit_remediation_20261101T0120Z_v1/cpu_audit/'+n+'.json' for n in ['feedback_distribution','matched_feedback','qwen_statistical_conclusion','metric_availability']]

def now():return datetime.now(timezone.utc).isoformat()
def read(p,d=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return d
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)
def text(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial');q.write_text(s,encoding='utf-8');q.replace(p)
def command(args,timeout=40):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=timeout);return p
def history(path,v):
 with Path(path).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def protected():return read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json')

def component_statuses(root=ROOT):
 """Claim-specific status does not turn paper NO-GO into a CPU/engineering ban."""
 report=read(Path(root)/'reports/SCIENTIFIC_GATE_STATUS.json',{})
 return report.get('statuses',{})
def fingerprint():
 paths=[p for p in (ROOT/KIT).rglob('*') if p.is_file()]+[ROOT/n for n in REUSE]+[ROOT/'scripts/overnight_evidence_cpu_20261101.py']
 d={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),d

def accounting(state):
 from scripts.submit_small_scale_20261031 import accounting as prior_accounting
 prior=prior_accounting();ids={'87921','88016','88029','88032','88084'}|{x['job_id'] for x in state['tasks'] if x['kind']=='gpu' and x.get('job_id')}
 p=command(['sacct','-X','-nP','-j',','.join(sorted(ids)),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,TimelimitRaw']);p.check_returncode();rows=[];consumed=prior['total'];reserved=0
 for line in p.stdout.splitlines():
  v=line.split('|')
  if len(v)<6 or v[0] not in ids:continue
  g=int((re.search(r'gres/gpu=(\d+)',v[3]) or [None,'0'])[1]);elapsed=int(v[2]);st=v[1].split()[0].rstrip('+');used=g*elapsed
  terminal=st in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'};remaining=0 if terminal else max(0,int(v[5])*60-elapsed)*g
  if st=='PENDING':remaining=next(x['cap_seconds'] for x in state['tasks'] if x.get('job_id')==v[0] and x['kind']=='gpu')
  consumed+=used;reserved+=remaining;rows.append(dict(job_id=v[0],state=st,gpu_seconds=used,reserved_seconds=remaining,exit_code=v[4]))
 assert {x['job_id'] for x in rows}==ids
 # Detect unknown newer canonical-project GPU allocations rather than assume a stale balance.
 q=command(['sacct','-X','-nP','-u',os.environ['USER'],'--starttime','2026-11-01T01:10:37','--format=JobID,State,ElapsedRaw,AllocTRES,WorkDir%240']);q.check_returncode();unknown=[]
 # The explicitly authorized10-hour overnight campaign owns a NEW budget.
 # Recognize its registered allocations without adding them to this3600s ledger.
 separate=read(ROOT/'reports/overnight_benign_20261102_v2/state.json',{})
 separate_ids={t['job_id'] for t in separate.get('GPU_jobs',[]) if t.get('job_id')} if separate.get('run_id')=='overnight_benign_20261102_v2' else set()
 debug=read(ROOT/'reports/numerical_fix_20261102_v1/state.json',{});debug_ids={t['job_id'] for t in debug.get('jobs',[]) if t.get('job_id')} if debug.get('run_id')=='numerical_fix_20261102_v1' else set()
 for line in q.stdout.splitlines():
  v=line.split('|')
  if len(v)>=5 and v[4].rstrip('/')==str(ROOT) and re.search(r'gres/gpu=[1-9]',v[3]) and v[0] not in ids and v[0] not in {r['job_id'] for r in prior['rows']} and v[0] not in separate_ids|debug_ids:unknown.append(v[:5])
 return dict(cap=3600,consumed=consumed,reserved=reserved,remaining=3600-consumed-reserved,scope='validation subtotal; includes failed and diagnostic allocations; not project lifetime',rows=prior['rows']+rows,unreconciled_project_GPU_jobs=unknown,separately_authorized_overnight_GPU_jobs=sorted(separate_ids),separate_overnight_ledger='reports/overnight_benign_20261102_v2/budget.json',separately_authorized_urgent_debug_GPU_jobs=sorted(debug_ids),urgent_debug_ledger='reports/numerical_fix_20261102_v1/budget.json')

def inspect_tasks(state):
 ids=[x['job_id'] for x in state['tasks'] if x.get('job_id')]
 if not ids:return []
 p=command(['sacct','-X','-nP','-j',','.join(ids),'--format=JobID,State,Start,End,ElapsedRaw,ExitCode']);p.check_returncode();by={v[0]:v for line in p.stdout.splitlines() if len(v:=line.split('|'))>=6};new=[]
 for task in state['tasks']:
  if not task.get('job_id'):raise RuntimeError('Uncertain submission: reconcile receipt before any new task')
  v=by.get(task['job_id']);task['scheduler_state']=v[1].split()[0].rstrip('+') if v else 'UNCONFIRMED'
  if not v:continue
  task.update(start_cluster_local=v[2],end_cluster_local=v[3],elapsed_seconds=int(v[4]),exit_code=v[5]);out=ROOT/'results'/task['run_id'];task['result']=read(out/'checks.json',{})
  task['watchdog_findings']=[]
  if task['scheduler_state']=='RUNNING' and task['elapsed_seconds']>task['cap_seconds']+60:task['watchdog_findings'].append('STALLED_OR_SLURM_WALLTIME_ENFORCEMENT_OVERDUE; inspect scheduler, never submit a replacement concurrently')
  if task['scheduler_state'] not in {'PENDING','RUNNING','CONFIGURING','COMPLETING'} and task.get('processed_job_id')!=task['job_id']:
   task['processed_job_id']=task['job_id'];new.append(task['job_id']);state['last_processed_Slurm_job_id']=task['job_id'];state['last_completed_task']=task['task_key']
   if task['scheduler_state']=='COMPLETED' and task['result'].get('status')=='PASS':task['classification']='ENGINEERING_PASS_SCOPED'
   elif task['result'].get('diagnostic_complete'):task['classification']='NUMERICAL_MISMATCH_PRESERVED'
   elif task['scheduler_state'] in {'NODE_FAIL','PREEMPTED','BOOT_FAIL'}:task['classification']='RECOVERABLE_TRANSIENT_INFRASTRUCTURE'
   else:task['classification']='FAILED_REQUIRES_DIAGNOSIS'
   if task['classification']!='ENGINEERING_PASS_SCOPED':
    logs=''.join((p.read_text(errors='replace')[-4000:] if p.exists() else '') for p in [out/'stderr.log',out/'stdout.log']);fp=hashlib.sha256((task['scheduler_state']+task['exit_code']+logs).encode()).hexdigest();task['failure_fingerprint']=fp;state['failure_fingerprints'][fp]=dict(job_id=task['job_id'],classification=task['classification'],run_id=task['run_id'],numerical_retry_allowed=False)
  # Reconsider output integrity every cycle using size/mtime-bound hash caching.
  if task.get('processed_job_id'):
   inv=read(out/'artifact_inventory.json',{});bad=[]
   for x in inv.get('files',[]):
    p=out/x['path'];cache=state.setdefault('integrity_cache',{});key=str(p);sig=[p.stat().st_size,p.stat().st_mtime_ns] if p.exists() else None
    if cache.get(key,{}).get('signature')!=sig or key not in cache:
     ok=p.is_file() and sha(p)==x['sha256'];cache[key]=dict(signature=sig,verified=ok)
    if not cache[key]['verified']:bad.append(x['path'])
   task['output_integrity']='FAIL' if bad else ('PASS' if inv else 'NO_INVENTORY');task['bad_outputs']=bad
 return new

def submit(state,key,kind,fp,inputs,attempt=0):
 out=ROOT/'results'/f'{CAMPAIGN}_{key}_a{attempt}';out.mkdir(exist_ok=False);cap=300 if kind=='gpu' else 120
 if kind=='gpu':
  parent=ROOT/'results/next_validation_20261101T0040Z_v1_BF16_instrumented/diagnostic';protocol=dict(task=key,order=['ordinary','instrumented','instrumented','ordinary'],protected_source=protected(),parent_checkpoint_directory=parent.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(parent/'before_next/training_state.pt'),rollout_sha256=sha(parent/'reference_rollout.pt'),cap_seconds=cap,coefficient=1.,bitwise_equality_required=True,acceptance_frozen_before_execution=True,scientific_reward_release=False,deviations=['B2/T2 vs historical B10/T10','single shared saved checkpoint and fresh process per worker','state and forward outputs saved at phase boundaries; generation wall timing excluded from equality by definition','instrumented mode has historical preclip CPU copies; ordinary mode has no added hooks'],ordinary_path_release_rule='This single matched-hardware pair cannot erase preserved historical failures or certify scientific batch10 training')
  script='scripts/overnight_bf16_pair_20261101.py'
 elif key=='P1_FORENSICS_v1':
  protocol=dict(task=key,parent_run=next(t['run_id'] for t in reversed(state['tasks']) if t['kind']=='gpu'),cap_seconds=cap,new_GPU_seconds=0);script='scripts/overnight_bf16_forensics_cpu_20261101.py'
 else:protocol=dict(task=key,inputs=inputs,kit=KIT,reuse_evidence=REUSE,cap_seconds=cap);script='scripts/overnight_evidence_cpu_20261101.py'
 write(out/'protocol.json',protocol);args=['sbatch','--parsable','--hold','--no-requeue','--account=allusers','--partition='+('epyc-gpu' if kind=='gpu' else 'epyc-cpu'),'--cpus-per-task='+('4' if kind=='gpu' else '2'),'--mem='+('48G' if kind=='gpu' else '8G'),'--time=00:%02d:00'%(cap//60),'--job-name=poisoning-overnight-'+kind,'--comment=poisoning-overnight:'+key+':a'+str(attempt),'--output='+str(out/'stdout.log'),'--error='+str(out/'stderr.log')]
 if kind=='gpu':args+=['--gres=gpu:tesla_a100_80gb:1']
 args+=['--wrap=cd '+shlex.quote(str(ROOT))+' && export OMP_NUM_THREADS='+('4' if kind=='gpu' else '2')+' OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && '+shlex.quote(str(ROOT/'.venv-editor/bin/python'))+' -B '+script+' '+shlex.quote(str(out))]
 record=dict(task_key=key,kind=kind,run_id=out.name,evidence_fingerprint=fp,attempt=attempt,cap_seconds=cap,job_id=None,submitted_UTC=now(),command=args);state['tasks'].append(record);write(BASE/'state.json',state);write(out/'submission_intent.json',record)
 p=command(args);write(out/'submission_reply.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));p.check_returncode();job=p.stdout.strip().split(';')[0];assert job.isdigit();record['job_id']=job;write(BASE/'state.json',state)
 # Register the completion check before releasing the held allocation.
 event=command(['sbatch','--parsable','--partition=epyc-cpu','--account=allusers','--cpus-per-task=1','--mem=2G','--time=00:05:00','--job-name=poisoning-overnight-event','--dependency=afterany:'+job,'--output='+str(out/'event-%j.out'),'--error='+str(out/'event-%j.err'),'--wrap=cd '+shlex.quote(str(ROOT))+' && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '+str(ROOT/'.venv/bin/python')+' -B scripts/overnight_supervisor_20261101.py event']);event.check_returncode();record['completion_CPU_job']=event.stdout.strip().split(';')[0];write(BASE/'state.json',state);command(['scontrol','release',job]).check_returncode();return job

def dashboard(root):
 root=Path(root);s=read(root/'reports/overnight_supervisor/heartbeat.json')
 if not s:return
 data=read(root/'dashboard/dashboard_data.json',{});data['overnight_supervisor']=s;encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']:
  p=root/'dashboard'/n
  if not p.exists():continue
  zh='_zh' in n or 'Latest' in n;b=s['budget'];block='<aside id="overnight-supervisor" style="padding:20px;background:#dcecf8;color:#152d48;overflow-wrap:anywhere"><h2>'+('夜间监督：真实调度与门槛' if zh else 'Overnight supervisor: actual scheduler and gates')+'</h2><p>'+html.escape(s['checked_UTC'])+' · Slurm '+str(s['supervisor_job_id'])+' · '+s['stage']+'</p><p>GPU '+str(b['consumed'])+'/3600s; reserved '+str(b['reserved'])+'s; remaining '+str(b['remaining'])+'s. Scientific NO-GO.</p><p>'+html.escape(s['next_action']['reason'])+'</p><p>Drive: '+html.escape(s['cloud']['status'])+'. '+('较早卡片为历史阶段快照。' if zh else 'Earlier cards are historical stage snapshots.')+'</p><p><a href="../chatgpt_reports/CURRENT_STAGE.md">Current stage</a> · <a href="../chatgpt_reports/LATEST_REPORT_INDEX.md">Report index</a> · <a href="../reports/overnight_supervisor/heartbeat.json">Heartbeat</a></p></aside>'
  t=p.read_text();t=re.sub(r'<!--OVERNIGHT_START-->.*?<!--OVERNIGHT_END-->','',t,flags=re.S);assert '<body' in t;t=re.sub(r'(<body[^>]*>)',lambda m:m[1]+'<!--OVERNIGHT_START-->'+block+'<!--OVERNIGHT_END-->',t,count=1);t=re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)',lambda m:m[1]+encoded+m[2],t,flags=re.S);text(p,t)
 write(root/'dashboard/state.json',data);write(root/'dashboard/dashboard_data.json',data)
 from scripts.render_execution_remediation_dashboard_20261102 import dashboard as execution_dashboard
 execution_dashboard(root)
 overnight=read(Path(root)/'reports/overnight_benign_20261102_v2/state.json')
 if overnight:
  from scripts.overnight_benign_reporting_20261102 import dashboard_only
  dashboard_only(overnight,Path(root))

def reports(state,receipt):
 package=ROOT/'chatgpt_reports';package.mkdir(exist_ok=True);header='# RSI-POISONING\n\nSnapshot '+receipt['checked_UTC']+'; supervisor Slurm '+str(receipt['supervisor_job_id'])+'.\n\n';b=receipt['budget'];jobs='\n'.join('- '+x['task_key']+' / Slurm '+x['job_id']+' / '+x.get('scheduler_state','SUBMITTED')+' / '+x.get('classification','await scheduler evidence') for x in state['tasks']) or 'None.'
 task_results=[dict(task=x['task_key'],run_id=x['run_id'],job_id=x['job_id'],state=x.get('scheduler_state','SUBMITTED'),checks=x.get('result',{}),summary=read(ROOT/'results'/x['run_id']/'summary.json',{})) for x in state['tasks']]
 body=header+'Stage: '+receipt['stage']+'. Scientific **NO-GO**.\n\n'+jobs+'\n\nCompleted since previous check: '+str(receipt['completed_since_previous'])+'.\n\nNext action: '+receipt['next_action']['task']+'; '+receipt['next_action']['reason']+'\n\nValidation GPU consumed '+str(b['consumed'])+'/3600s; reserved '+str(b['reserved'])+'s; remaining '+str(b['remaining'])+'s. Not project lifetime.\n\nExisting successor '+str(receipt['scheduler'].get('next_job_id'))+'; next eligibility '+str(receipt['scheduler'].get('next_due_utc'))+'. Last successful scheduler tick '+str(receipt['scheduler'].get('last_success_utc'))+'.\n\nCloud: '+receipt['cloud']['status']+'. Last verified cloud synchronization: '+str(state.get('last_successful_Google_Drive_synchronization'))+'. Cluster synchronization requires a valid existing googledrive grant.\n\n'
 text(package/'CURRENT_STAGE.md',body)
 text(package/'EXPERIMENT_SUMMARY.md',body+'Engineering and scientific results are separate. Local VJP PASS72 does not fix ordinary full-graph BF16 replay. Historical preintervention adapters and optimizer states differ. Semantic reviews remain BLOCKED unless genuine exported records validate. No ASR or poisoning reward is released.\n\n'+json.dumps(task_results,indent=2)+'\n')
 text(package/'NEXT_ACTIONS.md',header+json.dumps(dict(next=receipt['next_action'],dependencies=state['blocked_dependencies'],repairs=state['previously_attempted_repairs']),indent=2)+'\n\nNo repeat of completed scientific tasks; reconsider blocked tasks only if their evidence fingerprint changes. If no eligible task remains, inspect every7200 seconds with zero GPU allocation.\n')
 text(package/'EXPERIMENT_AUDIT_SUMMARY.md',header+'Unchanged historical feedback:230/500 at lower bound, S=.5A on500/500, constant U1/C1/3;70.625% matched candidate ties. No-op and unchanged labels do not establish instruction compliance. Historical LPIPS was missing; current availability is reported in SCIENTIFIC_GATE_STATUS.md. Clean Qwen paired confidence intervals include0; historically exposed development sources cannot become a new untouched test.\n\n'+json.dumps(dict(code_guard_unchanged=receipt['guard_intact'],failures=state['failure_fingerprints'],results=task_results),indent=2)+'\n')
 text(package/'RESULTS_SUMMARY.csv','PROJECT_ID,task,job_id,state,scope\n'+''.join('RSI-POISONING,'+x['task_key']+','+x['job_id']+','+x.get('scheduler_state','SUBMITTED')+',engineering_only\n' for x in state['tasks']))
 text(package/'GPU_COST_SUMMARY.csv','PROJECT_ID,cap_seconds,consumed_seconds,reserved_seconds,remaining_seconds\nRSI-POISONING,3600,'+','.join(str(b[k]) for k in ['consumed','reserved','remaining'])+'\n')
 write(package/'STATUS.json',dict(PROJECT_ID='RSI-POISONING',updated_UTC=receipt['checked_UTC'],stage=receipt['stage'],scientific_gate='NO-GO',budget={k:b[k] for k in ['cap','consumed','reserved','remaining']},last_cloud_verified=state.get('last_successful_Google_Drive_synchronization'),cloud_verification='PENDING_FOR_THIS_PUBLICATION',next_action=receipt['next_action']))
 hashes={p.name:sha(p) for p in package.iterdir() if p.name in REPORT_NAMES and p.name!='LATEST_REPORT_INDEX.md'}
 # Both publishers keep the connector-verified report index. Its digest enters
 # the version ID, so adoption cannot mutate an older immutable outbox version.
 external=ROOT/'reports/LATEST_REPORT_INDEX.md';verified=read(ROOT/'results/overnight_benign_20261102_v2/cloud_index_verification.json',{})
 if verified.get('status')=='CLOUD_VERIFIED' and (verified.get('PROJECT_ID')!='RSI-POISONING' or not external.is_file() or sha(external)!=verified.get('index_sha256')):raise RuntimeError('Verified external index missing/changed; preserve cloud latest instead of reverting')
 use_external=external.is_file() and verified.get('PROJECT_ID')=='RSI-POISONING' and verified.get('status')=='CLOUD_VERIFIED' and sha(external)==verified.get('index_sha256')
 version_basis={**hashes,**({'verified_external_index_sha256':verified['index_sha256']} if use_external else {})};version=hashlib.sha256(json.dumps(version_basis,sort_keys=True).encode()).hexdigest()[:24]
 index=external.read_text(encoding='utf8') if use_external else header+'Publication '+version+'. Reports are aggregate text only; no datasets, images, checkpoints or credentials.\n\n'+''.join('- '+n+' SHA256 '+h+'\n' for n,h in sorted(hashes.items()))+'\nLocal reports: chatgpt_reports/. Cloud text reports: existing RSI-Poisoning/reports/; STATUS and index at project folder root. This index is published last. Upload success alone is not cloud verification.\n'
 text(package/'LATEST_REPORT_INDEX.md',index)
 out=BASE/'outbox'/version;out.mkdir(parents=True,exist_ok=True)
 for n in REPORT_NAMES:
  p=package/n;blob=p.read_bytes();assert len(blob)<=128*1024 and b'RSI-POISONING' in blob and not re.search(rb'-----BEGIN .*PRIVATE KEY-----|(?i:access_token|refresh_token|client_secret)\s*[=:]',blob);(out/n).write_bytes(blob)
 write(out/'manifest.json',dict(version=version,files={n:sha(out/n) for n in REPORT_NAMES},folder_id='1MmOBPXEkKGdU03FcvOtiZuHcVYLfO8j9'))
 return out

def cloud_sync(state,out):
 conf=Path.home()/'.config/rclone/rclone.conf';key=str(conf.stat().st_mtime_ns) if conf.exists() else 'MISSING';cloud=state.get('cloud',{})
 if cloud.get('auth_fingerprint')==key and cloud.get('status')=='BLOCKED_INVALID_GRANT':return {**cloud,'outbox':str(out.relative_to(ROOT)),'last_attempt_skipped':'unchanged invalid grant; awaiting renewed authorization'}
 binary=ROOT/'tools/rclone/rclone';flags=['--drive-root-folder-id','1MmOBPXEkKGdU03FcvOtiZuHcVYLfO8j9','--retries','1','--low-level-retries','1','--contimeout','5s','--timeout','15s','--log-level','ERROR']
 try:
  probe=command([binary,'lsjson','googledrive:',*flags,'--max-depth','1'],timeout=30)
  if probe.returncode:
   reason='invalid_grant' if 'invalid_grant' in probe.stderr else 'Drive access failed; see redacted local attempt'
   result=dict(status='BLOCKED_INVALID_GRANT' if reason=='invalid_grant' else 'BLOCKED_CLOUD_ACCESS',auth_fingerprint=key,checked_UTC=now(),error=reason,outbox=str(out.relative_to(ROOT)),no_sensitive_upload=True);write(BASE/'cloud_last_attempt.json',result);return result
  manifest=read(out/'manifest.json');verified=[]
  for n in REPORT_NAMES: # Commit index last.
   dest='googledrive:'+('' if n in {'LATEST_REPORT_INDEX.md','STATUS.json'} else 'reports/')+n
   p=command([binary,'copyto',out/n,dest,*flags,'--checksum'],timeout=30);p.check_returncode();back=command([binary,'cat',dest,*flags],timeout=30);back.check_returncode();assert hashlib.sha256(back.stdout.encode()).hexdigest()==manifest['files'][n];verified.append(n)
  result=dict(status='PASS_REMOTE_READBACK',checked_UTC=now(),publication=manifest['version'],verified_files=verified,auth_fingerprint=key,outbox=str(out.relative_to(ROOT)));state['last_successful_Google_Drive_synchronization']=result['checked_UTC'];write(BASE/'cloud_last_success.json',result);return result
 except Exception as e:return dict(status='BLOCKED_CLOUD_ACCESS',checked_UTC=now(),error=type(e).__name__,auth_fingerprint=key,outbox=str(out.relative_to(ROOT)))

def cycle():
 BASE.mkdir(parents=True,exist_ok=True)
 with FileLock(BASE/'monitor.lock') as acquired:
  if not acquired:return dict(status='SKIPPED_LOCKED',checked_UTC=now())
  state=read(BASE/'state.json')
  # Existing two-hour chain is the only health fallback; no new polling timer.
  pointer=ROOT/'reports/OVERNIGHT_CONTROLLER_POINTER.json'
  if pointer.exists():
   try:
    p=command([ROOT/'.venv-editor/bin/python','-B',ROOT/'scripts/overnight_benign_controller_20261102.py','health'],timeout=90)
    write(BASE/'overnight_fallback_health.json',dict(checked_UTC=now(),returncode=p.returncode,stdout=p.stdout[-2000:],stderr=p.stderr[-2000:]))
   except Exception as e:write(BASE/'overnight_fallback_health.json',dict(checked_UTC=now(),status='FALLBACK_ERROR_ISOLATED',error=repr(e)))
  if state is None:
   state=dict(campaign=CAMPAIGN,current_stage='INSPECT',tasks=[],last_completed_task=None,last_processed_Slurm_job_id=None,previously_attempted_repairs=[],failure_fingerprints={},completed_dependencies={},blocked_dependencies={},last_successful_Google_Drive_synchronization=None,cloud=dict(status='NOT_ATTEMPTED'),created_UTC=now())
  phases=['INSPECT','CLASSIFY','DIAGNOSE','VERIFY'];state['current_stage']='INSPECT';write(BASE/'state.json',state)
  state['component_statuses']=component_statuses()
  for n in ['METHOD_IMPLEMENTATION_AUDIT.md','METRIC_AND_NORMALIZATION_AUDIT.md','METHOD_AUDIT_STATUS_20261101.json','METHOD_AUDIT_CLOSURE_20261101.json','AUDIT_DECISION_MATRIX.md','NEXT_VALIDATION_DECISION_20261031.md']:assert (ROOT/'reports'/n).is_file()
  guard=sha(ROOT/'src/rl/phase5_backend.py')==GUARD;unchanged=all(sha(ROOT/n)==h for n,h in protected().items());new=inspect_tasks(state);b=accounting(state);fp,inputs=fingerprint();active=[x for x in state['tasks'] if x.get('scheduler_state','SUBMITTED') in {'SUBMITTED','PENDING','RUNNING','CONFIGURING','COMPLETING','UNCONFIRMED'} and not x.get('processed_job_id')]
  state['active_and_pending_jobs']=[x['job_id'] for x in active];next_action=dict(task='INSPECT_ONLY',reason='No eligible independent task; retain scientific blockers and use zero GPU.')
  phases+=['SELECT_NEXT_ELIGIBLE_TASK'];p0key='P0_'+fp[:12];p0=[x for x in state['tasks'] if x['task_key']==p0key];gpu=[x for x in state['tasks'] if x['kind']=='gpu'];admissible=guard and unchanged and not b['unreconciled_project_GPU_jobs'] and b['remaining']>=0 and not any(x.get('output_integrity')=='FAIL' for x in state['tasks'])
  if not active and admissible:
   if not p0:
    next_action=dict(task=p0key,reason='Independent32-pair correspondence, official labels and geometry verification; new stage fingerprint. Reuse unchanged feedback and Qwen statistics.');submit(state,p0key,'cpu',fp,inputs)
   elif p0[-1].get('classification')=='ENGINEERING_PASS_SCOPED' and p0[-1].get('output_integrity')=='PASS' and not gpu and b['remaining']>=300:
    next_action=dict(task='P1_ABBA_v1',reason='One predefined same-A100 O/I/I/O fresh-process diagnostic addresses hardware and instrumentation confounds; max300 A100-seconds.');submit(state,'P1_ABBA_v1','gpu',fp,{})
   elif gpu and gpu[-1].get('result',{}).get('diagnostic_complete') and not any(t['task_key']=='P1_FORENSICS_v1' for t in state['tasks']):
    next_action=dict(task='P1_FORENSICS_v1',reason='New paired execution produced tensor evidence: locate raw versus clipping/optimizer divergence on CPU only; no GPU rerun.');submit(state,'P1_FORENSICS_v1','cpu',fp,{})
   else:
    retry=[x for x in state['tasks'] if x.get('classification') in {'RECOVERABLE_TRANSIENT_INFRASTRUCTURE','RECOVERABLE_ENGINEERING_FAILURE_FIXED'} and x['attempt']<2 and not any(y['task_key']==x['task_key'] and y['attempt']>x['attempt'] for y in state['tasks'])]
    if retry:
     t=retry[-1]
     if t['kind']=='cpu' or b['remaining']>=t['cap_seconds']:
      next_action=dict(task=t['task_key'],reason=t.get('verified_fix_reason','Confirmed Slurm transient '+t['scheduler_state'])+'; fresh unique attempt, at most2 retries.');state['previously_attempted_repairs'].append(dict(failed_job=t['job_id'],repair=t.get('verified_fix_reason','fresh process after transient scheduler failure'),attempt=t['attempt']+1,UTC=now()));submit(state,t['task_key'],t['kind'],fp,inputs,t['attempt']+1)
  elif active:next_action=dict(task='WAIT_FOR_REGISTERED_COMPLETION',reason='A registered task is active; afterany CPU hook will inspect it. No duplicate allocation.')
  else:next_action=dict(task='INSPECT_ONLY',reason='Guard/source/budget reconciliation blocks submission; never substitute an unapproved task.')
  b=accounting(state);state['GPU_budget']=b;state['next_action']=next_action;state['last_completed_task']=state.get('last_completed_task');state['blocked_dependencies']=dict(semantic='BLOCKED until genuine independent labels validate',ordinary_BF16='NOT VALIDATED; no confirmed correction to preserved historical failures',LPIPS='MISSING',four_arm_learning='BLOCKED: ordinary numerical reliability, semantic labels, scientific reward validity, independent evaluation',RSI_scientific_contribution='NOT ESTABLISHED; historical intervention confounded; tiny-task confidence intervals include0',untouched_evaluation='NOT CERTIFIED')
  if state['component_statuses']:
   state['blocked_dependencies']['LPIPS']='AVAILABLE_CALIBRATED; see execution remediation metrics'
   state['blocked_dependencies']['four_arm_learning']='FULL SCIENTIFIC METHOD blocked; development-only pretrained engineering pilot executed separately'
   state['blocked_dependencies']['benign_engineering_pilot']=state['component_statuses'].get('BENIGN_RL_STATUS',{})
  for t in state['tasks']:
   if t.get('classification')=='ENGINEERING_PASS_SCOPED':state['completed_dependencies'][t['task_key']]=dict(job_id=t['job_id'],scope='engineering only',result=t['result'])
  state['current_stage']='WAIT_REGISTERED_TASK' if any(not t.get('processed_job_id') for t in state['tasks']) else 'BLOCKED_EVIDENCE_MONITORING';state['last_inspection_UTC']=now();phases+=['EXECUTE','RECORD'];schedule=read(ROOT/'reports/monitoring/scheduler.json');receipt=dict(status='PASS_INSPECTION_ONLY',checked_UTC=now(),supervisor_job_id=os.getenv('SLURM_JOB_ID'),trigger=sys.argv[1] if len(sys.argv)>1 else 'periodic_callback',stage=state['current_stage'],completed_since_previous=new,active_jobs=[dict(job_id=t['job_id'],kind=t['kind'],task=t['task_key'],state=t.get('scheduler_state','SUBMITTED')) for t in state['tasks'] if not t.get('processed_job_id')],budget=b,guard_intact=guard,protected_sources_unchanged=unchanged,next_action=next_action,scheduler=schedule,cloud=state['cloud'],scientific_gate='NO-GO',state_machine_phases=phases+['SYNCHRONIZE'])
  receipt['component_statuses']=state['component_statuses'];receipt['scientific_gate_scope']='main-paper scientific claims, not an engineering or CPU prohibition'
  out=reports(state,receipt)
  try:
   from scripts.drive_sync_recovery_20261101 import integrate as drive_integrate
   state['cloud']=drive_integrate(state,out)
  except Exception as error:state['cloud']=dict(status='BLOCKED_SYNC_ISOLATED',checked_UTC=now(),error=type(error).__name__,reason='Cloud failure isolated; experiment monitoring continues.')
  receipt['cloud']=state['cloud'];receipt['publication']=read(out/'manifest.json')['version'];receipt['next_action']=next_action;state['last_supervisor_job_id']=receipt['supervisor_job_id'];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');write(BASE/f'check_{stamp}.json',receipt);write(BASE/'heartbeat.json',receipt);write(BASE/'state.json',state);history(BASE/'history.jsonl',receipt);history(ROOT/'experiments/overnight_supervisor_history.jsonl',dict(checked_UTC=receipt['checked_UTC'],supervisor_job_id=receipt['supervisor_job_id'],completed_since_previous=new,next_action=next_action,scientific_gate='NO-GO',receipt_path='reports/overnight_supervisor/check_'+stamp+'.json'));dashboard(ROOT)
  write(ROOT/'reports/validation_gpu_reservations.json',dict(cap=3600,consumed=b['consumed'],reserved=b['reserved'],reservations=[x for x in state['tasks'] if x['kind']=='gpu' and not x.get('processed_job_id')],scope=b['scope']))
  return receipt

if __name__=='__main__':print(json.dumps(cycle(),indent=2))

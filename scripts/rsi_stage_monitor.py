"""Bounded CPU extension to the EXISTING two-hour Slurm monitor; no busy loop."""
import argparse,copy,hashlib,html,json,math,os,re,shutil,subprocess,sys,time
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.monitoring.common import FileLock
BASE=ROOT/'reports/rsi_stage_monitor';ACTIVE={'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'};TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED'}
def now():return datetime.now(timezone.utc).isoformat()
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cmd(args):
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=45);return {'argv':list(map(str,args)),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def classify(state,log='',progress_age=0,elapsed=0):
    if state=='PENDING':return 'PENDING'
    if state in {'NODE_FAIL','BOOT_FAIL','PREEMPTED'} or (state=='FAILED' and re.search(r'Stale file handle|Input/output error|temporarily unavailable',log,re.I)):return 'RECOVERABLE_ENGINEERING_FAILURE'
    if state in ACTIVE:return 'STALLED' if state=='RUNNING' and progress_age>240 and elapsed>240 else state
    if state!='COMPLETED':return 'FAILED'
    return 'COMPLETED'
def inspect(task,job,state,elapsed):
    run=ROOT/'results'/job['run_id'];out=run/'clean_main' if task['kind']=='qwen' else run/'diagnostic';logs=''
    for ext in ['out','err']:
        path=ROOT/job[f'log_{ext}'];logs+=path.read_text(errors='replace')[-30000:] if path.exists() else ''
    progress=out/'live.json';age=time.time()-progress.stat().st_mtime if progress.exists() else elapsed
    status=classify(state,logs,age,elapsed);issues=[];receipt=read(out/'checks.json',{})
    if state in TERMINAL:
        if not receipt:issues.append('missing numerical assertions')
        elif receipt.get('status')!='PASS':issues.append('numerical/engineering checks failed')
        inv=read(out/'artifact_inventory.json',{})
        if not inv:issues.append('missing checkpoint/output hash inventory')
        for f in inv.get('files',[]):
            path=out/f['path']
            if not path.is_file() or sha(path)!=f['sha256']:issues.append('artifact checksum mismatch: '+f['path'])
        if task['kind']=='qwen' and receipt and state=='COMPLETED':
            if receipt.get('unique_training_steps')!=16 or receipt.get('prediction_records')!=128:issues.append('incomplete step/evaluation count')
            if not any(r['test']=='fresh_object_next_step_equal' and r['status']=='PASS' for r in receipt.get('checks',[])):issues.append('fresh-object continuation absent or failed')
        if issues and state=='COMPLETED':status='FAILED_INTEGRITY_OR_ASSERTIONS'
    # Finite metrics are validated as data, not by matching words in library warnings.
    updates=out/'updates.jsonl'
    if updates.exists():
        for line in updates.read_text().splitlines():
            try:
                r=json.loads(line)
                if any(not math.isfinite(r[k]) for k in ['loss','grad_norm_before_clip','max_parameter_change']):issues.append('nonfinite training metric')
            except (ValueError,KeyError):issues.append('incomplete progress line')
    return {'task':task['task'],'run_id':job['run_id'],'job_id':job['job_id'],'scheduler_state':state,'status':status,'elapsed_GPU_seconds':elapsed,'issues':issues,'checks_status':receipt.get('status'),'summary':read(out/'summary.json',{}),'last_progress':read(progress,{}),'log_out':job['log_out'],'log_err':job['log_err']}
def submit(task,run,ledger):
    child=ROOT/'results'/task['run_id'];logs=run/'logs';comment=f"rsi-stage:{run.name}:{task['task']}:a{task['attempt']}";seconds=task['wall_seconds']
    args=['sbatch','--parsable','--time',f'00:{seconds//60:02d}:00','--job-name','rsi-'+task['task'],'--comment',comment,'--output',str(logs/f"{task['task']}-a{task['attempt']}-%j.out"),'--error',str(logs/f"{task['task']}-a{task['attempt']}-%j.err"),str(ROOT/'slurm/rsi_repro_stage.sbatch'),task['kind'],task['run_id']]
    record={**task,'job_id':None,'submitted_UTC':now(),'command':args,'comment':comment};ledger.append(record);write(run/'job_ledger.json',ledger)
    result=cmd(args);record['submission_result']=result
    if result['exit_code'] or not result['stdout'].strip().split(';')[0].isdigit():write(run/'job_ledger.json',ledger);raise RuntimeError('Submission uncertain/failed; no automatic duplicate: '+json.dumps(result))
    record['job_id']=result['stdout'].strip().split(';')[0];record['log_out']=str((logs/f"{task['task']}-a{task['attempt']}-{record['job_id']}.out").relative_to(ROOT));record['log_err']=record['log_out'][:-3]+'err';write(run/'job_ledger.json',ledger)
    # Completion event is another bounded CPU pass, not another recurring chain.
    event=cmd(['sbatch','--parsable','--partition','epyc-cpu','--account','allusers','--cpus-per-task','1','--mem','2G','--time','00:03:00','--job-name','rsi-stage-event','--dependency','afterany:'+record['job_id'],'--output',str(logs/'event-%j.out'),'--error',str(logs/'event-%j.err'),'--wrap',str(ROOT/'.venv-editor/bin/python')+' -B '+str(ROOT/'scripts/rsi_stage_monitor.py')+' tick'])
    record['completion_monitor_submission']=event;write(run/'job_ledger.json',ledger);return record
def retry(task,last,attempt):
    new=copy.deepcopy(task);new['kind']=last.get('kind',task['kind']);new['attempt']=attempt;new['run_id']=re.sub(r'_a\d+$','_a'+str(attempt),last['run_id']);root=ROOT/'results'/new['run_id'];root.mkdir(exist_ok=False)
    old=ROOT/'results'/last['run_id'];cfg=read(old/'protocol.json');cfg['run_id']=new['run_id'];write(root/'protocol.json',cfg)
    if task['kind']=='qwen':
        manifest=read(old/'clean_data_manifest.json');manifest['protocol_sha256']=sha(root/'protocol.json');write(root/'clean_data_manifest.json',manifest);shutil.copy2(old/'accepted_clean_phases.json',root/'accepted_clean_phases.json')
    return new
def semantic_status():
    kit=ROOT/'results/semantic_clean_20261029_v1/semantic_review';files=sorted((kit/'submitted').glob('*.csv'));review=[];adjud=[]
    for p in files:
        header=p.open(encoding='utf-8-sig').readline()
        (adjud if 'adjudicator_id' in header else review).append(p)
    state=read(BASE/'semantic_state.json',{});digest=hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()
    if len(review)>=2 and len(adjud)<=1 and digest!=state.get('input_digest'):
        analysis_id='independent_reviews_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f');args=[ROOT/'.venv-editor/bin/python','-B',ROOT/'scripts/analyze_semantic_clean_stage.py','--run-id','semantic_clean_20261029_v1','--analysis-id',analysis_id]
        for p in review:args+=['--review-csv',p]
        if adjud:args+=['--adjudication-csv',adjud[0]]
        result=cmd(args);write(BASE/f'{analysis_id}_execution.json',result)
        if result['exit_code']==0:
            summary=read(ROOT/'results/semantic_clean_20261029_v1'/analysis_id/'analysis.json');state={'input_digest':digest,'analysis_id':analysis_id,'status':summary['human_semantic_status'],'resolved_cases':summary['human_final_case_count']};write(BASE/'semantic_state.json',state)
    return state or {'status':'BLOCKED','resolved_cases':0,'reason':'No two genuine independent review exports supplied'}
def run_stage_monitor():
    BASE.mkdir(parents=True,exist_ok=True);pointer=read(BASE/'active_stage.json')
    if not pointer:return {'status':'NO_ACTIVE_EXTENSION'}
    with FileLock(BASE/'monitor.lock') as acquired:
        if not acquired:return {'status':'SKIPPED_LOCKED'}
        started=time.time();run=ROOT/'results'/pointer['run_id'];cfg=read(run/'protocol.json');tasks=read(run/'tasks.json');ledger=read(run/'job_ledger.json',[]);observed=[];actions=[];errors=[]
        if any(not x.get('job_id') for x in ledger):raise RuntimeError('Uncertain submission in ledger; reconcile before another submission')
        result=cmd(['sacct','-X','-j',','.join(x['job_id'] for x in ledger),'-nP','--format=JobIDRaw,State,ElapsedRaw,ExitCode']) if ledger else {'exit_code':0,'stdout':'','stderr':''};states={}
        if result['exit_code']:raise RuntimeError('Accounting unavailable: '+result['stderr'])
        for line in result['stdout'].splitlines():
            f=line.split('|')
            if len(f)>=4:states[f[0]]=(f[1].split()[0].split('+')[0],int(f[2]),f[3])
        charged=cfg['charged_prior_seconds'];reserved=0
        for record in ledger:
            state,elapsed,exitcode=states.get(record['job_id'],('UNCONFIRMED',0,''));task=next(t for t in tasks if t['task']==record['task']);observed.append(inspect(task,record,state,elapsed))
            if state in TERMINAL:charged+=elapsed
            else:reserved+=record['wall_seconds']
        guard_source=ROOT/'src/rl/phase5_backend.py';guard_ok=sha(guard_source)==pointer['reward_guard_sha256']
        if not guard_ok:errors.append('reward guard source changed; submissions blocked')
        if all(states.get(x['job_id'],('UNCONFIRMED',))[0] in TERMINAL for x in ledger) and guard_ok:
            for task in tasks:
                attempts=[x for x in ledger if x['task']==task['task']]
                if attempts:
                    last=attempts[-1];lastobs=next(x for x in observed if x['job_id']==last['job_id'])
                    if lastobs['status']=='COMPLETED':continue
                    if lastobs['status']!='RECOVERABLE_ENGINEERING_FAILURE' or len(attempts)>cfg['max_retries_per_task']:continue
                    if charged+reserved+task['wall_seconds']>cfg['GPU_budget_seconds']:
                        errors.append('Budget blocks retry '+task['task']);continue
                    candidate=retry(task,last,len(attempts));actions.append({'repair':'fresh initialization after scheduler/transient I/O failure; preserve failed checkpoints','new_run_id':candidate['run_id']})
                else:candidate=task
                if charged+reserved+candidate['wall_seconds']>cfg['GPU_budget_seconds']:
                    errors.append('Budget blocks '+candidate['task']);continue
                submitted=submit(candidate,run,ledger);actions.append({'submitted_job_id':submitted['job_id'],'task':candidate['task'],'run_id':candidate['run_id']});reserved+=candidate['wall_seconds'];break
        schedule=read(ROOT/'reports/monitoring/scheduler.json',{});semantic=semantic_status();report={'status':'PASS','status_scope':'Successful inspection, not blanket experiment acceptance','checked_UTC':now(),'run_id':run.name,'observations':observed,'actions':actions,'findings':errors,'GPU_charged_seconds_including_prior':charged,'GPU_reserved_seconds':reserved,'GPU_remaining_unallocated_seconds':3600-charged-reserved,'GPU_budget_seconds':3600,'last_periodic_success_UTC':schedule.get('last_success_utc'),'next_periodic_job_id':schedule.get('next_job_id'),'next_periodic_eligible_UTC':schedule.get('next_due_utc'),'reward_guard_source_intact':guard_ok,'seconds':time.time()-started,'human_reviews':semantic,'failed_registered_attempts':[o['job_id'] for o in observed if o['scheduler_state']=='FAILED']}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');path=BASE/f'check_{stamp}.json';write(path,report);report['log_path']=str(path.relative_to(ROOT));write(BASE/'heartbeat.json',report);write(run/'monitor_latest.json',report)
        checked_local=datetime.fromisoformat(report['checked_UTC']).astimezone(ZoneInfo('America/New_York')).isoformat();next_local=datetime.fromisoformat(schedule['next_due_utc']).astimezone(ZoneInfo('America/New_York')).isoformat()
        body='<section style="max-width:1180px;margin:20px auto;padding:24px"><h2>Live two-hour monitoring · '+html.escape(checked_local)+'</h2><p>Stage '+run.name+' · persistent CPU Slurm successor chain. Next eligible '+next_local+' · job '+str(schedule.get('next_job_id'))+'. GPU charged '+str(charged)+'s; reserved '+str(reserved)+'s; unallocated '+str(3600-charged-reserved)+'s /3600s. Prior validation charges1002s are conservatively included.</p><table><tr><th>Task</th><th>Run</th><th>Job</th><th>State</th></tr>'
        for x in observed:body+='<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in ['task','run_id','job_id','status'])+'</tr>'
        for x in actions:
            if 'submitted_job_id' in x:body+='<tr><td>'+x['task']+'</td><td>'+x['run_id']+'</td><td>'+x['submitted_job_id']+'</td><td>SUBMITTED; scheduler confirmation next check</td></tr>'
        body+='</table><p>Independent semantic review: '+html.escape(str(semantic.get('status')))+'; resolved cases '+str(semantic.get('resolved_cases',0))+'/160. Training-induced backdoor evaluation not performed. Numerical failures are retained, never relabeled PASS.</p><p><a href="../reports/rsi_stage_monitor/heartbeat.json">Heartbeat JSON</a> · <a href="../'+report['log_path']+'">Timestamped check</a></p></section>'
        dashboard=ROOT/'dashboard/RSI_Experimental_Progress_Latest.html';text=dashboard.read_text();text=re.sub(r'<!--RSI_STAGE_MONITOR_START-->.*?<!--RSI_STAGE_MONITOR_END-->','',text,flags=re.S);block='<!--RSI_STAGE_MONITOR_START-->'+body+'<!--RSI_STAGE_MONITOR_END-->';text=text.replace('</header>','</header>'+block,1);temp=dashboard.with_suffix('.partial');temp.write_text(text);temp.replace(dashboard)
        with (BASE/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        return report
def selftest():
    cases=[('failed','FAILED','assertion failure',0,1,'FAILED'),('node failure','NODE_FAIL','',0,1,'RECOVERABLE_ENGINEERING_FAILURE'),('pending','PENDING','',99999,0,'PENDING'),('stalled','RUNNING','',300,300,'STALLED'),('completed','COMPLETED','',0,10,'COMPLETED')]
    results=[{'test':name,'passed':classify(state,log,age,elapsed)==expected} for name,state,log,age,elapsed,expected in cases]
    with FileLock(BASE/'selftest.lock') as one:
        with FileLock(BASE/'selftest.lock') as two:results.append({'test':'overlap_lock','passed':one and not two})
    scratch=ROOT/'tmp'/'replication_monitor_20261029_selftest';scratch.mkdir(parents=True,exist_ok=False)
    try:
        out=scratch/'clean_main';out.mkdir();write(out/'checks.json',{'status':'PASS','unique_training_steps':16,'prediction_records':128,'checks':[{'test':'fresh_object_next_step_equal','status':'PASS'}]});(out/'checkpoint.pt').write_bytes(b'SYNTHETIC_CHECKPOINT_ONLY')
        write(out/'artifact_inventory.json',{'files':[{'path':'checkpoint.pt','sha256':sha(out/'checkpoint.pt')}]})
        fixture={'run_id':'../tmp/'+scratch.name,'job_id':'SYNTHETIC_ONLY','log_out':'tmp/'+scratch.name+'/out.log','log_err':'tmp/'+scratch.name+'/err.log'};task={'task':'SYNTHETIC_ONLY','kind':'qwen'}
        good=inspect(task,fixture,'COMPLETED',1);results.append({'test':'fixture_valid_checkpoint','passed':not good['issues']})
        (out/'checkpoint.pt').write_bytes(b'SYNTHETIC_TAMPERING');bad=inspect(task,fixture,'COMPLETED',1);results.append({'test':'checkpoint_tamper_detected','passed':any('checksum mismatch' in x for x in bad['issues'])})
        write(out/'checks.json',{'status':'FAIL'});bad=inspect(task,fixture,'COMPLETED',1);results.append({'test':'numerical_assertion_failure_detected','passed':any('checks failed' in x for x in bad['issues'])})
    finally:
        target=scratch.resolve();assert target.parent==(ROOT/'tmp').resolve() and target.name=='replication_monitor_20261029_selftest';shutil.rmtree(target)
    report={'status':'PASS' if all(r['passed'] for r in results) else 'FAIL','timestamp_UTC':now(),'tests':results,'real_jobs_disrupted':0,'fixtures_only':True};write(BASE/'failure_detection_selftest.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tick','selftest']);a=p.parse_args();print(json.dumps(selftest() if a.action=='selftest' else run_stage_monitor(),indent=2))

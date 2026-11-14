"""Existing two-hour chain extension. No independent scheduler or GPU submission."""
import json
import shutil
from pathlib import Path
from ..monitoring.common import command,now,read_json
from ..utils.logging import write_json
from .completion import process_completions,bounded_repair
from .dashboard import refresh
from .controller import policy

CPU_STAGES={'S1_DATA','S1_REVIEW_PACKAGE','S1_REWARD_AUDIT','S2_REAL_HARNESS','S3_RL_ENGINEERING','S4_STATIC_COMPARISON'}

def invalidate_dependencies(state):
    changed=True
    while changed:
        changed=False
        for item in state.get('stages',{}).values():
            if any(state['stages'].get(d,{}).get('state') in {'FAILED','BLOCKED'} for d in item.get('required_inputs',[])) and item['state']!='BLOCKED':
                item.update(state='BLOCKED',decision='BLOCKED',reason='Current prerequisite failed integrity/completion');changed=True

def dispatch_ready_cpu(root,state,snapshot):
    """A one-shot CPU task worker, never a second recurring scheduler."""
    queue=snapshot.get('queue',{})
    if queue.get('exit_code')!=0:return {'action':'BLOCKED','reason':'scheduler unavailable'}
    if any('poisoning-phase6-cpu' in line for line in queue.get('stdout','').splitlines()):
        return {'action':'NO_SUBMISSION','reason':'CPU worker already active/pending'}
    eligible=[]
    for sid,item in state.get('stages',{}).items():
        if sid not in CPU_STAGES:continue
        transient=item['state']=='FAILED' and item.get('retryable_engineering_failure')
        key=sid+':'+item.get('failure_signature','unknown')
        if transient and state.get('retry_counts',{}).get(key,0)>=2:continue
        deps_ready=all(state['stages'].get(d,{}).get('state')=='PASSED' for d in item.get('required_inputs',[]))
        if (item['state'] in {'NOT_STARTED','BLOCKED'} and deps_ready) or transient:eligible.append(sid)
    if not eligible:return {'action':'NO_SUBMISSION','reason':'no eligible unfinished CPU task'}
    path=Path(root)/'reports/phase6/cpu_dispatch.json';dispatch=read_json(path,{})
    if dispatch.get('submission_uncertain'):return {'action':'BLOCKED','reason':'prior CPU submission uncertain'}
    dispatch.update(submission_uncertain=True,timestamp=now(),eligible_stages=eligible);write_json(path,dispatch)
    result=command(['sbatch','--parsable','--partition=epyc-cpu','--account=allusers','--cpus-per-task=1',
       '--mem=12G','--time=00:20:00','--job-name=poisoning-phase6-cpu',
       '--output=reports/phase6/cpu-%j.out','--error=reports/phase6/cpu-%j.err',
       '--wrap=export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; .venv-editor/bin/python -B scripts/run_phase6.py'],cwd=root)
    job=result['stdout'].strip().split(';')[0]
    dispatch.update(result=result,job_id=job if job.isdigit() else None,submission_uncertain=result['exit_code']!=0 or not job.isdigit())
    write_json(path,dispatch)
    from ..monitoring.common import append_json
    append_json(Path(root)/'reports/phase6/cpu_dispatch_events.jsonl',dispatch)
    return dispatch

def cycle(root,snapshot=None):
    root=Path(root);policy(root)
    from .review_workflow import refresh_review_analysis
    review_analysis=refresh_review_analysis(root)
    if snapshot is None:
        from ..monitoring.common import slurm_snapshot
        ids=[str(read_json(p/'manifest.json',{}).get('slurm_job_id')) for p in (root/'experiments').glob('EXP_*phase6_*')
             if read_json(p/'manifest.json',{}).get('slurm_job_id')]
        snapshot=slurm_snapshot(ids)
    completed=process_completions(root,snapshot.get('jobs',{}))
    retry_path=root/'reports/phase6/repair_attempts.json';attempts=read_json(retry_path,{})
    repairs=[]
    for path in (root/'experiments').glob('EXP_*phase6_*'):
        if read_json(path/'status.json',{}).get('status') not in {'CREATED','RUNNING','SUBMITTED'}:continue
        for kind,name in [('missing_images','images'),('missing_checkpoints','checkpoints'),('missing_run_log','run.log')]:
            if not (path/name).exists():repairs.append(bounded_repair(root,path,kind,attempts))
    write_json(retry_path,attempts)
    state=read_json(root/'reports/PHASE6_STAGE_STATUS.json',{})
    indexed={c['experiment_id']:c for c in completed}
    for sid,item in state.get('stages',{}).items():
        attempt=read_json(root/f'reports/phase6/attempts/{sid}.json',{})
        if item.get('state') in {'RUNNING','FAILED'} and attempt.get('created_at','')>=item.get('started_at',''):
            item['experiment_id']=attempt.get('experiment_id',item.get('experiment_id'))
        result=indexed.get(item.get('experiment_id'))
        if not result:continue
        if result['decision']=='PASS_ENGINEERING':item['completion_evidence_state']='VERIFIED_SLURM_AND_OUTPUTS'
        elif result.get('scheduler_state') in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'} or any('Slurm completion' not in reason for reason in result['reasons']):
            item.update(state='FAILED',decision='FAIL',completion_evidence_state='INVALID_OUTPUTS',reason='; '.join(result['reasons']))
            if result.get('scheduler_state') in {'NODE_FAIL','PREEMPTED'}:
                item.update(retryable_engineering_failure=True,failure_signature='Slurm:'+result['scheduler_state'])
        else:item['completion_evidence_state']='PENDING_OR_UNSUCCESSFUL_SLURM_ACCOUNTING'
    source_checks={}
    # Cheap exact manifest checks, no image/model/trajectory loading on idle monitor.
    manifest=root/'data/processed/vqav2_content_v2/provenance.json'
    if manifest.is_file():
        import hashlib
        for split,h in read_json(manifest)['split_sha256'].items():
            p=manifest.parent/(split+'.json');source_checks[split]=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h
    if source_checks and not all(source_checks.values()) and 'S1_DATA' in state.get('stages',{}):
        state['stages']['S1_DATA'].update(state='FAILED',decision='FAIL',reason='Current v2 manifest integrity failure')
    invalidate_dependencies(state)
    write_json(root/'reports/PHASE6_STAGE_STATUS.json',state)
    cpu_dispatch=dispatch_ready_cpu(root,state,snapshot)
    versions=read_json(root/'reports/phase6/S2_REAL_HARNESS.json',{}).get('results',{})
    live={'timestamp_utc':now(),'scheduler':snapshot,'completions':completed,
      'stage_states':{s:v['state'] for s,v in state.get('stages',{}).items()},
      'v2_manifest_hash_checks':source_checks,'disk':{'free_bytes':shutil.disk_usage(root).free},
      'phase6_gpu_seconds':0,'gpu_submission_permitted':False,'current_harness':versions.get('accepted_workflow'),
      'harness_diagnostic_overhead':versions.get('total_search_resources'),
      'bounded_engineering_repairs':repairs,
      'cpu_progression':cpu_dispatch,
      'active_investigations':'none after initial bounded CPU tasks',
      'review_submission_files':[p.name for p in (root/'reviews/incoming').glob('*.json')],
      'review_submission_scope':'presence only; independent provenance/adjudication required, never automatic human truth',
      'review_analysis':review_analysis,
      'stale_process_check':command(['ps','-u',__import__('os').environ.get('USER',''),'-o','pid,etime,args'],timeout=10),
      'monitor_scheduler_state':read_json(root/'reports/monitoring/scheduler.json',{}),
      'next_automatic_action':state.get('next_automatic_action'),
      'scope':'lightweight CPU metadata/integrity checks; no model loading or GPU work'}
    write_json(root/'reports/phase6/live.json',live)
    view=refresh(root,live)
    return {'status':'PASS_ENGINEERING_SNAPSHOT','dashboard':view,'completion_records':len(completed),'new_gpu_submissions':0}

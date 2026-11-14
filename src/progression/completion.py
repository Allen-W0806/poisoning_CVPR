"""Append-only completion observations and bounded cooperating-operator repairs."""
import json
import math
import hashlib
from pathlib import Path
from ..monitoring.common import read_json,append_json,now,safe_path
from ..utils.logging import write_json

def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True

def validate_completion(path,slurm):
    path=Path(path);manifest=read_json(path/'manifest.json',{});status=read_json(path/'status.json',{})
    contract=read_json(path/'phase6_run_contract.json',{});outputs=read_json(path/'output_manifest.json',{})
    metrics=read_json(path/'metrics.json',{})
    reasons=[]
    if manifest.get('experiment_id')!=path.name or outputs.get('experiment_id')!=path.name:reasons.append('manifest identity mismatch')
    required={'experiment_id','stage','parent_experiment_id','start_time','end_time','status','model_revision','dataset_version',
      'reward_version','harness_version','git_commit','seed','slurm_job_id','gpu_seconds','configuration','metrics',
      'failure_reason','checkpoint_path','evaluation_protocol'}
    if not required<=contract.keys():reasons.append('incomplete run contract')
    if status.get('status')!='PASS':reasons.append('run not engineering PASS')
    if not metrics or not finite(metrics):reasons.append('empty or nonfinite metrics')
    if not outputs.get('files') or not outputs.get('required_metrics') or any(k not in metrics for k in outputs.get('required_metrics',[])):
        reasons.append('missing required output/metric')
    for relative,expected in outputs.get('files',{}).items():
        p=safe_path(path,relative)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:reasons.append('output integrity: '+relative)
    job=manifest.get('slurm_job_id')
    if job:
        accounting=slurm.get(str(job),{})
        if accounting.get('State')!='COMPLETED' or accounting.get('ExitCode')!='0:0':reasons.append('Slurm completion unavailable or unsuccessful')
    return {'experiment_id':path.name,'decision':'PASS_ENGINEERING' if not reasons else 'BLOCKED',
      'scheduler_state':slurm.get(str(job),{}).get('State') if job else 'LOCAL_CPU',
      'reasons':reasons,'scheduler_scope':'actual accounting required for Slurm jobs; local CPU runs have null job',
      'scientific_success':False,'timestamp':now()}

def process_completions(root,slurm):
    root=Path(root);state_path=root/'reports/phase6/completion_state.json';state=read_json(state_path,{})
    results=[]
    for path in sorted((root/'experiments').glob('EXP_*phase6_*')):
        status=read_json(path/'status.json',{}).get('status')
        if status in {'CREATED','RUNNING','SUBMITTED'}:
            manifest=read_json(path/'manifest.json',{});job=str(manifest.get('slurm_job_id') or '')
            accounting=slurm.get(job,{})
            if accounting.get('State') not in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}:continue
            # Terminal Slurm evidence can reconcile an ACTIVE run, never an already finalized run.
            write_json(path/'status.json',{'status':'COMPLETED_UNVERIFIED' if accounting['State']=='COMPLETED' else 'FAIL',
              'updated_at':now(),'scheduler_state':accounting['State'],'scheduler_exit':accounting.get('ExitCode'),
              'reason':'Finalizer interrupted; scientific success not inferred'})
            append_json(root/'experiments/state_observations.jsonl',{'timestamp':now(),'experiment_id':path.name,
              'before':status,'after':'terminal_unverified','scheduler_state':accounting['State']})
        result=validate_completion(path,slurm);signature=json.dumps({k:v for k,v in result.items() if k!='timestamp'},sort_keys=True)
        if state.get(path.name)!=signature:
            append_json(root/'experiments/phase6_completion_observations.jsonl',result)
            append_json(root/'reports/phase6/completions.jsonl',result);state[path.name]=signature
        results.append(result)
    write_json(state_path,state)
    return results

def bounded_repair(root,run_path,kind,state):
    root=Path(root).resolve();path=Path(run_path).resolve()
    if path.parent!=root/'experiments':raise ValueError('Repair must stay in experiment registry')
    if read_json(path/'status.json',{}).get('status') not in {'CREATED','RUNNING','SUBMITTED'}:
        return {'action':'BLOCKED','reason':'finalized experiment is immutable'}
    allowed={'missing_images':'images','missing_checkpoints':'checkpoints','missing_run_log':'run.log'}
    if kind not in allowed:return {'action':'BLOCKED','reason':'no validated minimal repair rule'}
    signature=path.name+':'+kind;count=state.get(signature,0)
    if count>=2:return {'action':'BLOCKED','reason':'two retry ceiling'}
    state[signature]=count+1;target=path/allowed[kind];before=target.exists()
    if target.suffix=='.log':target.touch(exist_ok=True)
    else:target.mkdir(exist_ok=True)
    result={'action':'bounded_engineering_repair','kind':kind,'signature':signature,'attempt':count+1,
       'before_exists':before,'after_exists':target.exists(),'timestamp':now()}
    append_json(root/'reports/phase6/repairs.jsonl',result);return result

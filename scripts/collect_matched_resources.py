"""Reconcile Slurm allocation separately from measured process timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='matched_feedback_20261028_v2');p.add_argument('--cpu-job-id',default='86794');a=p.parse_args()
    run=ROOT/'results'/a.run_id;ledger=[json.loads(x) for x in (run/'job_ledger.jsonl').read_text().splitlines()]
    repair_path=run/'main_repair_v1/repair_receipt.json'
    repair=json.loads(repair_path.read_text()) if repair_path.exists() else None
    if repair and not any(r['job_id']==repair['slurm_job_id'] for r in ledger):
        from submit_phase2 import accounting
        charged,_=accounting(ledger)
        if charged+300>3600:raise RuntimeError('Repair reservation exceeds bounded stage budget')
        record={'job_id':repair['slurm_job_id'],'phase':'repair_main','reserved_seconds':300,'charged_before':charged,'reason':'single grayscale RGB input metadata repair'}
        with (run/'job_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        ledger.append(record)
    ids=[r['job_id'] for r in ledger]+[a.cpu_job_id];env=dict(os.environ,TZ='UTC')
    result=subprocess.run(['sacct','-X','-j',','.join(ids),'-nP','--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,AllocTRES'],capture_output=True,text=True,check=True,env=env)
    rows=[];gpu_seconds=0;cpu_only=0
    for line in result.stdout.splitlines():
        values=line.split('|')
        if not values or values[0] not in ids:continue
        row=dict(zip(['job_id','state','exit_code','start_UTC','end_UTC','allocated_elapsed_seconds','allocated_resources'],values))
        row['allocated_elapsed_seconds']=int(row['allocated_elapsed_seconds']);row['GPU_count']=1 if row['job_id']!=a.cpu_job_id else 0
        for name in ['start','end']:
            value=row[name+'_UTC']
            try:row[name+'_America_New_York']=datetime.fromisoformat(value).replace(tzinfo=timezone.utc).astimezone(ZoneInfo('America/New_York')).isoformat()
            except ValueError:row[name+'_America_New_York']=None
        if row['GPU_count']:gpu_seconds+=row['allocated_elapsed_seconds']
        else:cpu_only+=row['allocated_elapsed_seconds']
        rows.append(row)
    phases={s:json.loads((run/s/'checks.json').read_text()) for s in ['pilot','main'] if (run/s/'checks.json').exists()}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    resource={'status':'PASS' if len(rows)==len(ids) and all(r['state'].split()[0] in terminal for r in rows) and gpu_seconds<=3600 and (repair is None or repair['status']=='PASS') else 'NOT VALIDATED',
        'jobs':rows,'gpu_seconds':gpu_seconds,'gpu_hours':gpu_seconds/3600,'budget_gpu_seconds':3600,'remaining_gpu_seconds':3600-gpu_seconds,
        'cpu_only_allocated_seconds':cpu_only,'raw_sacct':result.stdout,
        'phase_process_timing':{s:{k:v[k] for k in ['start_unix','finish_unix','elapsed_process_seconds','generation_calls','VQA_records','generation_seconds','inference_seconds']} for s,v in phases.items()},
        'total_inference_records':sum(v['VQA_records'] for v in phases.values())+(repair['repair_inference_calls'] if repair else 0),
        'actual_VLM_inference_calls':sum(json.loads((run/s/'model_metadata.json').read_text())['model_calls'] for s in phases)+(repair['repair_inference_calls'] if repair else 0),
        'total_generation_calls_including_repeats':sum(v['generation_calls'] for v in phases.values()),
        'total_timed_generation_seconds':sum(v['generation_seconds'] for v in phases.values()),'total_timed_inference_seconds':sum(v['inference_seconds'] for v in phases.values())+(repair['repair_inference_seconds'] if repair else 0),
        'repair_receipt':repair,'dollar_cost':None,'dollar_cost_status':'unavailable; no price inferred','retry_count_GPU':1 if repair else 0,'failed_preparation_versions':['matched_feedback_20261028_v1'],
        'new_pretrained_model_optimizer_steps':0,'checkpoint_state':'Tiny synthetic checkpoint replay tested; full pretrained BF16 training gate remains NOT VALIDATED',
        'preparation_CPU_timing':'not instrumented as process time; frozen-protocol and manifest timestamps retained; not counted as GPU time'}
    (run/'resource_accounting.json').write_text(json.dumps(resource,indent=2));print(json.dumps(resource,indent=2))


if __name__=='__main__':main()

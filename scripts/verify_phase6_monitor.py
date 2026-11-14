"""Scheduler evidence only. Does not install, enqueue, tick or rerun stages."""
import json
import _bootstrap
from src.monitoring.common import command,read_json,now
from src.monitoring.scheduler import Scheduler
from src.monitoring.core_v6 import load_config
from src.progression.dashboard import atomic_text

if __name__=='__main__':
    root=_bootstrap.ROOT
    status=Scheduler(root,load_config(root)).status();s=status['state']
    ids=','.join(str(s[k]) for k in ['last_tick_job_id','next_job_id'] if s.get(k))
    evidence={'observed_utc':now(),'scheduler_status':status,
        'squeue':command(['squeue','-h','-u','ydk297','-o','%i|%T|%j|%b|%M|%l|%R']),
        'sacct':command(['sacct','-X','-j',ids,'-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,ReqTRES,AllocTRES,Timelimit']),
        'latest_manual_cycle':read_json(root/'reports/MONITORING_STATUS.json',{}),
        'single_successor':len(status['owned_jobs'])==1,
        'scientific_gpu_budget_seconds':read_json(root/'configs/phase6.json')['scientific_gpu_budget_seconds'],
        'recurrence_without_codex':'Slurm CPU successor; Windows mirror requires explicit synchronization',
        'no_scheduler_created_by_verification':True}
    atomic_text(root/'reports/phase6_followup/monitor_verification.json',json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'observed_utc':evidence['observed_utc'],'single_successor':evidence['single_successor'],'owned_jobs':status['owned_jobs'],'gpu_budget':evidence['scientific_gpu_budget_seconds']}))

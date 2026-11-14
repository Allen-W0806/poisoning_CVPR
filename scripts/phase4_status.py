"""Compact read-only progress for active Phase4 allocations."""
import json
from pathlib import Path
import _bootstrap
from submit_phase2 import accounting

root=_bootstrap.ROOT
ledger=root/'reports/phase4_jobs.jsonl'
jobs=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
charged,slurm=accounting(jobs)
runs=[]
for path in sorted((root/'experiments').glob('*phase4_*')):
    if not all((path/name).is_file() for name in ['status.json','metrics.json','run.log']):
        runs.append({'id':path.name,'status':'CREATING','updates':0,'rewarded_generations':0,'validation_looks':0,
            'last_reward_mean':None,'last_gradient_norm':None,'last_update_seconds':None,'last_generation':None})
        continue
    status=json.loads((path/'status.json').read_text())['status']
    metrics=json.loads((path/'metrics.json').read_text())
    events=[json.loads(x) for x in (path/'run.log').read_text().splitlines(keepends=True) if x.endswith('\n')]
    trajectories=[x for x in events if x['event']=='trajectory_reward']
    updates=[x['value'] for x in events if x['event']=='generator_rl_update'] or metrics.get('updates',[])
    if status in ['RUNNING','SUBMITTED'] or metrics.get('arm'):
        runs.append({'id':path.name,'status':status,'updates':len(updates),'rewarded_generations':len(trajectories),
            'validation_looks':len(metrics.get('evaluations',[])),
            'last_reward_mean':updates[-1]['reward_mean'] if updates else None,
            'last_gradient_norm':updates[-1]['gradient_norm'] if updates else None,
            'last_update_seconds':updates[-1]['step_seconds'] if updates else None,
            'last_generation':trajectories[-1]['value']['trajectory_id'] if trajectories else None})
print(json.dumps({'slurm':slurm,'charged_or_reserved_seconds':charged,'runs':runs}))

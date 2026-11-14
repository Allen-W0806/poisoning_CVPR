"""Phase4 CPU checks, separate budget; no automatic scientific retries."""
import json
import math
from .common import digest,TERMINAL
from .diagnostics import issue

PROTECTED=['scripts/run_phase4.py','src/evaluation/phase4_statistics.py']
PROTECTED+=['scripts/run_phase4_budgeted.py','configs/phase4_campaign_v2.yaml']
PROTECTED+=['scripts/run_phase4_rsi.py','src/rsi/phase4_schedule.py']

def phase4_checks(root,runs,state,snapshot,protect=True):
    findings=[];baseline=state.setdefault('phase4_protected_hashes_v1',{})
    if protect:
        for name in PROTECTED:
            if not (root/name).is_file():continue
            current=digest(root/name)
            if name not in baseline:baseline[name]=current
            elif baseline[name]!=current:findings.append(issue('phase4_configuration_drift',severity='error',path=name))
    ledger=root/'reports/phase4_jobs.jsonl'
    jobs=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    charged=0;diagnostic=0
    for job in jobs:
        row=snapshot.get('jobs',{}).get(str(job['job_id']))
        cost=int(row['ElapsedRaw']) if row and row['State'] in TERMINAL else job['reserved_seconds']
        charged+=cost
        if job['task'].startswith('diagnostic'):diagnostic+=cost
    if charged>14400 or diagnostic>2700:findings.append(issue('phase4_budget_exceeded',severity='error',charged_seconds=charged,diagnostic_seconds=diagnostic))
    summaries=[]
    for run in runs:
        if '_phase4_' not in run['id']:continue
        updates=run['metrics'].get('updates',[])
        for update in updates:
            good=all(math.isfinite(float(update.get(k,float('nan')))) for k in ['loss','gradient_norm','reward_mean','reward_variance'])
            good=good and update.get('finite_gradients') and update.get('gradient_norm',0)>0 and update.get('trainable_max_change',0)>0
            good=good and update.get('conditioning_verified') and update.get('behavior_logp_replay_max_difference',1)<=1e-6
            checkpoint=update.get('checkpoint')
            if not good:findings.append(issue('phase4_invalid_update',run['id'],'error',step=update.get('step')))
            if checkpoint and not (__import__('pathlib').Path(checkpoint)/'training_state.pt').is_file():
                findings.append(issue('phase4_missing_checkpoint',run['id'],'error',step=update.get('step')))
        summaries.append({'id':run['id'],'updates':len(updates),'status':run['status'],
            'peak_bytes':[u.get('peak_gpu_memory_bytes') for u in updates],
            'reward_mean':[u.get('reward_mean') for u in updates],'gradient_norm':[u.get('gradient_norm') for u in updates]})
    return {'findings':findings,'charged_or_reserved_gpu_seconds':charged,'max_gpu_seconds':14400,
        'diagnostic_charged_or_reserved_seconds':diagnostic,'max_diagnostic_seconds':2700,'runs':summaries,
        'recovery_policy':'No automatic Phase4 experiment submissions; exact resume not presumed'}

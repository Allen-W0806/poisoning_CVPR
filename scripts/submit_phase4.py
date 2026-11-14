"""Separate 14400 GPU-second Phase4 ledger, single GPU, conservative reservations."""
import argparse
import json
import subprocess
from pathlib import Path
import _bootstrap
from _phase4 import phase4_run
from submit_phase2 import accounting
from src.monitoring.common import FileLock
from src.utils.logging import write_json

TASKS={"diagnostic":"diagnose_phase4_restart.py", "frozen":"run_phase4.py --arm frozen --seed 42", "rl42":"run_phase4_budgeted.py --arm rl --seed 42", "rl123":"run_phase4_budgeted.py --arm rl --seed 123", "rsi42":"run_phase4_rsi.py --arm rsi --seed 42", "rsi123":"run_phase4_rsi.py --arm rsi --seed 123"}
TASKS['diagnostic_outcomes']='phase4_restart_outcomes.py'

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=TASKS,required=True); p.add_argument("--minutes",type=int,default=15)
    a=p.parse_args()
    if not 1<=a.minutes<=60: raise ValueError("Jobs must fit the hard Phase4 cap")
    root=_bootstrap.ROOT
    with FileLock(root/"reports/phase4_submit.lock") as locked:
        if not locked: raise RuntimeError("Submission lock already held")
        ledger=root/"reports/phase4_jobs.jsonl"
        previous=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        charged,rows=accounting(previous)
        terminal={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED"}
        observed={r[0]:r[1].split()[0] for r in rows}
        if any(observed.get(x['job_id']) not in terminal|{'RUNNING','PENDING','CONFIGURING','COMPLETING'} for x in previous):
            raise RuntimeError('Previous Phase4 allocation unconfirmed')
        active=[x for x in previous if observed.get(x['job_id']) not in terminal]
        if len(active)>=2 or (active and (a.task.startswith('diagnostic') or a.task=='frozen')):raise RuntimeError('At most two independent single-GPU pilot jobs; staged diagnostic/frozen alone')
        if any(x['task']==a.task for x in active):raise RuntimeError('Duplicate active scientific arm')
        if charged+60*a.minutes>14400: raise RuntimeError(f"Budget {charged}+{60*a.minutes}>14400")
        if (root/"reports/phase4_submission_uncertain.json").exists(): raise RuntimeError("Resolve prior unknown submission outcome first")
        queue=subprocess.check_output(["squeue","-u","ydk297","-h","-o","%i|%j|%b"],text=True)
        owned={x['job_id'] for x in active}
        if any('gpu:' in line and line.split('|')[0].strip() not in owned for line in queue.splitlines()):
            raise RuntimeError('Another non-Phase4 user GPU allocation is active/pending')
        diagnostic_charged=sum(int(r[2]) for r in rows if any(j['job_id']==r[0] and j['task'].startswith('diagnostic') for j in previous))
        if a.task.startswith('diagnostic') and diagnostic_charged+60*a.minutes>2700: raise RuntimeError('Diagnostic cap2700sec')
        launch_seed=123 if a.task.endswith('123') else 42
        with phase4_run("slurm_"+a.task,{"task":a.task,"seed":launch_seed,"additional_phase_budget_seconds":14400,"charged_before":charged,"reserved_seconds":60*a.minutes},splits=()) as run:
            script=f'''#!/bin/bash
#SBATCH --job-name=p4-{a.task}
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time={a.minutes//60:02d}:{a.minutes%60:02d}:00
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
source .venv-editor/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
trap 'rc=$?; .venv/bin/python scripts/finalize_job.py "$rc"; exit "$rc"' EXIT
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python scripts/{TASKS[a.task]}
'''
            jobfile=run.path/"job.sbatch"; jobfile.write_text(script)
            write_json(root/"reports/phase4_submission_uncertain.json",{"parent":run.id,"task":a.task,"reservation":60*a.minutes})
            job=subprocess.check_output(["sbatch","--parsable","--output",str(run.path/"slurm-%j.out"),"--error",str(run.path/"slurm-%j.err"),
                "--export","ALL,POISONING_JOB_RUN="+str(run.path),str(jobfile)],text=True).strip().split(";")[0]
            if not job.isdigit(): raise RuntimeError("Unknown sbatch response; reservation marker retained")
            record={"job_id":job,"task":a.task,"reserved_seconds":60*a.minutes,"parent_run":run.id}
            write_json(run.path/"job_submission.json",record)
            with ledger.open("a") as f:f.write(json.dumps(record)+"\n")
            (root/"reports/phase4_submission_uncertain.json").unlink()
            run.set_status("SUBMITTED",slurm_job_id=job); run.finished=True
            print(json.dumps(record))

"""Separate 7200 GPU-second Phase3 ledger, single GPU, conservative reservations."""
import argparse
import json
import subprocess
from pathlib import Path
import _bootstrap
from _phase3 import phase3_run
from submit_phase2 import accounting
from src.monitoring.common import FileLock
from src.utils.logging import write_json

TASKS={"one":"smoke_generator_rl.py --updates 1 --batch-size 1",
       "smoke":"smoke_generator_rl.py --updates 3 --batch-size 10",
       "backward":"diagnose_diffusion_backward.py",
       "semantics":"audit_semantic_validity.py"}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=TASKS,required=True); p.add_argument("--minutes",type=int,default=15)
    a=p.parse_args()
    if not 1<=a.minutes<=30: raise ValueError("Short jobs only")
    root=_bootstrap.ROOT
    with FileLock(root/"reports/phase3_submit.lock") as locked:
        if not locked: raise RuntimeError("Submission lock already held")
        ledger=root/"reports/phase3_jobs.jsonl"
        previous=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        charged,rows=accounting(previous)
        terminal={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED"}
        observed={r[0]:r[1].split()[0] for r in rows}
        if any(observed.get(x["job_id"]) not in terminal for x in previous): raise RuntimeError("Previous Phase3 allocation active/unconfirmed")
        if charged+60*a.minutes>7200: raise RuntimeError(f"Budget {charged}+{60*a.minutes}>7200")
        if (root/"reports/phase3_submission_uncertain.json").exists(): raise RuntimeError("Resolve prior unknown submission outcome first")
        queue=subprocess.check_output(["squeue","-u","ydk297","-h","-o","%j|%b"],text=True)
        if any("gpu:" in line for line in queue.splitlines()): raise RuntimeError("Another user GPU allocation is active/pending")
        if a.task=="smoke":
            pointer=json.loads((root/"reports/phase3_latest_one.json").read_text())
            if json.loads((Path(pointer["path"])/"status.json").read_text())["status"]!="PASS": raise RuntimeError("One-update gate not passed")
        with phase3_run("slurm_"+a.task,{"task":a.task,"seed":42,"additional_phase_budget_seconds":7200,"charged_before":charged,"reserved_seconds":60*a.minutes},splits=()) as run:
            script=f'''#!/bin/bash
#SBATCH --job-name=p3-{a.task}
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:{a.minutes:02d}:00
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
            write_json(root/"reports/phase3_submission_uncertain.json",{"parent":run.id,"task":a.task,"reservation":60*a.minutes})
            job=subprocess.check_output(["sbatch","--parsable","--output",str(run.path/"slurm-%j.out"),"--error",str(run.path/"slurm-%j.err"),
                "--export","ALL,POISONING_JOB_RUN="+str(run.path),str(jobfile)],text=True).strip().split(";")[0]
            if not job.isdigit(): raise RuntimeError("Unknown sbatch response; reservation marker retained")
            record={"job_id":job,"task":a.task,"reserved_seconds":60*a.minutes,"parent_run":run.id}
            write_json(run.path/"job_submission.json",record)
            with ledger.open("a") as f:f.write(json.dumps(record)+"\n")
            (root/"reports/phase3_submission_uncertain.json").unlink()
            run.set_status("SUBMITTED",slurm_job_id=job); run.finished=True
            print(json.dumps(record))

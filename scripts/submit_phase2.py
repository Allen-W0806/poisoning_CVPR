"""Bound each one-GPU Phase 2 allocation and enforce cumulative <=7200 seconds."""
import argparse
import csv
import io
import json
import subprocess
import _bootstrap
from _phase2 import phase2_run
from src.utils.logging import write_json

TASKS = {"editor": (".venv-editor", "smoke_pretrained_editor.py"),
         "training": (".venv-editor", "smoke_generator_training.py"),
         "vqa_dev": (".venv", "run_vqa_baseline_full.py --split validation"),
         "vqa_test": (".venv", "run_vqa_baseline_full.py --split test"),
         "semantics": (".venv", "validate_edited_images.py"),
         "controller": (".venv-editor", "compare_rl_rsi.py")}


def accounting(previous):
    if not previous:
        return 0, []
    result = subprocess.run(["sacct", "-X", "-j", ",".join(p["job_id"] for p in previous),
        "-nP", "--format=JobIDRaw,State,ElapsedRaw"], capture_output=True, text=True, check=True)
    rows = [r for r in csv.reader(io.StringIO(result.stdout), delimiter="|") if r and r[0].isdigit()]
    states = {r[0]: r for r in rows}
    charged = 0
    for p in previous:
        row = states.get(p["job_id"])
        # Reserve full wall limit for any still active or unreported allocation.
        terminal = row and row[1].split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
        charged += int(row[2]) if terminal else p["reserved_seconds"]
    return charged, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--minutes", type=int, default=15)
    a = p.parse_args()
    if not 1 <= a.minutes <= 30:
        raise ValueError("Only short 1–30 minute jobs are supported")
    registry = _bootstrap.ROOT / "reports/phase2_jobs.jsonl"
    previous = [json.loads(x) for x in registry.read_text().splitlines()] if registry.exists() else []
    charged, rows = accounting(previous)
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    observed = {r[0]: r[1].split()[0] for r in rows}
    active = [p["job_id"] for p in previous if observed.get(p["job_id"]) not in terminal_states]
    if active:
        raise RuntimeError(f"One GPU at a time: prior Phase 2 jobs still active/unconfirmed {active}")
    if charged + a.minutes * 60 > 7200:
        raise RuntimeError(f"Phase 2 budget exhausted/reserved: {charged}s; refusing +{a.minutes*60}s")
    preflight = subprocess.check_output(["sinfo", "-p", "epyc-gpu", "-o", "%P %a %l %G"]).decode()
    env, command = TASKS[a.task]
    with phase2_run("slurm_" + a.task, {"task": a.task, "phase2_budget_seconds": 7200,
                    "charged_before": charged, "reserved_seconds": a.minutes * 60,
                    "command": f"{env}/bin/python scripts/{command}"}) as run:
        run.event("preflight", {"sinfo": preflight, "accounting": rows})
        script = f"""#!/bin/bash
#SBATCH --job-name=p2-{a.task}
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:{a.minutes:02d}:00
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
source {env}/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
trap 'rc=$?; /storage/home/ydk297/projects/poisoning/.venv/bin/python scripts/finalize_job.py "$rc"; exit "$rc"' EXIT
hostname
nvidia-smi
python scripts/{command}
"""
        jobfile = run.path / "job.sbatch"
        jobfile.write_text(script)
        command_line = ["sbatch", "--parsable", "--output", str(run.path / "slurm-%j.out"),
            "--error", str(run.path / "slurm-%j.err"), "--export", "ALL,POISONING_JOB_RUN=" + str(run.path), str(jobfile)]
        result = subprocess.check_output(command_line, text=True).strip().split(";")[0]
        record = {"job_id": result, "task": a.task, "reserved_seconds": a.minutes * 60, "parent_run": run.id}
        write_json(run.path / "job_submission.json", {**record, "command": command_line})
        with registry.open("a") as f:
            f.write(json.dumps(record) + "\n")
        run.set_status("SUBMITTED", slurm_job_id=result)
        run.finished = True
        print(json.dumps(record))

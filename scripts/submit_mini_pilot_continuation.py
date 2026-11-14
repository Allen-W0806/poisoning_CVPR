"""Reserve the sole user-authorized continuation, within 5400 cumulative GPU seconds."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.startswith("pilot_run_") or not all(ch.isalnum() or ch in "_-" for ch in args.run_id):
        raise ValueError("Invalid run ID")
    output = ROOT / "outputs" / args.run_id
    state = json.loads((output / "run_state.json").read_text())
    if state["status"] != "FAILED" or not state.get("failure", "").startswith("TimeoutError:"):
        raise ValueError("Only a stopped time-cap run can continue; no retries for numerical/model failures")
    if state.get("numerical", {}).get("status") != "PASS" or not (output / "checkpoints/interrupted/training_state.pt").is_file():
        raise ValueError("A passed numerical gate and a real interrupted optimizer checkpoint are required")
    original = json.loads((ROOT / "outputs" / f"{args.run_id}_submission/submission.json").read_text())
    previous_job = original["job_id"]
    account = subprocess.run(["sacct", "-X", "-nP", "-j", previous_job, "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"],
                             check=True, capture_output=True, text=True).stdout
    record = next(line.split("|") for line in account.splitlines() if line.split("|")[0] == previous_job)
    if record[1].split()[0] not in {"FAILED", "TIMEOUT", "COMPLETED"} or "gres/gpu=1" not in record[3]:
        raise ValueError("The first one-GPU allocation must have ended unambiguously")
    elapsed = int(record[2])
    if elapsed + 1800 > 5400:
        raise ValueError("The next allocation would exceed the user-authorized cumulative cap")
    if subprocess.run(["squeue", "-h", "-j", previous_job], check=True, capture_output=True, text=True).stdout.strip():
        raise ValueError("The previous job is still active")
    folder = output / "continuation_submission"
    folder.mkdir(exist_ok=False)
    authorization = {"cumulative_cap_GPU_seconds": 5400, "max_continuations": 1,
                     "user_authorization": "User explicitly approved checkpoint continuation with cumulative single-GPU cap 90 minutes",
                     "created_UTC": datetime.now(timezone.utc).isoformat(), "previous_job": previous_job,
                     "previous_actual_GPU_seconds": elapsed, "new_reservation_GPU_seconds": 1800,
                     "worst_case_total_GPU_seconds": elapsed + 1800, "new_GPU_count": 1,
                     "scientific_settings_unchanged": True, "worker_SHA256": sha256(ROOT / "scripts/continue_mini_pilot.py")}
    write_json(folder / "authorization.json", authorization, exclusive=True)
    script = "\n".join(["#!/bin/bash", "#SBATCH --job-name=mini-pilot-cont", "#SBATCH --partition=epyc-gpu",
                          "#SBATCH --account=allusers", "#SBATCH --gres=gpu:tesla_a100_80gb:1", "#SBATCH --cpus-per-task=4",
                          "#SBATCH --mem=64G", "#SBATCH --time=00:30:00", f"#SBATCH --chdir={ROOT}",
                          f"#SBATCH --output={folder}/slurm-%j.log", f"#SBATCH --error={folder}/slurm-%j.log",
                          "set -euo pipefail", "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
                          "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4",
                          "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8",
                          "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
                          f"exec .venv-editor/bin/python scripts/continue_mini_pilot.py --output_dir outputs/{args.run_id}", ""])
    path = folder / "run.sbatch"
    path.write_text(script)
    process = subprocess.run(["sbatch", "--parsable", str(path)], text=True, capture_output=True)
    receipt = {"returncode": process.returncode, "stdout": process.stdout.strip(), "stderr": process.stderr.strip(),
               "status": "SUBMITTED" if process.returncode == 0 else "FAILED_OR_AMBIGUOUS_NO_RETRY"}
    if process.returncode == 0:
        receipt["job_id"] = process.stdout.strip().split(";")[0]
    write_json(folder / "submission.json", receipt, exclusive=True)
    print(json.dumps({"authorization": authorization, "submission": receipt}, indent=2))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())

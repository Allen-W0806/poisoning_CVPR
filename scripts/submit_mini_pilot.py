"""One bounded allocation for the explicitly requested benign mini-pilot; no retries."""
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
        raise ValueError("Expected a unique pilot_run_* ID")
    output = ROOT / "outputs" / args.run_id
    if output.exists():
        raise FileExistsError("A pilot run is never overwritten or automatically retried")
    # The sibling reservation prevents duplicate submission, including ambiguous sbatch responses.
    submission = ROOT / "outputs" / f"{args.run_id}_submission"
    submission.mkdir(parents=True, exist_ok=False)
    active = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T"],
                            check=True, text=True, capture_output=True).stdout
    if any("mini-pilot" in line for line in active.splitlines()):
        raise RuntimeError("An existing mini-pilot is pending/running; do not submit a duplicate")
    script = "\n".join(["#!/bin/bash", "#SBATCH --job-name=mini-pilot",
                          "#SBATCH --partition=epyc-gpu", "#SBATCH --account=allusers",
                          "#SBATCH --gres=gpu:tesla_a100_80gb:1", "#SBATCH --cpus-per-task=4",
                          "#SBATCH --mem=64G", "#SBATCH --time=01:00:00",
                          f"#SBATCH --chdir={ROOT}", f"#SBATCH --output={submission}/slurm-%j.log",
                          f"#SBATCH --error={submission}/slurm-%j.log", "set -euo pipefail",
                          f"export PILOT_OUTPUT_DIR=outputs/{args.run_id}", "exec bash run_mini_pilot.sh", ""])
    script_path = submission / "run.sbatch"
    script_path.write_text(script)
    receipt = {"run_id": args.run_id, "created_UTC": datetime.now(timezone.utc).isoformat(),
               "GPU_cap_seconds": 3600, "process_cap_seconds": 3300, "GPU_count": 1,
               "automatic_retries": 0, "authorization": "Current user request for 32/16/16 benign five-arm pilot",
               "separate_from_finalized_historical_budgets": True, "script_SHA256": sha256(script_path),
               "status": "SUBMISSION_RESERVED", "output": str(output)}
    write_json(submission / "submission.json", receipt, exclusive=True)
    process = subprocess.run(["sbatch", "--parsable", str(script_path)], text=True, capture_output=True)
    receipt.update(sbatch_returncode=process.returncode, stdout=process.stdout.strip(), stderr=process.stderr.strip())
    if process.returncode == 0 and process.stdout.strip().split(";")[0].isdigit():
        receipt.update(status="SUBMITTED", job_id=process.stdout.strip().split(";")[0])
    else:
        receipt["status"] = "FAILED_OR_AMBIGUOUS_NO_RETRY"
    write_json(submission / "submission.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "SUBMITTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

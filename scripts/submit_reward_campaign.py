"""Account for every one-GPU reservation in the user-authorized 20 GPU-hour campaign."""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import sha256, write_json


def accounting(campaign):
    records = []
    for path in sorted((campaign / "submissions").glob("*/submission.json")):
        rec = json.loads(path.read_text())
        if not rec.get("job_id"):
            if rec["status"] != "SBATCH_FAILED_NO_JOB":
                raise RuntimeError(f"Resolve ambiguous submission before more spending: {path}")
            continue
        job = rec["job_id"]
        data = subprocess.run(["sacct", "-X", "-nP", "-j", job, "--format=JobID,State,ElapsedRaw,AllocTRES"],
                              check=True, text=True, capture_output=True).stdout
        rows = [line.split("|") for line in data.splitlines() if line.split("|")[0] == job]
        if len(rows) != 1:
            raise RuntimeError(f"Cannot settle job {job}; refuse overcommit")
        _, status, elapsed, tres, *_ = rows[0]
        terminal = status.split()[0] in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
        if "gres/gpu=1" not in tres and status not in {"PENDING", "CANCELLED"}:
            raise RuntimeError(f"Unexpected GPU accounting for {job}: {tres}")
        records.append({"job_id": job, "stage": rec["stage"], "state": status,
            "elapsed_seconds": int(elapsed), "settled": terminal,
            "charged_or_reserved_GPU_seconds": int(elapsed) if terminal else rec["reservation_GPU_seconds"]})
    return {"records": records, "committed_GPU_seconds": sum(r["charged_or_reserved_GPU_seconds"] for r in records),
            "actual_GPU_seconds": sum(r["elapsed_seconds"] for r in records), "cap_GPU_seconds": 72000}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--stage")
    p.add_argument("--minutes", type=int)
    p.add_argument("--accounting-only", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    campaign = (ROOT / a.campaign).resolve()
    if not campaign.is_relative_to(ROOT / "outputs") or campaign == ROOT / "outputs":
        raise ValueError("Campaign must be a run-specific outputs directory")
    campaign.mkdir(parents=True, exist_ok=True)
    auth = campaign / "authorization.json"
    if not auth.exists():
        write_json(auth, {"seed": 42, "cap_GPU_seconds": 72000,
            "user_authorization": "最多20 GPU小时，允许继续外部复现；先跑1个seed就行",
            "scope": "Reward debug, benign seed42 pilot, verified external backdoor baseline reproduction",
            "old_historical_budgets_reopened": False, "created_UTC": datetime.now(timezone.utc).isoformat()}, exclusive=True)
    acc = accounting(campaign)
    write_json(campaign / "GPU_ACCOUNTING.json", acc)
    if a.accounting_only:
        print(json.dumps(acc, indent=2))
        return
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not a.stage or not all(c.isalnum() or c in "_-" for c in a.stage) or not command:
        raise ValueError("Safe unique stage and an explicit command required")
    if not a.minutes or not 1 <= a.minutes <= 1200 or acc["committed_GPU_seconds"] + a.minutes * 60 > 72000:
        raise ValueError("Reservation would exceed this campaign's 20 GPU-hour cap")
    sub = campaign / "submissions" / a.stage
    sub.mkdir(parents=True, exist_ok=False)
    import hashlib
    scratch_key = hashlib.sha256((campaign.name + ":" + a.stage).encode()).hexdigest()[:12]
    scratch = ROOT / "tmp" / f"rd_{scratch_key}"
    # multiprocessing adds pymp-*/listener-* below TMPDIR; AF_UNIX paths have
    # a hard108-byte limit, so a descriptive long campaign path is unsafe.
    if len(str(scratch).encode()) > 70:
        raise ValueError("Scratch prefix too long for multiprocessing AF_UNIX sockets")
    script = "\n".join(["#!/bin/bash", f"#SBATCH --job-name=reward-{a.stage[:16]}",
        "#SBATCH --partition=epyc-gpu", "#SBATCH --account=allusers", "#SBATCH --gres=gpu:tesla_a100_80gb:1",
        "#SBATCH --cpus-per-task=4", "#SBATCH --mem=80G", f"#SBATCH --time={a.minutes}",
        f"#SBATCH --chdir={ROOT}", f"#SBATCH --output={sub}/slurm-%j.log", f"#SBATCH --error={sub}/slurm-%j.log",
        "set -euo pipefail", "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false",
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1",
        "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128", f"mkdir -p {shlex.quote(str(scratch))}",
        f"export TMPDIR={shlex.quote(str(scratch))}", shlex.join(command), ""])
    (sub / "run.sbatch").write_text(script)
    rec = {"stage": a.stage, "reservation_GPU_seconds": a.minutes * 60, "seed": 42,
           "status": "RESERVED", "command": command, "prior_accounting": acc,
           "script_SHA256": sha256(sub / "run.sbatch"), "scratch": str(scratch),
           "created_UTC": datetime.now(timezone.utc).isoformat()}
    import shutil
    source_paths = [ROOT / part for part in command if part.endswith(".py") and (ROOT / part).is_file()]
    rec["entrypoint_SHA256"] = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    for source in source_paths:
        shutil.copy2(source, sub / source.name)
    write_json(sub / "submission.json", rec, exclusive=True)
    reply = subprocess.run(["sbatch", "--parsable", str(sub / "run.sbatch")], text=True, capture_output=True)
    rec.update(returncode=reply.returncode, stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
    if reply.returncode == 0 and reply.stdout.strip().split(";")[0].isdigit():
        rec.update(status="SUBMITTED", job_id=reply.stdout.strip().split(";")[0])
    else:
        rec["status"] = "FAILED_OR_AMBIGUOUS_NO_RETRY"
    write_json(sub / "submission.json", rec)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

"""Allocate an immutable parent directory before submitting Slurm."""
import argparse
import subprocess
import _bootstrap
from _runs import new_run
from src.utils.logging import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["smoke_gpu", "clean_baseline", "rl_validation"], default="smoke_gpu")
    args = p.parse_args()
    info = subprocess.run(["sinfo", "-p", "epyc-gpu", "-o", "%P %a %l %G"], capture_output=True, text=True, check=True)
    print(info.stdout)
    with new_run("slurm_" + args.job, {"job": args.job, "partition": "epyc-gpu", "gpus": 1, "cpus": 4, "mem": "32G", "time": "00:20:00"}) as run:
        run.event("resource_preflight", info.stdout)
        cmd = ["sbatch", "--parsable", "--chdir", str(_bootstrap.ROOT),
               "--output", str(run.path / "slurm-%j.out"), "--error", str(run.path / "slurm-%j.err"),
               "--export", "ALL,POISONING_JOB_RUN=" + str(run.path), str(_bootstrap.ROOT / f"slurm/{args.job}.sbatch")]
        submitted = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = submitted.stdout.strip().split(";")[0]
        write_json(run.path / "job_submission.json", {"job_id": job_id, "command": cmd})
        run.event("slurm_submitted", {"job_id": job_id})
        run.set_status("SUBMITTED", slurm_job_id=job_id)
        # Parent remains active; the allocated process will attach and finalize it.
        run.finished = True
        print("JOB_ID=" + job_id)
        print("RUN=" + str(run.path))

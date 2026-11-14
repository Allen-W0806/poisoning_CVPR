"""Read-only cluster inventory, exact dependency and scheduler-cost evidence."""
import json
import subprocess
import sys
from pathlib import Path
import _bootstrap
from src.utils.logging import write_json


def capture(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


if __name__ == "__main__":
    root = _bootstrap.ROOT
    jobs = []
    for path in sorted((root / "experiments").glob("EXP_*/job_submission.json")):
        jobs.append(json.loads(path.read_text())["job_id"])
    results = {
        "location": str(root), "python": sys.executable,
        "partitions": capture(["sinfo", "-o", "%P %a %l %D %G"]),
        "associations": capture(["sacctmgr", "-nP", "show", "assoc", "where", "user=ydk297", "format=Cluster,Account,Partition,QOS"]),
        "queue": capture(["squeue", "-u", "ydk297"]),
        "disk": capture(["df", "-h", str(root)]),
        "quota": capture(["/usr/lpp/mmfs/bin/mmlsquota", "-u", "ydk297", "--block-size", "auto"]),
        "retained_sizes": capture(["du", "-sh", str(root / ".venv"), str(root / "data"), str(root / "experiments"),
            "/storage/home/ydk297/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct"]),
        "scheduler_accounting": capture(["sacct", "-j", ",".join(jobs), "--format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS", "-P"]),
        "git_history": capture(["git", "log", "--format=%H %s"]),
        "freeze": capture([sys.executable, "-m", "pip", "freeze"])}
    write_json(root / "reports/cluster_inventory.json", results)
    print(json.dumps({"jobs": jobs, "inventory": str(root / "reports/cluster_inventory.json")}))

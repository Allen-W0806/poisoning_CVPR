"""Bounded orchestration of already specified jobs; never retunes or retries.

This is a foreground helper for this one run, not a recurring automation.
Every GPU submission uses the unique-stage cumulative20GPUh accountant.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256,write_json
from scripts.submit_reward_campaign import accounting


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def snapshot(c):
    state=read(c/"internal_pilot/run_state.json",{})
    updates=[]
    log=c/"internal_pilot/training.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row.get("scope")=="arm_training":updates.append(row)
    progress=read(c/"shadowcast/poison_progress.json",[])
    acc=accounting(c)
    return {"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"internal_status":state.get("status"),
        "internal_committed_updates":{arm:sum(r["arm"]==arm for r in updates) for arm in ["ddpo_only","rule_rsi","rl_rsi"]},
        "completed_PGD_pairs":sum(r["count"] for r in progress),"PGD_target_pairs":277,
        "external_train_done":{str(n):(c/f"shadowcast/train_{n}_completed.json").is_file() for n in [0,50]},
        "external_eval_done":{str(n):(c/f"shadowcast/eval_{n}_summary.json").is_file() for n in [0,50]},
        "frozen_VQA_done":(c/"lighting_vqa_audit/summary.json").is_file(),
        "actual_GPU_h":acc["actual_GPU_seconds"]/3600,"committed_GPU_h":acc["committed_GPU_seconds"]/3600,
        "attention":[r for r in acc["records"] if r["state"].split()[0] in {"FAILED","TIMEOUT","OUT_OF_MEMORY"}
                     and r["stage"] not in {"shadowcast_poison","shadowcast_clean_tmpfix"}],
        "all_GPU_settled":all(r["settled"] for r in acc["records"])}


def gpu(c,name,minutes,command):
    sub=c/"submissions"/name
    if sub.exists():return False
    args=[str(ROOT/".venv-editor/bin/python"),str(ROOT/"scripts/submit_reward_campaign.py"),
          "--campaign",str(c.relative_to(ROOT)),"--stage",name,"--minutes",str(minutes),"--",*command]
    result=subprocess.run(args,text=True,capture_output=True)
    with (c/"supervisor_submissions.log").open("a") as f:f.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f"Budgeted submission failed/ambiguous at {name}; inspect receipt and queue")
    print(json.dumps({"submitted_stage":name,"cap_minutes":minutes}),flush=True)
    return True


def quality_cpu(c):
    folder=c/"cpu_submissions/shadowcast_quality"
    if folder.exists():return
    folder.mkdir(parents=True,exist_ok=False)
    import hashlib
    scratch=ROOT/"tmp"/("rd_scq_"+hashlib.sha256(c.name.encode()).hexdigest()[:10])
    script="\n".join(["#!/bin/bash","#SBATCH --job-name=shadowcast-quality",
        "#SBATCH --partition=epyc-cpu","#SBATCH --account=allusers","#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=24G","#SBATCH --time=00:15:00",f"#SBATCH --chdir={ROOT}",
        f"#SBATCH --output={folder}/slurm-%j.log",f"#SBATCH --error={folder}/slurm-%j.log",
        "set -euo pipefail","export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=",f"mkdir -p {scratch}",f"export TMPDIR={scratch}",
        f"exec .venv-editor/bin/python scripts/audit_shadowcast_results.py --campaign {c} --stage quality --device cpu",""])
    (folder/"run.sbatch").write_text(script)
    write_json(folder/"submission.json",{"status":"RESERVED","GPU_count":0,"cap_seconds":900,"scratch":str(scratch)},exclusive=True)
    result=subprocess.run(["sbatch","--parsable",str(folder/"run.sbatch")],capture_output=True,text=True)
    receipt={"status":"SUBMITTED" if result.returncode==0 else "FAILED_OR_AMBIGUOUS_NO_RETRY",
        "returncode":result.returncode,"stdout":result.stdout.strip(),"stderr":result.stderr.strip(),
        "GPU_count":0,"cap_seconds":900,"scratch":str(scratch),"worker_SHA256":sha256(ROOT/"scripts/audit_shadowcast_results.py")}
    if result.returncode==0:receipt["job_id"]=result.stdout.strip().split(";")[0]
    write_json(folder/"submission.json",receipt)
    print(json.dumps({"CPU_quality_submission":receipt}),flush=True)
    if result.returncode:raise RuntimeError("CPU quality submission failed; no retry")


def advance(c):
    rel=str(c.relative_to(ROOT));worker=[".venv-shadowcast-20261104/bin/python","scripts/run_shadowcast_seed42.py","--campaign",rel]
    if (c/"shadowcast/poison_completed.json").is_file():
        quality_cpu(c)
        measured=read(c/"shadowcast/poison_quality.json",{})
        admitted=(measured.get("PGD_trace_audit",{}).get("all_batches_complete") is True and
                  measured.get("maximum_8bit_difference",999)<=8 and measured.get("selected_poison_images")==50)
        if admitted and gpu(c,"shadowcast_poisoned",180,[*worker,"--stage","train","--poison-count","50"]):return
    if all((c/f"shadowcast/train_{n}_completed.json").is_file() for n in [0,50]):
        lock=c/"shadowcast/checkpoint_evaluation_lock.json"
        if not lock.exists():
            models={}
            for n in [0,50]:
                record=read(c/f"shadowcast/train_{n}_completed.json")
                folder=Path(record["output"])
                for item in record["files"]:
                    if sha256(folder/item["path"])!=item["SHA256"]:raise ValueError("Trained checkpoint changed before evaluation")
                models[str(n)]=record
            write_json(lock,{"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"models":models,
                "rule":"Final configured one-epoch models only; locked before either evaluation"},exclusive=True)
        for n in [0,50]:
            if gpu(c,f"shadowcast_eval_{n}",45,[*worker,"--stage","eval","--poison-count",str(n)]):return
    state=read(c/"internal_pilot/run_state.json",{})
    if state.get("status")=="COMPLETED":
        registered=read(c/"quality_audit_preregistration_v3.json")
        if registered["protocol_SHA256"]!=sha256(ROOT/"configs/lighting_v3_posthoc_vqa.json"):
            raise ValueError("Frozen VQA protocol changed after preregistration")
        if gpu(c,"lighting_frozen_vqa",15,[".venv-editor/bin/python","scripts/audit_lighting_v3_vqa.py","--campaign",rel]):return
    if all((c/f"shadowcast/eval_{n}_summary.json").is_file() for n in [0,50]) and not (c/"shadowcast/verified_statistics.json").exists():
        subprocess.run([str(ROOT/".venv-editor/bin/python"),str(ROOT/"scripts/audit_shadowcast_results.py"),
                        "--campaign",str(c),"--stage","statistics"],check=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign",required=True,type=Path)
    p.add_argument("--read-only",action="store_true")
    p.add_argument("--max-seconds",type=int,default=21600)
    args=p.parse_args();c=args.campaign.resolve()
    if not c.is_relative_to(ROOT/"outputs") or not 1<=args.max_seconds<=21600:raise ValueError("Unsafe path/time cap")
    if args.read_only:
        print(json.dumps(snapshot(c),indent=2));return
    started=time.monotonic()
    write_json(c/"supervisor_process.json",{"PID":os.getpid(),"host":os.uname().nodename,
        "max_seconds":args.max_seconds,"worker_SHA256":sha256(__file__),"auto_retry":False,
        "scope":"bounded one-run orchestration of fixed preregistered stages; no tuning or recurring automation"})
    while time.monotonic()-started<args.max_seconds:
        advance(c)
        status=snapshot(c);write_json(c/"supervisor_latest.json",status)
        compact={"UTC":status["UTC"],"v3_updates":list(status["internal_committed_updates"].values()),
            "PGD_pairs":status["completed_PGD_pairs"],"external_trained":sum(status["external_train_done"].values()),
            "ASR_evaluated":sum(status["external_eval_done"].values()),"GPU_h":round(status["actual_GPU_h"],3),
            "attention":[r["stage"] for r in status["attention"]]}
        print(json.dumps(compact),flush=True)
        if (status["internal_status"]=="COMPLETED" and all(status["external_eval_done"].values()) and
                status["frozen_VQA_done"] and (c/"shadowcast/poison_quality.json").is_file() and
                (c/"shadowcast/verified_statistics.json").is_file() and status["all_GPU_settled"]):
            write_json(c/"supervisor_finished.json",{"status":"ALL_FIXED_EXPERIMENT_SPINES_COMPLETE",
                "UTC":status["UTC"],"auto_retuning_or_retry":False})
            print("ALL_FIXED_CORE_SPINES_COMPLETE",flush=True);return
        time.sleep(30)
    raise TimeoutError("Bounded supervisor wall cap; do not silently start another supervisor or any retry")


if __name__=="__main__":main()

"""Prespecified extra task-quality audit using a frozen VQA model; no ASR claim."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256,write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign",required=True,type=Path)
    args=p.parse_args();c=args.campaign.resolve();pilot=c/"internal_pilot"
    protocol=json.loads((ROOT/"configs/lighting_v3_posthoc_vqa.json").read_text())
    state=json.loads((pilot/"run_state.json").read_text())
    if state["status"]!="COMPLETED" or not (pilot/"assessment_checkpoint_lock.json").is_file():
        raise RuntimeError("Only audit completed locked outputs; never use this for model/config selection")
    import torch
    from PIL import Image
    from src.models.qwen_loader import QwenVQA
    from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
    if not os.getenv("SLURM_JOB_ID") or torch.cuda.device_count()!=1:
        raise RuntimeError("One Slurm GPU required")
    out=c/"lighting_vqa_audit";out.mkdir(exist_ok=False)
    write_json(out/"protocol.json",protocol,exclusive=True)
    cohort=json.loads((pilot/"cohort.json").read_text())["test"]
    model=QwenVQA(protocol["model"],revision=protocol["revision"],device="cuda",max_new_tokens=32,seed=42)
    observations=[];started=time.monotonic()
    source_answers={}
    for group in protocol["groups"]:
        for index,record in enumerate(cohort):
            if time.monotonic()-started>840:
                raise TimeoutError("VQA quality diagnostic cooperative time cap")
            if group=="pixel_source":path=pilot/"visuals/sources"/f"{record['sample_id']}.png"
            else:
                obs=state["assessment"][group][index]
                if obs["sample_id"]!=record["sample_id"]:raise ValueError("Source ordering drift")
                path=pilot/obs["image_path"]
                if sha256(path)!=obs["image_SHA256"]:raise ValueError("Edited image hash drift")
            with Image.open(path) as image:
                answer=model.answer([image.convert("RGB")],[record["question"]])[0]
            if group=="pixel_source":source_answers[record["sample_id"]]=normalize_answer(answer)
            row={"group":group,"sample_id":record["sample_id"],"image_id":record["image_id"],
                 "image_SHA256":sha256(path),"question":record["question"],**score_answer(answer,record),
                 "agrees_with_frozen_source_answer":normalize_answer(answer)==source_answers[record["sample_id"]]}
            observations.append(row)
            with (out/"predictions.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps({"group":group,"n":16,"score":sum(r["score"] for r in observations if r["group"]==group)/16}),flush=True)
    summaries={g:{"n":16,"source_answer_accuracy":sum(r["score"] for r in observations if r["group"]==g)/16,
                  "source_prediction_agreement":sum(r["agrees_with_frozen_source_answer"] for r in observations if r["group"]==g)/16}
               for g in protocol["groups"]}
    summaries["fixed"]={**summaries["frozen"],"equivalent_to":"frozen","physical_queries":0}
    write_json(out/"summary.json",{"status":"COMPLETED","groups":summaries,"model":model.metadata(),
        "scope":protocol["scope"],"ASR":"NOT TESTED","Delta_Clean":"NOT TESTED",
        "elapsed_seconds":time.monotonic()-started},exclusive=True)


if __name__=="__main__":main()

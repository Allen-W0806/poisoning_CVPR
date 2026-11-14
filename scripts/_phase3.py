import json
import os
import hashlib
import yaml
from pathlib import Path
import _bootstrap
from src.data.dataset import sha256
from src.utils.logging import Experiment,write_json

def phase3_run(name,config,splits=("train",),source_parent=None):
    metadata=json.loads((_bootstrap.ROOT/"reports/phase2_model_metadata.json").read_text())["timbrooks/instruct-pix2pix"]
    qwen=json.loads((_bootstrap.ROOT/"reports/model_ready.json").read_text())
    provenance={"model_identifiers":[metadata["id"],qwen["id"]],
        "config_sha256":hashlib.sha256(yaml.safe_dump({"phase":3,**config},sort_keys=True).encode()).hexdigest(),
        "model_revisions":{metadata["id"]:metadata["revision"],qwen["id"]:qwen["revision"]},
        "dataset_version":"VQAv2-train2014-2017-04-26",
        "split_hashes":{s:sha256(_bootstrap.ROOT/f"data/processed/vqav2_seed42_v1/{s}.json") for s in splits}}
    parent=source_parent or (Path(os.environ["POISONING_JOB_RUN"]).name if os.environ.get("POISONING_JOB_RUN") else None)
    return Experiment(_bootstrap.ROOT,"phase3_"+name,{"phase":3,**config},provenance,parent)

def mark_latest(name,run):
    write_json(_bootstrap.ROOT/f"reports/phase3_latest_{name}.json",{"id":run.id,"path":str(run.path)})

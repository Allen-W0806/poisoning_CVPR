import json
import os
from pathlib import Path
import _bootstrap
from src.utils.logging import Experiment, write_json
from src.data.dataset import sha256


def phase2_run(name, config, splits=(), editor=False, qwen=False):
    provenance = {"model_identifiers": [], "model_revisions": {}, "dataset_version": "VQAv2-train2014-2017-04-26",
                  "split_hashes": {s: sha256(_bootstrap.ROOT / f"data/processed/vqav2_seed42_v1/{s}.json") for s in splits}}
    if editor:
        m = json.loads((_bootstrap.ROOT / "reports/phase2_model_metadata.json").read_text())["timbrooks/instruct-pix2pix"]
        provenance["model_identifiers"].append(m["id"])
        provenance["model_revisions"][m["id"]] = m["revision"]
    if qwen:
        m = json.loads((_bootstrap.ROOT / "reports/model_ready.json").read_text())
        provenance["model_identifiers"].append(m["id"])
        provenance["model_revisions"][m["id"]] = m["revision"]
    parent = Path(os.environ["POISONING_JOB_RUN"]).name if os.environ.get("POISONING_JOB_RUN") else None
    return Experiment(_bootstrap.ROOT, "phase2_" + name, {"phase": 2, "seed": 42, **config}, provenance, parent)


def mark_latest(name, run):
    write_json(_bootstrap.ROOT / f"reports/phase2_latest_{name}.json", {"id": run.id, "path": str(run.path)})


def latest(name):
    return Path(json.loads((_bootstrap.ROOT / f"reports/phase2_latest_{name}.json").read_text())["path"])

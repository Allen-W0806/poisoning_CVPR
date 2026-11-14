import json
import os
from pathlib import Path
import _bootstrap
from src.data.dataset import sha256
from src.utils.logging import Experiment


def provenance(dataset=None, include_model=False):
    p = {}
    if dataset:
        path = Path(dataset)
        p.update(dataset_version=path.parent.name, split_hashes={path.stem: sha256(path)})
    ready = _bootstrap.ROOT / "reports/model_ready.json"
    if include_model and ready.exists():
        m = json.loads(ready.read_text())
        p.update(model_identifiers=[m["id"]], model_revisions={m["id"]: m["revision"]})
    return p


def new_run(name, config=None, dataset=None, parent=None, include_model=False):
    if parent is None and os.environ.get("POISONING_JOB_RUN"):
        parent = Path(os.environ["POISONING_JOB_RUN"]).name
    return Experiment(_bootstrap.ROOT, name, {"seed": 42, **(config or {})}, provenance(dataset, include_model), parent)


def model_options(device):
    ready = json.loads((_bootstrap.ROOT / "reports/model_ready.json").read_text())
    return {"model_id": ready["id"], "revision": ready["revision"], "device": device,
            "local_files_only": True, "seed": 42, "max_new_tokens": 64}

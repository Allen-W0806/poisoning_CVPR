"""Exclusive run allocation, append-only event metrics and atomic status updates."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback
import uuid
import yaml


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def git_commit(root):
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "UNCOMMITTED"


class Experiment:
    @classmethod
    def attach_active(cls, root, path):
        obj = cls.__new__(cls)
        obj.root, obj.path = Path(root).resolve(), Path(path).resolve()
        if obj.path.parent != obj.root / "experiments":
            raise ValueError("Run must belong to the project experiment directory")
        status = json.loads((obj.path / "status.json").read_text())
        if status["status"] not in {"CREATED", "RUNNING", "SUBMITTED"}:
            raise ValueError("Cannot attach a finalized experiment")
        obj.id = obj.path.name
        obj.manifest = json.loads((obj.path / "manifest.json").read_text())
        obj.config = yaml.safe_load((obj.path / "config.yaml").read_text())
        obj.finished = False
        return obj

    def __init__(self, root, name, config, provenance=None, parent=None):
        self.root = Path(root).resolve()
        self.id = "EXP_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12] + "_" + name
        self.path = self.root / "experiments" / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "checkpoints").mkdir()
        (self.path / "images").mkdir()
        self.config = config
        self.finished = False
        with (self.path / "config.yaml").open("x", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=True)
        commit = git_commit(self.root)
        source_status = subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=self.root,capture_output=True,text=True)
        import torch
        self.manifest = {"experiment_id": self.id, "created_at": utcnow(), "seed": config.get("seed", 42),
                         "hyperparameters": config, "hostname": platform.node(),
                         "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         "git_commit": commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                         "source_tree_clean_at_creation": source_status.returncode == 0 and not source_status.stdout.strip(),
                         "parent_experiment": parent, "output_paths": {"root": str(self.path)},
                         "model_identifiers": [], "model_revisions": {}, "dataset_version": None,
                         "split_hashes": {}, "status": "CREATED", **(provenance or {})}
        with (self.path / "manifest.json").open("x", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, allow_nan=False)
        (self.path / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (self.path / "environment.txt").write_text(platform.platform() + "\n" + os.sys.version + "\n" + freeze.stdout, encoding="utf-8")
        (self.path / "run.log").touch()
        (self.path / "train.log").touch()
        write_json(self.path / "metrics.json", {})
        (self.path / "metrics.csv").write_text("timestamp,step,key,value\n", encoding="utf-8")
        self.set_status("RUNNING")
        self.event("created", {"run": self.id})
        registry = self.root / "experiments" / "registry.jsonl"
        # One append operation per event. IDs use exclusive directories and UUIDs.
        with registry.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"experiment_id": self.id, "path": str(self.path), "created_at": self.manifest["created_at"]}) + "\n")
        self.refresh_index()

    def set_status(self, status, **details):
        if self.finished:
            raise RuntimeError("Finalized experiment is immutable")
        write_json(self.path / "status.json", {"status": status, "updated_at": utcnow(), **details})

    def event(self, name, value):
        if self.finished:
            raise RuntimeError("Finalized experiment is immutable")
        line = json.dumps({"timestamp": utcnow(), "event": name, "value": value}, allow_nan=False) + "\n"
        for filename in ["run.log", "train.log"]:
            with (self.path / filename).open("a", encoding="utf-8") as f:
                f.write(line)

    def metrics(self, value, step=0):
        if self.finished:
            raise RuntimeError("Finalized experiment is immutable")
        write_json(self.path / "metrics.json", value)
        with (self.path / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, item in value.items():
                writer.writerow([utcnow(), step, key, json.dumps(item, allow_nan=False)])

    def finish(self, status="PASS", summary="", **details):
        self.event("finished", {"status": status, **details})
        self.set_status(status, **details)
        (self.path / "summary.md").write_text(f"# {self.id}\n\nStatus: {status}\n\n{summary}\n", encoding="utf-8")
        self.finished = True
        write_json(self.root / "experiments" / "latest.json", {"id": self.id, "path": str(self.path)})
        self.refresh_index()
        if (self.root/'configs/phase6.json').is_file():
            try:
                from ..progression.events import finalized_event
                finalized_event(self.root,self.id)
            except Exception as error:
                # Dashboard failure is observable but cannot relabel an already finalized run.
                os.sys.stderr.write('Finalized experiment retained; dashboard notification failed: '+str(error)+'\n')

    def refresh_index(self):
        rows = ["# Experiment index", "", "| ID | Status | Path |", "|---|---|---|"]
        for path in sorted((self.root / "experiments").glob("EXP_*")):
            status_path = path / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status") if status_path.exists() else "CREATED"
            rows.append(f"| {path.name} | {status} | {path} |")
        text = "\n".join(rows) + "\n"
        for path in [self.root / "EXPERIMENT_INDEX.md", self.root / "reports" / "EXPERIMENT_INDEX.md"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.finish("FAIL", "".join(traceback.format_exception(exc_type, exc, tb)))
        elif not self.finished:
            self.finish()
        return False

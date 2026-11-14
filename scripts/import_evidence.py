"""Import compact cluster evidence without replacing independent local runs."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import _bootstrap
from src.utils.scratch import scratch_directory
from src.utils.logging import Experiment


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    a = p.parse_args()
    root = _bootstrap.ROOT
    with scratch_directory() as scratch:
        with tarfile.open(a.archive) as archive:
            archive.extractall(scratch, filter="data")
        for run in (scratch / "experiments").glob("EXP_*"):
            target = root / "experiments" / run.name
            if target.exists():
                for path in run.rglob("*"):
                    if path.is_file():
                        corresponding = target / path.relative_to(run)
                        if not corresponding.exists() or path.read_bytes() != corresponding.read_bytes():
                            raise RuntimeError(f"Existing immutable run differs: {corresponding}")
            else:
                shutil.copytree(run, target)
        registry = root / "experiments/registry.jsonl"
        known = {json.loads(line)["experiment_id"] for line in registry.read_text().splitlines()} if registry.exists() else set()
        with registry.open("a", encoding="utf-8") as out:
            for line in (scratch / "experiments/registry.jsonl").read_text().splitlines():
                entry = json.loads(line)
                if entry["experiment_id"] not in known:
                    out.write(line + "\n")
                    known.add(entry["experiment_id"])
        for name in ["reports", "logs"]:
            if (scratch / name).exists():
                shutil.copytree(scratch / name, root / name, dirs_exist_ok=True)
        manifests = scratch / "data/processed"
        if manifests.exists():
            shutil.copytree(manifests, root / "data/manifests/cluster", dirs_exist_ok=True)
    holder = Experiment.__new__(Experiment)
    holder.root = root
    holder.refresh_index()
    print(f"Imported evidence; local registry contains {len(known)} independent runs")

"""Verify and unpack this campaign's selected evidence on the fixed H: drive.

The archive is temporary transport. Model weights and tensor checkpoints remain
at the cluster paths in the manifest; this never claims they were copied.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--phase", required=True, choices=["review", "final"])
    a = p.parse_args()
    archive, destination = a.archive.resolve(), a.destination.resolve()
    if destination == ROOT / "outputs" or not destination.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected a run-specific output directory in this workspace")
    if not archive.is_relative_to(ROOT / "tmp"):
        raise ValueError("Transport archive must use project run scratch")
    receipt = json.loads(a.receipt.read_text(encoding="utf-8-sig"))
    if archive.stat().st_size != receipt["bytes"] or digest(archive) != receipt["SHA256"]:
        raise ValueError("Archive bytes/hash differ from the remote readback receipt")
    manifest_name = receipt["manifest_path"]
    expected_name = "REVIEW_INPUT_MANIFEST.json" if a.phase == "review" else "SELECTED_DELIVERY_MANIFEST.json"
    if manifest_name != expected_name:
        raise ValueError("Wrong delivery phase")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        payload = z.read(manifest_name)
        if hashlib.sha256(payload).hexdigest() != receipt["manifest_SHA256"]:
            raise ValueError("Manifest hash differs from remote receipt")
        manifest = json.loads(payload)
        if manifest["delivery_phase"] != a.phase or Path(manifest["source_root"]).name != destination.name:
            raise ValueError("Manifest belongs to a different run")
        entries = {r["path"]: r for r in manifest["included"]}
        if len(entries) != len(manifest["included"]) or receipt["files"] != len(entries):
            raise ValueError("Duplicate or inconsistent manifest entries")
        names = [r.filename for r in z.infolist()]
        if len(names) != len(set(names)) or set(names) != set(entries) | {manifest_name}:
            raise ValueError("Archive membership does not match its selected manifest")
        targets = []
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or not name.parts or any(v in {".", ".."} or ":" in v or "\\" in v for v in name.parts)
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("Unsafe archive member")
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination) or target == destination or target.is_symlink():
                raise ValueError("Archive member escapes the verified campaign")
            if item.filename in entries and item.file_size != entries[item.filename]["bytes"]:
                raise ValueError("Member size mismatch")
            targets.append((item, target))
        for item, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        for name, item in entries.items():
            target = destination / name
            if target.stat().st_size != item["bytes"] or digest(target) != item["SHA256"]:
                raise ValueError("Local readback differs from remote manifest: " + name)
    evidence = destination / "evidence"
    evidence.mkdir(exist_ok=True)
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "phase": a.phase,
              "local_root": str(destination), "archive_SHA256": receipt["SHA256"],
              "manifest_SHA256": receipt["manifest_SHA256"], "verified_files": len(entries),
              "verified_bytes": sum(r["bytes"] for r in entries.values()),
              "all_selected_files_passed_local_SHA256_readback": True,
              "cluster_tensor_checkpoints_or_model_weights_copied": False}
    (evidence / ("local_" + a.phase + "_delivery_readback.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

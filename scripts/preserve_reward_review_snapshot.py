"""Keep only amended review-input files so the exact reviewed snapshot is reconstructable.

Unchanged raw evidence is shared with the final tree. The temporary transport
archive can then be removed without discarding the review's original inputs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mirror_reward_campaign_delivery import digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    a = p.parse_args()
    c, archive = a.campaign.resolve(), a.archive.resolve()
    if c == ROOT / "outputs" or not c.is_relative_to(ROOT / "outputs") or not archive.is_relative_to(ROOT / "tmp"):
        raise ValueError("Expected campaign and run-specific transport scratch")
    receipt = json.loads(a.receipt.read_text(encoding="utf-8-sig"))
    if archive.stat().st_size != receipt["bytes"] or digest(archive) != receipt["SHA256"]:
        raise ValueError("Review transport differs from the original receipt")
    mappings, amended = [], []
    with zipfile.ZipFile(archive) as z:
        manifest_bytes = z.read(receipt["manifest_path"])
        import hashlib
        if hashlib.sha256(manifest_bytes).hexdigest() != receipt["manifest_SHA256"]:
            raise ValueError("Original review manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        for row in manifest["included"]:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or any(v in {".", ".."} or ":" in v or "\\" in v for v in relative.parts):
                raise ValueError("Unsafe original manifest path")
            current = c.joinpath(*relative.parts).resolve()
            if not current.is_relative_to(c):
                raise ValueError("Original input path escaped campaign")
            same = current.is_file() and current.stat().st_size == row["bytes"] and digest(current) == row["SHA256"]
            source = current
            if not same:
                source = c / "evidence/review_input_originals" / str(relative)
                source.parent.mkdir(parents=True, exist_ok=True)
                if not source.exists():
                    with z.open(row["path"]) as original, source.open("xb") as saved:
                        shutil.copyfileobj(original, saved)
                amended.append({"original_path": row["path"], "preserved_path": source.relative_to(c).as_posix()})
            if source.stat().st_size != row["bytes"] or digest(source) != row["SHA256"]:
                raise ValueError("Cannot reconstruct original reviewed input: " + row["path"])
            mappings.append({"original_path": row["path"], "read_path": source.relative_to(c).as_posix(),
                             "bytes": row["bytes"], "SHA256": row["SHA256"]})
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "review_manifest_SHA256": receipt["manifest_SHA256"],
              "review_archive_SHA256": receipt["SHA256"], "original_review_files": len(mappings),
              "shared_unchanged_files": len(mappings) - len(amended), "preserved_amended_files": amended,
              "all_original_review_inputs_reconstructable_and_hash_verified": True, "mapping": mappings,
              "claim_boundary": "Exact selected review-input reconstruction; omitted cluster tensors/weights remain omitted."}
    target = c / "evidence/review_snapshot_reconstruction.json"
    with target.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "mapping"}, indent=2))


if __name__ == "__main__":
    main()

"""Resolve an immutable model revision; reuse the central cache before download."""
import argparse
import json
from pathlib import Path
import _bootstrap
from huggingface_hub import HfApi, snapshot_download
from src.utils.logging import write_json


def download(model_id="Qwen/Qwen3-VL-2B-Instruct", revision="main", local_only=False):
    api = HfApi()
    if local_only:
        path = snapshot_download(model_id, revision=revision, local_files_only=True)
        sha = Path(path).name
        metadata = {"id": model_id, "revision": sha, "path": path, "source": "existing_cache"}
    else:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        metadata = {"id": model_id, "revision": info.sha, "license": getattr(info.card_data, "license", None),
                    "safetensors_bytes": total, "private": info.private, "gated": info.gated}
        write_json(_bootstrap.ROOT / "reports/model_metadata.json", metadata)
        path = snapshot_download(model_id, revision=info.sha,
                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "README.md"], max_workers=2)
        metadata["path"] = path
    write_json(_bootstrap.ROOT / "reports/model_ready.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="main")
    p.add_argument("--local-only", action="store_true")
    a = p.parse_args()
    download(revision=a.revision, local_only=a.local_only)

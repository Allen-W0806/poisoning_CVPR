"""Cache only the required editor components, resolve both candidate revisions."""
import json
from pathlib import Path
import _bootstrap
from huggingface_hub import HfApi, snapshot_download
from src.utils.logging import write_json

if __name__ == "__main__":
    api = HfApi()
    metadata = {}
    for name in ["timbrooks/instruct-pix2pix", "stepfun-ai/Step1X-Edit"]:
        info = api.model_info(name, files_metadata=True)
        metadata[name] = {"id": name, "revision": info.sha, "license": getattr(info.card_data, "license", None),
                          "weight_bytes_all_variants": sum(s.size or 0 for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin")))}
        if name == "timbrooks/instruct-pix2pix":
            names = [s.rfilename for s in info.siblings]
            variant = "fp16" if all(any(n.startswith(c + "/") and n.endswith(".fp16.safetensors") for n in names)
                                    for c in ["unet", "vae", "text_encoder"]) else None
            chosen = [n for n in names if n in ["model_index.json", "README.md", "LICENSE"] or
                      (n.split("/")[0] in {"unet", "vae", "text_encoder", "tokenizer", "scheduler"} and
                       (n.endswith((".json", ".txt")) or n.endswith(".fp16.safetensors" if variant else ".safetensors") and
                        (variant or ".fp16." not in n and ".non_ema." not in n)))]
            metadata[name]["variant"] = variant
            metadata[name]["files"] = chosen
            metadata[name]["download_bytes"] = sum(s.size or 0 for s in info.siblings if s.rfilename in chosen)
            write_json(_bootstrap.ROOT / "reports/phase2_model_metadata.json", metadata)
            path = snapshot_download(name, revision=info.sha, allow_patterns=chosen, max_workers=2)
            metadata[name]["path"] = path
    write_json(_bootstrap.ROOT / "reports/phase2_model_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))

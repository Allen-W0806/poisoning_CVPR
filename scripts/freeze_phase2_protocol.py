import json
import _bootstrap
from src.data.dataset import sha256
from src.evaluation.vqa_official_compatible import reference_metadata, PROTOCOL

if __name__ == "__main__":
    root = _bootstrap.ROOT
    ready = json.loads((root / "reports/model_ready.json").read_text())
    protocol = {"name": "phase2_clean_vqa_frozen_v1", "model_id": ready["id"], "model_revision": ready["revision"],
        "system_prompt": "Answer the question using only a short word or phrase. Do not include an explanation.",
        "do_sample": False, "max_new_tokens": 64, "precision": "bf16_if_supported", "seed": 42,
        "min_pixels": 4096, "max_pixels": 262144, "micro_batch": 1,
        "extraction": "strip outer whitespace only; no label-aware or explanatory-answer parsing",
        "scorer": PROTOCOL, "reference": reference_metadata(),
        "split_hashes": {s: sha256(root / f"data/processed/vqav2_seed42_v1/{s}.json") for s in ["train", "validation", "test"]},
        "test_scope": "300 internal held-out questions from official train2014; not official test server",
        "heldout_rule": "evaluate once after development with this same frozen protocol; never used for selection"}
    path = root / "configs/phase2_vqa_protocol.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    print(path, sha256(path))

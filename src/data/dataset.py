import hashlib
import json
from pathlib import Path
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class VQADataset:
    def __init__(self, manifest, root=None, expected_split=None):
        self.manifest = Path(manifest)
        self.root = Path(root or Path.cwd()).resolve()
        self.records = json.loads(self.manifest.read_text(encoding="utf-8"))
        ids = set()
        for r in self.records:
            required = {"sample_id", "image_id", "image_path", "question", "answers", "split"}
            if not required <= r.keys():
                raise ValueError("Incomplete annotation")
            if r["sample_id"] in ids:
                raise ValueError("Duplicate sample ID")
            ids.add(r["sample_id"])
            if expected_split and r["split"] != expected_split:
                raise ValueError("Split mismatch")
            if not isinstance(r["question"], str) or not r["question"] or not r["answers"]:
                raise ValueError("Invalid question or answers")
            if not self.image_path(r).is_file():
                raise FileNotFoundError(self.image_path(r))

    def image_path(self, record):
        p = Path(record["image_path"])
        return p if p.is_absolute() else self.root / p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = dict(self.records[index])
        with Image.open(self.image_path(r)) as image:
            r["image"] = image.convert("RGB")
        return r


def validate_disjoint(*datasets):
    sets = [{str(r["image_id"]) for r in d.records} for d in datasets]
    for i, ids in enumerate(sets):
        for other in sets[i + 1:]:
            if ids & other:
                raise ValueError("Image leakage between splits")
    return True

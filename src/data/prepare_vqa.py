"""Immutable synthetic fixtures and image-disjoint subsets of official VQAv2."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from .dataset import sha256

OFFICIAL = {
    "annotations": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "questions": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
}


def dump_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def prepare_fixture(root):
    root = Path(root)
    out = root / "data" / "processed" / "synthetic_v1"
    if (out / "provenance.json").exists():
        return out
    # Exclusive version directory: never overwrite a partial or existing dataset.
    out.mkdir(parents=True, exist_ok=False)
    for j, split in enumerate(["train", "validation", "test"]):
        rows = []
        for i in range(10):
            count = i % 3 + 1
            image_id = f"synthetic_{split}_{i:03d}"
            image = Image.new("RGB", (192, 128), "white")
            draw = ImageDraw.Draw(image)
            for k in range(count):
                x = 10 + 56 * k
                y = 28 + (i + j) % 5
                draw.ellipse((x, y, x + 38, y + 38), fill=(220, 25, 25))
            path = out / f"{image_id}.png"
            image.save(path)
            rows.append({"sample_id": image_id, "image_id": image_id,
                         "image_path": path.relative_to(root).as_posix(),
                         "question": "How many red circles are in the image? Answer with a number only.",
                         "answers": [str(count)] * 10, "split": split,
                         "source": "synthetic_v1", "semantic_oracle": {"red_components": count},
                         "image_sha256": sha256(path)})
        dump_exclusive(out / f"{split}.json", rows)
    dump_exclusive(out / "provenance.json", {"dataset": "synthetic_v1", "seed": 42,
                   "purpose": "engineering smoke only; NOT VQAv2", "split_hashes": {
                       s: sha256(out / f"{s}.json") for s in ["train", "validation", "test"]}})
    return out


def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "RSI-benign-research/0.1"})
    with urllib.request.urlopen(req, timeout=90) as response, part.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    part.replace(path)


def select_subsets(annotations, questions, sizes=(1000, 200, 300), seed=42):
    """Assign every image to one split first; then approximately balance answer types."""
    qmap = {q["question_id"]: q for q in questions}
    groups = defaultdict(list)
    for a in annotations:
        groups[a["image_id"]].append(a)
    image_ids = sorted(groups)
    random.Random(seed).shuffle(image_ids)
    # Train2014 pool has ample images. Allocate disjoint image pools proportionally.
    cuts = [int(len(image_ids) * sizes[0] / sum(sizes)), int(len(image_ids) * sum(sizes[:2]) / sum(sizes))]
    pools = [image_ids[:cuts[0]], image_ids[cuts[0]:cuts[1]], image_ids[cuts[1]:]]
    global_counts = Counter(a.get("answer_type", "unknown") for a in annotations)
    result = {}
    for split, n, pool in zip(["train", "validation", "test"], sizes, pools):
        buckets = defaultdict(list)
        for image_id in pool:
            for a in groups[image_id]:
                buckets[a.get("answer_type", "unknown")].append(a)
        rng = random.Random(seed + len(split))
        chosen = []
        for category in sorted(buckets):
            values = buckets[category]
            rng.shuffle(values)
            quota = int(n * global_counts[category] / len(annotations))
            chosen.extend(values[:quota])
        selected_ids = {a["question_id"] for a in chosen}
        extra = [a for values in buckets.values() for a in values if a["question_id"] not in selected_ids]
        rng.shuffle(extra)
        chosen.extend(extra[:n - len(chosen)])
        if len(chosen) != n:
            raise ValueError(f"Insufficient official data for {split}")
        result[split] = [{"sample_id": str(a["question_id"]), "image_id": str(a["image_id"]),
             "image_path": f"data/raw/vqav2/images/COCO_train2014_{a['image_id']:012d}.jpg",
             "question": qmap[a["question_id"]]["question"],
             "answers": [r["answer"] for r in a["answers"]], "split": split,
             "source": "VQAv2-train2014-2017-04-26", "answer_type": a.get("answer_type"),
             "question_type": a.get("question_type")} for a in chosen]
    return result


def prepare_official(root, seed=42):
    root = Path(root)
    raw = root / "data/raw/vqav2"
    out = root / "data/processed/vqav2_seed42_v1"
    if out.exists():
        raise FileExistsError(f"Dataset version already exists: {out}; use or audit it, never overwrite")
    paths = {}
    for kind, url in OFFICIAL.items():
        archive = raw / Path(url).name
        fetch(url, archive)
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith(".json") and not n.startswith("__MACOSX"))
            paths[kind] = json.loads(z.read(member))
    subsets = select_subsets(paths["annotations"]["annotations"], paths["questions"]["questions"], seed=seed)
    all_rows = [r for rows in subsets.values() for r in rows]
    unique = {r["image_id"]: r for r in all_rows}

    def download_image(row):
        name = Path(row["image_path"]).name
        # TLS-valid path-style endpoint for the same official COCO image bucket.
        fetch("https://s3.amazonaws.com/images.cocodataset.org/train2014/" + name, root / row["image_path"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download_image, unique.values()))
    out.mkdir(parents=True, exist_ok=False)
    for split, rows in subsets.items():
        for row in rows:
            row["image_sha256"] = sha256(root / row["image_path"])
        dump_exclusive(out / f"{split}.json", rows)
        dump_exclusive(out / f"{split}_ids.json", {"sample_ids": [r["sample_id"] for r in rows],
                         "image_ids": sorted({r["image_id"] for r in rows})})
    dump_exclusive(out / "fixture10.json", subsets["train"][:10])
    dump_exclusive(out / "provenance.json", {"dataset": "VQAv2", "source_urls": OFFICIAL,
        "source_archive_sha256": {k: sha256(raw / Path(u).name) for k, u in OFFICIAL.items()},
        "version": "train2014-2017-04-26", "seed": seed, "split_unit": "image_id",
        "test_scope": "internal held-out subset of official train2014; not official test",
        "split_hashes": {s: sha256(out / f"{s}.json") for s in subsets},
        "answer_type_counts": {s: dict(Counter(r["answer_type"] for r in rows)) for s, rows in subsets.items()},
        "image_source": "https://s3.amazonaws.com/images.cocodataset.org/train2014/",
        "terms": "https://visualqa.org/terms.html", "coco_terms": "https://cocodataset.org/#termsofuse"})
    return out

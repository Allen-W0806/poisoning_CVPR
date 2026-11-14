"""Create an image-disjoint mini cohort from previously exposed development data."""
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_image(root, record):
    root = Path(root).resolve()
    path = (root / record["image_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes canonical project")
    if not path.is_file() or sha256(path) != record["image_sha256"]:
        raise ValueError(f"Image missing or checksum mismatch: {path}")
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    return path


def validate_cohort(root, cohort, sizes):
    ids, hashes, samples = set(), set(), set()
    for split, count in sizes.items():
        if len(cohort[split]) != count:
            raise ValueError(f"Expected {count} {split} images")
        for record in cohort[split]:
            if (str(record["image_id"]) in ids or record["image_sha256"] in hashes
                    or str(record["sample_id"]) in samples):
                raise ValueError("Duplicate ID/content within or across mini splits")
            checked_image(root, record)
            ids.add(str(record["image_id"]))
            hashes.add(record["image_sha256"])
            samples.add(str(record["sample_id"]))
    return {"image_count": len(ids), "content_count": len(hashes),
            "split_counts": {key: len(value) for key, value in cohort.items()},
            "image_ID_and_SHA256_disjoint": True, "all_images_decoded_and_rehashed": True}


def prepare_cohort(root, directory, sizes, seed):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(root / "data"):
        raise ValueError("Mini cohort must live under canonical data/")
    if directory.exists():
        cohort = {key: json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
                  for key in sizes}
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if provenance["seed"] != seed or provenance["sizes"] != sizes:
            raise ValueError("Existing cohort has a different immutable protocol")
        return cohort, validate_cohort(root, cohort, sizes), provenance
    # Deliberately never read the historical test.json or prospective final pool.
    source = root / "data/processed/vqav2_content_v2/train.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    records = sorted(records, key=lambda item: (str(item["image_id"]), str(item["sample_id"])))
    random.Random(seed).shuffle(records)
    selected, ids, hashes = [], set(), set()
    for record in records:
        if str(record["image_id"]) in ids or record["image_sha256"] in hashes:
            continue
        checked_image(root, record)
        selected.append({key: record[key] for key in
                         ["sample_id", "image_id", "image_path", "image_sha256", "question", "answers"]})
        ids.add(str(record["image_id"]))
        hashes.add(record["image_sha256"])
        if len(selected) == sum(sizes.values()):
            break
    if len(selected) != sum(sizes.values()):
        raise ValueError("Insufficient verified distinct development images; no download attempted")
    cohort, offset = {}, 0
    for split, count in sizes.items():
        cohort[split] = selected[offset:offset + count]
        offset += count
    audit = validate_cohort(root, cohort, sizes)
    directory.mkdir(parents=True, exist_ok=False)
    for split, rows in cohort.items():
        write_json(directory / f"{split}.json", rows, exclusive=True)
    provenance = {"seed": seed, "sizes": sizes, "parent_manifest": str(source.relative_to(root)),
                  "parent_SHA256": sha256(source), "scope": "DEVELOPMENT ONLY",
                  "test_meaning": "held out within this run; historically exposed parent training pool",
                  "historical_test_or_prospective_pool_opened": False,
                  "copied_image_bytes": 0, "downloaded_bytes": 0}
    write_json(directory / "provenance.json", provenance, exclusive=True)
    return cohort, audit, provenance

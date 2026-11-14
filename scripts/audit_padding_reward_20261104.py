"""Pixel-only audit of exposed validation images; never reads sealed evaluation data."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def content_box(image, size=(512, 512)):
    contained = ImageOps.contain(image, size, Image.Resampling.BICUBIC)
    x, y = round((size[0] - contained.width) / 2), round((size[1] - contained.height) / 2)
    return (x, y, x + contained.width, y + contained.height), contained


def rgb_metrics(original, edited, box):
    a, b = np.asarray(original, dtype=np.float64) / 255., np.asarray(edited, dtype=np.float64) / 255.
    x0, y0, x1, y1 = box
    mask = np.zeros(a.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = True
    weights = np.array([.2126, .7152, .0722])
    linear = lambda rgb: np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    ca, cb = a[mask], b[mask]
    chroma = lambda rgb: rgb / np.maximum(rgb.sum(-1, keepdims=True), .03)
    return {
        "content_fraction": float(mask.mean()),
        "full_encoded_luma_delta": float(((b - a) @ weights).mean()),
        "content_encoded_luma_delta": float(((cb - ca) @ weights).mean()),
        "content_linear_luma_delta": float(((linear(cb) - linear(ca)) @ weights).mean()),
        "content_chromaticity_MAE": float(np.abs(chroma(linear(cb)) - chroma(linear(ca))).mean()),
        "padding_MAE": float(np.abs(b[~mask] - a[~mask]).mean()) if (~mask).any() else 0.,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-run", default="outputs/pilot_run_20261104T030649Z_s42")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prior = root / args.old_run
    cohort = json.loads((prior / "cohort.json").read_text())
    validation = {str(r["sample_id"]): r for r in cohort["validation"]}
    phases = {"shared_initial_validation", "ddpo_only/validation/round_3",
              "rule_rsi/validation/round_3", "continuation/rl_rsi/validation/round_3"}
    rows = []
    available = set()
    for line in (prior / "observations.jsonl").read_text().splitlines():
        obs = json.loads(line)
        available.add(obs["phase"])
        if obs["phase"] not in phases:
            continue
        record = validation[str(obs["sample_id"])]
        with Image.open(root / record["image_path"]) as raw:
            raw = raw.convert("RGB")
            box, contained = content_box(raw)
        with Image.open(prior / obs["source_visual_path"]) as source, Image.open(prior / obs["image_path"]) as edited:
            source, edited = source.convert("RGB"), edited.convert("RGB")
            if not np.array_equal(np.asarray(source.crop(box)), np.asarray(contained)):
                raise AssertionError("Content box does not reconstruct saved padded source")
            row = {"phase": obs["phase"], "sample_id": obs["sample_id"], "content_box": box,
                   **rgb_metrics(source, edited, box)}
        rows.append(row)
    summary = {}
    for phase in sorted(phases):
        group = [r for r in rows if r["phase"] == phase]
        if len(group) != 16:
            raise ValueError(f"Expected 16 {phase}; observed {len(group)}. Available: {sorted(available)}")
        summary[phase] = {"n": len(group), **{key: float(np.mean([r[key] for r in group]))
                          for key in group[0] if key not in {"phase", "sample_id", "content_box"}}}
    result = {"scope": "exposed development validation only; no historical/prospective final pool",
              "content_box_verified_against_saved_source": True, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

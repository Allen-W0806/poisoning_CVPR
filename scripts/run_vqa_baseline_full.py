import argparse
from collections import defaultdict
import json
import time
import numpy as np
import torch
import _bootstrap
from _phase2 import phase2_run, mark_latest
from src.data.dataset import VQADataset, sha256
from src.evaluation.vqa_official_compatible import score_answer, reference_metadata
from src.models.qwen_loader import QwenVQA
from src.evaluation.cost_tracker import CostTracker
from src.utils.logging import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    a = parser.parse_args()
    root = _bootstrap.ROOT
    protocol_path = root / "configs/phase2_vqa_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = sha256(protocol_path)
    assert protocol["reference"] == reference_metadata()
    assert sha256(root / "src/evaluation/_vendor/official_vqa.py") == protocol["reference"]["converted_sha256"]
    path = root / f"data/processed/vqav2_seed42_v1/{a.split}.json"
    assert sha256(path) == protocol["split_hashes"][a.split]
    data = VQADataset(path, root, a.split)
    if a.split == "test":
        development = json.loads((root / "reports/phase2_latest_vqa_validation.json").read_text())
        dev_metrics = json.loads((__import__('pathlib').Path(development["path"]) / "metrics.json").read_text())
        if dev_metrics["protocol_sha256"] != protocol_hash:
            raise ValueError("Held-out requires completed development under identical frozen protocol")
    with phase2_run("qwen_baseline_" + a.split, {"protocol": protocol, "protocol_sha256": protocol_hash,
                    "split": a.split, "samples": len(data)}, (a.split,), qwen=True) as run:
        if a.split == "test":
            with (root / "reports/phase2_test_opened.json").open("x") as f:
                json.dump({"run_id": run.id, "protocol_sha256": protocol_hash}, f)
        model = QwenVQA(model_id=protocol["model_id"], revision=protocol["model_revision"], device="cuda",
                        max_new_tokens=protocol["max_new_tokens"], seed=protocol["seed"],
                        min_pixels=protocol["min_pixels"], max_pixels=protocol["max_pixels"])
        assert model.system_prompt == protocol["system_prompt"]
        cost = CostTracker("cuda")
        rows, latencies = [], []
        with (run.path / "predictions.jsonl").open("x") as out:
            for i in range(len(data)):
                r = data[i]
                torch.cuda.synchronize()
                started = time.perf_counter()
                raw = model.answer([r["image"]], [r["question"]])[0]
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                result = {"sample_id": r["sample_id"], "image_id": r["image_id"], "question": r["question"],
                          "answer_type": r.get("answer_type"), "latency_seconds": latency, **score_answer(raw, r)}
                rows.append(result)
                latencies.append(latency)
                out.write(json.dumps(result) + "\n")
                out.flush()
                if i % 25 == 0:
                    run.event("progress", {"completed": i + 1, "samples": len(data)})
        repeated = [model.answer([data[i]["image"]], [data[i]["question"]])[0] for i in range(3)] if a.split == "validation" else []
        if repeated:
            assert repeated == [r["raw_answer"] for r in rows[:3]], "Greedy replay differs"
        groups = defaultdict(list)
        for r in rows:
            groups[r["answer_type"]].append(r["score"])
        cost.model_calls = model.calls
        cost.evaluation_calls = len(rows)
        metrics = {"protocol_sha256": protocol_hash, "samples": len(rows), "split": a.split,
            "official_vqa_accuracy": float(np.mean([r["score"] for r in rows])),
            "accuracy_percent": float(100 * np.mean([r["score"] for r in rows])),
            "by_answer_type": {k: {"samples": len(v), "accuracy": float(np.mean(v))} for k, v in groups.items()},
            "mean_latency_seconds": float(np.mean(latencies)), "latency_p95_seconds": float(np.percentile(latencies, 95)),
            "greedy_repeat_check": True if repeated else "not repeated; held-out consumed once",
            "model": model.metadata(), "error_cases": [r for r in rows if r["score"] < 1],
            "prediction_file": str(run.path / "predictions.jsonl"), **cost.snapshot(len(rows))}
        run.metrics(metrics)
        run.finish(summary=f"Complete {a.split} subset, pinned official evaluator, original questions/labels, no tuning on test.")
        mark_latest("vqa_" + a.split, run)
        print(json.dumps({"run": run.id, "samples": len(rows), "accuracy": metrics["official_vqa_accuracy"]}))

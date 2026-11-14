"""Offline read-only security fixture aggregation; use run_clean_baseline for VQA."""
import argparse
import json
from pathlib import Path
import _bootstrap
from _runs import new_run
from src.evaluation.security_metrics import security_metrics
from src.data.dataset import sha256
from src.evaluation.cost_tracker import CostTracker
from src.utils.logging import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True)
    p.add_argument("--fixture-id", required=True)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    fixture = Path(a.fixture)
    cost = CostTracker("cpu")
    rows = json.loads(fixture.read_text())
    result = security_metrics(rows, a.fixture_id, a.synthetic)
    with new_run("read_only_security_eval", {"fixture": str(fixture), "fixture_id": a.fixture_id,
                 "fixture_sha256": sha256(fixture), "synthetic": a.synthetic}) as run:
        write_json(run.path / "input_observations.json", rows)
        cost.evaluation_calls = 1
        result.update(cost.snapshot(len(rows)))
        result.update(image_quality=None, semantic_validity_rate=None,
                      input_quality_reason="No per-image measurements in this observation fixture",
                      clean_vqa_accuracy=None, clean_vqa_reason="Use separate clean VQA evaluation")
        run.metrics(result)
        run.finish(summary="Aggregate pre-existing independent observations; no training or image/label changes.")
        print(run.path)

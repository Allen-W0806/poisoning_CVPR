import argparse
import json
import _bootstrap
from _runs import new_run, model_options
from src.data.dataset import VQADataset
from src.data.prepare_vqa import prepare_fixture
from src.models.qwen_loader import QwenVQA
from src.evaluation.vqa_eval import evaluate_vqa, SyntheticOracleVQA
from src.evaluation.cost_tracker import CostTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/vqav2_seed42_v1/validation.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--oracle-fixture", action="store_true")
    parser.add_argument("--answer-style", choices=["short", "raw"], default="short")
    args = parser.parse_args()
    path = prepare_fixture(_bootstrap.ROOT) / "test.json" if args.oracle_fixture else _bootstrap.ROOT / args.dataset
    data = VQADataset(path, _bootstrap.ROOT)
    with new_run("oracle_clean_smoke" if args.oracle_fixture else "qwen_clean_baseline",
                 {"limit": args.limit, "device": args.device, "oracle_fixture": args.oracle_fixture,
                  "answer_style": args.answer_style}, path,
                 include_model=not args.oracle_fixture) as run:
        model = SyntheticOracleVQA() if args.oracle_fixture else QwenVQA(**model_options(args.device), answer_style=args.answer_style)
        cost = CostTracker("cpu" if args.oracle_fixture else model.device)
        result = evaluate_vqa(model, data, batch_size=1, limit=args.limit)
        if not args.oracle_fixture:
            result["model"] = model.metadata()
        cost.model_calls = 0 if args.oracle_fixture else model.calls
        result["oracle_calls"] = min(args.limit, len(data)) if args.oracle_fixture else 0
        cost.evaluation_calls = 1
        result.update(cost.snapshot(result["samples"]))
        result["security_metrics"] = {"trigger_conditioned_behavior_rate": None, "false_activation_rate": None,
            "detection_accuracy": None, "reason": "No authorized pre-existing real security fixture/checkpoint supplied"}
        result["image_quality"] = {"mse": 0.0, "scope": "unmodified inputs"}
        result["semantic_validity_rate"] = 1.0
        run.metrics(result)
        run.finish(summary="Clean inputs only; labels/questions unchanged. Limited preliminary evaluation; no model quality conclusion.")
        print(json.dumps({"run": str(run.path), "clean_vqa_consensus": result["clean_vqa_consensus"], "samples": result["samples"]}))


if __name__ == "__main__":
    main()

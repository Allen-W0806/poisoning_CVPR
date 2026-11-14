import argparse
import json
import _bootstrap
from _runs import new_run, model_options
from src.data.dataset import VQADataset
from src.data.prepare_vqa import prepare_fixture
from src.models.qwen_loader import QwenVQA
from src.evaluation.vqa_eval import normalize_answer
from src.evaluation.cost_tracker import CostTracker


def main(device="auto"):
    path = prepare_fixture(_bootstrap.ROOT) / "validation.json"
    data = VQADataset(path, _bootstrap.ROOT, "validation")
    with new_run("qwen_smoke", {**model_options(device), "microbatch": 1, "batched_check": 2}, path, include_model=True) as run:
        model = QwenVQA(**model_options(device))
        cost = CostTracker(model.device)
        records = [data[i] for i in range(2)]
        sequential = [model.answer([r["image"]], [r["question"]])[0] for r in records]
        first_peak = cost.snapshot(2)["peak_gpu_memory_bytes"]
        # Batch size two is attempted only after measured batch-one memory.
        if model.device.type == "cuda":
            import torch
            if first_peak * 2 > torch.cuda.get_device_properties(model.device).total_memory * .85:
                raise RuntimeError("Batch two deferred: insufficient measured memory margin")
        batched = model.answer([r["image"] for r in records], [r["question"] for r in records])
        repeat = [model.answer([r["image"]], [r["question"]])[0] for r in records]
        assert repeat == sequential, "Greedy deterministic replay differs"
        assert [normalize_answer(x) for x in batched] == [normalize_answer(x) for x in sequential], "Batch padding/preprocessing discrepancy"
        correctness = [normalize_answer(a) == r["answers"][0] for a, r in zip(sequential, records)]
        assert all(correctness), f"Fixture correctness failed: {sequential}"
        cost.model_calls = model.calls
        cost.evaluation_calls = 1
        metrics = {**cost.snapshot(6), "sequential": sequential, "batched": batched, "repeat": repeat,
                   "correctness": correctness, "batch_one_peak_memory_bytes": first_peak, "model": model.metadata()}
        run.metrics(metrics)
        run.finish(summary="Real frozen Qwen3-VL: two image questions, greedy replay, left-padded batch check and correctness.")
        print(json.dumps({"run": str(run.path), "answers": sequential, "cost": cost.snapshot(6)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    main(args.device)

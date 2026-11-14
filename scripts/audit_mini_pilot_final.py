"""CPU-only final evidence audit: identity proxy, device agreement and equivalent rules."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected an existing mini-pilot output")
    state = json.loads((output / "run_state.json").read_text())
    if state["status"] != "COMPLETED":
        raise RuntimeError("Run this read-only audit only after the five-arm comparison completes")
    import torch
    import numpy as np
    from PIL import Image
    from src.mini_pilot.data import sha256, write_json
    from src.mini_pilot.engine import compare_trees
    from src.mini_pilot.reward import ProxyRewardCalculator
    from src.mini_pilot.reporting import mean_metrics, paired_bootstrap
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    directory = output / "final_evidence_audit"
    directory.mkdir(exist_ok=False)
    (directory / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    protocol = json.loads((output / "protocol.json").read_text())
    cohort = json.loads((output / "cohort.json").read_text())
    recorded = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    reward = ProxyRewardCalculator(ROOT, device="cpu")
    original_resources = json.loads((output / "reward_resources.json").read_text())
    if any(reward.receipt[key] != original_resources[key]
           for key in ["CLIP_revision", "LPIPS_calibration_SHA256", "AlexNet_SHA256"]):
        raise RuntimeError("A frozen reward resource changed before the final evidence audit")
    identity, measured = [], {}
    def rgb_luma(image):
        values = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.
        return float((values * np.array([.2126, .7152, .0722])).sum(axis=2).mean())
    for record in cohort["validation"]:
        with Image.open(output / "visuals/sources" / f"{record['sample_id']}.png") as image:
            source = image.convert("RGB")
        identity.append({"sample_id": record["sample_id"], **reward.measure(source, source, protocol["target_prompt"])})
    discrepancies = []
    for method in ["frozen", "ddpo_only", "rule_rsi", "rl_rsi"]:
        candidates = (["shared_initial_validation"] if method == "frozen" else
                      [f"continuation/{method}/validation/round_3", f"{method}/validation/round_3"])
        rows = next((rows for phase in candidates if len(rows := [row for row in recorded if row["phase"] == phase]) == 16), None)
        if rows is None or len({row["sample_id"] for row in rows}) != 16:
            raise ValueError(f"Missing complete final validation evidence for {method}")
        lookup = {row["sample_id"]: row for row in rows}
        measured[method] = []
        for record in cohort["validation"]:
            row = lookup[record["sample_id"]]
            if sha256(output / row["image_path"]) != row["image_SHA256"]:
                raise ValueError("Saved validation edit changed")
            with Image.open(output / row["source_visual_path"]) as original:
                source = original.convert("RGB")
            with Image.open(output / row["image_path"]) as image:
                edit = image.convert("RGB")
            metric = reward.measure(source, edit, protocol["target_prompt"])
            metric["RGB_luma_change"] = rgb_luma(edit) - rgb_luma(source)
            measured[method].append({"sample_id": record["sample_id"], **metric})
            discrepancies.append({"arm": method, "sample_id": record["sample_id"],
                                  **{key: abs(row[key] - metric[key]) for key in ["CLIP_cosine", "LPIPS", "SSIM", "common_reward"]}})
    paired = {method: dict(zip(["mean_edit_minus_identity", "CI95_low", "CI95_high"], paired_bootstrap(rows, identity)))
              for method, rows in measured.items()}
    checks = {}
    for method, arm in state["arms"].items():
        if sha256(output / arm["checkpoint"] / "training_state.pt") != arm["checkpoint_SHA256"]:
            raise ValueError(f"Final checkpoint checksum changed: {method}")
    rule_actions_path = output / "rule_actions.jsonl"
    rule_actions = rule_actions_path.read_text().splitlines() if rule_actions_path.exists() else []
    if not rule_actions:
        first = torch.load(output / state["arms"]["ddpo_only"]["checkpoint"] / "training_state.pt", map_location="cpu", weights_only=False)
        second = torch.load(output / state["arms"]["rule_rsi"]["checkpoint"] / "training_state.pt", map_location="cpu", weights_only=False)
        keys = ["adapter", "optimizer", "rng_torch", "rng_cuda", "step", "version", "contract"]
        checks["untriggered_rule_vs_DDPO_100_update_state_comparison"] = compare_trees(
            {key: first[key] for key in keys}, {key: second[key] for key in keys})
        checks["comparison_scope"] = ("Observed equivalent settings; adapter/Adam/Torch+CUDA RNG/step/version/contract. "
                                      "Trajectory IDs differ by arm by design; full per-step raw gradient tensors were not retained. "
                                      "This is not an independent seed or a production BF16 repair.")
    else:
        checks["rule_vs_DDPO_equivalence"] = "NOT APPLICABLE: rule actually changed LR"
    maxima = {key: max(row[key] for row in discrepancies) for key in ["CLIP_cosine", "LPIPS", "SSIM", "common_reward"]}
    result = {"status": "MEASURED_READ_ONLY_AUDIT", "scope": "VALIDATION ONLY; CPU; no training-setting change",
              "identity": identity, "identity_mean": mean_metrics(identity), "edited_validation": measured,
              "edited_means": {method: mean_metrics(rows) for method, rows in measured.items()},
              "paired_vs_identity": paired, "CPU_vs_original_GPU_max_absolute_difference": maxima,
              "mean_RGB_luma_change": {method: sum(row["RGB_luma_change"] for row in rows)/len(rows) for method, rows in measured.items()},
              "RGB_luma_definition": "Mean encoded RGB weighted 0.2126/0.7152/0.0722, [0,1]; no gamma linearization; post-hoc direction diagnostic, not instruction-success accuracy",
              "CPU_GPU_agree_at_1e_minus_5": all(value <= 1e-5 for value in maxima.values()),
              "state_comparisons": checks, "GPU_seconds": 0, "CPU_wall_seconds": time.monotonic() - started,
              "Slurm_job_ID": os.getenv("SLURM_JOB_ID"), "ASR": "NOT TESTED"}
    write_json(directory / "results.json", result, exclusive=True)
    lines = ["# Final read-only proxy and numerical audit", "",
             "All values below were recomputed on CPU from saved validation images. The same calibrated reward evaluates "
             "the unedited source and final edits. This audit did not alter training, controller feedback or checkpoint selection.", "",
             "| Reference / method | Common reward | CLIP cosine | LPIPS | SSIM |", "|---|---:|---:|---:|---:|"]
    for name, metric in [("Unedited source (diagnostic)", result["identity_mean"]), *result["edited_means"].items()]:
        lines.append(f"| {name} | {metric['common_reward']:.6f} | {metric['CLIP_cosine']:.6f} | {metric['LPIPS']:.6f} | {metric['SSIM']:.6f} |")
    lines += ["", "The identity score is a proxy-validity diagnostic, not a sixth trained arm or instruction-success metric.", "",
              "Post-hoc signed RGB luma changes: " + json.dumps(result["mean_RGB_luma_change"]) + ". " + result["RGB_luma_definition"] + ".", "",
              "CPU/GPU metric differences and the untriggered-rule state comparison are fully retained in results.json. "
              "Exact and tolerance outcomes are separate; a rule that never activated is not evidence about an active heuristic.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"identity_mean": result["identity_mean"], "edited_means": result["edited_means"],
                      "CPU_GPU_differences": maxima, "state_checks": checks, "CPU_wall_seconds": result["CPU_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

"""Only report measured lighting-v2 endpoints, with explicit one-seed scope."""
import csv
import json
from pathlib import Path

from .reporting import METHODS, paired_bootstrap, display

METRICS = ["common_reward", "CLIP_cosine", "CLIP_source", "CLIP_delta", "LPIPS", "SSIM",
           "brightness_progress", "content_encoded_luma_delta", "content_linear_luma_delta",
           "content_chromaticity_MAE", "legacy_common_reward", "centered_CLIP_LPIPS"]


def mean_metrics(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in METRICS}


def generate_reports(output_dir, protocol, state):
    out = Path(output_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline = state.get("assessment", {}).get("frozen")
    for method in protocol["methods"]:
        arm = state.get("arms", {}).get(method, {})
        observed = state.get("assessment", {}).get(method)
        row = {"Method": METHODS[method], "arm": method, "training_seed": 42,
               "status": arm.get("status", "NOT TESTED"), "measured_images": len(observed or []),
               "optimizer_steps": arm.get("optimizer_steps", 0), "controller_actions": arm.get("controller_actions", 0),
               "controller_updates": arm.get("controller_updates", 0), **{k: None for k in METRICS},
               "delta_vs_frozen": None, "source_CI95_low": None, "source_CI95_high": None,
               "ASR": "NOT TESTED", "Delta_Clean": "NOT TESTED", "independent_seed_CI": "NOT ESTIMABLE (n=1)",
               "reward_version": "lighting_v2_20261104", "scope": "BENIGN DEVELOPMENT ONLY"}
        if observed:
            row.update(mean_metrics(observed))
            if baseline:
                delta, low, high = paired_bootstrap(observed, baseline, 42)
                row.update(delta_vs_frozen=delta, source_CI95_low=low, source_CI95_high=high)
        rows.append(row)
    with (out / "BASELINE_COMPARISON.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: display(v) for k, v in r.items()} for r in rows)
    cols = ["Method", "status", "optimizer_steps", "common_reward", "LPIPS", "SSIM",
            "content_encoded_luma_delta", "content_chromaticity_MAE", "ASR"]
    table = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    table += "\n".join("| " + " | ".join(str(display(r[k])) for k in cols) + " |" for r in rows)
    context = ("Only seed42. Source-image bootstrap intervals do not estimate training-seed uncertainty. "
        "Frozen/Fixed share the same observation by user choice. Development mini-assessment was previously exposed. "
        "No poisoned-victim training, ASR, clean accuracy or independent semantic labels.\n\n"
        "Reward v2 = 2*(CLIP_edit-CLIP_source) + brightness_progress - color_cost - LPIPS. "
        "The new task is explicit +3% linear-light exposure. Lighting/color use original content boxes; "
        "CLIP/LPIPS/SSIM retain full canvases. Old and new raw reward scales are not comparable. "
        "DDIM50 eta1, guidance5, image guidance1.5, rank16/alpha32, LR5e-6, clipping0.5. "
        "Full FP32 computation reuses upcast cached FP16-variant weights; BF16 repair is not asserted.\n")
    (out / "BASELINE_COMPARISON.md").write_text(f"# Lighting-v2 comparison\n\nStatus: {state['status']}\n\n{context}\n{table}\n")
    (out / "VALIDATION.json").write_text(json.dumps(state.get("validation", {}), indent=2) + "\n")
    (out / "NEXT_STEPS.md").write_text("# Next steps\n\nRead run_state.json, actual Slurm accounting and raw source-paired results. "
        "No v2 task-quality success, BF16 fix or RL-RSI superiority is inferred from completion alone. "
        "Victim endpoints remain NOT TESTED until real victim experiments finish.\n")


def make_figures(output_dir, state):
    # Export from saved evidence in the existing Windows Matplotlib environment.
    out = Path(output_dir) / "figures"
    out.mkdir(exist_ok=True)
    (out / "PLOTTING_STATUS.md").write_text("CPU export pending; no model rerun required.\n")
    state["figures_status"] = "PENDING_CPU_EXPORT"

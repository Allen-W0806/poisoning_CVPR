"""CPU-only exports from saved evidence; never rerun a model or select images."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--scratch", required=True, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(a.scratch / "mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    r = a.campaign
    d = json.loads((r / "diagnostics/debug_results.json").read_text())
    out = r / "figures"
    out.mkdir(exist_ok=True)
    settings = ["legacy10_g75_eta1", "steps50_g75_eta1", "steps50_g5_eta1", "steps50_g5_eta0"]
    labels = ["Original", "DDIM10, g=7.5, eta=1", "DDIM50, g=7.5, eta=1", "DDIM50, g=5, eta=1", "DDIM50, g=5, eta=0"]
    ids = [x["sample_id"] for x in d["quality"] if x["setting"] == settings[0]]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    for i, sample in enumerate(ids):
        paths = [r / f"diagnostics/quality_images/{settings[0]}/{sample}_source.png"] + [
                 r / f"diagnostics/quality_images/{s}/{sample}_edited.png" for s in settings]
        for j, path in enumerate(paths):
            axes[i, j].imshow(Image.open(path))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0: axes[i, j].set_title(labels[j], fontsize=10)
            if j == 0: axes[i, j].set_ylabel("ID " + sample, fontsize=9)
    fig.suptitle("Prespecified first four development-validation sources; frozen editor, seed42\n"
                 "Full padded canvases shown; lighting/color metrics exclude padding", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "editing_parameter_audit.png", dpi=130)
    fig.savefig(out / "editing_parameter_audit.pdf")
    plt.close(fig)
    variants = ["identity", "exposure_plus3pct", "exposure_minus3pct", "channel_swap", "unrelated_image"]
    vlabels = ["Identity", "+3% exposure", "-3% exposure", "Channel swap", "Unrelated image"]
    v3 = json.loads((r / "lighting_v3_calibration.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, records, key, title in zip(axes, [d["calibration"], d["calibration"], v3["controls"]],
            ["legacy_common_reward", "common_reward", "common_reward"],
            ["Legacy absolute CLIP - LPIPS", "Rejected lighting-v2 objective", "Lighting-v3 mean objective"]):
        for i, variant in enumerate(variants):
            values = [x[key] for x in records if x["variant"] == variant]
            ax.bar(i, np.mean(values), color="#3971a4" if i < 3 else "#a86655", alpha=.7)
            ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color="black", s=16, zorder=3)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(5), vlabels, rotation=25, ha="right")
        ax.set_ylabel("Reward (objective-specific scale)")
        ax.set_title(title)
    fig.suptitle("Calibration controls: first four TRAIN sources; dots are sources, not training seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out / "reward_calibration.png", dpi=160)
    fig.savefig(out / "reward_calibration.pdf")
    plt.close(fig)
    state_path = r / "internal_pilot/run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        styles = [("frozen", "Frozen = Fixed", "#777777", "--"), ("ddpo_only", "DDPO-only", "#3779af", "-"),
                  ("rule_rsi", "Rule RSI", "#40a076", ":"), ("rl_rsi", "RL-RSI", "#bd623b", "-")]
        for key, label, color, line in styles:
            rows = state.get("validation", {}).get(key, [])
            if rows:
                steps = [0, 34, 67, 100][:len(rows)] if key == "frozen" else [x["optimizer_step"] for x in rows]
                ax.plot(steps, [x["common_reward"] for x in rows],
                        marker="o", color=color, linestyle=line, label=label)
        ax.axhline(0, color="black", lw=.7, label="Pixel identity (v3=0)")
        ax.set_xlabel("Generator updates in trained arms (Frozen/Fixed reference)")
        ax.set_ylabel("Common lighting-v3 validation reward")
        ax.set_title("Seed42; 16 development-validation sources; no seed uncertainty")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "lighting_v3_validation.png", dpi=170)
        fig.savefig(out / "lighting_v3_validation.pdf")
        plt.close(fig)
        if state.get("status") == "COMPLETED":
            from matplotlib.backends.backend_pdf import PdfPages
            groups = ["frozen", "ddpo_only", "rule_rsi", "rl_rsi"]
            names = ["Original", "Frozen = Fixed", "DDPO-only", "Rule RSI", "RL-RSI"]
            with PdfPages(out / "final_editing_assessment_full16.pdf") as pages:
                for block in range(4):
                    cases = state["assessment"]["frozen"][4*block:4*block+4]
                    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
                    for i, record in enumerate(cases):
                        paths = [r / "internal_pilot" / record["source_visual_path"]]
                        for group in groups:
                            obs = state["assessment"][group][4*block+i]
                            if obs["sample_id"] != record["sample_id"]:
                                raise ValueError("Development-assessment pairing changed")
                            paths.append(r / "internal_pilot" / obs["image_path"])
                        for j, image_path in enumerate(paths):
                            axes[i,j].imshow(Image.open(image_path))
                            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
                            if i==0: axes[i,j].set_title(names[j],fontsize=11)
                            if j==0: axes[i,j].set_ylabel(record["sample_id"],fontsize=9)
                    fig.suptitle(f"Development-assessment sources {4*block+1}-{4*block+4} of 16, fixed cohort order\n"
                                 "Seed42, 100 updates per trained arm; full padded canvases",fontsize=12)
                    fig.tight_layout(rect=(0,0,1,.95))
                    pages.savefig(fig)
                    if block==0:
                        fig.savefig(out / "final_editing_assessment.png",dpi=140)
                        fig.savefig(out / "final_editing_assessment.pdf")
                    plt.close(fig)
    (out / "FIGURE_SCOPE.md").write_text(
        "# Figure scope\n\nReal saved generated images and measured diagnostics, not conceptual artwork. "
        "Four diagnostic images are the preregistered first four validation sources. "
        "Calibration uses first four train sources. No visual sample was selected by quality. "
        "The first editing grid uses the first four development-assessment sources; the full16 PDF shows all16 "
        "in fixed cohort order when the primary run is complete. Photos are inherently raster; PDF labels/charts are vector. "
        "No ASR or independent semantics inferred.\n")
    print(json.dumps({"figure_dir": str(out), "files": sorted(p.name for p in out.iterdir())}))


if __name__ == "__main__":
    main()

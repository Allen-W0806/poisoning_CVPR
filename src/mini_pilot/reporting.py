"""Evidence-derived reports. Missing measurements never become success claims."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

METHODS = {"frozen": "Frozen", "fixed": "Fixed Editing (equivalent control)",
           "ddpo_only": "DDPO-only", "rule_rsi": "Rule-based RSI", "rl_rsi": "RL-RSI"}
METRICS = ["common_reward", "CLIP_cosine", "LPIPS", "SSIM"]


def mean_metrics(observations):
    return {key: sum(row[key] for row in observations) / len(observations) for key in METRICS}


def paired_bootstrap(first, second, seed=42):
    import numpy as np
    # Source-image sampling, not independent training-seed uncertainty.
    if [row["sample_id"] for row in first] != [row["sample_id"] for row in second]:
        raise ValueError("Paired comparison requires identical source ordering")
    differences = np.array([a["common_reward"] - b["common_reward"] for a, b in zip(first, second)])
    draws = np.random.default_rng(seed).integers(0, len(differences), size=(10000, len(differences)))
    low, high = np.quantile(differences[draws].mean(1), [.025, .975])
    return float(differences.mean()), float(low), float(high)


def display(value):
    return "NOT TESTED" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def generate_reports(output_dir, protocol, state):
    directory = Path(output_dir) / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    rows, baseline = [], state.get("assessment", {}).get("frozen")
    for method in protocol["methods"]:
        arm = state.get("arms", {}).get(method, {})
        observations = state.get("assessment", {}).get(method)
        row = {"Method": METHODS[method], "arm": method, "status": arm.get("status", "NOT TESTED"),
               "training_seed": protocol["seed"], "measured_assessment_images": len(observations or []),
               "optimizer_steps": arm.get("optimizer_steps", 0),
               "controller_actions": arm.get("controller_actions", 0),
               "controller_updates": arm.get("controller_updates", 0),
               **{key: None for key in METRICS}, "delta_common_vs_Frozen": None,
               "paired_CI95_low": None, "paired_CI95_high": None,
               "ASR_Proxy": "NOT TESTED", "ASR": "NOT TESTED", "Delta_Clean": "NOT TESTED",
               "scope": "DEVELOPMENT ONLY; one training seed; source-bootstrap CI",
               "physical_equivalence_control": "frozen" if method == "fixed" else ""}
        if observations:
            row.update(mean_metrics(observations))
            if baseline:
                delta, low, high = paired_bootstrap(observations, baseline, protocol["seed"])
                row.update(delta_common_vs_Frozen=delta, paired_CI95_low=low, paired_CI95_high=high)
        rows.append(row)
    with (directory / "BASELINE_COMPARISON.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: display(value) for key, value in row.items()} for row in rows)
    columns = ["Method", "status", "optimizer_steps", "common_reward", "CLIP_cosine", "LPIPS", "SSIM",
               "delta_common_vs_Frozen", "ASR"]
    table = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    table += "\n".join("| " + " | ".join(display(row[key]) for key in columns) + " |" for row in rows)
    stamp = datetime.now(timezone.utc).isoformat()
    measured = [row for row in rows if row["common_reward"] is not None]
    complete = state.get("status") == "COMPLETED"
    best = max(measured, key=lambda row: row["common_reward"]) if measured else None
    context = ("All numbers below come from saved observations. This is a benign lighting-editing pilot. "
               "CLIP cosine is semantic text alignment, not ASR, instruction correctness, or VQA accuracy. "
               "The 16 mini-test images come from a historically exposed training pool and are development assessment. "
               "Frozen edits images with unchanged pretrained weights; it is not a pixel identity baseline. "
               "Fixed shares the identical prompt, seed, weights, sampler and observations by explicit user choice; "
               "it is an equivalence control, not independent evidence.\n")
    (directory / "BASELINE_COMPARISON.md").write_text(
        f"# Baseline comparison\n\nUTC: {stamp}\n\nStatus: {state.get('status')}\n\n" + context + "\n" + table +
        "\n\nCommon evaluator: 2 × CLIP cosine − calibrated AlexNet LPIPS. "
        "All arms use this same evaluator even when RL-RSI changes its training semantic weight. "
        "95% intervals in the CSV resample paired source images (10,000 draws); "
        "one training seed cannot estimate between-seed uncertainty.\n", encoding="utf-8")
    audit = state.get("data_audit", {})
    (directory / "FAST_VALIDATION_STATUS.md").write_text(
        f"# Fast validation status\n\n- Status: {state.get('status')}\n"
        f"- Dataset audit: `{json.dumps(audit, ensure_ascii=False)}`\n"
        f"- Requested: {protocol['num_train_images']}/{protocol['num_val_images']}/{protocol['num_test_images']}\n"
        f"- Total planned updates per trained arm: {protocol['max_inner_steps']}, divided over {protocol['rsi_rounds']} rounds.\n"
        f"- Figures: {state.get('figures_status', 'NOT GENERATED')}\n"
        f"- Numerically highest measured common proxy: {best['Method'] if best else 'NOT TESTED'}"
        f"{' (partial comparison; no final ranking)' if not complete else ' (single-seed descriptive ranking only)'}.\n"
        f"- Failure: {state.get('failure', 'none recorded')}\n\n" + context, encoding="utf-8")
    numerical = state.get("numerical", {})
    (directory / "BF16_IMPACT_ASSESSMENT.md").write_text(
        "# Precision and numerical assessment\n\n"
        f"Executed precision: {protocol['precision']}. LoRA storage/linear compute and Adam moments are FP32.\n\n"
        "Full FP32 is an isolated alternative, not proof that the existing BF16 ordinary-path issue is repaired. "
        "Casting adapters alone was already present in historical failing experiments. "
        "No upstream CUDA root cause is claimed.\n\n"
        f"Ordinary repeat/save-reload-next-update witness:\n\n```json\n{json.dumps(numerical, indent=2)}\n```\n\n"
        "A passing witness covers this recorded model, source images, precision, two-update sequence and node. "
        "It does not prove all-run reproducibility or cross-hardware equality. Nonfinite values and replay failures "
        "stop training; raw gradients, postclip gradients, adapter and Adam states are retained for the witness.\n",
        encoding="utf-8")
    qualitative = state.get("qualitative_review", {"status": "NOT PERFORMED"})
    (directory / "IMAGE_EDIT_QUALITY_AUDIT.md").write_text(
        "# Image-edit quality audit\n\n" + context + "\n" + table +
        "\n\nSSIM and LPIPS use the same aspect-preserving, padded 512 × 512 source as generation. "
        "Calibrated LPIPS receives RGB [-1,1]; CLIP receives RGB PIL via its processor. "
        "Semantic preservation is UNREVIEWED until actual source/edit pairs are inspected. "
        "No visual superiority sentence is generated from an arm name.\n\n"
        f"Qualitative inspection record: {json.dumps(qualitative, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8")
    independent = None
    assessments = state.get("assessment", {})
    if assessments.get("rl_rsi") and assessments.get("ddpo_only"):
        independent = paired_bootstrap(assessments["rl_rsi"], assessments["ddpo_only"], protocol["seed"])
    decision = ("The specified pilot completed. Inspect reward/quality failures and the paired RL-RSI versus DDPO-only "
                "difference before proposing a separately budgeted multi-seed replication. "
                "Do not infer a backdoor effect or scale up automatically." if complete else
                "Complete or diagnose the recorded blocking condition first. No automatic retry or scale-up is authorized by this report.")
    (directory / "NEXT_EXPERIMENT_DECISION.md").write_text(
        "# Next experiment decision\n\n" + decision + "\n\n"
        f"RL-RSI minus DDPO-only common reward (mean, source-bootstrap 95% limits): {independent or 'NOT TESTED'}.\n\n"
        "Only measured CLIP cosine, LPIPS, SSIM and their common reward may enter an explicitly labelled "
        "development Pilot table. ASR, Delta Clean, victim fine-tuning efficacy and an independent RSI gain "
        "cannot enter Table 8 as measured facts. Cross-round memory uses actual validation proxy feedback; "
        "victim fine-tuning/ASR is NOT TESTED in this benign pilot.\n", encoding="utf-8")
    tex = [r"% Development-only benign pilot; single training seed; no ASR or Delta Clean.",
           r"\begin{tabular}{lrrrr}", r"\hline", r"Method & Common proxy $\uparrow$ & CLIP $\uparrow$ & LPIPS $\downarrow$ & SSIM $\uparrow$ \\", r"\hline"]
    for row in rows:
        values = ["---" if row[key] is None else f"{row[key]:.6f}" for key in METRICS]
        tex.append(row["Method"] + " & " + " & ".join(values) + r" \\")
    tex.extend([r"\hline", r"\end{tabular}", "% Fixed is an equivalent Frozen control sharing observations.",
                "% Mini-test means this run's development assessment, not the sealed final pool."])
    (directory / "BASELINE_COMPARISON.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return rows


def make_figures(output_dir, state):
    # Figure dependencies belong to CPU reporting; never discard completed
    # measurements or extend a GPU allocation just to install a plotting package.
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        directory = Path(output_dir) / "figures"
        directory.mkdir(exist_ok=True)
        state["figures_status"] = "PENDING_CPU_REPORTING_MATPLOTLIB_UNAVAILABLE"
        (directory / "PLOTTING_STATUS.md").write_text(
            "# Plotting status\n\nMeasurements are retained. Matplotlib is unavailable in this runtime. "
            "Use a CPU Python with Matplotlib and Pillow to call src.mini_pilot.reporting.make_figures "
            "on the saved run_state.json and saved source/edit images. No model execution is required.\n",
            encoding="utf-8")
        print(json.dumps({"figures_status": state["figures_status"], "GPU_retry_required": False}), flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    directory = Path(output_dir) / "figures"
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 3.6), constrained_layout=True)
    for method in ["frozen", "ddpo_only", "rule_rsi", "rl_rsi"]:
        points = state.get("validation", {}).get(method, [])
        if points:
            ax.plot([point["round"] for point in points], [point["common_reward"] for point in points],
                    marker="o", label=METHODS[method])
    ax.set(xlabel="Completed round", ylabel="Common validation reward (2 CLIP − LPIPS)",
           title="Benign editing pilot · one seed · development only")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(directory / "validation_reward.pdf")
    fig.savefig(directory / "validation_reward.png", dpi=180)
    plt.close(fig)
    assessment = state.get("assessment", {})
    methods = [method for method in ["frozen", "ddpo_only", "rule_rsi", "rl_rsi"] if assessment.get(method)]
    if not methods:
        return
    # First four preregistered assessment sources; no best-looking sample selection.
    samples = min(4, len(assessment[methods[0]]))
    width, height, cell = (len(methods) + 1) * 224, samples * 248 + 32, 224
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(["Source"] + [METHODS[method] for method in methods]):
        draw.text((col * cell + 5, 8), label, fill="black")
    for row in range(samples):
        paths = [assessment[methods[0]][row]["source_visual_path"]] + [assessment[m][row]["image_path"] for m in methods]
        for col, path in enumerate(paths):
            with Image.open(Path(output_dir) / path) as image:
                sheet.paste(image.convert("RGB").resize((cell, cell)), (col * cell, row * 248 + 32))
            draw.text((col * cell + 4, row * 248 + 258), assessment[methods[0]][row]["sample_id"], fill="black")
    sheet.save(directory / "source_edit_contact_sheet.png")
    state["figures_status"] = "READY"

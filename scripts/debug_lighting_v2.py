"""One-GPU calibration/quality/BF16 diagnostic on exposed train/validation only."""
import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.mini_pilot.data import write_json, sha256
from scripts.audit_padding_reward_20261104 import content_box


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output escapes project outputs")
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.mini_pilot.lighting_reward_v2 import LightingRewardV2, analytic_exposure
    from src.mini_pilot.engine import PilotPolicy, PilotTrainer, compare_trees
    from src.editing.pretrained_editor import PretrainedEditor
    from src.models.pretrained_editor_loader import pretrained_config
    from train_and_evaluate_pilot import INSTRUCTION, TARGET
    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated GPU required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    kernel = torch.randn(8, 8, device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))
    result = {"seed": 42, "Slurm_job": os.environ["SLURM_JOB_ID"], "scope": "development only",
              "kernel_finite": bool(torch.isfinite(kernel @ kernel).all()), "GPU": torch.cuda.get_device_name(),
              "calibration": [], "quality": [], "BF16_repaired": False}
    def save():
        write_json(out / "debug_results.json", result)
    cohort = json.loads((ROOT / "outputs/pilot_run_20261104T030649Z_s42/cohort.json").read_text())
    reward = LightingRewardV2(ROOT)
    write_json(out / "reward_definition.json", reward.receipt, exclusive=True)
    # First four train items, all selected before any new reward measurement.
    for record in cohort["train"][:4]:
        with Image.open(ROOT / record["image_path"]) as raw:
            raw = raw.convert("RGB")
            box, _ = content_box(raw)
            from PIL import ImageOps
            source = ImageOps.pad(raw, (512, 512), color="white", method=3)
        path = out / "calibration_images" / str(record["sample_id"])
        path.mkdir(parents=True, exist_ok=False)
        unrelated = cohort["train"][7]
        with Image.open(ROOT / unrelated["image_path"]) as other:
            random_image = ImageOps.pad(other.convert("RGB"), (512, 512), color="white", method=3)
        variants = {"identity": source, "exposure_plus3pct": analytic_exposure(source, box, 1.03),
                    "exposure_minus3pct": analytic_exposure(source, box, .97),
                    "channel_swap": Image.fromarray(np.asarray(source)[:, :, ::-1]), "unrelated_image": random_image}
        for name, image in variants.items():
            metric = reward.measure_with_box(source, image, TARGET, box)
            image.save(path / f"{name}.png")
            result["calibration"].append({"sample_id": record["sample_id"], "variant": name, **metric})
        inputs = reward.processor(text=[TARGET, "brighten the image", "an X-ray image of a human hand"],
                                  images=[source] * 3, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            embeds = reward.clip_model(**inputs)
            dot = (F.normalize(embeds.image_embeds.float(), dim=-1) * F.normalize(embeds.text_embeds.float(), dim=-1)).sum(-1)
            cos = F.cosine_similarity(embeds.image_embeds.float(), embeds.text_embeds.float(), dim=-1)
        result.setdefault("CLIP_checks", []).append({"sample_id": record["sample_id"],
            "prompts": [TARGET, "brighten the image", "an X-ray image of a human hand"], "cosines": cos.cpu().tolist(),
            "normalized_dot_max_difference": float((dot - cos).abs().max()),
            "image_norms": embeds.image_embeds.norm(dim=-1).cpu().tolist(),
            "text_norms": embeds.text_embeds.norm(dim=-1).cpu().tolist()})
        save()
    result["calibration_gate"] = {
        "identity_zero": all(abs(r["common_reward"]) < 1e-6 for r in result["calibration"] if r["variant"] == "identity"),
        "positive_control_above_identity": all(r["common_reward"] > 0 for r in result["calibration"] if r["variant"] == "exposure_plus3pct"),
        "darken_and_color_controls_below_identity": all(r["common_reward"] < 0 for r in result["calibration"] if r["variant"] in {"exposure_minus3pct", "channel_swap"})}
    save()
    if not all(result["calibration_gate"].values()):
        raise RuntimeError("Task reward calibration failed; no training admitted")
    editor = PretrainedEditor(pretrained_config(ROOT, precision="fp32", seed=42, num_inference_steps=50, guidance_scale=5.), "cuda")
    from diffusers import DDIMScheduler
    editor.pipe.scheduler = DDIMScheduler.from_config(editor.pipe.scheduler.config, set_alpha_to_one=False,
        steps_offset=1, clip_sample=False, timestep_spacing="leading")
    settings = [("legacy10_g75_eta1", 10, 7.5, 1.), ("steps50_g75_eta1", 50, 7.5, 1.),
                ("steps50_g5_eta1", 50, 5., 1.), ("steps50_g5_eta0", 50, 5., 0.)]
    for name, steps, guidance, eta in settings:
        for index, record in enumerate(cohort["validation"][:4]):
            with Image.open(ROOT / record["image_path"]) as raw:
                raw = raw.convert("RGB")
                box, _ = content_box(raw)
                source = editor.prepare_image(raw)
            before = time.monotonic()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                edited = editor.pipe(prompt=INSTRUCTION, image=source, num_inference_steps=steps,
                    guidance_scale=guidance, image_guidance_scale=1.5, eta=eta,
                    generator=torch.Generator(device="cuda").manual_seed(42200000 + index)).images[0]
            directory = out / "quality_images" / name
            directory.mkdir(parents=True, exist_ok=True)
            source.save(directory / f"{record['sample_id']}_source.png")
            edited.save(directory / f"{record['sample_id']}_edited.png")
            metric = reward.measure_with_box(source, edited, TARGET, box)
            result["quality"].append({"setting": name, "sample_id": record["sample_id"], "steps": steps,
                "guidance_scale": guidance, "eta": eta, "generation_seconds": time.monotonic() - before, **metric})
            save()
        print(json.dumps({"quality_setting_done": name, "elapsed": time.monotonic() - started}), flush=True)
    del editor
    gc.collect()
    torch.cuda.empty_cache()
    # Reproduce whether the user's FP32-adapter recipe actually repairs ordinary BF16 updates.
    bf = PretrainedEditor(pretrained_config(ROOT, precision="bf16", seed=42, num_inference_steps=50, guidance_scale=5.), "cuda")
    cfg = {"lora_rank": 16, "lora_alpha": 32, "adapter_compute_precision": "fp32", "gradient_checkpointing": True,
           "learning_rate": 5e-6, "num_inference_steps": 50, "eta": 1., "logical_batch_size": 1,
           "physical_micro_batch_size": 1, "gradient_accumulation_steps": 1, "max_grad_norm": .5}
    policy = PilotPolicy(bf, cfg)
    trainer = PilotTrainer(policy, 5e-6, max_grad_norm=.5)
    record = cohort["train"][0]
    with Image.open(ROOT / record["image_path"]) as raw:
        condition = policy.condition({**record, "image": raw.convert("RGB")}, INSTRUCTION)
    initial = out / "bf16_witness/initial"
    trainer.save(initial)
    witnesses = []
    for label in ["first", "repeat"]:
        trainer.restore(initial)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            image, trajectory = policy.sample(condition, 42900001, label)
        metric = reward.measure_with_box(condition["source"], image, TARGET, content_box(Image.open(ROOT / record["image_path"]))[0])
        receipt = trainer.update([(trajectory, condition, metric["common_reward"])], keep_gradients=True)
        witness = trainer.witness()
        torch.save(witness, out / f"bf16_witness/{label}.pt")
        witnesses.append(witness)
        result.setdefault("BF16_updates", []).append(receipt)
        save()
    result["BF16_ordinary_repeat"] = compare_trees(*witnesses)
    result["BF16_repaired"] = False  # first-update repeat alone cannot certify full repair
    result["BF16_gate"] = "FAIL" if not result["BF16_ordinary_repeat"]["tolerance"] else "FIRST_UPDATE_ONLY_PASS_NOT_FULL_CERTIFICATION"
    result["status"] = "COMPLETED"
    result["elapsed_seconds"] = time.monotonic() - started
    save()
    print(json.dumps({k: v for k, v in result.items() if k in {"calibration_gate", "BF16_ordinary_repeat", "BF16_gate", "status", "elapsed_seconds"}}, indent=2))


if __name__ == "__main__":
    main()

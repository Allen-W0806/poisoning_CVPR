#!/usr/bin/env python3
"""Versioned lighting-v2 benign mini-pilot. Run from the canonical project inside one Slurm allocation.

--max_inner_steps is the TOTAL optimizer-update limit per trained arm, split over
--rsi_rounds. Test means this run's held-out DEVELOPMENT assessment, not the
historical sealed final pool. Neither ASR nor victim fine-tuning is simulated.
"""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import prepare_cohort, sha256, write_json
from src.mini_pilot.reporting_v3 import METHODS, generate_reports, mean_metrics
from scripts.audit_padding_reward_20261104 import content_box

INSTRUCTION = "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."
TARGET = "a naturally lit photograph with slightly brighter lighting and unchanged objects"
PROTECTED = ["src/models/pretrained_editor_loader.py", "src/editing/pretrained_editor.py",
             "src/rl/diffusion_policy.py", "src/rl/diffusion_rl_trainer.py", "src/rl/diffusion_trajectory.py",
             "src/rl/phase5_backend.py", "src/rl/reward.py", "src/training/generator_adapter.py"]


def boolean(value):
    if str(value).lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected True or False")
    return str(value).lower() == "true"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default="data/vqav2_mini_32_16_16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_images", type=int, default=32)
    parser.add_argument("--num_val_images", type=int, default=16)
    parser.add_argument("--num_test_images", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--base_learning_rate", type=float, default=5e-6)
    parser.add_argument("--force_fp32_lora", type=boolean, default=True)
    parser.add_argument("--max_inner_steps", type=int, default=100)
    parser.add_argument("--rsi_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--wall_seconds", type=int, default=17700)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--reports_only", action="store_true")
    return parser.parse_args()


def environment(root):
    from huggingface_hub import snapshot_download
    packages = {}
    for name in ["torch", "diffusers", "transformers", "peft", "scikit-image", "numpy", "Pillow"]:
        packages[name] = importlib.metadata.version(name)
    metadata = json.loads((root / "reports/phase2_model_metadata.json").read_text())["timbrooks/instruct-pix2pix"]
    for relative in metadata["files"]:
        if relative.endswith((".safetensors", ".json", ".txt")) and not (Path(metadata["path"]) / relative).is_file():
            raise FileNotFoundError(f"Incomplete cached editor: {relative}")
    clip_path = Path(snapshot_download("openai/clip-vit-large-patch14", local_files_only=True))
    if not any((clip_path / name).is_file() for name in ["model.safetensors", "pytorch_model.bin"]):
        raise FileNotFoundError("CLIP snapshot lacks model weights")
    dependency = json.loads((root / "results/execution_remediation_20261102_v1/"
                             "lpips_dependency_verification.json").read_text())
    calibration = Path(dependency["module_root"]) / "lpips/weights/v0.1/alex.pth"
    if sha256(calibration) != dependency["calibration_sha256"] or sha256(dependency["base_file"]) != dependency["base_sha256"]:
        raise ValueError("Cached LPIPS resource verification failed")
    disk = shutil.disk_usage(root)
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "editor_revision": metadata["revision"], "editor_snapshot": metadata["path"],
            "CLIP_snapshot": str(clip_path), "LPIPS_resources_verified": True,
            "disk_free_bytes": disk.free, "estimated_retained_run_bytes": 1_500_000_000,
            "downloaded_bytes": 0, "C_drive_model_bytes": 0, "Slurm_job_ID": os.getenv("SLURM_JOB_ID"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "hostname": os.uname().nodename}


def execute(root, output, protocol, cohort, state, persist):
    import numpy as np
    import torch
    from PIL import Image
    from src.editing.pretrained_editor import PretrainedEditor
    from src.models.pretrained_editor_loader import pretrained_config
    from src.mini_pilot.engine import PilotPolicy, PilotTrainer, compare_trees, frozen_digest
    from src.mini_pilot.lighting_reward_v3 import LightingRewardV3 as ProxyRewardCalculator
    from src.mini_pilot.controller_v3 import LearnedController, controller_state
    from src.mini_pilot.reporting_v3 import make_figures

    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Execute only inside an explicit one-GPU Slurm allocation; login-node CUDA prohibited")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    started = time.monotonic()
    def check_time():
        if time.monotonic() - started > protocol["wall_seconds"]:
            raise TimeoutError("Preregistered pilot wall-time cap reached; no automatic continuation")
    # A seeded device-kernel witness precedes all model work.
    kernel = torch.randn((8, 8), device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))
    state["kernel_witness"] = {"shape": list((kernel @ kernel).shape), "finite": bool(torch.isfinite(kernel @ kernel).all()),
                               "GPU": torch.cuda.get_device_name(), "total_memory": torch.cuda.get_device_properties(0).total_memory}
    print("KERNEL_WITNESS " + json.dumps(state["kernel_witness"]), flush=True)
    torch.cuda.reset_peak_memory_stats()
    configuration = pretrained_config(root, precision=protocol["precision"], seed=protocol["seed"],
                                       num_inference_steps=protocol["num_inference_steps"], guidance_scale=5.)
    editor = PretrainedEditor(configuration, device="cuda")
    diffusion = {"lora_rank": protocol["lora_rank"], "lora_alpha": protocol["lora_rank"] * 2,
                 "adapter_compute_precision": "fp32", "gradient_checkpointing": True,
                 "learning_rate": protocol["base_learning_rate"], "num_inference_steps": protocol["num_inference_steps"],
                 "eta": 1., "logical_batch_size": 1, "physical_micro_batch_size": 1,
                 "gradient_accumulation_steps": 1, "max_grad_norm": .5}
    policy = PilotPolicy(editor, diffusion)
    policy.contract["pilot_scope"] = "benign lighting v3 mean exposure/color task; separate from production reward admission"
    trainer = PilotTrainer(policy, protocol["base_learning_rate"], max_grad_norm=.5)
    reward = ProxyRewardCalculator(root)
    write_json(output / "reward_resources.json", reward.receipt, exclusive=True)
    state["precision_receipt"] = {"base": str(editor.dtype), "LoRA": "torch.float32",
                                  "LoRA_linear_compute": "FP32, autocast disabled",
                                  "Adam_moments": "FP32, checked at every update",
                                  "trainable_tensors": len(trainer.named), "BF16_repaired": False,
                                  "initial_GPU_peak_bytes": torch.cuda.max_memory_allocated()}
    conditions = {}
    for records in cohort.values():
        for record in records:
            check_time()
            with Image.open(root / record["image_path"]) as image:
                condition = policy.condition({**record, "image": image.convert("RGB")}, protocol["instruction"])
                condition["content_box"] = content_box(image.convert("RGB"))[0]
            conditions[record["sample_id"]] = condition
            path = output / "visuals/sources" / f"{record['sample_id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            condition["source"].save(path)
    initial_frozen_hash = frozen_digest(policy.pipe)
    initial = output / "checkpoints/shared_initial"
    trainer.save(initial, {"seed": protocol["seed"], "config": {"lr": protocol["base_learning_rate"],
                                                               "weight_sem": 2., "weight_stealth": 1.}})
    observations_log = output / "observations.jsonl"
    update_log = output / "training.jsonl"
    def append(path, value):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    def collect(record, label, seed, config, train=False):
        check_time()
        condition = conditions[record["sample_id"]]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            image, trajectory = policy.sample(condition, seed, label)
        path = output / "visuals" / label / f"{record['sample_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        metric = reward.measure_with_box(condition["source"], image, protocol["target_prompt"], condition["content_box"],
                               config["weight_sem"], config["weight_stealth"])
        observation = {"phase": label, "sample_id": record["sample_id"], "image_id": record["image_id"],
                       "source_SHA256": record["image_sha256"], "image_path": str(path.relative_to(output)),
                       "source_visual_path": f"visuals/sources/{record['sample_id']}.png",
                       "image_SHA256": sha256(path), "seed": seed, "config": dict(config), **metric}
        append(observations_log, observation)
        return (trajectory, condition, metric["training_reward"]), observation
    common_config = {"lr": protocol["base_learning_rate"], "weight_sem": 2., "weight_stealth": 1.}
    def evaluate(split, label):
        rows = []
        offset = 200000 if split == "validation" else 300000
        for index, record in enumerate(cohort[split]):
            _, observation = collect(record, label, protocol["seed"] * 1000000 + offset + index, common_config)
            rows.append(observation)
        return rows

    try:
        state["status"] = "NUMERICAL_GATE_RUNNING"
        persist()
        numerical_dir = output / "numerical_witness"
        numerical_dir.mkdir()
        def numeric_step(index, label):
            episode, _ = collect(cohort["train"][index - 1], label,
                                 protocol["seed"] * 1000000 + 900000 + index, common_config, train=True)
            receipt = trainer.update([episode], keep_gradients=True)
            append(update_log, {"scope": "numerical_witness_not_arm_training", "label": label, **receipt})
            witness = trainer.witness()
            torch.save(witness, numerical_dir / f"{label}.pt")
            return witness
        a1 = numeric_step(1, "ordinary_A1")
        after_one = numerical_dir / "after_one"
        trainer.save(after_one, {"seed": protocol["seed"], "purpose": "completed-next-update resume witness"})
        a2 = numeric_step(2, "ordinary_A2")
        trainer.restore(initial)
        b1 = numeric_step(1, "ordinary_B1")
        b2 = numeric_step(2, "ordinary_B2")
        trainer.restore(after_one)
        resumed = numeric_step(2, "resumed_next_update")
        checks = {"ordinary_first_update": compare_trees(a1, b1),
                  "ordinary_second_update": compare_trees(a2, b2),
                  "reload_completed_next_update": compare_trees(a2, resumed)}
        numerical_passed = all(value["exact"] and value["tolerance"] for value in checks.values())
        state["numerical"] = {"status": "PASS" if numerical_passed else "FAIL", "precision": protocol["precision"],
                              "actual_optimizer_updates": 5, "checks": checks, "BF16_fix_claim": False,
                              "protected_frozen_weights_unchanged": frozen_digest(policy.pipe) == initial_frozen_hash}
        numerical_passed &= state["numerical"]["protected_frozen_weights_unchanged"]
        write_json(numerical_dir / "comparison.json", state["numerical"], exclusive=True)
        del a1, a2, b1, b2, resumed
        trainer.restore(initial)
        torch.cuda.empty_cache()
        persist()
        if not numerical_passed:
            raise RuntimeError("Ordinary repeat/resume numerical gate failed; trained arms NOT TESTED")

        initial_validation = evaluate("validation", "shared_initial_validation")
        initial_mean = mean_metrics(initial_validation)
        state["validation"] = {method: [{"round": 0, "optimizer_step": 0, **initial_mean}]
                               for method in protocol["methods"]}
        state["arms"] = {method: {"status": "READY", "optimizer_steps": 0,
                                  "controller_actions": 0, "controller_updates": 0}
                         for method in protocol["methods"]}
        for method in ["frozen", "fixed"]:
            if method in protocol["methods"]:
                state["arms"][method].update(status="GENERATOR_READY", checkpoint=str(initial.relative_to(output)),
                                             checkpoint_SHA256=sha256(initial / "training_state.pt"))
                state["validation"][method] += [{"round": round_number, "optimizer_step": 0,
                                                 "shared_equivalent_checkpoint": True, **initial_mean}
                                                for round_number in range(1, protocol["rsi_rounds"] + 1)]
        # Identical source order/noise per step across every trained arm; cover all 32 sources each round.
        counts = [protocol["max_inner_steps"] // protocol["rsi_rounds"] +
                  (i < protocol["max_inner_steps"] % protocol["rsi_rounds"])
                  for i in range(protocol["rsi_rounds"])]
        order_rng = random.Random(protocol["seed"])
        schedule = []
        for count in counts:
            order = list(range(len(cohort["train"])))
            order_rng.shuffle(order)
            schedule.append([order[index % len(order)] for index in range(count)])
        write_json(output / "training_schedule.json", {"updates_by_round": counts, "source_indices": schedule}, exclusive=True)
        state["status"] = "TRAINING"
        controller = None
        active_method = None
        for method in ["ddpo_only", "rule_rsi", "rl_rsi"]:
            if method not in protocol["methods"]:
                continue
            active_method = method
            trainer.restore(initial)
            config, previous, before_previous = dict(common_config), dict(initial_mean), None
            controller = LearnedController(protocol["seed"] + 100) if method == "rl_rsi" else None
            state["arms"][method]["status"] = "RUNNING"
            persist()
            for round_index, indices in enumerate(schedule):
                action_record = None
                if controller:
                    feature = controller_state(trainer.step / protocol["max_inner_steps"], previous["common_reward"],
                                               previous["LPIPS"], config["lr"],
                                               1. - trainer.step / protocol["max_inner_steps"])
                    action, config, probabilities = controller.choose(feature, config)
                    action_record = {"round": round_index + 1, "action": action, "config": dict(config),
                                     "probabilities": probabilities, "state": feature.tolist()}
                    state["arms"][method]["controller_actions"] += 1
                    append(output / "controller_actions.jsonl", action_record)
                elif method == "rule_rsi" and before_previous is not None and previous["common_reward"] < before_previous["common_reward"]:
                    config["lr"] = max(config["lr"] * .5, 1e-6)
                    append(output / "rule_actions.jsonl", {"round": round_index + 1, "action": "halve_LR",
                                                            "config": dict(config), "basis": "previous common validation reward declined"})
                trainer.optimizer.param_groups[0]["lr"] = config["lr"]
                for source_index in indices:
                    step = trainer.step + 1
                    episode, observation = collect(cohort["train"][source_index], f"{method}/train/step_{step:03d}",
                                                   protocol["seed"] * 1000000 + 100000 + step, config, train=True)
                    check_time()
                    receipt = trainer.update([episode])
                    state["arms"][method]["optimizer_steps"] = trainer.step
                    append(update_log, {"scope": "arm_training", "arm": method, "round": round_index + 1,
                                        "source_id": observation["sample_id"], "config": dict(config),
                                        "training_reward": observation["training_reward"],
                                        "common_reward": observation["common_reward"], **receipt})
                    if step % 10 == 0:
                        print(json.dumps({"arm": method, "step": step, "round": round_index + 1,
                                          "training_reward": observation["training_reward"],
                                          "elapsed_seconds": time.monotonic() - started}), flush=True)
                        persist()
                validation = evaluate("validation", f"{method}/validation/round_{round_index + 1}")
                current = mean_metrics(validation)
                if controller:
                    learned = controller.learn(current["common_reward"] - previous["common_reward"],
                                               f"observations.jsonl:{method}/validation/round_{round_index + 1}")
                    append(output / "controller_updates.jsonl", learned)
                    state["arms"][method]["controller_updates"] += 1
                    write_json(output / "experience_memory.json", controller.memory)
                state["validation"][method].append({"round": round_index + 1, "optimizer_step": trainer.step,
                                                     "training_config": dict(config), **current})
                checkpoint = output / "checkpoints" / method / f"round_{round_index + 1}"
                trainer.save(checkpoint, {"seed": protocol["seed"], "config": dict(config),
                                           "controller": controller.state_dict() if controller else None,
                                           "next_training_step": trainer.step + 1})
                state["arms"][method].update(checkpoint=str(checkpoint.relative_to(output)),
                                             checkpoint_SHA256=sha256(checkpoint / "training_state.pt"))
                before_previous, previous = previous, current
                persist()
            if trainer.step != protocol["max_inner_steps"]:
                raise AssertionError("Incorrect total training-update count")
            state["arms"][method]["status"] = "GENERATOR_READY"
        active_method = None
        # No checkpoint selection by assessment. Final checkpoints are locked before any test reward query.
        write_json(output / "assessment_checkpoint_lock.json", state["arms"], exclusive=True)
        state["status"] = "ASSESSING_LOCKED_CHECKPOINTS"
        state["assessment"] = {}
        persist()
        for method in protocol["methods"]:
            if method == "fixed":
                state["assessment"][method] = copy.deepcopy(state["assessment"]["frozen"])
                state["arms"][method].update(status="MEASURED_EQUIVALENT_CONTROL", physical_assessment_queries=0)
            else:
                trainer.restore(output / state["arms"][method]["checkpoint"])
                state["assessment"][method] = evaluate("test", f"{method}/assessment")
                state["arms"][method].update(status="MEASURED", physical_assessment_queries=protocol["num_test_images"])
            persist()
        if frozen_digest(policy.pipe) != initial_frozen_hash:
            raise AssertionError("Frozen base weights changed")
        state.update(status="COMPLETED", frozen_weights_unchanged=True,
                     actual_forward_UNet_calls=policy.unet_calls,
                     GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     elapsed_GPU_process_seconds=time.monotonic() - started)
        make_figures(output, state)
        persist()
    except BaseException:
        # Preserve a recoverable actual optimizer/RNG state, but do not auto-resume a failed run.
        trainer.save(output / "checkpoints/interrupted", {"seed": protocol["seed"],
                                                          "scope": "failure evidence; no resume certification"})
        if "controller" in locals() and controller is not None:
            torch.save({"policy": controller.policy.state_dict(), "optimizer": controller.optimizer.state_dict(),
                        "generator_rng": controller.generator.get_state(), "memory": controller.memory,
                        "pending": controller.pending}, output / "interrupted_controller.pt")
        state.update(actual_forward_UNet_calls=policy.unet_calls,
                     GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     elapsed_GPU_process_seconds=time.monotonic() - started)
        raise


def main():
    args = arguments()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("New pilot output must be under this project's outputs/")
    if args.reports_only:
        protocol = json.loads((output / "protocol.json").read_text())
        state = json.loads((output / "run_state.json").read_text())
        generate_reports(output, protocol, state)
        return 0
    if not args.force_fp32_lora or args.max_inner_steps < args.rsi_rounds * args.num_train_images:
        raise ValueError("FP32 LoRA and coverage of every training image each round are required")
    if not 1e-6 <= args.base_learning_rate <= 5e-5 or not 1 <= args.wall_seconds <= 17700:
        raise ValueError("Outside the bounded mini-pilot LR/time protocol")
    if args.methods != list(METHODS):
        raise ValueError("This pilot records all five requested arms in fixed order")
    # Set before importing torch/CUDA, including into imported engine modules.
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        os.environ[key] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output.mkdir(parents=True, exist_ok=False)
    protocol = vars(args).copy()
    protocol.update(created_UTC=datetime.now(timezone.utc).isoformat(), project_root=str(ROOT),
                    instruction=INSTRUCTION, target_prompt=TARGET,
                    scope="BENIGN EDITING; DEVELOPMENT ONLY; NOT MAIN POISONING",
                    common_evaluator="lighting_v3: 2*(delta_CLIP + mean_exposure_progress) - color_cost - LPIPS",
                    reward_version="lighting_v3_mean_progress_20261104",
                    guidance_scale=5., image_guidance_scale=1.5, eta=1., gradient_clip=.5,
                    training_weights={"semantic": 2., "perceptual": 1.},
                    max_inner_steps_meaning="TOTAL optimizer updates per trained arm, split over RSI rounds",
                    frozen_fixed_equivalence="same prompt and checkpoint; explicitly approved by user",
                    victim_finetuning="NOT TESTED", ASR="NOT TESTED", Delta_Clean="NOT TESTED",
                    allocation_cap_GPU_seconds=18000, automatic_retries=0,
                    old_budget_or_controller_reopened=False)
    source_paths = ["train_lighting_v3.py", "train_lighting_v2.py", "train_and_evaluate_pilot.py", "run_mini_pilot.sh", "scripts/audit_padding_reward_20261104.py"] + [str(path.relative_to(ROOT)).replace("\\", "/")
                     for path in sorted((ROOT / "src/mini_pilot").glob("*.py"))] + PROTECTED
    protocol["source_SHA256"] = {name: sha256(ROOT / name) for name in source_paths}
    for name in source_paths:
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    write_json(output / "protocol.json", protocol, exclusive=True)
    state = {"status": "PREFLIGHT", "arms": {}, "assessment": {}}
    def persist():
        write_json(output / "run_state.json", state)
        generate_reports(output, protocol, state)
    persist()
    try:
        sizes = {"train": args.num_train_images, "validation": args.num_val_images, "test": args.num_test_images}
        cohort, audit, provenance = prepare_cohort(ROOT, ROOT / args.dataset_path, sizes, args.seed)
        state.update(data_audit=audit, dataset_provenance=provenance)
        write_json(output / "cohort.json", cohort, exclusive=True)
        state["environment"] = environment(ROOT)
        write_json(output / "environment.json", state["environment"], exclusive=True)
        print("ENVIRONMENT_AND_DATA " + json.dumps({"environment": state["environment"], "data": audit}), flush=True)
        persist()
        if args.preflight_only:
            state["status"] = "PREFLIGHT_COMPLETE_NO_GPU_EXPERIMENT"
        else:
            execute(ROOT, output, protocol, cohort, state, persist)
    except BaseException as exc:
        state.update(status="FAILED", failure=f"{type(exc).__name__}: {exc}")
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
    finally:
        state["source_protection"] = {name: sha256(ROOT / name) == protocol["source_SHA256"][name] for name in PROTECTED}
        state["finished_UTC"] = datetime.now(timezone.utc).isoformat()
        state["disk_free_bytes_after"] = shutil.disk_usage(ROOT).free
        persist()
        files = [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_inventory.json"]
        write_json(output / "artifact_inventory.json", {"files": [{"path": str(path.relative_to(output)),
                                                                  "bytes": path.stat().st_size, "SHA256": sha256(path)}
                                                                 for path in files],
                                                       "retained_bytes": sum(path.stat().st_size for path in files)}, exclusive=True)
    print(json.dumps({"status": state["status"], "output": str(output)}), flush=True)
    return 0 if state["status"] in {"COMPLETED", "PREFLIGHT_COMPLETE_NO_GPU_EXPERIMENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

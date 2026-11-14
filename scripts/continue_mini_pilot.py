"""One explicitly authorized time-cap continuation; never retries a numerical failure."""
import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONDONTWRITEBYTECODE"]:
    os.environ[key] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.mini_pilot.data import sha256, write_json, validate_cohort
from src.mini_pilot.reporting import generate_reports, mean_metrics, make_figures


def restore_controller(controller, saved):
    controller.policy.load_state_dict(saved["policy"])
    controller.optimizer.load_state_dict(saved["optimizer"])
    controller.generator.set_state(saved["generator_rng"])
    controller.memory, controller.pending = copy.deepcopy(saved["memory"]), copy.deepcopy(saved["pending"])
    controller.baseline = 0.
    for memory in controller.memory:
        controller.baseline = .9 * controller.baseline + .1 * memory["validation_gain_common_reward"]


def next_source_index(schedule, completed_rounds, step):
    cursor = step - sum(len(indices) for indices in schedule[:completed_rounds])
    if cursor < 0:
        raise ValueError("Checkpoint precedes its completed validation rounds")
    for indices in schedule[completed_rounds:]:
        if cursor < len(indices):
            return indices[cursor]
        cursor -= len(indices)
    if cursor != 0:
        raise ValueError("Checkpoint exceeds the preregistered training schedule")
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Continuation must use this project's original output")
    protocol = json.loads((output / "protocol.json").read_text())
    state = json.loads((output / "run_state.json").read_text())
    if state.get("status") != "FAILED" or not state.get("failure", "").startswith("TimeoutError:"):
        raise RuntimeError("Only the cooperative time-cap failure is eligible for authorized continuation")
    if state.get("numerical", {}).get("status") != "PASS":
        raise RuntimeError("An unpassed numerical gate is never bypassed")
    authorization = json.loads((output / "continuation_submission/authorization.json").read_text())
    if authorization["cumulative_cap_GPU_seconds"] != 5400 or authorization["max_continuations"] != 1:
        raise ValueError("Missing bounded 90-minute user authorization")
    continuation = output / "continuation"
    continuation.mkdir(exist_ok=False)
    write_json(continuation / "original_failed_state.json", state, exclusive=True)
    (continuation / "original_artifact_inventory.json").write_bytes((output / "artifact_inventory.json").read_bytes())
    (continuation / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    for name, expected in protocol["source_SHA256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Original scientific source changed before continuation: {name}")
    import numpy as np
    import torch
    from PIL import Image
    from src.editing.pretrained_editor import PretrainedEditor
    from src.models.pretrained_editor_loader import pretrained_config
    from src.mini_pilot.engine import PilotPolicy, PilotTrainer, compare_trees, frozen_digest
    from src.mini_pilot.reward import ProxyRewardCalculator
    from src.mini_pilot.controller import LearnedController, controller_state
    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A separately bounded one-GPU allocation is required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    started = time.monotonic()
    def check_time():
        if time.monotonic() - started > 1700:
            raise TimeoutError("The sole continuation's cooperative 1700-second cap was reached")
    def persist():
        write_json(output / "run_state.json", state)
        generate_reports(output, protocol, state)
    def append(path, value):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    cohort = json.loads((output / "cohort.json").read_text())
    validate_cohort(ROOT, cohort, {"train": 32, "validation": 16, "test": 16})
    editor = PretrainedEditor(pretrained_config(ROOT, precision=protocol["precision"], seed=protocol["seed"],
                                               num_inference_steps=protocol["num_inference_steps"]), device="cuda")
    original_checkpoint = torch.load(output / "checkpoints/shared_initial/training_state.pt", map_location="cpu", weights_only=False)
    policy = PilotPolicy(editor, original_checkpoint["contract"]["config"])
    policy.contract["pilot_scope"] = original_checkpoint["contract"]["pilot_scope"]
    trainer = PilotTrainer(policy, protocol["base_learning_rate"])
    reward = ProxyRewardCalculator(ROOT)
    original_resources = json.loads((output / "reward_resources.json").read_text())
    for key in ["CLIP_revision", "LPIPS_calibration_SHA256", "AlexNet_SHA256"]:
        if reward.receipt[key] != original_resources[key]:
            raise RuntimeError(f"Frozen reward resource changed before continuation: {key}")
    conditions = {}
    for records in cohort.values():
        for record in records:
            check_time()
            with Image.open(ROOT / record["image_path"]) as image:
                conditions[record["sample_id"]] = policy.condition({**record, "image": image.convert("RGB")}, protocol["instruction"])
    base_hash = frozen_digest(policy.pipe)
    common = {"lr": protocol["base_learning_rate"], "weight_sem": 2., "weight_stealth": 1.}
    def collect(record, label, seed, config):
        check_time()
        condition = conditions[record["sample_id"]]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            image, trajectory = policy.sample(condition, seed, label)
        path = output / "visuals/continuation" / label / f"{record['sample_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Never overwrite a previous continuation image")
        image.save(path)
        metric = reward.measure(condition["source"], image, protocol["target_prompt"], config["weight_sem"], config["weight_stealth"])
        observation = {"phase": "continuation/" + label, "sample_id": record["sample_id"], "image_id": record["image_id"],
                       "source_SHA256": record["image_sha256"], "image_path": str(path.relative_to(output)),
                       "source_visual_path": f"visuals/sources/{record['sample_id']}.png", "image_SHA256": sha256(path),
                       "seed": seed, "config": dict(config), **metric}
        append(output / "observations.jsonl", observation)
        return (trajectory, condition, metric["training_reward"]), observation
    def evaluate(split, label):
        offset = 200000 if split == "validation" else 300000
        return [collect(record, label, protocol["seed"] * 1000000 + offset + index, common)[1]
                for index, record in enumerate(cohort[split])]
    schedule = json.loads((output / "training_schedule.json").read_text())["source_indices"]
    updates = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    training = [row for row in updates if row.get("scope") == "arm_training"]
    active = next((method for method in ["ddpo_only", "rule_rsi", "rl_rsi"]
                   if state["arms"][method]["status"] == "RUNNING"), None)
    state["continuation"] = {"job_id": os.getenv("SLURM_JOB_ID"), "cumulative_cap_GPU_seconds": 5400,
                             "Slurm_cap_seconds": 1800, "initial_active_arm": active,
                             "hostname": os.uname().nodename, "GPU": torch.cuda.get_device_name(),
                             "worker_SHA256": sha256(__file__), "reason": "explicitly user-authorized remaining steps after time cap"}
    state["status"] = "CONTINUING_AUTHORIZED_REMAINING_STEPS"
    persist()
    controller = None
    try:
        for method in ["ddpo_only", "rule_rsi", "rl_rsi"]:
            if state["arms"][method]["status"] in {"GENERATOR_READY", "MEASURED", "MEASURED_EQUIVALENT_CONTROL"}:
                continue
            partial = method == active
            controller = LearnedController(protocol["seed"] + 100) if method == "rl_rsi" else None
            checkpoint = output / ("checkpoints/interrupted" if partial else "checkpoints/shared_initial")
            trainer.restore(checkpoint)
            method_updates = [row for row in training if row["arm"] == method]
            if len(method_updates) != trainer.step:
                raise ValueError("Partial optimizer step count disagrees with committed training log")
            config = dict(method_updates[-1]["config"]) if method_updates else dict(common)
            if partial:
                config["lr"] = trainer.optimizer.param_groups[0]["lr"]
            if controller and partial:
                saved = torch.load(output / "interrupted_controller.pt", map_location="cpu", weights_only=False)
                restore_controller(controller, saved)
                actions = [json.loads(line) for line in (output / "controller_actions.jsonl").read_text().splitlines()]
                if controller.pending is not None:
                    config = dict(actions[-1]["config"])
            points = state["validation"][method]
            completed_rounds = len(points) - 1
            previous = {key: points[-1][key] for key in ["common_reward", "CLIP_cosine", "LPIPS", "SSIM"]}
            before_previous = points[-2] if len(points) > 1 else None
            initial_step = trainer.step
            if partial and trainer.step < protocol["max_inner_steps"]:
                # Two throwaway real updates from the same partial disk state, reset afterward.
                next_index = next_source_index(schedule, completed_rounds, trainer.step)
                if next_index is not None:
                    source = cohort["train"][next_index]
                    witnesses = []
                    for attempt in [1, 2]:
                        trainer.restore(checkpoint)
                        episode, _ = collect(source, f"resume_witness_{attempt}",
                                             protocol["seed"] * 1000000 + 100000 + trainer.step + 1, config)
                        receipt = trainer.update([episode], keep_gradients=True)
                        append(output / "training.jsonl", {"scope": "continuation_resume_witness_not_arm_training", **receipt})
                        witness = trainer.witness()
                        torch.save(witness, continuation / f"resume_witness_{attempt}.pt")
                        witnesses.append(witness)
                    comparison = compare_trees(*witnesses)
                    write_json(continuation / "resume_comparison.json", comparison, exclusive=True)
                    if not comparison["exact"] or not comparison["tolerance"]:
                        raise RuntimeError("Partial-checkpoint next-update repeat failed; no continuation training")
                    trainer.restore(checkpoint)
                    state["continuation"]["partial_next_update_repeat"] = comparison
            for round_index in range(completed_rounds, len(schedule)):
                round_start = sum(len(indices) for indices in schedule[:round_index])
                offset = trainer.step - round_start
                if offset < 0 or offset > len(schedule[round_index]):
                    raise ValueError("Training schedule/round checkpoint disagreement")
                existing_pending = partial and round_index == completed_rounds
                if controller and controller.pending is None:
                    feature = controller_state(trainer.step / protocol["max_inner_steps"], previous["common_reward"],
                                               previous["LPIPS"], config["lr"], 1. - trainer.step / protocol["max_inner_steps"])
                    action, config, probabilities = controller.choose(feature, config)
                    append(output / "controller_actions.jsonl", {"round": round_index + 1, "action": action,
                                                                 "config": dict(config), "probabilities": probabilities,
                                                                 "state": feature.tolist(), "continuation": True})
                    state["arms"][method]["controller_actions"] += 1
                elif not controller and not existing_pending and method == "rule_rsi" and before_previous is not None and previous["common_reward"] < before_previous["common_reward"]:
                    config["lr"] = max(config["lr"] * .5, 1e-6)
                    append(output / "rule_actions.jsonl", {"round": round_index + 1, "action": "halve_LR",
                                                            "config": dict(config), "continuation": True,
                                                            "basis": "previous common validation reward declined"})
                trainer.optimizer.param_groups[0]["lr"] = config["lr"]
                for source_index in schedule[round_index][offset:]:
                    step = trainer.step + 1
                    episode, observation = collect(cohort["train"][source_index], f"{method}/train/step_{step:03d}",
                                                   protocol["seed"] * 1000000 + 100000 + step, config)
                    check_time()
                    receipt = trainer.update([episode])
                    append(output / "training.jsonl", {"scope": "arm_training", "arm": method, "round": round_index + 1,
                                                       "source_id": observation["sample_id"], "config": dict(config),
                                                       "training_reward": observation["training_reward"], "common_reward": observation["common_reward"],
                                                       "continuation": True, **receipt})
                    state["arms"][method]["optimizer_steps"] = trainer.step
                    if step % 10 == 0:
                        print(json.dumps({"arm": method, "step": step, "continuation": True}), flush=True)
                        persist()
                current = mean_metrics(evaluate("validation", f"{method}/validation/round_{round_index + 1}"))
                if controller:
                    receipt = controller.learn(current["common_reward"] - previous["common_reward"],
                                               f"observations.jsonl:continuation/{method}/validation/round_{round_index + 1}")
                    append(output / "controller_updates.jsonl", receipt)
                    state["arms"][method]["controller_updates"] += 1
                    write_json(output / "experience_memory.json", controller.memory)
                state["validation"][method].append({"round": round_index + 1, "optimizer_step": trainer.step,
                                                     "training_config": dict(config), "continuation": True, **current})
                final = output / "checkpoints" / method / f"round_{round_index + 1}"
                trainer.save(final, {"seed": protocol["seed"], "config": dict(config),
                                     "controller": controller.state_dict() if controller else None, "next_training_step": trainer.step + 1})
                state["arms"][method].update(checkpoint=str(final.relative_to(output)), checkpoint_SHA256=sha256(final / "training_state.pt"))
                before_previous, previous, partial = previous, current, False
                persist()
            state["arms"][method].update(status="GENERATOR_READY", optimizer_steps=trainer.step)
            state["continuation"].setdefault("remaining_arm_updates", {})[method] = trainer.step - initial_step
        lock_path = output / "assessment_checkpoint_lock.json"
        if lock_path.exists():
            locked = json.loads(lock_path.read_text())
            if any(locked[method]["checkpoint_SHA256"] != state["arms"][method]["checkpoint_SHA256"] for method in protocol["methods"]):
                raise RuntimeError("Assessment-locked checkpoint changed")
        else:
            write_json(lock_path, state["arms"], exclusive=True)
        state["status"] = "ASSESSING_LOCKED_CHECKPOINTS"
        state.setdefault("assessment", {})
        persist()
        for method in protocol["methods"]:
            if state["assessment"].get(method):
                continue
            if method == "fixed":
                state["assessment"][method] = copy.deepcopy(state["assessment"]["frozen"])
                state["arms"][method].update(status="MEASURED_EQUIVALENT_CONTROL", physical_assessment_queries=0)
            else:
                trainer.restore(output / state["arms"][method]["checkpoint"])
                state["assessment"][method] = evaluate("test", f"{method}/assessment")
                state["arms"][method].update(status="MEASURED", physical_assessment_queries=16)
            persist()
        if frozen_digest(policy.pipe) != base_hash:
            raise RuntimeError("Frozen base weights changed during continuation")
        state["continuation"].update(full_UNet_calls=policy.unet_calls, elapsed_GPU_process_seconds=time.monotonic() - started)
        state.update(status="COMPLETED", frozen_weights_unchanged=True,
                     failure_history=[state.pop("failure")], continued_after_time_cap=True)
        make_figures(output, state)
        persist()
        return 0
    except BaseException as exc:
        trainer.save(continuation / "interrupted", {"seed": protocol["seed"], "scope": "final continuation failure evidence"})
        (continuation / "failure.txt").write_text(traceback.format_exc())
        state.update(status="FAILED", failure=f"{type(exc).__name__}: {exc}")
        persist()
        return 1
    finally:
        files = [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_inventory.json"]
        write_json(output / "artifact_inventory.json", {"files": [{"path": str(path.relative_to(output)),
                                                                  "bytes": path.stat().st_size, "SHA256": sha256(path)}
                                                                 for path in files],
                                                       "retained_bytes": sum(path.stat().st_size for path in files),
                                                       "refreshed_after_continuation": True})


if __name__ == "__main__":
    raise SystemExit(main())

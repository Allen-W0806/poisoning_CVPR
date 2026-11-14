import copy
import gc
import json
import random
import time
import numpy as np
import torch
from PIL import Image
import _bootstrap
from _phase2 import phase2_run, mark_latest
from _runs import model_options
from src.data.dataset import VQADataset, sha256
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.models.qwen_loader import QwenVQA
from src.training.generator_adapter import frozen_hashes
from src.evaluation.semantic_validation import review_pair, semantic_decision
from src.evaluation.answer_consistency import answer_consistency
from src.evaluation.image_quality import image_quality
from src.editing.semantic_validator import SemanticResult
from src.rl.reward import validation_reward
from src.rl.cached_editor_trainer import CachedEditorController
from src.rsi.controller import RSIController
from src.utils.logging import write_json

INSTRUCTIONS = [
 "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.",
 "Slightly soften the lighting. Keep all objects, their colors, counts and positions unchanged.",
 "Make a subtle contrast adjustment. Keep all objects, their colors, counts and positions unchanged."]


def build_feedback(root):
    records = {}
    for split in ["train", "validation"]:
        data = VQADataset(root / f"data/processed/vqav2_seed42_v1/{split}.json", root, split)
        indices = list(range(len(data)))
        random.Random(42).shuffle(indices)
        records[split] = [data[i] for i in indices[:10]]
    with phase2_run("real_editor_feedback", {"images_per_split": 10, "actions": INSTRUCTIONS,
                    "steps": 10, "feedback_semantics": "conservative automatic proxies; not human verified"},
                    ("train", "validation"), editor=True, qwen=True) as run:
        editor = load_pretrained_editor(root)
        frozen_before = frozen_hashes(editor.pipe)
        generated = []
        for split, rows in records.items():
            for i, r in enumerate(rows):
                source_path = run.path / f"images/{split}_{i}_source.png"
                editor.prepare_image(r["image"]).save(source_path)
                for action, instruction in enumerate(INSTRUCTIONS):
                    result = editor.edit(r["image"], instruction, r["image_id"], seed=42 + i,
                                output_path=run.path / f"images/{split}_{i}_{action}.png", steps=10)
                    generated.append({"split": split, "index": i, "action": action, "source_path": str(source_path),
                        "sample_id": r["sample_id"], "question": r["question"], "answers": r["answers"], **result.metadata})
            run.event("generated_split", {"split": split, "count": len(rows) * 3})
        assert frozen_before == frozen_hashes(editor.pipe)
        del editor
        gc.collect()
        torch.cuda.empty_cache()
        qwen = QwenVQA(**model_options("cuda"))
        original_answers = {}
        feedback = []
        for item in generated:
            key = (item["split"], item["index"])
            r = records[item["split"]][item["index"]]
            source = Image.open(item["source_path"]).convert("RGB")
            edited = Image.open(item["output_path"]).convert("RGB")
            if key not in original_answers:
                original_answers[key] = qwen.answer([source], [r["question"]])[0]
            answer = qwen.answer([edited], [r["question"]])[0]
            consistency = answer_consistency(original_answers[key], answer, r)
            judgment = review_pair(qwen, source, edited, r)
            semantic = semantic_decision(source, edited, r, consistency, judgment)
            quality = image_quality(source, edited)
            reward = validation_reward(SemanticResult(semantic["verified"], semantic["score"], semantic["verifier"], semantic["reason"]),
                consistency["score"] if semantic.get("provisionally_valid") else 0., quality["distortion"], 10 / 30)
            feedback.append({**item, "semantic": semantic, "answer_consistency": consistency,
                             "structured_judgment": judgment, "quality": quality, "reward": reward})
            run.event("feedback", {"sample_id": r["sample_id"], "split": item["split"], "action": item["action"], "reward": reward})
        write_json(run.path / "feedback.json", feedback)
        run.metrics({"records": len(feedback), "qwen_calls": qwen.calls, "generator_frozen": True,
                     "human_verified_edits": 0, "training_split_records": 30, "validation_split_records": 30,
                     "generation_seconds": sum(x["generation_seconds"] for x in generated),
                     "instruction_conditioned_edits": True})
        run.finish(summary="Shared fixed real-editor feedback table, with conservative unverified-semantics penalties. No generator optimization.")
        mark_latest("feedback", run)
        return records, feedback, run


if __name__ == "__main__":
    root = _bootstrap.ROOT
    records, feedback, feedback_run = build_feedback(root)
    rewards = {split: [[next(x["reward"]["reward"] for x in feedback if x["split"] == split and x["index"] == i and x["action"] == a)
                        for a in range(3)] for i in range(10)] for split in ["train", "validation"]}
    with phase2_run("rl_rsi_validation", {"feedback_run": feedback_run.id, "batch_size": 10,
                    "updates_per_arm_total": 6, "validation_calls_per_arm": 2,
                    "rsi_schedule": "3-step baseline + 3-step candidate from common initial state",
                    "fixed_rl_schedule": "6 steps, validation after steps 3 and 6",
                    "compute_scope": "equal gradient/data/validation lookup counts; actual runtimes and shared precompute separately reported",
                    "selection_split": "validation", "generator_updated": False}, ("train", "validation"), editor=True, qwen=True) as run:
        fixed_start = time.perf_counter()
        fixed = CachedEditorController(records["train"], records["validation"], rewards["train"], rewards["validation"], 1e-5)
        fixed_history, fixed_validation = [], []
        for i in range(6):
            result = fixed.update()
            fixed_history.append(result)
            run.event("fixed_rl_update", result)
            if i in [2, 5]:
                fixed_validation.append(fixed.validate())
        torch.cuda.synchronize()
        fixed_seconds = time.perf_counter() - fixed_start
        torch.save({"policy": fixed.policy.state_dict(), "optimizer": fixed.optimizer.state_dict()}, run.path / "checkpoints/fixed_rl.pt")
        rsi_start = time.perf_counter()
        base = CachedEditorController(records["train"], records["validation"], rewards["train"], rewards["validation"], 1e-5)
        base_history = [base.update() for _ in range(3)]
        base_score = base.validate()
        base_seconds = time.perf_counter() - rsi_start
        val_hash = sha256(root / "data/processed/vqav2_seed42_v1/validation.json")
        controller = RSIController(run.path / "rsi", {"batch_size": 10, "learning_rate": 1e-5}, base_score,
            val_hash, .01, policy_version="controller_lr1e5", initial_cost_gpu_hours=base_seconds / 3600)
        observed_variance = float(np.mean([x["reward_variance"] for x in base_history]))
        proposed_lr = 5e-5 if observed_variance > 0 and max(x["gradient_norm"] for x in base_history) < 1 else 1e-5
        candidate_start = time.perf_counter()
        candidate = CachedEditorController(records["train"], records["validation"], rewards["train"], rewards["validation"], proposed_lr)
        candidate_history = [candidate.update() for _ in range(3)]
        candidate_score = candidate.validate()
        torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - candidate_start
        revision = controller.revise({"learning_rate": proposed_lr}, candidate_score, (time.perf_counter() - rsi_start - base_seconds) / 3600, val_hash,
            policy_version=f"controller_lr{proposed_lr}", proposal_reason="Use 5e-5 only if real sampled reward variance is positive and every gradient norm <1; otherwise retain 1e-5; accept only strict held-out validation gain",
            evidence={"feedback_run": feedback_run.id, "base_validation": base_score, "observed_reward_variance": observed_variance, "base_update_metrics": base_history},
            rollback={"previous_policy": "controller_lr1e5", "checkpoint": "checkpoints/base_controller.pt"})
        selected = candidate if revision["accepted"] else base
        for name, obj in [("base", base), ("candidate", candidate), ("selected", selected)]:
            torch.save({"policy": obj.policy.state_dict(), "optimizer": obj.optimizer.state_dict()}, run.path / f"checkpoints/{name}_controller.pt")
        if not revision["accepted"]:
            assert all(torch.equal(selected.policy.state_dict()[k], base.policy.state_dict()[k]) for k in base.policy.state_dict())
        rng = np.random.default_rng(42)
        random_actions = rng.integers(0, 3, size=10)
        metrics = {"frozen_editor_action0_validation_reward": float(np.mean(np.asarray(rewards["validation"])[:, 0])),
            "random_policy_validation_reward": float(np.mean(np.asarray(rewards["validation"])[np.arange(10), random_actions])),
            "fixed_rl_validation": fixed_validation, "rsi_base_validation": base_score,
            "rsi_candidate_validation": candidate_score,
            "rsi_selected_validation": candidate_score if revision["accepted"] else base_score,
            "revision": revision, "fixed_rl_updates": fixed_history, "rsi_base_updates": base_history,
            "rsi_candidate_updates": candidate_history, "fixed_rl_seconds": fixed_seconds,
            "rsi_seconds": time.perf_counter() - rsi_start,
            "shared_precomputation_run": feedback_run.id, "generator_rl_status": "BLOCKED; only separate controller trained",
            "exposure_per_arm": 60, "validation_lookups_per_arm": 2,
            "claim": "Preliminary fixed cached real-editor proxy experiment; no RSI superiority or semantic-guarantee claim"}
        run.metrics(metrics)
        run.finish(summary="Real-observation controller RL and RSI selection/rollback under equal nominal update/exposure/validation budgets. Actual time differs and is reported.")
        mark_latest("controller", run)
        print(run.path)

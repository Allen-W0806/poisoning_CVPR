import argparse
import copy
import json
import _bootstrap
from _runs import new_run
import numpy as np
import torch
import yaml
from src.data.prepare_vqa import prepare_fixture
from src.data.dataset import VQADataset, sha256
from src.models.editor_loader import load_editor
from src.rl.policy import EditPolicy
from src.rl.trainer import PolicyTrainer, DEFAULTS
from src.rl.reward import RewardWeights
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.seed import seed_everything, resolve_device
from src.evaluation.cost_tracker import CostTracker
from src.utils.scratch import scratch_directory


def make_trainer(device, config, weights=None):
    seed_everything(config["seed"])
    return PolicyTrainer(EditPolicy(), load_editor(), device, config, weights=weights)


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def run_smoke(device="auto", config=None, reward_config=None):
    device = resolve_device(device)
    config = {**DEFAULTS, **(config or {})}
    if config["total_updates"] != 3:
        raise ValueError("This smoke executes exactly three updates; total_updates must be 3")
    weights = RewardWeights(**(reward_config or {}))
    path = prepare_fixture(_bootstrap.ROOT) / "train.json"
    data = VQADataset(path, _bootstrap.ROOT, "train")
    batch = [data[i % len(data)] for i in range(config["batch_size"])]
    batches = [batch] * config["gradient_accumulation_steps"]
    contract = {"reward_weights": weights.__dict__, "editor": "pillow_brightness_v1",
                "device_type": device.type,
                "actions": [1.0, .97, 1.03], "evaluator": "synthetic_count_oracle_v1",
                "dataset_sha256": sha256(path), "sample_ids": [r["sample_id"] for r in batch],
                "batch_schedule": "same_microbatch_repeated_each_update"}
    with new_run("rl_smoke", {**config, "device": str(device), "feedback": "synthetic_count_oracle",
                            "reward_weights": weights.__dict__, "effective_smoke_checkpoint_steps": [2, 3],
                            "effective_smoke_log_interval": 1}, path) as run:
        cost = CostTracker(device)
        trainer = make_trainer(device, config, weights)
        frozen_before = {n: p.detach().clone() for n, p in trainer.policy.named_parameters() if not p.requires_grad}
        trainable_before = {n: p.detach().clone() for n, p in trainer.policy.named_parameters() if p.requires_grad}
        history = []
        for i in range(3):
            result = trainer.update(batches)
            history.append(result)
            run.event("update", result)
            if i == 1:
                save_checkpoint(run.path / "checkpoints/step_0002.pt", trainer.policy, trainer.optimizer,
                                trainer.scheduler, trainer.step, trainer.config, contract)
        original_state = copy.deepcopy(trainer.policy.state_dict())
        optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
        scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())
        assert all(torch.equal(p, frozen_before[n]) for n, p in trainer.policy.named_parameters() if not p.requires_grad)
        delta = max(float((p - trainable_before[n]).abs().max().detach()) for n, p in trainer.policy.named_parameters() if p.requires_grad)
        assert delta > 0
        save_checkpoint(run.path / "checkpoints/step_0003.pt", trainer.policy, trainer.optimizer,
                        trainer.scheduler, trainer.step, trainer.config, contract)
        replay = make_trainer(device, config, weights)
        replay_history = [replay.update(batches) for _ in range(3)]
        assert state_equal(original_state, replay.policy.state_dict()), "Seeded replay parameters diverged"
        assert history == replay_history, "Seeded replay metrics diverged"
        resumed = make_trainer(device, config, weights)
        payload = load_checkpoint(run.path / "checkpoints/step_0002.pt", resumed.policy, resumed.optimizer,
                                  resumed.scheduler, config, contract)
        try:
            load_checkpoint(run.path / "checkpoints/step_0002.pt", resumed.policy, resumed.optimizer,
                            resumed.scheduler, {**config, "learning_rate": 99}, contract)
        except ValueError:
            pass
        else:
            raise AssertionError("Resume accepted mismatched configuration")
        resumed.step = payload["step"]
        resumed_result = resumed.update(batches)
        assert history[2] == resumed_result, "Resumed next-step metrics diverged"
        assert state_equal(original_state, resumed.policy.state_dict()), "Resumed parameters diverged"
        assert state_equal(optimizer_state, resumed.optimizer.state_dict()), "Resumed optimizer diverged"
        assert state_equal(scheduler_state, resumed.scheduler.state_dict()), "Resumed scheduler diverged"
        total_samples = 7 * config["batch_size"] * config["gradient_accumulation_steps"]
        cost.evaluation_calls = total_samples
        result = {"updates": history, "trainable_max_change": delta, "frozen_unchanged": True,
                  "reproducible_exact_same_device": True, "checkpoint_next_step_equal": True,
                  "optimizer_resume_equal": True, "scheduler_resume_equal": True,
                  "policy_forward_calls": 7 * config["gradient_accumulation_steps"],
                  "vlm_calls": 0, "cost_reward_units": "edit operations / 10", **cost.snapshot(total_samples)}
        run.metrics(result)
        run.finish(summary="Three optimizer updates plus seeded replay and exact next-step checkpoint continuation; synthetic feedback only.")
        print(json.dumps({"run": str(run.path), "max_parameter_change": delta, "device": str(device)}))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default="configs/rl.yaml")
    args = parser.parse_args()
    loaded = yaml.safe_load((_bootstrap.ROOT / args.config).read_text(encoding="utf-8"))
    run_smoke(args.device, loaded.get("rl"), loaded.get("reward"))

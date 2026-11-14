"""Ordinary on-policy DDPO-SF updates, isolated from production training callers."""
import copy
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from src.rl.diffusion_policy import DiffusionPolicy, cfg_inputs, guided_prediction
from src.rl.diffusion_trajectory import assert_fresh_batch, transition
from src.training.generator_adapter import adapter_state, load_adapter_state


class PilotPolicy(DiffusionPolicy):
    def predict(self, latent, timestep, condition):
        return super().predict(latent, timestep, condition)


def frozen_digest(pipe):
    digest = hashlib.sha256()
    for prefix, model in [("unet", pipe.unet), ("vae", pipe.vae), ("text", pipe.text_encoder)]:
        for name, value in model.named_parameters():
            if not value.requires_grad:
                digest.update((prefix + "." + name).encode())
                digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def compare_trees(first, second):
    """Keep exact and tolerance outcomes separate; never skip failing tensors."""
    result = {"exact": True, "tolerance": True, "tensor_count": 0,
              "different_elements": 0, "max_absolute_error": 0., "structure_errors": []}
    def walk(a, b, key):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result["tensor_count"] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                result["structure_errors"].append(key)
                result["exact"] = result["tolerance"] = False
                return
            different = int((a != b).sum())
            result["different_elements"] += different
            result["exact"] &= different == 0
            result["tolerance"] &= bool(torch.allclose(a, b, rtol=1e-6, atol=1e-9))
            result["max_absolute_error"] = max(result["max_absolute_error"],
                                                float((a.double() - b.double()).abs().max()) if a.numel() else 0.)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            result["exact"] &= equal
            result["tolerance"] &= equal
            if not equal:
                result["structure_errors"].append(key)
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for name in a:
                walk(a[name], b[name], f"{key}/{name}")
        elif isinstance(a, (tuple, list)) and type(a) is type(b) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{key}/{index}")
        elif type(a) is not type(b) or a != b:
            result["exact"] = result["tolerance"] = False
            result["structure_errors"].append(key)
    walk(first, second, "root")
    return result


class PilotTrainer:
    def __init__(self, policy, lr, max_grad_norm=1.):
        self.policy = policy
        if not 0 < max_grad_norm <= 10:
            raise ValueError("A positive bounded gradient clipping norm is required")
        self.max_grad_norm = float(max_grad_norm)
        self.named = {name: value for name, value in policy.pipe.unet.named_parameters() if value.requires_grad}
        if not self.named or any(".lora_" not in name or value.dtype != torch.float32
                                 for name, value in self.named.items()):
            raise ValueError("Optimizer must exclusively own FP32 UNet LoRA parameters")
        self.params = list(self.named.values())
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.trainer_contract = {"schema": "benign_pilot_checkpoint_v2", "optimizer": "AdamW",
                                 "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
                                 "max_grad_norm": self.max_grad_norm}
        if {id(p) for group in self.optimizer.param_groups for p in group["params"]} != {id(p) for p in self.params}:
            raise ValueError("Optimizer ownership mismatch")
        self.step, self.consumed = 0, set()
        self.last_raw, self.last_postclip = {}, {}

    def update(self, episodes, keep_gradients=False):
        if not episodes:
            raise ValueError("Empty on-policy batch")
        assert_fresh_batch([trajectory for trajectory, _, _ in episodes], self.policy.version,
                           self.policy.contract, self.consumed)
        started, replay, losses = time.perf_counter(), [], []
        before = adapter_state(self.policy.pipe.unet)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for trajectory, condition, reward in episodes:
                if not np.isfinite(reward):
                    raise FloatingPointError("Nonfinite terminal reward")
                if (trajectory["sample_id"], trajectory["image_id"], trajectory["instruction"]) != (
                        condition["sample_id"], condition["image_id"], condition["instruction"]):
                    raise ValueError("Trajectory conditioning mismatch")
                if [item["t"] for item in trajectory["steps"]] != self.policy.pipe.scheduler.timesteps.tolist():
                    raise ValueError("Incomplete diffusion trajectory")
                for item in trajectory["steps"]:
                    state = item["state"].to(self.policy.editor.device)
                    action = item["action"].to(self.policy.editor.device)
                    _, logp, _, _ = transition(self.policy.pipe.scheduler,
                                               self.policy.predict(state, item["t"], condition),
                                               item["t"], state, self.policy.config["eta"], action=action)
                    error = float((logp.detach().cpu() - item["logp"]).abs().max())
                    replay.append(error)
                    if not np.isfinite(error) or error > 1e-6:
                        raise ValueError(f"Behavior log-probability replay failed: {error}")
                    # No batch-normalized advantages (B=1 would erase the gradient).
                    # Positive semantic and negative perceptual terms both enter this raw terminal reward.
                    loss = -float(reward) * logp.mean() / (len(episodes) * len(trajectory["steps"]))
                    loss.backward()
                    losses.append(float(loss.detach()))
        if any(p.grad is None or p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all()
               for p in self.params):
            raise FloatingPointError("Missing, non-FP32 or nonfinite LoRA gradient")
        if keep_gradients:
            self.last_raw = {name: value.grad.detach().cpu().clone() for name, value in self.named.items()}
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Zero or nonfinite raw gradient norm")
        if keep_gradients:
            self.last_postclip = {name: value.grad.detach().cpu().clone() for name, value in self.named.items()}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.params):
            raise FloatingPointError("Optimizer produced nonfinite LoRA parameters")
        self.step += 1
        self.policy.version += 1
        self.consumed.update(trajectory["trajectory_id"] for trajectory, _, _ in episodes)
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                    raise FloatingPointError("Adam moments must remain finite FP32")
        after = adapter_state(self.policy.pipe.unet)
        change = max(float((after[key] - before[key]).abs().max()) for key in before)
        if change <= 0:
            raise FloatingPointError("No actual LoRA update")
        if self.policy.editor.device.type == "cuda":
            torch.cuda.synchronize(self.policy.editor.device)
        return {"step": self.step, "loss": sum(losses), "gradient_norm_before_clip": float(norm),
                "finite_FP32_gradients_and_Adam_moments": True, "adapter_max_parameter_change": change,
                "behavior_logp_replay_max_difference": max(replay), "seconds": time.perf_counter() - started,
                "learning_rate": self.optimizer.param_groups[0]["lr"], "backward_transitions": len(losses),
                "gradient_clip_limit": self.max_grad_norm,
                "actual_optimizer_updates": 1,
                "adapter_dtypes": sorted({str(p.dtype) for p in self.params}),
                "optimizer_moment_dtypes": sorted({str(s[k].dtype) for s in self.optimizer.state.values()
                                                    for k in ("exp_avg", "exp_avg_sq")}),
                "algorithm": "DDPO score-function, one on-policy pass; not PPO"}

    def state(self):
        return {"adapter": adapter_state(self.policy.pipe.unet), "optimizer": cpu_copy(self.optimizer.state_dict()),
                "step": self.step, "version": self.policy.version, "contract": copy.deepcopy(self.policy.contract),
                "trainer_contract": self.trainer_contract,
                "consumed": set(self.consumed), "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate()}

    def save(self, directory, extra=None):
        directory.mkdir(parents=True, exist_ok=False)
        state = self.state()
        state["pilot_extra"] = extra
        path = directory / "training_state.pt"
        temporary = directory / "training_state.partial"
        torch.save(state, temporary)
        os.replace(temporary, path)
        (directory / "checkpoint_integrity.json").write_text(json.dumps({
            "schema": "benign_pilot_checkpoint_v2", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }, indent=2), encoding="utf-8")

    def restore(self, directory):
        path = directory / "training_state.pt"
        integrity = json.loads((directory / "checkpoint_integrity.json").read_text(encoding="utf-8"))
        if integrity["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Checkpoint byte checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["contract"] != self.policy.contract:
            raise ValueError("Immutable sampling/training contract changed")
        if state.get("trainer_contract") != self.trainer_contract:
            raise ValueError("Optimizer/clipping contract mismatch or unvalidated legacy checkpoint")
        load_adapter_state(self.policy.pipe.unet, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.policy.version, self.consumed = state["step"], state["version"], set(state["consumed"])
        torch.set_rng_state(state["rng_torch"])
        if state["rng_cuda"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint requires CUDA RNG restoration")
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        return state.get("pilot_extra")

    def witness(self):
        return {"adapter": adapter_state(self.policy.pipe.unet), "optimizer": cpu_copy(self.optimizer.state_dict()),
                "raw_gradients": self.last_raw, "postclip_gradients": self.last_postclip,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

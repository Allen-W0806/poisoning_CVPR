"""Categorical REINFORCE over a frozen, real pretrained-editor observation table.

No generator parameter is in this optimizer. No held-out test records are accepted.
"""
import time
import numpy as np
import torch
from .policy import EditPolicy
from ..utils.seed import seed_everything


class CachedEditorController:
    def __init__(self, train_records, validation_records, train_rewards, validation_rewards, learning_rate, seed=42):
        if len(train_records) != 10 or any(r["split"] != "train" for r in train_records):
            raise ValueError("Controller RL requires ten training-only records")
        if any(r["split"] != "validation" for r in validation_records):
            raise ValueError("Controller selection uses validation only")
        seed_everything(seed)
        self.policy = EditPolicy().cuda()
        self.optimizer = torch.optim.AdamW([p for p in self.policy.parameters() if p.requires_grad], lr=learning_rate, weight_decay=.01)
        self.train_images = [r["image"] for r in train_records]
        self.validation_images = [r["image"] for r in validation_records]
        self.train_rewards = torch.tensor(train_rewards, dtype=torch.float32, device="cuda")
        self.validation_rewards = torch.tensor(validation_rewards, dtype=torch.float32, device="cuda")
        self.step = 0
        self.initial = {n: p.detach().clone() for n, p in self.policy.named_parameters()}

    def update(self):
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        distribution = self.policy.distribution(self.train_images, "cuda")
        actions = distribution.sample()
        rewards = self.train_rewards[torch.arange(10, device="cuda"), actions]
        loss = -(distribution.log_prob(actions) * rewards.detach()).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in self.policy.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], 1.)
        self.optimizer.step()
        self.step += 1
        torch.cuda.synchronize()
        delta = max(float((p - self.initial[n]).abs().max().detach()) for n, p in self.policy.named_parameters() if p.requires_grad)
        assert delta > 0 and torch.isfinite(loss)
        assert all(torch.equal(p, self.initial[n]) for n, p in self.policy.named_parameters() if not p.requires_grad)
        return {"step": self.step, "batch_size": 10, "reward_mean": rewards.mean().item(),
            "reward_variance": rewards.var(unbiased=False).item(), "sample_rewards": rewards.tolist(),
            "actions": actions.tolist(), "loss": loss.item(), "gradient_norm": float(norm),
            "trainable_max_change": delta, "frozen_feature_projection_unchanged": True,
            "step_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}

    @torch.no_grad()
    def validate(self):
        probabilities = self.policy.distribution(self.validation_images, "cuda").probs
        return float((probabilities * self.validation_rewards).sum(1).mean())

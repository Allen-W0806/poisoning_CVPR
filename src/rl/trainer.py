"""REINFORCE trainer; evaluator supplies benign feedback, never security metrics."""
import math
import numpy as np
import torch
from .buffer import RolloutBuffer
from .reward import validation_reward, RewardWeights
from ..editing.semantic_validator import validate_semantics, synthetic_count
from ..evaluation.image_quality import image_quality

DEFAULTS = dict(seed=42, batch_size=10, gradient_accumulation_steps=1, learning_rate=1e-5,
                optimizer="adamw", weight_decay=.01, max_grad_norm=1., warmup_ratio=.05,
                mixed_precision="bf16", checkpoint_interval=100, log_interval=1, total_updates=3)


def synthetic_feedback(original, edited, record):
    if record.get("source") != "synthetic_v1":
        raise ValueError("Synthetic feedback is only defined on labelled synthetic fixtures")
    return {"semantics": validate_semantics(original, edited, record),
            "answer_consistency": float(synthetic_count(original) == synthetic_count(edited)),
            "quality": image_quality(original, edited),
            # Deterministic operation budget, not fabricated wall-clock cost.
            "cost": 1.0 / 10.0}


class PolicyTrainer:
    def __init__(self, policy, editor, device, config=None, evaluator=synthetic_feedback, weights=None):
        self.config = {**DEFAULTS, **(config or {})}
        if self.config["optimizer"] != "adamw":
            raise ValueError("Only AdamW is implemented")
        self.device = torch.device(device)
        self.policy = policy.to(device)
        self.editor = editor
        self.evaluator = evaluator
        self.weights = weights or RewardWeights()
        self.optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                           lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        warmup = math.ceil(self.config["warmup_ratio"] * self.config["total_updates"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: min(1., (s + 1) / max(1, warmup)))
        self.step = 0
        self.buffer = RolloutBuffer()
        self.amp = self.config["mixed_precision"] == "bf16" and self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def update(self, batches):
        if len(batches) != self.config["gradient_accumulation_steps"]:
            raise ValueError("Expected exactly gradient_accumulation_steps batches")
        for batch in batches:
            if len(batch) != self.config["batch_size"] or any(r["split"] != "train" for r in batch):
                raise ValueError("RL requires full training-only microbatches")
        self.optimizer.zero_grad(set_to_none=True)
        all_rewards, losses = [], []
        for j, records in enumerate(batches):
            images = [r["image"] for r in records]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                distribution = self.policy.distribution(images, self.device)
            # FP32 distribution arithmetic outside autocast for stable REINFORCE.
            distribution = torch.distributions.Categorical(logits=distribution.logits.float())
            actions = distribution.sample()
            edited = self.editor(images=images, conditions=[{"action": a} for a in actions.tolist()],
                                 seed=self.config["seed"] + self.step * len(batches) + j)
            rewards = []
            for record, original, output, action in zip(records, images, edited, actions.tolist()):
                feedback = self.evaluator(original, output, record)
                components = validation_reward(feedback["semantics"], feedback["answer_consistency"],
                              feedback["quality"]["distortion"], feedback["cost"], self.weights)
                self.buffer.append(record["sample_id"], action, components)
                rewards.append(components["reward"])
            r = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            if not torch.isfinite(r).all():
                raise FloatingPointError("Nonfinite rewards")
            loss = -(distribution.log_prob(actions) * r.detach()).mean() / len(batches)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            losses.append(float(loss.detach()))
            all_rewards.extend(rewards)
        gradients = [p.grad for p in self.policy.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], self.config["max_grad_norm"])
        used_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {"step": self.step, "loss": sum(losses), "batch_reward_mean": float(np.mean(all_rewards)),
                "grad_norm": float(norm), "learning_rate": used_lr, "amp_bf16": self.amp,
                "samples": len(all_rewards), "components": [t.__dict__ for t in self.buffer.drain()]}

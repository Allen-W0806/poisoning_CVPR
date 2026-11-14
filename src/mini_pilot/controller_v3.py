"""A separately optimized categorical REINFORCE controller, not a rule alias."""
import math

import torch
from torch import nn
from torch.distributions import Categorical


class RSIController(nn.Module):
    def __init__(self, state_dim=5, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, state):
        return Categorical(logits=self.net(state))


def apply_rsi_action(current_config, action):
    new_config = dict(current_config)
    if action == 1:
        new_config["lr"] = min(current_config["lr"] * 1.5, 5e-5)
    elif action == 2:
        new_config["lr"] = max(current_config["lr"] * .5, 1e-6)
    elif action == 3:
        new_config["weight_sem"] = min(current_config["weight_sem"] * 1.2, 5.)
    elif action != 0:
        raise ValueError("Unknown controller action")
    return new_config


def controller_state(step_fraction, common_reward, lpips, lr, budget_fraction):
    # Dimensionless scales prevent a raw step count from swamping a 1e-5 LR.
    return torch.tensor([step_fraction, common_reward, lpips, math.log10(lr) / 6., budget_fraction],
                        dtype=torch.float32)


class LearnedController:
    def __init__(self, seed):
        # Independent CPU RNG, checkpointed with the controller.
        self.policy = RSIController().float()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-3, weight_decay=0.)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.baseline = 0.
        self.memory = []
        self.pending = None

    def choose(self, state, config):
        if self.pending is not None:
            raise RuntimeError("Unsettled controller action")
        distribution = self.policy(state)
        action = int(torch.multinomial(distribution.probs.detach(), 1, generator=self.generator))
        self.pending = (state, action)
        return action, apply_rsi_action(config, action), distribution.probs.detach().tolist()

    def learn(self, validation_gain, evidence):
        if self.pending is None or not math.isfinite(validation_gain):
            raise ValueError("A finite causal validation outcome is required")
        state, action = self.pending
        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        self.optimizer.zero_grad(set_to_none=True)
        advantage = validation_gain - self.baseline
        loss = -self.policy(state).log_prob(torch.tensor(action)) * advantage
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.policy.parameters()):
            raise FloatingPointError("Nonfinite controller gradient")
        self.optimizer.step()
        change = max(float((a - b.detach()).abs().max()) for a, b in zip(before, self.policy.parameters()))
        item = {"state": state.tolist(), "action": action, "validation_gain_common_reward": validation_gain,
                "advantage": advantage, "baseline_before": self.baseline, "loss": float(loss.detach()),
                "controller_max_parameter_change": change, "evidence": evidence,
                "feedback_scope": "development validation lighting_v3 mean exposure/color/CLIP/LPIPS; ASR NOT TESTED"}
        self.memory.append(item)
        self.baseline = .9 * self.baseline + .1 * validation_gain
        self.pending = None
        return item

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Only settled controller states may be checkpointed")
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                "generator_rng": self.generator.get_state(), "baseline": self.baseline, "memory": self.memory}

    def load_state_dict(self, state):
        """Restore a settled REINFORCE controller; this does not make it PPO."""
        import copy
        if self.pending is not None:
            raise RuntimeError("Cannot restore over an unsettled controller action")
        if set(state) != {"policy", "optimizer", "generator_rng", "baseline", "memory"}:
            raise ValueError("Incomplete learned-controller checkpoint")
        expected = self.policy.state_dict()
        if set(state["policy"]) != set(expected) or not math.isfinite(state["baseline"]):
            raise ValueError("Controller checkpoint contract mismatch")
        for name, value in state["policy"].items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid learned-controller tensor")
        self.policy.load_state_dict(state["policy"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.generator.set_state(state["generator_rng"])
        self.baseline = float(state["baseline"])
        self.memory = copy.deepcopy(state["memory"])

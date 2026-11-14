import os
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, policy, optimizer, scheduler, step, config, contract):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    payload = {"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "step": step, "config": config, "contract": contract,
               "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
               "rng_torch": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_suffix(".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, optimizer, scheduler, expected_config, expected_contract):
    # Load only trusted project-owned checkpoints: RNG state requires weights_only=False.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["config"] != expected_config or payload["contract"] != expected_contract:
        raise ValueError("Checkpoint resume config/reward/editor/evaluator/data contract mismatch")
    policy.load_state_dict(payload["policy"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"])
    if payload["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload

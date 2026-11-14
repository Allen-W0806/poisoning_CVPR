from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class RewardWeights:
    semantic_weight: float = 1.0
    accuracy_weight: float = 0.5
    distortion_weight: float = 0.2
    cost_weight: float = 0.1
    invalid_semantics_penalty: float = 1.0


def validation_reward(semantic, answer_consistency, distortion, cost, weights=None):
    w = weights or RewardWeights()
    raw = [semantic.score, answer_consistency, distortion, cost]
    if not all(math.isfinite(float(v)) for v in raw):
        raise ValueError("Nonfinite reward component")
    v, a, d, c = [min(1., max(0., float(x))) for x in raw]
    valid = semantic.verified and v == 1.0
    penalty = 0.0 if valid else w.invalid_semantics_penalty
    total = w.semantic_weight * v + w.accuracy_weight * a - w.distortion_weight * d - w.cost_weight * c - penalty
    return {"semantic": v, "semantic_verified": semantic.verified, "answer_consistency": a,
            "distortion": d, "cost": c, "invalid_penalty": penalty,
            "reward_unclipped": total, "reward": min(1.5, max(-1., total))}

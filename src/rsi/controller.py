import copy
import math
from .memory import RevisionMemory
from .revision import validate_revision
from ..utils.logging import utcnow


class RSIController:
    def __init__(self, directory, config, initial_score, validation_hash, budget_gpu_hours, policy_version="v0", initial_cost_gpu_hours=0.):
        if not validation_hash or budget_gpu_hours <= 0 or not math.isfinite(initial_score):
            raise ValueError("Validation identity, finite score and positive budget are required")
        self.memory = RevisionMemory(directory)
        if not math.isfinite(initial_cost_gpu_hours) or not 0 <= initial_cost_gpu_hours <= budget_gpu_hours:
            raise ValueError("Invalid initial cost")
        self.state = {"policy_version": policy_version, "training_config": copy.deepcopy(config),
                      "validation_score": initial_score, "validation_hash": validation_hash,
                      "cost_gpu_hours": initial_cost_gpu_hours, "budget_gpu_hours": budget_gpu_hours,
                      "history": [], "failure_categories": [], "config_version": 0}
        self.memory.snapshot(self.state)

    def revise(self, changes, score, cost_gpu_hours, validation_hash, split="validation", policy_version=None,
               proposal_reason=None, evidence=None, rollback=None):
        if split != "validation" or validation_hash != self.state["validation_hash"]:
            raise ValueError("Only the fixed held-out validation split may select revisions")
        proposed = validate_revision(self.state["training_config"], changes)
        if not math.isfinite(score) or not math.isfinite(cost_gpu_hours) or cost_gpu_hours < 0:
            raise ValueError("Nonfinite score or invalid cost")
        total_cost = self.state["cost_gpu_hours"] + cost_gpu_hours
        accepted = score > self.state["validation_score"] and total_cost <= self.state["budget_gpu_hours"]
        reason = "accepted" if accepted else ("budget_exceeded" if total_cost > self.state["budget_gpu_hours"] else "no_validation_gain")
        revision = {"revision_number": len(self.state["history"]) + 1, "timestamp": utcnow(),
                    "previous_config": copy.deepcopy(self.state["training_config"]), "proposed_config": proposed,
                    "validation_hash": validation_hash, "validation_score": score,
                    "cost_gpu_hours": cost_gpu_hours, "accepted": accepted, "reason": reason,
                    "proposed_policy_version": policy_version or self.state["policy_version"]}
        revision.update(proposal_reason=proposal_reason, evidence=evidence, rollback=rollback)
        self.memory.append(revision)
        self.state["cost_gpu_hours"] = total_cost  # Rejected trials still count against total cost.
        self.state["history"].append(revision)
        if accepted:
            self.state["training_config"] = proposed
            self.state["validation_score"] = score
            self.state["policy_version"] = revision["proposed_policy_version"]
            self.state["config_version"] += 1
        else:
            self.state["failure_categories"].append(reason)
        self.memory.snapshot(self.state)
        return revision

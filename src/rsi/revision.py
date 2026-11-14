import math

ALLOWED_KEYS = {"batch_size", "learning_rate", "curriculum"}


def validate_revision(current, changes):
    if not changes or set(changes) - ALLOWED_KEYS:
        raise ValueError("Revision can change only benign training settings")
    proposal = {**current, **changes}
    if proposal.get("batch_size") not in {10, 20, 32}:
        raise ValueError("Unsupported batch size")
    lr = float(proposal.get("learning_rate", 1e-5))
    if not math.isfinite(lr) or not 0 < lr <= 1e-3:
        raise ValueError("Invalid learning rate")
    if proposal.get("curriculum", "uniform") not in {"uniform", "easy_first"}:
        raise ValueError("Unsupported curriculum")
    return proposal

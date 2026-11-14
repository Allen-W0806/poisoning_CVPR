from dataclasses import dataclass
import numpy as np


def synthetic_count(image):
    """Exact foreground-column oracle for our separated red-circle fixture only."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    foreground = (a[:, :, 0] > a[:, :, 1] + 80) & (a[:, :, 0] > a[:, :, 2] + 80)
    occupied = foreground.any(axis=0)
    return int(np.sum(occupied & ~np.r_[False, occupied[:-1]]))


@dataclass(frozen=True)
class SemanticResult:
    verified: bool
    score: float
    method: str
    reason: str


def validate_semantics(original, edited, record, external_verifier=None):
    if original.size != edited.size:
        return SemanticResult(True, 0.0, "geometry", "shape_changed")
    if record.get("source") == "synthetic_v1":
        expected = record["semantic_oracle"]["red_components"]
        good = synthetic_count(original) == expected == synthetic_count(edited)
        return SemanticResult(True, float(good), "synthetic_count_oracle", "preserved" if good else "count_changed")
    if external_verifier:
        return external_verifier(original, edited, record)
    return SemanticResult(False, 0.0, "unverified", "Natural-image semantics require independent review/oracle")

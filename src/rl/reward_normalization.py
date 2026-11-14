"""Fixed reference scaling, with no current-batch statistics in advantages."""
import math

def fixed_advantage(reward):
    if not math.isfinite(float(reward)) or not -1 <= reward <= 1.5:
        raise ValueError("Expected the preserved bounded benign reward")
    return float(reward)  # baseline 0; fixed reference scale 1

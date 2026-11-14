"""Harmless lighting proxies and bounded adaptive search. No security feedback."""
import math

import numpy as np


def lighting_metrics(source, edited, delta=.05, preservation_weight=.05, box=None):
    """Photometric instruction proxy, structural content proxy, independent SSIM.

    No edited ground truth or human labels are available. SSIM is evaluation
    only; the reward uses luminance error reduction and edge-angle agreement.
    """
    from skimage.metrics import structural_similarity
    first = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.
    second = np.asarray(edited.convert("RGB"), dtype=np.float64) / 255.
    if first.shape != second.shape or not np.isfinite(second).all():
        raise ValueError("Invalid edited image dimensions/pixels")
    if box is not None:
        left, top, right, bottom = box
        first, second = first[top:bottom, left:right], second[top:bottom, left:right]
    weights = np.array([.2126, .7152, .0722])
    a, b = first @ weights, second @ weights
    target = float(np.clip(a.mean() + delta, 0., 1.))
    before_error, after_error = abs(float(a.mean())-target), abs(float(b.mean())-target)
    ea = np.concatenate([x.ravel() for x in np.gradient(a)])
    eb = np.concatenate([x.ravel() for x in np.gradient(b)])
    denominator = float(np.linalg.norm(ea)*np.linalg.norm(eb))
    # Flat images have no measurable edge direction, so use zero penalty only
    # when both are flat. This is a proxy, never a semantic identity verdict.
    edge = (1. if np.linalg.norm(ea) == np.linalg.norm(eb) == 0 else 0.) if denominator == 0 else float(np.clip(ea @ eb / denominator, -1, 1))
    fidelity = before_error-after_error
    reward = fidelity-preservation_weight*(1-edge)
    win_size = min(7, min(a.shape))
    win_size -= 1-win_size % 2
    return {"training_reward_proxy": reward, "luminance_error_reduction": fidelity,
            "absolute_target_error": after_error, "source_luminance": float(a.mean()),
            "edited_luminance": float(b.mean()), "target_luminance": target,
            "edge_cosine_content_proxy": edge,
            "evaluation_ssim_source": float(structural_similarity(first, second, data_range=1.,
                                                        channel_axis=2, win_size=win_size)),
            "evaluation_clipped_channel_fraction": float(np.mean((second <= 0) | (second >= 1))),
            "evaluation_type": "photometric_and_structure_proxies_no_edited_ground_truth"}


def content_box(image, resolution):
    from PIL import ImageOps
    # Use the same resize implementation and floating-point operation order as
    # ImageOps.pad, including extreme aspect ratios and rounding ties.
    width, height = ImageOps.contain(image, (resolution, resolution), method=3).size
    # Match ImageOps.pad's round-to-nearest centering, including odd padding.
    left, top = round((resolution-width)/2), round((resolution-height)/2)
    return [left, top, left+width, top+height]


class RecipeSelector:
    """Small non-contextual UCB over LRs; explicitly adaptive search/HPO."""
    def __init__(self, method, rates, fixed_rate):
        if method not in ("fixed_recipe", "fixed_rule", "ucb_search"):
            raise ValueError("Unknown bounded search baseline")
        if not rates or any(not math.isfinite(rate) or not 0 < rate <= .001 for rate in rates):
            raise ValueError("Invalid benign learning-rate arm")
        self.method, self.rates, self.fixed = method, list(rates), fixed_rate
        self.counts, self.totals = [0]*len(rates), [0.]*len(rates)

    def propose(self, round_index):
        if self.method == "fixed_recipe": return self.fixed
        if self.method == "fixed_rule": return self.rates[round_index % len(self.rates)]
        if 0 in self.counts: return self.rates[self.counts.index(0)]
        total = sum(self.counts)
        scores = [value/count+math.sqrt(2*math.log(total)/count)
                  for count, value in zip(self.counts, self.totals)]
        return self.rates[max(range(len(scores)), key=lambda index: scores[index])]

    def observe(self, rate, measured_dev_delta):
        if not math.isfinite(measured_dev_delta): raise ValueError("Nonfinite dev measurement")
        if rate in self.rates:
            index = self.rates.index(rate)
            self.counts[index] += 1
            self.totals[index] += measured_dev_delta

    def state(self):
        return {"method": self.method, "rates": self.rates, "counts": self.counts,
                "totals": self.totals, "classification": "adaptive_search_HPO_not_PPO_or_verified_RSI"}


def paired_summary(baseline, candidate):
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired source observations required")
    if [(x['sample_id'], x['seed']) for x in baseline] != [(x['sample_id'], x['seed']) for x in candidate]:
        raise ValueError("Source/seed pairing mismatch")
    delta = np.array([b['training_reward_proxy']-a['training_reward_proxy'] for a,b in zip(baseline,candidate)])
    generator = np.random.default_rng(42)
    bootstrap = delta[generator.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return {"n_sources": len(delta), "n_training_seeds": 1,
            "mean_reward_delta": float(delta.mean()), "source_delta_std": float(delta.std(ddof=1)) if len(delta)>1 else None,
            "exploratory_source_bootstrap_95CI": np.quantile(bootstrap, [.025,.975]).tolist(),
            "source_deltas": delta.tolist(), "training_seed_uncertainty": "NOT_ESTIMATED",
            "learning_advantage_established": False}

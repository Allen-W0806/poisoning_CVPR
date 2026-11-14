"""Versioned, task-specific lighting objective; not a backdoor/ASR evaluator.

The source-relative CLIP term alone is a baseline subtraction. The explicit
linear-light target and chromaticity penalty are the substantive objective
revision. All task terms use the original content box, excluding letterboxing.
"""
import numpy as np
from PIL import Image

from .reward import ProxyRewardCalculator


VERSION = "lighting_v2_20261104"
EXPOSURE = 1.03
CHROMA_SCALE = .02


def linear_rgb(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def lighting_terms(original, edited, box):
    a, b = linear_rgb(original.crop(box)), linear_rgb(edited.crop(box))
    weights = np.array([.2126, .7152, .0722])
    target = np.minimum(EXPOSURE * a, 1.)
    source_error = float(np.abs((a - target) @ weights).mean())
    edited_error = float(np.abs((b - target) @ weights).mean())
    progress = float(np.clip((source_error - edited_error) / max(source_error, 1e-3), -1., 1.))
    valid = ((a @ weights) > .02) & ((a @ weights) < .9)
    chroma = lambda rgb: rgb / np.maximum(rgb.sum(-1, keepdims=True), 1e-9)
    chroma_mae = float(np.abs(chroma(b)[valid] - chroma(a)[valid]).mean()) if valid.any() else 0.
    return {"brightness_progress": progress, "target_exposure_multiplier": EXPOSURE,
            "source_target_luma_MAE": source_error, "edited_target_luma_MAE": edited_error,
            "content_linear_luma_delta": float(((b - a) @ weights).mean()),
            "content_encoded_luma_delta": float(((np.asarray(edited.crop(box), dtype=float)
                - np.asarray(original.crop(box), dtype=float)) / 255. @ weights).mean()),
            "content_chromaticity_MAE": chroma_mae,
            "color_cost": min(chroma_mae / CHROMA_SCALE, 5.)}


def analytic_exposure(image, box, factor):
    """Calibration control only, never a substitute for the diffusion generator."""
    rgb = linear_rgb(image.crop(box)) * factor
    rgb = np.clip(rgb, 0., 1.)
    encoded = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    result = image.copy()
    result.paste(Image.fromarray(np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)), box[:2])
    return result


class LightingRewardV2(ProxyRewardCalculator):
    def __init__(self, root, device="cuda"):
        super().__init__(root, device)
        self.source_cache = {}
        self.receipt.update(version=VERSION, task="benign +3% linear-light exposure",
            formula="w_sem*(CLIP_edit-CLIP_source) + brightness_progress - color_cost - w_lpips*LPIPS",
            metric_domain="CLIP/LPIPS/SSIM full 512 canvas; lighting/chroma original content box",
            brightness_progress="clip((source_target_MAE-edited_target_MAE)/max(source_target_MAE,0.001),-1,1)",
            color_cost="min(content linear-RGB chromaticity MAE / 0.02,5); source luma in (0.02,0.9)",
            calibration_control_is_generator=False)

    def measure_with_box(self, original, edited, target_prompt, box, weight_sem=2., weight_stealth=1.):
        import hashlib
        key = (hashlib.sha256(original.tobytes()).hexdigest(), target_prompt)
        if key not in self.source_cache:
            identity = super().measure(original, original, target_prompt)
            if abs(identity["LPIPS"]) > 1e-6 or identity["SSIM"] != 1.:
                raise AssertionError("Identity metric contract failed")
            self.source_cache[key] = identity["CLIP_cosine"]
        metric = super().measure(original, edited, target_prompt, weight_sem, weight_stealth)
        source_score = self.source_cache[key]
        terms = lighting_terms(original, edited, box)
        delta = metric["CLIP_cosine"] - source_score
        metric.update(legacy_common_reward=metric["common_reward"], CLIP_source=source_score,
                      CLIP_delta=delta, centered_CLIP_LPIPS=2 * delta - metric["LPIPS"], **terms)
        metric["training_reward"] = (weight_sem * delta + terms["brightness_progress"]
                                      - terms["color_cost"] - weight_stealth * metric["LPIPS"])
        metric["common_reward"] = (2 * delta + terms["brightness_progress"]
                                    - terms["color_cost"] - metric["LPIPS"])
        metric["reward_version"] = VERSION
        return metric

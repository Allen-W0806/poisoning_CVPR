"""Mean-property exposure target with continuous (non-hard-clipped) progress.

v2 was rejected on development outputs: per-pixel target error plus hard
clipping made the brightness term constant. Spatial preservation stays in LPIPS
and chromaticity; this term measures the requested mean lighting property.
"""
import numpy as np

from .lighting_reward_v2 import LightingRewardV2, linear_rgb, EXPOSURE

VERSION="lighting_v3_mean_progress_20261104"


def mean_lighting_terms(original,edited,box):
    weights=np.array([.2126,.7152,.0722])
    source=linear_rgb(original.crop(box));result=linear_rgb(edited.crop(box))
    source_mean=float((source@weights).mean());edited_mean=float((result@weights).mean())
    target_mean=float((np.minimum(EXPOSURE*source,1.)@weights).mean())
    baseline_error=abs(target_mean-source_mean);edited_error=abs(target_mean-edited_mean)
    raw=(baseline_error-edited_error)/max(baseline_error,1e-3)
    progress=raw/(1.+abs(raw))
    return {"source_mean_linear_luma":source_mean,"edited_mean_linear_luma":edited_mean,
            "target_mean_linear_luma":target_mean,"mean_brightness_error":edited_error,
            "mean_brightness_improvement_raw":raw,"brightness_progress":float(progress)}


class LightingRewardV3(LightingRewardV2):
    def __init__(self,root,device="cuda"):
        super().__init__(root,device)
        self.receipt.update(version=VERSION,
            formula="w_sem*(CLIP_edit-CLIP_source + brightness_progress) - color_cost - w_lpips*LPIPS",
            brightness_progress="x/(1+abs(x)); x=(source_mean_target_error-edited_mean_target_error)/max(source_mean_target_error,0.001)",
            mean_target="mean luma of clip(1.03*source_linear_RGB,0,1), original content only",
            rejected_prior="lighting_v2 per-pixel hard-clipped target saturated on42/43 development generations",
            limitations="mean lighting is not global semantic preservation; LPIPS/chroma are separate incomplete constraints")

    def measure_with_box(self,original,edited,target_prompt,box,weight_sem=2.,weight_stealth=1.):
        metric=super().measure_with_box(original,edited,target_prompt,box,weight_sem,weight_stealth)
        metric.update(rejected_v2_reward=metric["common_reward"],
                      rejected_v2_brightness_progress=metric["brightness_progress"])
        metric.update(mean_lighting_terms(original,edited,box))
        semantic=metric["CLIP_delta"]+metric["brightness_progress"]
        metric["training_reward"]=weight_sem*semantic-metric["color_cost"]-weight_stealth*metric["LPIPS"]
        metric["common_reward"]=2*semantic-metric["color_cost"]-metric["LPIPS"]
        metric["reward_version"]=VERSION
        return metric

"""CPU/source inspection; no model weights or GPU allocation."""
import argparse
import inspect
import json
from pathlib import Path
import _bootstrap
from src.utils.logging import write_json

def inspect_policy():
    import diffusers
    from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
        set_alpha_to_one=False, steps_offset=1, clip_sample=False)
    scheduler.set_timesteps(10)
    variances = [float(scheduler._get_variance(int(t), int(t)-100)) for t in scheduler.timesteps]
    assert min(variances) > 0
    source = inspect.getsource(StableDiffusionInstructPix2PixPipeline.__call__)
    assert "noise_pred_text, noise_pred_image, noise_pred_uncond" in source
    return {"diffusers": diffusers.__version__, "timesteps": scheduler.timesteps.tolist(),
        "variances": variances, "scheduler": dict(scheduler.config),
        "pipeline_source": inspect.getfile(StableDiffusionInstructPix2PixPipeline),
        "ddim_source": inspect.getfile(DDIMScheduler), "three_branch_cfg_source_present": True,
        "status": "SOURCE_INSPECTION_ONLY_NOT_GENERATOR_RL_PASS"}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output", default="tmp/phase3/inspection.json")
    a=p.parse_args(); result=inspect_policy(); write_json(Path(a.output),result); print(json.dumps(result,indent=2))

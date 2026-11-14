"""DDIM Gaussian transitions, adapted from DDIM eqs 12/16 and TRL DDPO.

Diffusers/TRL sources are Apache-2.0; see PHASE3_RL_ALGORITHM_REVIEW.md.
Only the explicitly tested epsilon/unclipped/leading DDIM contract is supported.
"""
import math
import torch


def transition(scheduler, epsilon, timestep, sample, eta=1., action=None, noise=None):
    c = scheduler.config
    if c.prediction_type != "epsilon" or c.clip_sample or c.thresholding or c.timestep_spacing != "leading":
        raise ValueError("Unvalidated DDIM configuration")
    if scheduler.num_inference_steps is None or not 0 < eta <= 1:
        raise ValueError("A stochastic initialized DDIM scheduler is required")
    if (sample.dtype != torch.float32 or sample.shape != epsilon.shape
            or sample.device != epsilon.device or not torch.isfinite(sample).all()
            or not torch.isfinite(epsilon).all()):
        raise ValueError("FP32 continuous states and matching epsilon shape required")
    t = int(timestep)
    if t not in scheduler.timesteps.tolist():
        raise ValueError("Timestep is outside the initialized sampling schedule")
    previous = t - c.num_train_timesteps // scheduler.num_inference_steps
    at = scheduler.alphas_cumprod[t].to(sample.device)
    ap = (scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod).to(sample.device)
    variance = ((1-ap)/(1-at) * (1-at/ap)).clamp_min(0)
    std = eta * variance.sqrt()
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Zero/nonfinite DDIM variance: deterministic transitions have no Gaussian density")
    epsilon = epsilon.float()
    x0 = (sample - (1-at).sqrt()*epsilon) / at.sqrt()
    mean = ap.sqrt()*x0 + (1-ap-std.square()).clamp_min(0).sqrt()*epsilon
    if action is not None and noise is not None:
        raise ValueError("Supply either action for evaluation or noise for sampling")
    if action is None:
        if noise is None:
            raise ValueError("Sampling requires explicit reproducible noise")
        if (noise.shape != sample.shape or noise.dtype != sample.dtype
                or noise.device != sample.device or not torch.isfinite(noise).all()):
            raise ValueError("Noise must be finite, matching FP32 state shape/device")
        action = mean + std*noise
    if (action.shape != mean.shape or action.dtype != sample.dtype
            or action.device != sample.device or not torch.isfinite(action).all()):
        raise ValueError("Invalid recorded action")
    # Detach sampled action: this is a likelihood-ratio gradient, not pathwise MSE.
    coordinate_logp = -(action.detach()-mean).square()/(2*std.square()) - std.log() - .5*math.log(2*math.pi)
    logp = coordinate_logp.flatten(1).mean(1)
    if not torch.isfinite(logp).all():
        raise FloatingPointError("Nonfinite transition likelihood")
    return action, logp, mean, std


def assert_fresh(trajectory, version, contract, consumed):
    if trajectory["policy_version"] != version or trajectory["contract"] != contract:
        raise ValueError("Stale or incompatible behavior trajectory")
    if trajectory["trajectory_id"] in consumed:
        raise ValueError("Trajectory already consumed; no off-policy reuse")


def assert_fresh_batch(trajectories, version, contract, consumed):
    """Reject duplicates before any gradients/optimizer activity, without mutation.

    Distinct trajectories of the same condition are permitted (e.g. controlled
    continuation tests); trajectory identity must always be unique.
    """
    seen = set(consumed)
    for trajectory in trajectories:
        assert_fresh(trajectory, version, contract, seen)
        seen.add(trajectory["trajectory_id"])

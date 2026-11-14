# Phase 3 algorithm review

Selected: DDPO score-function (DDPO-SF), a single on-policy likelihood-ratio update. The denoising MDP formulation supports terminal nondifferentiable rewards. [DDPO paper](https://arxiv.org/abs/2305.13301).

| Candidate | Stochastic likelihood | LoRA | Image conditioning | Decision |
|---|---|---|---|---|
| Native Euler ancestral editor | Not validated by this project | Yes | Native 8-channel IP2Pix | Preserve historical blocker |
| TRL DDPO StableDiffusion wrapper | DDIM Gaussian, coordinate mean log density | Yes | Ordinary text-to-image wrapper | Do not copy unchanged |
| IP2Pix + explicit stochastic DDIM + DDPO-SF | Positive-variance Gaussian transitions | Actual UNet attention LoRA | Source mode + instruction + 3 CFG branches | Validate numerically then GPU |
| Supervised DDPM MSE | Denoising target regression | Yes | Yes | Not RL |
| External categorical controller | Categorical action probability | External head only | Frozen generator | Not generator RL |

Inspected [TRL v0.11.4 scheduler/pipeline implementation](https://github.com/huggingface/trl/blob/v0.11.4/trl/models/modeling_sd_base.py) and [trainer](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/ddpo_trainer.py); algorithm reference only, no TRL package installation. The current [Diffusers DDPO guide](https://huggingface.co/docs/diffusers/training/ddpo) directs users to TRL; [TRL documentation](https://huggingface.co/docs/trl/) is version-sensitive. We inspect pinned historical implementation rather than assume an installed trainer supports IP2Pix.

The [pinned IP2Pix source](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py) orders text embeddings [instruction, empty, empty], image conditions [source, source, zero]. It concatenates 4 noisy plus 4 unscaled source-latent channels. Guidance is e_uncond + text_scale*(e_text-e_image) + image_scale*(e_image-e_uncond). These conditions must be identical during collection and likelihood evaluation. Fixed VAE and text encoder remain frozen.

We preserve the pretrained beta schedule and select [DDIM](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py), eta=1, offset=1, set_alpha_to_one=false, leading spacing and no clipping. CPU inspection finds all ten variances positive, including t=1 to final alpha[0]. No deterministic transition is assigned a Gaussian density; incompatible configurations fail. Float32 sampled states/noise avoid BF16 action quantization; BF16 UNet predictions define the floating-point policy mean. This scheduler change is an explicitly different editing policy from Phase2 Euler.

For state z_t and fixed conditions, transition mean mu follows DDIM equations 12/16 and sigma=eta*sqrt(variance). Joint log probability is sum of coordinate Gaussian log probabilities. Terminal image is the fixed VAE/postprocessing map of the final latent. Objective J=E[R]; loss is -mean_i R_i*mean_t mean_coordinate log p_psi(z_next|z_t,x,c). Fixed dimensions and step count multiply the exact score-function gradient by a positive constant. Reward is detached, baseline zero and scale one fixed. No batch whitening, importance ratios, PPO clipping, off-policy replay or KL reference. Behavior checkpoint/log probabilities are retained for provenance and replay, not a regularization reference. One optimizer update invalidates prior trajectories.

Numerical tests compare every stochastic transition with Diffusers and analytical Normal density, test sign reversal under reversed reward, and reject zero-variance/stale samples. GPU evidence is reported separately; CPU/source compatibility alone is not a generator RL PASS.

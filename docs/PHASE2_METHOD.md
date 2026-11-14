# Phase 2 method and claim boundaries

## Real generator updates

The native InstructPix2Pix UNet predicts DDPM noise conditioned jointly on noisy target latent, original image latent and instruction embedding. Training pairs are original training images and deterministic brightness ×1.02 targets. Full images are aspect-preservingly padded to 512, never cropped. Source conditioning uses VAE latent mode without target scaling; target latent sample uses the VAE scaling factor. A sampled diffusion timestep/noise determines supervised epsilon-prediction MSE.

Only attention projection LoRA A/B matrices receive AdamW updates. Base UNet, VAE and CLIP text encoder weights stay frozen and are checked tensor-by-tensor through hashes. Rank 8 and alpha 16 give scale alpha/r=2. Both scaling metadata and exact FP32 A/B values must survive public export. Generator computation uses BF16, trainable adapters/master optimizer state FP32. Public loading attaches LoRA while UNet is FP32, then restores frozen base BF16, avoiding irreversible adapter rounding. Math SDPA covers forward, activation recomputation and backward. Checkpoint tests keep exact same-device state assertions and a separately declared public PNG tolerance of two uint8 levels.

This is supervised denoising, not reinforcement learning. The reward optimizer interface for actual diffusion trajectories raises an explicit unsupported error. A future DDPO-style optimizer needs validated stochastic transitions/log probabilities, reference comparison and trajectory gradient/masking tests. No ASR or evasion objective is present.

## Valid categorical controller RL

For the fixed real editor and seed, each source image has three benign instruction-conditioned outputs. A categorical policy over instruction index defines a stochastic bandit: actions are sampled from the policy; the selected per-image reward is looked up from the fixed real-output table. Loss is negative mean log-probability times detached per-sample reward. Updating this head is genuine REINFORCE for the discrete controller, but it does not update the generator.

The table uses ten training and ten validation source images. Frozen and random controls share the same table. Fixed RL uses six batch-10 updates and two validation lookups; RSI uses two three-update candidates from common initialization and two validation lookups. Both expose 60 training samples. Actual initialization, generation, Qwen judging, optimization and selection time are reported, not assumed equal. Shared table construction is charged once in total phase allocation; it is not hidden behind cheap lookup timings.

The RSI rule reads actual sampled reward variance and gradient norms before proposing LR 5e-5 (otherwise retains 1e-5). Only strict fixed-validation score improvement within budget can accept. Every proposal retains prior/new config, reason, evidence, cost and rollback checkpoint. Test data and metric definitions are immutable. This small single-seed fixed-table check does not establish learning quality or RSI advantage.

## Semantic and VQA evaluation

Natural-image validation separates visual similarity, answer consistency, semantic preservation and pixel-change localization. The paired Qwen judgment is question-aware (object identity, color, count, spatial/other relevant attributes and answerability). Its agreement with a short-answer Qwen check is correlated automatic evidence, not an independent oracle. Changed pairs remain valid=false/verified=false and may only be provisionally_valid; they require independent review before answer-preserving accuracy aggregation. Exact identity and deliberately fully occluded diagnostic controls have deterministic decisions. Raw disagreements, parse failures and a human review form are retained.

Phase 2 VQA calls the licensed pinned upstream scorer directly after mechanical Python-3 syntax conversion. It preserves upstream conditional normalization behavior, all ten human references, raw output, whitespace-only short-answer extraction and a normalized diagnostic field. The protocol pins model revision, system prompt, pixels, greedy generation and scorer/split checksums before full development. The 300-question train2014-derived internal test is opened once after completed development under the same protocol; it is not the official test-server benchmark.

No poisoning, textual trigger, target-output training, attack discovery or detection evasion is implemented. Independent security aggregation is unchanged and receives no new real security fixture in this phase.

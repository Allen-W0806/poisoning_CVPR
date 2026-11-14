# Generator training report

Evidence: `EXP_20261027T052543_a20882fb60de_phase2_generator_lora_smoke`. Actual pretrained UNet attention LoRA, not a separate controller, received three AdamW updates. Objective is **supervised image-conditioned DDPM epsilon-prediction MSE**, not RL.

Total UNet-with-adapter parameters: 861,126,852; pipeline parameters: 1,067,841,195; trainable adapter parameters: 1,594,368 (0.1851% of generator). Rank 8, alpha 16, FP32 adapter/master parameters, BF16 frozen generator computation. VAE and text encoder frozen. Actual matched module list is in metrics.json.

Logical batch 10 = physical microbatch 1 × accumulation 10. Three updates, LR 1e-5, weight decay .01, max norm 1, constant LR for this smoke, gradient checkpointing. Ten training records only; targets are deterministic brightness ×1.02 pairs. Source latent mode is unscaled, target latent sample is scaled by the VAE factor, noisy target is concatenated with source latent, and sampled DDPM noise is the prediction target. No Qwen/security reward enters this loss. Official recipe has optional conditioning dropout; this minimal run uses none.

Maximum adapter parameter change: 3.00479169e-05; finite gradients at every update. All 1130 frozen parameter tensors were hashed before/after and stayed unchanged. Adapter artifact: 6,416,152 bytes; AdamW state after last update: 12,755,968 bytes.

Peak observed forward allocation: 3.677 GB; peak through backward: 4.603 GB. These include persistent model/accumulated gradient storage and are not incremental activation-only sizes. Mean update time 5.755s. The smoke also performs a resumed fourth diagnostic update; all allocation time is charged in the phase ledger.

Fresh training-state reload pixel max difference: 0; public exported adapter reload difference: 0 (predeclared tolerance ≤2 uint8 levels). Loaded rank/alpha checked as 8/16. Public adapter tensors retain float32 with exact max tensor difference 0.0. Resumed next-update adapter, AdamW and scheduler state matched exactly in this completed run. Checkpoints retain model revision, modules, data IDs, config and Python/NumPy/Torch/CUDA RNG. Public export must preserve PEFT metadata and exact FP32 A/B values; loading into BF16 and upcasting afterward loses information.

Earlier failures remain in reports/PHASE2_FAILURES.md and immutable runs: omitted offline weight_name, lost alpha metadata, a 4.41e-8 fused-attention restart difference, and a math-attention context lifetime bug. Final numerical isolation uses math SDPA through forward/backward/recomputation; no acceptance tolerance was enlarged. Success here does not erase the earlier non-bitwise result or prove universal cross-kernel reproducibility.

Generated images remain valid PNGs. This validates updates/restart plumbing, not visual quality improvement, instruction fidelity, robustness, or diffusion RL. Example after/reloaded images are in the experiment images directory.

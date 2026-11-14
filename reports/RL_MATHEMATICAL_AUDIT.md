# RL mathematical and implementation audit

**Generator RL correctness: CONDITIONAL GO at the formulation/wiring level; no authorization for new training. Strict continuation and scientific reward/evaluation gates remain open.**

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


## Estimator and assumptions

For source/instruction condition c, a denoising trajectory has density

`pθ(τ|c)=p(zT) ∏t N(zt−1; μθ(zt,t,c), σt²I)`.

For a fixed terminal reward functional, the score-function identity is

`∇θ E[R] = E[R ∑t ∇θ log pθ(zt−1|zt,c)]`.

It requires current-policy trajectories, parameter-independent initial noise, correct transition density, sampled states/actions held constant during the score derivative, positive transition variance, and no unaccounted θ-dependence in reward. The frozen nondifferentiable Qwen reward is permissible. The original [DDPO paper §4.3](https://arxiv.org/html/2305.13301) distinguishes one-step score-function updates from importance-sampled multi-update variants. The implementation here is the former.

The code minimizes `−(1/B) Σi Ri (1/T) Σt (1/d) Σj log Normal(action_ij;mean_ij,std_t)` where B10, T10, d=4×64×64=16384. Thus the expected raw gradient is the standard terminal-reward score gradient scaled by1/(T*d). It is a scaled log-density score, not an unscaled joint trajectory log probability. Scaling is constant for the fixed protocol; varying image dimensions/step counts would change weighting. Adam epsilon, gradient clipping and finite precision mean a constant gradient scale is not a guarantee of identical optimizer behavior or equivalence to another DDPO implementation.

## Transition derivation and CPU verification

Let αt be cumulative alpha and αp the previous timestep cumulative alpha. The implemented variance is `(1−αp)/(1−αt) * (1−αt/αp)`, σ=eta√variance; x0=(zt−√(1−αt)epsilon)/√αt; μ=√αp*x0+√(1−αp−σ²)*epsilon. The replay action is detached before the Gaussian score. Parameters are not optimized through sampled actions or the reward model.

The new CPU audit independently compares all ten synthetic steps to the installed [Diffusers 0.35.1 DDIM scheduler](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py): max output difference0.0. Gaussian log density and its epsilon gradient agree with `torch.distributions.Normal`. All ten standard deviations are positive, including the final0.0206482187. This checks algebra on CPU, not actual BF16 UNet backward reproducibility.

## Batch and update accounting

| Quantity | Per logical update | Phase4 trained-arm total |
|---|---:|---:|
| Source-image exposures / trajectories |10 |320 |
| Independent unique sources within each batch |10 |154 unique sources across seeds/arms |
| Denoising transitions with backward calls |10×10=100 |3200 |
| Optimizer steps |1 |32 |
| Physical source microbatch |1 |Sequential accumulation |
| CFG UNet branches per transition |3 sequential |Distinct from source batch count |

An additional180 development trajectories contribute1800 denoising transitions without optimizer steps. The CPU audit reads500 stored trajectories and verifies5000 complete, chained, FP32 transitions with expected timesteps, finite fields and positive variances. The two historical diagnostic optimizer replays are separate from32 training steps.

## Code-path assessment

- `DiffusionPolicy.sample` runs no_grad and stores source/image/instruction/seed/version/contract, states, sampled actions, noise and behavior score. The source latent and text embeddings are fixed conditions.
- `assert_fresh` checks version, contract and prior consumption. Each batch is fully validated before the optimizer step. There are no stale epochs or off-policy replay in recorded training.
- `update` recomputes scores at the unchanged behavior parameters; all recorded max likelihood discrepancies are0. It accumulates all100 scaled transition losses before one optimizer step. Terminal reward is assigned once per source and shared across its ten transitions; this is appropriate credit assignment, not100 independent rewards.
- Reward normalization is fixed baseline0/scale1, without batch whitening or self-max normalization. Stored behavior scores are a consistency check, not PPO ratios. There is no frozen-reference KL anchor; that is an explicit on-policy SF choice, not an omitted PPO requirement.
- AdamW uses1e-5 (M2 later5e-6), weight_decay0, constant LambdaLR and max_grad_norm1. Gradients are checked finite/nonzero; each recorded update changes the intended adapters; frozen hashes stay equal. Gradient sign-reversal checks verify wiring but do not prove a useful reward or convergence.
- The finite-gradient result and59 passing CPU tests are supportive engineering evidence. They do not prove an unbiased real-device implementation under every future configuration, statistical gain, semantic correctness or bitwise continuation.

## F09 — HIGH: Exact stochastic checkpoint continuation remains unaccepted

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_rl_trainer.py:78](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:78); [scripts/diagnose_phase4_restart.py:34](H:/Project/Poisoning/scripts/diagnose_phase4_restart.py:34); [scripts/audit_phase4_restart.py:24](H:/Project/Poisoning/scripts/audit_phase4_restart.py:24); [scripts/phase4_restart_outcomes.py:1](H:/Project/Poisoning/scripts/phase4_restart_outcomes.py:1).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`, `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`, `EXP_20261027T175526_5aee05413ae5_phase4_PH4_RESTART_OUTCOMES`.

Observed: Fixed-trajectory replay has exact checked restored fields and exact behavior likelihoods, but backward gradients and post-update adapters/optimizer states differ. Adapter max differences are 2.87986e-6 and 2.86177e-6. Two outcome sources produce unequal RGB images (max channel differences 103/37/93/40). The fixed-trajectory replay bypasses ten rollout seed draws; it does not establish full stochastic RNG continuation.

Proposed correction: Keep exact equality as the acceptance criterion. Permit no resume-equivalence claim until a separately authorized diagnostic closes it. No new numerical fix or threshold relaxation is made here.

## F12 — MEDIUM: Verified identity loses its answer-consistency credit at the caller

Evidence status: **VERIFIED**. Source/function: [src/evaluation/semantic_validation.py:37](H:/Project/Poisoning/src/evaluation/semantic_validation.py:37); [scripts/smoke_generator_rl.py:49](H:/Project/Poisoning/scripts/smoke_generator_rl.py:49).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The verified-identity branch omits provisionally_valid; collect gates A solely on that key. With raw consistency1, identity is passed A=0 and R=0.9666667 (C=1/3), rather than 1.4666667 if the declared consistency component were credited. This is a definite branch inconsistency, not an observed cause of Phase4 results: all 500 current edits were nonidentity/unverified.

Proposed correction: Under a future versioned implementation correction, explicitly define eligibility for verified-valid and provisional cases and add an identity branch test. Do not rewrite any historical reward.

## F15 — MEDIUM: Checkpoint configuration revisions require reconstructing the exact active contract

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_rl_trainer.py:80](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:80); [scripts/run_phase4_rsi.py:128](H:/Project/Poisoning/scripts/run_phase4_rsi.py:128); [scripts/run_phase4_rsi.py:129](H:/Project/Poisoning/scripts/run_phase4_rsi.py:129).

Supporting experiment IDs: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: LR revision updates optimizer LR, scheduler base_lrs and policy contract and creates a child revision record. The parent manifest/config remains the initial version; subsequent checkpoints carry the active revision. Reconstructing a parent-only default policy would fail the contract check. No production resume across this revision was accepted or used.

Proposed correction: Document the active revision lineage and reconstruction procedure; retain immutable parent and child records. Validate it before any future authorized resume; do not treat an intentional contract rejection as successful continuation.

## F17 — UNRESOLVED: The specific backward numerical divergence mechanism is not localized

Evidence status: **UNRESOLVED**. Source/function: [scripts/diagnose_phase4_restart.py:47](H:/Project/Poisoning/scripts/diagnose_phase4_restart.py:47); [scripts/audit_phase4_restart.py:29](H:/Project/Poisoning/scripts/audit_phase4_restart.py:29).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`, `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`.

Observed: The earliest inspected difference is in backward gradients after exact forward log-likelihoods. That does not identify the first differing CUDA primitive, exclude hidden execution-state differences, or prove an unavoidable universal nondeterminism mechanism.

Proposed correction: Retain competing hypotheses and exact diagnostics. Only separately authorized controlled diagnostics may identify a root cause; do not invent a fix or declare all GPU determinism impossible.

## F19 — MEDIUM: Reward configuration metadata is not the executed coefficient source

Evidence status: **VERIFIED**. Source/function: [configs/phase3_generator_rl.yaml:24](H:/Project/Poisoning/configs/phase3_generator_rl.yaml:24); [scripts/smoke_generator_rl.py:48](H:/Project/Poisoning/scripts/smoke_generator_rl.py:48); [src/rl/reward.py:15](H:/Project/Poisoning/src/rl/reward.py:15).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: collect calls validation_reward without a weights object, so dataclass defaults, not the YAML reward block, determine executed coefficients. They numerically match this campaign. Several YAML names also differ from dataclass field names. Editing YAML alone would silently fail to change the actual reward.

Proposed correction: Before future protocol changes, choose one explicit validated coefficient source and assert it equals recorded metadata. Current calculations are correct for the defaults; no coefficient or scientific source is changed in this audit.

## F23 — LOW: Generalized freshness checks have an unexercised duplicate-within-batch gap

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_trajectory.py:45](H:/Project/Poisoning/src/rl/diffusion_trajectory.py:45); [src/rl/diffusion_rl_trainer.py:31](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:31); [src/rl/diffusion_rl_trainer.py:57](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:57).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The trainer checks against previously consumed IDs before the batch, then adds all IDs only after the update. It does not independently reject two copies of the same fresh trajectory inside one input list. Actual collectors generate unique IDs and distinct sources, and all 32 recorded batches were verified distinct; no current duplicate update is found.

Proposed correction: Add an explicit per-batch trajectory/source uniqueness contract before generalizing this trainer. Do not infer contamination of current recorded batches.

No decisive score-sign, missing-source-conditioning or batch-accounting bug was found in the executed fixed protocol. The known identity reward branch and configuration wiring gaps are distinct from the sampled-action gradient. Numerical divergence remains an observed failure with an unresolved mechanism; it has not been repaired or excused.

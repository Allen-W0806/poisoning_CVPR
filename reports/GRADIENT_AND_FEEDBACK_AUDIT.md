# Gradient and Feedback Audit — 2026-10-28

**Decision: the bounded synthetic gradient and wiring checks pass. Frozen benign VQA is reproducible. Feedback is not validated as a GRPO candidate-ranking signal, and editor semantic preservation fails. Large-scale fine-tuning remains disabled.**

Canonical runtime: `/storage/home/ydk297/projects/poisoning`; local mirror: `H:/Project/Poisoning`. This stage uses existing `.venv-editor` and central model caches. No pretrained VLM or editor optimizer step was performed. Existing reward-release guards, old results, failed checkpoints, and security-evaluation records were preserved.

## Current implementation and parameter ownership

| Entry point / implementation | Actual algorithm | Optimizer ownership and schedule |
|---|---|---|
| `scripts/smoke_rl.py`; `src/rl/trainer.py:PolicyTrainer`; `src/rl/policy.py:EditPolicy` | Categorical REINFORCE, `-mean(log p(action) * detached reward)`; on-policy, no ratios or group normalization | `head.weight` (3×8), `head.bias` (3); 27 trainable elements. `encoder.weight` (8×3) and `encoder.bias` (8) frozen. AdamW, default LR 1e-5, weight decay .01, gradient norm cap 1; LambdaLR warmup then constant; no decay. |
| `scripts/compare_rl_rsi.py`; `src/rl/cached_editor_trainer.py:CachedEditorController` | Categorical REINFORCE over a cached frozen-editor reward table | Same head only; AdamW, configured LR, weight decay .01, gradient cap 1. No diffusion parameters. Validation selection distinct from train reward. CUDA-only implementation. |
| `scripts/smoke_generator_training.py`; `src/training/generator_trainer.py:GeneratorTrainer` | Supervised epsilon-denoising MSE; **not RL** | Attention LoRA only, AdamW configured LR/weight decay; constant LambdaLR. Frozen source VAE/text encoder. |
| `scripts/smoke_generator_rl.py`, `scripts/run_phase4.py`; `src/rl/diffusion_policy.py:DiffusionPolicy`; `src/rl/diffusion_rl_trainer.py:DiffusionRLTrainer` | Single-pass on-policy DDPO score-function estimator, with fixed coordinate/time scaling; not PPO or GRPO | All and only UNet parameters with `requires_grad=True`: attention LoRA A/B on `to_q`, `to_k`, `to_v`, `to_out.0`. Current pinned rank8/alpha16 config creates **256 tensors / 1,594,368 elements**. AdamW LR 1e-5, weight decay 0, betas (.9,.999), epsilon 1e-8, constant LambdaLR, norm cap 1. |
| `src/rl/diffusion_policy_v2.py`; `src/rl/diffusion_rl_trainer_v2.py`; `src/rl/phase5_backend.py` | Content-bound guarded wrapper of the same DDPO-SF backend | Adds source-content/uniqueness checks. `require_training_reward` always rejects training because no independently validated reward has been released. This guard was not bypassed. |
| `scripts/validate_policy_gradients.py` | Isolated categorical reference and an explicitly named one-step GRPO reference | Synthetic logits only. GRPO uses groups of 16 from the same context, eight groups per context, two contexts, three surrogate epochs per rollout, old log-probability snapshots, clipping .2, exact categorical KL to fixed uniform reference with coefficient .02, gradient cap 1, Adam LR .08. |

Actual instantiated optimizer inventories, including every tensor name and shape:

- [Categorical parameter groups](../results/gradient_feedback_20261028_remote_v2/optimizer_groups.json).
- [Pretrained IP2Pix parameter groups — zero optimizer steps](../results/ddpo_wiring_20261028_v3/pretrained_IP2Pix_optimizer_groups_ZERO_STEPS.json).
- [Tiny synthetic UNet parameter groups](../results/ddpo_wiring_20261028_v3/tiny_optimizer_groups.json).
- [Parameter updates and frozen hashes](../results/parameter_update_audit.csv).

LoRA initialization is implemented by `src/training/generator_adapter.py:add_attention_lora`: freeze UNet, attach PEFT adapters to matching attention linear layers, cast trainable A/B to FP32. Gaussian A initialization and initially zero B imply that zero A gradient on the first update is expected. Both A and B changed after two controlled tiny-UNet updates. Creating new pretrained-model adapters for ownership inspection is distinct from updating them; their optimizer performed zero steps.

## Gradient identity and statistical acceptance

For categorical actions with fixed reward vector `[0, .35, 1]`, the reference is `J(theta) = sum_a p_theta(a) R(a)`. We enumerate all three actions and compute its autodiff gradient. An independent score-function construction computes `sum_a p(a) R(a) grad(log p(a))`. The sampled loss is differentiated separately; rewards and sampled action indices are constants.

Both a direct three-logit policy and the **existing EditPolicy** agree with exact enumeration to approximately 3.5e-17. Thirty-two independent RNG seeds use N=128, 1,024, 8,192 and 65,536. Acceptance was frozen before sampling: coordinate mean error below four standard errors, final relative RMSE <10%, final cosine >.95, and log-RMSE/log-N slope between -.75 and -.25. Coordinate errors are correlated; the four-SE limit is conservative for the small tested gradient dimension. This is a finite diagnostic with Monte Carlo uncertainty, not universal proof of all training regimes.

| Estimator | Final relative RMSE | Error slope | Minimum final cosine |
|---|---:|---:|---:|
| Direct categorical logits | .0068014 | -.46855 | .9999737 |
| Existing EditPolicy head | .0061015 | -.47232 | .9999869 |
| Existing DDPO Gaussian transition, scaled by 1/16 | .0575715 | -.48429 | Scalar estimator; see exact/estimated plot |

The DDPO reference uses `R(action)=cos(action[first coordinate])`, with an analytically known Gaussian expectation `cos(mu) exp(-sigma^2/2)`. The existing `src/rl/diffusion_trajectory.py:transition` computes a **coordinate mean** log density; the reference therefore scales the exact gradient by 1/16 for its 4×2×2 synthetic latent. The derivative treats sampled actions and states as fixed. This directly tests the production transition implementation rather than an alternative density function.

The initial DDPO N=65,536 result **failed** the same 10% precision criterion (relative RMSE .227693), despite mean error z=1.078 and the correct convergence trend. This was diagnosed as high estimator variance, not evidence of bias. A separately frozen v2 protocol increased only DDPO sample counts to 8,192 / 65,536 / 524,288 / 1,048,576, streamed in chunks of 32,768. The cosine reward, gradient normalization, seeds, and all tolerances stayed unchanged. Final mean error z=2.410 and RMSE .0575715 pass. Both protocols and negative outputs remain available.

Production diffusion loss is proportional to `-mean_i R_i * mean_t mean_coordinate log p_theta(action_it | state_it,condition_i)`. For fixed trajectory length T and coordinate count d, its expected gradient is the ordinary terminal-reward policy gradient divided by T*d. It is not an unscaled joint trajectory log probability. Varying resolution or step count changes that scaling. Clipping, Adam epsilon, and BF16 arithmetic prevent assuming optimizer equivalence solely from equal gradient direction. See the [DDPO paper](https://arxiv.org/abs/2305.13301).

## Propagation, checkpoint, and group-relative checks

The existing categorical trainer was run with an isolated harmless identity editor that marks which categorical action was selected. Reward is exactly the declared finite action reward; no security or cost term is active. `loss.requires_grad`, exact optimizer ownership, finite nonzero gradients, nonzero per-step head changes, frozen encoder hashes, nonfinite-reward rejection, checkpoint reload, and exact next optimizer step all pass. Actual head weight/bias changes were approximately .01; the larger test LR .01 is explicitly diagnostic, distinct from repository default 1e-5.

The actual **DiffusionRLTrainer** also executed four optimizer calls on a randomly initialized tiny synthetic UNet: two initial updates plus one checkpoint continuation and its replay. It uses 4,096 trainable LoRA elements, two harmless trajectories per update and two stochastic DDIM transitions per trajectory. Base weights are frozen. Adapter A/B changed; frozen hashes remained equal; gradients were finite/nonzero; checkpoint tensors and the fixed-trajectory next update matched exactly. Sampling and replay both use math SDPA. A failed attempt used different attention backends and triggered the unchanged 1e-6 behavior-likelihood assertion (difference 3.71933e-5); the harness was corrected, not the assertion. This test does **not** close historical full-IP2Pix BF16 stochastic-continuation failures.

The repository has no production GRPO implementation. Its original DDPO stores old behavior scores to verify on-policy replay; it does not use them as PPO ratios. The isolated group reference implements required operations rather than renaming existing DDPO. Group rewards are normalized within the same context; zero-variance groups produce exact zeros; rewards, advantages and old log-probabilities are detached. Initial ratios are exactly one. Clipping is observed in training and also tested for both advantage signs. A separate clipping witness activates the norm cap, and KL minimization demonstrably moves toward the reference. Both ordinary REINFORCE and the group reference increase exactly enumerated expected reward over seeds 42/123/3407. They use different numbers of surrogate epochs, so their endpoint difference is not a compute-matched superiority result.

[Exact group-surrogate check](../results/group_reference_20261028_v3/checks.json) enumerates all 27 ordered three-action groups. Autodiff agrees with finite differences and Monte Carlo estimates (max z=1.046). The final clipping witness uses its declared 1e-10 tolerance against the exact clipped norm including PyTorch's 1e-6 denominator stabilizer; earlier reference receipts are retained. Its exact surrogate gradient is `[-.316157,-.003783,.319940]`, while the raw expected-reward gradient is `[-.177125,-.022902,.200027]`. **This difference is expected:** sample-relative standardization changes the surrogate. We do not call the normalized estimator an unbiased estimator of raw expected reward. The reference uses exact categorical KL, a tractable substitution for sampled KL in a one-action finite environment; it does not reproduce a full language-model GRPO trainer. Algorithm context: [DeepSeekMath](https://arxiv.org/html/2402.03300v3).

## Frozen models, rewards, harness, and prior evidence

- Target loading: `src/models/qwen_loader.py:QwenVQA`, pinned `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`; `.eval().requires_grad_(False)`, SDPA, greedy decoding, 32 new tokens for this stage, 4,096–262,144 image pixel budget, physical microbatch one. No target optimizer exists in the evaluation.
- Editor loading: `src/models/pretrained_editor_loader.py` and `src/editing/pretrained_editor.py:PretrainedEditor`, pinned `timbrooks/instruct-pix2pix@31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`, FP16 cached checkpoint variant, BF16 runtime, 256×256 output, ten steps, guidance 7.5 / image guidance 1.5, math SDPA. UNet/VAE/text encoder frozen during inference; no LoRA loaded in sanity evaluation.
- Existing evaluators: `src/rl/reward.py` combines semantic validity, answer consistency, distortion and compute cost with clipping [-1,1.5]. It is not mathematically equivalent to benign VQA accuracy. `reward_v2.py` exposes diagnostic compatibility arithmetic and `training_eligible=False`. Existing security evaluation remains separate. This validation uses only analytic synthetic reward or official benign VQA consensus, with **no attack-policy optimization and no cost optimization**.
- Official VQA scorer: `src/evaluation/vqa_official_compatible.py` and pinned vendored toolkit `src/evaluation/_vendor/official_vqa.py`. Complete raw answers and original ten annotators retained; no label-aware answer extraction. All 480 stored scores were independently recomputed by the report script and reconciled to CSV/JSONL/manifest.
- RSI: `src/rsi/controller.py` is validation-score/config revision with append-only `RevisionMemory` history and rollback records, not an autonomous neural meta-gradient. `src/rsi/harness/*` defines executable bounded checks (geometry/decode/content hash/independent review), proposals, versioning, costs and memory. `src/progression/real_harness.py` examines real stored cases. These modules are not renamed GRPO and were not used to tune the diagnostic evaluation set.
- Existing log/checkpoint inventory: 123 experiment directories and 72 `training_state.pt` files totaling 1,271,491,980 bytes on the canonical project at audit time. These are inventoried, not all revalidated. New model-stage data is in separate directories under `results/`. Historical `reports/RL_MATHEMATICAL_AUDIT.md`, `PHASE3_GENERATOR_RL_SMOKE.md`, and `PHASE4_RESTART_DIAGNOSTIC.md` retain their negative restart and reward-readiness findings.

[Local source inventory](../results/gradient_feedback_summary_20261028/inventory_local.json) and [canonical source inventory](../results/gradient_feedback_summary_20261028/inventory_cluster.json) contain 40 matching normalized-LF source hashes, environment versions, checkpoint paths and sizes. Initial executed source is retained under `source_versions/initial`; current source is under `source_snapshot`. No environment rebuild or new model download was necessary. Local Torch 2.8 CPU and canonical Torch 2.6 CUDA are explicitly separate witnesses.

## Acceptance boundaries

| Gate | Result |
|---|---|
| Exact-vs-MC synthetic gradients, convergence, finite gradients | PASS under frozen finite-task protocols; initial imprecision retained |
| Actual parameter updates, frozen invariance, bounded checkpoint checks | PASS for categorical head and tiny synthetic DDPO LoRA |
| Correct production algorithm identification | PASS: REINFORCE / DDPO-SF / supervised MSE / guarded backend; production GRPO absent |
| Complete traceable ten-batch feedback and repeated scores | PASS for record integrity and fixed-model reproducibility |
| Meaningful matched-candidate / GRPO batch feedback | NOT VALIDATED: different images/difficulty, saturation, weak batch separation |
| Frozen target VLM repeatability | PASS for 160 original images × three seeds; partial editor-log job failure explicitly reconciled |
| Editor file/seed/hash/inference correctness | PASS after serialization repair and editor-only rerun |
| Editor instruction/semantic preservation | FAIL / mixed: scene replacement, recoloring, darkening; ambiguous labels excluded |
| New untouched evaluation set and pretrained contamination audit | UNRESOLVED: development set reused; model pretraining exposure unknown |
| Full pretrained diffusion continuation and useful RL reward construct | UNRESOLVED; earlier negative evidence still applies |
| Large-scale VLM fine-tuning | DO NOT START |

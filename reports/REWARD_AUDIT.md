# Reward audit

**Reward validity: NO-GO for scientific improvement claims and further scaling.** Arithmetic is reproduced; construct validity remains insufficient. No coefficients changed.

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


The executed function is [src/rl/reward.py:14](H:/Project/Poisoning/src/rl/reward.py:14) with defaults(1,0.5,0.2,0.1,1); the actual caller is [scripts/smoke_generator_rl.py:34](H:/Project/Poisoning/scripts/smoke_generator_rl.py:34). Inputs S,A,D,C are each clamped to[0,1]. Let `valid = semantic.verified and S == 1`; U is0 if valid, otherwise1. Raw reward is S+0.5A−0.2D−0.1C−U; final reward is clipped to[−1,1.5]. With default coefficients, the unconstrained component-wise raw bound is[−1.3,1.5]. No prediction-dependent max normalization is used. `fixed_advantage` returns the bounded reward unchanged (baseline0, scale1).

| Component | Exact definition and range | Dependencies / uncertainty | Overlap and intended construct |
|---|---|---|---|
| S | Identity:1 and verified; invalid geometry/constructed negative:0 and verified; changed image:.5 only if structured checks and consistency both pass, otherwise0; all changed images unverified | [src/evaluation/semantic_validation.py:27](H:/Project/Poisoning/src/evaluation/semantic_validation.py:27); Qwen pair judgment, original/edited Qwen answers, original VQA references; malformed or uncertain judgments fail the gate | A conservative proxy, not independent semantic validity; duplicates the gated A for all500 saved observations |
| A | Raw consistency=`same_normalized_answer AND original_reference_score>0 AND edited_reference_score>0`; binary0/1. Caller further gates it on `provisionally_valid` | [src/evaluation/answer_consistency.py:4](H:/Project/Poisoning/src/evaluation/answer_consistency.py:4); official original references plus same Qwen; partial VQA consensus merely has to be positive | Not clean accuracy, not continuous edited consensus; verified identity gate defect in F12 |
| D | `clip((MAE+(1−SSIM)/2)/2,0,1)`; MAE uses RGB[0,1], SSIM data_range1; nominal D[0,1] | [src/evaluation/image_quality.py:4](H:/Project/Poisoning/src/evaluation/image_quality.py:4); installed windowed skimage SSIM, with explicitly named global fallback only if import unavailable; current receipt verifies skimage0.25.2 | Pixel/structure change, not perceptual quality or semantic validity; includes padding and penalizes intended brightening |
| C | `clip(num_inference_steps/30,0,1)`; current10/30 exactly1/3 | Caller above; independent of actual GPU seconds and reward-model overhead | Constant in this campaign; no ability to reward observed efficiency |
| U |0 only for verified S1; otherwise1 | `validation_reward`; uncertainty and invalidity are conflated | All500 U1, including proxy-pass images. Constant level penalty; induces floor censoring together with negative D/C |

For a changed edit let G be joint proxy agreement. Then S=.5G, A=G, U=1 and C=1/3:

`R = clip(G − .2D − 1/30 − 1, −1, 1.5)`.

G0 yields−1 for every D; G1 yields−1/30−.2D. S and A thus provide one binary signal, not independent semantic and correctness evidence. For G1, D can change reward by at most.2 across its entire allowed range; G flips produce a gap. A single flip contributes about.094 to a ten-image mean. Keeping an unverified image unchanged is not the same as producing a successful edit, and exact identity has privileged verification. All-negative rewards are **not** by themselves a policy-gradient sign error: an action-independent constant is a zero-expectation baseline in the ideal score estimator. The current clipped gate and finite-sample variance still require scrutiny.

## Recomputed distributions

All320 training and180 development observations were recomputed using saved PNGs, raw judgments and original answers; no new Qwen call. Train floor mass165/320=51.5625%; development floor mass65/180=36.1111%. All500 U1, all500 S=.5A, all500 human-unverified. Exact counts and per-arm min/max, means and raw/clipped values are in `phase45_recompute.json` → `saved_observation_recompute.statistics`.

| Arm/seed | Training proxy passes /80 | Mean raw reward | Mean clipped reward |
|---|---:|---:|---:|
| M1_42 | 45 | -0.505109402 | -0.473697641 |
| M1_123 | 32 | -0.667825709 | -0.626922154 |
| M2_42 | 46 | -0.492602724 | -0.462227997 |
| M2_123 | 32 | -0.667842608 | -0.627171706 |

U dominates absolute score level and the censoring regime; observed ranking is driven primarily by the proxy gate. It is inaccurate to say differences are caused by a varying U because U never varies. Semantic false negatives, wrong-but-consistent answers and model prompt instability cannot be disentangled from real preservation without independent labels.

## F02 — HIGH: Reward provider and semantic verifier are the same model

Evidence status: **VERIFIED**. Source/function: [scripts/smoke_generator_rl.py:34](H:/Project/Poisoning/scripts/smoke_generator_rl.py:34); [src/evaluation/semantic_validation.py:49](H:/Project/Poisoning/src/evaluation/semantic_validation.py:49); [src/models/qwen_loader.py:7](H:/Project/Poisoning/src/models/qwen_loader.py:7).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`.

Observed: Frozen Qwen supplies original answers, edited answers and pair judgments. All 500 observations remain independently unverified. Agreement between its two prompt formats is not independent verification; official original-image answers do not certify an edited image.

Proposed correction: Obtain blinded independent semantic labels and assess verifier errors on all relevant conditions. Keep proxy metrics and independent validity separate; do not tune against a purported independent evaluator that also supplies reward.

## F03 — HIGH: The reward collapses to one proxy gate with extensive floor censoring

Evidence status: **VERIFIED**. Source/function: [src/rl/reward.py:14](H:/Project/Poisoning/src/rl/reward.py:14); [src/evaluation/semantic_validation.py:27](H:/Project/Poisoning/src/evaluation/semantic_validation.py:27); [scripts/smoke_generator_rl.py:49](H:/Project/Poisoning/scripts/smoke_generator_rl.py:49).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Recomputed 500/500 observations have U=1 and S=0.5*A_reward. For changed edits with gate G, R=clip(G-0.2D-1/30-1,-1,1.5). G=0 is always -1, discarding D and C differences. 165/320 training and 65/180 development observations are at -1. All observed rewards are negative. U dominates the level and clipping; proxy-gate flips dominate between-arm changes. U itself is constant and does not rank changed samples.

Proposed correction: Review construct definitions, uncertainty treatment, redundant terms and clipping analytically under a new versioned protocol. Preserve current coefficients and all raw/clipped values. No coefficient search or automated reward repair in this phase.

## F04 — HIGH: Instruction success is missing from the objective

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_policy.py:9](H:/Project/Poisoning/src/rl/diffusion_policy.py:9); [src/evaluation/image_quality.py:4](H:/Project/Poisoning/src/evaluation/image_quality.py:4); [src/rl/reward.py:22](H:/Project/Poisoning/src/rl/reward.py:22).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: There is no explicit measure that lighting was slightly brightened. When the proxy gate passes (G=1), reducing D increases reward. When G=0, clipping makes reward flat at -1 regardless of D. Identity has S=1,U=0 and is strongly preferred. Thus preservation can improve by approaching no edit; reward improvement would not demonstrate instruction following.

Proposed correction: Before further training, define independently assessable task success alongside preservation, and document how no-op controls are interpreted. Do not silently add a brightness objective now.

## F10 — MEDIUM: Question-relative semantic checking is narrower than the global preservation instruction

Evidence status: **VERIFIED**. Source/function: [src/evaluation/semantic_validation.py:6](H:/Project/Poisoning/src/evaluation/semantic_validation.py:6); [src/evaluation/semantic_validation.py:51](H:/Project/Poisoning/src/evaluation/semantic_validation.py:51); [src/rl/diffusion_policy.py:9](H:/Project/Poisoning/src/rl/diffusion_policy.py:9).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`.

Observed: The verifier asks only about question-relevant properties and explicitly allows irrelevant properties to be true. The instruction asks to retain all objects, colors, counts and positions. Unchanged answers cannot validate all image semantics, nor guarantee the original labels remain valid after a question-relevant change.

Proposed correction: Specify separate question-answer validity, global semantic preservation and instruction-following rubrics for a future protocol; keep current labels diagnostic until verified.

## F11 — MEDIUM: C is a fixed denoising-step proxy, not measured compute cost

Evidence status: **VERIFIED**. Source/function: [scripts/smoke_generator_rl.py:49](H:/Project/Poisoning/scripts/smoke_generator_rl.py:49); [src/rl/reward.py:10](H:/Project/Poisoning/src/rl/reward.py:10).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: C=clip(10/30,0,1)=1/3 for every Phase4 trajectory, so -0.1C=-1/30. It excludes Qwen calls, model loading, diagnostics, validation and RSI bookkeeping. It cannot demonstrate compute savings. Slurm separately records actual total cost.

Proposed correction: Call C a normalized step-count proxy. Retain separate end-to-end Slurm accounting and define any future cost objective before comparing efficiency.

## F12 — MEDIUM: Verified identity loses its answer-consistency credit at the caller

Evidence status: **VERIFIED**. Source/function: [src/evaluation/semantic_validation.py:37](H:/Project/Poisoning/src/evaluation/semantic_validation.py:37); [scripts/smoke_generator_rl.py:49](H:/Project/Poisoning/scripts/smoke_generator_rl.py:49).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The verified-identity branch omits provisionally_valid; collect gates A solely on that key. With raw consistency1, identity is passed A=0 and R=0.9666667 (C=1/3), rather than 1.4666667 if the declared consistency component were credited. This is a definite branch inconsistency, not an observed cause of Phase4 results: all 500 current edits were nonidentity/unverified.

Proposed correction: Under a future versioned implementation correction, explicitly define eligibility for verified-valid and provisional cases and add an identity branch test. Do not rewrite any historical reward.

## F19 — MEDIUM: Reward configuration metadata is not the executed coefficient source

Evidence status: **VERIFIED**. Source/function: [configs/phase3_generator_rl.yaml:24](H:/Project/Poisoning/configs/phase3_generator_rl.yaml:24); [scripts/smoke_generator_rl.py:48](H:/Project/Poisoning/scripts/smoke_generator_rl.py:48); [src/rl/reward.py:15](H:/Project/Poisoning/src/rl/reward.py:15).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: collect calls validation_reward without a weights object, so dataclass defaults, not the YAML reward block, determine executed coefficients. They numerically match this campaign. Several YAML names also differ from dataclass field names. Editing YAML alone would silently fail to change the actual reward.

Proposed correction: Before future protocol changes, choose one explicit validated coefficient source and assert it equals recorded metadata. Current calculations are correct for the defaults; no coefficient or scientific source is changed in this audit.

Training uses train records; development images are source-disjoint from them. The same frozen verifier is reused across splits, and development is repeatedly used for adaptation. That is reward/evaluator dependence and adaptive evaluation, not evidence that validation images were directly included in optimizer batches. ASR and evasion do not enter reward or coefficient selection.

# Methodology and generator audit

**Decision for further scientific training: NO-GO.** Core wiring is supported, but the research objective, independent semantic evidence and causal RSI comparison are not ready.

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


## Research formulation and literature

[BadSem (2025), Sections 3–4](https://arxiv.org/html/2506.07214v1) studies training-time image/text semantic mismatches as implicit conditions for later malicious model behavior, focusing on color and object semantics. Its defining construct is cross-modal inconsistency. A changed image or a changed answer alone does not establish that construct. The present pilot trains an editor against preservation proxies and leaves Qwen frozen. It does not reproduce that threat model or demonstrate post-training persistence.

| Condition | Precise meaning | Present evidence |
|---|---|---|
| Image-only semantic manipulation | Change an object's identity, color, count, relation or other content in an image | Some saved edits have suspected changes; none independently labeled |
| Image-question semantic mismatch | The paired text asserts or presupposes something inconsistent with the image | Not systematically constructed/labeled here; ordinary questions need not assert an attribute |
| Semantic-preserving edit | An intentional image change that preserves a declared global or question-specific semantic contract | The optimization target; current validity is a same-Qwen proxy |
| Ordinary augmentation | A transformation such as brightness adjustment, independent of a backdoor target | Earlier PIL/controller smokes; not evidence that a diffusion edit preserves labels |
| Training-time data modification | Alter training examples for a learner | Generated trajectories train editor LoRA; they are not used to fine-tune Qwen |
| Inference-time model behavior | A fixed Qwen answer on original versus edited input | Measured diagnostic behavior; distinct from learned backdoor behavior |
| Post-training backdoor evaluation | Assess fixed victim artifacts for condition-dependent behavior after a documented training lineage | Real measurement absent; only explicitly synthetic aggregation fixtures |

M0 is zero-B frozen LoRA on the **same modified stochastic DDIM path** as trained arms; it is not necessarily stock pipeline-default inference. M1 is eight single-pass on-policy updates at 1e-5. M2 shares the exposure/noise protocol and makes one validation-triggered cooldown after update4. Earlier categorical-editor/controller RL and supervised LoRA smokes are separate methods and cannot be pooled as generator RL results.

## Generator source and evidence

`DiffusionPolicy.condition/predict/sample` in [src/rl/diffusion_policy.py:25](H:/Project/Poisoning/src/rl/diffusion_policy.py:25) uses full-image aspect-preserving resize and white pad to512, VAE mode **without source-latent scaling**, instruction embeddings in three CFG branches and an8-channel UNet input. CFG is unconditional +7.5(text-image)+1.5(image-unconditional). The VAE-decoded terminal latent is divided by the VAE scaling factor. This source-conditioning convention matches the [pinned Diffusers 0.35.1 IP2Pix implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py).

The policy changes the base scheduler to stochastic DDIM epsilon prediction, eta1, 10 leading timesteps901,...,1, offset1, clipping off and set_alpha_to_one=false. This avoids a zero-variance final transition. It is an explicit generation-protocol change, shared by M0/M1/M2, not evidence of equivalence with the pretrained editor's default scheduler. CPU algebra matches installed DDIM step exactly on all ten synthetic transitions.

LoRA attaches rank8/alpha16 adapters to `to_q`, `to_k`, `to_v`, `to_out.0`: 1,594,368 trainable parameters, FP32 adapter parameters/operators, BF16 frozen base. Saved before/after hashes cover1130 frozen named parameters; all agree. Conditioning controls show source-image and instruction influence and exact explicit-seed trajectory replay in the recorded control. These are wiring checks, not instruction-success measurements. No fresh model was loaded for this audit.

## Saved-image review boundary

Inspected the existing ten-row Phase3 contact sheet and Phase4 seed123 final development originals/edits for sources4 (skating) and6 (bananas), plus the exact duplicate source image. The displayed edits show warm/darker shifts and altered padding; fine contour changes are suspected. No count, object-identity or quality verdict is promoted to independent ground truth. See F14 and F18. The portable human-review package remains blank.

## F01 — CRITICAL: The implemented estimand is benign preservation, not BadSem reproduction

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_policy.py:9](H:/Project/Poisoning/src/rl/diffusion_policy.py:9); [scripts/smoke_generator_rl.py:34](H:/Project/Poisoning/scripts/smoke_generator_rl.py:34); [docs/METHOD.md:14](H:/Project/Poisoning/docs/METHOD.md:14).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The policy receives an image and one fixed lighting instruction. The question is used by the reward, not as policy text conditioning. Qwen is frozen. No victim fine-tuning or real post-training backdoor evaluation occurs. Semantic changes that break proxy preservation reduce reward; therefore this objective opposes introducing question-relevant semantic inconsistencies. This is a mismatch if results are presented as completing the broader BadSem objective, rather than a bug in the declared benign pilot.

Proposed correction: Freeze and name the benign preservation estimand explicitly; separately specify any future research question and label-validity contract before training. No reward changes or poisoning implementation are authorized by this audit.

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

## F14 — MEDIUM: Saved images show suspected instruction-following and padding failures

Evidence status: **VERIFIED**. Source/function: [src/editing/pretrained_editor.py:43](H:/Project/Poisoning/src/editing/pretrained_editor.py:43); [src/evaluation/image_quality.py:4](H:/Project/Poisoning/src/evaluation/image_quality.py:4).

Supporting experiment IDs: `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Direct executor inspection of existing Phase3 contact sheet and Phase4 seed123 eval8 sources4/6 shows large warm/darker shifts, altered white padding and suspected shape/detail changes. Banana source6 and skating source4 are visibly not a simple uniform slight brightening. This is a selected qualitative review, not blinded human ground truth or a failure-rate estimate. Pixel scores include padding, which can affect D independently of semantic content.

Proposed correction: Keep these as suspected failures and obtain blinded content/padding/instruction labels. Retain full-frame measurements; any content-only metric must be added as a separately documented derived metric, not replace old D.

## F16 — LOW: The top-level method document describes an obsolete stage

Evidence status: **VERIFIED**. Source/function: [docs/METHOD.md:7](H:/Project/Poisoning/docs/METHOD.md:7); [docs/METHOD.md:18](H:/Project/Poisoning/docs/METHOD.md:18).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`.

Observed: docs/METHOD.md still describes only a PIL brightness controller and preliminary normalization, while Phase3/4 use real UNet LoRA DDPO and the official-compatible scorer. Historical Phase4 reports are more accurate. This can confuse a new reader about which implementation generated the results.

Proposed correction: Add a versioned current-method entry point linking this audit and retain historical descriptions with explicit stage labels. No historical report has been edited during this audit.

## F18 — UNRESOLVED: Independent semantic validity and visual improvement remain unknown

Evidence status: **UNRESOLVED**. Source/function: [reports/PHASE4_SEMANTIC_REVIEW.md:1](H:/Project/Poisoning/reports/PHASE4_SEMANTIC_REVIEW.md:1); [experiments/EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW/review_form.csv:1](H:/Project/Poisoning/experiments/EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW/review_form.csv:1).

Supporting experiment IDs: `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The ten-pair portable review form is blank; all new training/development edits are also unverified. Inventory wording flags are not semantic error labels. No independent human/model agreement rate, valid-edit rate, or quality improvement can be computed.

Proposed correction: Complete a blinded review with explicit uncertainty and disagreement adjudication before scientific quality claims. Do not manufacture labels from the executor observations in this audit.

## F22 — HIGH: Real controlled post-training security assessment is not ready

Evidence status: **VERIFIED**. Source/function: [src/evaluation/security_metrics.py:9](H:/Project/Poisoning/src/evaluation/security_metrics.py:9); [experiments/EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval/metrics.json:3](H:/Project/Poisoning/experiments/EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval/metrics.json:3).

Supporting experiment IDs: `EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval`, `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Existing security rows are labeled synthetic fixture observations. Post-training behavior-change denominator is zero and its rate is null. Qwen remains frozen in this campaign. There is no evaluated victim checkpoint lineage or independently verified real post-training fixture. Inference answer changes on edited images are not backdoor ASR.

Proposed correction: Keep real security readiness NO-GO. First close task/label/evaluator/provenance gates and define a separately reviewed controlled assessment of fixed artifacts; never connect ASR or evasion feedback to automated optimization.


## Experiment map used throughout the audit

- **B**: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`
- **R42**: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`
- **R123**: `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`
- **S42**: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`
- **S123**: `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`
- **D**: `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`
- **DA**: `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`
- **O**: `EXP_20261027T175526_5aee05413ae5_phase4_PH4_RESTART_OUTCOMES`
- **H**: `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`
- **P3**: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`
- **T**: `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`
- **SEC**: `EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval`

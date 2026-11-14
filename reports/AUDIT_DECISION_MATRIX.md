# Audit decision matrix

**Overall further-scientific-training decision: NO-GO.** This is an audit recommendation, not a permission request or an automatic change to jobs. No scientific training was launched.

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


| Readiness dimension | Decision | Evidence and unmet gate |
|---|---|---|
|1 Engineering readiness |CONDITIONAL GO |Model/data provenance, real updates, logging, budget, monitoring and59 CPU tests supported. Exact continuation, runtime guards and active-contract reconstruction remain limited. CPU audits/reporting can continue; resume or new training is not cleared. |
|2 Generator RL correctness |CONDITIONAL GO |Fixed-contract score-function/DDIM algebra, actual LoRA gradients, B10/T10 and single optimizer pass supported. Real-device backward repeatability and continuation unresolved. This is not statistical or semantic correctness. |
|3 Reward validity |NO-GO |Same-model verification, S/A redundancy, constant uncertainty penalty plus censoring, no instruction-success measure, latent identity branch inconsistency. |
|4 RSI methodology |NO-GO |One LR cooldown; no demonstrated distinction from scheduling; major differences precede intervention; final intervals include zero. |
|5 Evaluation readiness |NO-GO |No independent semantic labels; ten adaptively reused sources/two seeds; ID-disjointness misses exact train/test duplicate. |
|6 Controlled post-training security assessment |NO-GO |No real fixed-victim lineage/independent real assessment; synthetic aggregator fixtures only; upstream task/semantic/evaluation gates unresolved. |

GO would mean all relevant gates are satisfied for the proposed scope; CONDITIONAL GO records supported components with explicit unresolved conditions. It does **not** override this phase's prohibition on new scientific training. No dimension grants approval for a poisoning campaign, automated ASR optimization or evasion tuning.

## All adverse and unresolved findings

| ID | Classification | Evidence status | Finding |
|---|---|---|---|
| F01 | CRITICAL | VERIFIED | The implemented estimand is benign preservation, not BadSem reproduction |
| F02 | HIGH | VERIFIED | Reward provider and semantic verifier are the same model |
| F03 | HIGH | VERIFIED | The reward collapses to one proxy gate with extensive floor censoring |
| F04 | HIGH | VERIFIED | Instruction success is missing from the objective |
| F05 | HIGH | VERIFIED | Executed RSI is a validation-driven LR schedule extension |
| F06 | CRITICAL | VERIFIED | RSI causal benefit is confounded by divergence before intervention |
| F07 | HIGH | VERIFIED | Two seeds and ten repeatedly reused development images cannot establish generalization |
| F08 | HIGH | VERIFIED | ID-disjoint splits contain an exact train/test image duplicate |
| F09 | HIGH | VERIFIED | Exact stochastic checkpoint continuation remains unaccepted |
| F10 | MEDIUM | VERIFIED | Question-relative semantic checking is narrower than the global preservation instruction |
| F11 | MEDIUM | VERIFIED | C is a fixed denoising-step proxy, not measured compute cost |
| F12 | MEDIUM | VERIFIED | Verified identity loses its answer-consistency credit at the caller |
| F13 | MEDIUM | VERIFIED | Exposure matching does not establish equal realized compute |
| F14 | MEDIUM | VERIFIED | Saved images show suspected instruction-following and padding failures |
| F15 | MEDIUM | VERIFIED | Checkpoint configuration revisions require reconstructing the exact active contract |
| F16 | LOW | VERIFIED | The top-level method document describes an obsolete stage |
| F17 | UNRESOLVED | UNRESOLVED | The specific backward numerical divergence mechanism is not localized |
| F18 | UNRESOLVED | UNRESOLVED | Independent semantic validity and visual improvement remain unknown |
| F19 | MEDIUM | VERIFIED | Reward configuration metadata is not the executed coefficient source |
| F20 | MEDIUM | VERIFIED | Runtime dataset guards are weaker than provenance audit guards |
| F21 | UNRESOLVED | UNRESOLVED | Near-duplicate and pretraining exposure are outside the present verification |
| F22 | HIGH | VERIFIED | Real controlled post-training security assessment is not ready |
| F23 | LOW | VERIFIED | Generalized freshness checks have an unexercised duplicate-within-batch gap |

## F01 — CRITICAL: The implemented estimand is benign preservation, not BadSem reproduction

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_policy.py:9](H:/Project/Poisoning/src/rl/diffusion_policy.py:9); [scripts/smoke_generator_rl.py:34](H:/Project/Poisoning/scripts/smoke_generator_rl.py:34); [docs/METHOD.md:14](H:/Project/Poisoning/docs/METHOD.md:14).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The policy receives an image and one fixed lighting instruction. The question is used by the reward, not as policy text conditioning. Qwen is frozen. No victim fine-tuning or real post-training backdoor evaluation occurs. Semantic changes that break proxy preservation reduce reward; therefore this objective opposes introducing question-relevant semantic inconsistencies. This is a mismatch if results are presented as completing the broader BadSem objective, rather than a bug in the declared benign pilot.

Proposed correction: Freeze and name the benign preservation estimand explicitly; separately specify any future research question and label-validity contract before training. No reward changes or poisoning implementation are authorized by this audit.

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

## F05 — HIGH: Executed RSI is a validation-driven LR schedule extension

Evidence status: **VERIFIED**. Source/function: [src/rsi/phase4_schedule.py:13](H:/Project/Poisoning/src/rsi/phase4_schedule.py:13); [src/rsi/controller.py:21](H:/Project/Poisoning/src/rsi/controller.py:21); [scripts/run_phase4_rsi.py:112](H:/Project/Poisoning/scripts/run_phase4_rsi.py:112).

Supporting experiment IDs: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Both arms make one LR change 1e-5 to 5e-6 after update4. The extension records accepted_schedule but explicitly quality_improvement_accepted=false. It bypasses the legacy candidate-gain acceptance/rollback semantics. Persistent revision history is implemented, but ordinary adaptive scheduling and a novel recursive improvement mechanism are not experimentally distinguished.

Proposed correction: Describe this result as an exploratory validation-driven cooldown. Define the proposed RSI mechanism and compare with a predeclared ordinary scheduler/control before a novelty claim. Do not relabel historical acceptance as quality acceptance.

## F06 — CRITICAL: RSI causal benefit is confounded by divergence before intervention

Evidence status: **VERIFIED**. Source/function: [scripts/run_phase4_rsi.py:81](H:/Project/Poisoning/scripts/run_phase4_rsi.py:81); [scripts/build_phase4_reports.py:81](H:/Project/Poisoning/scripts/build_phase4_reports.py:81).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Initial adapters, data order and noise seeds match. Before either LR change, step4 adapters already differ by max 4.34328e-5 (42) and 6.64559e-5 (123). Seed123 step4 M2-M1 reward is +0.0932581, comparable to final +0.0937077. Both final paired intervals include zero. Each final +~0.094 is driven by one gate flip. Matching seeds therefore does not isolate the intervention.

Proposed correction: Withdraw causal RSI-gain claims. A future authorized design must establish comparable pre-intervention behavior and separate numerical execution variability from adaptation effects; preserve all pre-intervention evidence and negative outcomes.

## F07 — HIGH: Two seeds and ten repeatedly reused development images cannot establish generalization

Evidence status: **VERIFIED**. Source/function: [scripts/run_phase4_budgeted.py:63](H:/Project/Poisoning/scripts/run_phase4_budgeted.py:63); [src/evaluation/phase4_statistics.py:4](H:/Project/Poisoning/src/evaluation/phase4_statistics.py:4); [src/evaluation/phase4_statistics.py:15](H:/Project/Poisoning/src/evaluation/phase4_statistics.py:15).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: There are 180 validation trajectories but only 10 unique development images. Four looks per trained arm reuse them; M2 uses these same looks to revise LR. Percentile source bootstrap CIs are conditional on a fitted seed, without repeated-selection correction. Two seeds mix training and evaluation-noise variation. Treating 180 trajectories as independent images would be invalid.

Proposed correction: Keep these estimates exploratory. Predeclare a separate selection/evaluation design and source-clustered analysis before any new campaign; do not reopen the old held-out subset to rescue this comparison.

## F08 — HIGH: ID-disjoint splits contain an exact train/test image duplicate

Evidence status: **VERIFIED**. Source/function: [src/data/dataset.py:49](H:/Project/Poisoning/src/data/dataset.py:49); [src/data/prepare_vqa.py:69](H:/Project/Poisoning/src/data/prepare_vqa.py:69); [data/manifests/cluster/vqav2_seed42_v1/train.json:13619](H:/Project/Poisoning/data/manifests/cluster/vqav2_seed42_v1/train.json:13619); [data/manifests/cluster/vqav2_seed42_v1/test.json:4810](H:/Project/Poisoning/data/manifests/cluster/vqav2_seed42_v1/test.json:4810).

Supporting experiment IDs: `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Canonical image hashing verifies train image397819/sample397819015 and test image388662/sample388662037 share SHA256 fcc1fd100ed50e30ea87b9d84597cb40876c76d955716fc9dd6e936787faf181. The questions differ. All split image IDs are disjoint, but bytes are not. No exact-byte train/validation overlap exists. Image397819 was not used in any of the four Phase4 training arms; no current Phase4 train/dev contamination is established. Frozen Qwen clean-test evaluation did not train on this internal train split, so this is not proof its historical accuracy was inflated.

Proposed correction: For a future dataset version, group exact duplicates before assigning splits and audit near duplicates separately. Preserve v1 and the already-opened test marker; restrict v1 independence claims. Do not silently delete a test record or recompute a favorable score.

## F09 — HIGH: Exact stochastic checkpoint continuation remains unaccepted

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_rl_trainer.py:78](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:78); [scripts/diagnose_phase4_restart.py:34](H:/Project/Poisoning/scripts/diagnose_phase4_restart.py:34); [scripts/audit_phase4_restart.py:24](H:/Project/Poisoning/scripts/audit_phase4_restart.py:24); [scripts/phase4_restart_outcomes.py:1](H:/Project/Poisoning/scripts/phase4_restart_outcomes.py:1).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`, `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`, `EXP_20261027T175526_5aee05413ae5_phase4_PH4_RESTART_OUTCOMES`.

Observed: Fixed-trajectory replay has exact checked restored fields and exact behavior likelihoods, but backward gradients and post-update adapters/optimizer states differ. Adapter max differences are 2.87986e-6 and 2.86177e-6. Two outcome sources produce unequal RGB images (max channel differences 103/37/93/40). The fixed-trajectory replay bypasses ten rollout seed draws; it does not establish full stochastic RNG continuation.

Proposed correction: Keep exact equality as the acceptance criterion. Permit no resume-equivalence claim until a separately authorized diagnostic closes it. No new numerical fix or threshold relaxation is made here.

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

## F13 — MEDIUM: Exposure matching does not establish equal realized compute

Evidence status: **VERIFIED**. Source/function: [scripts/submit_phase4.py:1](H:/Project/Poisoning/scripts/submit_phase4.py:1); [reports/phase4_sacct_final.log:1](H:/Project/Poisoning/reports/phase4_sacct_final.log:1); [reports/phase4_results.json:2787](H:/Project/Poisoning/reports/phase4_results.json:2787).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Each arm has 8 updates and 80 training exposures. M1/M2 elapsed GPU seconds are 3227/3211 for42 and 2707/3216 for123; nodes differ. No repeated same-node latency comparison was done. The entire campaign uses 13122 seconds, within 14400, including failures/diagnostics and evaluation.

Proposed correction: Report matched exposures plus actual cost, rather than strict compute equivalence or generation speedup. A future efficiency study needs a predeclared comparable hardware/timing protocol.

## F14 — MEDIUM: Saved images show suspected instruction-following and padding failures

Evidence status: **VERIFIED**. Source/function: [src/editing/pretrained_editor.py:43](H:/Project/Poisoning/src/editing/pretrained_editor.py:43); [src/evaluation/image_quality.py:4](H:/Project/Poisoning/src/evaluation/image_quality.py:4).

Supporting experiment IDs: `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Direct executor inspection of existing Phase3 contact sheet and Phase4 seed123 eval8 sources4/6 shows large warm/darker shifts, altered white padding and suspected shape/detail changes. Banana source6 and skating source4 are visibly not a simple uniform slight brightening. This is a selected qualitative review, not blinded human ground truth or a failure-rate estimate. Pixel scores include padding, which can affect D independently of semantic content.

Proposed correction: Keep these as suspected failures and obtain blinded content/padding/instruction labels. Retain full-frame measurements; any content-only metric must be added as a separately documented derived metric, not replace old D.

## F15 — MEDIUM: Checkpoint configuration revisions require reconstructing the exact active contract

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_rl_trainer.py:80](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:80); [scripts/run_phase4_rsi.py:128](H:/Project/Poisoning/scripts/run_phase4_rsi.py:128); [scripts/run_phase4_rsi.py:129](H:/Project/Poisoning/scripts/run_phase4_rsi.py:129).

Supporting experiment IDs: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: LR revision updates optimizer LR, scheduler base_lrs and policy contract and creates a child revision record. The parent manifest/config remains the initial version; subsequent checkpoints carry the active revision. Reconstructing a parent-only default policy would fail the contract check. No production resume across this revision was accepted or used.

Proposed correction: Document the active revision lineage and reconstruction procedure; retain immutable parent and child records. Validate it before any future authorized resume; do not treat an intentional contract rejection as successful continuation.

## F16 — LOW: The top-level method document describes an obsolete stage

Evidence status: **VERIFIED**. Source/function: [docs/METHOD.md:7](H:/Project/Poisoning/docs/METHOD.md:7); [docs/METHOD.md:18](H:/Project/Poisoning/docs/METHOD.md:18).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`.

Observed: docs/METHOD.md still describes only a PIL brightness controller and preliminary normalization, while Phase3/4 use real UNet LoRA DDPO and the official-compatible scorer. Historical Phase4 reports are more accurate. This can confuse a new reader about which implementation generated the results.

Proposed correction: Add a versioned current-method entry point linking this audit and retain historical descriptions with explicit stage labels. No historical report has been edited during this audit.

## F17 — UNRESOLVED: The specific backward numerical divergence mechanism is not localized

Evidence status: **UNRESOLVED**. Source/function: [scripts/diagnose_phase4_restart.py:47](H:/Project/Poisoning/scripts/diagnose_phase4_restart.py:47); [scripts/audit_phase4_restart.py:29](H:/Project/Poisoning/scripts/audit_phase4_restart.py:29).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`, `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`.

Observed: The earliest inspected difference is in backward gradients after exact forward log-likelihoods. That does not identify the first differing CUDA primitive, exclude hidden execution-state differences, or prove an unavoidable universal nondeterminism mechanism.

Proposed correction: Retain competing hypotheses and exact diagnostics. Only separately authorized controlled diagnostics may identify a root cause; do not invent a fix or declare all GPU determinism impossible.

## F18 — UNRESOLVED: Independent semantic validity and visual improvement remain unknown

Evidence status: **UNRESOLVED**. Source/function: [reports/PHASE4_SEMANTIC_REVIEW.md:1](H:/Project/Poisoning/reports/PHASE4_SEMANTIC_REVIEW.md:1); [experiments/EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW/review_form.csv:1](H:/Project/Poisoning/experiments/EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW/review_form.csv:1).

Supporting experiment IDs: `EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The ten-pair portable review form is blank; all new training/development edits are also unverified. Inventory wording flags are not semantic error labels. No independent human/model agreement rate, valid-edit rate, or quality improvement can be computed.

Proposed correction: Complete a blinded review with explicit uncertainty and disagreement adjudication before scientific quality claims. Do not manufacture labels from the executor observations in this audit.

## F19 — MEDIUM: Reward configuration metadata is not the executed coefficient source

Evidence status: **VERIFIED**. Source/function: [configs/phase3_generator_rl.yaml:24](H:/Project/Poisoning/configs/phase3_generator_rl.yaml:24); [scripts/smoke_generator_rl.py:48](H:/Project/Poisoning/scripts/smoke_generator_rl.py:48); [src/rl/reward.py:15](H:/Project/Poisoning/src/rl/reward.py:15).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: collect calls validation_reward without a weights object, so dataclass defaults, not the YAML reward block, determine executed coefficients. They numerically match this campaign. Several YAML names also differ from dataclass field names. Editing YAML alone would silently fail to change the actual reward.

Proposed correction: Before future protocol changes, choose one explicit validated coefficient source and assert it equals recorded metadata. Current calculations are correct for the defaults; no coefficient or scientific source is changed in this audit.

## F20 — MEDIUM: Runtime dataset guards are weaker than provenance audit guards

Evidence status: **VERIFIED**. Source/function: [src/data/dataset.py:15](H:/Project/Poisoning/src/data/dataset.py:15); [scripts/run_phase4_budgeted.py:60](H:/Project/Poisoning/scripts/run_phase4_budgeted.py:60).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The loader checks presence, basic schema and sample-ID uniqueness; it does not check each image SHA or the original question/annotation image join on every read. Training validates manifest SHA and train/dev image-ID disjointness. The new CPU audit closes these checks for current files; it does not turn the loader into a general corruption/duplicate detector.

Proposed correction: Use audited immutable dataset versions and record verified input hashes; add explicit source-join/image-hash/duplicate guards for future versions without changing v1.

## F21 — UNRESOLVED: Near-duplicate and pretraining exposure are outside the present verification

Evidence status: **UNRESOLVED**. Source/function: [src/data/prepare_vqa.py:69](H:/Project/Poisoning/src/data/prepare_vqa.py:69); [reports/phase45_recompute.json:33](H:/Project/Poisoning/reports/phase45_recompute.json:33).

Supporting experiment IDs: `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Exact file hashing detects one train/test duplicate and no train/dev duplicate. It cannot exclude re-encoded/cropped near duplicates or COCO/VQAv2 exposure during foundation-model pretraining. Internal image-ID disjointness is not evidence of foundation-model unseen data.

Proposed correction: Qualify the internal held-out claim; separately audit near duplicates and available training provenance if needed. Do not claim either zero pretraining exposure or definite memorization without evidence.

## F22 — HIGH: Real controlled post-training security assessment is not ready

Evidence status: **VERIFIED**. Source/function: [src/evaluation/security_metrics.py:9](H:/Project/Poisoning/src/evaluation/security_metrics.py:9); [experiments/EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval/metrics.json:3](H:/Project/Poisoning/experiments/EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval/metrics.json:3).

Supporting experiment IDs: `EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval`, `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Existing security rows are labeled synthetic fixture observations. Post-training behavior-change denominator is zero and its rate is null. Qwen remains frozen in this campaign. There is no evaluated victim checkpoint lineage or independently verified real post-training fixture. Inference answer changes on edited images are not backdoor ASR.

Proposed correction: Keep real security readiness NO-GO. First close task/label/evaluator/provenance gates and define a separately reviewed controlled assessment of fixed artifacts; never connect ASR or evasion feedback to automated optimization.

## F23 — LOW: Generalized freshness checks have an unexercised duplicate-within-batch gap

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_trajectory.py:45](H:/Project/Poisoning/src/rl/diffusion_trajectory.py:45); [src/rl/diffusion_rl_trainer.py:31](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:31); [src/rl/diffusion_rl_trainer.py:57](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:57).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: The trainer checks against previously consumed IDs before the batch, then adds all IDs only after the update. It does not independently reject two copies of the same fresh trajectory inside one input list. Actual collectors generate unique IDs and distinct sources, and all 32 recorded batches were verified distinct; no current duplicate update is found.

Proposed correction: Add an explicit per-batch trajectory/source uniqueness contract before generalizing this trainer. Do not infer contamination of current recorded batches.

## VERIFIED supported components

### V01 — VERIFIED: Dataset correspondence and deterministic replay

Source/function: [src/data/prepare_vqa.py:69](H:/Project/Poisoning/src/data/prepare_vqa.py:69). Supporting experiment: `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: All1500 selected original question/annotation/image joins, answer arrays, file hashes and sampling replay match. Split independence is qualified by F08.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V02 — VERIFIED: Stochastic DDIM/score algebra

Source/function: [src/rl/diffusion_trajectory.py:10](H:/Project/Poisoning/src/rl/diffusion_trajectory.py:10). Supporting experiment: `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: Ten CPU transitions match installed DDIM step with max error0; Gaussian score and epsilon gradient match torch.distributions.Normal; all variances positive. This is CPU algebra only.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V03 — VERIFIED: Actual generator RL and batch accounting

Source/function: [src/rl/diffusion_rl_trainer.py:27](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:27). Supporting experiment: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: 32 actual updates,10 source exposures and100 denoising backward transitions each; finite/nonzero gradients and adapter changes; no categorical-head substitution in Phase4.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V04 — VERIFIED: Preserved model revisions/frozen tensors

Source/function: [src/training/generator_adapter.py:22](H:/Project/Poisoning/src/training/generator_adapter.py:22). Supporting experiment: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: Pinned model revisions; 1130 named frozen parameters unchanged per arm; 1594368 intended LoRA parameters; does not imply useful editing.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V05 — VERIFIED: Saved metric and confidence-interval arithmetic

Source/function: [scripts/audit_phase45.py:110](H:/Project/Poisoning/scripts/audit_phase45.py:110). Supporting experiment: `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: 500 observations/trajectories,5000 transitions, reward/quality/consistency fields and paired CIs reproduced without model calls.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V06 — VERIFIED: Existing two-hour monitor operational at inspection

Source/function: [src/monitoring/scheduler.py:1](H:/Project/Poisoning/src/monitoring/scheduler.py:1). Supporting experiment: `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: One owned pending native CPU successor; enabled and failures0; no GPU research active. Snapshot records exact ID/times.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V07 — VERIFIED: Historical evidence and budget preserved

Source/function: [scripts/audit_phase45.py:40](H:/Project/Poisoning/scripts/audit_phase45.py:40). Supporting experiment: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: 1872 original Phase4 files checked against receipt;2130 historical files unchanged before/after; historical failed run remains failed; actual prior cost13122/14400GPU seconds; audit adds0.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

### V08 — VERIFIED: Security aggregation is separated from optimization

Source/function: [src/evaluation/security_metrics.py:9](H:/Project/Poisoning/src/evaluation/security_metrics.py:9). Supporting experiment: `EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.

Observed: No optimizer imports; synthetic fixture labeling and null absent denominators; no ASR/evasion reward path. This does not validate a real security experiment.

Proposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.

## Prioritized correction plan — proposed, not executed

1. **P0 — Freeze the scientific question and claim boundaries (F01/F04/F22).** Decide the precise preservation/augmentation research estimand and distinguish it from a BadSem-inspired security question. Produce a versioned label-validity contract. Keep all current results exploratory; no downstream security training is cleared.
2. **P0 — Validate semantic and task success independently (F02/F03/F10/F18).** Complete blinded review of existing pairs with uncertainty/adjudication; quantify proxy errors and instruction success. No model-generated substitute labels. Analyze reward redundancy, invalidity/uncertainty and clipping on existing evidence before any new objective is approved.
3. **P1 — Correct provenance and reward-interface gaps in a new version (F08/F12/F19/F20).** Group duplicate content for future splits, fix identity eligibility and explicit coefficient wiring, and validate new guards. Preserve v1, old coefficients, markers and all historical metrics; do not delete unfavorable samples.
4. **P1 — Resolve continuation/numerical comparability (F09/F15/F17).** First specify decisive controlled diagnostics. Keep exact acceptance unchanged. This phase authorizes no new GPU diagnostic; do not hypothesize a fix as if demonstrated.
5. **P2 — Define a testable RSI contribution and independent analysis (F05/F06/F07/F13).** Compare a defined adaptation mechanism against an ordinary schedule under a frozen fairness/selection design. State source/seed units and repeated-look handling in advance. Any later training needs its own authorized scope and budget.
6. **P3 — Refresh current-method documentation and only then assess security readiness (F16/F21/F22).** Evaluate fixed, documented artifacts with independent observations after upstream gates close; keep ASR/detection-evasion measurements outside automated reward optimization.

No scientific correction above was implemented by this audit. Existing monitoring continues. A passing engineering component does not waive another gate.

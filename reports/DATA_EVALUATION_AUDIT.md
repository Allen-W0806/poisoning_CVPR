# Dataset, evaluation, statistics and reproducibility audit

**Evaluation readiness: NO-GO for scientific claims; dataset provenance checks mostly pass with one new duplicate-image defect.**

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


## Official annotation and sampling checks

The CPU audit reads original annotation/question archives on the cluster. It independently joins all1500 selected question IDs to both archives and confirms matching image IDs, question text and all ten original answers, correct COCO filename, exact selected-image file SHA and exact deterministic sampling replay. It does not infer a person's identity or ethnicity from the reviewed duplicate photo; that question is simply an original dataset annotation.

| Internal split | Questions | Unique image IDs | Official source |
|---|---:|---:|---|
| Train |1000 |989 |VQAv2 official train2014 pool |
| Development |200 |197 |Same pool, image-ID-disjoint allocation |
| Internal held-out |300 |298 |Same pool; **not** official VQAv2 test |

All question IDs are unique and split image-ID intersections are empty. One train/test pair has identical JPEG bytes despite distinct IDs (F08). Train/development has zero exact-byte overlap. Four Phase4 training arms did not expose the duplicated train image. The v1 split is immutable; the audit accesses held-out annotations/file hashes only, with no model call, new metric evaluation, selection or marker change.

After editing, the label contract depends on what changed. Exact identity retains original annotation provenance. Independently verified question-relevant preservation can justify original answer use within that contract. A question-relevant color/count/object/relationship change may invalidate original answers; a mismatch can make the original question's premise false or unanswerable. A real semantic change irrelevant to this question may retain its answer but violate the global editing instruction. Ordinary brightness augmentation has no universal label-validity guarantee. Current edited-reference scores are correctly tagged diagnostic/eligible_for_accuracy=false.

## Metric dictionary

| Quantity | Unit / reference | What can be concluded |
|---|---|---|
| Training reward |320 source exposures, newly changing train batches; same-Qwen gated reward |Optimization observations; cross-update increases mix source difficulty and model change |
| Development reward |180 trajectories but10 distinct images; four looks/arm |Exploratory matched-source proxy comparison |
| Clean VQA accuracy |Official consensus scoring against ten dataset annotator answers on unmodified images |Historical frozen inference protocol accuracy; no Phase4 learned-Qwen improvement |
| Answer consistency |Same normalized answers; raw and positive-reference-gated versions separately stored |Agreement, possibly wrong in both cases; not semantic truth |
| Semantic-validity proxy |Same Qwen's pair check plus answer gate |Automatic agreement only |
| Independent semantic validity |No completed human labels |Unknown, no valid-edit success-rate estimate |
| Post-training security behavior |Real denominator0; synthetic fixtures labeled |No real behavior-change or ASR result |

Phase2 clean development81.15% and once-opened internal-test77.633333% belong to the old frozen baseline protocol. They are not new Phase4 training results. The official-compatible scorer uses dataset answers, not model-created ground truth. The auditor independently rechecks its output for500 saved original/edited answer pairs. No suspicious self-max score normalization or phantom result is identified by the fresh reviewer; the same-model semantic proxy remains a validity weakness.

## Final development reward, exact recomputation

| Condition | Seed42 | Seed123 | Mean across two seeds | Sample SD across two seeds |
|---|---:|---:|---:|---:|
| M0 | -0.437720587 | -0.254613674 | -0.346167130 | 0.129476139 |
| M1 | -0.624729042 | -0.347859168 | -0.486294105 | 0.195776565 |
| M2 | -0.530955535 | -0.254151467 | -0.392553501 | 0.195730033 |

Both fixed-RL seeds score below their frozen reference. The average RSI proxy reward is also below frozen. Raw same-answer fraction averaged across seeds is.75/.70/.70 for M0/M1/M2, while reference-positive raw consistency is.70/.65/.65. These differ from semantic-gated A_reward (.70/.55/.65). None should be relabeled clean accuracy or independent preservation.

`paired_summary` resamples ten source-level differences10,000 times with numpy default_rng7 and percentile2.5/97.5%. All16 stored frozen-paired intervals and final/intermediate M2−M1 intervals reproduce exactly. The unit is the source image **within one fitted seed**, not transitions, trajectories, questions on the same image or bootstrap draws. No pooled180-independent-image interval is valid. Repeated LR selection on the same development set, two fitted seeds and common ten images limit all inference; sample SD is descriptive, not a reliable population uncertainty estimate.

## Reproducibility and immutability

Pinned revisions: InstructPix2Pix `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`; Qwen `89644892e4d85e24eaac8bacfd4f463576704203`. Current environment versions reproduce saved runtime records: Torch2.6.0, Diffusers0.35.1, PEFT0.17.1, Transformers4.57.1, Accelerate1.11.0, hub0.35.3, skimage0.25.2. BF16 base/FP32 adapters, math SDPA, TF32 off and deterministic flags are recorded; flags are not an empirical bitwise guarantee.

CPU audit verifies1872 Phase4 files against their original per-file SHA receipt and2130 historical files before/after with no change. Frozen protocol SHA and once-opened test marker are unchanged. Existing source histories have different local/canonical commits but the Phase4 receipt provides32 normalized-LF source hashes; these are checked again at audit close. Some early wrapper/diagnostic manifests were dirty or mislabeled by parent seed metadata; scientific child configs and real command lines preserve the actual seeds. Those records remain untouched.

Fresh existing CPU tests:59 passed (`phase45_cpu_tests.log`). Read-only monitor snapshot: enabled/installed, interval2h, failures0, uncertain submission false; last successful native tick `2026-10-27T22:33:08.511954+00:00`, one pending owned CPU job `86533` due `2026-10-28T00:32:35+00:00`. No duplicate scheduler was created and no GPU research job was active at inspection. This is a point-in-time operational check, not a promise of future uptime.

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


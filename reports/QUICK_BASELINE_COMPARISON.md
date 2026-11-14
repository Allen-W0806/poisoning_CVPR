# Quick pretrained IP2Pix comparison

PROJECT_ID: RSI-POISONING. Generated UTC 2026-11-02T19:22:56.543053+00:00. **DEVELOPMENT ONLY: four independent source IDs, one training seed42.** Actual saved selected checkpoints and PNGs from job88392 are compared. Read-only evaluation does not depend on backward bitwise reproducibility. No parameters were updated, no checkpoint was reselected, and no sealed final-pool image was opened.

| Method | Training IDs attempted/retained | Selected step | Eval IDs | Brightness proxy (higher better) | Paired delta vs Frozen [95% source CI] | SSIM (higher similarity) | Calibrated LPIPS (lower distance) | Exact no-op |
|---|---|---|---|---|---|---|---|---|
| Frozen | 0 / 0 | 0 | 4 | -0.117785774 | +0.000000000 [0.00000000, 0.00000000] | 0.172757771 | 0.896046922 | 0/4 |
| DDPO-only | 20 / 20 | 2 | 4 | -0.118267060 | -0.000481285 [-0.00142581, 0.00045783] | 0.166480311 | 0.899597660 | 0/4 |
| Ordinary Scheduler | 20 / 20 | 2 | 4 | -0.118165582 | -0.000379808 [-0.00107563, 0.00036716] | 0.168439488 | 0.900560051 | 0/4 |
| DDPO + RSI | 20 / 10 | 1 | 4 | -0.118036655 | -0.000250881 [-0.00092899, 0.00031444] | 0.169974773 | 0.899786994 | 0/4 |

The trained-arm brightness means are all below Frozen. All paired brightness intervals include zero. This is a measured negative development result, not evidence of useful learning or causal RSI superiority. Negative brightness proxy means the edit moved global luminance farther from the target than doing nothing. Similarity to the source is not instruction-following or semantic correctness.

## Reproduction and metric definitions

Saved four-source score vectors and arm means reproduce within1e-12 (actual maximum error 0). Newly computed PNG brightness reproduces every saved score within1e-12. Same assessment source IDs `230462, 362793, 265622, 403306` and generation seeds4220000–4220003 appear in every arm. No source-ID or exact-content overlap with any of the20 attempted pilot training sources. The retained RSI checkpoint contains10 source exposures; its rejected branch attempted another10. All20 are conservatively excluded from held-out and expanded comparisons. Four separate validation sources were used for scheduling/RSI decisions and are excluded here.

Brightness proxy: |L_source−min(1,1.03 L_source)|−|L_edit−min(1,1.03 L_source)|. L is mean RGB[0,1] weighted by (0.2126,0.7152,0.0722), without sRGB linearization. Units are normalized pixel luminance, not percent accuracy. SSIM uses skimage structural_similarity on geometry-aligned512×512 RGB uint8 arrays, channel_axis2,data_range255,default7-pixel uniform window. LPIPS is calibrated AlexNet v0.1 from lpips0.1.4; normalize=True maps RGB[0,1] to[-1,1]. Learned calibration SHA256 `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` and trunk SHA256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` were checked; all loaded calibration tensors match and identity LPIPS is0. Exact no-op is pixel equality after identical source preprocessing; its operational definition detects exact identity only.

All means and paired contrasts, including SSIM, LPIPS and no-op, with intervals are in the CSV and table below. Intervals enumerate all256 equally weighted ordered bootstrap resamples of4 source IDs. The paired resample indices are shared across arms. These exploratory conditional intervals do not capture training-seed variability, dataset selection bias or independent semantic uncertainty. No-op marginal intervals use Wilson95%;0/4 has an upper bound around0.49, despite an empirical paired-bootstrap zero interval when every observed pair is zero.

| Method | Metric | Mean | 95% source interval | Delta Frozen | Paired95% interval |
|---|---|---|---|---|---|
| Frozen | brightness_proxy | -0.117785774 | [-0.19655650, -0.04142968] | +0.000000000 | [0.00000000, 0.00000000] |
| DDPO-only | brightness_proxy | -0.118267060 | [-0.19818598, -0.04076818] | -0.000481285 | [-0.00142581, 0.00045783] |
| Ordinary Scheduler | brightness_proxy | -0.118165582 | [-0.19776223, -0.04093242] | -0.000379808 | [-0.00107563, 0.00036716] |
| DDPO + RSI | brightness_proxy | -0.118036655 | [-0.19758429, -0.04101644] | -0.000250881 | [-0.00092899, 0.00031444] |
| Frozen | SSIM | 0.172757771 | [0.13745587, 0.20805967] | +0.000000000 | [0.00000000, 0.00000000] |
| DDPO-only | SSIM | 0.166480311 | [0.13106618, 0.20104796] | -0.006277460 | [-0.00929589, -0.00315475] |
| Ordinary Scheduler | SSIM | 0.168439488 | [0.13285354, 0.20402543] | -0.004318283 | [-0.00678410, -0.00213651] |
| DDPO + RSI | SSIM | 0.169974773 | [0.13452803, 0.20542151] | -0.002782999 | [-0.00456730, -0.00099869] |
| Frozen | calibrated_LPIPS | 0.896046922 | [0.82835835, 0.96738893] | +0.000000000 | [0.00000000, 0.00000000] |
| DDPO-only | calibrated_LPIPS | 0.899597660 | [0.82760033, 0.97159499] | +0.003550738 | [-0.00075802, 0.00785950] |
| Ordinary Scheduler | calibrated_LPIPS | 0.900560051 | [0.82880622, 0.97506462] | +0.004513130 | [0.00044787, 0.00857839] |
| DDPO + RSI | calibrated_LPIPS | 0.899786994 | [0.82936117, 0.97197746] | +0.003740072 | [0.00100282, 0.00647733] |
| Frozen | exact_noop | 0.000000000 | [0.00000000, 0.48989084] | +0.000000000 | [0.00000000, 0.00000000] |
| DDPO-only | exact_noop | 0.000000000 | [0.00000000, 0.48989084] | +0.000000000 | [0.00000000, 0.00000000] |
| Ordinary Scheduler | exact_noop | 0.000000000 | [0.00000000, 0.48989084] | +0.000000000 | [0.00000000, 0.00000000] |
| DDPO + RSI | exact_noop | 0.000000000 | [0.00000000, 0.48989084] | +0.000000000 | [0.00000000, 0.00000000] |
| Frozen | original_reference_VQA_agreement | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| DDPO-only | original_reference_VQA_agreement | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Ordinary Scheduler | original_reference_VQA_agreement | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| DDPO + RSI | original_reference_VQA_agreement | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |

## Checkpoint and training lineage

| Method | Selected checkpoint relative to canonical project | SHA256 | Training/selection history |
|---|---|---|---|
| Frozen | `results/execution_remediation_20261102_v1_pretrained_pilot/frozen/selected/training_state.pt` | `0aea815a9e423ecc6378b16787e2bee6cd803f729af1a26ee07cfb6888fd314d` | fixed step0 |
| DDPO-only | `results/execution_remediation_20261102_v1_pretrained_pilot/ddpo_only/selected/training_state.pt` | `60e4e6161578e7f1cccdae4c34db1c1a9d66933b27527bf0acb2db2e88cebea5` | predeclared step2 |
| Ordinary Scheduler | `results/execution_remediation_20261102_v1_pretrained_pilot/ordinary_scheduler/selected/training_state.pt` | `73f48d6ef732c645942cccfe070a7a829bd40515d63d9c573ce7f675d6f25f4b` | predeclared step2 |
| DDPO + RSI | `results/execution_remediation_20261102_v1_pretrained_pilot/rsi/selected/training_state.pt` | `b021852e5bc5f2329448b5bc3759d24533f6f718fa2ea0ea2698cb659dd48864` | own validation rejected candidate step2; restored own step1 prefix |

Pinned base: `timbrooks/instruct-pix2pix@31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`, fp16 artifact variant loaded in BF16; exact selected-state contracts and runtime sidecars agree. All four selected files were hashed now, and their hashes equal the original result receipts and sidecar bindings. Base model files were hashed now; safetensors hashes match their content-addressed cache blobs. Full file hashes and selected contracts are in checkpoint_and_base_lineage.json. These .pt files contain LoRA and optimizer state, not duplicated full base models; each has256 FP32 LoRA tensors /1,594,368 elements. LoRA was already FP32, which did not prevent the historical numerical mismatch.

Job88392 performed four physical optimizer calls total: one shared prefix plus three branch candidates. Each trained arm attempted two logical updates; RSI rejected its LR5e-6 step2 candidate using its own validation and restored its own LR1e-5 step1 prefix. DDPO retained step2 LR1e-5; ordinary ReduceLROnPlateau retained step2 LR5e-6. Fixed step rules and validation rollback preceded assessment. Common shared-prefix cost is counted once.

## Compute and live resource control

Live sacct rechecked job88392: submit 2026-11-02T04:07:10, start 2026-11-02T04:07:10, end 2026-11-02T04:11:12 UTC, COMPLETED, **242 A100 GPU-seconds shared by the entire pilot**. The allocation includes loading, training, selection, validation, assessment and writes; evaluation-only GPU time was not separately measured and cannot be attributed to individual rows. Four assessment images per arm imply24 UNet forward calls each,96 total. The frozen8-image validation+assessment branch component took 10.966034s; that includes checkpoint bookkeeping and is not a pure inference benchmark. Per-arm branch timings are retained in original receipts, not multiplied into four independent allocation costs.

This result-processing run used **0 GPU-seconds**, 10.955s CPU wall time and 11.034s CPU process time on the existing login CPU process (no new Slurm job). It made0 editor-generation and0 VQA calls. Current live queue and full allocation records are saved in live_resource_and_authorization.json. Overnight and urgent-debug admissions remain closed; the10-hour authorization expired at 2026-11-02T16:17:57.636501+00:00. No new valid GPU allocation authorization was found in the current request or live control ledgers. The remaining expired balances are not usable. Monitor 88532 remains due 2026-11-02T20:32:35+00:00; no duplicate monitor was created.

## Expanded read-only evaluation, prepared and separately priced

Prepared a fixed **64-source DEVELOPMENT ONLY** cohort: the existing4 assessment IDs plus60 deterministically selected available unique historical validation sources. Selection hashes source IDs only and does not inspect scores. All20 attempted pilot training IDs/content hashes,4 selection-validation IDs/content hashes,160 sealed final IDs/content hashes and duplicate IDs/content are excluded. Sealed metadata was consulted only for exclusion; no final image or answer was used. The same64 sources, fixed seeds, editing instruction,512 white-pad geometry,2-step DDIM eta1,7.5/1.5 guidance, sequential CFG and math SDPA apply to all four fixed selected checkpoints. Cohort and protocol hashes are frozen in `results/quick_comparison_20261102T192245Z/expanded_development_cohort.json` and expanded_readonly_protocol.json. No further checkpoint selection is permitted.

Price:256 generated images /1536 UNet forward calls. Scaling the measured frozen branch component gives about 350.9 A100-seconds, with substantial setup/throughput uncertainty. Proposed hard allocation cap **600 A100 GPU-seconds**,1 GPU,10-minute wall limit, followed by CPU-only SSIM/calibrated LPIPS/brightness/no-op processing. This is a proposed cap, not granted authorization or a guaranteed estimate. Dollar pricing is unavailable. The expanded evaluation has **NOT RUN** because no renewed GPU budget/window exists. Its eligibility does not depend on a BF16 backward fix. Optional original-reference VQA is a separate unpriced/unexecuted extension requiring its own pinned scorer and resource authorization; it is not silently included in the600-second quote.

## Scientific limits

All exposed sources remain DEVELOPMENT ONLY. Original-reference VQA agreement is **NOT TESTED for these four selected-checkpoint outputs**: the pilot made no Qwen calls. Historical frozen-pipeline answers cannot substitute. Independent edited-label correctness, Clean Accuracy and ASR are **NOT TESTED**. No human labels were created. The original scientific reward remains unreleased; this isolated global-brightness engineering proxy does not change it.

BF16 ordinary-path gradient/optimizer inconsistencies remain preserved. They limit reproducible training and attribution of arm differences, but do not invalidate read-only descriptions of these explicitly hashed fixed outputs. No numerical fix or security-effectiveness claim is made. Full per-source metrics, exact source bootstrap, live allocation evidence, checkpoint/base lineage and proposal are retained under `results/quick_comparison_20261102T192245Z`. No old C0–C5 or Tiny-UNet table was regenerated.

Additional CPU checkpoint history verification: actual consumed trajectory counts are0/20/20/10; saved Adam step values are empty/2/2/1. Frozen adapter/Adam match shared_initial exactly; selected RSI adapter/Adam match its own shared_prefix exactly. Selected file hashes remain unchanged. See selected_training_history_reverification.json.

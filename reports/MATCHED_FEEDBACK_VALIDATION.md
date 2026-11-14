# Matched Benign Feedback Validation

Run: `matched_feedback_20261028_v2`. Updated 2026-10-28T21:35:49-04:00 (America/New_York). **Frozen inference only; zero pretrained-model optimizer steps.**

**Feedback computation and tested reproducibility: PASS. Distinguishable inherited-reference agreement: PASS within the tested paired comparison scope. Useful semantically valid generator-learning feedback: NOT VALIDATED. Independently verified semantic secondary analysis: BLOCKED. Full pretrained BF16 training/restart: NOT VALIDATED.**

Primary results below use the main pool, n=160 unique source images. Pilot and main are separate pools and analyses. Candidate observations and repeated inference seeds are dependent, not additional independent images. There are five primary candidate conditions; C5 is an auxiliary preprocessing control.

## Design, data and provenance

The all-project census inspected 1050 JSON/JSONL/CSV records/manifests, including train, validation, test, gate, development, prior experiment and result records. It found 1638 previously referenced numeric source-image IDs, 1511 content hashes, 1484 cached originals, and **0 unused cached originals**. The official cached annotation pool contains 82783 images; 80731 unique image IDs meet the predeclared project-exposure exclusion and >=6/10 normalized human-answer agreement rule. Answer-category candidate populations overlap and must not be summed.

The script staged exactly 184 new official COCO images, preserving the original question and ten human answers. Twenty-four pilot sources and 160 main sources are disjoint by both image ID and content SHA256 and have no observed overlap with historical project records. Original project splits were not modified. Model pretraining contamination remains unknown. Category-balanced selection (pilot 8/8/8; main yes/no53, number53, other54) and the human-agreement rule limit population generalization.

Data source: cached VQAv2 train2014-2017-04-26 annotations/questions, and the fixed selected image paths at `https://s3.amazonaws.com/images.cocodataset.org/train2014/`. Annotation terms: CC BY 4.0; images retain original Flickr/COCO per-image terms. No new model weights or large packages were downloaded. The model/environment pins match the prior audit.

[Protocol](../results/matched_feedback_20261028_v2/protocol.json), [exposure census](../results/matched_feedback_20261028_v2/prior_exposure_census.json), [data checks](../results/matched_feedback_20261028_v2/data_checks.json), [pilot manifest](../results/matched_feedback_20261028_v2/pilot_manifest.json), [main manifest](../results/matched_feedback_20261028_v2/main_manifest.json). Source manifests were frozen before generation; candidate manifests with actual output hashes were frozen before VQA inference. Original and edited image hashes, raw answers and labels remain in every row.

## Fixed candidate conditions

| ID | Definition |
|---|---|
| C0 | Original unmodified RGB image |
| C1 | Exact original pixels re-encoded as identity PNG |
| C2 | Global brightness ×1.03, predefined |
| C3 | Global contrast ×1.03, predefined |
| C4 | Frozen pinned IP2Pix, lighting-only instruction, generation seed42, 256×256, 10steps |
| C5 | Auxiliary IP2Pix resize/padding only; no generation |

All target conditions use the original question and references, Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`, greedy short-answer decoding, max32 tokens, pixel budget4096–262144, physical microbatch1. The source order rotates condition order deterministically. IP2Pix revision is `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`; BF16 runtime, FP16 cached variant, guidance7.5/image guidance1.5, math SDPA, no adapter loaded. No strength/instruction/seed selection was made using target performance. Every attempted edit remains in the evidence.

C4 changes geometry as well as image content relative to C0. The predeclared auxiliary C4−C5 comparison helps diagnose generation beyond resizing/padding; C4−C0 is not attributed solely to generation.

## Main paired results

| Condition | Unique sources | Mean agreement | 95% CI | Difference C0 | Paired 95% CI | At score 1 |
|---|---|---|---|---|---|---|
| C0 | 160 | 0.84875 | [0.79500, 0.89812] | +0.00000 | [+0.00000, +0.00000] | 80.00% |
| C1 | 160 | 0.84875 | [0.79500, 0.89812] | +0.00000 | [+0.00000, +0.00000] | 80.00% |
| C2 | 160 | 0.85062 | [0.79687, 0.89938] | +0.00187 | [+0.00000, +0.00563] | 80.00% |
| C3 | 160 | 0.85062 | [0.79936, 0.90000] | +0.00188 | [-0.01688, +0.02063] | 80.00% |
| C4 | 160 | 0.64937 | [0.57812, 0.71687] | -0.19938 | [-0.26500, -0.13623] | 58.75% |
| C5 | 160 | 0.76437 | [0.70250, 0.82438] | -0.08438 | [-0.13438, -0.03687] | 70.62% |


| Comparison | Mean difference | Paired 95% CI | Permutation p | Holm p (4 planned) | Win | Tie | Loss |
|---|---|---|---|---|---|---|---|
| C1−C0 | +0.00000 | [+0.00000, +0.00000] | 1.000000 | 1.000000 | 0.00% | 100.00% | 0.00% |
| C2−C0 | +0.00187 | [+0.00000, +0.00563] | 1.000000 | 1.000000 | 0.62% | 99.38% | 0.00% |
| C3−C0 | +0.00188 | [-0.01688, +0.02063] | 1.000000 | 1.000000 | 1.25% | 98.12% | 0.62% |
| C4−C0 | -0.19938 | [-0.26500, -0.13623] | 0.000100 | 0.000400 | 1.88% | 72.50% | 25.62% |


The main planned family has 1 comparisons with Holm p<.05 and paired-bootstrap intervals excluding zero. This is evidence about reference-answer agreement under the fixed transformations, not certification of correct edited labels or useful learning. C1 pixel/answer/score identity is an engineering invariant, not a statistical equivalence claim.

The primary uncertainty analysis resamples source-image indices jointly across all candidates 10,000 times. Two-sided paired sign-permutation tests use 10,000 sign draws and a plus-one correction. Four predeclared C1–C4 vs C0 comparisons form one Holm family; ten exploratory primary-condition pairs form a separately labeled Holm family. C4−C5 is separately labeled auxiliary. No thresholds, transforms or pools were adjusted after observing target results.

## Group informativeness, saturation and reproducibility

Mean within-source C0–C4 reward variance is 0.045131; **70.62%** of source groups give exactly the same reward to all five candidates; **29.38%** are non-degenerate. C4 ties C0 on 72.50% of sources. Full win/tie/loss comparisons, category summaries and score distributions are retained.

Selected inference repeats cover six sources per phase (two per answer category), all six conditions, and seeds42/123/3407. Exact repeat agreement is 100.00% across 36 main repeated candidate cases. This is not full-pool repeated inference and does not increase n. Generation repeats cover the same six sources under identical generation settings. Frozen UNet, VAE, text encoder and target VLM parameter hashes are checked before/after each phase.

All-attempted score utility is zero for technical generation/inference failure; the raw official VQA score remains missing where no answer exists. Technical missingness and failure counts are preserved. For C2/C3/C4, inherited-reference scores are descriptive agreement until expected-answer preservation is verified. A high score alone does not establish semantic or instruction preservation.

## Secondary analysis and limits

The independently validated, all-candidate semantic-preserving matched subset is **BLOCKED**, with zero independently reviewed C4 cases. The original source labels and exact identity C1 are available; this does not certify nontrivial edits. We do not manufacture a filtered valid-edits effect from automated similarity or assistant screening. Technical complete-case sensitivity equals all-attempted analysis only when there are no technical failures; semantic robustness remains unvalidated.

The auxiliary C4−C5 mean difference is -0.11500, paired95% CI [-0.17125, -0.0625], permutation p=0.000100. This comparison still inherits the semantic-label limitation.

## Engineering failure and versioned control repair

The original main job86795 remains FAILED. One original JPEG (image361516) is encoded in grayscale L mode; the validation wrapper incorrectly rejected the encoded mode even though all VLM inputs are converted to RGB. It skipped the original C0 inference and failed both all-record validity and the identity-control assertion. No edited-image generation failed. The source pixels, question and ten references were unchanged.

Repair job86796 validated all160 decoded RGB inputs (159 RGB-encoded originals, one grayscale-encoded original), proved grayscale C0/C1 pixel equality, and reevaluated that source's C0 and C1 under seeds42/123/3407: six real inference calls, exact answer/score agreement and unchanged target hash. A versioned `main_repair_v1` accepted view replaces only the missing C0 observation; the remaining1031 observations and all generated images are unchanged. The original failed JSONL/checks/job state remain immutable. No data, score threshold, transform or question was changed to improve results.

Primary corrected-view outputs: [paired_feedback.csv](../results/matched_feedback_20261028_v2/paired_feedback.csv), [paired_statistics.json](../results/matched_feedback_20261028_v2/paired_statistics.json), [candidate_group_statistics.csv](../results/matched_feedback_20261028_v2/candidate_group_statistics.csv), [feedback_distribution.csv](../results/matched_feedback_20261028_v2/feedback_distribution.csv), [all plots](../results/matched_feedback_20261028_v2/main/plots/paired_differences_CI.png). All original attempts, including the skipped C0, plus six repair validations are separately retained in [all_attempts_feedback.csv](../results/matched_feedback_20261028_v2/all_attempts_feedback.csv). [Repair receipt](../results/matched_feedback_20261028_v2/main_repair_v1/repair_receipt.json) and [accepted-revision mapping](../results/matched_feedback_20261028_v2/accepted_phase_revisions.json) make the derivation explicit. Main and pilot numbers must not be pooled as one post-hoc favorable result.

## Reproduce the retained analysis

From `H:/Project/Poisoning`, with the existing local NumPy/SciPy/Matplotlib/Pillow environment:

```powershell
python -B scripts/analyze_matched_feedback.py --run-id matched_feedback_20261028_v2
python -B scripts/report_matched_validation.py --run-id matched_feedback_20261028_v2
python -B scripts/finalize_matched_validation.py --run-id matched_feedback_20261028_v2
```

These commands recompute from the frozen source-level predictions, retain failed attempts separately, and never rerun or tune the models. The original source census, selected image IDs, questions, labels, archived source-code versions, model pins, config and actual file hashes are preserved in the run. A future fresh-data census will reflect additional exposure; it must not silently replace this frozen pool. For new inference, use `run_matched_feedback.py --run-id <new-exclusive-run> --phase pilot|main` only after explicitly staging that run's versioned manifests and data checks. Use `submit_matched_feedback.py` for the one-hour stage budget guard. Do not run into an existing phase directory. Canonical `.venv-editor` is reused without installation; GPU work uses Slurm.

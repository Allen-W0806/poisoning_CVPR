<!-- semantic_clean_20261029_v1 -->
> Latest stage: bounded clean-target LoRA training now passes; independent edited-image review remains BLOCKED. Historical results below retain their original scope. See [current decision](NEXT_EXPERIMENT_DECISION.md) and [clean baseline](CLEAN_TARGET_TRAINING_BASELINE.md).

# RSI Gradient and Feedback Validation — Experimental Results

**Synthetic gradients pass; frozen VQA repeats exactly. Ten batch scores are measurable, but do not establish comparable GRPO feedback. The editor fails strict semantic preservation on some benign instructions. Large-scale fine-tuning remains disabled.**

Full code/algorithm audit: [GRADIENT_AND_FEEDBACK_AUDIT.md](GRADIENT_AND_FEEDBACK_AUDIT.md). English dashboard: [gradient_feedback_validation.html](../dashboard/gradient_feedback_validation.html).

## Answers to the six project questions

1. **Updated parameters:** the existing categorical policy's 27-element head, isolated synthetic group-policy logits, and 4,096 attention-LoRA elements in a randomly initialized tiny synthetic UNet. Its frozen base/encoder did not change. Pretrained IP2Pix, its VAE/text encoder, and Qwen3-VL received **zero optimizer steps**. A full IP2Pix optimizer was instantiated solely to inspect its 256 tensors / 1,594,368 LoRA elements.
2. **Gradient estimator:** verified within the finite synthetic protocols. Categorical gradient RMSE at N=65,536 is .68% for direct logits and .61% for existing EditPolicy. The existing DDPO transition achieves 5.76% relative RMSE at N=1,048,576, versus the analytic scaled Gaussian reference; error slopes are close to -.5. This does not certify every BF16 pretrained-model execution or the scientific reward construct.
3. **Ten batches:** every score is traceable and reproducible. Batch means span .5125–.89375, between-batch variance .0120421, pooled within-batch variance .1180125. Category-stratified composition permutation p=.07179; **0/45** exploratory pairwise comparisons survive Holm correction. These results do not demonstrate reliable candidate ranking or a system/policy effect.
4. **GRPO:** the original implementation is DDPO-SF / REINFORCE, not GRPO. A separate correctly identified one-step group reference passed detached-advantage, old-policy ratio, zero-variance, clipping, KL, learning and exact enumerated-surrogate checks. Group standardization is not an unbiased gradient of the raw expected-reward objective.
5. **Frozen VLM:** 480/480 original-image evaluations completed, with no missing/invalid responses. Every one of the 160 raw answers matched across seeds 42, 123 and 3407. Full parameter hashes matched before/after. Nine additional evaluations of three screened edits from two source images also repeated exactly.
6. **Remaining limits:** batches are different data, many scores saturate, the development set was previously used, target pretraining contamination is unknown, edited-label review is assistant screening rather than independent human certification, and editor outputs can change the scene or colors. Historical pretrained BF16 restart failures and reward-release guards remain open. No framework efficacy or attack-generation claim follows.

## Dataset and immutable protocol

Existing cached **VQAv2, train2014-2017-04-26**, using its project's internal **validation** split of 200 questions. This is a development subset of official COCO train2014, not official VQA test. Original ten annotator answers and questions were preserved.

Official annotation sources:

- [VQAv2 training annotations](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), saved archive SHA256 `fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853`.
- [VQAv2 training questions](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), SHA256 `05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e`.
- Images originally retrieved from the [official COCO bucket](https://s3.amazonaws.com/images.cocodataset.org/train2014/). No new image/model download was needed. [VQA terms](https://visualqa.org/terms.html) license annotations CC BY 4.0; COCO images retain individual Flickr terms under [COCO terms](https://cocodataset.org/#termsofuse), not a blanket annotation license. This stage reuses cached research images and does not redistribute the 160 original photos. Per-photo license IDs were not separately re-audited.

A fixed shuffle seed 20261028 precedes image-ID and content-SHA256 deduplication and selection of 160 unique images. Selected image IDs and hashes have no observed overlap with internal train/test. Known answer categories are balanced across ten batches of 16; this does not balance unknown item difficulty. The batch assignment, sample IDs, original labels, conditions, seeds, model revisions and protocol are frozen before inference in [evaluation_manifest.json](../results/benign_models_20261028_v1/evaluation_manifest.json), with a separate SHA256 receipt. Repeated seeds reuse exactly the same examples and groups. The effective image sample size is **160**, not 480.

Normalization and scoring use the pinned official upstream-compatible VQA toolkit. Main accuracy is leave-one-annotator-out consensus. Separately named `exact_any_reference` compares the normalized full response to any original reference; it is not a replacement for consensus. No extraction based on the correct label is performed. Invalid responses would score zero and remain in denominators; none occurred. All CSV scores were recomputed from raw predictions and ten reference answers.

## Frozen target performance

| Metric | Measured value |
|---|---:|
| Official VQA consensus | **80.9375%** |
| 95% image-bootstrap CI | **75.3750–86.0016%** |
| Exact match to any normalized reference | 88.7500% |
| Exact-any 95% image-bootstrap CI | 83.7500–93.1250% |
| Number category | 64.4444%, n=18 |
| Other category | 74.5977%, n=87 |
| Yes/no category | 96.3636%, n=55 |
| Invalid / missing original responses | 0 / 0 |
| Raw-answer seed agreement | 160/160 = 100% |
| Maximum per-image reward variance across seeds | 0 |
| Mean original-image latency | .211964 seconds |
| Seed mean latencies 42 / 123 / 3407 | .214097 / .210871 / .210924 seconds |
| Sum of timed original-image inference | 101.742567 seconds |

The confidence intervals use 5,000 percentile bootstrap resamples of unique images and quantify this diagnostic pool's item-sampling uncertainty. They do not account for unknown training contamination or prove population representativeness. Per-category intervals and complete timing data are in [feedback_statistics.json](../results/gradient_feedback_summary_20261028/feedback_statistics.json). Greedy decoding explains zero observed seed variation; it does not prove universal device- or version-independent determinism.

## Batch feedback

| Batch | Mean consensus reward |
|---|---:|
| 0 | .83750 |
| 1 | .89375 |
| 2 | .51250 |
| 3 | .86875 |
| 4 | .80000 |
| 5 | .84375 |
| 6 | .85625 |
| 7 | .87500 |
| 8 | .82500 |
| 9 | .78125 |

All seeds reproduce this table. Batch means have no exact ties, but example scores have many ties: **113/160 (70.625%) at 1 and 18/160 (11.25%) at 0**. Per-batch tie-pair counts, missingness, saturation, confidence intervals, category counts and exact sample lists appear in [batch_feedback.csv](../results/batch_feedback.csv).

The composition test permutes scores within category while preserving every batch's category counts and capacity. Observed between-batch variance .0120421 lies inside the permutation null's 95% range [.0021721,.0143349]; p=.07179. The batch-score range .38125 also lies inside the null range [.14375,.39375]. Failure to reject is not evidence of equivalence; the 16-image batches are small and actual difficulty remains unmeasured. All batches evaluate the **same frozen model**, so they cannot identify competing-policy quality. Validating an RL feedback signal will require different candidates evaluated on the same conditions, with retained independent rewards and an appropriately powered paired design.

## Editor sanity and edited VQA

Pinned InstructPix2Pix loaded locally cached weights and generated 9 images per sanity run: three sources × seeds `[42,42,123]`, 256×256, ten steps. All images decoded as valid finite uint8 RGB with expected dimensions. Each same-seed pair was pixel-identical; changing seed changed each output. Mean generation latency in the corrected run was **.903741 seconds**, excluding model loading and hashing. Complete UNet/VAE/text-encoder hashes were identical before and after. No LoRA was loaded during inference.

The harmless instruction was: “Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.” Global brightness did not always increase: the surfing image mean luminance fell from 187.64 to 171.30 / 140.04. One oven output became a sofa scene; restaurant cups changed color. **Engineering inference passes, but strict instruction and property preservation fail.** Inspect the [contact sheet](../results/benign_editor_20261028_v2/editor_contact_sheet.png) and [visual review](../results/editor_visual_review_20261028.json).

Assistant visual screening accepted only three edits for question-specific label consistency (tile remains visible; seated women are not playing tennis), excluded the scene replacement, and excluded two ambiguous time-of-day edits. Accepted edits are not certified as preserving every requested visual property. Three edit conditions from only two source images × three VLM seeds gave **9/9 correct, exactly repeated answers**. This small selected subset is descriptive, not a robustness estimate; it must not be combined into the original 160-image accuracy. Edited predictions and their own immutable manifest are in [reviewed_edits_20261028_v1](../results/reviewed_edits_20261028_v1/manifest.json).

## Failure and repair ledger

| Evidence | Failure / action | Final interpretation |
|---|---|---|
| Local `gradient_feedback_20261028_v1` | `diffusers` absent in CPU environment | Blocked dependency retained; existing canonical environment used |
| Canonical `gradient_feedback_20261028_remote_v1` | DDPO gradient RMSE .227693 > .10 | Failed precision retained; increase samples only under v2; final .0575715 |
| CPU submission to `epyc` | Invalid partition, no allocation | Corrected to observed `epyc-cpu`; no hidden compute charge |
| Job86736, `benign_models_20261028_v1` | NumPy Boolean JSON serialization failed after VQA/editor inference | Job remains FAILED. All 480 VQA records and hashes reconciled; editor-only rerun after explicit bool conversion passes. Truncated original `editor_checks.json` retained as negative evidence and never read as valid JSON |
| Job86739, `ddpo_wiring_20261028_v1` | Synthetic pipe omitted VAE/text-encoder placeholders needed by parameter accounting | Test-fixture construction repaired, failed result retained |
| Job86742 tiny substage, `ddpo_wiring_20261028_v2` | Sample/replay attention backend mismatch, behavior logp error 3.71933e-5 | Strict assertion retained; math SDPA applied consistently in v3. Job86742 exit0 reflects its successful final edited-VQA command, not success of its failed tiny substage |
| Benign editor semantic review | Scene replacement, recoloring and darkening | Not “repaired” by relabeling or selection of only good seeds. All outputs retained; invalid/ambiguous labels excluded from edited VQA |

No assertion was removed or acceptance threshold relaxed. The tested production RL source was not edited. Twelve targeted pre-existing transition/reward/VQA regression tests pass in the canonical environment; log: [targeted_regression.log](../logs/gradient_feedback_20261028/targeted_regression.log).

## Resources and reproduction

One A100 per GPU job, sequential allocation. Slurm elapsed times including failures: job86736 138s; 86739 6s; 86741 20s; 86742 24s; 86745 10s: **198 allocated GPU-seconds (.055 GPU-hours)**. CPU-only job86738 used 28 elapsed allocation seconds. Process timing is separate: successful canonical synthetic stage 22.8221s, tiny wiring 4.1579s, screened-edit VQA 13.0570s, first local synthetic stage 4.7049s. Login/local orchestration overhead is not fabricated as a total compute measurement. [Resource ledger](../results/gradient_feedback_summary_20261028/resource_accounting.json).

All scratch and archives were on verified H: or canonical shared storage. No model download, installation, multi-GB C: artifact, or large-scale fine-tuning occurred. Reproducible source, plots, diagnostic checkpoints and negative evidence are retained. End-of-run size and cleanup receipts are in the summary directory.

From the local project, regenerate statistical tables, dashboard and PNG/SVG plots:

```powershell
python -B scripts/report_gradient_feedback.py
```

In the existing canonical `.venv-editor`, use new unique run IDs (directories are exclusive):

```bash
.venv-editor/bin/python -B scripts/validate_policy_gradients.py --protocol configs/gradient_feedback_validation_v2.json --run-id gradient_reproduction_UNIQUE
# Allocate one GPU via the bounded Slurm wrapper; no VLM fine-tuning:
sbatch slurm/gradient_feedback_validation.sbatch benign_reproduction_UNIQUE
.venv-editor/bin/python -B scripts/validate_ddpo_wiring.py --run-id wiring_reproduction_UNIQUE
```

The last command also requires an allocated GPU. Group exact-surrogate checks run locally with `python -B scripts/validate_group_reference.py --run-id group_reproduction_UNIQUE`. Reviewed-edit replay uses `python -B scripts/validate_reviewed_edits.py --run-id edited_reproduction_UNIQUE` inside a GPU allocation and consumes the retained visual-review manifest. Plot dependencies are NumPy/SciPy/Matplotlib, available locally; canonical `.venv-editor` has no Matplotlib, so the documented plotting invocation belongs to the local environment. Core source snapshots, hashes, all protocols and model identifiers are retained. No tests consume security metrics or optimize attack generation.

Requested deliverables: [gradient_validation.csv](../results/gradient_validation.csv), [batch_feedback.csv](../results/batch_feedback.csv), [benign_vlm_evaluation.csv](../results/benign_vlm_evaluation.csv), [parameter_update_audit.csv](../results/parameter_update_audit.csv), [plots and source data](../results/gradient_feedback_summary_20261028/feedback_statistics.json). The combined VLM CSV labels original and reviewed-edit conditions; the two conditions must be analyzed separately.

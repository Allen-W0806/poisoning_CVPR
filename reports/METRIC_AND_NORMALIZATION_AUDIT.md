# Metric and normalization audit

Snapshot 2026-11-01T05:16:56.482704+00:00. Original research question and scientific coefficients are unchanged.

| Metric | Unit | Executed source | Normalization/definition | Limit |
|---|---|---|---|---|
| Gaussian transition logp | per latent transition | `src/rl/diffusion_trajectory.py:transition` | Mean log density over latent coordinates; actions detached for score-function differentiation. | Not VQA probability, semantic confidence or post-training ASR. |
| Terminal benign reward R | per trajectory/image-question pair | `src/rl/reward.py:validation_reward` | clip(S+.5A-.2D-.1C-U,-1,1.5); U=0 only verified valid score1 else1. | Legacy preservation proxy; coefficients/scientific objective unchanged. |
| Advantage/coefficient | per trajectory | `src/rl/reward_normalization.py:fixed_advantage` | Baseline0 and scale1; terminal reward detached. No group mean/std. | No GRPO; clipping reward and clipping gradient norm are separate operations. |
| Loss and gradients | logical optimizer batch | `src/rl/diffusion_rl_trainer.py:update` | -1/N sum_i R_i *1/T sum_t mean_coordinate(logp_it); physical microbatch1, accumulation N. Norm capped1. | Fixed coordinate/time scaling; batch-mean reward is logging only, not broadcast per-sample feedback. |
| S/validity | per image-question pair | `src/evaluation/semantic_validation.py:semantic_decision` | Identity score1 verified; same-Qwen provisional agreement .5 unverified; contradiction0. | Question-relative proxy does not certify global semantics; manual labels0. |
| A/answer consistency | original/edited question pair | `src/evaluation/answer_consistency.py:answer_consistency` | 1 only equal normalized predictions and both reference scores>0. | Not exact-match accuracy or independent factual preservation; legacy identity caller gates credit incorrectly. |
| VQA consensus / exact_any | per answer, then dataset average | `src/evaluation/vqa_official_compatible.py:score_answer` | Mean ten leave-one-out reference agreement scores min(count_other9/3,1); exact_any separately reported. | For nonidentity edits, original-reference agreement is not verified edited-image accuracy. |
| MSE/MAE/SSIM and D | per edited image pair | `src/evaluation/image_quality.py:image_quality` | RGB float32/255; skimage SSIM data_range1, explicit global fallback name if dependency absent; D=clip((MAE+(1-SSIM)/2)/2,0,1). | Image similarity does not validate labels or successful editing. New aligned C4/C5 SSIM uses same256 RGB geometry; original native geometry is separate. |
| LPIPS | per image pair if available | `cached environment availability receipts` | Unavailable; null. No substitute or calibrated weights assumed. | Classifier AlexNet weights alone are not a verified LPIPS metric. |
| C and resource cost | per trajectory proxy versus job/component | `legacy collect num_inference_steps/30; Slurm sacct` | C=1/3 for10 steps; actual UNet calls may be3 per step under sequential CFG. Slurm allocated GPU-seconds and synchronized held-GPU wall time reported separately. | Step proxy is not elapsed GPU compute, active-kernel utilization, query cost or equal real-time budget. |
| Group/batch feedback | per fixed source candidate group / diagnostic batch | `matched_feedback_20261028_v2; audit_remediation CPU receipts` | C0-C4 paired on same source;160 source units; ten16-source diagnostic blocks; repeated seeds do not multiply source count. | 70.625% all-candidate ties; composition/difficulty variation is not automatically policy variation or a valid GRPO group. |
| RSI Gate score | per fixed evaluation set | `src/rsi/controller.py:revise; harness/cost_aware_gate.py:IndependentGate` | Generic controller strict validation gain and cost bound; separate constructed-integrity Gate min24, gain>=.05 or operation saving>=.20 and no new errors. | Different protocols; thresholds are not interchangeable. Neither validates natural-image semantics without independent labels. |
| Clean paired accuracy delta | per dataset and victim-training seed | `existing Qwen clean recheck/statistics` | 3 training seeds on same32 historical sources; paired image bootstrap. Both changing seeds affect one common source. | CI includes0; exact paired sign-flip p1 for one nonzero pair. No generalization or equivalence claim. |
| Post-training ASR / behavior rate | per trained condition/dataset; aggregated victim seeds | `src/evaluation/security_metrics.py:security_metrics; manuscript Algorithm1` | Requires fixed trained victim lineage, predeclared event and explicit denominator on independent assessment. Missing rates stay null. | Not an independently observed per-image reward. No real ASR is fabricated or connected to generator optimization. |

## Historical empirical construct checks

All500 old observations have S=.5*A, constant U1 and C1/3.230/500(46%) hit the lower reward bound; all rewards are negative. Centered5-component rank is2. No-op C1 utility equals original C0 across160 matched sources. These observations do not validate instruction success; their coefficients were not tuned or repaired here.

## Guarded compatibility v2 versus legacy

{
  "status": "PASS",
  "checks": [
    {
      "test": "v2_verified_pixel_identity_consistency_credit",
      "passed": true,
      "observed": 1.4666666666666666,
      "training_eligible": false
    },
    {
      "test": "v2_executed_metadata_mismatch_rejected",
      "passed": true
    },
    {
      "test": "v2_guard_rejects_even_true_receipt_before_optimizer",
      "passed": true
    },
    {
      "test": "v2_duplicate_logical_source_batch_rejected",
      "passed": true
    }
  ],
  "scope": "existing diagnostic v2 interface only; not a released training reward; legacy historical caller remains separate",
  "new_human_labels": 0,
  "new_model_calls": 0,
  "new_GPU_seconds": 0
}

Existing reward_v2 explicitly verifies executed/recorded configuration and correctly credits verified pixel identity; both are diagnostic interface fixes. Legacy historical caller and defaults remain unchanged. The v2 receipt always has training_eligible=False, and require_training_reward rejects even a true flag. A successful bookkeeping check neither validates the reward construct nor releases scientific training.

## Aggregation and exposure

Do not assign dataset-level post-training ASR uniformly to image trajectories without an independently specified credit rule. No such rule is implemented here. Do not equate random-seed repeats with new images or use different-difficulty batches as exchangeable GRPO candidates. Existing real-image feedback/final evaluation has prior exposure; final-test notation in Figure1/Algorithm1 is a protocol requirement, not a completed independent test.

## Final execution and claim boundary

{
  "run_id": "method_code_audit_20261101T0510Z_v1",
  "updated_UTC": "2026-11-01T05:25:12.799177+00:00",
  "scientific_gate": "NO-GO",
  "jobs": [
    {
      "job_id": "88083",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:08:44",
      "end_EDT": "2026-11-01T01:08:55",
      "elapsed_seconds": 11,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "88084",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:10:02",
      "end_EDT": "2026-11-01T01:10:37",
      "elapsed_seconds": 35,
      "gpu_seconds": 35,
      "exit_code": "0:0"
    },
    {
      "job_id": "88085",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:16:53",
      "end_EDT": "2026-11-01T01:16:56",
      "elapsed_seconds": 3,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "88090",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:20:32",
      "end_EDT": "2026-11-01T01:20:34",
      "elapsed_seconds": 2,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    }
  ],
  "budget": {
    "cap": 3600,
    "prior": 2246,
    "new": 35,
    "consumed": 2281,
    "reserved": 0,
    "remaining": 1319,
    "scope": "validation-only subtotal including all earlier failures, not lifetime whole-project cost"
  },
  "local_VJP": "PASS_LOCAL_VJP_ONLY",
  "full_graph_BF16_restart": "NOT VALIDATED",
  "independent_edited_labels": 0,
  "actual_task_learning": "NOT VALIDATED",
  "scientific_improvement": "NOT VALIDATED",
  "four_arm_image_training": "BLOCKED_NOT_LAUNCHED",
  "monitor": {
    "method": "slurm_cpu_successor",
    "last_success_UTC": "2026-11-01T04:32:51.929261+00:00",
    "next_job_id": "88064",
    "next_due_UTC": "2026-11-01T06:32:35+00:00",
    "scheduler_evidence": "88064|PENDING|2026-11-01T01:32:35|poisoning-monitor-v1:f916f24e0de6:ec957b65b56e40c1b1d83b61080d5d4b",
    "duplicate_created": false
  },
  "tensor_integrity_files": 27
}

Direct initial checkpoint contract values are retained in results/method_code_audit_20261101T0510Z_v1/historical_initial_contracts.json. Both seeds initially share parameters, optimizer, scheduler and RNG. Before LR intervention at step4, all256 exp_avg and256 exp_avg_sq tensors differ, although saved RNG, data order, rollout seeds and LR match. This is a confounded historical comparison, not evidence of RSI benefit.

## Retained reporting failures and bounded repair

CPU88093 and88094 failed the final dashboard-contract assertion: legacy Latest HTML used an implicit body and had no embedded JSON. Chinese/English data already matched. The passive renderer now makes that existing body explicit and adds the authoritative JSON/banner; the original payload-equality/unique-banner assertions pass. Evidence: results/method_code_audit_20261101T0510Z_v1/dashboard_data_QA.json. This is an interface repair; no numerical test, reward or threshold was changed.

## Final scheduler receipt and initial contract explanation

Initial contract differences for both seeds are metadata config.arm=rl versusrsi only; initial optimizer/RNG/adapter states match. Step4 optimizer/adapter differences precede LR intervention and remain a scientific confound.

| Job | State | Start EDT | End EDT | Seconds | Exit |
|---|---|---|---|---|---|
|88083|COMPLETED|2026-11-01T01:08:44|2026-11-01T01:08:55|11|0:0|
|88084|COMPLETED|2026-11-01T01:10:02|2026-11-01T01:10:37|35|0:0|
|88085|COMPLETED|2026-11-01T01:16:53|2026-11-01T01:16:56|3|0:0|
|88090|COMPLETED|2026-11-01T01:20:32|2026-11-01T01:20:34|2|0:0|
|88093|FAILED|2026-11-01T01:25:10|2026-11-01T01:25:13|3|1:0|
|88094|FAILED|2026-11-01T01:26:24|2026-11-01T01:26:25|1|1:0|
|88095|COMPLETED|2026-11-01T01:27:24|2026-11-01T01:27:25|1|0:0|

The two failures are reporting-interface jobs only;88095 passes the unchanged HTML/data assertions. All model evidence is unchanged, and no new GPU job is running.

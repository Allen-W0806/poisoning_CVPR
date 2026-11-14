# Next validation decision — 2026-10-31

Evidence snapshot: 2026-11-01T00:41:48.281335+00:00. Run `next_validation_20261101T0040Z_v1`. **NO-GO remains.** No independent human semantic labels or untouched evaluation are available. Engineering checks do not establish scientific efficacy.

## Actual execution and budget

CPU 88013: {'job_id': '88013', 'state': 'COMPLETED', 'elapsed': 8, 'gpu_seconds': 0, 'exit_code': '0:0', 'start_cluster_local': '2026-10-31T20:36:02', 'end_cluster_local': '2026-10-31T20:36:10'}
GPU 88016: {'job_id': '88016', 'state': 'COMPLETED', 'elapsed': 279, 'gpu_seconds': 279, 'exit_code': '0:0', 'start_cluster_local': '2026-10-31T20:36:58', 'end_cluster_local': '2026-10-31T20:41:37'}
Prior validation 1676 + this job 279 = 1955/3600 A100 seconds; remaining 1645. New job hard cap600s. No Qwen optimizer steps. No new monitor. Other project jobs excluded.

## Completion matrix

| Check | Status | Evidence |
|---|---|---|
| Source separation within the 80-source pilot | PASS | results/small_scale_validation_20261031T1928Z_v1/audit/data_integrity.json; unique source/hash and cross-split dHash screen; does not certify historical nonexposure |
| Semantic validity | BLOCKED | results/next_validation_20261101T0040Z_v1/semantic_review32/selection_receipt.json; 0 genuine reviews; no valid/changed rate estimable |
| Image quality metrics | BLOCKED | results/next_validation_20261101T0040Z_v1/aligned_image_quality.csv; aligned SSIM measured for 160; LPIPS implementation absent |
| Pretrained generator numerical reliability | FAIL | results/next_validation_20261101T0040Z_v1_BF16_ordinary/diagnostic/checks.json and instrumented/diagnostic/checks.json; path-dependent failures and historical full batch continuation unresolved |
| Clean target fine-tuning and existing checkpoint recheck | PASS | results/next_validation_20261101T0040Z_v1/qwen_recheck.json; engineering-only; paired accuracy is development evidence |
| Benign actual generator RL learning | PASS | results/next_validation_20261101T0040Z_v1/integrated/checks.json; tiny random UNet actual production DDPO, not pretrained reliability |
| Integrated RSI revision, memory and rollback | PASS | results/next_validation_20261101T0040Z_v1/integrated/results.json; equal-compute fixed control, fixed gate, disk rollback |
| Independent evaluation readiness | BLOCKED | results/next_validation_20261101T0040Z_v1/evaluation_provenance.json; historical exposure unresolved; no untouched test |
| Training-induced behavioral efficacy | NOT TESTED | No backdoor training or ASR optimization; benign changes do not establish this claim |

## Semantic and image-quality audit

Random subset32/160 frozen before metrics with seed20261031032; source manifest and protocol hashes retained. Kit: `results/next_validation_20261101T0040Z_v1/semantic_review32/index.html`. Two independent reviewers and separate adjudication exports required. No reviewers have been contacted or simulated. Full160 kit preserved.
{
  "n_sources": 160,
  "all_pairs_decode_hash_PASS": true,
  "generated_SSIM_mean": 0.7340322949377474,
  "generated_SSIM_range": [
    0.3026932613381966,
    0.94053636185029
  ],
  "generated_MAE_mean": 18.34335912068685,
  "LPIPS": null,
  "cached_metric_modules": {
    "lpips": false,
    "torchmetrics": false,
    "piq": false
  },
  "geometry_interpretation": "SSIM 1 for C5 vs itself is a pipeline control, not native-resolution fidelity or semantic proof"
}

SSIM compares generated C4 with stored geometry-matched original C5. Original native dimensions remain distinct. Geometry C5 versus itself is a control, not a semantic or resolution-fidelity measurement. LPIPS null means missing; no substitute is relabeled LPIPS.

## BF16 diagnostic — ordinary versus instrumented

### ordinary

Status: PASS; execution: `results/next_validation_20261101T0040Z_v1/executions/BF16_ordinary.json`.
{
  "status": "PASS",
  "comparisons": [
    {
      "mode": "matched_checkpoint_enabled",
      "earliest_divergence": null,
      "comparisons": {
        "checkpoint_restore": true,
        "conditioning": true,
        "RNG_seed_draw": true,
        "stochastic_forward": true,
        "decoded_pixels": true,
        "backward_gradients": true,
        "optimizer_parameters": true,
        "optimizer_state": true,
        "scheduler": true,
        "post_step_RNG": true
      },
      "gradient_error": {
        "exact": true,
        "max_abs": 0.0,
        "relative_l2": 0.0
      },
      "frozen_parameters_unchanged": true,
      "behavior_logp_replay_max_difference": 0.0,
      "optimizer_call": 3,
      "interpretation": "locked strict fresh-object stochastic continuation"
    },
    {
      "mode": "checkpoint_disabled_control",
      "earliest_divergence": null,
      "comparisons": {
        "checkpoint_restore": true,
        "conditioning": true,
        "RNG_seed_draw": true,
        "stochastic_forward": true,
        "decoded_pixels": true,
        "backward_gradients": true,
        "optimizer_parameters": true,
        "optimizer_state": true,
        "scheduler": true,
        "post_step_RNG": true
      },
      "gradient_error": {
        "exact": true,
        "max_abs": 0.0,
        "relative_l2": 0.0
      },
      "frozen_parameters_unchanged": true,
      "behavior_logp_replay_max_difference": 0.0,
      "optimizer_call": 4,
      "interpretation": "checkpoint-disabled control deliberately changes execution; not the strict acceptance comparison"
    }
  ],
  "optimizer_calls": 4,
  "scope": "Actual pretrained IP2Pix BF16 / rank8 attention LoRA; two stochastic trajectories from one fixed benign original source, two diffusion steps; constant content-independent coefficient. No target VLM or attack reward.",
  "historical_full_batch10_continuation": "NOT VALIDATED; old backward differences remain preserved; this smaller controlled test does not certify the historical run.",
  "frozen_hash": "eb372d4dafdbafb69e0cc03437bc59406760b5da66f80efacf4587596e782dd9",
  "seconds": 65.0853955745697
}
null
### instrumented

Status: FAIL; execution: `results/next_validation_20261101T0040Z_v1/executions/BF16_instrumented.json`.
{
  "status": "FAIL",
  "comparisons": [
    {
      "mode": "matched_checkpoint_enabled",
      "earliest_divergence": "backward_gradients",
      "comparisons": {
        "checkpoint_restore": true,
        "conditioning": true,
        "RNG_seed_draw": true,
        "stochastic_forward": true,
        "decoded_pixels": true,
        "backward_gradients": false,
        "optimizer_parameters": false,
        "optimizer_state": false,
        "scheduler": true,
        "post_step_RNG": true
      },
      "gradient_error": {
        "exact": false,
        "max_abs": 1.5500118024647236e-05,
        "relative_l2": 0.0001647921355888719
      },
      "frozen_parameters_unchanged": true,
      "behavior_logp_replay_max_difference": 0.0,
      "optimizer_call": 3,
      "interpretation": "locked strict fresh-object stochastic continuation"
    },
    {
      "mode": "checkpoint_disabled_control",
      "earliest_divergence": null,
      "comparisons": {
        "checkpoint_restore": true,
        "conditioning": true,
        "RNG_seed_draw": true,
        "stochastic_forward": true,
        "decoded_pixels": true,
        "backward_gradients": true,
        "optimizer_parameters": true,
        "optimizer_state": true,
        "scheduler": true,
        "post_step_RNG": true
      },
      "gradient_error": {
        "exact": true,
        "max_abs": 0.0,
        "relative_l2": 0.0
      },
      "frozen_parameters_unchanged": true,
      "behavior_logp_replay_max_difference": 0.0,
      "optimizer_call": 4,
      "interpretation": "checkpoint-disabled control deliberately changes execution; not the strict acceptance comparison"
    }
  ],
  "optimizer_calls": 4,
  "scope": "Actual pretrained IP2Pix BF16 / rank8 attention LoRA; two stochastic trajectories from one fixed benign original source, two diffusion steps; constant content-independent coefficient. No target VLM or attack reward.",
  "historical_full_batch10_continuation": "NOT VALIDATED; old backward differences remain preserved; this smaller controlled test does not certify the historical run.",
  "frozen_hash": "eb372d4dafdbafb69e0cc03437bc59406760b5da66f80efacf4587596e782dd9",
  "seconds": 53.79189372062683
}
{
  "earliest_gradient_divergence": "raw_backward_gradients",
  "raw_gradient_error": {
    "exact": false,
    "max_abs": 2.834317274391651e-05,
    "relative_l2": 0.0001648017040489566
  },
  "preclip_norm_reference": 1.8282486200332642,
  "preclip_norm_resumed": 1.8282517194747925,
  "gradient_layout_equal": true,
  "postclip_exact": false,
  "all_instrumentation_is_readonly": true,
  "acceptance_unchanged": "bitwise equality; original failed run retained"
}

One preregistered repetition per path at seed42; bitwise criteria unchanged. Historical86853 ordinary FAIL (postclip maxabs7.6610595e-6, relativeL2 6.3059489e-5) and86856 instrumented PASS both ran on epyc08. This new paired run88016 ran on epyc01: ordinary PASS but instrumented FAIL, reversing the historical pattern. Instrumented raw-gradient first mismatch maxabs2.834317274391651e-5, relativeL2 0.0001648017040489566; preclipnorm1.8282486200332642 versus1.8282517194747925. No discrepancy existed in restored state/conditioning/RNG draws/stochastic forward/decoded pixels; postclip gradients, optimizer parameters/state diverged. This fails reliability and does not reproduce a stable ordinary-fails/instrumented-passes pattern. Node, allocation and lifetime effects are possible confounders; not a causal localization to one kernel. No assertion relaxed and no extra attempts run. Preserve all failed gradients/checkpoints.

## Existing clean checkpoints — development only

[
  {
    "seed": null,
    "n": 32,
    "accuracy": 0.7843749999999999,
    "paired_difference": 0.0,
    "paired_ci95": [
      0.0,
      0.0
    ],
    "changed_answers": 0,
    "fresh_object_repeat_exact": true,
    "mean_latency": 0.26470065202738624,
    "scope": "historical development only"
  },
  {
    "seed": 42,
    "n": 32,
    "accuracy": 0.7843749999999999,
    "paired_difference": 0.0,
    "paired_ci95": [
      0.0,
      0.0
    ],
    "changed_answers": 0,
    "fresh_object_repeat_exact": true,
    "mean_latency": 0.31916784221539274,
    "scope": "historical development only"
  },
  {
    "seed": 123,
    "n": 32,
    "accuracy": 0.8062499999999999,
    "paired_difference": 0.021875,
    "paired_ci95": [
      0.0,
      0.06562499999999999
    ],
    "changed_answers": 1,
    "fresh_object_repeat_exact": true,
    "mean_latency": 0.3188282387036452,
    "scope": "historical development only"
  },
  {
    "seed": 3407,
    "n": 32,
    "accuracy": 0.8062499999999999,
    "paired_difference": 0.021875,
    "paired_ci95": [
      0.0,
      0.06562499999999999
    ],
    "changed_answers": 1,
    "fresh_object_repeat_exact": true,
    "mean_latency": 0.3191470800265961,
    "scope": "historical development only"
  }
]

Pinned Qwen revision89644892e4d85e24eaac8bacfd4f463576704203; greedy decoding, original references, fixed pixel budgets/math SDPA. Fresh model loads repeated twice per adapter. Checkpoint inventory verified before use; inference frozen. Paired image bootstrap5000 resamples with fixedseed20261031; zero remains compatible with the observed differences, and empirical zero intervals do not prove equivalence. Repeated seeds changing the same image do not multiply independent sample size. Historical split exposure remains unresolved; no independent test claim.

## Integrated benign RL–RSI

{
  "status": "PASS",
  "decision": {
    "revision_number": 1,
    "timestamp": "2026-11-01T00:39:43.648447+00:00",
    "previous_config": {
      "batch_size": 10,
      "learning_rate": 1e-05,
      "curriculum": "uniform"
    },
    "proposed_config": {
      "batch_size": 10,
      "learning_rate": 2e-05,
      "curriculum": "uniform"
    },
    "validation_hash": "117c4a5ea7d968128229bb9c777902e28da8cae7d20cce2c6d161c0f8e2dc49e",
    "validation_score": 0.0832411574665457,
    "cost_gpu_hours": 0.008662270762223569,
    "accepted": true,
    "reason": "accepted",
    "proposed_policy_version": "rsi_candidate_v1",
    "proposal_reason": "LR_PROPOSAL_0001",
    "evidence": "integrated/results.json",
    "rollback": "restore fixed/final when candidate rejected"
  },
  "equality_rejection": {
    "revision_number": 2,
    "timestamp": "2026-11-01T00:39:43.677975+00:00",
    "previous_config": {
      "batch_size": 10,
      "learning_rate": 2e-05,
      "curriculum": "uniform"
    },
    "proposed_config": {
      "batch_size": 10,
      "learning_rate": 2e-05,
      "curriculum": "uniform"
    },
    "validation_hash": "117c4a5ea7d968128229bb9c777902e28da8cae7d20cce2c6d161c0f8e2dc49e",
    "validation_score": 0.0832411574665457,
    "cost_gpu_hours": 0.0,
    "accepted": false,
    "reason": "no_validation_gain",
    "proposed_policy_version": "rsi_candidate_v1",
    "proposal_reason": "locked equality rejection check using cached selected score",
    "evidence": null,
    "rollback": "selected checkpoint remains incumbent"
  },
  "selected": "rsi",
  "paired_gate_difference": 3.591412678360939e-06,
  "paired_bootstrap95": [
    -9.631318971514701e-06,
    1.6934884479269377e-05
  ],
  "memory_persistence": true,
  "actual_rollback": true,
  "scope": "single tiny random UNet seed; 16 synthetic validation conditions; selected on this gate, no untouched test, no autonomous scientific efficacy claim",
  "selection_cost_GPU_hours": 0.008662270762223569,
  "seconds": 31.25424368295353
}
{
  "fixed": {
    "gate_mean": 0.08323756605386734,
    "cost": {
      "training_updates": 4,
      "recovery_reference_updates": 1,
      "recovery_replay_updates": 1,
      "actual_optimizer_calls": 6,
      "training_trajectories": 40,
      "recovery_trajectories": 20,
      "gate_trajectories": 16,
      "forward_NFE": 272,
      "backward_transition_calls": 120
    },
    "trainable_tensors": 64,
    "trainable_elements": 4096,
    "parameter_changes": 64,
    "final_checkpoint_sha256": "c895c0caa1fabb4c5be3ea6e798594f3982b55b42a304f79847551958ca43c60"
  },
  "rsi": {
    "gate_mean": 0.0832411574665457,
    "cost": {
      "training_updates": 4,
      "recovery_reference_updates": 1,
      "recovery_replay_updates": 1,
      "actual_optimizer_calls": 6,
      "training_trajectories": 40,
      "recovery_trajectories": 20,
      "gate_trajectories": 16,
      "forward_NFE": 272,
      "backward_transition_calls": 120
    },
    "trainable_tensors": 64,
    "trainable_elements": 4096,
    "parameter_changes": 64,
    "final_checkpoint_sha256": "e4ab1e8edcb742338fd2ba2f0923915b95e4962d12b50d9902e68d90d80ae22c"
  }
}

Production algorithm is single-pass on-policy DDPO_SF_v1, not GRPO/PPO. Tiny randomly initialized UNet attention LoRA rank2alpha4 only; fixedcos terminal reward. Four logical steps and two extra recovery/reference optimizer calls per arm are all charged. Common initialization/training seeds; one LR proposal1e-5→2e-5; 16 common synthetic validation conditions. Exact NFE/update parity does not mean equal wall-clock latency. Selected gate is development evidence, not a fresh test. Actual disk rollback restores weights, optimizer and CPU/CUDA RNG; memory has persisted decisions. Any engineering acceptance is not statistical/autonomous research benefit.

## Monitoring and source integrity

Existing `slurm_cpu_successor`, enabled=True; last successful tick=2026-11-01T00:32:52.066830+00:00; next=88008 eligible2026-11-01T02:32:35+00:00. Actual sacct:
```
87971|COMPLETED|2026-10-31T20:32:48|2026-10-31T20:32:52|0:0
88008|PENDING|Unknown|Unknown|0:0

```
The original stage-specific heartbeat still reports its old1344-second subtotal; use this report/validation_gpu_reservations.json for aggregate budget. New phase artifacts/logs have their own immutable ledger. No duplicate recurring monitor installed.
Reward guard unchanged=True; exact config/source hashes: `results/next_validation_20261101T0040Z_v1/source_manifest.json`. Frozen executed source retained under audit/executed_source. Cached environment reused, no rebuild/download.

## Failures and next decision

A first CPU submission was rejected because intel-cpu is not a valid partition; no Job ID/GPU time was allocated. The error receipt is preserved. Same CPU work succeeded on epyc-cpu. All numerical failures and successful witnesses remain in their original unique directories.

Next scientific priority: obtain two real independent semantic reviews and adjudication for the predeclared32; extend to remaining128 without outcome-based selection. Establish verifiable private evaluation exposure before efficacy claims. Localize ordinary-path backward divergence with a preregistered diagnostic before pretrained generator training. No further GPU job is authorized by this report, and no attack-driven optimization is enabled.

Google Drive publication contains separate deidentified prose and aggregates only; raw questions/answers/images/source IDs, host paths, models, checkpoints and logs remain local/cluster. Publication readback receipt is separate.

## Delivery verification

CPU88013, GPU88016 and one-shot finalizer88019 are COMPLETED0:0. Last live check20:44 EDT October31: no active Poisoning experiment; existing monitor88008 PENDING BeginTime, eligible22:32:35 EDT. Other account GPU jobs belong to unrelated projects and were not modified. Last successful monitor87971 finished20:32:52 EDT. Cluster cron is absent and user systemd bus unavailable; existing native Slurm successor chain remains in use.

New BF16 checkpoint/output inventories: 32 files rehashed successfully, including saved negative-gradient evidence. All160 C5 geometry-control hashes match the frozen candidate manifest. Pinned editor31519b5cb02a7fd89b906d88731cd4d6a7bbf88d; Qwen89644892e4d85e24eaac8bacfd4f463576704203. QA: reports/next_validation_browser_20261031/QA.json; bilingual desktop and mobile payload equality and32-pair image loading PASS;0 human labels entered. New phase retained about303MB of cluster evidence and about13MB of Windows metadata/review kit, plus QA screenshots; no new pretrained weights or multi-GB C: artifacts. QA run-specific scratch was verified and removed.

Only the new deidentified summary and aggregate CSV were uploaded to the existing RSI-Poisoning reports folder. Remote connector readback matched both byte hashes and folder parents. Receipt: reports/deidentified_next_validation_20261031/PUBLICATION_VERIFIED.json. The older scheduled report-sync package is stage-specific and still describes its prior completed stage; it was not repurposed or overwritten.

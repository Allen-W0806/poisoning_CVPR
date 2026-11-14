# Small-scale validation: execution and next decision

Actual snapshot 2026-11-01T01:28:15.514837+00:00; NO-GO for scientific natural-image training. Eligible benign synthetic engineering work was executed.

## Actual jobs and resource cost

[
  {
    "job_id": "88029",
    "state": "COMPLETED",
    "elapsed_seconds": 147,
    "gpu_seconds": 147,
    "exit_code": "0:0",
    "start_EDT": "2026-10-31T21:18:48",
    "end_EDT": "2026-10-31T21:21:15",
    "node": "epyc01"
  },
  {
    "job_id": "88030",
    "state": "COMPLETED",
    "elapsed_seconds": 9,
    "gpu_seconds": 0,
    "exit_code": "0:0",
    "start_EDT": "2026-10-31T21:21:37",
    "end_EDT": "2026-10-31T21:21:46",
    "node": "epyc16"
  },
  {
    "job_id": "88032",
    "state": "COMPLETED",
    "elapsed_seconds": 144,
    "gpu_seconds": 144,
    "exit_code": "0:0",
    "start_EDT": "2026-10-31T21:22:42",
    "end_EDT": "2026-10-31T21:25:06",
    "node": "epyc01"
  }
]
{
  "cap": 3600,
  "prior": 1955,
  "new": 291,
  "consumed": 2246,
  "reserved": 0,
  "remaining": 1354,
  "scope": "bounded validation subtotal only; includes failed child tests and failed first three-way attempt; not whole-project lifetime"
}

Job completion is separate from child assertion status. All failed attempts count. Slurm GPU-seconds include model loading, diagnostics, writes, rejected trials and idle overhead while allocated. Synchronized component GPU-held wall seconds count one held GPU; they are not active-kernel utilization. Configured steps alone are not cost.

## Preregistered three-way design

{
  "run_id": "audit_remediation_20261101T0120Z_v1_threeway_repair_v1",
  "created_UTC": "2026-11-01T01:18:18.347042+00:00",
  "gpu_cap_seconds": 360,
  "global_validation_cap_seconds": 3600,
  "training_seeds": [
    42,
    123,
    3407
  ],
  "arms": [
    "fixed",
    "adaptive_scheduler",
    "rsi"
  ],
  "objective": "same benign terminal cos(first latent coordinate) as previous tiny test",
  "pre_adaptation_steps": 2,
  "post_adaptation_steps": 2,
  "logical_batch_size": 10,
  "diffusion_steps": 2,
  "initial_lr": 1e-05,
  "ordinary_scheduler": "ReduceLROnPlateau max, factor .5, patience0, threshold0; initial and preadaptation validation",
  "RSI_proposal": "cooldown .5 on no initial-to-prefix gain, else LR2x; candidate accepted iff validation mean strictly improves own prefix and cost gate passes; rejection restores OWN prefix",
  "validation_conditions": {
    "seed": 100000,
    "n": 16
  },
  "untouched_synthetic_test_conditions": {
    "seed": 200000,
    "n": 32
  },
  "per_arm_evaluation_queries": 80,
  "per_arm_actual_forward_NFE": 320,
  "per_arm_optimizer_calls": 4,
  "common_initial_full_state_and_preadaptation_full_state_must_be_bitwise_equal": true,
  "bootstrap": "5000 crossed resamples of3 training seeds and32 shared synthetic conditions, seed20261101",
  "BF16": {
    "repetitions_per_path": 1,
    "paths": [
      "ordinary",
      "instrumented"
    ],
    "seed": 42,
    "acceptance": "existing bitwise assertions unchanged; first mismatch preserved; no run-until-pass"
  },
  "human_review": "reuse existing predeclared32,0 genuine labels",
  "source_guard_sha256": "73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712",
  "no_VLM_reward": true,
  "no_attack_metrics": true,
  "parent_run": "audit_remediation_20261101T0120Z_v1",
  "rerun_scope": "threeway only after serialization failure; no new BF16 attempts",
  "engineering_change": "mode_worse negative infinity sentinel encoded as a string only; gradients/rewards remain strict finite and objective/criteria/seeds unchanged"
}

| Seed | Arm | Selected synthetic test mean | Proposed LR | Retained LR | Updates | Forward NFE | Eval queries | Measured held-GPU s |
|---|---|---|---|---|---|---|---|
| 42 | fixed | 0.1571662167 | 1e-05 | 1e-05 | 4 | 320 | 80 | 15.650035 |
| 42 | adaptive_scheduler | 0.1571662167 | 1e-05 | 1e-05 | 4 | 320 | 80 | 15.025347 |
| 42 | rsi | 0.1571633513 | 2e-05 | 1e-05 | 4 | 320 | 80 | 15.045673 |
| 123 | fixed | 0.0839195922 | 1e-05 | 1e-05 | 4 | 320 | 80 | 15.008279 |
| 123 | adaptive_scheduler | 0.0839116753 | 5e-06 | 5e-06 | 4 | 320 | 80 | 15.014150 |
| 123 | rsi | 0.0839116753 | 5e-06 | 5e-06 | 4 | 320 | 80 | 15.042912 |
| 3407 | fixed | 0.1394203220 | 1e-05 | 1e-05 | 4 | 320 | 80 | 15.015150 |
| 3407 | adaptive_scheduler | 0.1394203220 | 1e-05 | 1e-05 | 4 | 320 | 80 | 15.013946 |
| 3407 | rsi | 0.1394269639 | 2e-05 | 2e-05 | 4 | 320 | 80 | 15.027921 |

Three independent training initializations42/123/3407. Every branch independently replays two identical prefix updates. Full parameters, adapter/optimizer moments, scheduler, CPU/CUDA/NumPy/Python RNG, contract/version/consumed IDs, last gradients and validation predictions are compared online before adaptation. Twelve independent saved initial/prefix state comparisons across the six nonfixed branches were also performed. See audit/preadaptation_disk_verification.json. Equality failure aborts the experiment.

Fixed keeps1e-5. Ordinary ReduceLROnPlateau sees the same initial/prefix validation, cooling by.5 if no gain. RSI proposes.5 on no gain or2x on improvement; candidate acceptance requires a strictly higher mean than its OWN prefix. Rejection restores its OWN checkpoint and configuration. It never borrows fixed-control weights/results. All candidate work is charged even when rejected. This is a bounded controller comparison, not a novel autonomous method demonstration.

All arms:4 optimizer calls;80 backward transition calls;320 actual forward calls;48 validation and32 held-out synthetic test queries. Test conditions are first evaluated after selection; they are distinct from all training/validation noise IDs. Equal query/update budgets do not imply identical wall time. Sequential same-node timing has cache/order confounds and cannot establish speedup.

[
  {
    "comparison": "rsi_minus_fixed",
    "mean": -1.3801618479192257e-06,
    "per_seed_means": [
      -2.86538852378726e-06,
      -7.916940376162529e-06,
      6.641843356192112e-06
    ],
    "crossed_seed_condition_bootstrap95": [
      -1.1316767389265198e-05,
      8.26225732453165e-06
    ],
    "independent_training_seeds": 3,
    "shared_test_conditions": 32,
    "scope": "synthetic held-out noise, conditional on tiny architecture; only three training seeds, exploratory"
  },
  {
    "comparison": "rsi_minus_adaptive_scheduler",
    "mean": 1.258818277468284e-06,
    "per_seed_means": [
      -2.86538852378726e-06,
      0.0,
      6.641843356192112e-06
    ],
    "crossed_seed_condition_bootstrap95": [
      -5.2800169214606285e-06,
      9.22750623431055e-06
    ],
    "independent_training_seeds": 3,
    "shared_test_conditions": 32,
    "scope": "synthetic held-out noise, conditional on tiny architecture; only three training seeds, exploratory"
  }
]

Crossed bootstrap resamples3 training seeds and32 shared synthetic test conditions jointly across methods5000 times. It avoids treating96 correlated pairs as96 independent images. With only3 training seeds, intervals remain exploratory and do not address natural-image generalization.

## BF16 diagnosis

- replication_monitor_20261029_v1_ip2pix_a0: first divergence `backward_gradients`; error {'exact': False, 'max_abs': 7.661059498786926e-06, 'relative_l2': 6.305948877976716e-05}; raw localization None.
- replication_monitor_20261029_v1_ip2pix_a1: first divergence `None`; error {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}; raw localization {'earliest_gradient_divergence': None, 'raw_gradient_error': {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}, 'preclip_norm_reference': 1.834847331047058, 'preclip_norm_resumed': 1.834847331047058, 'gradient_layout_equal': True, 'postclip_exact': True, 'all_instrumentation_is_readonly': True, 'acceptance_unchanged': 'bitwise equality; original failed run retained'}.
- next_validation_20261101T0040Z_v1_BF16_ordinary: first divergence `None`; error {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}; raw localization None.
- next_validation_20261101T0040Z_v1_BF16_instrumented: first divergence `backward_gradients`; error {'exact': False, 'max_abs': 1.5500118024647236e-05, 'relative_l2': 0.0001647921355888719}; raw localization {'earliest_gradient_divergence': 'raw_backward_gradients', 'raw_gradient_error': {'exact': False, 'max_abs': 2.834317274391651e-05, 'relative_l2': 0.0001648017040489566}, 'preclip_norm_reference': 1.8282486200332642, 'preclip_norm_resumed': 1.8282517194747925, 'gradient_layout_equal': True, 'postclip_exact': False, 'all_instrumentation_is_readonly': True, 'acceptance_unchanged': 'bitwise equality; original failed run retained'}.
- audit_remediation_20261101T0120Z_v1_BF16_ordinary: first divergence `None`; error {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}; raw localization None.
- audit_remediation_20261101T0120Z_v1_BF16_instrumented: first divergence `None`; error {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}; raw localization {'earliest_gradient_divergence': None, 'raw_gradient_error': {'exact': True, 'max_abs': 0.0, 'relative_l2': 0.0}, 'preclip_norm_reference': 1.8282486200332642, 'preclip_norm_resumed': 1.8282486200332642, 'gradient_layout_equal': True, 'postclip_exact': True, 'all_instrumentation_is_readonly': True, 'acceptance_unchanged': 'bitwise equality; original failed run retained'}.

Historical ordinary FAIL/instrumented PASS was later reversed on88016; the new88029 pair both passes on epyc01. One small fixed repetition per path is not evidence that intermittent instability is fixed. The earliest preserved failed comparison is raw/postclip backward after exact forward/RNG restoration; no unique primitive has been identified. No numerical tolerance or assertion was relaxed, and no extra BF16 retries were launched.

## Next actions and blocked work

1. Obtain two genuine independent reviews of the existing32-pair kit, lock original labels separately, and adjudicate disagreements without target predictions. Do not fabricate labels or estimate rates from blank forms.
2. Freeze an actually unused/private real-image assessment with provenance; current records remain historical development. Do not reopen old test data to tune.
3. Preregister a primitive-level BF16 backward diagnostic only if a later budgeted stage is authorized; retain current first-divergence snapshots. No pretrained scientific training cleared.
4. Before another objective protocol, resolve legacy identity credit and explicit coefficient wiring through versioned contracts. No reward search or ASR-driven loop.
5. No further GPU job is submitted by these reports. Existing two-hour monitor retained without duplication.

Evidence roots: results/audit_remediation_20261101T0120Z_v1 (CPU audit, BF16, failed three-way, source archives); results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1 (repaired three-way). All checksums/checkpoints/failures retained.

## Final delivery verification

Jobs88029/88030/88032/88034 are COMPLETED0:0. Nested first comparison serialization failure is retained. Both new BF16 diagnostic inventories and all18 selected/prefix three-way checkpoints were rehashed: 50 files PASS. Twelve disk initial/prefix equality comparisons PASS. Repair component held-GPU time sums to135.84341243002564s out of144 allocated seconds; remaining8.156587569974363s is startup/cleanup/bookkeeping. Two GPU allocations147+144=291s; cumulative2246/3600, remaining1354, reserved0.

Browser QA: reports/audit_remediation_browser_20261101/QA.json. Chinese/English desktop and390px mobile: same embedded authoritative payload,23 findings+9 branch rows, no JS errors or overflow. Existing32-pair review kit loads with blank labels and0 saved reviews; no human labels entered. QA scratch under project tmp was verified and removed.

Existing monitor remains single Slurm CPU successor: next job88008 eligible2026-11-01T02:32:35+00:00. No duplicate monitor or further GPU job installed. Original snapshots, historical matrix, failures, checkpoints and code versions remain intact.

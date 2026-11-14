# RL–RSI controlled validation

Run `method_code_audit_20261101T0510Z_v1`, snapshot 2026-11-01T05:16:56.482704+00:00.

## Actually executed in this stage

[
  {
    "job_id": "88083",
    "state": "COMPLETED",
    "elapsed_seconds": 11,
    "gpu_seconds": 0,
    "exit_code": "0:0",
    "start_cluster_local": "2026-11-01T01:08:44",
    "end_cluster_local": "2026-11-01T01:08:55",
    "node": "epyc16"
  },
  {
    "job_id": "88084",
    "state": "COMPLETED",
    "elapsed_seconds": 35,
    "gpu_seconds": 35,
    "exit_code": "0:0",
    "start_cluster_local": "2026-11-01T01:10:02",
    "end_cluster_local": "2026-11-01T01:10:37",
    "node": "epyc11"
  }
]
{
  "cap": 3600,
  "prior": 2246,
  "new": 35,
  "consumed": 2281,
  "reserved": 0,
  "remaining": 1319,
  "scope": "validation-only subtotal including all earlier failures, not lifetime whole-project cost"
}

CPU88083: named raw-gradient forensic mapping, exact optimizer inventory, official original-question/answer retention1040 rows, data-version overlap and review-status audit. A distinct subsequent CPU verification checks v2 diagnostic credit/configuration/guard contracts and rehashes actual dataset bytes. No Qwen inference or fine-tuning is performed.

GPU88084: one bounded A100 local-VJP diagnostic from a preserved failed checkpoint, constant content-independent coefficient1, no target VLM/security metrics. It uses existing core numerical-test entry points; the Phase5 scientific guard is untouched and no new training reward is released.

{
  "status": "PASS_LOCAL_VJP_ONLY",
  "selected_parameter": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
  "selected_module": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
  "captures": {
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default": 12,
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default": 12
  },
  "parameter_hook_contributions": {
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default": 4,
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default": 4
  },
  "isolated_VJP_calls": 72,
  "optimizer_updates": 1,
  "update": {
    "step": 2,
    "loss": -2.1162105202674866,
    "reward_mean": 1.0,
    "reward_variance": 0.0,
    "individual_rewards": [
      1.0,
      1.0
    ],
    "gradient_norm": 1.8282486200332642,
    "finite_gradients": true,
    "probabilities_finite": true,
    "behavior_logp_replay_max_difference": 0.0,
    "conditioning_verified": true,
    "logical_batch_size": 2,
    "physical_micro_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "backward_transition_calls": 4,
    "optimizer_updates": 1,
    "changes": {
      "max_abs_change": 1.0013571227318607e-05,
      "mean_abs_change": 5.929565599565257e-06,
      "updated_tensors": 256,
      "adapter_tensor_count": 256
    },
    "trainable_max_change": 1.0013571227318607e-05,
    "step_seconds": 7.981272341683507,
    "peak_gpu_memory_bytes": 4947492864,
    "algorithm": "DDPO_SF_v1"
  },
  "maximum_FP64_CPU_relative_error": 2.379282048123678e-07,
  "earliest_local_repeat_mismatch": null,
  "ordinary_full_graph_reproducibility": "NOT VALIDATED; captures synchronize and alter lifetimes; local PASS cannot close historical failures",
  "CPU_FP64_reference": "analytical g.T @ x; descriptive floating-point comparison, not a changed acceptance threshold",
  "device": "NVIDIA A100 80GB PCIe",
  "seconds": 28.681606197729707
}

## What is localized

{
  "source": "results/next_validation_20261101T0040Z_v1_BF16_instrumented/diagnostic",
  "source_sha256": "d7af390cad13036adf3440533e02938368a9d80056a9c586fa89605ad204aa92",
  "first_in_parameter_order": {
    "name": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "shape": [
      8,
      320
    ],
    "exact": false,
    "nonfinite": false,
    "different_elements": 2560,
    "max_abs": 2.629860773595283e-10,
    "relative_L2": 0.00033405186547753906
  },
  "temporal_kernel_order": "NOT DETERMINED by parameter registration order",
  "different_tensors": 224,
  "raw_before_clipping": true,
  "gradient_layout_equal": true,
  "clip_norms": [
    1.8282486200332642,
    1.8282517194747925
  ],
  "all_finite": true,
  "interpretation": "raw mismatch excludes clipping as sole origin; no unique GEMM/attention/accumulation cause follows"
}

224/256 old raw gradient tensors differ before clipping. The first registered parameter is not necessarily the first temporal operation. Its local A/B linears generated12 activation/upstream pairs each; each was independently replayed3 times. All72 CUDA VJPs are exact across repeats, and their FP64 analytical CPU references differ by at most2.379282048123678e-7 relative error (descriptive, not a changed threshold). This narrows the observed local computation, but does not exclude attention/upstream BF16 gradients, fan-in accumulation, allocation/lifetime effects or the full graph. Captures synchronize and alter lifetimes. No primitive is asserted as the unique root cause.

One constant-coefficient diagnostic update changed256 adapter tensors by max1.0013571e-5; frozen base parameters did not change. Its reward variance is0 by design. For a constant terminal coefficient the expected score-function update is zero; finite-sample parameter movement is not evidence of meaningful image-edit learning or reward discrimination. Updated adapter/optimizer/RNG state and all probe inputs/gradients are retained.

## Existing matched controls — different scope

The prior tiny random-UNet cosine task has3 training seeds×fixed/ordinary ReduceLROnPlateau/RSI,4 updates and80 evaluation queries per branch,320 actual forward calls and80 backward transitions. Initial/prefix full state and predictions match;12 saved checkpoint cross-comparisons pass. Its two RSI-minus-control intervals include0. This is real tiny-model optimization and engineering control evidence, not a harmless natural-image editing validation or a frozen-generator four-arm study. No frozen arm is invented retrospectively.

## Conditional four-arm image-edit validation

**NOT LAUNCHED / BLOCKED.** The prerequisite is ordinary pretrained full-graph exact backward/continuation, not local instrumented VJP equality. Independent edited-image task/label validity is also unavailable, and an unused real-image final assessment is uncertified. Therefore this run does not progress into frozen-generation/DDPO-only/ordinary-scheduler/RSI image-edit learning. No alternative scientific loss or surrogate task is silently substituted.

When eligible, all four predefined arms must share original sources, instruction, questions/labels, initial generator/optimizer/RNG, preadaptation trajectory/data order, fixed protocol and measured budget; the frozen arm has zero generator updates and receives the same evaluation cases. Adaptation uses development feedback only; locked final assessment is report-only. Do not optimize post-training security outcomes, attack cost or stealth. No such future job is submitted by this report.

## Direct historical checkpoint/RNG follow-up

A distinct read-only CPU task compared the preserved initial and pre-LR-step4 checkpoints directly. Evidence: results/method_code_audit_20261101T0510Z_v1/historical_prefix_state_comparison.json. Matching data order and RNG do not imply matched model/optimizer trajectories.

{
  "status": "COMPLETE_HISTORICAL_DIFFERENCES_PRESERVED",
  "rows": [
    {
      "seed": 42,
      "step": 0,
      "baseline": "EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": true,
        "optimizer": true,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 0.0,
      "different_optimizer_moment_tensors": {
        "exp_avg": 0,
        "exp_avg_sq": 0
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL/checkpoints/step_0000/training_state.pt": "6b2446ae5b14b4033a493beafc6d661e940cfacb990d563c72ef35a0c45bf31c",
        "experiments/EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI/checkpoints/step_0000/training_state.pt": "d47bbb1a322743d0393dcd2c06590dc3eeacdd232898f0b99e4bdcbfa6e14b8d"
      }
    },
    {
      "seed": 42,
      "step": 4,
      "baseline": "EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": false,
        "optimizer": false,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 4.343278851592913e-05,
      "different_optimizer_moment_tensors": {
        "exp_avg": 256,
        "exp_avg_sq": 256
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL/checkpoints/step_0004/training_state.pt": "18c82539a230ddcbe5d82028ac69b903052bfded9d77c21c381719e3768162bc",
        "experiments/EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI/checkpoints/step_0004/training_state.pt": "2a77e297fdd793c6c1456f1a76201c412c250ba1f530c99e21a83bcd68ad2348"
      }
    },
    {
      "seed": 123,
      "step": 0,
      "baseline": "EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": true,
        "optimizer": true,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 0.0,
      "different_optimizer_moment_tensors": {
        "exp_avg": 0,
        "exp_avg_sq": 0
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL/checkpoints/step_0000/training_state.pt": "5d6fcefab465d7d0e275eb5497425dff804438da1f1056b56c9f796e7e0f6cdc",
        "experiments/EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI/checkpoints/step_0000/training_state.pt": "e15d426b3f7b375e4990e140ea7c1082cca9e2b44158cb7e782f0d4f5f32be1e"
      }
    },
    {
      "seed": 123,
      "step": 4,
      "baseline": "EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": false,
        "optimizer": false,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 6.645589019171894e-05,
      "different_optimizer_moment_tensors": {
        "exp_avg": 256,
        "exp_avg_sq": 256
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL/checkpoints/step_0004/training_state.pt": "de7023da79fe70f116974b30ad3a20d0c28c636db283c3bcdf7dedfdbd7f6ddb",
        "experiments/EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI/checkpoints/step_0004/training_state.pt": "1f3d1a0b886b47f86d392e936fa6c94a21ae669e717c7d26e2ae0a95cb33e055"
      }
    }
  ],
  "scope": "saved checkpoints step0 and step4 before LR intervention; matching RNG/order is not proof of equal backward execution"
}

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

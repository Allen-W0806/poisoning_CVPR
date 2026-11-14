# Real-observation RSI validation

Feedback evidence: `EXP_20261027T053043_7b4b5ce3fe61_phase2_real_editor_feedback`; controller/RSI experiment: `EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation`.

Shared feedback construction measured 60 edited observations, 140 Qwen calls, and 82.570s editor generation. Human-verified changed edits: 0. All model loading and judging time is included in the enclosing Slurm allocation, beyond the generation subtotal.

The real pretrained editor is frozen. Shared table contains 10 training and 10 validation source images ×3 benign instructions, fixed source/seed/action mapping. Qwen answer/structured proxies and quality/resource measurements are saved per edited image. Training uses only train rows; selection uses only validation; no test rows are opened by this code.

Fixed RL: six logical-batch-10 updates, validation after updates 3/6. RSI: three-update base plus three-update candidate from common initial state, two validation lookups total. Both have 60 gradient sample exposures and two validation lookups. This matches nominal operation/data budgets; **actual wall/GPU time is not forced equal** and is reported. Shared generation/verification cost belongs to the comparison infrastructure and is charged once, not hidden in optimizer timings.

Revision rule: propose 5e-5 only when observed sampled reward variance is positive and all gradient norms <1, otherwise retain 1e-5; strict expected-validation-reward gain within budget accepts. Every proposal preserves old/new config, reason, raw evidence, cost, decision and rollback checkpoint. This is controlled configuration revision, not code self-rewriting.

Observed comparison:
```json
{
  "frozen_editor_action0_validation_reward": -0.5261084448794524,
  "random_policy_validation_reward": -0.4320706140995026,
  "fixed_rl_validation": [
    -0.46183091402053833,
    -0.4618317782878876
  ],
  "rsi_base_validation": -0.46183091402053833,
  "rsi_candidate_validation": -0.46183642745018005,
  "rsi_selected_validation": -0.46183091402053833,
  "fixed_rl_seconds": 0.48461304511874914,
  "rsi_seconds": 0.40818353509530425
}
```

The recorded random baseline uses one seed-42 action per validation image (ten draws), whereas trained-policy validation integrates all three actions' probabilities. To expose this sampling difference, a read-only derived table audit also reports the exact uniform-random expectation; it does not replace the original run:
```json
{
  "experiment_id": "EXP_20261027T053043_7b4b5ce3fe61_phase2_real_editor_feedback",
  "records": 60,
  "random_uniform_expected_validation_reward": -0.4628590846020314,
  "automatic_provisional_agreements": 30,
  "independently_verified_changed_edits": 0,
  "note": "Derived read-only from saved feedback; original single-draw random baseline and immutable run metrics retained"
}
```

Revision:
```json
{
  "previous_config": {
    "batch_size": 10,
    "learning_rate": 1e-05
  },
  "proposed_config": {
    "batch_size": 10,
    "learning_rate": 5e-05
  },
  "validation_score": -0.46183642745018005,
  "accepted": false,
  "reason": "no_validation_gain",
  "cost_gpu_hours": 5.6133556904064286e-05,
  "proposal_reason": "Use 5e-5 only if real sampled reward variance is positive and every gradient norm <1; otherwise retain 1e-5; accept only strict held-out validation gain",
  "rollback": {
    "previous_policy": "controller_lr1e5",
    "checkpoint": "checkpoints/base_controller.pt"
  }
}
```

Raw sample rewards, variance, gradient norms, parameter deltas, time and memory are retained for all updates. This small fixed-table automatic-proxy experiment is preliminary and does not establish RSI superiority, improved natural-image semantics, generator learning by RL, or robust generalization. Budget metadata includes actual measured selection overhead and all scientific jobs are charged to the phase ledger.

Observed per-update diagnostic ranges (all twelve updates across arms):
```json
{
  "reward_variance": {
    "min": 0.21591825783252716,
    "max": 0.2253696173429489
  },
  "gradient_norm": {
    "min": 0.1055053174495697,
    "max": 0.321592777967453
  },
  "trainable_max_change": {
    "min": 1.004338264465332e-05,
    "max": 0.0001481175422668457
  },
  "step_seconds": {
    "min": 0.04924934799782932,
    "max": 0.1279380910564214
  },
  "peak_gpu_memory_bytes": {
    "min": 67122176,
    "max": 68183552
  }
}
```

Separate CPU reload audit of the saved selection and rollback artifacts:
```json
{
  "experiment_id": "EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation",
  "loaded_from_disk": true,
  "selected": "base",
  "selected_policy_and_optimizer_equal": true,
  "candidate_policy_differs_from_base": true,
  "configuration_matches_selection": true,
  "rejected_trial_cost_retained": true
}
```

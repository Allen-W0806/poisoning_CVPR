# Failures and numerical limits

## EXP_20261027T141314_7f2c3dc26636_phase3_slurm_one

Slurm `None`; source `00950716195e20f9a08b2175b75d16221e1a231d`.

```json
{
  "restart": null,
  "inputs": null,
  "real_updates_before_failure": null
}
```
## EXP_20261027T141335_bc288d994eca_phase3_generator_rl_one

Slurm `86403`; source `00950716195e20f9a08b2175b75d16221e1a231d`.

```json
{
  "restart": {
    "fresh_model": true,
    "adapter_exact": false,
    "optimizer_exact": false,
    "scheduler_exact": true,
    "rng_exact": true,
    "next_step_loss_exact": true,
    "rollout_seeds_exact": true,
    "reward_components_exact": true,
    "adapter_max_difference": 8.372412594326306e-07
  },
  "inputs": null,
  "real_updates_before_failure": [
    {
      "step": 1,
      "loss": -0.014984687557443976,
      "reward_mean": -0.08180490632851922,
      "reward_variance": 0.0,
      "individual_rewards": [
        -0.08180490632851922
      ],
      "gradient_norm": 0.0063325329683721066,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 1,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 1,
      "backward_transition_calls": 10,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 9.999729627452325e-06,
        "mean_abs_change": 4.728110876947896e-06,
        "updated_tensors": 128,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 9.999729627452325e-06,
      "step_seconds": 8.19261509552598,
      "peak_gpu_memory_bytes": 13520258560,
      "algorithm": "DDPO_SF_v1"
    }
  ]
}
```
## EXP_20261027T141845_9b4a88c37076_phase3_slurm_one

Slurm `None`; source `3dcf5947d32673a1a7075fdd9be8c099420372aa`.

```json
{
  "restart": null,
  "inputs": null,
  "real_updates_before_failure": null
}
```
## EXP_20261027T141857_470631a915ce_phase3_generator_rl_one

Slurm `86404`; source `3dcf5947d32673a1a7075fdd9be8c099420372aa`.

```json
{
  "restart": {
    "fresh_model": true,
    "adapter_exact": false,
    "optimizer_exact": false,
    "scheduler_exact": true,
    "rng_exact": true,
    "next_step_loss_exact": true,
    "rollout_seeds_exact": true,
    "reward_components_exact": true,
    "adapter_max_difference": 1.725631591398269e-05
  },
  "inputs": {
    "adapter_exact": true,
    "conditions_exact": true
  },
  "real_updates_before_failure": [
    {
      "step": 1,
      "loss": -0.014984687557443976,
      "reward_mean": -0.08180490632851922,
      "reward_variance": 0.0,
      "individual_rewards": [
        -0.08180490632851922
      ],
      "gradient_norm": 0.006332202814519405,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 1,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 1,
      "backward_transition_calls": 10,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 9.999730536947027e-06,
        "mean_abs_change": 4.728011842776017e-06,
        "updated_tensors": 128,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 9.999730536947027e-06,
      "step_seconds": 8.159428771585226,
      "peak_gpu_memory_bytes": 13512691200,
      "algorithm": "DDPO_SF_v1"
    }
  ]
}
```
## EXP_20261027T142201_648c36328684_phase3_slurm_one

Slurm `None`; source `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`.

```json
{
  "restart": null,
  "inputs": null,
  "real_updates_before_failure": null
}
```
## EXP_20261027T142212_e407797b1bef_phase3_generator_rl_one

Slurm `86405`; source `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`.

```json
{
  "restart": {
    "fresh_model": true,
    "adapter_exact": false,
    "optimizer_exact": false,
    "scheduler_exact": true,
    "rng_exact": true,
    "next_step_loss_exact": true,
    "rollout_seeds_exact": true,
    "reward_components_exact": true,
    "adapter_max_difference": 1.97364242922049e-05
  },
  "inputs": {
    "adapter_exact": true,
    "conditions_exact": true
  },
  "real_updates_before_failure": [
    {
      "step": 1,
      "loss": -0.014943636022508144,
      "reward_mean": -0.08158079882462821,
      "reward_variance": 0.0,
      "individual_rewards": [
        -0.08158079882462821
      ],
      "gradient_norm": 0.006334240082651377,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 1,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 1,
      "backward_transition_calls": 10,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 9.999707799579483e-06,
        "mean_abs_change": 4.7274101088747865e-06,
        "updated_tensors": 128,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 9.999707799579483e-06,
      "step_seconds": 21.46960769407451,
      "peak_gpu_memory_bytes": 9186541056,
      "algorithm": "DDPO_SF_v1"
    }
  ]
}
```
## EXP_20261027T143457_03808d628b67_phase3_slurm_smoke

Slurm `None`; source `7d548a34efa2558c9d5b61f8a559d06a953019e6`.

```json
{
  "restart": null,
  "inputs": null,
  "real_updates_before_failure": null
}
```
## EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke

Slurm `86411`; source `7d548a34efa2558c9d5b61f8a559d06a953019e6`.

```json
{
  "restart": {
    "fresh_model": true,
    "adapter_exact": false,
    "optimizer_exact": false,
    "scheduler_exact": true,
    "rng_exact": true,
    "next_step_loss_exact": true,
    "rollout_seeds_exact": true,
    "reward_components_exact": true,
    "adapter_max_difference": 2.8720805858029053e-06
  },
  "inputs": {
    "adapter_exact": true,
    "conditions_exact": true
  },
  "real_updates_before_failure": [
    {
      "step": 1,
      "loss": -0.09631205416735611,
      "reward_mean": -0.5298925384754936,
      "reward_variance": 0.22106646921637135,
      "individual_rewards": [
        -0.08158079882462821,
        -0.048346738268931655,
        -0.05351584193607173,
        -1.0,
        -1.0,
        -1.0,
        -0.058407892783482906,
        -0.05707411294182141,
        -1.0,
        -1.0
      ],
      "gradient_norm": 0.020313914865255356,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 10,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 10,
      "backward_transition_calls": 100,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 9.999949725170154e-06,
        "mean_abs_change": 4.9101590065005174e-06,
        "updated_tensors": 128,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 9.999949725170154e-06,
      "step_seconds": 106.60176666802727,
      "peak_gpu_memory_bytes": 20591604736,
      "algorithm": "DDPO_SF_v1"
    },
    {
      "step": 2,
      "loss": -0.0801527940275264,
      "reward_mean": -0.4391602098941803,
      "reward_variance": 0.20979520742226368,
      "individual_rewards": [
        -0.06970587720473609,
        -0.046452045192321156,
        -0.05908528541525204,
        -1.0,
        -1.0,
        -1.0,
        -0.0664728378256162,
        -0.060631347199281094,
        -1.0,
        -0.0892547061045964
      ],
      "gradient_norm": 0.01845010556280613,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 10,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 10,
      "backward_transition_calls": 100,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 1.0013297469413374e-05,
        "mean_abs_change": 3.3825389054331775e-06,
        "updated_tensors": 256,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 1.0013297469413374e-05,
      "step_seconds": 106.81316138384864,
      "peak_gpu_memory_bytes": 20604359680,
      "algorithm": "DDPO_SF_v1"
    },
    {
      "step": 3,
      "loss": -0.06233439334027935,
      "reward_mean": -0.34077073776473604,
      "reward_variance": 0.18626746597361832,
      "individual_rewards": [
        -0.05796729897459352,
        -0.06762201984723404,
        -0.05326672208805883,
        -1.0,
        -1.0,
        -1.0,
        -0.055387588342030836,
        -0.06379546051224072,
        -0.05459383701284726,
        -0.05507445087035501
      ],
      "gradient_norm": 0.01666080392897129,
      "finite_gradients": true,
      "probabilities_finite": true,
      "behavior_logp_replay_max_difference": 0.0,
      "conditioning_verified": true,
      "logical_batch_size": 10,
      "physical_micro_batch_size": 1,
      "gradient_accumulation_steps": 10,
      "backward_transition_calls": 100,
      "optimizer_updates": 1,
      "changes": {
        "max_abs_change": 1.003468241833616e-05,
        "mean_abs_change": 2.6940149640102263e-06,
        "updated_tensors": 256,
        "adapter_tensor_count": 256
      },
      "trainable_max_change": 1.003468241833616e-05,
      "step_seconds": 106.83462042198516,
      "peak_gpu_memory_bytes": 20604359680,
      "algorithm": "DDPO_SF_v1"
    }
  ]
}
```

Initial batched-CFG/BF16-autocast-adapter exact continuation failed; FP32 A/B isolation also failed; sequential CFG with BF16 reduced-precision reductions disabled also failed. No precise CUDA-kernel root cause is established. All original forward losses, rewards and RNG could match while backward gradients/AdamW updates differed. Fixed-state job86408 retained comparisons with/without gradient checkpointing: disabling it matched one fresh-instance gradient exactly, but same-instance repeat still differed. That diagnostic is not universal determinism proof and performs no optimizer update. Any subsequent full witness has its own run and configuration; no failed run is relabeled.

Local system Python lacks Diffusers; three numerical test dependency errors were not hidden by installing into a shared runtime. Canonical editor environment is the declared acceptance environment. No OOM has been observed unless a later raw run explicitly reports it. Tests/thresholds remain fixed; no reward, split, answer annotation or heldout protocol was changed to turn a failure into PASS.

Semantic inventory schema v1 rejected valid integer-count JSON output and marked all10 pairs unparseable. Package PASS meant files were created, never semantic certification. A separate CPU v2 reparse preserves all raw JSON and original table/forms, accepts only the observed integer representation, and still leaves every human label blank. No GPU re-evaluation or post-hoc semantic truth was fabricated.

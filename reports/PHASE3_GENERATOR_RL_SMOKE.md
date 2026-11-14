# Generator RL smoke

BLOCKED: no complete three-update logical-batch10 generator RL witness.

One-update full witnesses: ['EXP_20261027T143134_4f17668ff54a_phase3_generator_rl_one']. Three-update witnesses: [].

Three-update batch10 training was executed; the full integration run status is FAIL. Completed optimizer updates must not be confused with full restart acceptance.

Actual terminal-reward updates are distinct from supervised MSE and external controller RL. Failed exact-restart attempts retain their real updates and all negative evidence but are not full PASS.

```json
{
  "experiment": "EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke",
  "updates": [
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
  ],
  "parameters": {
    "trainable_parameters": 1594368
  },
  "adapter_changes": {
    "max_abs_change": 3.0045215680729598e-05,
    "mean_abs_change": 9.150057718931672e-06,
    "updated_tensors": 256,
    "adapter_tensor_count": 256
  },
  "frozen_tensor_count": 1130,
  "frozen_weights_unchanged": true,
  "implementation_tests": {
    "official_source_latent_exact": true,
    "image_ablation_max_prediction_difference": 0.26953125,
    "instruction_ablation_max_prediction_difference": 0.205078125,
    "explicit_seed_exact_trajectory_replay": true,
    "actual_generator_reward_sign_reversal_exact": true,
    "reward_test_gradient_norm": 0.022235674279249534,
    "scope": "Wiring and conditioning, not image quality improvement"
  },
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
  "restart_diagnostic": {
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
  "checkpoint_audit": {
    "scope": "read-only CPU audit after terminal/update evidence; original metrics unchanged",
    "adapter_changes": {
      "max_abs_change": 3.0045215680729598e-05,
      "mean_abs_change": 9.150057718931672e-06,
      "updated_tensors": 256,
      "adapter_tensor_count": 256
    },
    "trainable_parameters": 1594368,
    "frozen_tensor_count": 1130,
    "frozen_hashes_equal": true
  },
  "source_images": 10,
  "training_trajectories": 30,
  "restart_diagnostic_trajectories": 10,
  "unet_calls": null,
  "qwen_calls": 81
}
```

Tests A–G are actual runtime assertions in smoke_generator_rl.py plus analytical/official-transition CPU tests. H requires the separate three-update run; one-source witnesses never substitute for batch10. Full raw trajectories, per-step adapter/optimizer/RNG states, pre-update conditions, gradients, observations and restart diagnostics are under each run directory. Model call counter excludes internal checkpoint recomputations; allocation costs include them.

- `EXP_20261027T141314_7f2c3dc26636_phase3_slurm_one`: **FAIL**; `00950716195e20f9a08b2175b75d16221e1a231d`; Slurm `None`; config SHA256 `c258b5e71131a7e90253b025b6c69e49129f073e0d68e5134c7caca02c6472d6`.
- `EXP_20261027T141335_bc288d994eca_phase3_generator_rl_one`: **FAIL**; `00950716195e20f9a08b2175b75d16221e1a231d`; Slurm `86403`; config SHA256 `36388e7b5ed912fd6285b06dcbecff6b71ab945e912d6bef78ba32adfdc63832`.
- `EXP_20261027T141845_9b4a88c37076_phase3_slurm_one`: **FAIL**; `3dcf5947d32673a1a7075fdd9be8c099420372aa`; Slurm `None`; config SHA256 `7cacbbd0ff299647979d54908d05b755520b8a1a7f611b8055971d49b183d7e0`.
- `EXP_20261027T141857_470631a915ce_phase3_generator_rl_one`: **FAIL**; `3dcf5947d32673a1a7075fdd9be8c099420372aa`; Slurm `86404`; config SHA256 `21cdb9a46b19722adc02ec3a4e252aefc786ec1adf35015e6daca284278b036f`.
- `EXP_20261027T142201_648c36328684_phase3_slurm_one`: **FAIL**; `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`; Slurm `None`; config SHA256 `6bdb90d51116c1d9a2b61cf9b3836be57399df07c5a81e873107c78a0deddbeb`.
- `EXP_20261027T142212_e407797b1bef_phase3_generator_rl_one`: **FAIL**; `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`; Slurm `86405`; config SHA256 `6672be0a9e91b191361ec76dd496baf617a73c5d1ef75d5e615c53683e71fd2e`.
- `EXP_20261027T142708_33e331a1db43_phase3_slurm_backward`: **PASS**; `bf29d2e530bf89e7255104d6a340c2f3c4cafb40`; Slurm `None`; config SHA256 `b75e7249855760a2630a5496b9bec03a2879f66fe157fdf678157d1e3d99a586`.
- `EXP_20261027T142716_ee26d8788a67_phase3_backward_diagnostic`: **PASS**; `bf29d2e530bf89e7255104d6a340c2f3c4cafb40`; Slurm `86408`; config SHA256 `10350b3b71aed00b60207a8fcc11b4d4594d1e2d5bf9439adb78cc8fcce91bde`.
- `EXP_20261027T143117_b4453cea2d66_phase3_slurm_one`: **PASS**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `None`; config SHA256 `a05bacfdef50c57b61521283d0a8da130916bf01fd584e7dde22e6636c0e56bf`.
- `EXP_20261027T143134_4f17668ff54a_phase3_generator_rl_one`: **PASS**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `86409`; config SHA256 `77eb5528c4ef433a3642a7b4a22acf76a6d50d3caae5146ae913f49fc136a7a4`.
- `EXP_20261027T143457_03808d628b67_phase3_slurm_smoke`: **FAIL**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `None`; config SHA256 `776352cf5ab2668d993db235f6d2cf5fcefe58cc7318447a0d5d65f24d304274`.
- `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`: **FAIL**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `86411`; config SHA256 `95ed27bdd7157147c1298c74e632dad0796f5b17fa1dd2e757ad608ac4f6acdb`.
- `EXP_20261027T145338_d0c6788762ba_phase3_slurm_semantics`: **PASS**; `643b63c10f634635cda412462ff4e0588a2a39db`; Slurm `None`; config SHA256 `9cbbaed8e01b3aa6583b425e5285b1e5a30ee0e323e4b0ef698cc04fce6c13ef`.
- `EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review`: **PASS**; `643b63c10f634635cda412462ff4e0588a2a39db`; Slurm `86412`; config SHA256 `fc9565f863054222fbe7c60a0b879cd6a3e14c2d9f9beafa30ed1152890c3137`.
- `EXP_20261027T150420_0c1cc77f77e2_phase3_semantic_schema_audit`: **PASS**; `9e203d6012d8d0690c1505b9e99420def82ffb3d`; Slurm `None`; config SHA256 `c030a7caf3e2a50adf5fa03d9bfbffb34fda77edab2bf65c43b26e93b7d25231`.

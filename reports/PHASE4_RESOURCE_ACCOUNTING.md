Generated from immutable runs and current Slurm accounting. Development proxies only; no internal held-out selection.

Additional Phase4 allocation: **13122 seconds = 3.645000 A100 GPU-hours**, hard cap14400sec. Diagnostics 373/2700sec. Active/unreported jobs reserve their full wall limit. One GPU per job. Revision child runs share the same allocation and are not double charged.

| Job | State | GPU seconds |
|---|---|---|
| 86430 | COMPLETED | 314 |
| 86433 | COMPLETED | 388 |
| 86434 | COMPLETED | 3227 |
| 86435 | COMPLETED | 2707 |
| 86444 | COMPLETED | 59 |
| 86445 | COMPLETED | 3211 |
| 86446 | COMPLETED | 3216 |

Costs include loading, generation, Qwen judgments, training, fixed validation, failed attempts and RSI decisions. Reward compute term remains steps/30; actual costs are separately measured. Source exposures, trajectories and optimization counts are distinct. No third seed or unrestricted sweep. Existing environments and central model cache reused; retained experiment evidence is on H and canonical storage, not C.

```json
{
  "training_source_exposures": 320,
  "training_diffusion_trajectories": 320,
  "unique_training_source_images_across_seeds_and_arms": 154,
  "validation_trajectories_including_frozen": 180,
  "conditioning_seed_control_extra_generations": 4,
  "outcome_diagnostic_generations": 6,
  "measured_qwen_model_calls": 1050
}
```

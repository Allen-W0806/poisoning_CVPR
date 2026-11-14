# Resource report

Actual terminal one-A100 allocation: **1610 GPU seconds = 0.447222 GPU-hours**, against a separate Phase3 cap7200 seconds. Pending/unconfirmed jobs: []. All failures and diagnostics are charged; Phase2 historical1187 seconds are not subtracted from the new budget.

| Job | State | GPU seconds | Exit |
|---|---|---:|---|
| 86403 | FAILED | 88 | 1:0 |
| 86404 | FAILED | 73 | 1:0 |
| 86405 | FAILED | 119 | 1:0 |
| 86408 | COMPLETED | 131 | 0:0 |
| 86409 | COMPLETED | 90 | 0:0 |
| 86411 | FAILED | 953 | 1:0 |
| 86412 | COMPLETED | 156 | 0:0 |

```json
{
  "recorded_rewarded_trajectories_including_restart_and_failures": 48,
  "sampling_seconds_recorded": 212.51751920813695,
  "reward_judging_seconds_recorded": 356.99388954089954,
  "retained_phase3_experiment_bytes": 515947616,
  "selected_run": "EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke",
  "training_seconds_by_update": [
    106.60176666802727,
    106.81316138384864,
    106.83462042198516
  ],
  "peak_allocated_bytes_by_update": [
    20591604736,
    20604359680,
    20604359680
  ],
  "generation_mean_seconds": 4.427448316836187
}
```

Sampling subtotals exclude seed-control extra generations unless explicitly saved as observations; total allocation includes all generation, model loading, numerical tests, judging, hashing and replay. Peak values are measured CUDA allocation with co-resident frozen Qwen, not total device reservation. No larger comparison was launched. The first batch10 measured ~4.55 seconds/generation, ~7.55 seconds/reward audit and ~106.6 seconds/optimizer update. Six batch10 updates plus two10-source validation looks therefore extrapolate to ~80*(4.55+7.55)+6*106.6 =1608 seconds/arm, with loading/failures/selection overhead additional. Prepared per-arm cap was revised from600 to1800 seconds based on this measured cost; it does not authorize a sweep or expand the global7200-second budget. Multi-seed M0–M4 evaluation is not covered by one successful feasibility check. No new model download/environment installation or multi-GB C artifact was required.

Generated from immutable runs and current Slurm accounting. Development proxies only; no internal held-out selection.

| Arm/seed | Status | Updates | Train exposures | Validation trajectories | Run |
|---|---|---|---|---|---|
| frozen | PASS | 0 | 0 | 20 | EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR |
| rl42 | PASS | 8 | 80 | 40 | EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL |
| rl123 | PASS | 8 | 80 | 40 | EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL |
| rsi42 | PASS | 8 | 80 | 40 | EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI |
| rsi123 | PASS | 8 | 80 | 40 | EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI |

| Arm/seed | Gradient min | Gradient max | Largest per-update LoRA delta | Initial-to-final LoRA delta | Mean optimization sec | Peak allocated GB | Frozen unchanged |
|---|---|---|---|---|---|---|---|
| rl42 | 0.016203900799155235 | 0.021066956222057343 | 1.008517574518919e-05 | 7.985760021256283e-05 | 137.46169962975546 | 20.601747456 | True |
| rl123 | 0.015744445845484734 | 0.024953380227088928 | 1.0155858035432175e-05 | 7.98227047198452e-05 | 113.97888981923461 | 20.602238976 | True |
| rsi42 | 0.015911821275949478 | 0.02066863141953945 | 1.0059231499326415e-05 | 5.990559293422848e-05 | 138.51566491861013 | 20.601747456 | True |
| rsi123 | 0.015056087635457516 | 0.02548864856362343 | 1.0064721209346317e-05 | 5.9828282246598974e-05 | 137.97846691170707 | 20.602238976 | True |

Per-update loss, reward variance/components, LR, source IDs, generation/judging/optimization time, cumulative elapsed GPU-hour estimates and checkpoint paths remain in each run metrics.json and append-only generator_rl_update events. Actual allocation time is reconciled separately with Slurm; memory values include the co-resident frozen Qwen and are CUDA allocation, not total reservation.

| Arm | Seed | Step | S | A | D | C | U | Reward | Generation sec |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 42 | 0 | 0.30000000 | 0.60000000 | 0.17595693 | 0.33333333 | 1.00000000 | -0.43772059 | 5.86701735 |
| M0 | 123 | 0 | 0.40000000 | 0.80000000 | 0.17685662 | 0.33333333 | 1.00000000 | -0.25461367 | 5.85765360 |
| M1 | 42 | 2 | 0.25000000 | 0.50000000 | 0.17577835 | 0.33333333 | 1.00000000 | -0.53108172 | 5.86272320 |
| M1 | 42 | 4 | 0.25000000 | 0.50000000 | 0.17502882 | 0.33333333 | 1.00000000 | -0.53099800 | 5.86090998 |
| M1 | 42 | 6 | 0.30000000 | 0.60000000 | 0.17464070 | 0.33333333 | 1.00000000 | -0.43756522 | 5.85234215 |
| M1 | 42 | 8 | 0.20000000 | 0.40000000 | 0.17425495 | 0.33333333 | 1.00000000 | -0.62472904 | 5.84563341 |
| M1 | 123 | 2 | 0.40000000 | 0.80000000 | 0.17620533 | 0.33333333 | 1.00000000 | -0.25448911 | 5.01778469 |
| M1 | 123 | 4 | 0.35000000 | 0.70000000 | 0.17481829 | 0.33333333 | 1.00000000 | -0.34756741 | 5.00171387 |
| M1 | 123 | 6 | 0.40000000 | 0.80000000 | 0.17458045 | 0.33333333 | 1.00000000 | -0.25414654 | 5.00463540 |
| M1 | 123 | 8 | 0.35000000 | 0.70000000 | 0.17363562 | 0.33333333 | 1.00000000 | -0.34785917 | 5.00263241 |
| M2 | 42 | 2 | 0.25000000 | 0.50000000 | 0.17560767 | 0.33333333 | 1.00000000 | -0.53105470 | 5.90750999 |
| M2 | 42 | 4 | 0.25000000 | 0.50000000 | 0.17459392 | 0.33333333 | 1.00000000 | -0.53090812 | 5.90019618 |
| M2 | 42 | 6 | 0.20000000 | 0.40000000 | 0.17440748 | 0.33333333 | 1.00000000 | -0.62468413 | 5.90430296 |
| M2 | 42 | 8 | 0.25000000 | 0.50000000 | 0.17468956 | 0.33333333 | 1.00000000 | -0.53095553 | 5.90358058 |
| M2 | 123 | 2 | 0.40000000 | 0.80000000 | 0.17564662 | 0.33333333 | 1.00000000 | -0.25439507 | 5.88153185 |
| M2 | 123 | 4 | 0.40000000 | 0.80000000 | 0.17524291 | 0.33333333 | 1.00000000 | -0.25430930 | 5.88018408 |
| M2 | 123 | 6 | 0.40000000 | 0.80000000 | 0.17468789 | 0.33333333 | 1.00000000 | -0.25420182 | 5.88138553 |
| M2 | 123 | 8 | 0.40000000 | 0.80000000 | 0.17451205 | 0.33333333 | 1.00000000 | -0.25415147 | 5.88167561 |

Raw VQA diagnostics at the final checkpoint (A above is the semantic-gated reward credit):

| Arm | Seed | Raw answer consistency | Same normalized answer fraction | Reference score, diagnostic only |
|---|---|---|---|---|
| M0 | 42 | 0.6 | 0.6 | 0.55 |
| M0 | 123 | 0.8 | 0.9 | 0.6199999999999999 |
| M1 | 42 | 0.5 | 0.5 | 0.45 |
| M1 | 123 | 0.8 | 0.9 | 0.6199999999999999 |
| M2 | 42 | 0.5 | 0.5 | 0.45 |
| M2 | 123 | 0.8 | 0.9 | 0.6199999999999999 |

Raw reference scores are not accuracy on verified semantics-preserving edits. Image semantics remain unverified.

All intermediate checkpoints and images are retained. Measured baseline generation/judging and diagnostic optimization extrapolated four10-update arms beyond14400sec, so before training all arms were uniformly reduced to8 updates /80 exposures /4 validation looks, finalstep8 (configs/phase4_campaign_v2.yaml). Original10-update driver is retained unchanged. Primary comparison uses finalstep8, never the best observed development look. Training sources are seed-shuffled and image-distinct within each pilot; evaluation sources are fixed and image-disjoint from training. Logical10 / physical1 / accumulation10, rank8 alpha16, BF16 base and FP32 LoRA, fixed benign reward. No training reward-only quality claim.

Each optimizer update draws a different10-source batch, so between-update training reward changes do not isolate learning. Fixed-source matched development outcomes determine the comparison. Every trained arm also performs one repeat-seed conditioning-control generation, charged to the allocation but excluded from its80 training trajectories.

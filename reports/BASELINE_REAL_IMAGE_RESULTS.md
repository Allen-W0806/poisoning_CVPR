# Natural-image control comparison — internally measured

Recomputed UTC 2026-11-01T22:57:54.484248+00:00. Immutable run: baseline_comparison_20261101T2255Z_v1. **160 unique matched sources,960 primary records;72 dependent repeats excluded from n.** No new inference, generation or training.

Metric: **VQA original-reference agreement**, not verified edited-image accuracy or post-training ASR. Same original questions, ten references and pinned scorer across all conditions. Historically exposed development data.

| Condition | n | Agreement % | Source95%CI % | Delta C0 pp | Paired95%CI pp | Delta C5 pp | Paired95%CI pp | Changed/C0 | SSIM aligned256 | Invalid/missing |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 Original | 160 | 84.8750 | [79.562, 89.750] | +0.0000 | [0.000, 0.000] | +8.4375 | [3.687, 13.312] | 0 | 1.000000 | 0/0 |
| C1 Identity | 160 | 84.8750 | [79.562, 89.750] | +0.0000 | [0.000, 0.000] | +8.4375 | [3.687, 13.312] | 0 | 1.000000 | 0/0 |
| C2 Brightness x1.03 | 160 | 85.0625 | [79.812, 89.877] | +0.1875 | [0.000, 0.563] | +8.6250 | [3.875, 13.500] | 3 | 0.998914 | 0/0 |
| C3 Contrast x1.03 | 160 | 85.0625 | [79.812, 89.875] | +0.1875 | [-1.687, 2.062] | +8.6250 | [3.500, 13.813] | 6 | 0.996474 | 0/0 |
| C4 Frozen IP2Pix | 160 | 64.9375 | [57.625, 71.750] | -19.9375 | [-26.500, -13.562] | -11.5000 | [-17.187, -6.125] | 51 | 0.734032 | 0/0 |
| C5 Geometry only | 160 | 76.4375 | [70.125, 82.312] | -8.4375 | [-13.312, -3.687] | +0.0000 | [0.000, 0.000] | 27 | 1.000000 | 0/0 |

Historical C0=.84875, C4=.649375 and C5=.764375 point estimates reconcile using the unchanged1e-12 arithmetic check. Intervals use10,000 paired source resamples, fixed seed202611012255. These are descriptive, unadjusted intervals. Historical interval endpoint differences reflect bootstrap streams, not changed data. All C5 contrasts are pipeline diagnostics; C4-C5 isolates generation beyond geometry, not semantic correctness.

SSIM uses a common256 RGB reference: stored C5, white padding, BICUBIC. C0-C3 pass through this same geometry (640 new CPU measurements); C4/C5 reuse hash-bound aligned measurements. No human prerequisite is imposed on descriptive image metrics. Identity/geometry SSIM=1 is not instruction success. LPIPS is null because calibrated implementation/weights are unavailable, not replaced by another metric.

| Condition | SSIM95%CI | VLM primary-record seconds | VLM mean seconds | C4 generation seconds / timed sources |
|---|---|---|---|---|
| C0 | [1.000000, 1.000000] | 31.033 | 0.193958 | not separately measured |
| C1 | [1.000000, 1.000000] | 31.400 | 0.196250 | not separately measured |
| C2 | [0.998782, 0.999031] | 31.397 | 0.196230 | not separately measured |
| C3 | [0.995623, 0.997215] | 31.408 | 0.196300 | not separately measured |
| C4 | [0.714090, 0.753742] | 29.098 | 0.181862 | 138.645 / 160 |
| C5 | [1.000000, 1.000000] | 29.421 | 0.183881 | not separately measured |

Timing sums cover identified saved primary records. Historical job costs also include pilot, repairs, repeats, loads/writes. Shared allocation is never charged to every condition; full receipt is retained in this run real_image/summary.json. Component timing is not a synchronized latency benchmark. All missing metrics remain null.

The C4 decline is a frozen-pipeline observation, not backdoor implantation. Independent edited labels remain0. C2/C3 tie for largest observed agreement, without established superiority.

Evidence: /storage/home/ydk297/projects/poisoning/results/matched_feedback_20261028_v2/main_repair_v1/predictions.jsonl; /storage/home/ydk297/projects/poisoning/results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json; /storage/home/ydk297/projects/poisoning/results/baseline_comparison_20261101T2255Z_v1/real_image/per_source_recomputed.csv.

## Shared historical allocation (not per-condition costs)

The matched-feedback campaign allocated **584 GPU-seconds (0.162222 GPU-hours)**: 86793 pilot 129s; 86795 failed main 436s; 86796 repair 19s. CPU86794 consumed 32 allocated seconds and zero GPU. These jobs include pilots, failed execution, repeats, loading and writes. They cannot be partitioned exactly over C0-C5. The 160 C4 primary-generation records total 138.644700s; this is included within, not added to, allocation. This recomputation job88317 allocated 20 CPU-job seconds, zero GPUs. Historical failure86795 is retained; no failed run was discarded.

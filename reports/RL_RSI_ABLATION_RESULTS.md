# Benign DDPO / scheduler / RSI ablation

**Controlled Tiny FP32 engineering/benign-learning ablation; not natural-image or backdoor efficacy.** Three independent training seeds share32 synthetic test conditions. Objective: predeclared harmless cosine of the first terminal latent coordinate. No ASR/security reward or new optimizer steps.

| Arm | Seeds | Test mean | Sample SD across seeds | GPU-held seconds mean±SD | Updates/NFE per seed | Validation/test queries |
|---|---|---|---|---|---|---|
| fixed | 3 | 0.126835376975 | 0.038210635057 | 15.224488 ± 0.368551 | 4 /320 | 48 /32 |
| adaptive_scheduler | 3 | 0.126832737995 | 0.038215080977 | 15.017814 ± 0.006524 | 4 /320 | 48 /32 |
| rsi | 3 | 0.126833996813 | 0.038215037948 | 15.038835 ± 0.009552 | 4 /320 | 48 /32 |

| Seed | Arm | Outcome | Delta fixed | Paired source95%CI | GPU-held seconds | LR candidate→retained |
|---|---|---|---|---|---|---|
| 42 | fixed | 0.157166216697 | +0.000000000000 | [0.000000000000, 0.000000000000] | 15.650035 | 1e-05→1e-05 |
| 123 | fixed | 0.083919592202 | +0.000000000000 | [0.000000000000, 0.000000000000] | 15.008279 | 1e-05→1e-05 |
| 3407 | fixed | 0.139420322026 | +0.000000000000 | [0.000000000000, 0.000000000000] | 15.015150 | 1e-05→1e-05 |
| 42 | adaptive_scheduler | 0.157166216697 | +0.000000000000 | [0.000000000000, 0.000000000000] | 15.025347 | 1e-05→1e-05 |
| 123 | adaptive_scheduler | 0.083911675261 | -0.000007916940 | [-0.000020035793, 0.000004628213] | 15.014150 | 5e-06→5e-06 |
| 3407 | adaptive_scheduler | 0.139420322026 | +0.000000000000 | [0.000000000000, 0.000000000000] | 15.013946 | 1e-05→1e-05 |
| 42 | rsi | 0.157163351309 | -0.000002865389 | [-0.000010409043, 0.000004624634] | 15.045673 | 2e-05→1e-05 |
| 123 | rsi | 0.083911675261 | -0.000007916940 | [-0.000020035793, 0.000004628213] | 15.042912 | 5e-06→5e-06 |
| 3407 | rsi | 0.139426963869 | +0.000006641843 | [-0.000002930980, 0.000015493552] | 15.027921 | 2e-05→2e-05 |

| Contrast | Mean difference | Crossed seed/source95%CI | Conclusion |
|---|---|---|---|
| rsi_minus_fixed | -0.000001380162 | [-0.000011316767, 0.000008262257] | RSI superiority NOT ESTABLISHED |
| rsi_minus_adaptive_scheduler | +0.000001258818 | [-0.000005280017, 0.000009227506] | RSI superiority NOT ESTABLISHED |

Original crossed-bootstrap intervals reproduce exactly from saved paired observations. SD uses three seed means;96 seed-condition outcomes are not96 independent conditions. Intervals are exploratory with only3 training seeds.

Existing full-state preintervention evidence is reused and SHA-bound, including12 initial/prefix cross-branch disk comparisons: parameters, optimizer/scheduler, saved RNG, gradients, predictions, contracts/version/trajectory IDs. Four updates include rejected computation. RSI rollback restores its OWN prefix. Every branch reports320 measured forward NFE,48 validation and32 test queries.

Branch GPU-held seconds total135.843412. Successful job88032 allocated144 GPU-seconds SHARED across9 branches; do not charge144 to each row. Prior job88029 allocated147 seconds to mixed diagnostics and a partial ablation with strict-JSON sentinel failure. Its failure/costs remain and cannot be precisely assigned to an arm. Sequential same-node timings do not establish speed superiority.

## Historical pretrained IP2Pix — separate, noncausal

M0 is the frozen reference (zero optimizer updates; one checkpoint evaluated at seeds42/123). M1 is DDPO-only and M2 is DDPO+RSI (two training seeds, eight updates each). These labels are distinct from the later Tiny fixed/adaptive_scheduler/rsi arms.

| Arm | seed | Final legacy proxy reward |
|---|---|---|
| M0 | 42 | -0.437720586509 |
| M0 | 123 | -0.254613674382 |
| M1 | 42 | -0.624729041619 |
| M1 | 123 | -0.347859167804 |
| M2 | 42 | -0.530955534565 |
| M2 | 123 | -0.254151467408 |

Each trained historical arm uses8 updates for seeds42/123. Before LR intervention at step4, adapter max differences are4.34328e-5/6.64559e-5; optimizer moments already differ. Thus final pretrained RL/RSI differences cannot support causal improvement. New Tiny FP32 equality does not retroactively repair the historical BF16 comparison. No ranking is applied to confounded historical arms.

Evidence: /storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/threeway/results.json; /storage/home/ydk297/projects/poisoning/results/baseline_comparison_20261101T2255Z_v1/ablation/recomputed.json; original protocol and preadaptation disk verification.

## Supplemental ordinary-scheduler contrast

Scheduler minus Fixed = -0.000002638980, crossed 95% CI [-0.000011719065, 0.000001498483]. 5000 draws, fixed supplemental seed2026110123; original two RSI intervals remain bitwise identical. This is exploratory, and also includes zero. No training or image measurement was repeated.

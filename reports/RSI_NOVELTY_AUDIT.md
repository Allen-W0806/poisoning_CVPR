# RSI mechanism, novelty and fairness audit

**RSI methodology: NO-GO for causal improvement or novelty claims.** Revision persistence is implemented; benefit is not identified.

Audit date: 2026-10-28T00:00:38.995156+00:00 (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\Project\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: [reports/phase45_recompute.json:1](H:/Project/Poisoning/reports/phase45_recompute.json:1); audit source: [scripts/audit_phase45.py:1](H:/Project/Poisoning/scripts/audit_phase45.py:1).
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.


`RSIController.revise` accepts a candidate only for strict validation-score gain within budget, recording rejected costs and optional rollback evidence. Phase4 instead subclasses it with `revise_schedule`: after at least two looks, if both `mean_delta <= (CI_hi−CI_lo)/2`, make a single LR reduction. It records `quality_improvement_accepted=false`. The controller did not learn a proposal policy, edit code, search multiple learning mechanisms or validate a candidate improvement before adoption.

The trigger is a gain-magnitude-versus-width heuristic, not the hypothesis test “CI contains zero”, not a plateau test on successive score slopes, and not evidence of no improvement. For asymmetric percentile intervals the CI can be entirely positive while the rule still fires. Its raw reason string “two unresolved validation gains” is imprecise; the Phase4 assumption ledger already qualifies this. We preserve the raw history.

Both seeds change LR after step4, so updates1–4 use1e-5; updates5–8 use5e-6. Step4 checkpoints are saved **before** revision. Parent manifests stay initial; child config revisions and active checkpoint contracts record the actual new setting. No retroactive schedule tuning is found.

## Pre-intervention and final comparison

| Seed | Step4 M2−M1 mean (before LR revision) | Step8 M2−M1 mean | Final descriptive paired95% interval |
|---|---:|---:|---|
| 42 | 0.000089882 | 0.093773507 | [-0.000018919, 0.281345849] |
| 123 | 0.093258113 | 0.093707700 | [-0.000275419, 0.281527407] |

The seed123 difference largely predates intervention. Final differences each include a single source's roughly.938 proxy-gate change divided by10; remaining small photometric differences cannot establish independent semantic improvement. Both final intervals include zero. A small positive M2-versus-M0 interval for seed123 is retained in the raw results; after repeated development reuse and with only one fitted seed, it is not confirmatory generalization evidence.

Initial adapter tensors are bitwise equal per matched seed, and train source order/noise seeds, validation IDs/seeds, batch counts and model revisions match. They do not force subsequent numerical execution to match. Real cost differs (3227/3211 and2707/3216 seconds), and compute nodes differ. No matched intervention fork from a verified continuation was executed. No counterfactual claim that LR caused the difference is justified.

Supported: revision memory, one-variable validation-conditioned configuration change, budget accounting, fresh trajectories and honest proxy logging. Unverified: self-improvement beyond scheduling, quality-accepted revision selection, superiority to a conventional cooldown, causal RL benefit, generalization, or improved security behavior.

## F05 — HIGH: Executed RSI is a validation-driven LR schedule extension

Evidence status: **VERIFIED**. Source/function: [src/rsi/phase4_schedule.py:13](H:/Project/Poisoning/src/rsi/phase4_schedule.py:13); [src/rsi/controller.py:21](H:/Project/Poisoning/src/rsi/controller.py:21); [scripts/run_phase4_rsi.py:112](H:/Project/Poisoning/scripts/run_phase4_rsi.py:112).

Supporting experiment IDs: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Both arms make one LR change 1e-5 to 5e-6 after update4. The extension records accepted_schedule but explicitly quality_improvement_accepted=false. It bypasses the legacy candidate-gain acceptance/rollback semantics. Persistent revision history is implemented, but ordinary adaptive scheduling and a novel recursive improvement mechanism are not experimentally distinguished.

Proposed correction: Describe this result as an exploratory validation-driven cooldown. Define the proposed RSI mechanism and compare with a predeclared ordinary scheduler/control before a novelty claim. Do not relabel historical acceptance as quality acceptance.

## F06 — CRITICAL: RSI causal benefit is confounded by divergence before intervention

Evidence status: **VERIFIED**. Source/function: [scripts/run_phase4_rsi.py:81](H:/Project/Poisoning/scripts/run_phase4_rsi.py:81); [scripts/build_phase4_reports.py:81](H:/Project/Poisoning/scripts/build_phase4_reports.py:81).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Initial adapters, data order and noise seeds match. Before either LR change, step4 adapters already differ by max 4.34328e-5 (42) and 6.64559e-5 (123). Seed123 step4 M2-M1 reward is +0.0932581, comparable to final +0.0937077. Both final paired intervals include zero. Each final +~0.094 is driven by one gate flip. Matching seeds therefore does not isolate the intervention.

Proposed correction: Withdraw causal RSI-gain claims. A future authorized design must establish comparable pre-intervention behavior and separate numerical execution variability from adaptation effects; preserve all pre-intervention evidence and negative outcomes.

## F07 — HIGH: Two seeds and ten repeatedly reused development images cannot establish generalization

Evidence status: **VERIFIED**. Source/function: [scripts/run_phase4_budgeted.py:63](H:/Project/Poisoning/scripts/run_phase4_budgeted.py:63); [src/evaluation/phase4_statistics.py:4](H:/Project/Poisoning/src/evaluation/phase4_statistics.py:4); [src/evaluation/phase4_statistics.py:15](H:/Project/Poisoning/src/evaluation/phase4_statistics.py:15).

Supporting experiment IDs: `EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR`, `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: There are 180 validation trajectories but only 10 unique development images. Four looks per trained arm reuse them; M2 uses these same looks to revise LR. Percentile source bootstrap CIs are conditional on a fitted seed, without repeated-selection correction. Two seeds mix training and evaluation-noise variation. Treating 180 trajectories as independent images would be invalid.

Proposed correction: Keep these estimates exploratory. Predeclare a separate selection/evaluation design and source-clustered analysis before any new campaign; do not reopen the old held-out subset to rescue this comparison.

## F13 — MEDIUM: Exposure matching does not establish equal realized compute

Evidence status: **VERIFIED**. Source/function: [scripts/submit_phase4.py:1](H:/Project/Poisoning/scripts/submit_phase4.py:1); [reports/phase4_sacct_final.log:1](H:/Project/Poisoning/reports/phase4_sacct_final.log:1); [reports/phase4_results.json:2787](H:/Project/Poisoning/reports/phase4_results.json:2787).

Supporting experiment IDs: `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL`, `EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL`, `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: Each arm has 8 updates and 80 training exposures. M1/M2 elapsed GPU seconds are 3227/3211 for42 and 2707/3216 for123; nodes differ. No repeated same-node latency comparison was done. The entire campaign uses 13122 seconds, within 14400, including failures/diagnostics and evaluation.

Proposed correction: Report matched exposures plus actual cost, rather than strict compute equivalence or generation speedup. A future efficiency study needs a predeclared comparable hardware/timing protocol.

## F15 — MEDIUM: Checkpoint configuration revisions require reconstructing the exact active contract

Evidence status: **VERIFIED**. Source/function: [src/rl/diffusion_rl_trainer.py:80](H:/Project/Poisoning/src/rl/diffusion_rl_trainer.py:80); [scripts/run_phase4_rsi.py:128](H:/Project/Poisoning/scripts/run_phase4_rsi.py:128); [scripts/run_phase4_rsi.py:129](H:/Project/Poisoning/scripts/run_phase4_rsi.py:129).

Supporting experiment IDs: `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI`, `EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI`.

Observed: LR revision updates optimizer LR, scheduler base_lrs and policy contract and creates a child revision record. The parent manifest/config remains the initial version; subsequent checkpoints carry the active revision. Reconstructing a parent-only default policy would fail the contract check. No production resume across this revision was accepted or used.

Proposed correction: Document the active revision lineage and reconstruction procedure; retain immutable parent and child records. Validate it before any future authorized resume; do not treat an intentional contract rejection as successful continuation.

## F17 — UNRESOLVED: The specific backward numerical divergence mechanism is not localized

Evidence status: **UNRESOLVED**. Source/function: [scripts/diagnose_phase4_restart.py:47](H:/Project/Poisoning/scripts/diagnose_phase4_restart.py:47); [scripts/audit_phase4_restart.py:29](H:/Project/Poisoning/scripts/audit_phase4_restart.py:29).

Supporting experiment IDs: `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`, `EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC`, `EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT`.

Observed: The earliest inspected difference is in backward gradients after exact forward log-likelihoods. That does not identify the first differing CUDA primitive, exclude hidden execution-state differences, or prove an unavoidable universal nondeterminism mechanism.

Proposed correction: Retain competing hypotheses and exact diagnostics. Only separately authorized controlled diagnostics may identify a root cause; do not invent a fix or declare all GPU determinism impossible.
